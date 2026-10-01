"""Train and select source-task masked MLP teachers for meta-learning."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from pattern.length_interp.mlp import BatchedMaskedMLP
from .core import EDGES, HIDDEN, SEQ_LEN, build_experiment_data
from .convergence import loss_plateau


@dataclass(frozen=True)
class BankConfig:
    models_by_density: tuple[int, ...] = (128, 128, 32)
    keep_per_density: int = 32
    batch_size: int = 128
    learning_rate: float = 0.005
    decay_every: int = 2000
    decay_factor: float = 0.5
    minimum_lr: float = 0.0005
    min_steps: int = 2000
    max_steps: int = 8000
    eval_every: int = 100
    plateau_relative: float = 0.01
    plateau_patience: int = 400
    plateau_passes: int = 3
    probe_size: int = 128
    densities: tuple[int, ...] = (32, 44, EDGES)

    def validate(self) -> None:
        if not self.densities or len(self.models_by_density) != len(self.densities):
            raise ValueError("models_by_density and densities must align")
        if self.keep_per_density < 1 or any(
                count < self.keep_per_density for count in self.models_by_density):
            raise ValueError("each density needs at least keep_per_density teacher models")
        if self.batch_size < 2 or self.batch_size % 2:
            raise ValueError("bank batch_size must be a positive even number")
        if not self.min_steps <= self.max_steps or self.eval_every < 1:
            raise ValueError("invalid bank step schedule")
        if any(not 0 <= density <= EDGES for density in self.densities):
            raise ValueError("invalid mask density")


def exact_k_masks(n_masks: int, k_active: int, seed: int) -> torch.Tensor:
    """Draw independent uniformly random masks with exactly ``k_active`` edges."""
    if n_masks < 1 or not 0 <= k_active <= EDGES:
        raise ValueError("invalid mask count or active-edge count")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    # Sorting independent random keys avoids Python-level per-model loops.
    keys = torch.rand((n_masks, EDGES), generator=generator)
    active = keys.argsort(dim=1)[:, :k_active]
    masks = torch.zeros((n_masks, EDGES), dtype=torch.float32)
    masks.scatter_(1, active, 1.0)
    return masks.reshape(n_masks, SEQ_LEN, HIDDEN)


def _hash_ids(ids: torch.Tensor) -> str:
    raw = ids.detach().cpu().to(torch.int64).contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _balanced_eval(model: BatchedMaskedMLP, pool: dict[str, torch.Tensor],
                   device: torch.device, batch_size: int = 1024) -> dict[str, torch.Tensor]:
    x = pool["x"].to(device)
    y = pool["y"].to(device)
    n_models = model.n_models
    total = torch.zeros(n_models, device=device)
    correct_total = torch.zeros(n_models, device=device)
    positive_loss = torch.zeros_like(total)
    negative_loss = torch.zeros_like(total)
    positive_correct = torch.zeros_like(total)
    negative_correct = torch.zeros_like(total)
    positive_count = int((y > 0.5).sum())
    negative_count = int((y <= 0.5).sum())
    model.eval()
    with torch.no_grad():
        for start in range(0, x.size(0), batch_size):
            xb = x[start:start + batch_size]
            yb = y[start:start + batch_size]
            logits = model(xb)
            losses = F.binary_cross_entropy_with_logits(
                logits, yb[:, None].expand_as(logits), reduction="none")
            positive = yb > 0.5
            negative = ~positive
            if positive.any():
                positive_loss += losses[positive].sum(dim=0)
                positive_correct += ((logits[positive] > 0).to(y.dtype)
                                     == yb[positive, None]).to(y.dtype).sum(dim=0)
            if negative.any():
                negative_loss += losses[negative].sum(dim=0)
                negative_correct += ((logits[negative] > 0).to(y.dtype)
                                     == yb[negative, None]).to(y.dtype).sum(dim=0)
            total += losses.sum(dim=0)
            correct_total += ((logits > 0) == (yb[:, None] > 0.5)).to(y.dtype).sum(dim=0)
    model.train()
    return {
        "natural_bce": total / max(1, x.size(0)),
        "balanced_bce": 0.5 * (
            positive_loss / max(1, positive_count) + negative_loss / max(1, negative_count)
        ),
        "natural_accuracy": correct_total / max(1, x.size(0)),
        "balanced_accuracy": 0.5 * (
            positive_correct / max(1, positive_count) + negative_correct / max(1, negative_count)
        ),
    }


def _cpu_state(model: BatchedMaskedMLP) -> dict[str, torch.Tensor]:
    return {name: getattr(model, name).detach().cpu().clone()
            for name in ("w1", "b1", "w2", "b2", "masks")}


def _copy_best(best: dict[str, torch.Tensor], current: dict[str, torch.Tensor],
               improved: torch.Tensor) -> None:
    for name in ("w1", "b1", "w2", "b2"):
        mask_shape = (improved.size(0),) + (1,) * (current[name].ndim - 1)
        best[name] = torch.where(improved.view(mask_shape), current[name], best[name])


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _train_pattern(
    pattern: str,
    pools: dict[str, dict[str, torch.Tensor]],
    probe_ids: torch.Tensor,
    out: Path,
    seed: int,
    device: torch.device,
    config: BankConfig,
    resume: bool,
    extend: bool,
) -> dict[str, Any]:
    task_id = f"k4:{pattern}"
    task_seed = int(seed) * 100 + int(pattern, 2)
    final_path = out / "teachers" / f"pattern_{pattern}.pt"
    checkpoint_path = out / "checkpoints" / f"bank_{pattern}.pt"
    prior_final = None
    if final_path.exists():
        stored = torch.load(final_path, map_location="cpu", weights_only=False)
        old_config = stored.get("config", {})
        current_config = asdict(config)
        if old_config == current_config and stored.get("seed") == task_seed:
            return stored
        same_except_cap = (
            old_config.get("max_steps", 0) < config.max_steps
            and {key: value for key, value in old_config.items() if key != "max_steps"}
            == {key: value for key, value in current_config.items() if key != "max_steps"}
        )
        if stored.get("seed") != task_seed:
            raise ValueError(f"existing teacher protocol mismatch: {final_path}")
        if extend and same_except_cap and not stored.get("cap_hit", False):
            # A converged legacy teacher is already complete under the shared
            # protocol; do not retrain it just because capped peers are extended.
            return stored
        extension_compatible = (extend and same_except_cap
                                and stored.get("cap_hit", False))
        if not extension_compatible:
            raise ValueError(f"existing teacher protocol mismatch: {final_path}")
        prior_final = stored

    support = pools[task_id]["support"]
    query = pools[task_id]["query"]
    masks = torch.cat([
        exact_k_masks(count, density, task_seed + 11 + index)
        for index, (density, count) in enumerate(zip(config.densities, config.models_by_density))
    ], dim=0)
    n_models = masks.size(0)
    model = BatchedMaskedMLP(masks, seed=task_seed + 91).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    rng = torch.Generator(device="cpu").manual_seed(task_seed + 171)
    pos = torch.nonzero(support["y"] > 0.5, as_tuple=False).flatten()
    neg = torch.nonzero(support["y"] <= 0.5, as_tuple=False).flatten()
    if not pos.numel() or not neg.numel():
        raise ValueError(f"source task {task_id} lacks a support class")
    support_x = support["x"].to(device)
    support_y = support["y"].to(device)

    best = _cpu_state(model)
    best_score = torch.full((n_models,), float("inf"))
    best_step = torch.zeros(n_models, dtype=torch.int64)
    curves: dict[str, list[torch.Tensor]] = {
        key: [] for key in ("step", "train_natural_bce", "train_balanced_bce",
                            "train_natural_accuracy", "train_balanced_accuracy",
                            "query_natural_bce", "query_balanced_bce",
                            "query_natural_accuracy", "query_balanced_accuracy")
    }
    step = 0
    converged = torch.zeros(n_models, dtype=torch.bool)
    plateau_passes = torch.zeros(n_models, dtype=torch.int64)
    plateau_step = torch.zeros(n_models, dtype=torch.int64)
    stop_reason = "step_cap"
    if resume and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        old_config = saved.get("config", {})
        current_config = asdict(config)
        config_match = old_config == current_config
        extension_match = (extend and old_config.get("max_steps", 0) < config.max_steps
                           and {key: value for key, value in old_config.items() if key != "max_steps"}
                           == {key: value for key, value in current_config.items() if key != "max_steps"}
                           and saved.get("stop_reason", "step_cap") == "step_cap")
        if not (config_match or extension_match) or saved.get("seed") != task_seed:
            raise ValueError(f"bank resume protocol mismatch: {checkpoint_path}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.set_state(saved["sampler_rng"])
        step = int(saved["step"])
        best = saved["best"]
        best_score = saved["best_score"]
        best_step = saved["best_step"]
        curves = saved["curves"]
        converged = saved.get("converged", torch.zeros(n_models, dtype=torch.bool))
        plateau_passes = saved.get(
            "plateau_passes",
            torch.where(converged, torch.full((n_models,), config.plateau_passes),
                        torch.zeros(n_models, dtype=torch.int64)),
        )
        plateau_step = saved.get("plateau_step", torch.zeros(n_models, dtype=torch.int64))
    elif resume and prior_final is not None:
        saved = dict(prior_final)
        saved["model"] = saved["last"]
        saved["best_score"] = saved["best_query_balanced_bce"]
        if isinstance(saved["curves"]["step"], torch.Tensor):
            saved["curves"] = {key: list(value.unbind(0))
                                for key, value in saved["curves"].items()}
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.set_state(saved["sampler_rng"])
        step = int(saved["step"])
        best = saved["best"]
        best_score = saved["best_score"]
        best_step = saved["best_step"]
        curves = saved["curves"]
        converged = saved.get("converged_per_model", saved.get("converged"))
        if converged is None:
            converged = torch.full((n_models,), not saved.get("cap_hit", False),
                                   dtype=torch.bool)
        plateau_passes = saved.get(
            "plateau_passes",
            torch.where(converged, torch.full((n_models,), config.plateau_passes),
                        torch.zeros(n_models, dtype=torch.int64)),
        )
        plateau_step = saved.get("plateau_step", torch.zeros(n_models, dtype=torch.int64))

    density_order = torch.cat([
        torch.full((count,), density, dtype=torch.int64)
        for density, count in zip(config.densities, config.models_by_density)
    ])
    while step < config.max_steps:
        if step and step % config.decay_every == 0:
            lr = max(config.minimum_lr, config.learning_rate *
                     config.decay_factor ** (step // config.decay_every))
            for group in optimizer.param_groups:
                group["lr"] = lr
        n_pos = config.batch_size // 2
        pos_ids = pos[torch.randint(pos.numel(), (n_pos,), generator=rng)]
        neg_ids = neg[torch.randint(neg.numel(), (config.batch_size - n_pos,), generator=rng)]
        indices = torch.cat((pos_ids, neg_ids))
        indices = indices[torch.randperm(indices.numel(), generator=rng)]
        chosen = indices.to(device)
        xb = support_x.index_select(0, chosen)
        yb = support_y.index_select(0, chosen)
        logits = model(xb)
        per_model = F.binary_cross_entropy_with_logits(
            logits, yb[:, None].expand_as(logits), reduction="none").mean(dim=0)
        optimizer.zero_grad(set_to_none=True)
        per_model.sum().backward()
        with torch.no_grad():
            previous = {name: getattr(model, name)[converged.to(device)].clone()
                        for name in ("w1", "b1", "w2", "b2")}
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad[converged.to(device)] = 0
                state = optimizer.state.get(parameter, {})
                for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    if name in state:
                        state[name][converged.to(device)] = 0
        optimizer.step()
        with torch.no_grad():
            for name, old_values in previous.items():
                getattr(model, name)[converged.to(device)] = old_values
        step += 1

        if step % config.eval_every:
            continue
        train_metrics = _balanced_eval(model, support, device)
        query_metrics = _balanced_eval(model, query, device)
        curves["step"].append(torch.tensor(step, dtype=torch.int64))
        for key in ("natural_bce", "balanced_bce", "natural_accuracy", "balanced_accuracy"):
            curves[f"train_{key}"].append(train_metrics[key].detach().cpu())
            curves[f"query_{key}"].append(query_metrics[key].detach().cpu())
        query_score = query_metrics["balanced_bce"].detach().cpu()
        improved = query_score < best_score
        best_score = torch.minimum(best_score, query_score)
        best_step = torch.where(improved, torch.full_like(best_step, step), best_step)
        _copy_best(best, _cpu_state(model), improved)
        if step >= config.min_steps:
            train_plateau = loss_plateau(curves["train_balanced_bce"], width=8,
                                         tolerance=config.plateau_relative)
            query_plateau = loss_plateau(curves["query_balanced_bce"], width=8,
                                         tolerance=config.plateau_relative)
            plateau_now = train_plateau & query_plateau & ~converged
            plateau_passes = torch.where(plateau_now, plateau_passes + 1,
                                         torch.where(converged, plateau_passes,
                                                     torch.zeros_like(plateau_passes)))
            newly_converged = plateau_now & (plateau_passes >= config.plateau_passes)
            converged |= newly_converged
            plateau_step = torch.where(newly_converged,
                                       torch.full_like(plateau_step, step), plateau_step)
        if bool(converged.all()):
            stop_reason = "empirical_train_query_plateau"
        snapshot = {
            "protocol": "task_quality_source_bank_v1",
            "config": asdict(config), "seed": task_seed, "pattern": pattern,
            "step": step,
            "stop_reason": ("empirical_train_query_plateau" if bool(converged.all())
                            else ("step_cap" if step >= config.max_steps else "running")),
            "model": _cpu_state(model), "best": best,
            "best_score": best_score, "best_step": best_step, "optimizer": optimizer.state_dict(),
            "sampler_rng": rng.get_state(), "converged": converged,
            "plateau_passes": plateau_passes, "plateau_step": plateau_step,
            "curves": curves,
            "support_ids_sha256": _hash_ids(support["ids"]),
            "query_ids_sha256": _hash_ids(query["ids"]),
            "probe_ids_sha256": _hash_ids(probe_ids),
        }
        _atomic_save(snapshot, checkpoint_path)
        if stop_reason == "empirical_train_query_plateau":
            break

    if step >= config.max_steps and stop_reason != "empirical_train_query_plateau":
        stop_reason = "step_cap"
    last = _cpu_state(model)
    # A final curve row is retained even when the cap does not align to eval_every.
    if not curves["step"] or int(curves["step"][-1]) != step:
        train_metrics = _balanced_eval(model, support, device)
        query_metrics = _balanced_eval(model, query, device)
        curves["step"].append(torch.tensor(step, dtype=torch.int64))
        for key in ("natural_bce", "balanced_bce", "natural_accuracy", "balanced_accuracy"):
            curves[f"train_{key}"].append(train_metrics[key].detach().cpu())
            curves[f"query_{key}"].append(query_metrics[key].detach().cpu())
        final_score = query_metrics["balanced_bce"].detach().cpu()
        improved = final_score < best_score
        best_score = torch.minimum(best_score, final_score)
        best_step = torch.where(improved, torch.full_like(best_step, step), best_step)
        _copy_best(best, last, improved)
    stacked_curves = {key: torch.stack(values) for key, values in curves.items()}
    result = {
        "protocol": "task_quality_source_bank_v1",
        "config": asdict(config), "seed": task_seed, "pattern": pattern,
        "step": step, "stop_reason": stop_reason,
        "converged": bool(converged.all()), "converged_per_model": converged,
        "stop_reason_per_model": ["empirical_train_query_plateau" if value else "step_cap"
                                  for value in converged.tolist()],
        "plateau_step": plateau_step, "cap_hit": not bool(converged.all()),
        "selection_complete": True, "best": best, "last": last,
        "best_query_balanced_bce": best_score, "best_step": best_step,
        "optimizer": optimizer.state_dict(), "sampler_rng": rng.get_state(),
        "curves": stacked_curves, "plateau_passes": plateau_passes,
        "support_ids_sha256": _hash_ids(support["ids"]),
        "query_ids_sha256": _hash_ids(query["ids"]),
        "probe_ids": probe_ids.clone(), "probe_ids_sha256": _hash_ids(probe_ids),
        "density": density_order, "models_by_density": config.models_by_density,
    }
    _atomic_save(result, final_path)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    return result


def _functional_features(
    state: dict[str, torch.Tensor],
    probe_x: torch.Tensor,
    quality: torch.Tensor,
    density: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Build per-hidden-unit source features from probe activation gradients."""
    mask = state["masks"].float()
    w_eff = state["w1"] * mask
    bias = state["b1"]
    readout = state["w2"]
    preactivation = torch.einsum("pi,mih->pmh", probe_x, w_eff) + bias.unsqueeze(0)
    relu = F.relu(preactivation)
    psi = relu * readout.unsqueeze(0)
    active = (preactivation > 0).to(w_eff.dtype)
    q = (probe_x[:, None, :, None] * w_eff[None, :, :, :]
         * readout[None, :, None, :] * active[:, :, None, :])
    q_signed = q.mean(dim=0)
    q_abs = q.abs().mean(dim=0)
    q_var = q.var(dim=0, unbiased=False)

    # Feature order per neuron: complete probe response, three q summaries,
    # mask row, effective-weight row, bias, readout, density, source quality.
    features = torch.cat([
        psi.permute(1, 2, 0),
        q_signed.permute(0, 2, 1),
        q_abs.permute(0, 2, 1),
        q_var.permute(0, 2, 1),
        mask.permute(0, 2, 1),
        w_eff.permute(0, 2, 1),
        bias.unsqueeze(-1),
        readout.unsqueeze(-1),
        (density.float() / EDGES).view(-1, 1, 1).expand(-1, HIDDEN, 1),
        quality.view(-1, 1, 1).expand(-1, HIDDEN, 1),
    ], dim=-1)
    return features, {
        "probe_psi": psi.permute(1, 0, 2).contiguous(),
        "edge_q": q.permute(1, 0, 2, 3).contiguous(),
        "q_signed": q_signed,
        "q_abs": q_abs,
        "q_variance": q_var,
        "masks": mask,
        "weff": w_eff,
        "b": bias,
        "a": readout,
        "c": state["b2"],
    }


def build_bank(
    out: str | Path,
    seed: int = 8100,
    device: str | torch.device = "cpu",
    config: BankConfig = BankConfig(),
    resume: bool = True,
    extend: bool = False,
) -> Path:
    """Train source teachers, retain query-quality exemplars, and save bank.pt."""
    config.validate()
    out = Path(out)
    device = torch.device(device)
    data = build_experiment_data(task_seed=42, split_seed=1729,
                                 probe_size=config.probe_size, probe_seed=seed)
    probe_x = data["probe"]["x"].to(device)
    probe_ids = data["probe"]["ids"]
    teacher_results: list[dict[str, Any]] = []
    selected_states: list[dict[str, torch.Tensor]] = []
    selected_quality: list[torch.Tensor] = []
    selected_density: list[torch.Tensor] = []
    selected_pattern: list[str] = []
    selected_index: list[torch.Tensor] = []
    selection_curves: dict[str, Any] = {}
    source_patterns = [task.pattern for task in data["splits"]["train"]]
    for pattern in source_patterns:
        result = _train_pattern(pattern, data["pools"], probe_ids, out, seed,
                                device, config, resume, extend)
        teacher_results.append(result)
        selection_curves[pattern] = result["curves"]
        start = 0
        for density_index, (density, model_count) in enumerate(
                zip(config.densities, config.models_by_density)):
            stop = start + model_count
            scores = result["best_query_balanced_bce"][start:stop]
            keep = torch.argsort(scores, stable=True)[:config.keep_per_density] + start
            selection_curves[pattern].setdefault("selected_by_density", {})[density] = keep
            selected_states.append({key: result["best"][key].index_select(0, keep)
                                    for key in ("w1", "b1", "w2", "b2", "masks")})
            selected_quality.append(scores.index_select(0, keep - start))
            selected_density.append(torch.full((keep.numel(),), density, dtype=torch.int64))
            selected_pattern.extend([pattern] * keep.numel())
            selected_index.append(keep)
            start = stop

    flat_state = {key: torch.cat([state[key] for state in selected_states], dim=0)
                  for key in ("w1", "b1", "w2", "b2", "masks")}
    quality = torch.cat(selected_quality)
    density = torch.cat(selected_density)
    features, functional = _functional_features(flat_state, probe_x.detach().cpu(), quality, density)
    feature_mean = features.mean(dim=(0, 1), keepdim=True)
    feature_std = features.std(dim=(0, 1), unbiased=False, keepdim=True).clamp_min(1e-6)
    normalized = (features - feature_mean) / feature_std
    bank = {
        "protocol": "task_quality_functional_bank_v1",
        "seed": int(seed), "config": asdict(config),
        "feature": normalized.float().contiguous(),
        "raw_feature": features.float().contiguous(),
        "feature_mean": feature_mean.float(), "feature_std": feature_std.float(),
        "masks": functional["masks"].float(), "weff": functional["weff"].float(),
        "w": flat_state["w1"].float(), "b": functional["b"].float(),
        "a": functional["a"].float(), "c": functional["c"].float(),
        "probe_psi": functional["probe_psi"].float(),
        "edge_q": functional["edge_q"].float(),
        "q_signed": functional["q_signed"].float(),
        "q_abs": functional["q_abs"].float(),
        "q_variance": functional["q_variance"].float(),
        "density": density, "quality": quality.float(),
        "source_pattern": selected_pattern,
        "selected_index": torch.cat(selected_index),
        "probe_ids": probe_ids,
        "support_ids_sha256": {
            result["pattern"]: result["support_ids_sha256"] for result in teacher_results
        },
        "query_ids_sha256": {
            result["pattern"]: result["query_ids_sha256"] for result in teacher_results
        },
        "task_seed": data["task_seed"], "split_seed": data["split_seed"],
        "teacher_paths": [str(out / "teachers" / f"pattern_{p}.pt") for p in source_patterns],
        "teacher_summaries": [
            {"pattern": r["pattern"], "steps": r["step"], "stop_reason": r["stop_reason"],
             "converged": r["converged"], "cap_hit": r["cap_hit"],
             "mean_best_query_balanced_bce": float(r["best_query_balanced_bce"].mean())}
            for r in teacher_results
        ],
        "source_curves": selection_curves,
    }
    destination = out / "bank.pt"
    _atomic_save(bank, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=8100)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--models-by-density", type=int, nargs="+",
                        default=list(BankConfig.models_by_density))
    parser.add_argument("--keep-per-density", type=int, default=BankConfig.keep_per_density)
    parser.add_argument("--min-steps", type=int, default=BankConfig.min_steps)
    parser.add_argument("--max-steps", type=int, default=BankConfig.max_steps)
    parser.add_argument("--extend", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    config = BankConfig(models_by_density=tuple(args.models_by_density),
                        keep_per_density=args.keep_per_density,
                        min_steps=args.min_steps, max_steps=args.max_steps)
    path = build_bank(args.out, args.seed, args.device, config,
                      resume=not args.no_resume, extend=args.extend)
    print(path)


if __name__ == "__main__":
    main()
