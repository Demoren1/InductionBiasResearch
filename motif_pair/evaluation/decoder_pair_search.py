"""Agreement search between two independently trained frozen scalar CVAEs.

The search objective contains only the two decoded masks, after an optimal
hidden-column assignment.  Gold masks, task labels and importance maps are
strictly post-hoc diagnostics/baselines and cannot affect latent selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from data.generate import ideal_mask  # noqa: E402
from evaluation.baselines import TrainOnlyGapMean, gold_mask, random_exact  # noqa: E402
from evaluation.eval_generated_masks import _check_provenance, _load_model, _wrong_gaps  # noqa: E402
from evaluation.oracle_ideal import align_columns, hard_topk, soft_topk  # noqa: E402
from evaluation.structural import best_permutation_iou  # noqa: E402
from models.cvae import checkpoint_condition_metadata, read_split_provenance  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _project_(z: torch.Tensor, radius: float) -> None:
    with torch.no_grad():
        z.mul_((radius / z.norm(dim=1, keepdim=True).clamp_min(1e-12)).clamp(max=1.0))


def _task_for_gap(gap: int) -> config.Task:
    # The pair is arbitrary and deliberately never reaches either decoder.
    return config.Task("000", "001", int(gap))


def _condition_batch(model: torch.nn.Module, gaps: tuple[int, ...], n_starts: int,
                     device: torch.device) -> torch.Tensor:
    return torch.cat([
        model.condition([_task_for_gap(gap)], device=device).expand(n_starts, -1)
        for gap in gaps
    ])


def _decode(model: torch.nn.Module, z: torch.Tensor, condition: torch.Tensor,
            temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    logits = model.decode(z, condition)
    if logits.ndim != 2 or logits.shape[1] != config.MASK_DIM:
        raise ValueError("decoder output is not a flat 16x16 mask")
    return (soft_topk(logits, config.K_ACTIVE, temperature).reshape(-1, config.SEQ_LEN, config.H),
            hard_topk(logits, config.K_ACTIVE).reshape(-1, config.SEQ_LEN, config.H))


def _pair_values(model1: torch.nn.Module, model2: torch.nn.Module,
                 z1: torch.Tensor, z2: torch.Tensor, condition: torch.Tensor,
                 temperature: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    soft1, hard1 = _decode(model1, z1, condition, temperature)
    soft2, hard2 = _decode(model2, z2, condition, temperature)
    aligned_soft2 = align_columns(soft1, soft2)
    loss = (soft1 - aligned_soft2).square().mean((1, 2))
    with torch.no_grad():
        aligned_hard2 = align_columns(hard1, hard2)
        intersection = (hard1 * aligned_hard2).sum((1, 2))
        hard_iou = intersection / (2 * config.K_ACTIVE - intersection)
    return loss, hard_iou, hard1, hard2


def _freeze(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    model.eval()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return before


def _verify_frozen(model: torch.nn.Module, before: dict[str, torch.Tensor], label: str) -> None:
    for name, value in model.state_dict().items():
        if not torch.equal(before[name], value):
            raise AssertionError(f"{label} decoder state changed: {name}")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError(f"{label} decoder received gradients")


def optimize_agreement(model1: torch.nn.Module, model2: torch.nn.Module,
                        initial_z1: torch.Tensor, initial_z2: torch.Tensor,
                        condition: torch.Tensor, *, steps: int = 1000,
                        lr: float = .03, temperature: float = .5,
                        radius: float = 8.) -> dict[str, Any]:
    """Optimize two latent batches; selection is solely per-row soft agreement."""
    if initial_z1.shape != initial_z2.shape or initial_z1.ndim != 2:
        raise ValueError("initial latent tensors must have the same [batch, latent] shape")
    if condition.shape != (len(initial_z1), getattr(model1, "cond_dim", -1)):
        raise ValueError("condition shape does not match the latent batch/model")
    if getattr(model2, "cond_dim", None) != condition.shape[1]:
        raise ValueError("the two decoders have different condition dimensions")
    if steps < 0 or lr <= 0 or temperature <= 0 or radius <= 0:
        raise ValueError("invalid search settings")
    before1, before2 = _freeze(model1), _freeze(model2)
    z1 = torch.nn.Parameter(initial_z1.detach().clone())
    z2 = torch.nn.Parameter(initial_z2.detach().clone())
    _project_(z1, radius); _project_(z2, radius)
    start1, start2 = z1.detach().clone(), z2.detach().clone()
    optimizer = torch.optim.Adam([z1, z2], lr=lr)
    n = len(z1)
    best_loss = torch.full((n,), float("inf"), device=z1.device)
    best_z1, best_z2 = z1.detach().clone(), z2.detach().clone()
    best_step = torch.zeros(n, dtype=torch.long, device=z1.device)
    history: list[dict[str, float | int]] = []
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, hard_iou, _, _ = _pair_values(model1, model2, z1, z2, condition, temperature)
        if not bool(torch.isfinite(loss).all()):
            raise RuntimeError("non-finite agreement objective")
        with torch.no_grad():
            better = loss < best_loss
            best_loss = torch.where(better, loss, best_loss)
            best_z1[better], best_z2[better] = z1[better], z2[better]
            best_step[better] = step
            history.append({"step": step, "soft_pair_loss_mean": float(loss.mean()),
                            "hard_pair_iou_mean": float(hard_iou.mean())})
        if step % 100 == 0 or step == steps:
            print(f"[agreement] step={step}/{steps} loss={float(loss.mean()):.6f} "
                  f"hard-IoU={float(hard_iou.mean()):.4f}", flush=True)
        if step < steps:
            loss.sum().backward(); optimizer.step(); _project_(z1, radius); _project_(z2, radius)

    def pack(a: torch.Tensor, b: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            loss, hard_iou, hard1, hard2 = _pair_values(model1, model2, a, b, condition, temperature)
        return {"z1": a.detach().cpu(), "z2": b.detach().cpu(), "soft_pair_loss": loss.cpu(),
                "hard_pair_iou": hard_iou.cpu(), "mask1": hard1.flatten(1).cpu(), "mask2": hard2.flatten(1).cpu(),
                "norm1": a.detach().norm(dim=1).cpu(), "norm2": b.detach().norm(dim=1).cpu()}

    _verify_frozen(model1, before1, "first"); _verify_frozen(model2, before2, "second")
    return {"prior": pack(start1, start2), "best": pack(best_z1, best_z2),
            "best_step": best_step.cpu(), "history": history, "decoder_unchanged": True}


@torch.no_grad()
def random_pair_search(model1: torch.nn.Module, model2: torch.nn.Module,
                       proposals_z1: torch.Tensor, proposals_z2: torch.Tensor,
                       condition: torch.Tensor, *, temperature: float = .5,
                       radius: float = 8.) -> dict[str, torch.Tensor]:
    """Choose the best of independent paired latent proposals by soft agreement."""
    if proposals_z1.shape != proposals_z2.shape or proposals_z1.ndim != 3:
        raise ValueError("proposals must have shape [proposals, batch, latent]")
    p, n, _ = proposals_z1.shape
    if condition.shape[0] != n or p < 1:
        raise ValueError("proposal/batch shape mismatch")
    best_loss = torch.full((n,), float("inf"), device=condition.device)
    best_z1 = torch.empty_like(proposals_z1[0], device=condition.device)
    best_z2 = torch.empty_like(proposals_z2[0], device=condition.device)
    best_index = torch.zeros(n, dtype=torch.long, device=condition.device)
    for ordinal in range(p):
        z1 = proposals_z1[ordinal].to(condition.device).clone()
        z2 = proposals_z2[ordinal].to(condition.device).clone()
        _project_(z1, radius); _project_(z2, radius)
        loss, _, _, _ = _pair_values(model1, model2, z1, z2, condition, temperature)
        better = loss < best_loss
        best_loss = torch.where(better, loss, best_loss)
        best_z1[better], best_z2[better] = z1[better], z2[better]
        best_index[better] = ordinal
        if ordinal % 100 == 0 or ordinal + 1 == p:
            print(f"[random-pair] proposal={ordinal + 1}/{p} loss={float(loss.mean()):.6f}", flush=True)
    loss, hard_iou, hard1, hard2 = _pair_values(model1, model2, best_z1, best_z2, condition, temperature)
    return {"z1": best_z1.cpu(), "z2": best_z2.cpu(), "soft_pair_loss": loss.cpu(),
            "hard_pair_iou": hard_iou.cpu(), "mask1": hard1.flatten(1).cpu(), "mask2": hard2.flatten(1).cpu(),
            "norm1": best_z1.norm(dim=1).cpu(), "norm2": best_z2.norm(dim=1).cpu(),
            "proposal_index": best_index.cpu()}


def _validate_checkpoints(path1: Path, path2: Path, split_path: Path, device: torch.device,
                          top_frac: float, importance_name: str):
    expected_split = read_split_provenance(split_path)
    model1, payload1 = _load_model(path1, device); model2, payload2 = _load_model(path2, device)
    for label, payload, model in (("checkpoint1", payload1, model1), ("checkpoint2", payload2, model2)):
        _check_provenance(payload, expected_split, label, importance_name=importance_name, top_frac=top_frac)
        expected = config.condition_metadata(config.CONDITION_ENCODING_SCALAR)
        if (model.cond_dim != 1 or payload.get("condition_encoding") != config.CONDITION_ENCODING_SCALAR
                or checkpoint_condition_metadata(payload) != expected
                or any(payload.get(k) != v for k, v in expected.items())):
            raise ValueError(f"{label} must be an explicitly scalar one-dimensional CVAE")
    same_keys = ("mask_dim", "latent_dim", "hidden", "cond_dim", "beta", "importance_name", "top_frac",
                 "split_sha256", "split_train_tasks", "train_tasks")
    mismatched = [key for key in same_keys if payload1.get(key) != payload2.get(key)]
    if mismatched:
        raise ValueError(f"checkpoint metadata mismatch: {mismatched}")
    if payload1.get("seed") != 42 or payload2.get("seed") != 43:
        raise ValueError("expected independently initialized checkpoint seeds 42 and 43")
    # Old seed-42 checkpoints did not persist this field; their only loader
    # seed was the training seed.  A new seed-43 replicate must state it.
    loader1 = int(payload1.get("loader_seed", 42))
    loader2 = payload2.get("loader_seed")
    if loader1 != 42 or loader2 != 42:
        raise ValueError("both CVAEs must use loader_seed=42")
    return model1, model2, payload1, payload2, expected_split


def _validate_protocol(protocol_path: Path, args: argparse.Namespace,
                       payload1: dict[str, Any], expected_split: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Bind the run to the pre-registered first-checkpoint profile/settings."""
    protocol = json.loads(protocol_path.read_text())
    matching = [(name, profile) for name, profile in protocol.get("profiles", {}).items()
                if Path(profile.get("checkpoint", "")).resolve() == args.checkpoint1.resolve()
                and Path(profile.get("split", "")).resolve() == args.split.resolve()]
    if len(matching) != 1:
        raise ValueError("checkpoint1/split must identify one pre-registered protocol profile")
    name, profile = matching[0]
    if profile.get("checkpoint_sha256") != _sha256(args.checkpoint1):
        raise ValueError("checkpoint1 hash does not match protocol")
    if profile.get("split_sha256") != expected_split["split_sha256"]:
        raise ValueError("split hash does not match protocol")
    if float(profile.get("beta")) != float(payload1.get("beta")):
        raise ValueError("checkpoint beta does not match protocol profile")
    declared = protocol.get("search", {})
    current = {"seed": args.seed, "n_starts": args.n_starts, "steps": args.steps,
               "lr": args.lr, "temperature": args.temperature, "radius": args.radius,
               "random_pair_proposals": args.random_proposals, "k": config.K_ACTIVE,
               "gaps": list(config.GAPS)}
    expected = {key: declared.get(key) for key in current}
    if current != expected:
        raise ValueError(f"search settings differ from protocol: current={current}, protocol={expected}")
    return name, protocol


def _split_blocks(value: torch.Tensor, n_gaps: int, n_starts: int) -> torch.Tensor:
    return value.reshape(n_gaps, n_starts, *value.shape[1:]).detach().cpu()


def _repeat_proposals_across_gaps(proposals: torch.Tensor, n_gaps: int) -> torch.Tensor:
    """Make gap-major batches while preserving each proposal/start verbatim."""
    if proposals.ndim != 3 or n_gaps < 1:
        raise ValueError("expected [proposals, starts, latent] and a positive gap count")
    return proposals[:, None].expand(-1, n_gaps, -1, -1).reshape(
        len(proposals), n_gaps * proposals.shape[1], proposals.shape[2])


def _structure(mask: torch.Tensor, gap: int) -> dict[str, Any]:
    gold = ideal_mask(_task_for_gap(gap)).float()
    return best_permutation_iou(mask.reshape(config.SEQ_LEN, config.H).cpu(), gold)


def _method_metrics(masks: torch.Tensor, gaps: tuple[int, ...]) -> dict[str, Any]:
    return {str(gap): {"mean_best_permutation_iou": float(sum(
        _structure(mask, gap)["iou"] for mask in masks[i]) / len(masks[i])),
        "structural": [_structure(mask, gap) for mask in masks[i]]}
            for i, gap in enumerate(gaps)}


def _wrong_condition_masks(model: torch.nn.Module, z: torch.Tensor, target_gap: int,
                           train_gaps: tuple[int, ...], device: torch.device) -> dict[str, torch.Tensor]:
    wrong = _wrong_gaps(_task_for_gap(target_gap).id, train_gaps)
    result = {}
    for gap in wrong:
        c = model.condition([_task_for_gap(gap)], device=device).expand(len(z), -1)
        _, hard = _decode(model, z.to(device), c, .5)
        result[str(gap)] = hard.flatten(1).cpu()
    return result


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run outside the sandbox")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    device = torch.device(args.device)
    model1, model2, payload1, payload2, split_provenance = _validate_checkpoints(
        args.checkpoint1, args.checkpoint2, args.split, device, args.top_frac, args.importance_name)
    protocol_profile, _ = _validate_protocol(args.protocol, args, payload1, split_provenance)
    split = json.loads(args.split.read_text())
    train_gaps = tuple(sorted({config.parse_task(task).gap for task in split_provenance["split_train_tasks"]}))
    gaps = tuple(config.GAPS)
    args.out.mkdir(parents=True, exist_ok=False)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    starts1 = torch.randn(args.n_starts, model1.latent_dim, generator=generator)
    starts2 = torch.randn(args.n_starts, model1.latent_dim, generator=generator)
    initial1, initial2 = starts1.repeat(len(gaps), 1).to(device), starts2.repeat(len(gaps), 1).to(device)
    condition = _condition_batch(model1, gaps, args.n_starts, device)
    agreed = optimize_agreement(model1, model2, initial1, initial2, condition, steps=args.steps,
                                lr=args.lr, temperature=args.temperature, radius=args.radius)
    # Each proposal has one start per ordinal, copied verbatim to every gap.
    proposals1 = torch.randn(args.random_proposals, args.n_starts, model1.latent_dim, generator=generator)
    proposals2 = torch.randn(args.random_proposals, args.n_starts, model1.latent_dim, generator=generator)
    random1 = _repeat_proposals_across_gaps(proposals1, len(gaps))
    random2 = _repeat_proposals_across_gaps(proposals2, len(gaps))
    random = random_pair_search(model1, model2, random1, random2, condition,
                                temperature=args.temperature, radius=args.radius)

    def masks_from(record: dict[str, torch.Tensor], key: str) -> torch.Tensor:
        return _split_blocks(record[key], len(gaps), args.n_starts)
    masks = {"prior1": masks_from(agreed["prior"], "mask1"), "prior2": masks_from(agreed["prior"], "mask2"),
             "agreement1": masks_from(agreed["best"], "mask1"), "agreement2": masks_from(agreed["best"], "mask2"),
             "random_search1": masks_from(random, "mask1"), "random_search2": masks_from(random, "mask2")}
    mean = TrainOnlyGapMean(split_provenance["split_train_tasks"], ckpt_root=args.ckpt_root,
                            importance_name=args.importance_name, top_frac=args.top_frac,
                            device=device, split_path=args.split,
                            condition_encoding=config.CONDITION_ENCODING_SCALAR)
    masks["conditional_mean"] = torch.stack([mean(_task_for_gap(gap)) for gap in gaps]).unsqueeze(1).expand(
        -1, args.n_starts, -1).clone().cpu()
    masks["random_exact96"] = random_exact(len(gaps) * args.n_starts,
                                             generator=torch.Generator(device="cpu").manual_seed(args.seed + 40_000)).reshape(len(gaps), args.n_starts, -1)
    masks["ideal"] = torch.stack([gold_mask(_task_for_gap(gap)) for gap in gaps]).unsqueeze(1).expand(-1, args.n_starts, -1).clone()
    pair_stats = {"prior": {key: _split_blocks(agreed["prior"][key], len(gaps), args.n_starts)
                             for key in ("soft_pair_loss", "hard_pair_iou", "norm1", "norm2")},
                  "agreement": {key: _split_blocks(agreed["best"][key], len(gaps), args.n_starts)
                                for key in ("soft_pair_loss", "hard_pair_iou", "norm1", "norm2")},
                  "random_search": {key: _split_blocks(random[key], len(gaps), args.n_starts)
                                    for key in ("soft_pair_loss", "hard_pair_iou", "norm1", "norm2", "proposal_index")}}
    wrong: dict[str, Any] = {}
    for method, record, zkey in (("prior1", agreed["prior"], "z1"), ("agreement1", agreed["best"], "z1"),
                                 ("random_search1", random, "z1")):
        z = _split_blocks(record[zkey], len(gaps), args.n_starts)
        wrong[method] = {str(gap): _wrong_condition_masks(model1, z[i], gap, train_gaps, device)
                         for i, gap in enumerate(gaps)}
    metrics = {method: _method_metrics(value, gaps) for method, value in masks.items()}
    for method, by_target in wrong.items():
        for gap, by_wrong_gap in by_target.items():
            metrics[method][gap]["wrong_condition"] = {
                condition_gap: _method_metrics(block.unsqueeze(0), (int(gap),))[gap]
                for condition_gap, block in by_wrong_gap.items()}
    metadata = {"experiment": "two_frozen_scalar_cvae_latent_agreement", "no_gold_in_optimizer": True,
                "checkpoint1": str(args.checkpoint1.resolve()), "checkpoint2": str(args.checkpoint2.resolve()),
                "checkpoint1_sha256": _sha256(args.checkpoint1), "checkpoint2_sha256": _sha256(args.checkpoint2),
                "split": str(args.split.resolve()), **split_provenance,
                "checkpoint_seeds": [payload1["seed"], payload2["seed"]], "loader_seed": 42,
                "old_seed42_loader_seed_inferred": "loader_seed" not in payload1,
                "protocol": str(args.protocol.resolve()), "protocol_sha256": _sha256(args.protocol),
                "protocol_profile": protocol_profile,
                "source_sha256": _sha256(Path(__file__)), "settings": {"n_starts": args.n_starts,
                    "steps": args.steps, "lr": args.lr, "temperature": args.temperature, "radius": args.radius,
                    "random_proposals": args.random_proposals, "seed": args.seed},
                "gaps": list(gaps), "train_gaps": list(train_gaps),
                "wrong_gap_policy": "nearest structurally non-equivalent train gaps; retain ties; same z"}
    torch.save({"metadata": metadata, "gaps": torch.tensor(gaps), "masks": masks, "latents": {
        "prior": {k: agreed["prior"][k] for k in ("z1", "z2")}, "agreement": {k: agreed["best"][k] for k in ("z1", "z2")},
        "random_search": {k: random[k] for k in ("z1", "z2")}}, "pair_stats": pair_stats,
        "history": agreed["history"], "best_step": agreed["best_step"]}, args.out / "masks.pt")
    serial_pair = {name: {key: value.tolist() for key, value in values.items()} for name, values in pair_stats.items()}
    summary = {"metadata": metadata, "metrics": metrics, "pair_stats": serial_pair,
               "conditional_mean": {"policy": mean.interpolation_policy, "available_train_gaps": list(mean.available_gaps),
                                    "per_gap": {str(g): mean.describe_gap(g) for g in gaps}}}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"saved {args.out / 'masks.pt'}", flush=True)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint1", type=Path, required=True); p.add_argument("--checkpoint2", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--ckpt-root", type=Path, required=True, help="profile-local MLP importance checkpoint root")
    p.add_argument("--device", choices=("cuda",), default="cuda")
    p.add_argument("--n-starts", type=int, default=64); p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--random-proposals", type=int, default=1001); p.add_argument("--seed", type=int, default=20260906)
    p.add_argument("--lr", type=float, default=.03); p.add_argument("--temperature", type=float, default=.5); p.add_argument("--radius", type=float, default=8.)
    p.add_argument("--top-frac", type=float, default=.1); p.add_argument("--importance-name", default="importance.pt")
    return p


if __name__ == "__main__":
    parsed = parser().parse_args(); torch.set_num_threads(2); torch.use_deterministic_algorithms(True); run(parsed)
