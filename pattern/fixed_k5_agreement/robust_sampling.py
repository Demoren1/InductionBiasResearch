"""R=4 robust binary-mask latent sampling for the fixed seq-32/k5 VAEs."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from pattern.evaluation.decoder_agreement import align_columns, hard_topk
from pattern.evaluation.oracle_ideal import optimize_ideal
from pattern.evaluation.z_star_noise import ci
from pattern.evaluation.z_star_sampling_r4 import full_schedule

from .common import (atomic_save, ideal_mask, load_protocol, pair_dir, seed_for,
                     sha256_file, write_json)
from .model import load_model


@dataclass(frozen=True)
class Settings:
    starts_per_group: int = 64
    oracle_steps: int = 2000
    oracle_lr: float = 0.03
    temperature: float = 0.5
    radius: float = 4.0
    radii_schedule: tuple[tuple[float, ...], ...] = full_schedule()
    screen_steps: int = 400
    refine_steps: int = 2000
    refine_replicates: int = 2
    finalists: int = 4
    final_eval_steps: int = 2000
    min_improvement: float = 1e-4
    seed: int = 20260923


def _project(latent: torch.Tensor, radius: float) -> torch.Tensor:
    norm = latent.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(latent.dtype).tiny)
    return latent * (radius / norm).clamp(max=1.0)


def sample_candidates(parents: torch.Tensor, radii: Sequence[float], radius: float,
                      seed: int) -> torch.Tensor:
    generator = torch.Generator(device=parents.device).manual_seed(seed)
    directions = torch.randn(len(parents), len(radii), parents.size(1),
                             generator=generator, device=parents.device)
    directions /= directions.norm(dim=2, keepdim=True).clamp_min(
        torch.finfo(directions.dtype).tiny)
    mutations = _project(parents[:, None] + parents.new_tensor(radii)[None, :, None] * directions,
                         radius)
    return torch.cat([parents[:, None], mutations], dim=1)


def unique_mask_counts(masks: torch.Tensor) -> list[int]:
    flat = masks.reshape(masks.size(0), masks.size(1), -1).cpu()
    return [int(torch.unique(row, dim=0).size(0)) for row in flat]


def _fresh(n: int, seq_len: int, hidden: int, seed: int,
           device: torch.device) -> list[torch.nn.Parameter]:
    generator = torch.Generator(device=device).manual_seed(seed)
    return [
        torch.nn.Parameter(torch.randn(n, seq_len, hidden, generator=generator, device=device) * .1),
        torch.nn.Parameter(torch.zeros(n, hidden, device=device)),
        torch.nn.Parameter(torch.randn(n, hidden, generator=generator, device=device) * .1),
        torch.nn.Parameter(torch.zeros(n, device=device)),
    ]


def _paired_params(parents: int, candidates: int, seq_len: int, hidden: int,
                   seed: int, device: torch.device) -> list[torch.nn.Parameter]:
    base = _fresh(parents, seq_len, hidden, seed, device)
    return [torch.nn.Parameter(value.detach().repeat_interleave(candidates, dim=0)) for value in base]


def _forward(x: torch.Tensor, params: list[torch.Tensor], masks: torch.Tensor) -> torch.Tensor:
    w1, b1, w2, b2 = params
    hidden = F.relu(torch.einsum("bl,nlh->bnh", x, w1 * masks) + b1)
    return torch.einsum("bnh,nh->bn", hidden, w2) + b2


def _bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        logits, labels[:, None].expand_as(logits), reduction="none").mean(0)


def fit_and_score(masks: torch.Tensor, data: dict, config, device: torch.device, *,
                  steps: int, model_seed: int, batch_seed: int,
                  split: str, return_accuracy: bool = False) -> dict[str, torch.Tensor]:
    if masks.ndim != 4 or masks.shape[-2:] != (config.seq_len, config.hidden):
        raise ValueError("invalid mask shape")
    if not torch.equal(masks, masks.round()) or not (masks.sum((-1, -2)) == config.k_active).all():
        raise ValueError("only binary exact-K masks are accepted")
    if split == "search":
        train, evaluation = data["task_support"], data["task_query"]
    elif split == "final":
        train, evaluation = data["eval_support"], data["eval_test"]
    else:
        raise ValueError(split)
    parents, candidates = masks.shape[:2]
    flat = masks.to(device).reshape(-1, config.seq_len, config.hidden)
    params = _paired_params(parents, candidates, config.seq_len, config.hidden,
                            model_seed, device)
    optimizer = torch.optim.Adam(params, lr=config.eval_lr)
    train_x, train_y = train["x"].to(device), train["y"].to(device)
    generator = torch.Generator().manual_seed(batch_seed)
    for _ in range(steps):
        index = torch.randint(len(train_x), (config.eval_batch,), generator=generator).to(device)
        loss = _bce(_forward(train_x[index], params, flat), train_y[index])
        optimizer.zero_grad(set_to_none=True)
        loss.sum().backward()
        optimizer.step()
    eval_x, eval_y = evaluation["x"].to(device), evaluation["y"].to(device)
    total_bce = torch.zeros(len(flat), device=device)
    total_correct = torch.zeros(len(flat), device=device)
    with torch.no_grad():
        for start in range(0, len(eval_x), 512):
            xb, yb = eval_x[start:start + 512], eval_y[start:start + 512]
            logits = _forward(xb, params, flat)
            total_bce += F.binary_cross_entropy_with_logits(
                logits, yb[:, None].expand_as(logits), reduction="none").sum(0)
            if return_accuracy:
                total_correct += ((logits > 0) == yb.bool()[:, None]).sum(0)
    result = {"bce": (total_bce / len(eval_x)).reshape(parents, candidates).cpu()}
    if return_accuracy:
        result["accuracy"] = (total_correct / len(eval_x)).reshape(parents, candidates).cpu()
    return result


def decode_hard(model: torch.nn.Module, latent: torch.Tensor, config) -> torch.Tensor:
    shape = latent.shape
    flat = latent.reshape(-1, shape[-1])
    with torch.no_grad():
        logits = model.decode(flat, flat.new_empty(len(flat), 0))
        masks = hard_topk(logits, config.k_active).reshape(-1, config.seq_len, config.hidden)
    return masks.reshape(*shape[:-1], config.seq_len, config.hidden)


def robust_winners(scores: torch.Tensor, masks: torch.Tensor,
                   min_improvement: float) -> tuple[torch.Tensor, torch.Tensor]:
    mean_scores = scores.mean(0)
    mutation_index = mean_scores[:, 1:].argmin(1) + 1
    parents = torch.arange(scores.size(1), device=scores.device)
    candidate = scores[:, parents, mutation_index]
    parent = scores[:, :, 0]
    differs = (masks[parents, mutation_index] != masks[:, 0]).any(-1).any(-1)
    accepted = (candidate < parent - min_improvement).all(0) & differs
    return torch.where(accepted, mutation_index, torch.zeros_like(mutation_index)), accepted


def evolutionary_search(model: torch.nn.Module, initial_z: torch.Tensor, pattern: str,
                        data: dict, config, device: torch.device, settings: Settings,
                        seed: int) -> dict[str, Any]:
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    parents = initial_z.detach().to(device).float()
    norms = parents.norm(dim=1, keepdim=True).clamp_min(torch.finfo(parents.dtype).tiny)
    parents = parents * (settings.radius / norms).clamp(max=1.)
    initial = parents.cpu().clone()
    initial_masks = decode_hard(model, parents, config).cpu()
    history = []
    pat_int = int(pattern, 2)
    for generation, radii in enumerate(settings.radii_schedule):
        candidates_z = sample_candidates(
            parents, radii, settings.radius,
            seed_for("k5-sampling", seed, pat_int, generation))
        masks = decode_hard(model, candidates_z, config)
        screen = fit_and_score(
            masks, data, config, device, steps=settings.screen_steps,
            model_seed=seed_for("k5-screen-model", seed, pat_int, generation),
            batch_seed=seed_for("k5-screen-batch", seed, pat_int, generation), split="search")["bce"].to(device)
        mutation_indices = screen[:, 1:].topk(settings.finalists - 1, dim=1, largest=False).indices + 1
        refined_indices = torch.cat([
            torch.zeros(len(parents), 1, dtype=torch.long, device=device), mutation_indices], dim=1)
        batch = torch.arange(len(parents), device=device)[:, None]
        refined_z = candidates_z[batch, refined_indices]
        refined_masks = masks[batch, refined_indices]
        scores = []
        for replicate in range(settings.refine_replicates):
            scores.append(fit_and_score(
                refined_masks, data, config, device, steps=settings.refine_steps,
                model_seed=seed_for("k5-refine-model", seed, pat_int, generation, replicate),
                batch_seed=seed_for("k5-refine-batch", seed, pat_int, generation, replicate),
                split="search")["bce"])
        scores = torch.stack(scores).to(device)
        winners, accepted = robust_winners(scores, refined_masks, settings.min_improvement)
        previous = parents
        parents = refined_z[torch.arange(len(parents), device=device), winners].detach()
        different = (masks[:, 1:] != masks[:, :1]).any(-1).any(-1)
        history.append({
            "generation": generation, "radii": list(map(float, radii)),
            "accepted": accepted.cpu(), "accepted_fraction": float(accepted.float().mean()),
            "candidate_different_support_fraction": float(different.float().mean()),
            "unique_mask_counts": unique_mask_counts(masks), "screen_parent_bce": screen[:, 0].cpu(),
            "screen_best_mutation_bce": screen[:, 1:].min(1).values.cpu(),
            "refine_scores": scores.cpu(), "selected_candidate_index": refined_indices[
                torch.arange(len(parents), device=device), winners].cpu(),
            "step_l2": (parents - previous).norm(dim=1).cpu(),
        })
        print(f"[k5-robust] pattern={pattern} generation={generation + 1}/{len(settings.radii_schedule)} "
              f"accepted={float(accepted.float().mean()):.3f}", flush=True)
    final = parents.cpu()
    final_masks = decode_hard(model, parents, config).cpu()
    final_eval = fit_and_score(
        torch.stack([initial_masks, final_masks], 1), data, config, device,
        steps=settings.final_eval_steps,
        model_seed=seed_for("k5-final-model", seed, pat_int),
        batch_seed=seed_for("k5-final-batch", seed, pat_int), split="final", return_accuracy=True)
    for name, value in model.state_dict().items():
        if not torch.equal(before[name], value.detach().cpu()):
            raise AssertionError(f"frozen decoder changed: {name}")
    return {
        "initial_z": initial, "final_z": final, "initial_masks": initial_masks,
        "final_masks": final_masks, "history": history,
        "final_eval_initial_bce": final_eval["bce"][:, 0],
        "final_eval_final_bce": final_eval["bce"][:, 1],
        "final_eval_initial_accuracy": final_eval["accuracy"][:, 0],
        "final_eval_final_accuracy": final_eval["accuracy"][:, 1],
        "decoder_unchanged": True, "settings": asdict(settings), "uses_gold": False,
    }


def _structure(masks: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    targets = target[None].expand(len(masks), -1, -1).to(masks)
    aligned = align_columns(targets.float(), masks.float())
    intersection = (targets * aligned).sum((1, 2))
    k = float(target.sum())
    iou = intersection / (2 * k - intersection)
    exact = (aligned == targets).all(2).all(1)
    return iou.cpu(), exact.cpu()


def _describe(values: torch.Tensor) -> dict:
    values = values.float().cpu()
    return {"mean": float(values.mean()), "std": float(values.std(unbiased=False)),
            "min": float(values.min()), "max": float(values.max())}


def _analyze(result: dict, group_slice: slice, keep: torch.Tensor,
             endpoints: torch.Tensor, target: torch.Tensor) -> dict:
    output = {}
    for stage in ("initial", "final"):
        masks = result[f"{stage}_masks"][group_slice][keep]
        latent = result[f"{stage}_z"][group_slice][keep]
        iou, exact = _structure(masks, target)
        prefix = "initial" if stage == "initial" else "final"
        output[stage] = {
            "count": len(masks), "exact_fraction": float(exact.float().mean()),
            "exact_count": int(exact.sum()), "gold_iou": _describe(iou),
            "task_bce_2000": _describe(result[f"final_eval_{prefix}_bce"][group_slice][keep]),
            "task_accuracy_2000": _describe(result[f"final_eval_{prefix}_accuracy"][group_slice][keep]),
            "paired_oracle_distance": _describe((latent - endpoints[keep]).norm(dim=1)),
        }
    accepted = torch.stack([row["accepted"] for row in result["history"]])[:, group_slice][:, keep]
    output["search"] = {
        "ever_accepted_fraction": float(accepted.any(0).float().mean()),
        "mean_acceptances_per_start": float(accepted.sum(0).float().mean()),
    }
    return output


def prepare(out: Path, parent: Path) -> None:
    config, source = load_protocol(parent)
    if len(config.pairs) != 32:
        raise ValueError("the seq32 sampling study requires exactly 32 VAE pairs")
    settings = Settings()
    artifacts = {}
    for pair in config.pairs:
        for seed in pair:
            root = pair_dir(parent, pair) / f"vae_{seed}"
            checkpoint, metadata = root / "best.pt", root / "metadata.json"
            if not checkpoint.is_file() or not metadata.is_file():
                raise FileNotFoundError(f"missing VAE artifact for seed {seed}")
            artifacts[str(seed)] = {
                "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                "metadata": str(metadata), "metadata_sha256": sha256_file(metadata),
            }
    payload = {
        "experiment": "R=4 robust hard-mask sampling, seq_len=32, pattern_len=5",
        "settings": asdict(settings), "parent": str(parent),
        "parent_protocol_sha256": sha256_file(parent / "protocol.json"),
        "pairs": [list(pair) for pair in config.pairs],
        "model_seeds": list(config.vae_seeds),
        "patterns": source["task_split"]["test_patterns"], "artifacts": artifacts,
        "oracle": {"uses_gold": True, "target": "analytic exact-160 ideal", "radius": 4.0},
        "groups": ["prior", "oracle_best_r4"],
        "sampling": {"uses_gold": False, "masks": "hard binary exact-160",
                     "selection_data": "task_support/task_query",
                     "final_data": "disjoint eval_support/eval_test"},
        "independence_unit": "VAE pair; two decoders, tasks and starts nested",
    }
    path = out / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise FileExistsError(f"{path} differs from immutable protocol")
        return
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    write_json(path, payload)


def _load_sampling(out: Path) -> tuple[Settings, dict, Any]:
    protocol = json.loads((out / "protocol.json").read_text())
    raw = dict(protocol["settings"])
    raw["radii_schedule"] = tuple(tuple(row) for row in raw["radii_schedule"])
    settings = Settings(**raw)
    config, _ = load_protocol(protocol["parent"])
    return settings, protocol, config


def run_seed(out: Path, seed: int, device_name: str) -> None:
    settings, protocol, config = _load_sampling(out)
    if seed not in protocol["model_seeds"]:
        raise ValueError(f"seed {seed} absent from protocol")
    root = out / f"seed_{seed}"
    summary_path = root / "summary.json"
    if summary_path.exists():
        print(f"[k5-robust] seed={seed}: complete", flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    artifact = protocol["artifacts"][str(seed)]
    if sha256_file(artifact["checkpoint"]) != artifact["checkpoint_sha256"]:
        raise ValueError("checkpoint changed")
    model = load_model(artifact["checkpoint"], device)
    generator = torch.Generator().manual_seed(seed_for("k5-robust-starts", config.task_split_seed))
    initial = torch.randn(settings.starts_per_group, config.latent_dim, generator=generator).to(device)
    target = ideal_mask(config).to(device)
    oracle_path = root / "oracle_same_start.pt"
    if oracle_path.exists():
        oracle = torch.load(oracle_path, map_location="cpu", weights_only=True)
    else:
        oracle = optimize_ideal(model, initial, target, steps=settings.oracle_steps,
                                lr=settings.oracle_lr, temperature=settings.temperature,
                                radius=settings.radius)
        oracle.update({"model_seed": seed, "checkpoint_sha256": artifact["checkpoint_sha256"]})
        atomic_save(oracle_path, oracle)
    endpoints = oracle["best_soft"]["z"].float()
    exact_oracle = oracle["best_soft"]["iou"] == 1
    all_oracle = torch.ones(settings.starts_per_group, dtype=torch.bool)
    combined = torch.cat([initial.cpu(), endpoints])
    tasks = {}
    for pattern in protocol["patterns"]:
        path = root / f"task_{pattern}.pt"
        if path.exists():
            result = torch.load(path, map_location="cpu", weights_only=True)
        else:
            data = torch.load(Path(protocol["parent"]) / "task_data" / f"pattern_{pattern}.pt",
                              map_location="cpu", weights_only=True)
            result = evolutionary_search(model, combined, pattern, data, config, device,
                                         settings, settings.seed + seed)
            result.update({"model_seed": seed, "pattern": pattern,
                           "checkpoint_sha256": artifact["checkpoint_sha256"],
                           "oracle_sha256": sha256_file(oracle_path)})
            atomic_save(path, result)
        tasks[pattern] = result
    n = settings.starts_per_group
    summary = {"model_seed": seed, "oracle_exact_count": int(exact_oracle.sum()), "tasks": {}}
    for pattern, result in tasks.items():
        summary["tasks"][pattern] = {
            "prior": _analyze(result, slice(0, n), torch.ones(n, dtype=torch.bool), endpoints, target.cpu()),
            "oracle_best_r4": _analyze(result, slice(n, 2 * n), all_oracle, endpoints, target.cpu()),
        }
    write_json(summary_path, summary)
    print(f"[k5-robust] seed={seed}: complete; exact oracle={int(exact_oracle.sum())}/{n}", flush=True)


def aggregate(out: Path) -> dict:
    settings, protocol, _ = _load_sampling(out)
    records = {seed: json.loads((out / f"seed_{seed}/summary.json").read_text())
               for seed in protocol["model_seeds"]}
    metrics = ("exact_fraction", "gold_iou", "paired_oracle_distance",
               "task_bce_2000", "task_accuracy_2000")
    output = {"protocol": protocol, "groups": {}, "oracle": {}}
    count = sum(row["oracle_exact_count"] for row in records.values())
    total = len(records) * settings.starts_per_group
    output["oracle"] = {"exact_count": count, "total": total, "fraction": count / total}
    for group in protocol["groups"]:
        output["groups"][group] = {}
        for stage in ("initial", "final"):
            per_pair = []
            for pair in protocol["pairs"]:
                decoders = []
                for seed in pair:
                    task_rows = [records[seed]["tasks"][pattern][group][stage]
                                 for pattern in protocol["patterns"]]
                    decoders.append({metric: float(np.mean([
                        row[metric] if metric == "exact_fraction" else row[metric]["mean"]
                        for row in task_rows])) for metric in metrics})
                per_pair.append({metric: float(np.mean([row[metric] for row in decoders]))
                                 for metric in metrics})
            output["groups"][group][stage] = {
                metric: ci([row[metric] for row in per_pair]) for metric in metrics}
        per_pair_search = []
        for pair in protocol["pairs"]:
            decoders = []
            for seed in pair:
                rows = [records[seed]["tasks"][pattern][group]["search"]
                        for pattern in protocol["patterns"]]
                decoders.append({key: float(np.mean([row[key] for row in rows])) for key in rows[0]})
            per_pair_search.append({key: float(np.mean([row[key] for row in decoders]))
                                    for key in decoders[0]})
        output["groups"][group]["search"] = {
            key: ci([row[key] for row in per_pair_search]) for key in per_pair_search[0]}
    return output


def _fmt(item: dict) -> str:
    return f"{item['mean']:.4f} [{item['ci95'][0]:.4f}; {item['ci95'][1]:.4f}]"


def report(out: Path) -> None:
    summary = aggregate(out)
    write_json(out / "summary.json", summary)
    lines = ["# Robust sampling: sequence length 32, pattern length 5", "",
             "Дата: 2026-09-13.", "",
             "32 пары VAE (64 замороженных декодера), 8 held-out паттернов, "
             "64 prior- и 64 oracle-стартов на декодер. Интервалы 95% построены по парам.", "",
             "Sampling использует только жёсткие binary exact-160 маски и target labels; "
             "Gold доступен лишь oracle-построению `z*` и структурной диагностике.", "",
             "| Старт | Стадия | Exact ideal | Gold IoU | Test accuracy | Test BCE | L2 до paired z* |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for group, label in (("prior", "Prior"), ("oracle_best_r4", "Oracle-best R=4")):
        for stage in ("initial", "final"):
            row = summary["groups"][group][stage]
            lines.append(f"| {label} | {stage} | {_fmt(row['exact_fraction'])} | {_fmt(row['gold_iou'])} | "
                         f"{_fmt(row['task_accuracy_2000'])} | {_fmt(row['task_bce_2000'])} | "
                         f"{_fmt(row['paired_oracle_distance'])} |")
    lines += ["", "| Старт | Хотя бы одно принятие | Принято мутаций / start |", "|---|---:|---:|"]
    for group, label in (("prior", "Prior"), ("oracle_best_r4", "Oracle-best R=4")):
        row = summary["groups"][group]["search"]
        lines.append(f"| {label} | {_fmt(row['ever_accepted_fraction'])} | "
                     f"{_fmt(row['mean_acceptances_per_start'])} |")
    oracle = summary["oracle"]
    lines += ["", "## Контроль достижимости", "",
              f"При `R=4` oracle получил exact ideal для **{oracle['exact_count']}/{oracle['total']}** "
              f"стартов ({oracle['fraction']:.2%}). Ветка `oracle-best` включает все лучшие "
              "Gold-oracle приближения и не обозначается как `z*`, если exact не достигнут.", "", "## Вывод", "",
              "Численный вывод дополняется после полного запуска на основании `summary.json`.", ""]
    (out / "RESULTS.md").write_text("\n".join(lines))
    print(f"[k5-robust] report -> {out / 'RESULTS.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--stage", choices=("prepare", "run", "report"), required=True)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.stage == "prepare":
        if args.parent is None:
            parser.error("prepare requires --parent")
        prepare(out, args.parent.resolve())
    elif args.stage == "run":
        if args.model_seed is None:
            parser.error("run requires --model-seed")
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        run_seed(out, args.model_seed, args.device)
    else:
        report(out)


if __name__ == "__main__":
    main()
