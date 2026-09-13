"""Two-seed matched R=4 pilot with binary-forward task-z optimization."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PATTERN_ROOT = HERE.parent
REPO_ROOT = PATTERN_ROOT.parent
sys.path.insert(0, str(PATTERN_ROOT))

from data.generate import ideal_mask  # noqa: E402
from evaluation.single_z_task_hard_ste import optimize_task_z_hard_ste  # noqa: E402
from evaluation.z_star_noise import (  # noqa: E402
    _hard_stats,
    atomic_save,
    ci,
    load_model,
    sha256_file,
    write_json,
)


SEEDS = (186, 188)
RADIUS = 4.0
STARTS = 64
REFERENCE = PATTERN_ROOT / "outputs/z_star_reachable/r4_multiseed8_20260912"
DEFAULT_OUT = PATTERN_ROOT / "outputs/z_star_reachable/r4_hard_ste_seeds186_188_20260912"


def make_protocol(reference: Path) -> dict:
    parent_protocol = json.loads((reference / "protocol.json").read_text())
    sources = [Path(__file__), HERE / "single_z_task_hard_ste.py",
               HERE / "single_z_task.py", HERE / "z_star_noise.py"]
    artifacts = {}
    for seed in SEEDS:
        oracle = reference / f"seed_{seed}/oracle_same_start.pt"
        initial_source = reference / f"seed_{seed}/task_0100.pt"
        artifacts[str(seed)] = {
            "oracle": str(oracle), "oracle_sha256": sha256_file(oracle),
            "initial_source": str(initial_source),
            "initial_source_sha256": sha256_file(initial_source),
            "checkpoint": parent_protocol["checkpoints"][str(seed)]["checkpoint"],
            "checkpoint_sha256": parent_protocol["checkpoints"][str(seed)]["checkpoint_sha256"],
        }
    return {
        "experiment": "two-seed matched R=4 hard-forward soft-backward STE pilot",
        "model_seeds": list(SEEDS), "starts": STARTS, "radius": RADIUS,
        "patterns": parent_protocol["heldout_patterns"],
        "reference": str(reference), "reference_protocol_sha256": sha256_file(reference / "protocol.json"),
        "parent": parent_protocol["parent"], "artifacts": artifacts,
        "task_z": {"outer_steps": 30, "warmup_steps": 300, "grad_steps": 100,
                   "z_lr": 0.05, "temperature": 0.5, "seed": 20260912,
                   "forward_mask": "binary hard top-32",
                   "backward_mask_jacobian": "soft top-32 straight-through estimator"},
        "comparison": "same starts, oracle endpoints, tasks, seeds, and budget as soft matched R=4",
        "source_sha256": {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in sources},
    }


def prepare(out: Path, reference: Path) -> None:
    payload = make_protocol(reference)
    path = out / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise FileExistsError(f"{path} differs from immutable protocol")
        return
    out.mkdir(parents=True)
    (out / "logs").mkdir()
    write_json(path, payload)


def describe(values: torch.Tensor) -> dict:
    values = values.float().cpu()
    return {"mean": float(values.mean()), "min": float(values.min()),
            "max": float(values.max()), "values": values.tolist()}


def stage_metrics(z: torch.Tensor, masks: torch.Tensor, endpoints: torch.Tensor,
                  eligible: torch.Tensor, offset: int) -> dict:
    z = z[offset:offset + STARTS][eligible].float()
    masks = masks[offset:offset + STARTS][eligible]
    iou, exact = _hard_stats(masks, ideal_mask().float())
    refs = endpoints[eligible].float()
    return {
        "count": len(z), "exact_count": int(exact.sum()),
        "exact_fraction": float(exact.float().mean()), "gold_iou": describe(iou),
        "paired_distance": describe((z - refs).norm(dim=1)),
        "nearest_exact_distance": describe(torch.cdist(z, refs).min(dim=1).values),
    }


def run_seed(out: Path, seed: int, device_name: str) -> None:
    protocol = json.loads((out / "protocol.json").read_text())
    if seed not in SEEDS:
        raise ValueError(seed)
    root = out / f"seed_{seed}"
    summary_path = root / "summary.json"
    if summary_path.exists():
        print(f"[hard-ste-r4] seed={seed}: complete", flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    model, provenance = load_model(Path(protocol["parent"]), seed, device)
    sources = protocol["artifacts"][str(seed)]
    oracle = torch.load(sources["oracle"], map_location="cpu", weights_only=True)
    reference_task = torch.load(sources["initial_source"], map_location="cpu", weights_only=True)
    combined = reference_task["initial_z"].float()
    if combined.shape != (2 * STARTS, 32):
        raise ValueError("reference starts have unexpected shape")
    endpoints = oracle["best_soft"]["z"].float()
    eligible = oracle["best_soft"]["iou"] == 1
    if not torch.allclose(combined[STARTS:], endpoints, atol=2e-6, rtol=0):
        raise ValueError("oracle endpoints differ from matched reference starts")

    tasks = {}
    for pattern in protocol["patterns"]:
        destination = root / f"task_{pattern}.pt"
        if destination.exists():
            result = torch.load(destination, map_location="cpu", weights_only=True)
        else:
            result = optimize_task_z_hard_ste(
                model, combined, pattern, device,
                outer_steps=protocol["task_z"]["outer_steps"],
                warmup_steps=protocol["task_z"]["warmup_steps"],
                grad_steps=protocol["task_z"]["grad_steps"],
                z_lr=protocol["task_z"]["z_lr"], radius=RADIUS,
                temperature=protocol["task_z"]["temperature"],
                seed=protocol["task_z"]["seed"],
            )
            result.update({"model_seed": seed, "pattern": pattern,
                           "checkpoint_sha256": provenance["checkpoint_sha256"],
                           "oracle_sha256": sources["oracle_sha256"],
                           "uses_gold": False})
            atomic_save(destination, result)
        tasks[pattern] = result

    stages = {"initial": ("initial_z", "initial_masks"),
              "final": ("final_z", "final_masks"),
              "best_query": ("best_val_z", "best_val_masks")}
    summary = {"model_seed": seed, "eligible_count": int(eligible.sum()),
               "tasks": {}, "provenance": provenance}
    for pattern, task in tasks.items():
        summary["tasks"][pattern] = {}
        for group, offset in (("same_start", 0), ("oracle_endpoint", STARTS)):
            summary["tasks"][pattern][group] = {
                stage: stage_metrics(task[z_key], task[mask_key], endpoints, eligible, offset)
                for stage, (z_key, mask_key) in stages.items()
            }
    write_json(summary_path, summary)
    print(f"[hard-ste-r4] seed={seed}: complete", flush=True)


def aggregate_method(records: list[dict], patterns: list[str]) -> dict:
    metrics = ("exact_fraction", "gold_iou", "paired_distance", "nearest_exact_distance")
    output = {}
    for group in ("same_start", "oracle_endpoint"):
        output[group] = {}
        for stage in ("initial", "final", "best_query"):
            per_seed = []
            for record in records:
                rows = [record["tasks"][pattern][group][stage] for pattern in patterns]
                per_seed.append({
                    metric: float(np.mean([
                        row[metric] if isinstance(row[metric], (int, float))
                        else row[metric]["mean"]
                        for row in rows]))
                    for metric in metrics
                })
            output[group][stage] = {
                metric: ci([row[metric] for row in per_seed]) for metric in metrics
            }
    return output


def soft_reference(protocol: dict) -> dict:
    reference = Path(protocol["reference"])
    patterns = protocol["patterns"]
    refined = json.loads((reference / "refined_summary.json").read_text())
    records = [row for row in refined["per_seed"] if row["model_seed"] in SEEDS]
    return aggregate_method(records, patterns)


def fmt(value: dict) -> str:
    return f"{value['mean']:.3f} [{value['ci95'][0]:.3f}; {value['ci95'][1]:.3f}]"


def report(out: Path) -> None:
    protocol = json.loads((out / "protocol.json").read_text())
    records = [json.loads((out / f"seed_{seed}/summary.json").read_text()) for seed in SEEDS]
    hard = aggregate_method(records, protocol["patterns"])
    soft = soft_reference(protocol)
    payload = {"protocol": protocol, "hard_ste": hard, "soft_reference_same_two_seeds": soft,
               "eligible_counts": {str(row["model_seed"]): row["eligible_count"] for row in records}}
    write_json(out / "summary.json", payload)
    lines = [
        "# Pattern R=4: hard-mask STE pilot", "",
        "Два frozen VAE seeds (186, 188), 64 matched старта, четыре held-out задачи. "
        "MLP получает бинарную top-32 маску в forward; backward к z использует soft-top-k STE.", "",
        "| Режим | Старт | Стадия | Exact ideal | Gold IoU | L2 до paired z* |",
        "|---|---|---|---:|---:|---:|",
    ]
    labels = {"same_start": "тот же prior", "oracle_endpoint": "exact z*"}
    for method, values in (("Soft reference", soft), ("Hard STE", hard)):
        for group in ("same_start", "oracle_endpoint"):
            for stage in ("initial", "final", "best_query"):
                row = values[group][stage]
                lines.append(f"| {method} | {labels[group]} | {stage} | "
                             f"{fmt(row['exact_fraction'])} | {fmt(row['gold_iou'])} | "
                             f"{fmt(row['paired_distance'])} |")
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(f"[hard-ste-r4] report -> {out / 'RESULTS.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "report"), required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    out, reference = args.out.resolve(), args.reference.resolve()
    if args.stage == "prepare":
        prepare(out, reference)
        print(f"[hard-ste-r4] protocol -> {out / 'protocol.json'}", flush=True)
        return
    if args.stage == "run":
        if args.model_seed is None:
            parser.error("--model-seed is required")
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
