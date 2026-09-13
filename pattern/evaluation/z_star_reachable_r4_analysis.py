"""Post-hoc eligible-only statistics and figures for the matched R=4 run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PATTERN_ROOT = HERE.parent
sys.path.insert(0, str(PATTERN_ROOT))

from data.generate import ideal_mask  # noqa: E402
from evaluation.decoder_agreement import align_columns  # noqa: E402
from evaluation.z_star_noise import _hard_stats, ci, sha256_file, write_json  # noqa: E402


STAGES = {"initial": ("initial_z", "initial_masks"),
          "final": ("final_z", "final_masks"),
          "best_query": ("best_val_z", "best_val_masks")}


def seed_rows(out: Path, seed: int, patterns: list[str], starts: int) -> dict:
    oracle_path = out / f"seed_{seed}/oracle_same_start.pt"
    oracle = torch.load(oracle_path, map_location="cpu", weights_only=True)
    endpoints = oracle["best_soft"]["z"].float()
    eligible = oracle["best_soft"]["iou"] == 1
    references = endpoints[eligible]
    target = ideal_mask().float()
    result = {"model_seed": seed, "eligible_count": int(eligible.sum()), "tasks": {},
              "oracle_sha256": sha256_file(oracle_path)}
    for pattern in patterns:
        task_path = out / f"seed_{seed}/task_{pattern}.pt"
        task = torch.load(task_path, map_location="cpu", weights_only=True)
        result["tasks"][pattern] = {}
        for group, offset in (("same_start", 0), ("oracle_endpoint", starts)):
            result["tasks"][pattern][group] = {}
            for stage, (z_key, mask_key) in STAGES.items():
                z = task[z_key][offset:offset + starts].float()[eligible]
                masks = task[mask_key][offset:offset + starts][eligible]
                iou, exact = _hard_stats(masks, target)
                paired = (z - endpoints[eligible]).norm(dim=1)
                nearest = torch.cdist(z, references).min(dim=1).values
                result["tasks"][pattern][group][stage] = {
                    "count": len(z), "exact_fraction": float(exact.float().mean()),
                    "exact_count": int(exact.sum()), "gold_iou": float(iou.mean()),
                    "paired_distance": float(paired.mean()),
                    "nearest_exact_distance": float(nearest.mean()),
                }
    return result


def aggregate(rows: list[dict], patterns: list[str]) -> dict:
    result = {"groups": {}, "per_task": {}}
    metrics = ("exact_fraction", "gold_iou", "paired_distance", "nearest_exact_distance")
    for group in ("same_start", "oracle_endpoint"):
        result["groups"][group] = {}
        for stage in STAGES:
            nested = []
            for row in rows:
                nested.append({metric: float(np.mean([
                    row["tasks"][pattern][group][stage][metric] for pattern in patterns
                ])) for metric in metrics})
            result["groups"][group][stage] = {
                metric: ci([item[metric] for item in nested]) for metric in metrics
            }
    for pattern in patterns:
        result["per_task"][pattern] = {}
        for group in ("same_start", "oracle_endpoint"):
            result["per_task"][pattern][group] = {}
            for stage in STAGES:
                result["per_task"][pattern][group][stage] = {
                    metric: ci([row["tasks"][pattern][group][stage][metric] for row in rows])
                    for metric in metrics
                }
    return result


def figures(out: Path, summary: dict, patterns: list[str], starts: int) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Before", "Final", "Best query"]
    stages = list(STAGES)
    colors = {"same_start": "#377f8d", "oracle_endpoint": "#c86b42"}
    group_labels = {"same_start": "From same prior start",
                    "oracle_endpoint": "Starting at oracle z*"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), layout="constrained")
    x = np.arange(len(stages))
    width = .34
    for index, group in enumerate(("same_start", "oracle_endpoint")):
        means = [summary["groups"][group][stage]["exact_fraction"]["mean"] for stage in stages]
        cis = [summary["groups"][group][stage]["exact_fraction"]["ci95"] for stage in stages]
        errors = np.asarray([[mean - max(0, bounds[0]) for mean, bounds in zip(means, cis)],
                             [min(1, bounds[1]) - mean for mean, bounds in zip(means, cis)]])
        axes[0].bar(x + (index - .5) * width, means, width, yerr=errors,
                    color=colors[group], label=group_labels[group], capsize=3)
        means = [summary["groups"][group][stage]["gold_iou"]["mean"] for stage in stages]
        axes[1].plot(x, means, marker="o", color=colors[group], label=group_labels[group])
    axes[0].set(ylabel="Exact ideal fraction", ylim=(0, 1.05), xticks=x, xticklabels=labels)
    axes[1].set(ylabel="Gold IoU", ylim=(.55, 1.02), xticks=x, xticklabels=labels)
    for axis in axes:
        axis.grid(axis="y", alpha=.25)
        axis.legend(fontsize=8)
    fig.suptitle("Matched R=4: Gold optimum is feasible, task-z does not recover it")
    fig.savefig(out / "matched_reachability.png", dpi=170)
    fig.savefig(out / "matched_reachability.pdf")
    plt.close(fig)

    seed, pattern = 186, "0011"
    oracle = torch.load(out / f"seed_{seed}/oracle_same_start.pt",
                        map_location="cpu", weights_only=True)
    task = torch.load(out / f"seed_{seed}/task_{pattern}.pt",
                      map_location="cpu", weights_only=True)
    eligible = oracle["best_soft"]["iou"] == 1
    _, best_exact = _hard_stats(task["best_val_masks"][starts:], ideal_mask().float())
    candidates = torch.where(eligible & best_exact)[0]
    if not len(candidates):
        candidates = torch.where(eligible)[0]
    index = int(candidates[0])
    target = ideal_mask().float()

    def aligned(mask: torch.Tensor) -> torch.Tensor:
        return align_columns(target[None], mask.float()[None])[0].cpu()

    panels = [
        ("Same prior: before", aligned(task["initial_masks"][index])),
        ("Same prior: final", aligned(task["final_masks"][index])),
        ("Same prior: best", aligned(task["best_val_masks"][index])),
        ("Ideal", target),
        ("Oracle z*: before", aligned(task["initial_masks"][starts + index])),
        ("From z*: final", aligned(task["final_masks"][starts + index])),
        ("From z*: best", aligned(task["best_val_masks"][starts + index])),
        ("Ideal", target),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(9.5, 5), layout="constrained")
    for axis, (title, mask) in zip(axes.flat, panels):
        axis.imshow(mask, cmap="Greys", vmin=0, vmax=1)
        iou, _ = _hard_stats(mask[None], target)
        axis.set_title(f"{title}\nIoU={float(iou[0]):.3f}", fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(f"Mask example: VAE seed {seed}, task {pattern}, matched start {index}")
    fig.savefig(out / "matched_mask_example.png", dpi=170)
    fig.savefig(out / "matched_mask_example.pdf")
    plt.close(fig)
    return {"example_seed": seed, "example_pattern": pattern, "example_start": index}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    protocol = json.loads((out / "protocol.json").read_text())
    seeds = protocol["settings"]["model_seeds"]
    patterns = protocol["heldout_patterns"]
    starts = protocol["settings"]["starts"]
    rows = [seed_rows(out, seed, patterns, starts) for seed in seeds]
    summary = aggregate(rows, patterns)
    example = figures(out, summary, patterns, starts)
    payload = {
        "description": "Eligible-only matched R=4 analysis",
        "eligibility": "Gold-oracle best-soft endpoint has exact ideal hard mask",
        "patterns": patterns, "aggregate": summary, "per_seed": rows,
        "figure_example": example, "source_sha256": sha256_file(__file__),
    }
    write_json(out / "refined_summary.json", payload)
    print(json.dumps({group: {stage: {
        "exact": round(summary["groups"][group][stage]["exact_fraction"]["mean"], 6),
        "iou": round(summary["groups"][group][stage]["gold_iou"]["mean"], 6),
        "paired_d": round(summary["groups"][group][stage]["paired_distance"]["mean"], 6),
    } for stage in STAGES} for group in summary["groups"]}, indent=2))


if __name__ == "__main__":
    main()
