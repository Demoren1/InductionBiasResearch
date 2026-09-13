"""Paired deltas and figures for the completed robust hard-sampling run."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    protocol = json.loads((out / "protocol.json").read_text())
    n = protocol["settings"]["starts_per_group"]
    per_seed = []
    pooled = {group: {"exact": 0, "count": 0} for group in protocol["groups"]}
    for seed in protocol["settings"]["model_seeds"]:
        oracle = torch.load(protocol["artifacts"][str(seed)]["oracle"],
                            map_location="cpu", weights_only=True)
        eligible = oracle["best_soft"]["iou"][:n] == 1
        row = {"model_seed": seed, "groups": {}}
        for group, group_slice, keep in (
            ("prior", slice(0, n), torch.ones(n, dtype=torch.bool)),
            ("oracle_z_star", slice(n, 2 * n), eligible),
        ):
            nested = {"accuracy_delta": [], "bce_delta": [],
                      "fraction_accuracy_better": [], "fraction_bce_better": []}
            for pattern in protocol["patterns"]:
                task = torch.load(out / f"seed_{seed}/task_{pattern}.pt",
                                  map_location="cpu", weights_only=True)
                initial_bce = task["final_eval_initial_bce"][group_slice][keep]
                final_bce = task["final_eval_final_bce"][group_slice][keep]
                initial_acc = task["final_eval_initial_accuracy"][group_slice][keep]
                final_acc = task["final_eval_final_accuracy"][group_slice][keep]
                nested["accuracy_delta"].append(float((final_acc - initial_acc).mean()))
                nested["bce_delta"].append(float((final_bce - initial_bce).mean()))
                nested["fraction_accuracy_better"].append(float((final_acc > initial_acc).float().mean()))
                nested["fraction_bce_better"].append(float((final_bce < initial_bce).float().mean()))
                _, exact = _hard_stats(task["final_masks"][group_slice][keep], ideal_mask().float())
                pooled[group]["exact"] += int(exact.sum())
                pooled[group]["count"] += len(exact)
            row["groups"][group] = {key: float(np.mean(values)) for key, values in nested.items()}
        per_seed.append(row)
    aggregate = {group: {
        metric: ci([row["groups"][group][metric] for row in per_seed])
        for metric in per_seed[0]["groups"][group]
    } for group in protocol["groups"]}
    for group in pooled:
        pooled[group]["fraction"] = pooled[group]["exact"] / pooled[group]["count"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summary = json.loads((out / "summary.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4), layout="constrained")
    labels = {"prior": "Prior", "oracle_z_star": "Oracle z*"}
    colors = {"prior": "#347f8b", "oracle_z_star": "#c76e42"}
    for group in labels:
        before, after = summary["groups"][group]["initial"], summary["groups"][group]["final"]
        for axis, metric in zip(axes, ("exact_fraction", "gold_iou", "task_accuracy_2000")):
            axis.plot([0, 1], [before[metric]["mean"], after[metric]["mean"]],
                      marker="o", color=colors[group], label=labels[group])
    axes[0].set(ylabel="Exact ideal fraction", ylim=(-.02, 1.02))
    axes[1].set(ylabel="Gold IoU", ylim=(.55, 1.02))
    axes[2].set(ylabel="Fresh-MLP test accuracy", ylim=(.9, .94))
    for axis in axes:
        axis.set(xticks=[0, 1], xticklabels=["Initial", "Sampling final"])
        axis.grid(alpha=.25)
    axes[0].legend()
    fig.suptitle("Robust hard-mask sampling: task quality improves without recovering Gold")
    fig.savefig(out / "robust_sampling_summary.png", dpi=170)
    fig.savefig(out / "robust_sampling_summary.pdf")
    plt.close(fig)

    seed, pattern = 186, "0100"
    task = torch.load(out / f"seed_{seed}/task_{pattern}.pt", map_location="cpu", weights_only=True)
    oracle = torch.load(protocol["artifacts"][str(seed)]["oracle"],
                        map_location="cpu", weights_only=True)
    eligible = oracle["best_soft"]["iou"][:n] == 1
    prior_changed = (task["final_masks"][:n] != task["initial_masks"][:n]).any(-1).any(-1)
    candidates = torch.where(
        eligible & prior_changed
        & (task["final_eval_final_bce"][:n] < task["final_eval_initial_bce"][:n])
        & (task["final_eval_final_bce"][n:] <= task["final_eval_initial_bce"][n:])
    )[0]
    index = int(candidates[0]) if len(candidates) else int(torch.where(eligible)[0][0])
    gold = ideal_mask().float()

    def aligned(mask: torch.Tensor) -> torch.Tensor:
        return align_columns(gold[None], mask.float()[None])[0]

    panels = [
        ("Prior: initial", task["initial_masks"][index], task["final_eval_initial_bce"][index]),
        ("Prior: sampling final", task["final_masks"][index], task["final_eval_final_bce"][index]),
        ("Ideal", gold, None),
        ("z*: initial", task["initial_masks"][n + index], task["final_eval_initial_bce"][n + index]),
        ("z*: sampling final", task["final_masks"][n + index], task["final_eval_final_bce"][n + index]),
        ("Ideal", gold, None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(8.2, 5.4), layout="constrained")
    for axis, (title, mask, bce) in zip(axes.flat, panels):
        mask = aligned(mask)
        iou, _ = _hard_stats(mask[None], gold)
        suffix = f"IoU={float(iou[0]):.3f}" if bce is None else f"IoU={float(iou[0]):.3f}, BCE={float(bce):.3f}"
        axis.imshow(mask, cmap="Greys", vmin=0, vmax=1)
        axis.set_title(f"{title}\n{suffix}", fontsize=9)
        axis.set_xticks([]); axis.set_yticks([])
    fig.suptitle(f"Seed {seed}, task {pattern}, matched start {index}")
    fig.savefig(out / "robust_sampling_masks.png", dpi=170)
    fig.savefig(out / "robust_sampling_masks.pdf")
    plt.close(fig)

    payload = {"paired_deltas": aggregate, "pooled_exact": pooled,
               "per_seed": per_seed,
               "mask_example": {"seed": seed, "pattern": pattern, "start": index},
               "source_sha256": sha256_file(__file__)}
    write_json(out / "paired_analysis.json", payload)
    print(json.dumps({"paired_deltas": aggregate, "pooled_exact": pooled}, indent=2))


if __name__ == "__main__":
    main()
