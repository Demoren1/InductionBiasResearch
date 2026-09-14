"""Aggregate plots for generated parameter-sharing experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pattern.evaluation.decoder_agreement import align_columns
from .generated_sharing import SharingConfig, analytic_assignment


METHODS = ("generated_sharing", "generated_connectivity", "dense", "random_sharing", "analytic_sharing")
LABELS = ("Generated\nsharing", "Same connectivity,\nindependent weights", "Dense", "Random\nsharing", "Analytic\nsharing")
COLORS = ("#2864DC", "#7A65C7", "#8895A7", "#E7A51A", "#18996A")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--comparison-output", type=Path, required=True)
    parser.add_argument("--masks-output", type=Path, required=True)
    args = parser.parse_args()
    summaries = [json.loads(path.read_text()) for path in sorted(args.input_root.glob("seed*/summary.json"))]
    if not summaries:
        raise ValueError("no summaries found")

    fields = (("mean_query_bce", "Test BCE ↓", (0.1, 0.66)),
              ("mean_query_accuracy", "Accuracy ↑", (0.60, 1.01)),
              ("mean_active_iou", "Active IoU ↑", (0.0, 1.05)))
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    rng = np.random.default_rng(20260915)
    x = np.arange(len(METHODS))
    t95 = 2.365 if len(summaries) == 8 else 1.96
    for axis, (field, title, ylim) in zip(axes, fields):
        values = np.asarray([[summary["evaluations"]["test"]["strategies"][method][field]
                              for summary in summaries] for method in METHODS])
        means = values.mean(1)
        errors = t95 * values.std(1, ddof=1) / np.sqrt(values.shape[1])
        axis.bar(x, means, color=COLORS, width=0.72, alpha=0.88)
        axis.errorbar(x, means, yerr=errors, fmt="none", ecolor="#20242A", capsize=4)
        for index, row in enumerate(values):
            axis.scatter(index + rng.uniform(-0.15, 0.15, len(row)), row,
                         s=18, color="#20242A", alpha=0.58, linewidths=0, zorder=3)
            axis.text(index, means[index] + errors[index] + 0.015 * (ylim[1] - ylim[0]),
                      f"{means[index]:.3f}", ha="center", fontsize=8.5)
        axis.set_title(title, fontweight="bold")
        axis.set_xticks(x, LABELS, rotation=20, ha="right")
        axis.set_ylim(*ylim)
        axis.grid(axis="y", alpha=0.22)
        axis.set_axisbelow(True)
    figure.suptitle("Generated parameter sharing — length 32, pattern length 4, 8 seeds",
                    fontsize=14, fontweight="bold")
    figure.text(0.5, 0.01, "Bars: mean; error bars: 95% CI; dots: seeds", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    args.comparison_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.comparison_output, dpi=180, bbox_inches="tight")
    plt.close(figure)

    config = SharingConfig(**summaries[0]["config"])
    gold = analytic_assignment(config, torch.device("cpu")).sum(-1)
    figure, axes = plt.subplots(3, 3, figsize=(10, 10))
    for axis, summary in zip(axes.flat, summaries):
        mask = torch.tensor(summary["evaluations"]["test"]["strategies"]["generated_sharing"]
                            ["chosen_active_masks"][0])
        axis.imshow(align_columns(gold, mask), cmap="Greys", vmin=0, vmax=1, aspect="auto")
        axis.set_title(f"Seed {summary['config']['seed']}")
        axis.set(xticks=[], yticks=[])
    axes.flat[-1].imshow(gold, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    axes.flat[-1].set_title("Analytic")
    axes.flat[-1].set(xticks=[], yticks=[])
    figure.suptitle("Learned active structures after hidden-column alignment", fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    args.masks_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.masks_output, dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
