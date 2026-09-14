"""Plot the aggregate comparison from length-32 joint-search summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = ("joint", "frozen", "dense", "random", "analytic")
LABELS = ("Joint w/z", "Frozen w", "Dense-pretrained w", "Random best-of-64", "Analytic")
COLORS = ("#2F6BFF", "#7D8CA3", "#A66DD4", "#E69F00", "#169B62")
METRICS = (
    ("mean_query_bce", "Query BCE ↓", (0.30, 0.76)),
    ("mean_query_accuracy", "Accuracy ↑", (0.55, 0.88)),
    ("mean_iou", "IoU with analytic ↑", (0.20, 1.05)),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summaries = [json.loads(path.read_text()) for path in sorted(args.input_root.glob("seed*/summary.json"))]
    if not summaries:
        raise ValueError(f"no seed*/summary.json files under {args.input_root}")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    rng = np.random.default_rng(20260915)
    x = np.arange(len(METHODS))
    for ax, (field, title, ylim) in zip(axes, METRICS):
        values = np.asarray([
            [summary["evaluations"]["test"]["strategies"][method][field] for summary in summaries]
            for method in METHODS
        ])
        means = values.mean(axis=1)
        ci95 = 2.131 * values.std(axis=1, ddof=1) / np.sqrt(values.shape[1])
        ax.bar(x, means, color=COLORS, alpha=0.86, width=0.72)
        ax.errorbar(x, means, yerr=ci95, fmt="none", ecolor="#20242A", capsize=4, linewidth=1.4)
        for index, row in enumerate(values):
            jitter = rng.uniform(-0.17, 0.17, size=len(row))
            ax.scatter(index + jitter, row, s=15, color="#20242A", alpha=0.52, linewidths=0, zorder=3)
            ax.text(index, means[index] + ci95[index] + 0.012 * (ylim[1] - ylim[0]),
                    f"{means[index]:.3f}", ha="center", va="bottom", fontsize=8.5)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x, LABELS, rotation=24, ha="right")
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)

    fig.suptitle("Length 32, pattern length 4 — test comparison over 16 seeds", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.01, "Bars: mean; error bars: 95% CI; dots: individual seeds", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
