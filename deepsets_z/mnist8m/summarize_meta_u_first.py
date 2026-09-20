"""Summarize the first-layer direct-U MNIST8m translation experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEEDS = (42, 43, 44)
LENGTHS = (1, 3, 5, 10, 20)
COLORS = {"learned": "#1769aa", "random": "#dc6b23",
          "convolution": "#348659", "dense": "#8a55a0"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("deepsets_z/mnist8m/meta_u_first_results"))
    parser.add_argument("--out", type=Path,
                        default=Path("deepsets_z/mnist8m"))
    args = parser.parse_args()
    runs = {}
    for seed in SEEDS:
        for arm in ("learned", "random", "dense"):
            path = args.root / f"{arm}_seed{seed}.json"
            data = json.loads(path.read_text())
            if "test" not in data or data["history"][-1]["step"] != 5000:
                raise ValueError(f"Incomplete result: {path}")
            runs[(arm, seed)] = data
    conv_path = args.root / "convolution_seed42.json"
    conv = json.loads(conv_path.read_text())
    if "test" not in conv or conv["history"][-1]["step"] != 5000:
        raise ValueError(f"Incomplete result: {conv_path}")
    for seed in SEEDS:
        a, b, c = (runs[(arm, seed)]["config"] for arm in
                   ("learned", "random", "dense"))
        for key in ("seed", "inner_steps", "lr_v1", "lr_readout",
                    "outer_lr", "tasks_per_step", "support", "query",
                    "test_tasks", "train_images_per_digit",
                    "eval_images_per_digit"):
            if a[key] != b[key] or a[key] != c[key]:
                raise ValueError(f"Mismatched {key} for seed {seed}")
    summary = {"source": str(args.root), "seeds": list(SEEDS),
               "test_tasks_per_seed": runs[("learned", 42)]["config"][
                   "test_tasks"], "by_length": {}, "overlap": {}}
    for length in LENGTHS:
        row = {}
        for arm in ("learned", "random", "dense"):
            values = np.array([runs[(arm, seed)]["test"][str(length)]["mae"]
                               for seed in SEEDS])
            row[arm] = {"mean_mae": float(values.mean()),
                        "sd_across_seeds": float(values.std(ddof=1)),
                        "by_seed": values.tolist()}
        delta = (np.array(row["learned"]["by_seed"]) -
                 np.array(row["random"]["by_seed"]))
        row["learned_minus_random_mae"] = {
            "mean": float(delta.mean()), "by_seed": delta.tolist()}
        dense_delta = (np.array(row["learned"]["by_seed"]) -
                       np.array(row["dense"]["by_seed"]))
        row["learned_minus_dense_mae"] = {
            "mean": float(dense_delta.mean()), "by_seed": dense_delta.tolist()}
        row["convolution_seed42_mae"] = conv["test"][str(length)]["mae"]
        summary["by_length"][str(length)] = row
    for arm in ("learned", "random"):
        summary["overlap"][arm] = [runs[(arm, seed)]["convolution_overlap"]
                                   for seed in SEEDS]
    summary["overlap"]["convolution_seed42"] = conv["convolution_overlap"]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "meta_u_first_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharey=True)
    for ax, seed in zip(axes, SEEDS):
        for arm in ("learned", "random", "dense"):
            history = runs[(arm, seed)]["history"]
            ax.plot([r["step"] for r in history],
                    [r["validation_score"] for r in history],
                    label=arm, color=COLORS[arm], linewidth=1.7)
        if seed == 42:
            history = conv["history"]
            ax.plot([r["step"] for r in history],
                    [r["validation_score"] for r in history],
                    label="analytic conv", color=COLORS["convolution"],
                    linewidth=1.5)
        ax.set_title(f"seed {seed}")
        ax.set_xlabel("Meta-training step")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Validation MSE / set length ↓")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out / "meta_u_first_convergence.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    for arm in ("learned", "random", "dense"):
        by_seed = np.array([[runs[(arm, seed)]["test"][str(length)]["mae"]
                             for length in LENGTHS] for seed in SEEDS])
        for values in by_seed:
            ax.plot(LENGTHS, values, color=COLORS[arm], alpha=0.2,
                    linewidth=0.8)
        ax.plot(LENGTHS, by_seed.mean(axis=0), color=COLORS[arm],
                marker="o", label=arm, linewidth=2)
    ax.plot(LENGTHS, [conv["test"][str(length)]["mae"] for length in LENGTHS],
            color=COLORS["convolution"], marker="s", linewidth=1.7,
            label="analytic conv, seed 42")
    ax.set_xlabel("Set length")
    ax.set_ylabel("Shifted-query test MAE ↓")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out / "meta_u_first_lengths.png", dpi=180)
    plt.close(fig)
    print(json.dumps({length: summary["by_length"][length]
                      for length in ("5", "10", "20")}, indent=2))


if __name__ == "__main__":
    main()
