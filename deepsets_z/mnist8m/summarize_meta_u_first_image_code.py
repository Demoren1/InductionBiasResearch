"""One-seed summary of image-conditioned first-layer U on MNIST8m."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ARMS = ("learned", "random", "generated_ortho",
        "generated_shuffled_ortho", "generated", "generated_shuffled")
LENGTHS = (1, 3, 5, 10, 20)
LABELS = {
    "learned": "Direct U",
    "random": "Random U",
    "generated_ortho": "Spatial generator, orthogonal U",
    "generated_shuffled_ortho": "Shuffled generator, orthogonal U",
    "generated": "Spatial generator",
    "generated_shuffled": "Shuffled generator",
    "dense": "Dense first layer",
}
COLORS = {
    "learned": "#145ca8", "random": "#6d7887",
    "generated_ortho": "#078367",
    "generated_shuffled_ortho": "#bc792d",
    "generated": "#66b79c", "generated_shuffled": "#deb479",
    "dense": "#8349a5",
}
CONFIG_KEYS = ("seed", "steps", "inner_steps", "lr_v1", "lr_readout",
               "outer_lr", "tasks_per_step", "support", "query", "test_tasks",
               "train_images_per_digit", "eval_images_per_digit")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root", type=Path, default=Path(
        "deepsets_z/mnist8m/meta_u_first_image_code_results"))
    parser.add_argument("--static-root", type=Path, default=Path(
        "deepsets_z/mnist8m/meta_u_first_results"))
    parser.add_argument("--out", type=Path, default=Path("deepsets_z/mnist8m"))
    args = parser.parse_args()
    runs = {arm: json.loads((args.root / f"{arm}_seed{args.seed}.json").read_text())
            for arm in ARMS}
    static = {arm: json.loads((args.static_root /
                               f"{arm}_seed{args.seed}.json").read_text())
              for arm in (*ARMS, "dense")}
    reference = runs[ARMS[0]]["config"]
    for arm, run in runs.items():
        if "test" not in run or "test_no_image_code" not in run:
            raise ValueError(f"Incomplete test: {arm}")
        if run["history"][-1]["step"] != run["config"]["steps"]:
            raise ValueError(f"Incomplete training: {arm}")
        for key in CONFIG_KEYS:
            if run["config"][key] != reference[key]:
                raise ValueError(f"Mismatched {key}: {arm}")
            if run["config"][key] != static[arm]["config"][key]:
                raise ValueError(f"Mismatched static {key}: {arm}")
    if static["dense"]["config"]["seed"] != args.seed:
        raise ValueError("Mismatched dense seed")

    summary = {
        "seed": args.seed,
        "test_tasks": reference["test_tasks"],
        "metric": "mean absolute error on held-out shifted-query tasks; lower is better",
        "note": "no_code disables the image gate after training; static is a separate training run without the image encoder",
        "lengths": list(LENGTHS), "by_arm": {},
    }
    for arm in ARMS:
        run, old = runs[arm], static[arm]
        summary["by_arm"][arm] = {
            "parameters": run["model"]["trainable_parameters"],
            "best_step": run["best_step"],
            "with_code": {str(n): run["test"][str(n)]["mae"] for n in LENGTHS},
            "code_disabled_after_training": {
                str(n): run["test_no_image_code"][str(n)]["mae"] for n in LENGTHS},
            "separately_trained_without_code": {
                str(n): old["test"][str(n)]["mae"] for n in LENGTHS},
        }
    summary["dense_static"] = {
        str(n): static["dense"]["test"][str(n)]["mae"] for n in LENGTHS}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"meta_u_first_image_code_seed{args.seed}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12})
    fig = plt.figure(figsize=(13.3, 9.0), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(1.1, 0.9))
    ax_lengths = fig.add_subplot(grid[0, 0])
    ax_ablation = fig.add_subplot(grid[0, 1])
    ax_training = fig.add_subplot(grid[1, :])
    for arm in ARMS:
        values = [runs[arm]["test"][str(n)]["mae"] for n in LENGTHS]
        ax_lengths.plot(LENGTHS, values, color=COLORS[arm], marker="o",
                        linewidth=2, markersize=4.5, label=LABELS[arm])
    for arm in ("learned", "dense"):
        values = [static[arm]["test"][str(n)]["mae"] for n in LENGTHS]
        ax_lengths.plot(LENGTHS, values, linestyle="--", color=COLORS[arm],
                        linewidth=1.6, label=f"{LABELS[arm]}, prior run")
    ax_lengths.set_title("A. Held-out tasks: test error by set length")
    ax_lengths.set_xlabel("Number of images in the query set")
    ax_lengths.set_ylabel("Mean absolute error in the sum ↓")
    ax_lengths.set_xticks(LENGTHS)
    ax_lengths.grid(alpha=0.25)
    ax_lengths.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")

    y = np.arange(len(ARMS))
    for index, arm in enumerate(ARMS):
        code = runs[arm]["test"]["5"]["mae"]
        disabled = runs[arm]["test_no_image_code"]["5"]["mae"]
        ax_ablation.plot([code, disabled], [index, index], color=COLORS[arm],
                         linewidth=2, zorder=1)
        ax_ablation.scatter(code, index, marker="o", s=55,
                            color=COLORS[arm], edgecolor="white", zorder=3)
        ax_ablation.scatter(disabled, index, marker="x", s=70,
                            color=COLORS[arm], linewidth=2, zorder=3)
        if disabled >= 4:
            ax_ablation.annotate(f"{disabled:.1f}", (disabled, index),
                                 xytext=(-5, 6), textcoords="offset points",
                                 ha="right", fontsize=8)
    ax_ablation.set_yticks(y, [LABELS[arm] for arm in ARMS])
    ax_ablation.invert_yaxis()
    ax_ablation.set_xscale("log")
    ax_ablation.set_xlim(1.0, 25)
    ax_ablation.set_xticks([1, 1.5, 2, 3, 5, 10, 20])
    ax_ablation.set_xticklabels(["1", "1.5", "2", "3", "5", "10", "20"])
    ax_ablation.set_title("B. Length 5: disable image code after training")
    ax_ablation.set_xlabel("Test MAE ↓    ● code on    × code off")
    ax_ablation.grid(axis="x", alpha=0.25)

    for arm in ARMS:
        history = runs[arm]["history"]
        ax_training.plot([point["step"] for point in history],
                         [point["validation_score"] for point in history],
                         color=COLORS[arm], linewidth=1.7, label=LABELS[arm])
    ax_training.set_title("C. Validation during meta-training")
    ax_training.set_xlabel("Outer optimization step")
    ax_training.set_ylabel("Normalized validation MSE ↓")
    ax_training.set_yscale("log")
    ax_training.grid(alpha=0.25)
    ax_training.legend(frameon=False, ncol=3, fontsize=8)
    fig.suptitle(f"Image-conditioned U in the first layer · MNIST8m · seed {args.seed}",
                 fontsize=14)
    figure_path = args.out / f"meta_u_first_image_code_seed{args.seed}_summary.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    print(figure_path)
    for arm in ARMS:
        row = summary["by_arm"][arm]
        print(f"{arm:27s} code={row['with_code']['5']:.4f} "
              f"disabled={row['code_disabled_after_training']['5']:.4f} "
              f"static={row['separately_trained_without_code']['5']:.4f}")


if __name__ == "__main__":
    main()
