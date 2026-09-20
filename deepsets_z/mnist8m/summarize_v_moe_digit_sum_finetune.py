"""Summarize frozen-U digit-sum fine-tuning against the dense MLP control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .run import TEST_LENGTHS


VARIANTS = (
    ("conv_only", "centered", "conv_router_v_seed42", "Conv · router + v", "#9c8c87", ":"),
    ("conv_coeff", "centered", "conv_router_v_coeff_seed42", "Conv · + coefficients", "#8585a3", ":"),
    ("conv_readout", "centered", "conv_router_v_readout_seed42", "Conv · + readout", "#2077a4", "--"),
    ("conv_head", "centered", "conv_router_v_head_seed42", "Conv · + head", "#147f6f", "-"),
    ("mlp_head", "centered", "mlp_router_v_head_seed42", "MLP · + head", "#c46b39", "-"),
    ("conv_readout_raw", "raw", "conv_router_v_readout_seed42", "Conv · + readout", "#2077a4", "--"),
    ("conv_head_raw", "raw", "conv_router_v_head_seed42", "Conv · + head", "#147f6f", "-"),
    ("mlp_head_raw", "raw", "mlp_router_v_head_seed42", "MLP · + head", "#c46b39", "-"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--centered-dir", type=Path, default=Path(
        "deepsets_z/mnist8m/outputs/v_moe_digit_sum_finetune"))
    parser.add_argument("--raw-dir", type=Path, default=Path(
        "deepsets_z/mnist8m/outputs/v_moe_digit_sum_finetune_raw"))
    parser.add_argument("--dense-json", type=Path, default=Path(
        "deepsets_z/mnist8m/results_authors_scaled/paper_mlp_seed42.json"))
    parser.add_argument("--out", type=Path, default=Path("deepsets_z/mnist8m"))
    args = parser.parse_args()
    dense = json.loads(args.dense_json.read_text())
    summary = {"protocol": {
        "seed": 42, "train_sets": 148148, "test_sets_per_length": 10000,
        "train_lengths": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "zero_digits_excluded": True, "test_image_shift": 0,
        "frozen_U": True, "frozen_middle_layers": True,
        "checkpoint_selection": "validation MAE on sets of lengths 1–9",
        "centered_target": "(digit_sum - 4.5 * set_length) / sqrt(8.25)",
        "raw_target": "digit_sum"},
        "dense": {"parameter_count": dense["parameter_count"],
                  "epochs": len(dense["history"]),
                  "test": {str(length): dense["metrics"][f"test_{length}"]
                           for length in TEST_LENGTHS}},
        "variants": {}}
    for key, target, stem, label, color, linestyle in VARIANTS:
        directory = args.centered_dir if target == "centered" else args.raw_dir
        run = json.loads((directory / f"{stem}.json").read_text())
        if "test" not in run or run["config"]["seed"] != 42:
            raise ValueError(f"Incomplete run: {directory / stem}")
        if run["config"]["trainable"] != stem.split("_seed")[0].split("_", 1)[1]:
            raise ValueError(f"Mismatched trainable parameters: {directory / stem}")
        expected_center = 0.0 if target == "raw" else 4.5
        expected_scale = 1.0 if target == "raw" else 8.25 ** 0.5
        if (run["config"].get("target_center", 4.5) != expected_center
                or abs(run["config"].get("target_scale", 8.25 ** 0.5)
                       - expected_scale) > 1e-12):
            raise ValueError(f"Mismatched target transform: {directory / stem}")
        summary["variants"][key] = {
            "label": label, "target": target,
            "router": run["config"]["router"],
            "trainable": run["config"]["trainable"],
            "source_checkpoint": run["config"]["checkpoint"],
            "target_center": expected_center,
            "target_scale": expected_scale,
            "trained_parameters": run["trained_parameters"],
            "best_epoch": run["best_epoch"],
            "stop_epoch": run["stopping"]["epoch"],
            "stop_reason": run["stopping"]["reason"],
            "best_validation_mae": run["best_validation_mae"],
            "training_seconds": run["history"][-1]["elapsed_seconds"],
            "test": run["test"],
            "history": [{"epoch": row["epoch"],
                         "validation_mae": row["validation"]["mae"],
                         "probe50_accuracy": row["probe"]["50"]["exact_round_accuracy"]}
                        for row in run["history"]]}
    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "v_moe_digit_sum_finetune_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8),
                             sharey=True, layout="constrained")
    for ax, target in zip(axes, ("raw", "centered")):
        ax.plot(TEST_LENGTHS,
                [100 * summary["dense"]["test"][str(n)]["exact_round_accuracy"]
                 for n in TEST_LENGTHS], color="#242424", marker="o",
                linewidth=2, label="Dense MLP · direct training")
        for key, group, _, label, color, linestyle in VARIANTS:
            if group != target:
                continue
            run = summary["variants"][key]
            ax.plot(TEST_LENGTHS,
                    [100 * run["test"][str(n)]["exact_round_accuracy"]
                     for n in TEST_LENGTHS],
                    color=color, linestyle=linestyle, marker="o", markersize=3.5,
                    linewidth=1.8, label=label)
        ax.set_xticks(TEST_LENGTHS)
        ax.set_xlabel("Set length")
        ax.set_title("Raw sum target" if target == "raw" else "Centered sum target")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Exact rounded sum, % ↑")
    fig.suptitle("MNIST8m · frozen generated U · seed 42 · 10,000 test sets per length")
    figure = args.out / "v_moe_digit_sum_finetune_accuracy.png"
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.7),
                             layout="constrained")
    for key, target, _, label, color, linestyle in VARIANTS:
        if key in ("conv_only", "conv_coeff"):
            continue
        run = summary["variants"][key]
        history = run["history"]
        name = f"{label} ({target})"
        axes[0].plot([row["epoch"] for row in history],
                     [100 * row["probe50_accuracy"] for row in history],
                     color=color, linestyle=linestyle if target == "centered" else ":",
                     linewidth=1.8, label=name)
        axes[1].plot([row["epoch"] for row in history],
                     [row["validation_mae"] for row in history],
                     color=color, linestyle=linestyle if target == "centered" else ":",
                     linewidth=1.8, label=name)
    axes[0].set_ylabel("Exact sum on 1,000 length-50 probe sets, % ↑")
    axes[1].set_ylabel("Validation MAE in digit-sum units ↓")
    for ax in axes:
        ax.set_xlabel("Fine-tuning epoch")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Fine-tuning progress · checkpoints selected by validation MAE")
    convergence = args.out / "v_moe_digit_sum_finetune_convergence.png"
    fig.savefig(convergence, dpi=180)
    plt.close(fig)
    print(f"Summary: {summary_path}\nFigures: {figure}, {convergence}")


if __name__ == "__main__":
    main()
