"""Aggregate the controlled Deep Sets z ablation and draw diagnostic plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .run import ARMS, TEST_LENGTHS


LABELS = {
    "learned_z": "Learned z, 16 codes",
    "fixed_z": "Fixed z, 16 codes",
    "no_z_one": "No z, one generator",
    "no_z_16": "No z, 16 generators",
    "oracle": "Exact shared U",
}
COLORS = {
    "learned_z": "#2675c2", "fixed_z": "#e68225",
    "no_z_one": "#8e69ad", "no_z_16": "#238b64", "oracle": "#222222",
}


def selected(run: dict, policy: str) -> dict:
    return run if run["arm"] == "oracle" else run["selections"][policy]


def aggregate(runs: dict[str, list[dict]], policy: str) -> dict:
    answer = {}
    for arm, values in runs.items():
        rows = [selected(row, policy) for row in values]
        answer[arm] = {
            "seeds": [row["seed"] for row in values],
            "training_seconds_mean": float(np.mean([row["training_seconds"] for row in values])),
            "training_seconds_std": float(np.std([row["training_seconds"] for row in values], ddof=1)),
            "selected_step_median": float(np.median([row["selected_step"] for row in rows])),
            "selected_step_1_count": sum(row["selected_step"] == 1 for row in rows),
            "constant_assignment_count": sum(
                all(slot == row["slot_assignments"][0]
                    for slot in row["slot_assignments"])
                for row in rows
            ),
            "permutation_error_10_mean": float(np.mean([row["permutation_error_10"] for row in rows])),
            "lengths": {},
        }
        for length in TEST_LENGTHS:
            metrics = [row["metrics"][f"test_{length}"] for row in rows]
            answer[arm]["lengths"][str(length)] = {
                key: {"mean": float(np.mean([metric[key] for metric in metrics])),
                      "std": float(np.std([metric[key] for metric in metrics], ddof=1)),
                      "median": float(np.median([metric[key] for metric in metrics])),
                      "q25": float(np.quantile([metric[key] for metric in metrics], 0.25)),
                      "q75": float(np.quantile([metric[key] for metric in metrics], 0.75)),
                      "per_seed": [float(metric[key]) for metric in metrics]}
                for key in ("mae", "rmse", "exact_round_accuracy")
            }
    return answer


def plot_size(summary: dict, path: Path, metric: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7.3, 4.8))
    for arm in ARMS:
        values = summary[arm]["lengths"]
        median = np.array([values[str(length)][metric]["median"] for length in TEST_LENGTHS])
        q25 = np.array([values[str(length)][metric]["q25"] for length in TEST_LENGTHS])
        q75 = np.array([values[str(length)][metric]["q75"] for length in TEST_LENGTHS])
        ax.plot(TEST_LENGTHS, median, marker="o", linewidth=2.2,
                label=LABELS[arm], color=COLORS[arm])
        if arm != "oracle":
            ax.fill_between(TEST_LENGTHS, q25, q75,
                            color=COLORS[arm], alpha=0.10)
    ax.set_xlabel("Number of digits in the set")
    ax.set_ylabel(ylabel)
    ax.set_xticks(TEST_LENGTHS)
    ax.grid(alpha=0.25)
    ax.set_title("Median across 8 seeds; band = 25th–75th percentile")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_paired(runs: dict[str, list[dict]], path: Path, policy: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    for ax, length in zip(axes, (10, 100), strict=True):
        learned = [selected(row, policy)["metrics"][f"test_{length}"]["mae"]
                   for row in runs["learned_z"]]
        for arm in ("fixed_z", "no_z_one", "no_z_16"):
            control = [selected(row, policy)["metrics"][f"test_{length}"]["mae"]
                       for row in runs[arm]]
            ax.scatter(learned, control, label=LABELS[arm], color=COLORS[arm], alpha=0.8)
        upper = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, upper], [0, upper], "--", color="0.45", linewidth=1)
        ax.set_xlabel("Learned z: MAE")
        ax.set_ylabel("Control: MAE")
        ax.set_title(f"Set length {length}")
        ax.grid(alpha=0.2)
    axes[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_assignments(runs: dict[str, list[dict]], path: Path, policy: str,
                     placement: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 5.4), sharex=True, sharey=True)
    for ax, arm in zip(axes.flat, ARMS[:4], strict=True):
        row = selected(runs[arm][0], policy)
        codes = np.asarray(row["slot_assignments"])
        ax.imshow(codes.T, aspect="auto", interpolation="nearest", cmap="tab10", vmin=0, vmax=9)
        if placement == "prefix":
            ax.axvline(9.5, color="white", linewidth=1.5, linestyle="--")
            ax.axvline(19.5, color="white", linewidth=1.0, linestyle=":")
        ax.set_title(LABELS[arm] + ", seed " + str(runs[arm][0]["seed"]))
        ax.set_yticks(range(4), labels=["bit 1", "bit 2", "bit 4", "bit 8"])
        ax.set_xlim(-0.5, 99.5)
    for ax in axes[-1]:
        ax.set_xlabel("Element position")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("deepsets_z/results"))
    parser.add_argument("--output", type=Path, default=Path("deepsets_z"))
    parser.add_argument("--plots", choices=("none", "all"), default="none")
    args = parser.parse_args()
    cfg = json.loads((args.results / "config.json").read_text())
    runs = {arm: [json.loads((args.results / f"{arm}_seed{seed}.json").read_text())
                  for seed in cfg["seeds"]] for arm in ARMS}
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = {policy: aggregate(runs, policy)
                 for policy in ("in_distribution", "length_20")}
    (args.output / "summary.json").write_text(json.dumps({
        "config": cfg, "selection_policies": summaries,
    }, indent=2) + "\n")
    if args.plots == "all":
        for policy, summary in summaries.items():
            plot_size(summary, args.output / f"{policy}_mae.png", "mae", "Sum MAE ↓")
            plot_size(summary, args.output / f"{policy}_accuracy.png", "exact_round_accuracy",
                      "Exact sum after rounding ↑")
            plot_paired(runs, args.output / f"{policy}_paired.png", policy)
            plot_assignments(runs, args.output / f"{policy}_assignments.png", policy,
                             cfg.get("placement", "prefix"))
    for policy, summary in summaries.items():
        print(f"SELECTION {policy}")
        for arm in ARMS:
            row = summary[arm]
            print(f"{arm:12s} | time {row['training_seconds_mean']:6.2f}s | "
                  f"MAE10 {row['lengths']['10']['mae']['mean']:7.4f} | "
                  f"MAE20 {row['lengths']['20']['mae']['mean']:7.4f} | "
                  f"MAE100 {row['lengths']['100']['mae']['mean']:7.4f} | "
                  f"Acc100 {row['lengths']['100']['exact_round_accuracy']['mean']:5.3f}")


if __name__ == "__main__":
    main()
