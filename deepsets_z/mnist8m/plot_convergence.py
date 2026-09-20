"""Plot held-out image-sum accuracy and MAE versus optimizer steps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .run import ARMS, TEST_LENGTHS


LABELS = {"paper_mlp": "Dense MLP", "learned_z": "Learned z",
          "fixed_z": "Fixed z", "no_z": "No z"}
COLORS = {"paper_mlp": "#222222", "learned_z": "#d55e00",
          "fixed_z": "#7b61a8", "no_z": "#0072b2"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=Path("deepsets_z/mnist8m/results_convergence"))
    parser.add_argument("--output", type=Path,
                        default=Path("deepsets_z/mnist8m/convergence.json"))
    args = parser.parse_args()
    summary: dict = {"source": str(args.results), "arms": {}}
    runs_by_arm = {}
    for arm in ARMS:
        paths = sorted(args.results.glob(f"{arm}_seed*.json"))
        if not paths:
            continue
        runs = [json.loads(path.read_text()) for path in paths]
        if not all(any("probe" in row for row in run["history"]) for run in runs):
            raise ValueError(f"Missing per-length probes in {arm}")
        runs_by_arm[arm] = runs
        rows = {}
        for length in TEST_LENGTHS:
            by_seed = {}
            for run in runs:
                by_seed[str(run["seed"])] = [
                    {"epoch": row["epoch"], "optimizer_steps": row["optimizer_steps"],
                     "elapsed_seconds": row["elapsed_seconds"],
                     **row["probe"][str(length)]}
                    for row in run["history"] if "probe" in row]
            rows[str(length)] = by_seed
        summary["arms"][arm] = {"seeds": [run["seed"] for run in runs],
                                 "lengths": rows}

    for metric, title, suffix in (
        ("exact_round_accuracy", "Exact sum accuracy", "accuracy"),
        ("mae", "Sum MAE", "mae"),
    ):
        fig, axes = plt.subplots(2, 5, figsize=(17, 6.8), sharex=True)
        for ax, length in zip(axes.flat, TEST_LENGTHS):
            for arm, runs in runs_by_arm.items():
                curves = {}
                for run in runs:
                    rows = [row for row in run["history"] if "probe" in row]
                    curves[run["seed"]] = {
                        row["optimizer_steps"]: row["probe"][str(length)][metric]
                        for row in rows}
                common = sorted(set.intersection(*(set(curve) for curve in curves.values())))
                values = np.array([[curve[step] for step in common]
                                   for curve in curves.values()])
                ax.plot(np.array(common) / 1000, np.median(values, axis=0), color=COLORS[arm],
                        linewidth=1.8, label=f"{LABELS[arm]} (n={len(runs)})")
                if len(runs) > 1:
                    ax.fill_between(np.array(common) / 1000, values.min(0),
                                    values.max(0), color=COLORS[arm], alpha=0.13)
                    for individual in values:
                        ax.plot(np.array(common) / 1000, individual,
                                color=COLORS[arm], linewidth=0.7, alpha=0.3)
            ax.set_title(f"{length} images")
            ax.grid(alpha=0.2)
            if metric == "exact_round_accuracy":
                ax.set_ylim(0, 1)
            else:
                ax.set_yscale("log")
            if length >= 30:
                ax.set_xlabel("Optimizer steps (thousands)")
            if length in (5, 30):
                ax.set_ylabel(title)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(labels),
                   bbox_to_anchor=(0.5, 0.99))
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(args.output.with_name(f"{args.output.stem}_{suffix}.png"),
                    dpi=180)
        plt.close(fig)
        for seed in sorted({run["seed"] for runs in runs_by_arm.values() for run in runs}):
            fig, axes = plt.subplots(2, 5, figsize=(17, 6.8), sharex=True)
            for ax, length in zip(axes.flat, TEST_LENGTHS):
                for arm, runs in runs_by_arm.items():
                    run = next((item for item in runs if item["seed"] == seed), None)
                    if run is None:
                        continue
                    rows = [row for row in run["history"] if "probe" in row]
                    ax.plot([row["optimizer_steps"] / 1000 for row in rows],
                            [row["probe"][str(length)][metric] for row in rows],
                            color=COLORS[arm], linewidth=1.8, label=LABELS[arm])
                ax.set_title(f"{length} images")
                ax.grid(alpha=0.2)
                if metric == "exact_round_accuracy":
                    ax.set_ylim(0, 1)
                else:
                    ax.set_yscale("log")
                if length >= 30:
                    ax.set_xlabel("Optimizer steps (thousands)")
                if length in (5, 30):
                    ax.set_ylabel(title)
            handles, labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=len(labels),
                       bbox_to_anchor=(0.5, 0.99))
            fig.tight_layout(rect=(0, 0, 1, 0.93))
            fig.savefig(args.output.with_name(
                f"{args.output.stem}_seed{seed}_{suffix}.png"), dpi=180)
            plt.close(fig)
    if "learned_z" in runs_by_arm and "no_z" in runs_by_arm:
        fig, axes = plt.subplots(2, 5, figsize=(17, 6.8), sharex=True)
        learned = {run["seed"]: run for run in runs_by_arm["learned_z"]}
        without = {run["seed"]: run for run in runs_by_arm["no_z"]}
        for ax, length in zip(axes.flat, TEST_LENGTHS):
            paired = []
            for seed in sorted(learned.keys() & without.keys()):
                a = {row["optimizer_steps"]: row["probe"][str(length)]["exact_round_accuracy"]
                     for row in learned[seed]["history"]
                     if "probe" in row}
                b = {row["optimizer_steps"]: row["probe"][str(length)]["exact_round_accuracy"]
                     for row in without[seed]["history"]
                     if "probe" in row}
                paired.append({step: a[step] - b[step] for step in a.keys() & b.keys()})
            if paired:
                steps = sorted(set.intersection(*(set(row) for row in paired)))
                differences = np.array([[row[step] for step in steps]
                                        for row in paired])
                ax.plot(np.array(steps) / 1000, np.median(differences, axis=0),
                        color=COLORS["learned_z"], linewidth=1.8)
                if len(paired) > 1:
                    ax.fill_between(np.array(steps) / 1000,
                                    differences.min(0), differences.max(0),
                                    color=COLORS["learned_z"], alpha=0.15)
                    for individual in differences:
                        ax.plot(np.array(steps) / 1000, individual,
                                color=COLORS["learned_z"], linewidth=0.7, alpha=0.35)
            ax.axhline(0, color="#222222", linewidth=0.8)
            ax.set_title(f"{length} images")
            ax.grid(alpha=0.2)
            if length >= 30:
                ax.set_xlabel("Optimizer steps (thousands)")
            if length in (5, 30):
                ax.set_ylabel("Learned z minus no z\nexact accuracy")
        fig.tight_layout()
        fig.savefig(args.output.with_name(f"{args.output.stem}_z_difference.png"),
                    dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for arm, runs in runs_by_arm.items():
        for ax, metric in zip(axes, ("mae", "exact_round_accuracy")):
            curves = [{row["optimizer_steps"]: row["validation"][metric]
                       for row in run["history"]} for run in runs]
            common = sorted(set.intersection(*(set(curve) for curve in curves)))
            values = np.array([[curve[step] for step in common] for curve in curves])
            ax.plot(np.array(common) / 1000, np.median(values, axis=0), color=COLORS[arm],
                    linewidth=1.6, label=f"{LABELS[arm]} (n={len(runs)})")
            if len(runs) > 1:
                ax.fill_between(np.array(common) / 1000, values.min(0),
                                values.max(0), color=COLORS[arm], alpha=0.12)
                for individual in values:
                    ax.plot(np.array(common) / 1000, individual,
                            color=COLORS[arm], linewidth=0.7, alpha=0.3)
    axes[0].set(xlabel="Optimizer steps (thousands)", ylabel="Validation MAE")
    axes[0].set_yscale("log")
    axes[1].set(xlabel="Optimizer steps (thousands)",
                ylabel="Validation exact sum accuracy", ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output.with_name(f"{args.output.stem}_validation.png"), dpi=180)
    plt.close(fig)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    for arm, data in summary["arms"].items():
        for length in (10, 50):
            curves = data["lengths"][str(length)]
            for epoch in (10, 50, 100, 200):
                matches = [next((row for row in rows if row["epoch"] == epoch), None)
                           for rows in curves.values()]
                found = [row for row in matches if row is not None]
                if found:
                    print(arm, "length", length, "epoch", epoch,
                          "accuracy_median", round(float(np.median([
                              row["exact_round_accuracy"] for row in found])), 4),
                          "MAE_median", round(float(np.median([row["mae"] for row in found])), 4),
                          "n", len(found))


if __name__ == "__main__":
    main()
