"""Aggregate image digit-sum runs and compare to approximate paper points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .run import ARMS, TEST_LENGTHS


LABELS = {"paper_mlp": "Original MLP, our run",
          "learned_z": "Generated W, learned z",
          "fixed_z": "Generated W, fixed z",
          "no_z": "Generated W, no z"}
COLORS = {"paper_mlp": "#222222", "learned_z": "#d55e00",
          "fixed_z": "#7b61a8", "no_z": "#0072b2"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=Path("deepsets_z/mnist8m/results"))
    parser.add_argument("--output", type=Path,
                        default=Path("deepsets_z/mnist8m/summary.json"))
    args = parser.parse_args()
    reference = json.loads(Path(__file__).with_name("paper_reference.json").read_text())
    summary = {"paper_reference": reference, "arms": {}, "paired_comparisons": {}}
    fig, ax = plt.subplots(figsize=(8, 5))
    mae_fig, mae_ax = plt.subplots(figsize=(8, 5))
    training_fig, training_ax = plt.subplots(figsize=(8, 5))
    seed_runs: dict[int, dict[str, dict]] = {}
    lengths = np.asarray(TEST_LENGTHS)
    ax.plot(lengths, [reference["accuracy"][str(n)] for n in lengths],
            linestyle="--", color="#777777", marker="o",
            label="Deep Sets, paper Fig. 2(b) (approx.)")
    ax.scatter([50], [reference["notebook_reference"]["accuracy_50"]],
               color="#a50f15", marker="D", s=65,
               label="Deep Sets, released notebook at 50 (approx.)")
    for arm in ARMS:
        paths = sorted(args.results.glob(f"{arm}_seed*.json"))
        if not paths:
            continue
        runs = [json.loads(path.read_text()) for path in paths]
        for run in runs:
            seed_runs.setdefault(run["seed"], {})[arm] = run
        values = np.asarray([[run["metrics"][f"test_{length}"]["exact_round_accuracy"]
                              for length in lengths] for run in runs])
        mae = np.asarray([[run["metrics"][f"test_{length}"]["mae"]
                           for length in lengths] for run in runs])
        mean = values.mean(0)
        line, = ax.plot(lengths, mean, marker="o", linewidth=2,
                        color=COLORS[arm],
                        label=f"{LABELS[arm]} (n={len(runs)})")
        if len(runs) > 1:
            ax.fill_between(lengths, values.min(0), values.max(0),
                            color=line.get_color(), alpha=0.10)
            for individual in values:
                ax.plot(lengths, individual, color=line.get_color(),
                        linewidth=0.8, alpha=0.35)
        mae_line, = mae_ax.plot(lengths, mae.mean(0), marker="o", linewidth=2,
                                color=COLORS[arm],
                                label=f"{LABELS[arm]} (n={len(runs)})")
        if len(runs) > 1:
            mae_ax.fill_between(lengths, mae.min(0), mae.max(0),
                                color=mae_line.get_color(), alpha=0.10)
            for individual in mae:
                mae_ax.plot(lengths, individual, color=mae_line.get_color(),
                            linewidth=0.8, alpha=0.35)
        summary["arms"][arm] = {
            "seeds": [run["seed"] for run in runs],
            "parameter_count": runs[0]["parameter_count"],
            "best_epoch": [run["best_epoch"] for run in runs],
            "evaluated_epoch": [run.get("evaluated_epoch", run["best_epoch"])
                                for run in runs],
            "training_seconds": [run["training_seconds"] for run in runs],
            "accuracy_mean": dict(zip(map(str, lengths), mean.tolist())),
            "accuracy_per_seed": {str(length): values[:, i].tolist()
                                  for i, length in enumerate(lengths)},
            "mae_mean": dict(zip(map(str, lengths), mae.mean(0).tolist())),
            "mae_per_seed": {str(length): mae[:, i].tolist()
                             for i, length in enumerate(lengths)},
        }
        histories = [[entry["validation"]["mae"] for entry in run["history"]]
                     for run in runs]
        curves = np.full((len(runs), max(map(len, histories))), np.nan)
        for index, history in enumerate(histories):
            curves[index, :len(history)] = history
        epochs = np.arange(1, curves.shape[1] + 1)
        train_line, = training_ax.plot(epochs, np.nanmean(curves, axis=0),
                                       linewidth=2, color=COLORS[arm],
                                       label=f"{LABELS[arm]} (n={len(runs)})")
        if len(runs) > 1:
            training_ax.fill_between(epochs, np.nanmin(curves, axis=0),
                                     np.nanmax(curves, axis=0),
                                     color=train_line.get_color(), alpha=0.12)
    for arm in ("learned_z", "fixed_z"):
        shared = sorted(set(summary["arms"].get(arm, {}).get("seeds", [])) &
                        set(summary["arms"].get("no_z", {}).get("seeds", [])))
        if not shared:
            continue
        paired = {}
        for length in lengths:
            values = {}
            for seed in shared:
                a = json.loads((args.results / f"{arm}_seed{seed}.json").read_text())
                b = json.loads((args.results / f"no_z_seed{seed}.json").read_text())
                values[str(seed)] = (a["metrics"][f"test_{length}"]["exact_round_accuracy"] -
                                     b["metrics"][f"test_{length}"]["exact_round_accuracy"])
            paired[str(length)] = values
        summary["paired_comparisons"][f"{arm}_minus_no_z"] = paired
    difference_fig, difference_ax = plt.subplots(figsize=(8, 4))
    for key, color, label in (("learned_z_minus_no_z", COLORS["learned_z"], "Learned z minus no z"),
                              ("fixed_z_minus_no_z", COLORS["fixed_z"], "Fixed z minus no z")):
        if key not in summary["paired_comparisons"]:
            continue
        pairs = summary["paired_comparisons"][key]
        values = np.asarray([list(pairs[str(length)].values()) for length in lengths])
        difference_ax.plot(lengths, values.mean(1), marker="o", color=color,
                           label=f"{label} (n={values.shape[1]})")
        if values.shape[1] > 1:
            difference_ax.fill_between(lengths, values.min(1), values.max(1),
                                       color=color, alpha=0.12)
    difference_ax.axhline(0, color="#222222", linewidth=1)
    difference_ax.set(xlabel="Number of images in a set",
                      ylabel="Paired exact-accuracy difference", xticks=lengths)
    difference_ax.grid(alpha=0.25)
    difference_ax.legend(fontsize=9)
    difference_fig.tight_layout()
    difference_fig.savefig(args.output.with_name(args.output.stem + "_z_effect.png"), dpi=180)
    plt.close(difference_fig)
    ax.set(xlabel="Number of images in a set", ylabel="Exact sum accuracy after rounding",
           xticks=lengths, ylim=(0, 1.02))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output.with_suffix(".png"), dpi=180)
    plt.close(fig)
    mae_ax.set(xlabel="Number of images in a set", ylabel="Test sum MAE",
               xticks=lengths)
    mae_ax.grid(alpha=0.25)
    mae_ax.legend(fontsize=8)
    mae_fig.tight_layout()
    mae_fig.savefig(args.output.with_name(args.output.stem + "_test_mae.png"), dpi=180)
    plt.close(mae_fig)
    for seed, by_arm in sorted(seed_runs.items()):
        seed_fig, (accuracy_ax, error_ax) = plt.subplots(1, 2, figsize=(13, 4.6))
        accuracy_ax.plot(lengths, [reference["accuracy"][str(n)] for n in lengths],
                         linestyle="--", color="#777777", marker="o",
                         label="Paper Fig. 2(b), approx.")
        accuracy_ax.scatter([50], [reference["notebook_reference"]["accuracy_50"]],
                            color="#a50f15", marker="D", s=50,
                            label="Released notebook at 50, approx.")
        for arm in ARMS:
            if arm not in by_arm:
                continue
            run = by_arm[arm]
            accuracy_ax.plot(lengths, [run["metrics"][f"test_{n}"]
                                       ["exact_round_accuracy"] for n in lengths],
                             color=COLORS[arm], marker="o", label=LABELS[arm])
            error_ax.plot(lengths, [run["metrics"][f"test_{n}"]["mae"]
                                    for n in lengths],
                          color=COLORS[arm], marker="o", label=LABELS[arm])
        accuracy_ax.set(xlabel="Number of images in a set",
                        ylabel="Exact sum accuracy after rounding",
                        xticks=lengths, ylim=(0, 1.02))
        error_ax.set(xlabel="Number of images in a set", ylabel="Test sum MAE",
                     xticks=lengths)
        for seed_ax in (accuracy_ax, error_ax):
            seed_ax.grid(alpha=0.25)
        accuracy_ax.legend(fontsize=7)
        seed_fig.suptitle(f"Seed {seed}")
        seed_fig.tight_layout()
        seed_fig.savefig(args.output.with_name(f"{args.output.stem}_seed{seed}.png"),
                         dpi=180)
        plt.close(seed_fig)
    training_ax.set(xlabel="Epoch", ylabel="Validation MAE")
    training_ax.set_yscale("log")
    training_ax.grid(alpha=0.25)
    training_ax.legend(fontsize=8)
    training_fig.tight_layout()
    training_fig.savefig(args.output.with_name(args.output.stem + "_training.png"), dpi=180)
    plt.close(training_fig)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    for arm, row in summary["arms"].items():
        print(arm, "n=", len(row["seeds"]), "acc10=", row["accuracy_mean"]["10"],
              "acc50=", row["accuracy_mean"]["50"], "params=", row["parameter_count"])


if __name__ == "__main__":
    main()
