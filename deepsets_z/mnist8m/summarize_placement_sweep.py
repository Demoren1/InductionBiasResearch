"""Summarize completed one-seed layer-placement runs and draw comparison plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .placement_sweep import BY_NAME, SPECS
from .run import TEST_LENGTHS


LAYER_LABELS = {None: "Dense control", "first": "First 784→300",
                "second": "Middle 300→100", "third": "Last 100→30"}
COLORS = {None: "#222222", "first": "#d55e00", "second": "#0072b2",
          "third": "#009e73"}
PROBE_LENGTHS = (5, 20, 35, 50)


def plot_probes(records: dict, output: Path, metric: str) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(16, 15), constrained_layout=True,
                             sharex=True, sharey=True)
    for column, layer in enumerate(("third", "second", "first")):
        for row, length in enumerate(PROBE_LENGTHS):
            ax = axes[row, column]
            for spec in SPECS:
                if spec.layer not in (None, layer):
                    continue
                record = records.get(spec.name)
                if record is None:
                    continue
                history = [entry for entry in record["history"]
                           if str(length) in entry.get("probe", {})]
                if not history:
                    continue
                scale = 100 if metric == "exact_round_accuracy" else 1
                ax.plot([entry["optimizer_steps"] for entry in history],
                        [scale * entry["probe"][str(length)][metric]
                         for entry in history], marker="o", markersize=2.5,
                        linewidth=1.5, label=spec.name)
            if row == 0:
                ax.set_title(LAYER_LABELS[layer])
            if column == 0:
                ax.set_ylabel(f"Length {length}\n" +
                              ("Exact accuracy, %" if metric == "exact_round_accuracy"
                               else "MAE"))
            if row == len(PROBE_LENGTHS) - 1:
                ax.set_xlabel("Optimizer steps")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)
    if metric == "exact_round_accuracy":
        axes[0, 0].set_ylim(0, 102)
        filename = "placement_sweep_probe_accuracy.png"
    else:
        for row in axes:
            for ax in row:
                ax.set_yscale("log")
        filename = "placement_sweep_probe_mae.png"
    fig.savefig(output.with_name(filename), dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(
        "deepsets_z/mnist8m/results_placement_sweep"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path(
        "deepsets_z/mnist8m/placement_sweep_summary.md"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = {}
    for spec in SPECS:
        path = args.results / f"{spec.name}_seed{args.seed}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record["config"]["spec"] != vars(spec):
                raise ValueError(f"Spec mismatch in {path}")
            records[spec.name] = record
    if not records:
        raise SystemExit(f"No completed runs for seed {args.seed} in {args.results}")
    epoch_counts = {record["config"]["max_epochs"] for record in records.values()}
    test_counts = {record["config"]["test_set_count"] for record in records.values()}
    if len(epoch_counts) != 1 or len(test_counts) != 1:
        raise ValueError("Runs with different epoch or test-set counts cannot be compared")

    lines = ["# Generated-layer placement on MNIST8m", "",
             f"Seed: **{args.seed}**. Completed: **{len(records)}/{len(SPECS)}** runs.",
             f"All runs use explicit padding masks, the same sets and optimizer. "
             f"Training epochs: **{next(iter(epoch_counts))}**. One seed gives an exploratory comparison, "
             "not an uncertainty estimate.", "",
             "| Model | Generated layer | Generator | Parameters | Test accuracy, L=50 | Test MAE, L=50 | Train time |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for spec in SPECS:
        record = records.get(spec.name)
        if record is None:
            continue
        profile = ("dense" if spec.layer is None else
                   (f"{spec.mode}, width {spec.hidden}, K={spec.values}" if
                    spec.mode == "discrete" else f"direct, width {spec.hidden}"))
        metric = record["test"]["50"]
        lines.append(f"| `{spec.name}` | {LAYER_LABELS[spec.layer]} | {profile} | "
                     f"{record['parameter_count']:,} | "
                     f"{100 * metric['exact_round_accuracy']:.2f}% | "
                     f"{metric['mae']:.3f} | "
                     f"{record['training_seconds'] / 60:.1f} min |")
    missing = [spec.name for spec in SPECS if spec.name not in records]
    if missing:
        lines.extend(("", "Pending: " + ", ".join(f"`{name}`" for name in missing) + "."))
    lines.extend(("", "![Accuracy versus model size](placement_sweep_size.png)", "",
                  "![Accuracy by set length](placement_sweep_lengths.png)", "",
                  "![Validation convergence](placement_sweep_training.png)", "",
                  "![Accuracy over steps by set length](placement_sweep_probe_accuracy.png)", "",
                  "![MAE over steps by set length](placement_sweep_probe_mae.png)", ""))
    args.output.write_text("\n".join(lines))

    size_fig, (size_ax, error_ax) = plt.subplots(
        1, 2, figsize=(12, 5), constrained_layout=True)
    for spec in SPECS:
        record = records.get(spec.name)
        if record is None:
            continue
        metric = record["test"]["50"]
        count = record["parameter_count"]
        color = COLORS[spec.layer]
        size_ax.scatter(count, 100 * metric["exact_round_accuracy"],
                        color=color, s=55)
        error_ax.scatter(count, metric["mae"], color=color, s=55)
        size_ax.annotate(spec.name, (count, 100 * metric["exact_round_accuracy"]),
                         xytext=(3, 4), textcoords="offset points", fontsize=7)
        error_ax.annotate(spec.name, (count, metric["mae"]),
                          xytext=(3, 4), textcoords="offset points", fontsize=7)
    for ax in (size_ax, error_ax):
        ax.set_xscale("log")
        ax.set_xlabel("Trainable parameters")
        ax.grid(alpha=0.2)
    size_ax.set_ylabel("Exact sum accuracy at length 50, %")
    size_ax.set_ylim(0, 102)
    error_ax.set_ylabel("MAE at length 50")
    error_ax.set_yscale("log")
    size_fig.savefig(args.output.with_name("placement_sweep_size.png"), dpi=180)
    plt.close(size_fig)

    length_fig, axes = plt.subplots(1, 3, figsize=(15, 4.5),
                                   sharey=True, constrained_layout=True)
    dense = records.get("dense")
    for ax, layer in zip(axes, ("third", "second", "first")):
        if dense:
            ax.plot(TEST_LENGTHS,
                    [100 * dense["test"][str(n)]["exact_round_accuracy"]
                     for n in TEST_LENGTHS], color=COLORS[None],
                    linewidth=2, label="dense")
        for spec in SPECS:
            record = records.get(spec.name)
            if spec.layer != layer or record is None:
                continue
            ax.plot(TEST_LENGTHS,
                    [100 * record["test"][str(n)]["exact_round_accuracy"]
                     for n in TEST_LENGTHS], marker="o", markersize=3,
                    label=spec.name)
        ax.set(title=LAYER_LABELS[layer], xlabel="Set length",
               xticks=TEST_LENGTHS, ylim=(0, 102))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Exact sum accuracy, %")
    length_fig.savefig(args.output.with_name("placement_sweep_lengths.png"), dpi=180)
    plt.close(length_fig)

    training_fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for spec in SPECS:
        record = records.get(spec.name)
        if record is None:
            continue
        history = record["history"]
        ax.plot([row["optimizer_steps"] for row in history],
                [row["validation"]["mae"] for row in history],
                label=spec.name, color=COLORS[spec.layer],
                linestyle="-" if spec.mode == "dense" else "--", alpha=0.8)
    ax.set(xlabel="Optimizer steps", ylabel="Short-set validation MAE")
    ax.set_yscale("log")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    training_fig.savefig(args.output.with_name("placement_sweep_training.png"), dpi=180)
    plt.close(training_fig)
    plot_probes(records, args.output, "exact_round_accuracy")
    plot_probes(records, args.output, "mae")
    print(args.output)


if __name__ == "__main__":
    main()
