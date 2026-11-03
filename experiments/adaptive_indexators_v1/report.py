"""Summarize completed GPU fits and render research figures."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("experiments/adaptive_indexators_v1/results")
FIGURES = Path("mds/assets/2026-11-02/adaptive_indexators")
LABELS = {
    "dense_min_norm": "Dense min-norm",
    "dense_ridge": "Dense ridge",
    "adaptive_partition": "Adaptive partition",
    "base_partition": "Base partition",
    "source_pca": "Source PCA",
    "random_partition": "Random partition",
    "dense_gd": "Dense GD",
    "adaptive_gd": "Adaptive GD",
}


def write_csv(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    complete = json.loads((ROOT / "complete.json").read_text())
    records = list(csv.DictReader((ROOT / "metrics.csv").open()))
    selections = json.loads((ROOT / "selection.json").read_text())
    curves = json.loads((ROOT / "curves.json").read_text())
    timings = json.loads((ROOT / "timings.json").read_text())
    FIGURES.mkdir(parents=True, exist_ok=True)
    bank_groups = defaultdict(list)
    for row in records:
        key = (
            int(row["truth_groups"]),
            float(row["residual"]),
            int(row["size"]),
            row["method"],
            int(row["bank"]),
        )
        bank_groups[key].append(row)
    bank_rows, summaries = [], []
    metrics = [
        "population_mse",
        "train_mse",
        "test_mse",
        "positive_recall",
        "success",
    ]
    aggregate = defaultdict(list)
    for (groups, residual, size, method, bank), rows in bank_groups.items():
        values = {
            name: float(np.mean([float(row[name]) for row in rows]))
            for name in metrics
        }
        bank_rows.append(
            {
                "truth_groups": groups,
                "residual": residual,
                "size": size,
                "method": method,
                "bank": bank,
                **values,
            }
        )
        aggregate[(groups, residual, size, method)].append(values)
    for (groups, residual, size, method), banks in aggregate.items():
        row = {
            "truth_groups": groups,
            "residual": residual,
            "size": size,
            "method": method,
        }
        for name in metrics:
            values = [bank[name] for bank in banks]
            row[name + "_mean"] = float(np.mean(values))
            row[name + "_bank_sd"] = float(np.std(values, ddof=1))
        summaries.append(row)
    write_csv(ROOT / "bank_summary.csv", bank_rows)
    write_csv(ROOT / "summary.csv", summaries)
    summary_map = {
        (row["truth_groups"], row["residual"], row["size"], row["method"]): row
        for row in summaries
    }
    recovery = []
    for choice in selections:
        tag = f"k{choice['truth_groups']}_r{choice['residual']:.2f}_b{choice['bank']}"
        saved = np.load(ROOT / f"{tag}.npz")
        truth = saved["true_indices"]
        found = saved[f"n{choice['size']}_indices"]
        truth_pairs = (truth[:, None] == truth[None]) & (truth[:, None] > 0)
        found_pairs = (found[:, None] == found[None]) & (found[:, None] > 0)
        recovery.append(
            {
                "truth_groups": choice["truth_groups"],
                "residual": choice["residual"],
                "bank": choice["bank"],
                "size": choice["size"],
                "selected_groups": choice["groups"],
                "selected_rank": choice["rank"],
                "active_weights": int((found > 0).sum()),
                "base_partition_exact": bool(
                    np.array_equal(truth_pairs, found_pairs)
                ),
            }
        )
    write_csv(ROOT / "recovery.csv", recovery)
    plt.rcParams.update(
        {"font.size": 10, "axes.grid": True, "grid.alpha": 0.25}
    )
    methods = [
        "dense_ridge",
        "adaptive_partition",
        "source_pca",
        "base_partition",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for row, residual in enumerate([0.0, 0.75]):
        for col, size in enumerate([32, 96]):
            ax = axes[row, col]
            for method in methods:
                selected = [
                    summary_map[(k, residual, size, method)]
                    for k in [4, 8, 16]
                ]
                ax.errorbar(
                    [4, 8, 16],
                    [s["population_mse_mean"] for s in selected],
                    yerr=[s["population_mse_bank_sd"] for s in selected],
                    label=LABELS[method],
                    marker="o",
                    capsize=3,
                )
            ax.set(
                title=f"Residual {residual:.2f}, support {size}",
                xlabel="Base group count",
                ylabel="Population MSE",
                xticks=[4, 8, 16],
                yscale="log",
            )
            ax.legend(fontsize=8)
    fig.suptitle("Mean and standard deviation across five source banks")
    fig.savefig(FIGURES / "population_mse.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for col, (residual, size) in enumerate([(0.0, 32), (0.75, 96)]):
        for method, color in [
            ("dense_gd", "tab:orange"),
            ("adaptive_gd", "tab:blue"),
        ]:
            steps = sorted({record["step"] for record in curves})
            selected = [
                record
                for record in curves
                if record["truth_groups"] == 8
                and record["residual"] == residual
                and record["size"] == size
                and record["method"] == method
            ]
            for split, style in [("train", "-"), ("query", "--")]:
                for row, metric in [(0, "mse"), (1, "positive_recall")]:
                    means = [
                        np.mean(
                            [
                                record[f"{split}_{metric}"]
                                for record in selected
                                if record["step"] == step
                            ]
                        )
                        for step in steps
                    ]
                    axes[row, col].plot(
                        steps,
                        means,
                        color=color,
                        linestyle=style,
                        label=f"{LABELS[method]} {split}",
                    )
        axes[0, col].set(
            title=f"8 base groups, residual {residual:.2f}, n={size}",
            xlabel="GD updates",
            ylabel="Observed-label MSE",
        )
        axes[0, col].set_yscale("symlog", linthresh=0.0001)
        axes[1, col].set(
            xlabel="GD updates", ylabel="Positive-sign recall", ylim=(0, 1.02)
        )
        axes[0, col].legend(fontsize=8)
        axes[1, col].legend(fontsize=8)
    fig.suptitle(
        "Fixed-budget training; query curves do not select checkpoints"
    )
    fig.savefig(FIGURES / "training_curves.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for row, residual in enumerate([0.0, 0.75]):
        for col, size in enumerate([32, 96]):
            ax = axes[row, col]
            for k in [4, 8, 16]:
                chosen = [
                    record["groups"]
                    for record in selections
                    if record["truth_groups"] == k
                    and record["residual"] == residual
                    and record["size"] == size
                ]
                ax.scatter(
                    k + np.linspace(-0.5, 0.5, len(chosen)), chosen, s=35
                )
            ax.plot([4, 8, 16], [4, 8, 16], "k--", label="Base group count")
            ax.set(
                title=f"Residual {residual:.2f}, support {size}",
                xlabel="Base group count",
                ylabel="Selected trainable groups",
                xticks=[4, 8, 16],
                ylim=(0, 135),
            )
            ax.legend(fontsize=8)
    fig.suptitle("Each dot is a separately rebuilt source bank")
    fig.savefig(FIGURES / "selected_groups.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, residual in zip(axes, [0.0, 0.75]):
        for size in [32, 96]:
            paths = [
                row["partition_query_path"]
                for row in selections
                if row["truth_groups"] == 8
                and row["residual"] == residual
                and row["size"] == size
            ]
            ax.plot(
                np.arange(129), np.mean(paths, axis=0), label=f"Support {size}"
            )
        ax.set(
            title=f"8 base groups, residual {residual:.2f}",
            xlabel="Candidate group count",
            ylabel="Source query MSE",
            yscale="log",
        )
        ax.legend()
    fig.savefig(FIGURES / "calibration_path.png", dpi=170)
    plt.close(fig)

    totals = {
        key: sum(row[key] for row in timings)
        for key in [
            "source_seconds",
            "hierarchy_seconds",
            "calibration_seconds",
            "target_seconds",
        ]
    }
    exact = [row for row in recovery if row["residual"] == 0]
    evidence = {
        **complete,
        "exact_structure_choices": sum(
            row["base_partition_exact"] for row in exact
        ),
        "total_exact_condition_choices": len(exact),
        "timing_seconds": totals,
        "aggregation": "Mean of bank means; SD across five banks, not across 80 tasks",
    }
    (ROOT / "summary.json").write_text(json.dumps(evidence, indent=2) + "\n")
    for key in [
        (k, 0.0, size, "adaptive_partition")
        for k in [4, 8, 16]
        for size in [32, 96]
    ]:
        print(key, summary_map[key])
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
