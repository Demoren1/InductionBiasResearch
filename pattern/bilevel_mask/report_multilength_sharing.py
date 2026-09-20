"""Aggregate the 3/5/7 multi-length sharing experiment and draw comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VARIANTS = ("global", "length_latent")
VARIANT_LABELS = {"global": "One global z", "length_latent": "One z per length"}
METHODS = (
    "generated_sharing", "generated_connectivity", "random_sharing",
    "analytic_sharing", "dense",
)
METHOD_LABELS = {
    "generated_sharing": "Generated sharing",
    "generated_connectivity": "Connectivity only",
    "random_sharing": "Random sharing",
    "analytic_sharing": "Analytic sharing",
    "dense": "Dense",
}


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "ci95": (3.182 if len(values) == 4 else 1.96) *
        (stdev(values) if len(values) > 1 else 0.0) / len(values) ** 0.5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    candidates = [json.loads(path.read_text()) for path in sorted(args.root.glob("*/summary.json"))]
    # A longer convergence continuation may coexist with its fixed-budget run.
    # Keep the largest outer budget for every (variant, seed).
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in candidates:
        key = (row["variant"], row["config"]["seed"])
        grouped.setdefault(key, []).append(row)
    records = [
        max(group, key=lambda row: row["config"]["outer_steps"])
        for group in grouped.values()
    ]
    if len(records) != 8:
        raise ValueError(f"expected 8 unique (variant, seed) summaries, found {len(records)}")

    aggregate: dict[str, object] = {"variants": {}}
    for variant in VARIANTS:
        selected = [row for row in records if row["variant"] == variant]
        if len(selected) != 4:
            raise ValueError(f"expected 4 {variant} summaries, found {len(selected)}")
        methods: dict[str, object] = {}
        for method in METHODS:
            methods[method] = {
                field: _stats([
                    row["evaluations"]["test"]["strategies"][method][field]
                    for row in selected
                ])
                for field in ("mean_query_bce", "mean_query_accuracy", "mean_active_iou")
            }
        per_length: dict[str, object] = {}
        for length in ("3", "5", "7"):
            per_length[length] = {
                method: {
                    field: _stats([
                        row["evaluations"]["test"]["strategies"][method]
                        ["per_length"][length][field]
                        for row in selected
                    ])
                    for field in ("mean_query_bce", "mean_query_accuracy")
                }
                for method in ("generated_sharing", "analytic_sharing")
            }
        convergence = {
            "selected_steps": [row["history"][-1]["selected_outer_step"] for row in selected],
            "completed_steps": [row["history"][-1]["completed_outer_steps"] for row in selected],
            "converged": [bool(row["history"][-1]["converged"]) for row in selected],
        }
        aggregate["variants"][variant] = {
            "seeds": [row["config"]["seed"] for row in selected],
            "methods": methods,
            "per_length": per_length,
            "convergence": convergence,
        }

    (args.root / "aggregate.json").write_text(json.dumps(aggregate, indent=2))

    lines = [
        "# Multi-length generated sharing: lengths 3, 5, 7",
        "",
        "Sequence length 32; four seeds per variant. Values are test mean ± sample SD over seeds.",
        "",
        "## Aggregate",
        "",
        "| Variant | Method | BCE ↓ | Accuracy ↑ | Active IoU ↑ |",
        "|---|---|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        methods = aggregate["variants"][variant]["methods"]  # type: ignore[index]
        for method in METHODS:
            row = methods[method]
            lines.append(
                f"| {VARIANT_LABELS[variant]} | {METHOD_LABELS[method]} | "
                f"{row['mean_query_bce']['mean']:.4f} ± {row['mean_query_bce']['std']:.4f} | "
                f"{row['mean_query_accuracy']['mean']:.4f} ± {row['mean_query_accuracy']['std']:.4f} | "
                f"{row['mean_active_iou']['mean']:.4f} ± {row['mean_active_iou']['std']:.4f} |"
            )
    lines.extend([
        "", "## Generated sharing by pattern length", "",
        "| Variant | Length | BCE ↓ | Accuracy ↑ | Analytic BCE ↓ |",
        "|---|---:|---:|---:|---:|",
    ])
    for variant in VARIANTS:
        per_length = aggregate["variants"][variant]["per_length"]  # type: ignore[index]
        for length in ("3", "5", "7"):
            generated = per_length[length]["generated_sharing"]
            analytic = per_length[length]["analytic_sharing"]
            lines.append(
                f"| {VARIANT_LABELS[variant]} | {length} | "
                f"{generated['mean_query_bce']['mean']:.4f} ± {generated['mean_query_bce']['std']:.4f} | "
                f"{generated['mean_query_accuracy']['mean']:.4f} ± {generated['mean_query_accuracy']['std']:.4f} | "
                f"{analytic['mean_query_bce']['mean']:.4f} ± {analytic['mean_query_bce']['std']:.4f} |"
            )
    lines.extend(["", "## Outer-loop convergence", "",
                  "| Variant | Selected steps | Completed steps | Patience stops |",
                  "|---|---|---|---:|"])
    for variant in VARIANTS:
        convergence = aggregate["variants"][variant]["convergence"]  # type: ignore[index]
        lines.append(
            f"| {VARIANT_LABELS[variant]} | {convergence['selected_steps']} | "
            f"{convergence['completed_steps']} | {sum(convergence['converged'])}/4 |"
        )
    lines.extend(["", "![Test comparison](comparison.png)", ""])
    (args.root / "RESULTS.md").write_text("\n".join(lines))

    colors = {"global": "#2864DC", "length_latent": "#E38927"}
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    x = np.arange(len(METHODS))
    width = 0.36
    for offset, variant in zip((-width / 2, width / 2), VARIANTS):
        methods = aggregate["variants"][variant]["methods"]  # type: ignore[index]
        for axis, field, title in zip(
            axes[:2], ("mean_query_bce", "mean_query_accuracy"),
            ("Balanced test BCE ↓", "Balanced test accuracy ↑"),
        ):
            means = [methods[method][field]["mean"] for method in METHODS]
            errors = [methods[method][field]["ci95"] for method in METHODS]
            axis.bar(x + offset, means, width, color=colors[variant], alpha=0.86,
                     label=VARIANT_LABELS[variant])
            axis.errorbar(x + offset, means, yerr=errors, fmt="none", ecolor="#20242A", capsize=3)
            axis.set_title(title, fontweight="bold")
            axis.set_xticks(x, [METHOD_LABELS[method] for method in METHODS], rotation=23, ha="right")
            axis.grid(axis="y", alpha=0.22)
            axis.set_axisbelow(True)

    lengths = ("3", "5", "7")
    x_length = np.arange(len(lengths))
    for offset, variant in zip((-width / 2, width / 2), VARIANTS):
        per_length = aggregate["variants"][variant]["per_length"]  # type: ignore[index]
        means = [per_length[length]["generated_sharing"]["mean_query_bce"]["mean"] for length in lengths]
        errors = [per_length[length]["generated_sharing"]["mean_query_bce"]["ci95"] for length in lengths]
        axes[2].bar(x_length + offset, means, width, color=colors[variant], alpha=0.86,
                    label=VARIANT_LABELS[variant])
        axes[2].errorbar(x_length + offset, means, yerr=errors, fmt="none",
                         ecolor="#20242A", capsize=3)
    analytic = [
        aggregate["variants"]["global"]["per_length"][length]
        ["analytic_sharing"]["mean_query_bce"]["mean"]  # type: ignore[index]
        for length in lengths
    ]
    axes[2].plot(x_length, analytic, "o--", color="#18996A", label="Analytic sharing")
    axes[2].set_title("Generated BCE by length ↓", fontweight="bold")
    axes[2].set_xticks(x_length, lengths)
    axes[2].set_xlabel("Pattern length")
    axes[2].grid(axis="y", alpha=0.22)
    axes[2].set_axisbelow(True)
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    figure.suptitle("Generated sharing across pattern lengths 3, 5, 7 (4 seeds)",
                    fontweight="bold", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(args.root / "comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
