"""Render completed G23 comparisons from frozen structural and target audits."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.darts_structured_tasks_v1.evaluate import align_indices

from .shared_structure_bilevel_report import read_frozen_audit

BASE = Path(__file__).resolve().parent
PRIMARY = BASE / "results/20261105/shared_structure_g23"
SUPPLEMENT = BASE / "results/20261105/shared_bilevel_g23"
SHORT = {
    "prefix": "Prefix",
    "dense_dp": "Dense GN+DP",
    "lc": "LC",
    "lutq": "LUT-Q",
    "yeh": "Joint P/C",
    "yeh_bilevel_direct": "T/V direct",
    "yeh_bilevel_literal": "T/V Neumann20",
}
COLORS = ["#087f8c", "#677489", "#907aa9", "#d3a73f", "#cf8275", "#7a9b76", "#ad697a"]


def verified_targets(root):
    phase = root / "transfer"
    freeze = json.loads((phase / "fit_frozen.json").read_text())
    for relative, expected in freeze["sha256"].items():
        if digest(phase / relative) != expected:
            raise RuntimeError("Frozen target model changed")
    audit = json.loads((phase / "audit.json").read_text())
    if audit["target_fit_freeze_sha256"] != digest(phase / "fit_frozen.json"):
        raise RuntimeError("Target audit/freeze mismatch")
    return audit["records"]


def comparison(destination, structures, targets):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    rows = []
    positions = np.arange(len(SHORT))
    for column, sigma in enumerate((0.2, 2.0)):
        reference = [
            row["mean_target_clean_mse"]
            for row in targets
            if row["method"] == "oracle_true_I"
            and row["noise"] == sigma
            and row["origin"] == "primary"
        ]
        axes[1, column].axhline(
            np.mean(reference),
            color="black",
            linestyle="--",
            label="True-I numeric control mean",
        )
        for position, (method, label) in enumerate(SHORT.items()):
            cases = [
                row
                for row in structures
                if row["noise"] == sigma and row["method"] == method
            ]
            risk = [
                row
                for row in targets
                if row["noise"] == sigma
                and row["method"] == method
                and row["scope"] == "common4"
            ]
            if (
                len(cases) != 6
                or len(risk) != 6
                or not all(row["complete"] for row in risk)
            ):
                raise RuntimeError("Incomplete final comparison")
            exact = sum(row["exact_I"] for row in cases)
            axes[0, column].bar(position, exact, color=COLORS[position], width=0.7)
            axes[0, column].text(position, exact + 0.12, f"{exact}/6", ha="center")
            family_risks = np.array([row["mean_target_clean_mse"] for row in risk])
            axes[1, column].scatter(
                position + np.linspace(-0.17, 0.17, 6),
                family_risks,
                facecolors="none",
                edgecolors=COLORS[position],
                s=50,
            )
            axes[1, column].scatter(
                position,
                family_risks.mean(),
                color=COLORS[position],
                marker="D",
                s=90,
                zorder=5,
            )
            rows.append(
                {
                    "noise": sigma,
                    "method": method,
                    "exact_I": exact,
                    "exact_mask": sum(row["exact_mask"] for row in cases),
                    "cases": 6,
                    "target_tasks": 48,
                    "mean_clean_MSE": float(family_risks.mean()),
                    "mean_excess_MSE": float(
                        np.mean([row["mean_excess_mse_over_true_I"] for row in risk])
                    ),
                    "true_I_control_mean_MSE": float(np.mean(reference)),
                }
            )
        axes[0, column].set_title(f"Observation noise SD = {sigma}")
        axes[0, column].set_ylabel("Exactly recovered signed I")
        axes[0, column].set_ylim(0, 6.8)
        axes[0, column].set_yticks(range(7))
        axes[1, column].set_yscale("log")
        axes[1, column].set_ylabel("Clean target MSE: 8 fresh C per I")
        axes[1, column].legend(fontsize=8, loc="lower right")
        for axis in axes[:, column]:
            axis.set_xticks(positions, SHORT.values(), rotation=30, ha="right")
            axis.grid(axis="y", alpha=0.2)
            axis.set_axisbelow(True)
    fig.suptitle(
        "Completed G23 pilot: 6 independent structures, paired noise conditions\n"
        "Transfer dots: one mean per structure; diamonds: six-structure mean; compute not matched",
        fontsize=14,
    )
    fig.savefig(destination / "structure_and_transfer.png", dpi=150)
    plt.close(fig)
    with (destination / "comparison.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def hard_case(destination):
    with np.load(PRIMARY / "common/audit_only.npz") as arrays:
        expected = arrays["family_4__truth_I"]
    values = [(expected, "True I")]
    records = []
    for method, label in SHORT.items():
        origin = SUPPLEMENT if method.startswith("yeh_bilevel") else PRIMARY
        with np.load(origin / "common/selected/job_009.npz") as arrays:
            index = arrays[f"{method}__common4__index"]
        _, aligned, _ = align_indices(index, expected, 4)
        errors = int((aligned != expected).sum())
        values.append((aligned, f"{label}: {errors} errors"))
        records.append({"method": method, "categorical_errors": errors})
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    for axis, (value, title) in zip(axes.ravel(), values, strict=True):
        rendered = axis.imshow(value, cmap="RdBu_r", vmin=-4, vmax=4)
        if title != "True I":
            for row, column in np.argwhere(value != expected):
                axis.add_patch(
                    Rectangle(
                        (column - 0.5, row - 0.5),
                        1,
                        1,
                        fill=False,
                        linewidth=0.45,
                        edgecolor="black",
                    )
                )
        axis.set_title(title, fontsize=11)
        axis.set_xticks((0, 7, 15))
        axis.set_yticks((0, 7, 15))
    fig.colorbar(
        rendered, ax=axes.ravel().tolist(), shrink=0.65, label="Signed code ID"
    )
    fig.suptitle(
        "Remaining prefix failure: random I=9921, SD=2\n"
        "Black outlines mark categorical errors after permitted audit-only alignment"
    )
    fig.savefig(destination / "all_methods_hard_case.png", dpi=150)
    plt.close(fig)
    return records


def main():
    destination = PRIMARY / "final_report"
    destination.mkdir(exist_ok=False)
    structures = read_frozen_audit(PRIMARY) + read_frozen_audit(SUPPLEMENT)
    targets = []
    for origin, name in ((PRIMARY, "primary"), (SUPPLEMENT, "supplement")):
        targets.extend({**row, "origin": name} for row in verified_targets(origin))
    rows = comparison(destination, structures, targets)
    errors = hard_case(destination)
    inputs = [
        origin / relative
        for origin in (PRIMARY, SUPPLEMENT)
        for relative in (
            "protocol.json",
            "common/selection_frozen.json",
            "common/audit.json",
            "transfer/protocol.json",
            "transfer/data_frozen.json",
            "transfer/fit_frozen.json",
            "transfer/audit.json",
        )
    ]
    save_json(
        destination / "provenance.json",
        {
            "inputs_sha256": {str(path): digest(path) for path in inputs},
            "execution_code_sha256": digest(Path(__file__)),
            "comparison": rows,
            "hard_case": errors,
            "new_training_or_selection": False,
            "overall_research_goal_complete": False,
        },
    )
    source = Path(__file__).read_text()
    (destination / "executed_code.py").write_text(source)


if __name__ == "__main__":
    main()
