"""Render the completed G24 ablation from frozen selections and late audits."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Rectangle

from experiments.darts_structured_tasks_v1.evaluate import align_indices


def digest(path):
    """Record the exact input artifact used by this report."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root
    audit = json.loads((root / "common/audit.json").read_text())
    freeze = root / "common/selection_frozen.json"
    if audit["selection_freeze_sha256"] != digest(freeze):
        raise RuntimeError("Late audit does not match the frozen selections")
    records = audit["records"]
    report = root / "report"
    method_labels = {
        "bootstrap": "Frozen bootstrap",
        "axis_single_refit": "Axis C / single / refit",
        "bootstrap_single_fixed": "Calibrated / single / fixed",
        "bootstrap_single_refit": "Calibrated / single / refit",
        "bootstrap_joint_fixed": "Calibrated / joint / fixed",
        "bootstrap_joint_refit": "Calibrated / joint / refit",
    }
    source_rows = []
    for noise in (0.2, 2.0):
        for method in method_labels:
            rows = [
                row
                for row in records
                if row["scope"] == "source1"
                and row["noise"] == noise
                and row["method"] == method
            ]
            if rows:
                source_rows.append(
                    {
                        "noise": noise,
                        "method": method,
                        "cases": len(rows),
                        "exact_I": sum(row["exact_I"] for row in rows),
                        "complete": sum(row["complete"] for row in rows),
                    }
                )
    with (report / "source0_comparison.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
    positions = np.arange(len(method_labels))
    for axis, noise in zip(axes, (0.2, 2.0), strict=True):
        counts = []
        for method in method_labels:
            rows = [
                row
                for row in records
                if row["scope"] == "common4"
                and row["method"] == method
                and row["noise"] == noise
            ]
            if len(rows) != 6:
                raise RuntimeError("Incomplete common4 comparison")
            counts.append(sum(row["exact_I"] for row in rows))
        colors = ["#90979e", "#cc9a64"] + ["#338a9e"] * 4
        bars = axis.barh(positions, counts, color=colors)
        axis.bar_label(bars, labels=[f"{n}/6" for n in counts], padding=5)
        source = [row for row in source_rows if row["noise"] == noise]
        axis.scatter(
            [row["exact_I"] for row in source],
            [list(method_labels).index(row["method"]) for row in source],
            marker="D",
            facecolor="white",
            edgecolor="#222222",
            s=40,
            zorder=3,
            label="Only source0 candidates",
        )
        axis.set_yticks(positions, method_labels.values(), fontsize=9)
        axis.invert_yaxis()
        axis.set_xlim(0, 6.8)
        axis.set_xticks(np.arange(7))
        axis.set_xlabel("Exact signed indexators / 6 structures")
        axis.set_title(f"Raw observation noise SD = {noise:g}")
        axis.grid(axis="x", alpha=0.18)
        axis.set_axisbelow(True)
    axes[1].legend(loc="lower right", fontsize=9)
    fig.suptitle(
        "G24: codebook calibration fixes the remaining sharing error\n"
        "Bars: bank of four sources; same six opened structures, fresh query noise"
    )
    fig.savefig(report / "g24_ablation.png", dpi=170)
    plt.close(fig)

    family, case = 4, 9
    selected_path = root / "common/selected" / f"job_{case:03d}.npz"
    truth_path = root / "common/audit_only.npz"
    methods = (
        "bootstrap",
        "axis_single_refit",
        "bootstrap_single_fixed",
        "bootstrap_joint_fixed",
    )
    short_labels = (
        "Bootstrap I",
        "Axis C, fresh search",
        "Calibrated C, single",
        "Calibrated C, joint",
    )
    with np.load(truth_path) as truth, np.load(selected_path) as selected:
        expected = truth[f"family_{family}__truth_I"]
        values = [(expected, "True I")]
        for method, label in zip(methods, short_labels, strict=True):
            _, aligned, _ = align_indices(
                selected[f"{method}__common4__index"], expected, 4
            )
            errors = int(np.count_nonzero(aligned != expected))
            values.append((aligned, f"{label}\n{errors} index errors"))
        fig, axes = plt.subplots(
            1, len(values), figsize=(15, 3.8), constrained_layout=True
        )
        colormap = plt.get_cmap("RdBu_r", 9)
        normalization = BoundaryNorm(np.arange(-4.5, 5.5), colormap.N)
        for axis, (value, label) in zip(axes, values, strict=True):
            rendered = axis.imshow(value, cmap=colormap, norm=normalization)
            for row, column in np.argwhere(value != expected):
                axis.add_patch(
                    Rectangle(
                        (column - 0.5, row - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="black",
                        linewidth=1.7,
                    )
                )
            axis.set_title(label, fontsize=11)
            axis.set_xticks((0, 15))
            axis.set_yticks((0, 15))
            axis.set_xlabel("Hidden-unit column")
        axes[0].set_ylabel("Input coordinate")
        fig.colorbar(
            rendered, ax=axes.tolist(), ticks=np.arange(-4, 5), shrink=0.66
        ).set_label("Signed index")
        fig.suptitle(
            "random9921, SD=2: exact mask throughout, four sharing errors corrected\n"
            "Labels, signs and hidden columns matched only in the late audit"
        )
        fig.savefig(report / "g24_hard_case.png", dpi=170)
        plt.close(fig)

    inputs = (
        root / "protocol.json",
        root / "common/audit.json",
        freeze,
        selected_path,
        truth_path,
    )
    provenance = {
        "source_sha256": {str(path): digest(path) for path in inputs},
        "report_code_sha256": digest(Path(__file__)),
        "teacher_access": "Late display/audit only; no fitting or selection",
        "new_fits_or_queries": False,
        "independent_new_structures": False,
        "raw_labels_per_function_noise_condition": 184549376,
        "joint_specific_accuracy_gain_over_calibrated_single_bank": False,
    }
    (report / "final_figures_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
