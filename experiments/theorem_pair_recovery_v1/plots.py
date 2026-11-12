"""Display every frozen condition, with truth used only for alignment."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from experiments.paper_structure_benchmark_v1.reporting.evaluate import align_index


def render(output):
    with (output / "recovery.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    arrays = dict(np.load(output / "selected_structures.npz"))
    palette = ListedColormap(
        [
            "#5b2c83",
            "#225ea8",
            "#d6604d",
            "#238b45",
            "#ffffff",
            "#a1d99b",
            "#f4a582",
            "#9ecae1",
            "#c2a5cf",
        ]
    )
    norm = BoundaryNorm(np.arange(-4.5, 5.5), palette.N)
    for family in sorted({row["family"] for row in rows}):
        cases = [
            row
            for row in rows
            if row["family"] == family and row["method"] == "baseline_full"
        ]
        figure, axes = plt.subplots(
            len(cases), 5, figsize=(15, 2.9 * len(cases)), squeeze=False
        )
        for row_number, entry in enumerate(cases):
            prefix = f"{entry['cohort']}_{entry['case']}_c{entry['condition']}"
            truth = arrays[f"{prefix}_baseline_full_truth"]
            for column, method in enumerate(
                (
                    "truth",
                    "baseline_full",
                    "geometry_top1",
                    "signature_only",
                    "augmented",
                )
            ):
                axis = axes[row_number, column]
                key = f"{prefix}_{method}_aligned"
                if method == "truth":
                    image, title = truth, "True structure"
                elif method == "geometry_top1":
                    candidates = np.load(
                        output
                        / entry["cohort"]
                        / f"{entry['case']}_c{entry['condition']}"
                        / "candidates.npz"
                    )["indices"]
                    if not len(candidates):
                        axis.set_axis_off()
                        continue
                    image, metrics = align_index(
                        candidates[0], truth, int(np.abs(truth).max())
                    )
                    title = f"Geometry top 1: {metrics['signed_entry_errors']} errors"
                elif key not in arrays:
                    axis.text(
                        0.5,
                        0.5,
                        "No complete candidate",
                        ha="center",
                        va="center",
                        transform=axis.transAxes,
                    )
                    axis.set_axis_off()
                    continue
                else:
                    image = arrays[key]
                    errors = int((image != truth).sum())
                    title = {
                        "baseline_full": "G34",
                        "signature_only": "Signatures",
                        "augmented": "Combined",
                    }[method]
                    title += f": {errors} errors"
                axis.imshow(
                    image,
                    cmap=palette,
                    norm=norm,
                    aspect="auto",
                    interpolation="nearest",
                )
                axis.set_title(title, fontsize=10)
                axis.set_xticks([])
                axis.set_yticks([])
                if column == 0:
                    axis.set_ylabel(
                        f"{entry['case']}, c{entry['condition']}", fontsize=10
                    )
        figure.suptitle(
            f"{family}: signed indexators after source-only selection", fontsize=14
        )
        figure.tight_layout(rect=(0, 0, 1, 0.98))
        figure.savefig(output / f"{family}_structures.png", dpi=150)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.output)


if __name__ == "__main__":
    main()
