"""Plot descriptive transfer, numerical fitting and all recovered structures."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .score import LABELS

METHODS = ("current", "signature", "dense", "signature_random")
COLORS = dict(zip(METHODS, ("#17649b", "#c74927", "#363636", "#8b66a8")))


def plots(output):
    with (output / "transfer_per_task.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    plan = json.loads((output / "plan.json").read_text())
    root = Path(plan["root"])
    jobs = {job["id"]: job for job in plan["jobs"]}
    assignments = {
        (a["case"], a["condition"], a["method"]): a["job"] for a in plan["assignments"]
    }
    sources = json.loads((output / "structure_freeze.json").read_text())["cases"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for col, geometry in enumerate(("local", "quartet_mix", "random")):
        for condition in (0, 1):
            axis = axes[condition, col]
            available = [
                r["case"]
                for r in sources
                if r["geometry"] == geometry
                and r["condition"] == condition
                and r["signature_available"]
            ]
            for method in METHODS:
                means = [
                    np.mean(
                        [
                            float(r["mse"])
                            for r in rows
                            if r["case"] in available
                            and int(r["condition"]) == condition
                            and r["method"] == method
                            and int(r["budget"]) == n
                        ]
                    )
                    for n in (64, 256, 1024)
                ]
                axis.plot(
                    (64, 256, 1024),
                    np.maximum(means, 1e-14),
                    marker="o" if method != "signature" else "x",
                    linestyle="--" if method == "signature" else "-",
                    color=COLORS[method],
                    label=LABELS[method],
                )
            axis.set_title(
                f"{geometry}, source noise {0.2 * condition:.1f}; {len(available)}/2 strata"
            )
            axis.set(
                xscale="log",
                yscale="log",
                xticks=(64, 256, 1024),
                xlabel="Support observations N",
                ylabel="Mean query MSE",
            )
            axis.set_xticklabels((64, 256, 1024))
            axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Fresh numerical transfer; identical available strata within each panel",
        fontsize=13,
    )
    for extension in ("png", "pdf"):
        fig.savefig(output / f"transfer.{extension}", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for col, geometry in enumerate(("local", "quartet_mix", "random")):
        case = f"case_{2 * col:02d}"
        for condition in (0, 1):
            axis = axes[condition, col]
            for method in METHODS:
                job_id = assignments.get((case, condition, method))
                if job_id is None:
                    continue
                job = jobs[job_id]
                folder = (
                    root / "targets" / f"{case}_c{job['condition']}" / job["method"]
                )
                with np.load(folder / "curves.npz") as stored:
                    history = stored["validation_history_1024"]
                    running = np.minimum.accumulate(history, axis=0).min(1).mean(1)
                    axis.plot(
                        stored["steps_1024"],
                        np.maximum(running, 1e-14),
                        color=COLORS[method],
                        linestyle="--" if method == "signature" else "-",
                        label=LABELS[method],
                    )
            axis.set(
                title=f"{geometry}, source noise {0.2 * condition:.1f}, stratum 0",
                yscale="log",
                xlabel="Adam updates",
                ylabel="Normalized numerical-selection MSE",
            )
            axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "N=1024: best numerical-selection checkpoint so far; before GN", fontsize=13
    )
    fig.savefig(output / "numerical_selection.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    with np.load(output / "recovery_structures.npz") as stored:
        structures = dict(stored)
    for number, geometry in enumerate(("local", "quartet_mix", "random")):
        fig, axes = plt.subplots(4, 5, figsize=(13, 10), layout="constrained")
        image = None
        for row, (stratum, condition) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            case = f"case_{2 * number + stratum:02d}"
            for col, method in enumerate(
                ("truth", "current", "signature", "signature_random")
            ):
                key = (
                    f"{case}_truth"
                    if method == "truth"
                    else f"{case}_c{condition}_{method}"
                )
                axis = axes[row, col]
                if key in structures:
                    image = axis.imshow(
                        structures[key],
                        cmap="bwr",
                        vmin=-4,
                        vmax=4,
                        interpolation="nearest",
                    )
                else:
                    axis.text(
                        0.5,
                        0.5,
                        "No candidate",
                        ha="center",
                        va="center",
                        transform=axis.transAxes,
                    )
                axis.set_xticks([])
                axis.set_yticks([])
                if row == 0:
                    axis.set_title(
                        {
                            "truth": "Generating I",
                            "current": "G34",
                            "signature": "Weighted SVD",
                            "signature_random": "Shuffled I",
                        }[method],
                        fontsize=10,
                    )
                if col == 0:
                    axis.set_ylabel(
                        f"O={1 + stratum}, source noise={0.2 * condition:.1f}"
                    )
            job = jobs[assignments[(case, condition, "dense")]]
            folder = root / "targets" / f"{case}_c{job['condition']}" / job["method"]
            with np.load(folder / "frozen_predictions.npz") as stored:
                weight = stored["parameters_1024"][0, :256].reshape(16, 16)
            axis = axes[row, 4]
            limit = max(float(np.abs(weight).max()), 1e-8)
            dense = axis.imshow(
                weight, cmap="PiYG", vmin=-limit, vmax=limit, interpolation="nearest"
            )
            fig.colorbar(dense, ax=axis, fraction=0.045, pad=0.02)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title("Dense target W1, task 0", fontsize=10)
        if image is not None:
            fig.colorbar(
                image,
                ax=axes[:, :4].ravel().tolist(),
                shrink=0.4,
                label="Signed code label; zero is inactive",
            )
        fig.suptitle(
            f"{geometry}: late reference alignment; Dense uses native hidden-unit order",
            fontsize=13,
        )
        fig.savefig(output / f"{geometry}_structures.png", dpi=170, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plots(args.output.resolve())


if __name__ == "__main__":
    main()
