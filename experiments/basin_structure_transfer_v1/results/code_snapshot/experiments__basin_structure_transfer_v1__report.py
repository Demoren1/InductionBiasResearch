"""Report every completed frequency regime without additional fitting."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import betainc

from experiments.basin_structure_transfer_v1.core import canonicalize

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT.parents[1] / "mds/assets/2026-11-02/basin_structure_transfer"


def collect(root, frequency, bank):
    plan = json.loads((root / "protocol.json").read_text())
    directory = root / "evaluation"
    if not (directory / "complete.json").exists():
        raise RuntimeError(f"Incomplete evaluation: {root}")
    teacher = np.load(directory / "teacher_diagnostic.npz")
    known = teacher["families"] < plan["families"]
    rows, curves, proposals = [], {}, {}
    for path in sorted(directory.glob("*/models.npz")):
        model = np.load(path)
        query, risk = model["query"][:, 0], model["population"][:, 0]
        selected = risk[np.arange(len(risk)), query.argmin(1)]
        timing = json.loads((path.parent / "timing.json").read_text())
        rows.append({
            "frequency": frequency, "method": path.parent.name,
            "known_success": int((selected[known] < plan["success_mse"]).sum()),
            "known_tasks": int(known.sum()),
            "known_mean_mse": float(selected[known].mean()),
            "known_median_mse": float(np.median(selected[known])),
            "foreign_success": int((selected[~known] < plan["success_mse"]).sum()),
            "foreign_tasks": int((~known).sum()),
            "foreign_mean_mse": float(selected[~known].mean()),
            "known_attempt_success": float((risk[known] < plan["success_mse"]).mean()),
            "numeric_fit_seconds": timing["seconds"],
            "starts": timing["starts"], "steps": timing["steps"],
        })
        method_curve = []
        for count in (1, 2, 4, 8, 16, 32):
            if count > risk.shape[1]:
                continue
            choice = query[:, :count].argmin(1)
            losses = risk[np.arange(len(risk)), choice]
            method_curve.append({
                "updates": count * timing["steps"], "attempts": count,
                "known_success": float((losses[known] < plan["success_mse"]).mean()),
                "known_mean_mse": float(losses[known].mean()),
            })
        curves[path.parent.name] = method_curve
        if "indices" in model.files:
            indices = canonicalize(model["indices"])
            exact = np.all(indices == teacher["indices"][:, None], axis=2)
            valid = np.any(np.all(indices[:, :, None] == bank[None, None], axis=3),
                           axis=2)
            success = risk < plan["success_mse"]
            same = exact[known]
            proposals[path.parent.name] = {
                "library_fraction": float(valid[known].mean()),
                "target_fraction": float(same.mean()),
                "tasks_receiving_target": int(same.any(1).sum()),
                "successful_exact_attempts": int((success & exact)[known].sum()),
                "exact_attempts": int(same.sum()),
                "successful_other_attempts": int((success & ~exact)[known].sum()),
                "total_known_attempts": int(success[known].size),
            }
    result = {"metrics": rows, "curves": curves, "proposals": proposals}
    (root / "summary.json").write_text(json.dumps(result, indent=2))
    with (root / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def figures(reports, geometry):
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    methods = ["ar_0", "ar_1", "ar_2", "retrieval", "dense_sgd",
               "dense_sgdm", "dense_adam", "long_adam", "source_transfer_adam",
               "frequency", "random", "oracle"]
    labels = ["AR seed 1", "AR seed 2", "AR seed 3", "Whole-I retrieval",
              "Dense SGD", "Dense SGDM", "Dense Adam", "Long dense Adam",
              "Source init + Adam", "Independent labels", "Random I", "Oracle I"]
    fig, axes = plt.subplots(1, len(reports), figsize=(15, 6), sharey=True)
    for axis, (frequency, report) in zip(axes, sorted(reports.items())):
        by_method = {row["method"]: row for row in report["metrics"]}
        values = [100 * by_method[method]["known_success"] /
                  by_method[method]["known_tasks"] for method in methods]
        axis.barh(np.arange(len(methods)), values,
                  color=["#277f8e"] * 4 + ["#d77741"] * 5 + ["#9aa7ad"] * 3)
        axis.set_yticks(np.arange(len(methods)), labels)
        axis.set_xlim(0, 110)
        axis.set_xlabel("Known-family tasks solved (%)")
        axis.set_title(f"High frequency near {frequency}")
        axis.grid(axis="x", alpha=0.15)
        for index, value in enumerate(values):
            axis.text(value + 1, index, f"{value:.1f}", va="center", fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle("Fresh codebooks; 32 new functions per regime; equal update budget")
    fig.tight_layout()
    fig.savefig(FIGURES / "comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(reports), figsize=(15, 4.5), sharey=True)
    styles = {"ar_0": ("AR seed 1", "#1a8790"),
              "retrieval": ("Retrieval", "#325aab"),
              "dense_sgd": ("Dense SGD", "#888888"),
              "dense_sgdm": ("Dense SGDM", "#cfa027"),
              "dense_adam": ("Dense Adam", "#d45c34"),
              "long_adam": ("Long Adam", "#854da6")}
    for axis, (frequency, report) in zip(axes, sorted(reports.items())):
        for method, (label, color) in styles.items():
            curve = report["curves"][method]
            axis.plot([item["updates"] for item in curve],
                      [item["known_success"] for item in curve],
                      marker="o", label=label, color=color)
        axis.set_xscale("log", base=2)
        axis.set_ylim(-0.03, 1.03)
        axis.set_xlabel("Model updates per search")
        axis.set_title(f"High frequency near {frequency}")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Fraction of known-family functions solved")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "budget.png", dpi=180)
    plt.close(fig)
    toy = geometry["toy"]
    dimensions = np.arange(2, 33)
    probabilities = 0.5 * betainc((dimensions - 1) / 2, 0.5,
                                 toy["sin_squared_cap_angle"])
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.semilogy(dimensions, probabilities, color="#277f8e")
    for dimension in (3, 24):
        value = toy["dimensions"][str(dimension)]["continuous_flow_probability"]
        axis.scatter([dimension], [value], color="#d77741", zorder=3)
        axis.annotate(f"d={dimension}: {value:.3g}", (dimension, value),
                      xytext=(8, 10), textcoords="offset points")
    axis.set_xlabel("Dimension of the optimized subspace")
    axis.set_ylabel("Exact probability of reaching the global minimum")
    axis.set_title("Artificial two-well loss; same initial norm R=2.5")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "basin_probability.png", dpi=180)
    plt.close(fig)


def main():
    plan = json.loads((ROOT / "protocol.json").read_text())
    extension = json.loads((ROOT / "frequency_sweep.json").read_text())
    source = Path(plan["output"])
    bank = np.unique(np.load(source / "structure_bank.npz")["train"], axis=0)
    reports = {4: collect(source, 4, bank)}
    for frequency in extension["frequencies"]:
        root = Path(extension["output"]) / f"frequency_{int(frequency)}"
        reports[int(frequency)] = collect(root, frequency, bank)
    geometry = json.loads((source / "geometry_audit.json").read_text())
    figures(reports, geometry)
    combined = ROOT / "results/frequency_comparison.json"
    combined.write_text(json.dumps(reports, indent=2))
    print(json.dumps({frequency: report["metrics"]
                      for frequency, report in reports.items()}, indent=2))


if __name__ == "__main__":
    main()
