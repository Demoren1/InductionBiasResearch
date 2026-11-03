"""Summarize frozen routing experiments and paired task differences."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.basin_structure_transfer_v1.core import canonicalize

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT.parents[1] / "mds/assets/2026-11-02/basin_routing"


def paired_interval(difference, families):
    rng = np.random.default_rng(261103470)
    draws = np.concatenate(
        [
            rng.choice(np.flatnonzero(families == family), (5000, 32))
            for family in range(4)
        ],
        axis=1,
    )
    averages = difference[draws].mean(1)
    return {
        "mean_percentage_points": float(100 * difference.mean()),
        "bootstrap_95_percentage_points": (
            100 * np.quantile(averages, [0.025, 0.975])
        ).tolist(),
        "unit": "task, stratified by fixed family; prior seeds averaged within task",
    }


def main():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    root = Path(protocol["output"])
    rows, reports = [], {}
    for frequency in protocol["frequencies"]:
        run = root / f"frequency_{frequency}"
        complete = json.loads((run / "complete.json").read_text())
        if not complete["frozen_inputs_unchanged"]:
            raise RuntimeError("Experiment inputs changed during evaluation")
        teacher = np.load(run / "teacher_diagnostic.npz")
        known = teacher["families"] < 4
        success, metrics = {}, {}
        for method in protocol["methods"]:
            saved = np.load(run / method / "selected.npz")
            timing = json.loads((run / method / "timing.json").read_text())
            good = saved["population"] < protocol["success_mse"]
            success[method] = good[known].astype(float)
            metrics[method] = {
                "known_solved": int(good[known].sum()),
                "known_tasks": int(known.sum()),
                "foreign_solved": int(good[~known].sum()),
                "foreign_tasks": int((~known).sum()),
                "known_mean_mse": float(saved["population"][known].mean()),
                "foreign_mean_mse": float(saved["population"][~known].mean()),
                "numeric_batch_seconds": timing["seconds"]
                + timing.get("screening_seconds", 0),
                "query_candidate_checkpoints": timing[
                    "query_candidate_checkpoints"
                ],
                "model_updates_per_task": timing["model_updates_per_task"],
            }
            if "route_exact" in saved.files:
                metrics[method]["known_correct_routes"] = int(
                    saved["route_exact"][known].sum()
                )
                metrics[method]["known_solved_with_wrong_route"] = int(
                    (good & ~saved["route_exact"])[known].sum()
                )
            if method.startswith(("raw_ar_", "canonical_ar_")):
                proposals = canonicalize(
                    np.load(run / method / "models.npz")["indices"]
                )
                library = np.unique(teacher["indices"][known], axis=0)
                valid = (
                    (proposals[:, :, None] == library[None, None])
                    .all(3)
                    .any(2)
                )
                exact = (proposals == teacher["indices"][:, None]).all(2)
                metrics[method]["library_proposal_fraction"] = float(
                    valid[known].mean()
                )
                metrics[method]["correct_proposal_fraction"] = float(
                    exact[known].mean()
                )
            for task in range(len(known)):
                rows.append(
                    {
                        "frequency": frequency,
                        "method": method,
                        "task": task,
                        "family": int(teacher["families"][task]),
                        "known": bool(known[task]),
                        "solved": bool(good[task]),
                        "population_mse": float(saved["population"][task]),
                    }
                )
        contrasts = {}
        for left, right in (
            ("routed_ar", "canonical_ar"),
            ("canonical_ar", "raw_ar"),
        ):
            difference = np.mean(
                [
                    success[f"{left}_{seed}"] - success[f"{right}_{seed}"]
                    for seed in protocol["prior_seeds"]
                ],
                axis=0,
            )
            contrasts[f"{left}_minus_{right}"] = paired_interval(
                difference, teacher["families"][known]
            )
        reports[frequency] = {"metrics": metrics, "contrasts": contrasts}
    (root / "summary.json").write_text(json.dumps(reports, indent=2))
    with (root / "tasks.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    FIGURES.mkdir(parents=True, exist_ok=True)
    methods = [
        "raw_ar_0",
        "canonical_ar_0",
        "routed_ar_0",
        "resampled_ar_0",
        "random_route_ar_0",
        "retrieval",
        "balanced_retrieval",
        "routed_retrieval",
        "oracle",
        "dense_sgdm",
        "dense_adam",
        "source_transfer_adam",
    ]
    labels = [
        "Raw AR",
        "Canonical AR",
        "Route by task data",
        "Resample after screen",
        "Random route",
        "Retrieval",
        "Balanced retrieval",
        "Routed retrieval",
        "Oracle structure",
        "Dense SGDM",
        "Dense Adam",
        "Source weights + Adam",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    for axis, (frequency, report) in zip(axes, reports.items()):
        values = [
            report["metrics"][method]["known_solved"] for method in methods
        ]
        axis.barh(np.arange(len(methods)), values, color="#267788")
        axis.set_yticks(np.arange(len(methods)), labels)
        axis.set_xlim(0, 140)
        axis.set_title(f"Frequency {frequency}")
        axis.set_xlabel("New known-family functions solved (of 128)")
        for y, value in enumerate(values):
            axis.text(value + 1, y, str(value), va="center", fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Fresh codebooks, frozen settings, equal 65,536-update budgets"
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "routing_comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
