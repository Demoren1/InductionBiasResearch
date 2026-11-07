"""Report all task-cue conditions and the supplementary strong controls."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.basin_routing_v1.report import FIGURES, paired_interval
from experiments.basin_routing_v1.run import ROOT


def main():
    protocol = json.loads((ROOT / "cue_protocol.json").read_text())
    control = json.loads((ROOT / "cue_control_protocol.json").read_text())
    reports, rows = {}, []
    for frequency in protocol["frequencies"]:
        for cue in protocol["cue_modes"]:
            name = f"frequency_{frequency}_{cue}"
            run = Path(protocol["output"]) / name
            competitor = Path(control["output"]) / name
            for directory in (run, competitor):
                complete = json.loads(
                    (directory / "complete.json").read_text()
                )
                if not complete["frozen_inputs_unchanged"]:
                    raise RuntimeError(f"Changed inputs: {directory}")
            teacher = np.load(run / "teacher_diagnostic.npz")
            known = teacher["families"] < 4
            feature = np.load(run / "feature_diagnostic.npz")[
                "empirical_moment"
            ]
            direction = teacher["direction"].astype(float)
            squared_norm = (direction**2).sum(1)
            high, low, alpha = (
                teacher[key].astype(float) for key in ("high", "low", "alpha")
            )
            coefficient = alpha * low * np.exp(
                -(low**2) * squared_norm / 2
            ) + high * np.exp(-(high**2) * squared_norm / 2)
            population_feature = coefficient[:, None] * direction
            cosine = np.abs((feature * direction).sum(1)) / (
                np.linalg.norm(feature, axis=1) * np.sqrt(squared_norm)
            )
            methods, success = {}, {}
            for method in protocol["methods"] + control["methods"]:
                directory = competitor if method in control["methods"] else run
                selected = np.load(directory / method / "selected.npz")
                timing = json.loads(
                    (directory / method / "timing.json").read_text()
                )
                good = selected["population"] < 0.1
                success[method] = good[known].astype(float)
                metrics = {
                    "known_solved": int(good[known].sum()),
                    "foreign_solved": int(good[~known].sum()),
                    "known_mean_mse": float(
                        selected["population"][known].mean()
                    ),
                    "foreign_mean_mse": float(
                        selected["population"][~known].mean()
                    ),
                    "numeric_batch_seconds": timing["seconds"]
                    + timing.get("screening_seconds", 0),
                    "model_updates_per_task": timing["model_updates_per_task"],
                }
                if "route_exact" in selected.files:
                    metrics["known_correct_route"] = int(
                        selected["route_exact"][known].sum()
                    )
                    metrics["foreign_correct_route"] = int(
                        selected["route_exact"][~known].sum()
                    )
                if "phase_two_exact_attempts" in selected.files:
                    metrics["known_phase_two_contains_true_I"] = int(
                        (selected["phase_two_exact_attempts"][known] > 0).sum()
                    )
                if "selected_initial_radius" in selected.files:
                    radii = selected["selected_initial_radius"][known]
                    metrics["selected_radius_counts"] = {
                        str(radius): int((radii == radius).sum())
                        for radius in np.unique(radii)
                    }
                methods[method] = metrics
                for task in range(len(known)):
                    rows.append(
                        {
                            "frequency": frequency,
                            "cue": cue,
                            "method": method,
                            "task": task,
                            "family": int(teacher["families"][task]),
                            "known": bool(known[task]),
                            "solved": bool(good[task]),
                            "population_mse": float(
                                selected["population"][task]
                            ),
                        }
                    )
            difference = np.mean(
                [
                    success[f"moment_ar_{seed}"]
                    - success[f"canonical_ar_{seed}"]
                    for seed in range(3)
                ],
                axis=0,
            )
            reports[name] = {
                "metrics": methods,
                "moment_minus_canonical_ar": paired_interval(
                    difference, teacher["families"][known]
                ),
                "feature_diagnostic": {
                    "median_alignment_to_true_direction": float(
                        np.median(cosine[known])
                    ),
                    "median_true_feature_norm": float(
                        np.median(
                            np.linalg.norm(population_feature[known], axis=1)
                        )
                    ),
                    "median_empirical_feature_norm": float(
                        np.median(np.linalg.norm(feature[known], axis=1))
                    ),
                    "median_feature_error": float(
                        np.median(
                            np.linalg.norm(
                                feature[known] - population_feature[known],
                                axis=1,
                            )
                        )
                    ),
                },
            }
    output = Path(protocol["output"])
    (output / "summary.json").write_text(json.dumps(reports, indent=2))
    with (output / "tasks.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    FIGURES.mkdir(parents=True, exist_ok=True)
    names = [
        "canonical_ar_0",
        "routed_ar_0",
        "top_two_ar_0",
        "hedged_ar_0",
        "moment_ar_0",
        "moment_retrieval",
        "moment_kmeans",
        "moment_balanced_groups",
        "moment_dense_adam",
        "moment_dense_sgdm",
        "oracle",
        "dense_adam",
        "source_transfer_adam",
    ]
    labels = [
        "Unconditional AR",
        "Hard early route",
        "Two early structures",
        "Half routed / half prior",
        "Moment chooses AR structure",
        "Moment chooses bank structure",
        "Moment k-means groups",
        "Moment balanced groups",
        "Moment init + dense Adam",
        "Moment init + dense SGDM",
        "Oracle structure",
        "Random init + dense Adam",
        "Source init + dense Adam",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True, sharey=True)
    for row, cue in enumerate(("present", "absent")):
        for column, frequency in enumerate((6, 8)):
            axis = axes[row, column]
            metrics = reports[f"frequency_{frequency}_{cue}"]["metrics"]
            values = [metrics[name]["known_solved"] for name in names]
            colors = ["#277f8e"] * 6 + ["#d77741"] * 4 + ["#888888"] * 3
            axis.barh(np.arange(len(names)), values, color=colors)
            axis.set_yticks(np.arange(len(names)), labels)
            axis.set_xlim(0, 141)
            axis.set_title(f"Frequency {frequency}; low-frequency cue {cue}")
            axis.set_xlabel("Known-family functions solved (of 128)")
            for y, value in enumerate(values):
                axis.text(value + 1, y, str(value), va="center", fontsize=8)
    axes[0, 0].invert_yaxis()
    fig.suptitle(
        "Same new tasks and update budgets; orange = no learned structural library"
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "task_cue_and_controls.png", dpi=160)
    plt.close(fig)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
