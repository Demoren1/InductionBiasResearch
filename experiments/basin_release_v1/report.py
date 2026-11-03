"""Report paired held-out structural and released-weight comparisons."""

import csv
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.basin_structure_transfer_v1.core import canonicalize

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT.parents[1] / "mds/assets/2026-11-03/basin_release"


def sharing_residual(weights, indices, groups):
    """Relative distance from the parameter subspace of the initial I."""
    projection = np.zeros_like(weights)
    for group in range(groups):
        mask = indices == group
        mean = (weights * mask).sum(-1) / mask.sum(-1).clip(min=1)
        projection += mean[..., None] * mask
    return np.linalg.norm(weights - projection, axis=-1) / np.maximum(
        np.linalg.norm(weights, axis=-1), 1e-12
    )


def paired_interval(difference, families, protocol):
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    draws = np.concatenate(
        [
            rng.choice(
                np.flatnonzero(families == family),
                (protocol["bootstrap_draws"], np.sum(families == family)),
            )
            for family in np.unique(families)
        ],
        axis=1,
    )
    return {
        "mean_percentage_points": float(100 * difference.mean()),
        "bootstrap_95_percentage_points": (
            100 * np.quantile(difference[draws].mean(1), [0.025, 0.975])
        ).tolist(),
        "unit": "task, stratified by fixed family; prior seeds averaged",
    }


def main():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    output = Path(protocol["output"])
    base = json.loads(
        (Path(protocol["source_run"]) / "protocol.json").read_text()
    )
    reports, rows, curves = {}, [], {}
    for frequency in protocol["frequencies"]:
        directory = output / f"frequency_{frequency}"
        complete = json.loads((directory / "complete.json").read_text())
        if not complete["frozen_inputs_unchanged"]:
            raise RuntimeError("Frozen run inputs changed")
        teacher = np.load(directory / "teacher_diagnostic.npz")
        families = teacher["families"]
        known = families < base["families"]
        settings = json.loads(
            (directory / "frozen_selection.json").read_text()
        )
        success, metrics, curves[frequency] = {}, {}, {}
        for method in complete["methods"]:
            saved = np.load(directory / method / "selected.npz")
            models = np.load(directory / method / "models.npz")
            timing = json.loads(
                (directory / method / "timing.json").read_text()
            )
            updates = timing["steps"] * timing["starts"]
            if updates != protocol["steps"] * protocol["starts"]:
                raise RuntimeError("Target update budgets differ")
            if timing["source_numeric_initialization"] or timing[
                "task_direction_initialization"
            ]:
                raise RuntimeError("Unexpected numerical transfer")
            good = saved["population"] < protocol["success_mse"]
            success[method] = good[known].astype(float)
            metric = {
                "known_solved": int(good[known].sum()),
                "known_total": int(known.sum()),
                "foreign_solved": int(good[~known].sum()),
                "foreign_total": int((~known).sum()),
                "known_mean_population_mse": float(
                    saved["population"][known].mean()
                ),
                "foreign_mean_population_mse": float(
                    saved["population"][~known].mean()
                ),
                "numeric_seconds": timing["seconds"],
                "model_updates_per_task": updates,
                "numeric_config": timing["configs"][0],
            }
            if "indices" in models.files:
                indices = models["indices"]
                exact = (
                    canonicalize(indices) == teacher["indices"][:, None]
                ).all(2)
                attempt_good = models["population"][:, 0] < protocol[
                    "success_mse"
                ]
                metric["known_correct_proposals"] = int(exact[known].sum())
                metric["known_successful_correct_attempts"] = int(
                    (exact & attempt_good)[known].sum()
                )
                metric["known_tasks_without_correct_proposal"] = int(
                    (~exact[known].any(1)).sum()
                )
                residual = sharing_residual(
                    saved["weights"], saved["indices"], base["groups"]
                )
                metric["known_median_selected_sharing_residual"] = float(
                    np.median(residual[known])
                )
                final_residual = sharing_residual(
                    models["final_weights"][:, 0], indices, base["groups"]
                )
                metric["median_final_residual_correct_known_attempts"] = (
                    float(np.median(final_residual[known][exact[known]]))
                    if exact[known].any() else None
                )
            metrics[method] = metric
            with gzip.open(directory / method / "history.json.gz", "rt") as f:
                history = json.load(f)
            series = []
            for item in history:
                query = np.asarray(item["query"])
                risk = np.asarray(item["population"])
                choice = query.argmin(1)
                selected_risk = risk[np.arange(len(query)), choice]
                series.append({
                    "updates": item["step"] * protocol["starts"],
                    "known_solved": int((
                        selected_risk[known] < protocol["success_mse"]
                    ).sum()),
                })
            curves[frequency][method] = series
            for task in range(len(known)):
                rows.append({
                    "frequency": frequency,
                    "method": method,
                    "task": task,
                    "family": int(families[task]),
                    "known": bool(known[task]),
                    "solved": bool(good[task]),
                    "population_mse": float(saved["population"][task]),
                })
        contrasts = {}
        for competitor in ("released_matched", "released_tuned"):
            difference = np.mean(
                [
                    success[f"tied_ar_{seed}"]
                    - success[f"{competitor}_ar_{seed}"]
                    for seed in protocol["prior_seeds"]
                ],
                axis=0,
            )
            contrasts[f"tied_minus_{competitor}"] = paired_interval(
                difference, families[known], protocol
            )
        calibration_seconds = sum(
            json.loads(path.read_text())["seconds"]
            for path in (directory / "calibration").glob("*/timing.json")
        )
        reports[frequency] = {
            "metrics": metrics,
            "contrasts": contrasts,
            "settings": settings,
            "matching": complete["matching"],
            "calibration_numeric_seconds": calibration_seconds,
        }
    (output / "summary.json").write_text(json.dumps(reports, indent=2))
    (output / "curves.json").write_text(json.dumps(curves, indent=2))
    with (output / "tasks.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex="row")
    modes = ("tied", "released_matched", "released_tuned")
    labels = (
        "Fixed learned I", "Released, same start", "Released, tuned norm"
    )
    colors = ("#19758b", "#d58831", "#765ca8")
    for column, frequency in enumerate(protocol["frequencies"]):
        metrics = reports[frequency]["metrics"]
        values = [
            [metrics[f"{mode}_ar_{seed}"]["known_solved"]
             for seed in protocol["prior_seeds"]]
            for mode in modes
        ]
        means = np.mean(values, axis=1)
        errors = np.array([
            means - np.min(values, axis=1),
            np.max(values, axis=1) - means,
        ])
        axes[0, column].barh(
            np.arange(3), means, xerr=errors, color=colors, capsize=4
        )
        axes[0, column].set_yticks(np.arange(3), labels)
        axes[0, column].invert_yaxis()
        axes[0, column].set_xlim(0, 132)
        axes[0, column].set_title(f"Frequency {frequency}")
        axes[0, column].set_xlabel("Solved known tasks (of 128)")
        for y, value in enumerate(means):
            axes[0, column].text(
                max(values[y]) + 2, y, f"{value:.1f}", va="center"
            )
        for mode, label, color in zip(modes, labels, colors):
            records = [curves[frequency][f"{mode}_ar_{seed}"]
                       for seed in protocol["prior_seeds"]]
            updates = [point["updates"] for point in records[0]]
            solved = np.mean(
                [[point["known_solved"] for point in record]
                 for record in records],
                axis=0,
            )
            axes[1, column].plot(updates, solved, label=label, color=color)
        axes[1, column].set_xlabel("Model updates per task")
        axes[1, column].set_ylabel("Solved known tasks (mean over 3 priors)")
        axes[1, column].set_ylim(0, 132)
        axes[1, column].grid(alpha=0.2)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle(
        "Fresh codebooks: retain sharing or release the same start\n"
        "Bars: mean; whiskers: range across three frozen priors"
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "release_comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps({
        frequency: {
            "contrasts": report["contrasts"],
            "solved": {
                method: [metric["known_solved"], metric["foreign_solved"]]
                for method, metric in report["metrics"].items()
            },
        }
        for frequency, report in reports.items()
    }, indent=2))


if __name__ == "__main__":
    main()
