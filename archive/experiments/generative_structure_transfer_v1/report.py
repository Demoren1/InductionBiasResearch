"""Summarize the completed structure-only transfer pilot without training."""

import csv
import gzip
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.adaptive_tanh_v1.source import align_indices

ROOT = Path(__file__).resolve().parent
LABELS = {
    "ar": "AR structures",
    "frequency": "Independent frequencies",
    "retrieval": "Whole-structure retrieval",
    "random": "Random balanced structure",
    "oracle": "Oracle structure",
    "dense_sgd": "Dense SGDM",
    "dense_adam": "Dense Adam",
    "dense_sgd_long": "Dense SGDM, 4x steps",
    "dense_adam_long": "Dense Adam, 4x steps",
    "source_transfer": "Source weight transfer",
    "portfolio": "AR + dense selection",
}


def main():
    plan = json.loads((ROOT / "protocol.json").read_text())
    output = Path(plan["output"])
    assets = Path("mds/assets/2026-11-02/generative_structure_transfer")
    assets.mkdir(parents=True, exist_ok=True)
    summaries, audits, curves, rows = {}, {}, {}, []
    alignment_cache = {}
    budgets = [1, 2, 4, 8, 16]
    for support in plan["support_sizes"]:
        folder = output / f"evaluation_{support}"
        complete = json.loads((folder / "complete.json").read_text())
        if not complete["frozen_inputs_unchanged"]:
            raise RuntimeError("An evaluation reports changed frozen inputs")
        selected = json.loads((folder / "selected_summary.json").read_text())
        summaries[support] = selected
        families = np.array(selected["families"])
        known = families < plan["families"]
        truth = np.load(folder / "teacher.npz")["indexators"]
        family_truth = [truth[np.flatnonzero(families == family)[0]]
                        for family in range(plan["families"] + 1)]
        audits[support], curves[support] = {}, {}
        for method, item in selected["methods"].items():
            for condition, mask in [("known", known), ("foreign", ~known)]:
                errors = np.array(item["test_nmse"])[mask]
                rows.append({
                    "support": support, "method": method,
                    "condition": condition, "tasks": int(mask.sum()),
                    "mean_nmse": float(errors.mean()),
                    "median_nmse": float(np.median(errors)),
                    "successes": int((errors < plan["success_nmse"]).sum()),
                    "fit_seconds": item["seconds"],
                    "sampling_seconds": item["sampling_seconds"],
                })
            if method == "portfolio":
                continue
            models = np.load(folder / method / "models.npz")
            metrics = np.load(folder / method / "test_metrics.npz")
            query = models["query_nmse"][:, 0]
            test = metrics["test_nmse"]
            curve = []
            for budget in budgets:
                choice = query[:, :budget].argmin(1)
                error = test[np.arange(len(query)), choice]
                candidate = (test[:, :budget] < plan["success_nmse"]).any(1)
                curve.append({
                    "attempts": budget,
                    "selected_success": float(
                        (error[known] < plan["success_nmse"]).mean()
                    ),
                    "candidate_success": float(candidate[known].mean()),
                    "mean_nmse": float(error[known].mean()),
                })
            curves[support][method] = curve
            if method not in {"ar", "frequency", "retrieval", "random"}:
                continue
            indices = models["indices"].reshape(
                -1, plan["inputs"], plan["hidden"]
            )
            matched = []
            for matrix in indices:
                key = matrix.tobytes()
                if key not in alignment_cache:
                    alignment_cache[key] = [
                        bool(np.array_equal(
                            align_indices(matrix, target, plan["groups"])[0],
                            target,
                        ))
                        for target in family_truth
                    ]
                matched.append(alignment_cache[key])
            matched = np.array(matched).reshape(
                len(families), plan["starts"], plan["families"] + 1
            )
            correct = np.take_along_axis(
                matched, families[:, None, None], axis=2
            )[:, :, 0]
            valid = matched[:, :, :plan["families"]].any(2)
            numeric_success = test < plan["success_nmse"]
            matched_count = int(correct[known].sum())
            conditional_success = (
                float(numeric_success[known][correct[known]].mean())
                if matched_count else None
            )
            audit = {
                "valid_library_structure_fraction": float(valid[known].mean()),
                "correct_target_structure_fraction": float(
                    correct[known].mean()
                ),
                "known_tasks_with_correct_structure": int(
                    correct[known].any(1).sum()
                ),
                "exact_structure_attempts": matched_count,
                "numeric_success_given_exact_structure": conditional_success,
                "functional_success_without_exact_structure": int(
                    (numeric_success[known] & ~correct[known]).sum()
                ),
                "sampled_family_counts": matched[known].sum((0, 1)).tolist(),
                "known_task_exact_counts": correct[known].sum(1).tolist(),
            }
            audits[support][method] = audit
            np.savez_compressed(
                folder / method / "structure_audit.npz",
                matches_family=matched,
                correct_for_task=correct,
            )
        reference = selected["methods"]["portfolio"]["dense_reference"]
        delta = np.array(selected["methods"]["portfolio"]["test_nmse"])
        delta -= np.array(selected["methods"][reference]["test_nmse"])
        audits[support]["portfolio"] = {
            "dense_reference": reference,
            "maximum_test_nmse_increase": float(delta.max()),
            "tasks_worse_by_more_than_0.001": int((delta > 0.001).sum()),
            "selected_ar_known": int(np.array(
                selected["methods"]["portfolio"]["choose_ar"]
            )[known].sum()),
            "selected_ar_foreign": int(np.array(
                selected["methods"]["portfolio"]["choose_ar"]
            )[~known].sum()),
        }

    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "diagnostics.json").write_text(json.dumps(audits, indent=2))
    (output / "budget_curves.json").write_text(json.dumps(curves, indent=2))

    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
    methods = list(LABELS)
    for column, support in enumerate(plan["support_sizes"]):
        for row, condition in enumerate(["known", "foreign"]):
            ax = axes[row, column]
            values = [next(item["mean_nmse"] for item in rows
                           if item["support"] == support
                           and item["method"] == method
                           and item["condition"] == condition)
                      for method in methods]
            colors = ["#2378b5" if method == "ar" else "#92a4b2"
                      for method in methods]
            ax.barh(np.arange(len(methods)), np.maximum(values, 1e-14),
                    color=colors)
            ax.set_yticks(
                np.arange(len(methods)), [LABELS[x] for x in methods]
            )
            ax.invert_yaxis()
            ax.set_xscale("log")
            ax.set_xlim(1e-14, 2)
            ax.axvline(plan["success_nmse"], color="black", linestyle=":")
            ax.set_title(f"{support} training examples; {condition} structure")
            ax.set_xlabel("Mean test NMSE (lower is better)")
            for i, value in enumerate(values):
                ax.text(max(value, 1e-14) * 1.2, i, f"{value:.2g}",
                        va="center", fontsize=8)
    fig.savefig(assets / "comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, support in zip(axes, plan["support_sizes"]):
        for method in [
            "ar", "retrieval", "frequency", "dense_sgd", "dense_adam"
        ]:
            ax.plot(budgets, [row["selected_success"]
                             for row in curves[support][method]],
                    marker="o", label=LABELS[method])
        ax.plot(budgets, 1 - (1 - 1 / plan["families"]) ** np.array(budgets),
                "k--", label="Ideal uniform structure coverage")
        ax.set_title(f"{support} training examples; 16 known-family tasks")
        ax.set_xlabel("Numeric fits with independent initializations")
        ax.set_ylabel("Fraction of selected solutions with NMSE < 0.001")
        ax.set_ylim(-0.03, 1.04)
        ax.set_xticks(budgets)
        ax.grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.savefig(assets / "budget.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for column, support in enumerate(plan["support_sizes"]):
        known = np.array(summaries[support]["families"]) < plan["families"]
        for method in [
            "ar", "retrieval", "dense_sgd", "dense_adam", "dense_sgd_long"
        ]:
            path = (
                output / f"evaluation_{support}" / method / "history.json.gz"
            )
            with gzip.open(path, "rt") as handle:
                history = json.load(handle)
            best_query = np.full((len(known), plan["starts"]), np.inf)
            best_train = best_query.copy()
            train_curve, query_curve = [], []
            for record in history:
                query = np.array(record["query_nmse"]).reshape(
                    best_query.shape
                )
                train = np.array(record["train_nmse"]).reshape(
                    best_query.shape
                )
                better = query < best_query
                best_query[better] = query[better]
                best_train[better] = train[better]
                choice = best_query.argmin(1)
                row_ids = np.arange(len(known))
                query_curve.append(best_query[row_ids, choice][known].mean())
                train_curve.append(best_train[row_ids, choice][known].mean())
            steps = [record["step"] for record in history]
            axes[0, column].plot(steps, query_curve, label=LABELS[method])
            axes[1, column].plot(steps, train_curve, label=LABELS[method])
        for row, name in enumerate(["query", "train of query-selected model"]):
            ax = axes[row, column]
            ax.set_yscale("log")
            ax.set_xlabel("Gradient updates per attempt")
            ax.set_ylabel("Mean NMSE")
            ax.set_title(f"{support} examples: {name}")
            ax.grid(alpha=0.2)
    axes[0, 1].legend(fontsize=8)
    fig.savefig(assets / "learning.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"rows": rows, "diagnostics": audits}, indent=2))


if __name__ == "__main__":
    main()
