"""Aggregate leakage-safe task-quality records and write the final report."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .evaluate import (
    EXPECTED_OUTER_SEEDS,
    EXPECTED_TEST_TASK_IDS,
    METHODS,
    paired_seed_interval,
    validate_artifact_root,
    validate_eval_record,
)
from .structure import signed_weight_audit, summarize_structure


METHOD_LABELS = {
    "transformer_mask": "Transformer mask",
    "free_mask": "Free mask",
    "functional_centroid_mean": "Functional centroid mean",
    "random_exact32": "Random exact-32",
    "dense": "Dense",
    "oracle_mask": "Oracle mask",
}
METHOD_COLORS = {
    "transformer_mask": "#0072B2",
    "free_mask": "#D55E00",
    "functional_centroid_mean": "#009E73",
    "random_exact32": "#CC79A7",
    "dense": "#555555",
    "oracle_mask": "#E69F00",
}
EXPECTED_BUDGETS = (32, 128)
EXPECTED_INIT_IDS = (0, 1, 2, 3)
EXAMPLE_SEED = 8100
EXAMPLE_TASK_INDEX = 0
EXAMPLE_BUDGET = 128
EXAMPLE_INIT_ID = 0


def _records_from_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                validate_eval_record(record)
            except Exception as error:
                raise ValueError(f"invalid record at {path}:{line_number}: {error}") from error
            records.append(record)
    if not records:
        raise ValueError(f"no records found in {path}")
    return records


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _record_key(record: Mapping[str, Any]) -> tuple[str, int, str, int, int]:
    return (str(record["method"]), int(record["seed"]), str(record["task_id"]),
            int(record["budget"]), int(record["init_id"]))


def validate_complete_design(records: Iterable[Mapping[str, Any]]) -> dict[str, list[Any]]:
    """Require the planned 6-method × 4-seed × 4-task × 2-budget × 4-init grid."""
    rows = list(records)
    methods = sorted({str(row["method"]) for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    task_ids_by_index: dict[int, str] = {}
    for row in rows:
        index, task_id = int(row["task_index"]), str(row["task_id"])
        if index in task_ids_by_index and task_ids_by_index[index] != task_id:
            raise ValueError(f"task_index {index} maps to multiple task IDs")
        task_ids_by_index[index] = task_id
    task_indices = sorted(task_ids_by_index)
    budgets = sorted({int(row["budget"]) for row in rows})
    init_ids = sorted({int(row["init_id"]) for row in rows})
    if methods != sorted(METHODS):
        raise ValueError(f"expected the six methods {sorted(METHODS)}, got {methods}")
    if tuple(seeds) != EXPECTED_OUTER_SEEDS:
        raise ValueError(f"expected outer seeds {EXPECTED_OUTER_SEEDS}, got {seeds}")
    if task_indices != [0, 1, 2, 3]:
        raise ValueError(f"expected held-out task indices 0..3, got {task_indices}")
    if tuple(task_ids_by_index[index] for index in task_indices) != EXPECTED_TEST_TASK_IDS:
        raise ValueError(f"held-out task order must be {EXPECTED_TEST_TASK_IDS}")
    if tuple(budgets) != EXPECTED_BUDGETS:
        raise ValueError(f"expected budgets {EXPECTED_BUDGETS}, got {budgets}")
    if tuple(init_ids) != EXPECTED_INIT_IDS:
        raise ValueError(f"expected child initialization IDs {EXPECTED_INIT_IDS}, got {init_ids}")
    expected_keys = {
        (method, seed, task_ids_by_index[index], budget, init_id)
        for method in METHODS for seed in seeds for index in task_indices
        for budget in budgets for init_id in init_ids
    }
    observed_keys = {_record_key(row) for row in rows}
    missing = sorted(expected_keys - observed_keys)
    extra = sorted(observed_keys - expected_keys)
    if missing or extra:
        raise ValueError(f"incomplete or inconsistent design: {len(missing)} missing, {len(extra)} extra keys")
    if len(rows) != len(expected_keys):
        raise ValueError("duplicate evaluation records are present")
    if EXAMPLE_SEED not in seeds:
        raise ValueError(f"seed {EXAMPLE_SEED} is required for the prespecified heatmap example")
    for method in METHODS:
        for budget in budgets:
            selections = [row["lr_selection"] for row in rows
                          if row["method"] == method and int(row["budget"]) == budget]
            signatures = {
                (float(selection["selected_lr"]),
                 tuple(float(selection.get("scores_by_lr", {}).get(str(rate),
                              selection.get("scores_by_lr", {}).get(f"{rate:g}", np.nan)))
                       for rate in (0.001, 0.003, 0.01)))
                for selection in selections
            }
            if len(signatures) != 1:
                raise ValueError(f"{method} budget {budget}: LR tuning must be frozen globally before test tasks")
    return {"methods": methods, "seeds": seeds,
            "task_ids": [task_ids_by_index[index] for index in task_indices],
            "task_indices": task_indices, "budgets": budgets, "init_ids": init_ids}


def _index_records(records: Iterable[Mapping[str, Any]]) -> dict[tuple[str, int, int, int, int], Mapping[str, Any]]:
    """Index runs by method, seed, task index, budget and initialization."""
    return {(str(row["method"]), int(row["seed"]), int(row["task_index"]),
             int(row["budget"]), int(row["init_id"])): row for row in records}


def aggregate_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute equally weighted task means and seed-paired dense differences."""
    rows = list(records)
    design = validate_complete_design(rows)
    indexed = _index_records(rows)
    aggregate: dict[str, Any] = {"design": design, "conditions": [], "overall": [],
                                "lr_selection": [], "training_status": []}

    for method in METHODS:
        for budget in design["budgets"]:
            subset = [row for row in rows
                      if row["method"] == method and int(row["budget"]) == budget]
            lr = subset[0]["lr_selection"]
            raw_scores = lr["scores_by_lr"]
            scores = {str(rate): float(raw_scores.get(str(rate), raw_scores.get(f"{rate:g}")))
                      for rate in (0.001, 0.003, 0.01)}
            aggregate["lr_selection"].append({
                "method": method, "budget": budget,
                "selected_lr": float(lr["selected_lr"]),
                "selection_split": str(lr.get("selection_split", lr.get("split"))),
                "validation_task_ids": list(lr["task_ids"]),
                "validation_query_bce_by_lr": scores,
                "selection_score": str(lr["score"]),
                "used_test": False,
            })
            converged = [bool(row["fit_status"]["converged"]) for row in subset]
            stop_reasons = defaultdict(int)
            for row in subset:
                stop_reasons[str(row["fit_status"]["stop_reason"])] += 1
            cap_count = sum(
                int(row["fit_status"]["steps"]) >= int(row["fit_status"]["max_steps"])
                or str(row["fit_status"]["stop_reason"]).lower() in {"max_steps", "step_cap", "cap"}
                for row in subset
            )
            aggregate["training_status"].append({
                "method": method, "budget": budget,
                "n_runs": len(subset),
                "converged_runs": int(sum(converged)),
                "unconverged_runs": int(len(converged) - sum(converged)),
                "step_cap_runs": int(cap_count),
                "query_selection_complete_runs": int(sum(
                    bool(row["fit_status"]["selection_complete"]) for row in subset)),
                "mean_steps": float(np.mean([row["fit_status"]["steps"] for row in subset])),
                "stop_reason_counts": dict(stop_reasons),
            })

    for budget in design["budgets"]:
        task_results: dict[str, dict[str, Any]] = {}
        overall_seed_acc_deltas: dict[str, dict[int, list[float]]] = {
            method: defaultdict(list) for method in METHODS
        }
        overall_seed_bce_improvements: dict[str, dict[int, list[float]]] = {
            method: defaultdict(list) for method in METHODS
        }
        overall_seed_balanced_accuracy_deltas: dict[str, dict[int, list[float]]] = {
            method: defaultdict(list) for method in METHODS
        }
        overall_seed_accuracy: dict[str, dict[int, list[float]]] = {
            method: defaultdict(list) for method in METHODS
        }
        overall_seed_balanced_bce: dict[str, dict[int, list[float]]] = {
            method: defaultdict(list) for method in METHODS
        }
        for task_index, task_id in enumerate(design["task_ids"]):
            task_results[task_id] = {}
            for method in METHODS:
                seed_accuracy: dict[int, list[float]] = defaultdict(list)
                seed_balanced_accuracy: dict[int, list[float]] = defaultdict(list)
                seed_balanced_bce: dict[int, list[float]] = defaultdict(list)
                seed_natural_bce: dict[int, list[float]] = defaultdict(list)
                seed_acc_delta: dict[int, list[float]] = defaultdict(list)
                seed_balanced_accuracy_delta: dict[int, list[float]] = defaultdict(list)
                seed_balanced_bce_improvement: dict[int, list[float]] = defaultdict(list)
                for seed in design["seeds"]:
                    for init_id in design["init_ids"]:
                        row = indexed[(method, seed, task_index, budget, init_id)]
                        dense = indexed[("dense", seed, task_index, budget, init_id)]
                        accuracy = float(row["test"]["accuracy"])
                        balanced_accuracy = float(row["test"]["balanced_accuracy"])
                        balanced_bce = float(row["test"]["balanced_bce"])
                        natural_bce = float(row["test"]["bce"])
                        seed_accuracy[seed].append(accuracy)
                        seed_balanced_accuracy[seed].append(balanced_accuracy)
                        seed_balanced_bce[seed].append(balanced_bce)
                        seed_natural_bce[seed].append(natural_bce)
                        seed_acc_delta[seed].append(accuracy - float(dense["test"]["accuracy"]))
                        seed_balanced_accuracy_delta[seed].append(
                            balanced_accuracy - float(dense["test"]["balanced_accuracy"]))
                        seed_balanced_bce_improvement[seed].append(
                            float(dense["test"]["balanced_bce"]) - balanced_bce)
                seed_mean_accuracy = {seed: float(np.mean(values))
                                      for seed, values in seed_accuracy.items()}
                seed_mean_balanced_accuracy = {seed: float(np.mean(values))
                                               for seed, values in seed_balanced_accuracy.items()}
                seed_mean_balanced_bce = {seed: float(np.mean(values))
                                          for seed, values in seed_balanced_bce.items()}
                seed_mean_natural_bce = {seed: float(np.mean(values))
                                         for seed, values in seed_natural_bce.items()}
                seed_mean_acc_delta = {seed: float(np.mean(values))
                                       for seed, values in seed_acc_delta.items()}
                seed_mean_balanced_accuracy_delta = {
                    seed: float(np.mean(values)) for seed, values in seed_balanced_accuracy_delta.items()}
                seed_mean_balanced_bce_improvement = {
                    seed: float(np.mean(values)) for seed, values in seed_balanced_bce_improvement.items()}
                accuracy_ci = paired_seed_interval(np.array(list(seed_mean_accuracy.values())))
                balanced_accuracy_ci = paired_seed_interval(
                    np.array(list(seed_mean_balanced_accuracy.values())))
                balanced_bce_ci = paired_seed_interval(
                    np.array(list(seed_mean_balanced_bce.values())))
                bce_improvement_ci = paired_seed_interval(
                    np.array(list(seed_mean_balanced_bce_improvement.values())))
                balanced_accuracy_delta_ci = paired_seed_interval(
                    np.array(list(seed_mean_balanced_accuracy_delta.values())))
                accuracy_delta_ci = paired_seed_interval(np.array(list(seed_mean_acc_delta.values())))
                method_result = {
                    "mean_accuracy": float(np.mean(list(seed_mean_accuracy.values()))),
                    "accuracy_ci95_over_seed_means": accuracy_ci,
                    "mean_balanced_accuracy": float(np.mean(list(seed_mean_balanced_accuracy.values()))),
                    "balanced_accuracy_ci95_over_seed_means": balanced_accuracy_ci,
                    "mean_balanced_bce": float(np.mean(list(seed_mean_balanced_bce.values()))),
                    "balanced_bce_ci95_over_seed_means": balanced_bce_ci,
                    "mean_natural_bce": float(np.mean(list(seed_mean_natural_bce.values()))),
                    "balanced_bce_improvement_vs_dense": float(
                        np.mean(list(seed_mean_balanced_bce_improvement.values()))),
                    "balanced_bce_improvement_ci95_over_seed_means": bce_improvement_ci,
                    "balanced_accuracy_delta_vs_dense": float(
                        np.mean(list(seed_mean_balanced_accuracy_delta.values()))),
                    "balanced_accuracy_delta_ci95_over_seed_means": balanced_accuracy_delta_ci,
                    "accuracy_delta_vs_dense": float(np.mean(list(seed_mean_acc_delta.values()))),
                    "accuracy_delta_ci95_over_seed_means": accuracy_delta_ci,
                    "seed_accuracy_means": {str(seed): value for seed, value in seed_mean_accuracy.items()},
                    "seed_balanced_bce_improvements": {
                        str(seed): value for seed, value in seed_mean_balanced_bce_improvement.items()},
                }
                task_results[task_id][method] = method_result
                for seed, value in seed_mean_accuracy.items():
                    overall_seed_accuracy[method][seed].append(value)
                    overall_seed_balanced_bce[method][seed].append(seed_mean_balanced_bce[seed])
                for seed, value in seed_mean_acc_delta.items():
                    overall_seed_acc_deltas[method][seed].append(value)
                for seed, value in seed_mean_balanced_bce_improvement.items():
                    overall_seed_bce_improvements[method][seed].append(value)
                for seed, value in seed_mean_balanced_accuracy_delta.items():
                    overall_seed_balanced_accuracy_deltas[method][seed].append(value)

        for method in METHODS:
            task_improvements = [task_results[task_id][method]["balanced_bce_improvement_vs_dense"]
                                 for task_id in design["task_ids"]]
            overall_seed_bce_improvements_mean = [
                float(np.mean(overall_seed_bce_improvements[method][seed]))
                for seed in design["seeds"]]
            overall_seed_accuracy_deltas = [
                float(np.mean(overall_seed_acc_deltas[method][seed]))
                for seed in design["seeds"]]
            overall_seed_balanced_accuracy_deltas_mean = [
                float(np.mean(overall_seed_balanced_accuracy_deltas[method][seed]))
                for seed in design["seeds"]]
            overall_seed_acc = [float(np.mean(overall_seed_accuracy[method][seed]))
                                for seed in design["seeds"]]
            overall_seed_balanced_bce_mean = [float(np.mean(overall_seed_balanced_bce[method][seed]))
                                              for seed in design["seeds"]]
            aggregate["overall"].append({
                "budget": budget,
                "method": method,
                "mean_accuracy_equal_task_weight": float(np.mean(overall_seed_acc)),
                "accuracy_ci95_over_seed_means": paired_seed_interval(np.asarray(overall_seed_acc)),
                "mean_balanced_bce_equal_task_weight": float(np.mean(overall_seed_balanced_bce_mean)),
                "balanced_bce_ci95_over_seed_means": paired_seed_interval(
                    np.asarray(overall_seed_balanced_bce_mean)),
                "mean_balanced_bce_improvement_vs_dense": float(
                    np.mean(overall_seed_bce_improvements_mean)),
                "balanced_bce_improvement_ci95_over_seed_means": paired_seed_interval(
                    np.asarray(overall_seed_bce_improvements_mean)),
                "mean_balanced_accuracy_delta_vs_dense": float(
                    np.mean(overall_seed_balanced_accuracy_deltas_mean)),
                "balanced_accuracy_delta_ci95_over_seed_means": paired_seed_interval(
                    np.asarray(overall_seed_balanced_accuracy_deltas_mean)),
                "mean_natural_accuracy_delta_vs_dense": float(np.mean(overall_seed_accuracy_deltas)),
                "natural_accuracy_delta_ci95_over_seed_means": paired_seed_interval(
                    np.asarray(overall_seed_accuracy_deltas)),
                "fraction_tasks_with_positive_balanced_bce_improvement": float(
                    np.mean(np.asarray(task_improvements) > 0)),
                "worst_task_balanced_bce_improvement": float(np.min(task_improvements)),
                "task_balanced_bce_improvements": {
                    task_id: task_results[task_id][method]["balanced_bce_improvement_vs_dense"]
                    for task_id in design["task_ids"]},
            })
        aggregate["conditions"].append({"budget": budget, "tasks": task_results})
    return aggregate


def write_summary_npz(records: Iterable[Mapping[str, Any]], destination: str | Path) -> Path:
    """Save per-run numeric test scores and their condition keys without pickle."""
    rows = sorted(list(records), key=lambda row: _record_key(row))
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        method=np.asarray([row["method"] for row in rows], dtype="U32"),
        seed=np.asarray([row["seed"] for row in rows], dtype=np.int64),
        task_id=np.asarray([row["task_id"] for row in rows], dtype="U16"),
        task_index=np.asarray([row["task_index"] for row in rows], dtype=np.int64),
        budget=np.asarray([row["budget"] for row in rows], dtype=np.int64),
        init_id=np.asarray([row["init_id"] for row in rows], dtype=np.int64),
        accuracy=np.asarray([row["test"]["accuracy"] for row in rows], dtype=np.float64),
        balanced_accuracy=np.asarray([row["test"]["balanced_accuracy"] for row in rows], dtype=np.float64),
        bce=np.asarray([row["test"]["bce"] for row in rows], dtype=np.float64),
        balanced_bce=np.asarray([row["test"]["balanced_bce"] for row in rows], dtype=np.float64),
        brier=np.asarray([row["test"]["brier"] for row in rows], dtype=np.float64),
        selected_lr=np.asarray([row["lr_selection"]["selected_lr"] for row in rows], dtype=np.float64),
        test_n=np.asarray([row["test"]["n"] for row in rows], dtype=np.int64),
        child_steps=np.asarray([row["fit_status"]["steps"] for row in rows], dtype=np.int64),
        child_max_steps=np.asarray([row["fit_status"]["max_steps"] for row in rows], dtype=np.int64),
        child_converged=np.asarray([row["fit_status"]["converged"] for row in rows], dtype=bool),
        child_selection_complete=np.asarray([row["fit_status"]["selection_complete"] for row in rows], dtype=bool),
        child_stop_reason=np.asarray([row["fit_status"]["stop_reason"] for row in rows], dtype="U32"),
    )
    return output


def write_structure_artifacts(records: Iterable[Mapping[str, Any]], root: Path) -> dict[str, int]:
    """Write full post-hoc support/weight audits and numeric structure arrays."""
    rows = sorted(list(records), key=lambda row: _record_key(row))
    audits = []
    for row in rows:
        audit = summarize_structure(np.asarray(row["mask"]), np.asarray(row["weight"]),
                                    readout=np.asarray(row["readout"]), bias=np.asarray(row["bias"]))
        audits.append({**{name: row[name] for name in
                          ("method", "seed", "task_id", "task_index", "budget", "init_id")},
                       "structure": audit})
    structure_path = root / "structure.jsonl"
    with structure_path.open("w", encoding="utf-8") as stream:
        for row in audits:
            stream.write(json.dumps(_jsonable(row), allow_nan=False) + "\n")

    metrics = [row["structure"] for row in audits]
    numeric_values: dict[str, list[float]] = {
        "mask_iou": [value["iou"] for value in metrics],
        "mask_precision": [value["precision"] for value in metrics],
        "mask_recall": [value["recall"] for value in metrics],
        "weight_all_offset_toeplitz_energy": [
            value["masked_effective_weight_structure"]["all_offset_toeplitz_explained_energy"]
            for value in metrics],
        "weight_oracle_band_toeplitz_energy": [
            value["masked_effective_weight_structure"]["oracle_band_0_to_k_minus_1_explained_energy"]
            for value in metrics],
        "weight_affine_normalized_all_offset_toeplitz_energy": [
            value["affine_column_normalized_structure"]["all_offset_toeplitz_explained_energy"]
            for value in metrics],
        "weight_affine_normalized_oracle_band_toeplitz_energy": [
            value["affine_column_normalized_structure"]["oracle_band_0_to_k_minus_1_explained_energy"]
            for value in metrics],
        "weight_readout_all_offset_toeplitz_energy": [
            value["readout_scaled_structure"]["all_offset_toeplitz_explained_energy"]
            for value in metrics],
        "weight_readout_oracle_band_toeplitz_energy": [
            value["readout_scaled_structure"]["oracle_band_0_to_k_minus_1_explained_energy"]
            for value in metrics],
    }
    np.savez_compressed(
        root / "structure_metrics.npz",
        method=np.asarray([row["method"] for row in audits], dtype="U32"),
        seed=np.asarray([row["seed"] for row in audits], dtype=np.int64),
        task_id=np.asarray([row["task_id"] for row in audits], dtype="U16"),
        task_index=np.asarray([row["task_index"] for row in audits], dtype=np.int64),
        budget=np.asarray([row["budget"] for row in audits], dtype=np.int64),
        init_id=np.asarray([row["init_id"] for row in audits], dtype=np.int64),
        **{name: np.asarray(values, dtype=np.float64) for name, values in numeric_values.items()},
    )
    return {"n_records": len(audits), "array_fields": len(numeric_values)}


def _record_lookup(records: Iterable[Mapping[str, Any]]) -> dict[tuple[str, int, int, int, int], Mapping[str, Any]]:
    return _index_records(records)


def write_selected_example(records: Iterable[Mapping[str, Any]], root: Path) -> bool:
    """Save the predeclared common-condition masks and signed weights."""
    lookup = _record_lookup(records)
    keys = [(method, EXAMPLE_SEED, EXAMPLE_TASK_INDEX, EXAMPLE_BUDGET, EXAMPLE_INIT_ID)
            for method in METHODS]
    missing = [key for key in keys if key not in lookup]
    if missing:
        return False
    selected = [lookup[key] for key in keys]
    masks = np.asarray([row["mask"] for row in selected], dtype=np.uint8)
    weights = np.asarray([row["weight"] for row in selected], dtype=np.float64)
    readout = np.asarray([row["readout"] for row in selected], dtype=np.float64)
    bias = np.asarray([row["bias"] for row in selected], dtype=np.float64)
    effective = weights * masks
    np.savez_compressed(
        root / "selected_example.npz",
        method=np.asarray(METHODS, dtype="U32"),
        seed=np.asarray(EXAMPLE_SEED, dtype=np.int64),
        task_id=np.asarray(selected[0]["task_id"], dtype="U16"),
        task_index=np.asarray(EXAMPLE_TASK_INDEX, dtype=np.int64),
        budget=np.asarray(EXAMPLE_BUDGET, dtype=np.int64),
        init_id=np.asarray(EXAMPLE_INIT_ID, dtype=np.int64),
        mask=masks, raw_weight=weights, effective_weight=effective,
        readout=readout, bias=bias,
    )
    return True


def _plot_quality(aggregate: Mapping[str, Any], destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = aggregate["conditions"]
    design = aggregate["design"]
    fig, axes = plt.subplots(len(conditions), len(design["task_ids"]),
                             figsize=(16, 7), sharey=True, squeeze=False)
    x = np.arange(len(METHODS))
    for row_index, condition in enumerate(conditions):
        for col_index, task_id in enumerate(design["task_ids"]):
            ax = axes[row_index, col_index]
            for method_index, method in enumerate(METHODS):
                result = condition["tasks"][task_id][method]
                ci = result["accuracy_ci95_over_seed_means"]
                center = result["mean_accuracy"]
                err = [[center - ci["ci_low"]], [ci["ci_high"] - center]]
                ax.errorbar(method_index, center, yerr=err, fmt="o", capsize=2,
                            color=METHOD_COLORS[method], markersize=5)
            ax.set_title(f"{task_id}, budget {condition['budget']}")
            ax.set_xticks(x, [METHOD_LABELS[method] for method in METHODS], rotation=55,
                          ha="right", fontsize=8)
            if col_index == 0:
                ax.set_ylabel("Held-out test accuracy")
            ax.set_ylim(0.45, 1.01)
            ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Per-task test accuracy; points average four child initializations per seed")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _plot_balanced_bce(aggregate: Mapping[str, Any], destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = aggregate["conditions"]
    design = aggregate["design"]
    fig, axes = plt.subplots(len(conditions), len(design["task_ids"]),
                             figsize=(16, 7), sharey=True, squeeze=False)
    x = np.arange(len(METHODS))
    for row_index, condition in enumerate(conditions):
        for col_index, task_id in enumerate(design["task_ids"]):
            ax = axes[row_index, col_index]
            for method_index, method in enumerate(METHODS):
                result = condition["tasks"][task_id][method]
                ci = result["balanced_bce_ci95_over_seed_means"]
                center = result["mean_balanced_bce"]
                err = [[center - ci["ci_low"]], [ci["ci_high"] - center]]
                ax.errorbar(method_index, center, yerr=err, fmt="o", capsize=2,
                            color=METHOD_COLORS[method], markersize=5)
            ax.set_title(f"{task_id}, budget {condition['budget']}")
            ax.set_xticks(x, [METHOD_LABELS[method] for method in METHODS], rotation=55,
                          ha="right", fontsize=8)
            if col_index == 0:
                ax.set_ylabel("Balanced test BCE (lower is better)")
            ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Per-task balanced test BCE; points average child inits within four seeds")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _plot_paired_deltas(aggregate: Mapping[str, Any], destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = aggregate["conditions"]
    design = aggregate["design"]
    fig, axes = plt.subplots(len(conditions), 1, figsize=(11, 7), sharex=True, squeeze=False)
    for row_index, condition in enumerate(conditions):
        ax = axes[row_index, 0]
        group_x = np.arange(len(design["task_ids"]))
        offsets = np.linspace(-0.22, 0.22, len(METHODS) - 1)
        delta_methods = [method for method in METHODS if method != "dense"]
        for method_index, method in enumerate(delta_methods):
            centers, lows, highs = [], [], []
            for task_id in design["task_ids"]:
                result = condition["tasks"][task_id][method]
                ci = result["balanced_bce_improvement_ci95_over_seed_means"]
                center = result["balanced_bce_improvement_vs_dense"]
                centers.append(center)
                lows.append(center - ci["ci_low"])
                highs.append(ci["ci_high"] - center)
            x = group_x + offsets[method_index]
            ax.errorbar(x, centers, yerr=[lows, highs], fmt="o", capsize=3,
                        color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        ax.axhline(0, color="#333333", linewidth=1, linestyle="--")
        ax.set_ylabel(f"Dense BCE − method BCE\nbudget {condition['budget']}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncol=3, fontsize=8, loc="best")
    axes[-1, 0].set_xticks(np.arange(len(design["task_ids"])), design["task_ids"])
    axes[-1, 0].set_xlabel("Held-out test pattern")
    fig.suptitle("Paired balanced-BCE improvement from dense on identical seed/task/budget/init")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _plot_accuracy_deltas(aggregate: Mapping[str, Any], destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = aggregate["conditions"]
    design = aggregate["design"]
    fig, axes = plt.subplots(len(conditions), 1, figsize=(11, 7), sharex=True, squeeze=False)
    methods = [method for method in METHODS if method != "dense"]
    for row_index, condition in enumerate(conditions):
        ax = axes[row_index, 0]
        group_x = np.arange(len(design["task_ids"]))
        offsets = np.linspace(-0.22, 0.22, len(methods))
        for method_index, method in enumerate(methods):
            centers, lows, highs = [], [], []
            for task_id in design["task_ids"]:
                result = condition["tasks"][task_id][method]
                ci = result["accuracy_delta_ci95_over_seed_means"]
                center = result["accuracy_delta_vs_dense"]
                centers.append(center)
                lows.append(center - ci["ci_low"])
                highs.append(ci["ci_high"] - center)
            ax.errorbar(group_x + offsets[method_index], centers, yerr=[lows, highs],
                        fmt="o", capsize=3, color=METHOD_COLORS[method],
                        label=METHOD_LABELS[method])
        ax.axhline(0, color="#333333", linewidth=1, linestyle="--")
        ax.set_ylabel(f"Accuracy Δ\nbudget {condition['budget']}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncol=3, fontsize=8, loc="best")
    axes[-1, 0].set_xticks(np.arange(len(design["task_ids"])), design["task_ids"])
    axes[-1, 0].set_xlabel("Held-out test pattern")
    fig.suptitle("Secondary paired natural-accuracy difference from dense")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _plot_toeplitz_quality(records: Iterable[Mapping[str, Any]], root: Path,
                           destination: Path) -> bool:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(records)
    if not (root / "structure.jsonl").is_file():
        return False
    structure_rows = _records_from_structure(root / "structure.jsonl")
    indexed = {_record_key(row): row for row in structure_rows}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for method in METHODS:
        xs_support, balanced_bces, xs_weight = [], [], []
        for row in rows:
            if row["method"] != method:
                continue
            structure = indexed[_record_key(row)]["structure"]
            xs_support.append(float(structure["iou"]))
            xs_weight.append(float(structure["masked_effective_weight_structure"]
                                 ["all_offset_toeplitz_explained_energy"]))
            balanced_bces.append(float(row["test"]["balanced_bce"]))
        axes[0].scatter(xs_support, balanced_bces, s=18, alpha=0.55,
                        color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        axes[1].scatter(xs_weight, balanced_bces, s=18, alpha=0.55,
                        color=METHOD_COLORS[method], label=METHOD_LABELS[method])
    axes[0].set_xlabel("Mask support IoU after gold-only Hungarian alignment")
    axes[1].set_xlabel("Signed W×mask energy explained by all-offset diagonals")
    for ax in axes:
        ax.set_ylabel("Held-out balanced BCE (lower is better)")
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle("Post-hoc structure and primary held-out quality; each point is one frozen run")
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    return True


def _records_from_structure(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    return records


def _plot_selected_example(root: Path, destination: Path) -> bool:
    archive_path = root / "selected_example.npz"
    if not archive_path.is_file():
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    with np.load(archive_path, allow_pickle=False) as data:
        methods = data["method"].astype(str).tolist()
        masks = data["mask"].astype(float)
        effective = data["effective_weight"].astype(float)
        task_id = str(data["task_id"].item())
        seed = int(data["seed"].item())
        budget = int(data["budget"].item())
        init_id = int(data["init_id"].item())
    vmax = max(float(np.max(np.abs(effective))), 1e-12)
    fig, axes = plt.subplots(2, len(methods), figsize=(16, 5.5), squeeze=False)
    for column, method in enumerate(methods):
        mask_artist = axes[0, column].imshow(masks[column], origin="upper", aspect="auto",
                               cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        axes[0, column].set_title(METHOD_LABELS[method])
        axes[0, column].set_xlabel("Hidden column h")
        axes[0, column].set_xticks(range(8))
        axes[0, column].set_yticks(range(11))
        weight_artist = axes[1, column].imshow(effective[column], origin="upper", aspect="auto",
                               cmap="coolwarm", norm=Normalize(-vmax, vmax),
                               interpolation="nearest")
        axes[1, column].set_xlabel("Hidden column h")
        axes[1, column].set_xticks(range(8))
        axes[1, column].set_yticks(range(11))
    axes[0, 0].set_ylabel("Input coordinate i\nBinary mask")
    axes[1, 0].set_ylabel("Input coordinate i\nSigned W×mask")
    fig.suptitle(f"Common frozen condition: seed {seed}, {task_id}, support budget {budget}, init {init_id}")
    fig.tight_layout(rect=(0, 0, 0.94, 0.94))
    mask_scale = fig.add_axes([0.95, 0.58, 0.012, 0.25])
    weight_scale = fig.add_axes([0.95, 0.14, 0.012, 0.25])
    fig.colorbar(mask_artist, cax=mask_scale, ticks=[0, 1]).set_label("Allowed edge")
    fig.colorbar(weight_artist, cax=weight_scale).set_label("Signed W×mask")
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    return True


def _plot_optional_source_quality(root: Path, destination: Path) -> bool:
    import glob
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bank_paths = sorted(root.glob("seed_*/bank/bank.pt"))
    if (root / "bank.pt").is_file():
        bank_paths.insert(0, root / "bank.pt")
    if not bank_paths:
        return False
    distributions = []
    for bank_path in bank_paths:
        payload = torch.load(bank_path, map_location="cpu", weights_only=True)
        if "quality" not in payload:
            continue
        quality = np.asarray(payload["quality"], dtype=np.float64).reshape(-1)
        if quality.size and np.isfinite(quality).all():
            seed = next((part.replace("seed_", "") for part in bank_path.parts
                         if part.startswith("seed_")), "bank")
            distributions.append((seed, quality))
    if not distributions:
        return False
    fig, ax = plt.subplots(figsize=(8, 4.5))
    all_quality = np.concatenate([quality for _, quality in distributions])
    bins = np.histogram_bin_edges(all_quality, bins=30)
    for seed, quality in distributions:
        ax.hist(quality, bins=bins, alpha=0.45, label=f"seed {seed}")
    ax.set_xlabel("Minimum source-task query balanced BCE per selected teacher")
    ax.set_ylabel("Number of source maps")
    ax.set_title(f"Frozen source-bank quality/loss; {all_quality.size} maps across {len(distributions)} seed banks")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    return True


def _plot_optional_source_loss(root: Path, destination: Path) -> bool:
    """Plot the saved train/query balanced-BCE trajectories of source teachers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traces: dict[tuple[int, str, int], list[float]] = defaultdict(list)
    for bank_path in sorted(root.glob("seed_*/bank/bank.pt")):
        try:
            import torch
            bank = torch.load(bank_path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        seed_text = next((part.replace("seed_", "") for part in bank_path.parts
                          if part.startswith("seed_")), None)
        if seed_text is None:
            continue
        seed = int(seed_text)
        for payload in bank.get("source_curves", {}).values():
            steps = np.asarray(payload.get("step", []), dtype=np.int64).reshape(-1)
            for name in ("train_balanced_bce", "query_balanced_bce"):
                values = payload.get(name)
                if values is None:
                    continue
                values = np.asarray(values, dtype=np.float64)
                if values.ndim != 2 or values.shape[0] != steps.size:
                    continue
                means = np.nanmean(values, axis=1)
                for step, mean in zip(steps, means):
                    if np.isfinite(mean):
                        traces[(seed, name, int(step))].append(float(mean))
    if not traces:
        return False

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for seed in sorted({key[0] for key in traces}):
        for name, linestyle, label_suffix in (
            ("train_balanced_bce", "-", "train"),
            ("query_balanced_bce", "--", "query"),
        ):
            points = sorted((step, float(np.mean(values)))
                            for (row_seed, row_name, step), values in traces.items()
                            if row_seed == seed and row_name == name)
            if not points:
                continue
            ax.plot([point[0] for point in points], [point[1] for point in points],
                    linestyle=linestyle, linewidth=1.4,
                    label=f"seed {seed} {label_suffix}")
    ax.set_xlabel("Source-teacher training step")
    ax.set_ylabel("Balanced BCE across saved source teachers")
    ax.set_title("Frozen source-bank teacher curves; source query scores also curate the bank")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    return True


def _plot_optional_meta_curves(root: Path, destination: Path) -> bool:
    curves_path = root / "meta_curves.npz"
    if not curves_path.is_file():
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(curves_path, allow_pickle=False) as data:
        required = {"method", "seed", "lr", "step", "train_loss", "val_loss"}
        if required - set(data.files):
            return False
        method = data["method"].astype(str)
        seed = data["seed"].astype(int)
        lr = data["lr"].astype(float)
        step = data["step"].astype(float)
        train = data["train_loss"].astype(float)
        val = data["val_loss"].astype(float)
    if not all(array.ndim == 1 for array in (method, seed, lr, step, train, val)):
        return False
    if len({array.shape[0] for array in (method, seed, lr, step, train, val)}) != 1:
        return False
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in METHODS:
        for run_seed in np.unique(seed[method == name]):
            selected = (method == name) & (seed == run_seed)
            order = np.argsort(step[selected])
            ax.plot(step[selected][order], train[selected][order], alpha=0.22,
                    color=METHOD_COLORS[name], linewidth=0.8)
            ax.plot(step[selected][order], val[selected][order], alpha=0.85,
                    color=METHOD_COLORS[name], linewidth=1.3,
                    label=f"{METHOD_LABELS[name]} val, seed={run_seed}, initial lr={lr[selected][0]:g}")
    ax.set_xlabel("Meta-training step")
    ax.set_ylabel("Utility BCE (train and meta-validation tasks)")
    ax.set_title("Meta-training curves; LR and checkpoint decisions use only meta-validation")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    return True


def _plot_optional_child_curves(root: Path, destination: Path) -> bool:
    """Plot saved support/query balanced-BCE traces and observed-run counts."""
    curves_path = root / "child_curves.npz"
    if not curves_path.is_file():
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(curves_path, allow_pickle=False) as data:
        method_key = "method"
        train_key = "train_loss" if "train_loss" in data.files else "support_balanced_bce"
        query_key = "query_loss" if "query_loss" in data.files else "query_balanced_bce"
        required = {method_key, "budget", "step", "seed", "task_id", "init_id", train_key, query_key}
        if required - set(data.files):
            return False
        method = data[method_key].astype(str)
        budget = data["budget"].astype(int)
        step = data["step"].astype(int)
        train = data[train_key].astype(float)
        query = data[query_key].astype(float)
        seed = data["seed"].astype(int)
        task_id = data["task_id"].astype(str)
        init_id = data["init_id"].astype(int)
    if not all(array.ndim == 1 for array in (method, budget, step, train, query)):
        return False
    if len({array.shape[0] for array in (method, budget, step, train, query)}) != 1:
        return False
    valid = np.isfinite(train) & np.isfinite(query) & (step >= 1)
    # A batch retains frozen rows while its slower peers continue. Display each
    # child's trajectory only through its own stopping step, not these repeated
    # snapshots of an unchanged model.
    records_path = root / "records.jsonl"
    if not records_path.is_file():
        return False
    stopped_at = {
        _record_key(row): int(row["fit_status"]["steps"])
        for row in _records_from_jsonl(records_path)
    }
    limits = np.asarray([
        stopped_at[(str(m), int(s), str(t), int(b), int(i))]
        for m, s, t, b, i in zip(method, seed, task_id, budget, init_id)
    ])
    valid &= step <= limits
    method, budget, step = method[valid], budget[valid], step[valid]
    train, query = train[valid], query[valid]
    if not step.size:
        return False
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex="col")
    for column, budget_value in enumerate(EXPECTED_BUDGETS):
        loss_ax, count_ax = axes[0, column], axes[1, column]
        for name in METHODS:
            selected_method = (method == name) & (budget == budget_value)
            if not selected_method.any():
                continue
            steps = np.unique(step[selected_method])
            train_means, query_means, counts = [], [], []
            for iteration in steps:
                at_step = selected_method & (step == iteration)
                train_means.append(float(np.mean(train[at_step])))
                query_means.append(float(np.mean(query[at_step])))
                counts.append(int(at_step.sum()))
            loss_ax.plot(steps, train_means, color=METHOD_COLORS[name], alpha=0.45,
                         linewidth=1, linestyle="-", label=f"{METHOD_LABELS[name]} train")
            loss_ax.plot(steps, query_means, color=METHOD_COLORS[name], alpha=0.95,
                         linewidth=1.5, linestyle="--", label=f"{METHOD_LABELS[name]} query")
            count_ax.plot(steps, counts, color=METHOD_COLORS[name], linewidth=1.2,
                          label=METHOD_LABELS[name])
        loss_ax.axvline(1000, color="#333333", linewidth=0.8, linestyle=":")
        loss_ax.set_title(f"Support budget {budget_value}")
        loss_ax.set_ylabel("Balanced BCE")
        loss_ax.grid(alpha=0.25)
        loss_ax.legend(fontsize=6, ncol=2)
        count_ax.set_xlabel("Child training step")
        count_ax.set_ylabel("Runs observed at step")
        count_ax.grid(alpha=0.25)
        count_ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Fresh-child support/query balanced-BCE traces; points average available runs at each step")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    return True


def _write_captions(path: Path, has_source: bool, has_source_loss: bool,
                    has_meta: bool, has_child: bool) -> None:
    captions = [
        "# Figure captions and scope",
        "",
        "`per_task_balanced_bce.png`: each panel is one held-out pattern and support budget. The x-axis names the six mask methods; the y-axis is test balanced BCE (equal positive/negative class weight, lower is better). Points average four child initializations within each of outer seeds 8100–8103. Error bars are 95% Student-t intervals over the four seed means (df=3). Each panel uses all 414 test IDs.",
        "",
        "`per_task_accuracy.png`: each panel is one held-out pattern and support budget. The x-axis names the six mask methods; the y-axis is natural test accuracy at the strict logit threshold >0. Points average child initializations within each seed; error bars are descriptive 95% t intervals across four seed means (df=3). Accuracy is secondary to balanced BCE.",
        "",
        "`paired_deltas.png`: balanced-BCE improvement from dense on the same task, seed, budget and child initialization (dense balanced BCE minus method balanced BCE). The x-axis is held-out pattern; positive y values favor the named mask method. Error bars are descriptive 95% t intervals over four paired seed means (df=3).",
        "",
        "`paired_accuracy_deltas.png`: secondary natural-accuracy difference from dense on the same task, seed, budget and child initialization. Positive values favor the named method. Error bars are descriptive 95% t intervals over four paired seed means (df=3).",
        "",
        "`toeplitz_quality.png`: left x-axis is exact-mask support IoU after Hungarian matching to gold windows; right x-axis is the fraction of signed `W×mask` squared norm explained by projection onto every constant-offset diagonal. The y-axis is held-out balanced BCE (lower is better). This plot shows the raw, coordinate- and positive-ReLU-gauge-dependent weight score; readout-scaled and affine-column-normalized scores are retained in `structure_metrics.npz`. Gold is used only after all checkpoints and masks are frozen. These associations do not establish that structure caused quality.",
        "",
        "`selected_masks_and_weights.png`: columns are methods, top row is the binary 11×8 mask and bottom row is the signed trained `W×mask` matrix. All panels use seed 8100, test task index 0, budget 128 and child initialization 0. Mask colors share [0,1]; signed weights share one symmetric color range across all methods. Rows are input coordinate i and columns are hidden unit h.",
        "",
    ]
    if has_source:
        captions.extend([
            "`source_bank_quality.png`: overlapping histograms of each seed's frozen source-bank teacher quality, measured as minimum query balanced BCE across saved source-training snapshots. The x-axis is balanced BCE and the y-axis is number of source maps. The source query labels may curate the bank; no target or held-out test labels enter this plot.",
            "",
        ])
    else:
        captions.extend(["Source-bank quality figure was not rendered because no `seed_*/bank/bank.pt` quality vector was found.", ""])
    if has_source_loss:
        captions.extend([
            "`source_bank_loss.png`: x-axis is source-teacher training step and y-axis is balanced BCE averaged over saved teachers within each seed and task. Solid lines show source-train scores and dashed lines show source-query scores. Source-query labels curate the frozen bank; these curves are source diagnostics, not target-task validation or evidence of causal bank use.",
            "",
        ])
    else:
        captions.extend(["Source-teacher loss curves were not rendered because saved source training histories were unavailable.", ""])
    if has_meta:
        captions.extend([
            "`meta_curves.png`: x-axis is meta-training step and y-axis is utility BCE. Thin lines show the fixed train-monitor query curves; opaque lines show meta-validation query curves for each method and outer seed. Each seed has its own trace; values from different fits are never connected. Labels record the initial outer learning rate, which may decay later. The checkpoints were selected on validation data; held-out test tasks are not included.",
            "",
        ])
    else:
        captions.extend(["Meta-training curve figure was not rendered because `meta_curves.npz` is absent or incomplete.", ""])
    if has_child:
        captions.extend([
            "`child_convergence.png`: columns are support budgets 32 and 128; top-row y-axis is support/query balanced BCE, and bottom-row y-axis is number of runs with a saved observation at that step. Curves average available runs by method; lower panels disclose changing run counts after individual stopping. The vertical dotted line marks the 1,000-step minimum. Query loss selects checkpoints; test loss is absent.",
            "",
        ])
    else:
        captions.extend(["Child convergence figure was not rendered because `child_curves.npz` is absent or incomplete.", ""])
    path.write_text("\n".join(captions), encoding="utf-8")


def _format_ci(result: Mapping[str, Any], field: str = "accuracy_ci95_over_seed_means") -> str:
    interval = result[field]
    return f"{result['mean_accuracy']:.3f} [{interval['ci_low']:.3f}, {interval['ci_high']:.3f}]"


def _write_reports(root: Path, aggregate: Mapping[str, Any], figures: Mapping[str, bool]) -> None:
    design = aggregate["design"]
    lines = [
        "# Task-quality and Toeplitz-structure evaluation",
        "",
        "This report compares six fixed-mask methods on four held-out length-4 patterns. The four test patterns are members of a single reversal/complement orbit, so they are related tasks rather than four independent task orbits. Utility tasks use binary length-11 inputs and width-8 ReLU children. Every sparse method has exactly 32 active edges; dense has 88; the oracle mask is a gold-only reference. Each condition uses outer seeds 8100–8103 and four paired fresh-child initializations.",
        "",
        "The primary score is class-balanced held-out BCE; lower is better. Natural accuracy, balanced accuracy, natural BCE and Brier score are secondary metrics. Checkpoints are selected by query balanced BCE and frozen before test scoring. Each test uses the complete 414-ID test partition from the finite 2,048 input space; query selection uses the complete 408-ID query partition. Balanced 32/128 support IDs, query IDs and test IDs are disjoint. Test labels are used only for the final report.",
        "",
        "## Primary per-task balanced BCE",
        "",
        "Each point averages the four child initializations within a seed, then averages the four outer-seed means. Brackets show descriptive 95% Student-t intervals across those four seed means (df=3). Balanced BCE gives equal weight to positive-class and negative-class BCE.",
        "",
        "| Budget | Test pattern | " + " | ".join(METHOD_LABELS[m] for m in METHODS) + " |",
        "|---:|---|" + "---:|" * len(METHODS),
    ]
    for condition in aggregate["conditions"]:
        budget = condition["budget"]
        for task_id in design["task_ids"]:
            text_cells = []
            for method in METHODS:
                result = condition["tasks"][task_id][method]
                ci = result["balanced_bce_ci95_over_seed_means"]
                text_cells.append(f"{result['mean_balanced_bce']:.3f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]")
            lines.append(f"| {budget} | `{task_id}` | " + " | ".join(text_cells) + " |")
    lines += [
        "",
        "![Per-task balanced test BCE](figures/per_task_balanced_bce.png)",
        "",
        "## Secondary accuracy",
        "",
        "Natural accuracy uses a strict logit threshold greater than zero, matching the child model. Balanced accuracy weights positive and negative recall equally. These values are secondary to balanced BCE.",
        "",
        "![Per-task natural test accuracy](figures/per_task_accuracy.png)",
        "",
        "## Paired balanced-BCE improvement over dense",
        "",
        "Each difference pairs identical outer seed, held-out task, budget and fresh-child initialization. Positive improvement means the method's balanced BCE is lower than dense's. `Fraction tasks improved` counts held-out patterns with positive mean paired improvement; `Worst task improvement` is the minimum of the four task means. These four related patterns do not support a guarantee for unseen tasks.",
        "",
        "| Budget | Method | Mean improvement (dense BCE − method BCE) | 95% t interval across seeds | Fraction tasks improved | Worst task improvement |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for result in aggregate["overall"]:
        ci = result["balanced_bce_improvement_ci95_over_seed_means"]
        lines.append(
            f"| {result['budget']} | {METHOD_LABELS[result['method']]} | "
            f"{result['mean_balanced_bce_improvement_vs_dense']:+.3f} | "
            f"[{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}] | "
            f"{result['fraction_tasks_with_positive_balanced_bce_improvement']:.2f} | "
            f"{result['worst_task_balanced_bce_improvement']:+.3f} |"
        )
    lines += [
        "",
        "![Paired balanced-BCE improvements from dense](figures/paired_deltas.png)",
        "",
        "Natural-accuracy paired differences are retained as a secondary outcome in [`figures/paired_accuracy_deltas.png`](figures/paired_accuracy_deltas.png) and `summary.json`.",
        "",
        "## Frozen learning-rate selection",
        "",
        "Each method and support budget uses one learning rate from $\\{0.001,0.003,0.01\\}$ selected by mean query balanced BCE on the two meta-validation patterns, across seeds 8100–8103 and child initializations 0–3. The selected rate is frozen across all four test patterns. No test score contributes to this choice.",
        "",
        "| Budget | Method | Selected LR | Query balanced BCE at 0.001 | at 0.003 | at 0.01 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in aggregate["lr_selection"]:
        scores = row["validation_query_bce_by_lr"]
        lines.append(f"| {row['budget']} | {METHOD_LABELS[row['method']]} | {row['selected_lr']:.3g} | "
                     f"{scores['0.001']:.4f} | {scores['0.003']:.4f} | {scores['0.01']:.4f} |")
    lines += [
        "",
        "## Child convergence and checkpoint completion",
        "",
        "Train and query traces are retained at their recorded steps. Runs that reach the configured step cap without meeting the convergence rule remain labeled unconverged; the cap is never treated as successful convergence. Query checkpoint selection completion is reported separately.",
        "",
        "| Budget | Method | Runs | Converged | Unconverged | At step cap | Query checkpoint selected | Mean steps |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate["training_status"]:
        lines.append(f"| {row['budget']} | {METHOD_LABELS[row['method']]} | {row['n_runs']} | "
                     f"{row['converged_runs']} | {row['unconverged_runs']} | {row['step_cap_runs']} | "
                     f"{row['query_selection_complete_runs']} | {row['mean_steps']:.0f} |")
    lines += [
        "",
        "![Train and query convergence traces](figures/child_convergence.png)",
        "",
        "## Mask and signed-weight structure",
        "",
        "Hidden columns are matched to canonical four-edge windows by a Hungarian assignment that uses only binary mask overlap. This assignment is post-hoc and uses the gold support only for reporting. Mask IoU measures support recovery. The weight audit separately measures how much squared norm of the signed trained `W×mask` lies in the constant-offset diagonal subspace across all offsets, and in the oracle band $i-h\\in\\{0,1,2,3\\}$. These are distinct quantities: support recovery does not imply repeated signed weights.",
        "",
        "ReLU units have a positive rescaling gauge: scaling a hidden unit's incoming weights and bias by a positive factor while inversely scaling its readout leaves the function unchanged. The raw signed `W×mask` projection is therefore coordinate- and gauge-dependent. We also report `a_j W_j M_j`, invariant to positive hidden rescaling, and incoming affine columns `[W_j M_j,b_j]` normalized by their joint L2 norm. The heatmap retains raw trained `W×mask`; readout signs are preserved. Binary support recovery is independent of this weight gauge. A low raw-weight projection alone cannot establish absence of functionally Toeplitz behavior.",
        "",
        "![Support and signed-weight structure versus quality](figures/toeplitz_quality.png)",
        "",
        "![Common-condition masks and trained signed weights](figures/selected_masks_and_weights.png)",
        "",
        "Full per-run masks, aligned weights, projection values, offset histograms and Hungarian permutations are in `structure.jsonl` and `structure_metrics.npz`. `selected_example.npz` stores the fixed-condition matrices behind the heatmap.",
        "",
        "## Figures and numeric artifacts",
        "",
        "Figure axes, color scales, conditions and interpretation limits are recorded in [`figures/captions.md`](figures/captions.md). `records.jsonl` stores run-level IDs, masks, weights, provenance and status; `summary.npz` stores aligned numeric test metrics and selected rates; paired dense deltas are in `paired_deltas.npz`; structure metrics are in `structure_metrics.npz`.",
        "",
        f"Source-bank quality figure rendered: **{figures['source_quality']}**. Source-teacher loss curves rendered: **{figures['source_loss']}**. Meta-training curves rendered: **{figures['meta_curves']}**. Child convergence curves rendered: **{figures['child_curves']}**.",
        "",
        "## Limitations",
        "",
        "Intervals are descriptive with four outer seeds and use $t_{0.975,3}$. The four held-out patterns are members of one reversal/complement orbit and are related rather than independent tasks. The task-averaged score and fraction improved do not guarantee improvement on every future pattern. One frozen source bank is used per independent fit; bank content is not varied, so this pilot does not measure bank generalization or establish causal bank use. `free_mask` differs from the Transformer in context and architecture as well as bank input, so it is not a pure bank ablation. Meta-training differentiates through a 64-step SGD inner loop, but the binary top-$K$ mask uses a hard-forward sigmoid straight-through estimator; its backward signal is a biased surrogate, not the exact gradient of the discrete-mask objective. Final children use Adam with a validation-tuned rate and query-selected checkpoint, so their optimizer and horizon differ from the finite meta-training utility objective. Structure is audited after training and does not affect checkpoint selection. The analytic oracle mask is a structural reference and must not be treated as a learned result.",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    appendix = [
        "# Appendix: utility-trained masks and post-hoc Toeplitz audit",
        "",
        "## Evaluation protocol",
        "",
        "A bank Transformer proposes an exact-$K=32$ support mask from source-map features and adapts only through utility on the training-pattern tasks. `free_mask` learns the same number of mask logits directly, `functional_centroid_mean` uses only source-map probe features, and `random_exact32` draws one of four independent exact-cardinality supports paired by child initialization. Dense uses all 88 links. `oracle_mask` equals the analytic $11\\times8$ gold support and is an evaluation reference only. Each selected mask is frozen before a fresh child is initialized and trained on target support examples; query labels select the child checkpoint; target test labels are scored once after freeze.",
        "",
        "Patterns are split by reversal/complement orbits, with 10 length-4 patterns for meta-training, 2 for meta-validation, and the four members of one held-out reversal/complement orbit for final testing. For each input ID, the 11-bit sequence is assigned to support/query/test by a task-independent hash partition. All labels are analytic substring-presence labels. The final report uses the test pool only after mask, learning rate and query checkpoint selection are frozen.",
        "",
        "## Support and weight metrics",
        "",
        "The analytic support is $M_{ih}=1$ exactly when $0\\le i-h<4$. A Hungarian assignment maps each learned hidden column to one gold window by maximizing binary overlap. The reported support IoU, precision and recall follow that assignment. Per-column window coverage is its best overlap divided by 4; exact recovery requires every one of the eight canonical windows to appear in exactly one learned column. Column locality span and the aligned relative-offset histogram are also reported.",
        "",
        "Signed weights are analyzed only after the support-derived permutation is fixed. The effective matrix is $W_{ih}M_{ih}$. Projection energy is reported for the full family of constant-offset diagonals (offsets $-7$ through $10$) and separately for the gold offsets $0$ through $3$. Raw, column-normalized, joint `[W_jM_j,b_j]`-normalized and readout-scaled forms are included. Raw projection varies under the positive ReLU rescaling gauge; `a_j W_j M_j` and the joint-normalized incoming affine column are invariant to positive rescaling. The actual raw weights remain visible in the heatmap. These metrics distinguish a Toeplitz-shaped support from repeated signed weights without treating a raw score as functionally invariant.",
        "",
        "## Results",
        "",
        "The complete task-by-method balanced-BCE table and paired differences are in [`REPORT.md`](REPORT.md). Natural accuracy and balanced accuracy are secondary. The numerical archive stores every condition, not only averaged values. The paired comparison reuses seed, task, budget and child initialization across methods; intervals are descriptive 95% Student-t intervals over four outer-seed means (df=3). Per-task improvements and the worst task remain visible because a positive grand mean alone does not show whether every test task improved.",
        "",
        "![Paired balanced-BCE improvements](figures/paired_deltas.png)",
        "",
        "![Post-hoc structure versus held-out quality](figures/toeplitz_quality.png)",
        "",
        "![Masks and signed effective weights at the predeclared common condition](figures/selected_masks_and_weights.png)",
        "",
        "## Interpretation limits",
        "",
        "Gold is used only in post-hoc support alignment and oracle-reference evaluation. It is never a Transformer input, training target, checkpoint criterion or test-time selector. The four test patterns are related members of one reversal/complement orbit. Each independent fit uses one fixed source bank, so bank content is not varied and bank generalization or causal bank use is not measured. `free_mask` changes architecture/context as well as bank input, so it is not a pure bank ablation. Meta-training differentiates through a finite 64-step SGD inner loop and uses a biased sigmoid straight-through gradient for the hard top-$K$ mask; this is not an exact hypergradient of the discrete-mask objective. Final child fitting uses Adam, so the optimizer and horizon differ from the meta-training utility objective. A high support IoU does not establish repeated signed weights, while raw weight projection varies under positive ReLU rescaling; the supplementary readout-scaled and joint-normalized scores address that gauge but do not establish causal utility.",
        "",
        "See [`figures/captions.md`](figures/captions.md) for axes, units, color scales, conditions and figure limitations.",
        "",
    ]
    (root / "REPORT27_APPENDIX.md").write_text("\n".join(appendix), encoding="utf-8")

    optional_figure_links: list[str] = ["", "## Source and training diagnostics", ""]
    if figures["source_quality"]:
        optional_figure_links.extend([
            "![Frozen source-bank teacher quality](figures/source_bank_quality.png)", "",
        ])
    if figures["source_loss"]:
        optional_figure_links.extend([
            "![Frozen source-teacher train/query loss](figures/source_bank_loss.png)", "",
        ])
    if figures["meta_curves"]:
        optional_figure_links.extend([
            "![Meta-training query curves](figures/meta_curves.png)", "",
        ])
    if figures["child_curves"]:
        optional_figure_links.extend([
            "![Fresh-child convergence curves](figures/child_convergence.png)", "",
        ])
    if len(optional_figure_links) > 3:
        (root / "REPORT.md").write_text(
            (root / "REPORT.md").read_text(encoding="utf-8")
            .replace("## Limitations\n", "\n".join(optional_figure_links) + "\n## Limitations\n"),
            encoding="utf-8",
        )


def generate_report(root: str | Path, *, require_complete: bool = True) -> dict[str, Any]:
    """Aggregate canonical records and write summaries, figures and Markdown."""
    path = Path(root)
    records = _records_from_jsonl(path / "records.jsonl")
    if require_complete:
        design = validate_complete_design(records)
    else:
        design = None
    write_summary_npz(records, path / "summary.npz")
    aggregate = aggregate_records(records) if require_complete else None
    if aggregate is None:
        raise ValueError("report generation requires the full prespecified design")
    # This checks key alignment, one-time test records, and finite numeric fields.
    validation = validate_artifact_root(path)
    path.mkdir(parents=True, exist_ok=True)
    figures_dir = path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    structure_status = write_structure_artifacts(records, path)
    deltas = []
    lookup = _index_records(records)
    for method in METHODS:
        for seed in aggregate["design"]["seeds"]:
            for task_index, task_id in enumerate(aggregate["design"]["task_ids"]):
                for budget in aggregate["design"]["budgets"]:
                    for init_id in aggregate["design"]["init_ids"]:
                        row = lookup[(method, seed, task_index, budget, init_id)]
                        dense = lookup[("dense", seed, task_index, budget, init_id)]
                        deltas.append((method, seed, task_id, task_index, budget, init_id,
                                       float(dense["test"]["balanced_bce"])
                                       - float(row["test"]["balanced_bce"]),
                                       float(row["test"]["accuracy"])
                                       - float(dense["test"]["accuracy"]),
                                       float(row["test"]["balanced_accuracy"])
                                       - float(dense["test"]["balanced_accuracy"])))
    np.savez_compressed(
        path / "paired_deltas.npz",
        method=np.asarray([row[0] for row in deltas], dtype="U32"),
        seed=np.asarray([row[1] for row in deltas], dtype=np.int64),
        task_id=np.asarray([row[2] for row in deltas], dtype="U16"),
        task_index=np.asarray([row[3] for row in deltas], dtype=np.int64),
        budget=np.asarray([row[4] for row in deltas], dtype=np.int64),
        init_id=np.asarray([row[5] for row in deltas], dtype=np.int64),
        balanced_bce_improvement=np.asarray([row[6] for row in deltas], dtype=np.float64),
        accuracy_delta=np.asarray([row[7] for row in deltas], dtype=np.float64),
        balanced_accuracy_delta=np.asarray([row[8] for row in deltas], dtype=np.float64),
    )
    selected_example_available = write_selected_example(records, path)
    _plot_quality(aggregate, figures_dir / "per_task_accuracy.png")
    _plot_balanced_bce(aggregate, figures_dir / "per_task_balanced_bce.png")
    _plot_paired_deltas(aggregate, figures_dir / "paired_deltas.png")
    _plot_accuracy_deltas(aggregate, figures_dir / "paired_accuracy_deltas.png")
    toeplitz_available = _plot_toeplitz_quality(records, path, figures_dir / "toeplitz_quality.png")
    selected_plot_available = _plot_selected_example(path, figures_dir / "selected_masks_and_weights.png")
    source_quality = _plot_optional_source_quality(path, figures_dir / "source_bank_quality.png")
    source_loss = _plot_optional_source_loss(path, figures_dir / "source_bank_loss.png")
    meta_curves = _plot_optional_meta_curves(path, figures_dir / "meta_curves.png")
    child_curves = _plot_optional_child_curves(path, figures_dir / "child_convergence.png")
    _write_captions(figures_dir / "captions.md", source_quality, source_loss,
                    meta_curves, child_curves)
    if not selected_example_available or not selected_plot_available:
        raise ValueError("required fixed-condition heatmap is unavailable; expected seed 8100/task 0/budget 128/init 0")
    if not toeplitz_available:
        raise ValueError("post-hoc Toeplitz structure plot was not produced")

    aggregate["artifact_validation"] = validation
    aggregate["structure_artifacts"] = structure_status
    aggregate["figures"] = {
        "per_task_accuracy": True,
        "per_task_balanced_bce": True,
        "paired_deltas": True,
        "paired_accuracy_deltas": True,
        "toeplitz_quality": toeplitz_available,
        "selected_masks_and_weights": selected_plot_available,
        "source_bank_quality": source_quality,
        "source_loss": source_loss,
        "meta_curves": meta_curves,
        "child_curves": child_curves,
    }
    (path / "summary.json").write_text(
        json.dumps(_jsonable(aggregate), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_reports(path, aggregate, {"source_quality": source_quality,
                                     "source_loss": source_loss,
                                     "meta_curves": meta_curves,
                                     "child_curves": child_curves})
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = generate_report(args.root, require_complete=True)
    print(json.dumps(_jsonable({"design": result["design"],
                                "artifact_validation": result["artifact_validation"],
                                "figures": result["figures"]}), indent=2))


if __name__ == "__main__":
    main()
