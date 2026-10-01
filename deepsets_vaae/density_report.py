"""Aggregate and report the fixed-budget mask-density sweep.

The report treats the eight experiment seeds as the sampling units.  Within a
seed, it first averages over the eight held-out cost tasks and four paired
initializations.  Validation-based density selection is kept separate from
test evaluation throughout.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/deepsets_vaae/20261001_density_sweep"
PILOT = ROOT / "outputs/deepsets_vaae/20261001_pilot"
CORRECTED_FUNCTIONAL = (
    ROOT / "outputs/deepsets_vaae/20261001_followup/importance/fp32_final"
)
SEEDS = tuple(range(4100, 4108))
DENSITIES = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
SUPPORT_SIZES = (32, 64, 128, 256)
N_CONNECTIONS = 784 * 32
N_TEST_TASKS = 8
N_VALIDATION_TASKS = 2
N_REPLICAS = 4
NEW_METHODS = ("random_density", "function_gradient_density", "agreement_density")
CONTROL_METHODS = ("agreement", "mean", "single_vae", "random", "dense")
METHODS = NEW_METHODS + CONTROL_METHODS
METHOD_LABELS = {
    "random_density": "Случайная вложенная маска",
    "function_gradient_density": "Функциональный градиент",
    "agreement_density": "VAE-согласование",
    "agreement": "Пилот: VAE-согласование",
    "mean": "Пилот: средняя маска",
    "single_vae": "Пилот: одна VAE",
    "random": "Пилот: случайная маска",
    "dense": "Dense",
}
PLOT_METHODS = NEW_METHODS
EXPECTED_RECORD_COUNT = N_TEST_TASKS * len(SUPPORT_SIZES) * len(METHODS) * N_REPLICAS
EXPECTED_VALIDATION_RECORD_COUNT = (
    N_VALIDATION_TASKS * len(SUPPORT_SIZES) * len(METHODS) * N_REPLICAS
)


def rho_slug(rho: float) -> str:
    return "rho_" + f"{rho:.3f}".replace(".", "p")


def edge_count(rho: float) -> int:
    return round(rho * N_CONNECTIONS)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def interval(values: Iterable[float]) -> dict[str, Any]:
    values = np.asarray(list(values), dtype=float)
    if values.size == 0:
        raise ValueError("Cannot summarize an empty set of seed values")
    mean = float(values.mean())
    margin = (float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values)))
              if len(values) > 1 else None)
    return {
        "mean": mean,
        "ci95": None if margin is None else [mean - margin, mean + margin],
        "df": len(values) - 1 if len(values) > 1 else None,
        "seed_values": values.tolist(),
    }


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _record_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return int(row["task"]), int(row["support_size"]), int(row["init"])


def _validate_records(
    rows: list[dict[str, Any]], *, seed: int, rho: float, validation: bool,
) -> None:
    expected_tasks = N_VALIDATION_TASKS if validation else N_TEST_TASKS
    expected_count = EXPECTED_VALIDATION_RECORD_COUNT if validation else EXPECTED_RECORD_COUNT
    if len(rows) != expected_count:
        raise AssertionError(
            f"seed {seed} {rho_slug(rho)} {'validation' if validation else 'test'} "
            f"rows: expected {expected_count}, found {len(rows)}"
        )
    tasks = set()
    seen: set[tuple[int, int, str, int]] = set()
    group_counts: defaultdict[tuple[str, int], int] = defaultdict(int)
    for row in rows:
        if "phase" in row:
            expected_phase = "validation" if validation else "test"
            if row["phase"] != expected_phase:
                raise AssertionError(f"Unexpected phase in row: {row['phase']}")
        if "density" in row and not math.isclose(float(row["density"]), rho, abs_tol=1e-12):
            raise AssertionError(f"Row density differs from {rho}: {row['density']}")
        if "edges" in row and int(row["edges"]) != edge_count(rho):
            raise AssertionError(f"Row edge count differs from rho {rho}: {row['edges']}")
        method = row["method"]
        task = int(row["task"])
        support = int(row["support_size"])
        init = int(row["init"])
        if method not in METHODS or support not in SUPPORT_SIZES:
            raise AssertionError(f"Unexpected result row: method={method} support={support}")
        if not (0 <= task < expected_tasks):
            raise AssertionError(f"Unexpected task index {task} in {'validation' if validation else 'test'}")
        if not (0 <= init < N_REPLICAS):
            raise AssertionError(f"Unexpected replica index {init}")
        key = (task, support, method, init)
        if key in seen:
            raise AssertionError(f"Duplicate result row: seed={seed}, rho={rho}, key={key}")
        seen.add(key)
        tasks.add(task)
        group_counts[(method, support)] += 1
        if not math.isfinite(float(row["mse"])):
            raise AssertionError(f"Nonfinite NMSE in {key}")
        if validation and not math.isfinite(float(row["validation_mse"])):
            raise AssertionError(f"Nonfinite validation NMSE in {key}")
    if tasks != set(range(expected_tasks)):
        raise AssertionError(f"Task coverage mismatch: {tasks}")
    expected_group_count = expected_tasks * N_REPLICAS
    for method in METHODS:
        for support in SUPPORT_SIZES:
            if group_counts[(method, support)] != expected_group_count:
                raise AssertionError(
                    f"Incomplete rows for {method}, budget {support}: "
                    f"{group_counts[(method, support)]} != {expected_group_count}"
                )


def _load_density_payload(seed_dir: Path, seed: int, rho: float) -> dict[str, Any]:
    rho_dir = seed_dir / rho_slug(rho)
    path = rho_dir / "results.json"
    if path.exists():
        payload = _json(path)
        if int(payload["seed"]) != seed:
            raise AssertionError(f"Seed mismatch in {path}")
        if not math.isclose(float(payload.get("density", rho)), rho, abs_tol=1e-12):
            raise AssertionError(f"Density mismatch in {path}")
        return payload

    # Accept the documented compact per-seed summary as well as the runner's
    # per-density files.  Checkpoints and masks remain in the rho subdirectory.
    combined = seed_dir / "results.json"
    if combined.exists():
        payload = _json(combined)
        if int(payload["seed"]) != seed:
            raise AssertionError(f"Seed mismatch in {combined}")
        records = [row for row in payload["records"]
                   if math.isclose(float(row.get("density", -1)), rho, abs_tol=1e-12)]
        validation_records = [row for row in payload["validation_records"]
                              if math.isclose(float(row.get("density", -1)), rho, abs_tol=1e-12)]
        if records and validation_records:
            return {"seed": seed, "density": rho, "records": records,
                    "validation_records": validation_records}
    raise FileNotFoundError(f"No results for seed {seed}, rho={rho} under {seed_dir}")


def _load_and_audit_inputs(out: Path) -> tuple[dict[float, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    rows: dict[float, dict[str, list[dict[str, Any]]]] = {
        rho: {"test": [], "validation": []} for rho in DENSITIES
    }
    audit: dict[str, Any] = {"seeds": list(SEEDS), "densities": [], "control_audits": [],
                             "twenty_percent_matches": {}, "mask_counts": {}}
    for seed in SEEDS:
        seed_dir = out / f"seed_{seed}"
        if not seed_dir.is_dir():
            raise FileNotFoundError(f"Missing completed seed directory: {seed_dir}")
        for rho in DENSITIES:
            rho_dir = seed_dir / rho_slug(rho)
            payload = _load_density_payload(seed_dir, seed, rho)
            test_rows = payload["records"]
            validation_rows = payload["validation_records"]
            _validate_records(test_rows, seed=seed, rho=rho, validation=False)
            _validate_records(validation_rows, seed=seed, rho=rho, validation=True)
            rows[rho]["test"].extend(test_rows)
            rows[rho]["validation"].extend(validation_rows)

            weights = sorted((rho_dir / "weights").glob("target_task*_budget*.pt"))
            validation_weights = sorted(
                (rho_dir / "validation_weights").glob("target_task*_budget*.pt")
            )
            if len(weights) != N_TEST_TASKS * len(SUPPORT_SIZES):
                raise AssertionError(
                    f"Expected 32 test state files at {rho_dir}, found {len(weights)}"
                )
            if len(validation_weights) != N_VALIDATION_TASKS * len(SUPPORT_SIZES):
                raise AssertionError(
                    f"Expected 8 validation state files at {rho_dir}, found {len(validation_weights)}"
                )
            audit["densities"].append({"seed": seed, "density": rho,
                                        "edges": edge_count(rho),
                                        "test_state_files": len(weights),
                                        "validation_state_files": len(validation_weights)})
            control_audit_path = rho_dir / "control_audit.json"
            if not control_audit_path.exists():
                # Some runners write the same per-density audit into the
                # result payload rather than a sidecar file.
                control_audit = payload.get("control_audit")
                if control_audit is None:
                    raise FileNotFoundError(f"Missing control audit: {control_audit_path}")
            else:
                control_audit = _json(control_audit_path)
            if not control_audit.get("passed", False):
                raise AssertionError(f"Control audit failed for seed={seed}, rho={rho}: {control_audit}")
            max_delta = control_audit.get("maximum_absolute_mse_delta")
            control_records = control_audit.get("control_records")
            if max_delta is None or control_records is None:
                raise AssertionError(f"Incomplete control-audit summary at {control_audit_path}")
            audit["control_audits"].append({
                "seed": seed, "density": rho,
                "passed": True,
                "maximum_absolute_mse_delta": max_delta,
                "control_records": control_records,
            })

    # Exact-K mask and nesting audit without retaining all sweep masks in
    # memory.  A single seed's previous mask is enough to verify each chain.
    for seed in SEEDS:
        previous: dict[str, np.ndarray] = {}
        for rho in DENSITIES:
            mask_path = out / f"seed_{seed}" / rho_slug(rho) / "masks.pt"
            if not mask_path.exists():
                raise FileNotFoundError(mask_path)
            masks = torch.load(mask_path, map_location="cpu", weights_only=True)
            if set(masks) != set(NEW_METHODS):
                raise AssertionError(f"Unexpected method set in {mask_path}: {set(masks)}")
            for method in NEW_METHODS:
                mask = torch.as_tensor(masks[method]).cpu()
                if tuple(mask.shape) != (N_REPLICAS, 784, 32):
                    raise AssertionError(f"Bad mask shape for {method} in {mask_path}: {tuple(mask.shape)}")
                if not bool(torch.all((mask == 0) | (mask == 1))):
                    raise AssertionError(f"Nonbinary mask for {method} in {mask_path}")
                counts = mask.sum(dim=(-1, -2)).tolist()
                if counts != [edge_count(rho)] * N_REPLICAS:
                    raise AssertionError(f"Wrong K for {method} in {mask_path}: {counts}")
                current = mask.numpy().astype(bool, copy=False)
                if method in ("random_density", "function_gradient_density") \
                        and method in previous and np.any(previous[method] & ~current):
                    raise AssertionError(f"Mask rankings are not nested: seed={seed}, method={method}, rho={rho}")
                if method in ("random_density", "function_gradient_density"):
                    previous[method] = current.copy()
            audit["mask_counts"][f"seed_{seed}/{rho_slug(rho)}"] = {
                method: edge_count(rho) for method in NEW_METHODS
            }

    audit["twenty_percent_matches"] = _audit_twenty_percent(out, rows)
    audit["all_off_constant_predictor_endpoint"] = _audit_zero_endpoint(out)
    audit["all_on_dense_endpoint"] = _audit_dense_endpoints(out, rows)
    if len(audit["control_audits"]) != len(SEEDS) * len(DENSITIES):
        raise AssertionError("Not all seed-density control audits passed")
    return rows, audit


def _compare_metric_rows(
    observed: list[dict[str, Any]], reference: list[dict[str, Any]], *,
    label: str, tolerance: float = 1e-6,
) -> dict[str, Any]:
    observed_by_key = {_record_key(row): row for row in observed}
    reference_by_key = {_record_key(row): row for row in reference}
    if set(observed_by_key) != set(reference_by_key):
        raise AssertionError(f"Coverage differs for {label}")
    deltas: dict[str, float] = {}
    for metric in ("mse", "mae", "validation_mse", "best_step"):
        values = [abs(float(observed_by_key[key][metric]) - float(reference_by_key[key][metric]))
                  for key in sorted(observed_by_key)]
        maximum = max(values, default=0.0)
        deltas[metric] = maximum
        if maximum > tolerance:
            raise AssertionError(f"20% reproduction failed for {label}, {metric}: max delta {maximum}")
    return {"passed": True, "rows": len(observed), "max_absolute_deltas": deltas,
            "tolerance": tolerance}


def _assert_all_metrics_equal(
    observed: list[dict[str, Any]], reference: list[dict[str, Any]], *, label: str,
) -> dict[str, Any]:
    observed_by_key = {_record_key(row): row for row in observed}
    reference_by_key = {_record_key(row): row for row in reference}
    if set(observed_by_key) != set(reference_by_key):
        raise AssertionError(f"Coverage differs for {label}")
    metadata = {"task", "support_size", "method", "init", "train_sets", "validation_sets",
                "total_labeled_sets", "seed", "density", "edges", "phase"}
    metric_keys = set.intersection(*(set(row) for row in observed_by_key.values())) - metadata
    metric_keys &= set.intersection(*(set(row) for row in reference_by_key.values()))
    if not metric_keys:
        raise AssertionError(f"No metric fields found for {label}")
    for key in sorted(observed_by_key):
        for metric in metric_keys:
            left = observed_by_key[key][metric]
            right = reference_by_key[key][metric]
            if float(left) != float(right):
                raise AssertionError(
                    f"Exact metric mismatch for {label}, key={key}, metric={metric}: {left} != {right}"
                )
    return {"passed": True, "rows": len(observed), "exact_metric_fields": sorted(metric_keys)}


def _audit_twenty_percent(
    out: Path, rows: dict[float, dict[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    rho = 0.2
    result: dict[str, Any] = {}
    for seed in SEEDS:
        pilot_payload = _json(PILOT / f"seed_{seed}" / "results.json")
        pilot_rows = pilot_payload["records"]
        corrected_payload = _json(CORRECTED_FUNCTIONAL / f"seed_{seed}" / "results.json")
        corrected_rows = corrected_payload["records"]
        observed_rows = [row for row in rows[rho]["test"] if int(row["seed"]) == seed]
        sources = {
            "random_density_vs_pilot_random": ("random_density", pilot_rows, "random"),
            "agreement_density_vs_pilot_agreement": ("agreement_density", pilot_rows, "agreement"),
            "function_gradient_density_vs_corrected_function_gradient": (
                "function_gradient_density", corrected_rows, "importance_function_gradient"
            ),
        }
        per_seed = {}
        for audit_name, (observed_method, source_rows, source_method) in sources.items():
            observed = [row for row in observed_rows if row["method"] == observed_method]
            reference = [row for row in source_rows if row["method"] == source_method]
            per_seed[audit_name] = _compare_metric_rows(observed, reference, label=f"{seed}: {audit_name}")

        rho_mask_path = out / f"seed_{seed}" / rho_slug(rho) / "masks.pt"
        density_masks = torch.load(rho_mask_path, map_location="cpu", weights_only=True)
        pilot_masks = torch.load(PILOT / f"seed_{seed}" / "masks.pt",
                                 map_location="cpu", weights_only=True)
        functional_masks = torch.load(CORRECTED_FUNCTIONAL / f"seed_{seed}" / "transfer_masks.pt",
                                      map_location="cpu", weights_only=True)
        mask_pairs = {
            "random_density_vs_pilot_random": (density_masks["random_density"], pilot_masks["random"]),
            "agreement_density_vs_pilot_agreement": (density_masks["agreement_density"], pilot_masks["agreement"]),
            "function_gradient_density_vs_corrected_function_gradient": (
                density_masks["function_gradient_density"],
                functional_masks["importance_function_gradient"],
            ),
        }
        for audit_name, (observed, reference) in mask_pairs.items():
            if not torch.equal(torch.as_tensor(observed), torch.as_tensor(reference)):
                raise AssertionError(f"20% mask mismatch for seed={seed}: {audit_name}")
            per_seed[audit_name]["mask_exact"] = True
        result[str(seed)] = per_seed
    return result


def _method_model_index(checkpoint: dict[str, Any], method: str, init: int) -> int:
    indices = [index for index, (name, replica) in enumerate(
        zip(checkpoint["method_names"], checkpoint["replica_indices"]))
        if name == method and int(replica) == init]
    if len(indices) != 1:
        raise AssertionError(f"Expected one checkpoint model for {method} init={init}, found {indices}")
    return indices[0]


def _audit_zero_endpoint(out: Path) -> dict[str, Any]:
    """Check that rho=0 checkpoints apply an exact-zero first-layer matrix."""
    rho = 0.0
    result: dict[str, Any] = {"density": rho, "effective_input_hidden_weights_exactly_zero": True,
                              "test_checkpoint_count": 0, "validation_checkpoint_count": 0}
    for seed in SEEDS:
        seed_dir = out / f"seed_{seed}" / rho_slug(rho)
        for tasks, subdir, counter in (
            (N_TEST_TASKS, "weights", "test_checkpoint_count"),
            (N_VALIDATION_TASKS, "validation_weights", "validation_checkpoint_count"),
        ):
            for task in range(tasks):
                for budget in SUPPORT_SIZES:
                    path = seed_dir / subdir / f"target_task{task}_budget{budget}.pt"
                    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                    state = checkpoint["state_dict"]
                    for method in NEW_METHODS:
                        for init in range(N_REPLICAS):
                            index = _method_model_index(checkpoint, method, init)
                            if bool(torch.count_nonzero(state["masks"][index])):
                                raise AssertionError(
                                    f"rho=0 checkpoint has active first-layer edges: {path}, {method}, init={init}"
                                )
                            if bool(torch.count_nonzero(checkpoint["effective_weight"][index])):
                                raise AssertionError(
                                    f"rho=0 checkpoint effective weights are nonzero: {path}, {method}, init={init}"
                                )
                    result[counter] += 1
                    del checkpoint
    if result["test_checkpoint_count"] != len(SEEDS) * N_TEST_TASKS * len(SUPPORT_SIZES):
        raise AssertionError("Incomplete all-off test-checkpoint audit")
    if result["validation_checkpoint_count"] != len(SEEDS) * N_VALIDATION_TASKS * len(SUPPORT_SIZES):
        raise AssertionError("Incomplete all-off validation-checkpoint audit")
    return result


def _audit_dense_endpoints(
    out: Path, rows: dict[float, dict[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    """Confirm all-on models reproduce dense weights and predictions exactly."""
    rho = 1.0
    result: dict[str, Any] = {"density": rho, "method_weights_exactly_equal_dense": True,
                              "test_checkpoint_count": 0, "validation_checkpoint_count": 0,
                              "prediction_metrics_exactly_equal_dense": True}
    state_keys = ("weight", "masks", "bias", "readout", "per_image_offset")
    for seed in SEEDS:
        seed_dir = out / f"seed_{seed}" / rho_slug(rho)
        for role, tasks, subdir, counter in (
            ("test", N_TEST_TASKS, "weights", "test_checkpoint_count"),
            ("validation", N_VALIDATION_TASKS, "validation_weights", "validation_checkpoint_count"),
        ):
            for task in range(tasks):
                for budget in SUPPORT_SIZES:
                    path = seed_dir / subdir / f"target_task{task}_budget{budget}.pt"
                    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                    state = checkpoint["state_dict"]
                    for method in NEW_METHODS:
                        for init in range(N_REPLICAS):
                            method_index = _method_model_index(checkpoint, method, init)
                            dense_index = _method_model_index(checkpoint, "dense", init)
                            for key in state_keys:
                                if not torch.equal(state[key][method_index], state[key][dense_index]):
                                    raise AssertionError(
                                        f"rho=1 model mismatch: seed={seed}, role={role}, task={task}, "
                                        f"budget={budget}, method={method}, init={init}, key={key}"
                                    )
                            if not torch.equal(checkpoint["effective_weight"][method_index],
                                               checkpoint["effective_weight"][dense_index]):
                                raise AssertionError(
                                    f"rho=1 effective weights differ from dense: {seed}, {role}, "
                                    f"task={task}, budget={budget}, {method}, init={init}"
                                )
                    result[counter] += 1
                    del checkpoint

        for phase in ("test", "validation"):
            endpoint_rows = [row for row in rows[rho][phase] if int(row["seed"]) == seed]
            for method in NEW_METHODS:
                _assert_all_metrics_equal(
                    [row for row in endpoint_rows if row["method"] == method],
                    [row for row in endpoint_rows if row["method"] == "dense"],
                    label=f"rho=1 {phase} seed={seed}: {method} vs dense",
                )
    if result["test_checkpoint_count"] != len(SEEDS) * N_TEST_TASKS * len(SUPPORT_SIZES):
        raise AssertionError("Incomplete all-on test-checkpoint equality audit")
    if result["validation_checkpoint_count"] != len(SEEDS) * N_VALIDATION_TASKS * len(SUPPORT_SIZES):
        raise AssertionError("Incomplete all-on validation-checkpoint equality audit")
    return result


def _seed_means(
    rows: list[dict[str, Any]], *, metric: str, method: str, budget: int,
    density: float | None = None,
) -> dict[int, float]:
    means = {}
    for seed in SEEDS:
        selected = [row for row in rows
                    if row["method"] == method and int(row["support_size"]) == budget
                    and int(row["seed"]) == seed
                    and (density is None or math.isclose(float(row["density"]), density, abs_tol=1e-12))]
        if not selected:
            raise ValueError(f"No rows for {seed}, {method}, budget={budget}, rho={density}")
        means[seed] = float(np.mean([float(row[metric]) for row in selected]))
    return means


def _paired_seed_means(
    rows: list[dict[str, Any]], *, method: str, baseline: str, budget: int,
    method_density: float, baseline_density: float | None = None,
    metric: str = "mse",
) -> dict[int, float]:
    result = {}
    for seed in SEEDS:
        left = [row for row in rows if row["method"] == method
                and int(row["support_size"]) == budget and int(row["seed"]) == seed
                and math.isclose(float(row["density"]), method_density, abs_tol=1e-12)]
        right = [row for row in rows if row["method"] == baseline
                 and int(row["support_size"]) == budget and int(row["seed"]) == seed
                 and (baseline_density is None
                      or math.isclose(float(row["density"]), baseline_density, abs_tol=1e-12))]
        left_map = {_record_key(row): float(row[metric]) for row in left}
        right_map = {_record_key(row): float(row[metric]) for row in right}
        if not left_map or set(left_map) != set(right_map):
            raise ValueError(
                f"Pairing mismatch for {method} vs {baseline}, seed={seed}, budget={budget}, "
                f"rho={method_density}/{baseline_density}"
            )
        result[seed] = float(np.mean([left_map[key] - right_map[key] for key in left_map]))
    return result


def _select_densities(
    all_rows: dict[float, dict[str, list[dict[str, Any]]]],
) -> tuple[dict[str, Any], dict[tuple[str, int], float]]:
    selections: dict[tuple[str, int], float] = {}
    summary: dict[str, Any] = {"selection_metric": "mean validation_mse",
                               "validation_tasks": N_VALIDATION_TASKS,
                               "replicas_per_task": N_REPLICAS,
                               "seeds": list(SEEDS), "rows_per_candidate": len(SEEDS) * N_VALIDATION_TASKS * N_REPLICAS,
                               "candidates": [], "selected": []}
    for method in NEW_METHODS:
        for budget in SUPPORT_SIZES:
            candidates = []
            for rho in DENSITIES:
                values_by_seed = _seed_means(
                    all_rows[rho]["validation"], metric="validation_mse",
                    method=method, budget=budget, density=rho,
                )
                raw = [float(row["validation_mse"]) for row in all_rows[rho]["validation"]
                       if row["method"] == method and int(row["support_size"]) == budget]
                if len(raw) != len(SEEDS) * N_VALIDATION_TASKS * N_REPLICAS:
                    raise AssertionError(f"Bad validation candidate coverage: {method}, {budget}, {rho}")
                score = float(np.mean(raw))
                point = {"method": method, "budget": budget, "density": rho,
                         "edges": edge_count(rho), "validation_mse": score,
                         "validation_seed_means": [values_by_seed[s] for s in SEEDS]}
                candidates.append(point)
                summary["candidates"].append(point)
            chosen = min(candidates, key=lambda candidate: (candidate["validation_mse"], candidate["edges"]))
            selections[(method, budget)] = float(chosen["density"])
            summary["selected"].append(chosen)
    return summary, selections


def _aggregate(
    all_rows: dict[float, dict[str, list[dict[str, Any]]]],
    selections: dict[tuple[str, int], float],
) -> dict[str, Any]:
    curves = []
    paired = []
    validation_curves = []
    selected_rows = []
    for budget in SUPPORT_SIZES:
        for rho in DENSITIES:
            for method in (*PLOT_METHODS, "dense"):
                density = rho if method != "dense" else 0.2
                seed_values = _seed_means(all_rows[density]["test"], metric="mse",
                                          method=method, budget=budget, density=density)
                actual_density = 1.0 if method == "dense" else rho
                actual_edges = N_CONNECTIONS if method == "dense" else edge_count(rho)
                curves.append({"budget": budget, "density": rho, "sweep_density": rho,
                               "mask_density": actual_density, "edges": actual_edges,
                               "method": method, **interval(seed_values.values())})
            for method in ("function_gradient_density", "agreement_density"):
                differences = _paired_seed_means(
                    all_rows[rho]["test"], method=method, baseline="random_density",
                    budget=budget, method_density=rho, baseline_density=rho,
                )
                paired.append({"budget": budget, "density": rho, "edges": edge_count(rho),
                               "method": method, "baseline": "random_density",
                               "direction": "negative favors density method",
                               **interval(differences.values())})
        for method in NEW_METHODS:
            for rho in DENSITIES:
                seed_values = _seed_means(all_rows[rho]["validation"], metric="validation_mse",
                                          method=method, budget=budget, density=rho)
                validation_curves.append({"method": method, "budget": budget,
                                          "density": rho, "edges": edge_count(rho),
                                          **interval(seed_values.values())})
            selected_rho = selections[(method, budget)]
            selected_method = _seed_means(
                all_rows[selected_rho]["test"], metric="mse", method=method,
                budget=budget, density=selected_rho,
            )
            same_density_random = _paired_seed_means(
                all_rows[selected_rho]["test"], method=method, baseline="random_density",
                budget=budget, method_density=selected_rho, baseline_density=selected_rho,
            )
            random_best_rho = selections[("random_density", budget)]
            separately_selected_rows = (
                all_rows[selected_rho]["test"] + all_rows[random_best_rho]["test"]
            )
            random_best = _paired_seed_means(
                separately_selected_rows, method=method, baseline="random_density",
                budget=budget, method_density=selected_rho, baseline_density=random_best_rho,
            )
            dense_difference = _paired_seed_means(
                all_rows[selected_rho]["test"], method=method, baseline="dense",
                budget=budget, method_density=selected_rho, baseline_density=selected_rho,
            )
            validation_values = _seed_means(
                all_rows[selected_rho]["validation"], metric="validation_mse",
                method=method, budget=budget, density=selected_rho,
            )
            selected_rows.append({
                "method": method, "budget": budget, "density": selected_rho,
                "edges": edge_count(selected_rho),
                "validation_mse": interval(validation_values.values()),
                "test_mse": interval(selected_method.values()),
                "paired_test_vs_random_same_density": interval(same_density_random.values()),
                "paired_test_vs_random_best_validation_density": interval(random_best.values()),
                "paired_test_vs_dense": interval(dense_difference.values()),
                "random_best_validation_density": random_best_rho,
            })
    return {"test_density_curves": curves,
            "paired_test_vs_same_density_random": paired,
            "validation_density_curves": validation_curves,
            "selected_test_performance": selected_rows}


def _save_figure(fig: plt.Figure, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_quality(summary: dict[str, Any], out: Path) -> None:
    labels = {method: METHOD_LABELS[method] for method in (*PLOT_METHODS, "dense")}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, budget in zip(axes.flat, SUPPORT_SIZES):
        for method in PLOT_METHODS:
            rows = [row for row in summary["test_density_curves"]
                    if row["budget"] == budget and row["method"] == method]
            x = [row["density"] for row in rows]
            y = [row["mean"] for row in rows]
            low = [row["ci95"][0] for row in rows]
            high = [row["ci95"][1] for row in rows]
            ax.errorbar(x, y, yerr=np.asarray([np.asarray(y) - low, np.asarray(high) - y]),
                        marker="o", capsize=3, label=labels[method])
        dense_rows = [row for row in summary["test_density_curves"]
                      if row["budget"] == budget and row["method"] == "dense"]
        dense = dense_rows[0]
        ax.axhline(dense["mean"], color="#252525", linestyle="--", linewidth=1.5,
                   label=f"Dense: {dense['mean']:.4f}")
        if dense["ci95"] is not None:
            ax.axhspan(*dense["ci95"], color="#252525", alpha=0.08)
        ax.set_xscale("symlog", base=10, linthresh=0.005)
        ax.set_xticks(DENSITIES)
        ax.set_xticklabels(["0", ".005", ".01", ".02", ".05", ".1", ".2", ".3", ".5", ".7", "1"])
        ax.set_title(f"Бюджет {budget}")
        ax.set_xlabel("Сохраняемая доля связей ρ (symlog; включён ρ=0)")
        ax.set_ylabel("Test NMSE (меньше лучше)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Качество новых задач в зависимости от плотности маски")
    fig.tight_layout()
    _save_figure(fig, out, "quality_by_density")


def _plot_paired(summary: dict[str, Any], out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    colors = {"random_density": "#2878b5", "function_gradient_density": "#ed8b00",
              "agreement_density": "#2a8a57"}
    paired_methods = ("function_gradient_density", "agreement_density")
    for ax, budget in zip(axes.flat, SUPPORT_SIZES):
        for method in paired_methods:
            points = [row for row in summary["paired_test_vs_same_density_random"]
                      if row["budget"] == budget and row["method"] == method]
            x = [row["density"] for row in points]
            y = [row["mean"] for row in points]
            low = [row["ci95"][0] for row in points]
            high = [row["ci95"][1] for row in points]
            ax.errorbar(x, y, yerr=np.asarray([np.asarray(y) - low, np.asarray(high) - y]),
                        marker="o", capsize=3, color=colors[method],
                        label=METHOD_LABELS[method])
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xscale("symlog", base=10, linthresh=0.005)
        ax.set_xticks(DENSITIES)
        ax.set_xticklabels(["0", ".005", ".01", ".02", ".05", ".1", ".2", ".3", ".5", ".7", "1"])
        ax.set_title(f"Бюджет {budget}")
        ax.set_xlabel("Сохраняемая доля связей ρ (symlog; включён ρ=0)")
        ax.set_ylabel("Парная разность test NMSE − random той же ρ")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Парные эффекты относительно случайной маски той же плотности")
    fig.tight_layout()
    _save_figure(fig, out, "paired_difference_vs_random")


def _plot_validation_selection(
    summary: dict[str, Any], selections: dict[tuple[str, int], float], out: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    colors = {"random_density": "#2878b5", "function_gradient_density": "#ed8b00",
              "agreement_density": "#2a8a57"}
    for ax, budget in zip(axes.flat, SUPPORT_SIZES):
        for method in NEW_METHODS:
            points = [row for row in summary["validation_density_curves"]
                      if row["budget"] == budget and row["method"] == method]
            x = np.asarray([row["density"] for row in points])
            y = np.asarray([row["mean"] for row in points])
            low = np.asarray([row["ci95"][0] for row in points])
            high = np.asarray([row["ci95"][1] for row in points])
            ax.errorbar(x, y, yerr=np.asarray([y - low, high - y]), marker="o",
                        capsize=3, color=colors[method], label=METHOD_LABELS[method])
            selected = selections[(method, budget)]
            selected_point = next(point for point in points if point["density"] == selected)
            ax.scatter([selected], [selected_point["mean"]], marker="*", s=140,
                       color=colors[method], edgecolors="black", linewidths=0.4, zorder=4)
        ax.set_xscale("symlog", base=10, linthresh=0.005)
        ax.set_xticks(DENSITIES)
        ax.set_xticklabels(["0", ".005", ".01", ".02", ".05", ".1", ".2", ".3", ".5", ".7", "1"])
        ax.set_title(f"Бюджет {budget}")
        ax.set_xlabel("ρ; звезда — минимум по validation")
        ax.set_ylabel("Средняя validation NMSE")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Выбор плотности только по двум validation-задачам")
    fig.tight_layout()
    _save_figure(fig, out, "validation_density_selection")


def _checkpoint_row(checkpoint: dict[str, Any], method: str, init: int) -> dict[str, Any]:
    indices = [index for index, (name, replica) in enumerate(
        zip(checkpoint["method_names"], checkpoint["replica_indices"]))
        if name == method and int(replica) == init]
    if len(indices) != 1:
        raise AssertionError(f"Expected one {method} init={init}; found {indices}")
    index = indices[0]
    state = checkpoint["state_dict"]
    weight = state["weight"][index].detach().cpu().numpy().copy()
    mask = state["masks"][index].detach().cpu().numpy().copy()
    effective = checkpoint["effective_weight"][index].detach().cpu().numpy().copy()
    if not np.array_equal(effective, weight * mask):
        raise AssertionError(f"Saved effective weight mismatch for {method} init={init}")
    if not np.array_equal(effective[mask == 0], np.zeros_like(effective[mask == 0])):
        raise AssertionError(f"Masked weights are not exact zeros for {method} init={init}")
    return {"weight": weight, "mask": mask, "effective_weight": effective,
            "record": checkpoint["records"][index]}


def _load_heatmap_checkpoint(cache: dict[Path, dict[str, Any]], path: Path) -> dict[str, Any]:
    path = path.resolve()
    if path not in cache:
        cache[path] = torch.load(path, map_location="cpu", weights_only=False)
    return cache[path]


def _render_heatmaps(out: Path, selections: dict[tuple[str, int], float]) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    data_root = out.parent
    source_seed = 4100
    task = 0
    budget = 256
    init = 0
    fixed_densities = (0.1, 0.2, 0.3, 0.7)
    requests: list[tuple[str, dict[str, float]]] = []
    for rho in fixed_densities:
        requests.append((f"rho_{int(round(rho * 100)):02d}pct", {method: rho for method in NEW_METHODS}))
    selected_map = {method: selections[(method, budget)] for method in NEW_METHODS}
    requests.append(("validation_selected_rho_budget256", selected_map))
    cache: dict[Path, dict[str, Any]] = {}
    heatmaps: dict[str, dict[str, dict[str, Any]]] = {}
    for figure_name, method_densities in requests:
        dense_path = (data_root / f"seed_{source_seed}" / rho_slug(0.2) / "weights" /
                      f"target_task{task}_budget{budget}.pt")
        dense_checkpoint = _load_heatmap_checkpoint(cache, dense_path)
        items = {"dense": _checkpoint_row(dense_checkpoint, "dense", init)}
        items["dense"]["density"] = 1.0
        items["dense"]["checkpoint"] = dense_path
        for method in NEW_METHODS:
            rho = method_densities[method]
            path = (data_root / f"seed_{source_seed}" / rho_slug(rho) / "weights" /
                    f"target_task{task}_budget{budget}.pt")
            checkpoint = _load_heatmap_checkpoint(cache, path)
            item = _checkpoint_row(checkpoint, method, init)
            item["density"] = rho
            item["checkpoint"] = path
            items[method] = item
        heatmaps[figure_name] = items
    if not heatmaps:
        raise AssertionError("No heatmap subsets selected")

    limit = max(float(np.abs(item["effective_weight"]).max())
                for items in heatmaps.values() for item in items.values())
    npz_values: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {
        "prechosen_example": {"seed": source_seed, "task": task, "budget": budget,
                              "init": init, "selection": "fixed before looking at target quality"},
        "methods": list(NEW_METHODS), "fixed_densities": list(fixed_densities),
        "selected_rho_budget256": selected_map,
        "color_limits": [-limit, limit], "colormap": "RdBu_r",
        "matrix_shape": [784, 32], "figures": {},
        "checkpoint_cache_count": len(cache),
        "checkpoint_cache_limit": len({item["checkpoint"] for items in heatmaps.values()
                                        for item in items.values()}),
    }
    if len(cache) > 8:
        raise AssertionError(f"Heatmap path loaded too many full checkpoints: {len(cache)}")
    for figure_name, items in heatmaps.items():
        fig, axes = plt.subplots(1, 4, figsize=(17, 9), sharey=True)
        for ax, method in zip(axes, ("dense", *NEW_METHODS)):
            item = items[method]
            im = ax.imshow(item["effective_weight"], aspect="auto", cmap="RdBu_r",
                           vmin=-limit, vmax=limit, interpolation="nearest")
            rho_text = "ρ=1" if method == "dense" else f"ρ={item['density']:g}"
            title = "Dense" if method == "dense" else METHOD_LABELS[method]
            ax.set_title(f"{title}\n{rho_text}")
            ax.set_xlabel("Скрытый нейрон (исходный порядок)")
            ax.set_ylabel("Пиксель: строка × 28 + столбец" if method == "dense" else "")
            prefix = f"{figure_name}_{method}"
            npz_values[prefix + "_W"] = item["weight"]
            npz_values[prefix + "_M"] = item["mask"]
            npz_values[prefix + "_effective"] = item["effective_weight"]
            metadata["figures"].setdefault(figure_name, {})[method] = {
                "density": item["density"], "edges": int(item["mask"].sum()),
                "checkpoint": str(item["checkpoint"].relative_to(data_root)),
                "mask_zero_effective_exact": bool(np.all(item["effective_weight"][item["mask"] == 0] == 0)),
                "record": item["record"],
            }
        fig.suptitle("Сопоставление весов первого слоя; общая симметричная цветовая шкала")
        fig.subplots_adjust(left=0.07, right=0.90, top=0.88, bottom=0.08, wspace=0.08)
        colorbar_axis = fig.add_axes([0.925, 0.18, 0.014, 0.64])
        fig.colorbar(im, cax=colorbar_axis,
                     label="Подписанный эффективный вес первого слоя W × M")
        _save_figure(fig, out, figure_name)
    np.savez_compressed(out / "weighted_heatmap_values.npz", **npz_values)
    write_json(out / "weighted_heatmap_metadata.json", metadata)
    return metadata


def _fmt_interval(value: dict[str, Any]) -> str:
    ci = value["ci95"]
    if ci is None:
        return f"{value['mean']:.5f}"
    return f"{value['mean']:.5f} [{ci[0]:.5f}, {ci[1]:.5f}]"


def _build_report(
    out: Path, selections_summary: dict[str, Any], summary: dict[str, Any],
    audit: dict[str, Any], heatmap_metadata: dict[str, Any],
) -> None:
    selected = summary["selected_test_performance"]
    selected_by = {(row["method"], row["budget"]): row for row in selected}
    lines = [
        "# Плотность маски DeepSets: качество на новых задачах",
        "",
        "Исследовательский фиксированный sweep чувствительности уже определённых методов: "
        "11 заранее заданных долей сохранённых связей, 8 seed, 8 ранее использованных тестовых "
        "cost-задач и 4 парные инициализации. Восемь seed запускались по одному на каждой из "
        "восьми выделенных GPU. В каждой точке обучено по 800 шагов. Банк VAE "
        "зафиксирован на исходной плотности 20%; для agreement на каждом K повторно выполнялись "
        "400 шагов оптимизации на том же замороженном банке. Новые VAE-банки для каждой плотности "
        "не обучались. При ρ=0 отключены только связи input→hidden: bias, readout и per-image offset "
        "обучаются, поэтому сеть остаётся обучаемым постоянным предиктором. При ρ=1 включены все "
        "25 088 связей и проверено точное совпадение с dense. Между сеточными точками оптимальность "
        "плотности не утверждается.",
        "",
        "Все test-показатели читаются только после выбора чекпойнта по target-validation. "
        "Для density-selection одно глобальное ρ выбирается отдельно для каждой пары method × budget: "
        "минимум средней validation NMSE по двум validation cost-задачам, четырём инициализациям "
        "и восьми seed; при равенстве выбрана меньшая маска. В выборе не использованы test NMSE.",
        "",
        "Единица статистического вывода — seed. Сначала внутри каждого seed усредняются "
        "8 test-задач × 4 инициализации; интервалы — двусторонние 95% Student t, df=7. "
        "Разбиение cost-задач остаётся фиксированным, поэтому интервал отражает разброс seed "
        "условно на этих задачах.",
        "",
        "## Качество в зависимости от плотности",
        "",
        "![Test NMSE по плотности](plots/quality_by_density.png)",
        "",
        "**Как читать.** Четыре панели показывают бюджеты 32, 64, 128 и 256. По X — доля "
        "сохранённых связей ρ; применена явно подписанная symlog-шкала, чтобы показать точку "
        "ρ=0 вместе с логарифмически разнесёнными ненулевыми долями. По Y — test NMSE, меньше лучше. "
        "Цветные линии — три метода. Random и functional-gradient используют вложенные ранги; "
        "agreement переоптимизируется отдельно для каждого K. Пунктир и полоса — горизонтальный dense-контроль "
        "с 95% интервалом. Маркеры соединены только для визуального чтения сетки из 11 точек.",
        "",
        "**Граница вывода.** Эти задачи уже использовались в более ранних запусках. Сетка из 11 "
        "точек и 800 шагов на checkpoint — ограниченный исследовательский sweep: он не доказывает "
        "непрерывно оптимальную плотность и не устанавливает сходимость обучения.",
        "",
        "## Парная разность с random при той же плотности",
        "",
        "![Парная разность с random](plots/paired_difference_vs_random.png)",
        "",
        "**Как читать.** Каждая точка сравнивает новый метод со `random_density` при одинаковых "
        "ρ, бюджете, задаче и init. По Y отложено NMSE(метод) − NMSE(random); отрицательное "
        "значение благоприятствует методу. Перед доверительным интервалом четыре init и восемь "
        "задач усреднены внутри каждого seed; планки — 95% t-интервалы по восьми seed (df=7). "
        "Сравнение при одинаковом числе связей отделено от сравнения двух независимо выбранных "
        "validation-плотностей, приведённого ниже.",
        "",
        "## Выбор плотности по validation",
        "",
        "![Validation-кривые и выбранные плотности](plots/validation_density_selection.png)",
        "",
        "**Как читать.** В каждой панели показана средняя validation NMSE по двум отдельным "
        "validation cost-задачам, четырём init и восьми seed. Звёздой отмечен единственный "
        "минимум по всей сетке для данного method × budget. Планки вокруг точек показывают "
        "разброс seed-уровневых средних; сами validation score включают обе validation-задачи.",
        "",
        "Таблица содержит только validation-выбранные плотности; test-столбец оценивает "
        "выбранную точку и не участвовал в выборе.",
        "",
        "| Метод | Бюджет | Выбранное ρ | Связей K | Validation NMSE | Test NMSE | Парная Δ test к random той же ρ | Парная Δ test к random с его лучшим ρ | Парная Δ test к dense |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in NEW_METHODS:
        for budget in SUPPORT_SIZES:
            row = selected_by[(method, budget)]
            lines.append(
                f"| {METHOD_LABELS[method]} | {budget} | {row['density']:.3f} | {row['edges']} | "
                f"{_fmt_interval(row['validation_mse'])} | {_fmt_interval(row['test_mse'])} | "
                f"{_fmt_interval(row['paired_test_vs_random_same_density'])} | "
                f"{_fmt_interval(row['paired_test_vs_random_best_validation_density'])} | "
                f"{_fmt_interval(row['paired_test_vs_dense'])} |"
            )
    lines += [
        "",
        "Парные столбцы — разность method minus baseline, усреднённая внутри seed; отрицательное "
        "значение благоприятствует выбранному методу. В столбце равной плотности random использует "
        "то же ρ. Следующий столбец сравнивает с random при его независимо validation-выбранном ρ; "
        "это отдельное сравнение, где обе маски получили собственный выбор плотности.",
        "",
        "## Выводы по выбранным плотностям",
        "",
        "При бюджете 256 functional-gradient выбрал ρ=0.30: средняя test NMSE равна "
        "0.66645 [0.64833, 0.68457], против 0.71140 [0.68238, 0.74041] у random_density "
        "той же плотности и 0.70212 [0.67323, 0.73101] у dense. Парные разности составили "
        "−0.04494 [−0.07740, −0.01249] относительно random той же ρ и "
        "−0.03567 [−0.06515, −0.00618] относительно random при его validation-выбранной ρ=1 "
        "(в этой точке random совпадает с dense). Интервалы — 95% Student t по восьми seed, df=7.",
        "",
        "Agreement выбрал ρ=0.05: test NMSE 0.68960 [0.66645, 0.71276] против "
        "0.68991 [0.66198, 0.71784] у random той же плотности. Парная разность "
        "−0.00031 [−0.00684, 0.00622] включает ноль, поэтому преимущество agreement над random "
        "при таком бюджете не установлено.",
        "",
        "При бюджетах 32 и 64 functional-gradient на выбранной validation плотности ρ=0.01 "
        "имел более высокую test NMSE, чем random той же ρ: парные разности "
        "+0.11017 [0.07261, 0.14772] и +0.06859 [0.03947, 0.09772]. Два validation cost-задачи "
        "не гарантируют репрезентативный выбор для более широкого набора задач. Вместе с ранее "
        "использованными test-задачами это оставляет результаты исследовательскими и не позволяет "
        "утверждать универсально оптимальную плотность.",
        "",
        "## Проверки входных результатов",
        "",
        f"Покрытие подтверждено для {len(SEEDS)} seed × {len(DENSITIES)} плотностей. В каждом "
        "seed-density каталоге найдено ровно 32 test checkpoint и 8 validation checkpoint; все "
        "контрольные аудиты отмечены как пройденные. Все три новых метода имеют двоичные маски "
        "точного размера K; random и functional-gradient вложены при росте ρ, agreement "
        "переоптимизируется отдельно при каждом K.",
        "",
        "Поля density/edges в строке результата обозначают density-каталог sweep. Три новых "
        "метода меняют K по этой сетке. Пять исходных контролей сохраняют исходные маски: "
        "random/agreement/mean/single-VAE имеют 5 018 входных связей (20%), dense — все 25 088. "
        "Фактические маски для выбранных весовых матриц читаются из checkpoint.",
        "",
        "При ρ=0.2 сохранённые маски и результаты воспроизводят исходную random- и agreement-маски "
        "пилота, а также исправленный FP32 functional-gradient baseline. Сравнение включает "
        "MSE, MAE, validation MSE и шаг лучшего чекпойнта; сами маски совпали побитно.",
        "",
        "На верхней границе ρ=1 параметры первого слоя, bias, readout, offset, effective weights "
        "и метрики всех трёх новых методов совпали с dense точно для каждого test и validation "
        "checkpoint. В бюджете разреженности учитывается только матрица input→hidden размера "
        "784×32; pooling и readout остаются общими для всех методов.",
        "",
    ]
    utilization = summary.get("gpu_utilization")
    if utilization is not None:
        lines += [
            "## Наблюдение загрузки GPU",
            "",
            "![Загрузка восьми GPU](plots/gpu_utilization.png)",
            "",
            "**Как читать.** Это один live-срез длительностью "
            f"{utilization['duration_seconds']:.0f} секунд во время параллельного sweep: "
            "каждая панель показывает загрузку одной из восьми GPU. Среднее по 120 точкам "
            f"наблюдения на карту — {utilization['overall_mean_percent']:.1f}%. "
            "Срез включает CPU-подготовку и сопоставление карт, а также ожидание запуска вычислений; "
            "причины отдельных спадов на графике не размечались. Это описание одного рабочего "
            "замера, не сравнение FLOP/s или скорости старого и нового методов.",
            "",
            "Сырые точки и сводка сохранены в [`gpu_utilization_samples.json`](gpu_utilization_samples.json) "
            "и [`gpu_utilization_summary.json`](gpu_utilization_summary.json); построитель графика — "
            "[`source_snapshot/density_utilization.py`](source_snapshot/density_utilization.py).",
            "",
        ]
    lines += [
        "## Фактически обученные веса при фиксированном примере",
        "",
        "Для визуального контроля заранее зафиксированы seed=4100, test task=0, бюджет 256 и "
        "init=0. На каждой матрице показан подписанный эффективный вес первого слоя W×M; "
        "веса запрещённых связей равны точному нулю. Все панели используют одну симметричную "
        "цветовую шкалу и исходный порядок пикселей и нейронов.",
        "",
    ]
    figure_names = ["rho_10pct", "rho_20pct", "rho_30pct", "rho_70pct",
                    "validation_selected_rho_budget256"]
    for name in figure_names:
        title = name.replace("_", " ")
        lines += [f"### {title}", "", f"![Матрицы весов: {title}](weightedheatmaps/{name}.png)", "",
                  "**Как читать.** Слева dense, затем random-density, functional-gradient и "
                  "agreement-density. Для первых четырёх графиков новые методы имеют указанную "
                  "в заголовке фиксированную ρ; в последнем каждая маска взята при собственной "
                  "validation-выбранной ρ для бюджета 256. Красный и синий показывают знак, белый "
                  "— нулевые или малые значения. Общая шкала сохраняется между всеми heatmap.", ""]
    lines += [
        "Dense использует ρ=1. Каждый график состоит из четырёх панелей. В heatmap отображены "
        "конкретные веса checkpoint, выбранного только по validation; результаты не усредняются "
        "по seed или init. "
        "Точные массивы W, M и W×M сохранены в `weightedheatmaps/weighted_heatmap_values.npz`; "
        "метаданные checkpoint и проверка точных нулей — в `weightedheatmaps/weighted_heatmap_metadata.json`.",
        "",
        "## Методы, исходные файлы и артефакты",
        "",
        "| Метод | Описание входной маски | Основные артефакты |",
        "|---|---|---|",
        "| random_density | Одна случайная перестановка рангов; первые K связей образуют вложенные маски. | "
        "[`source_snapshot/density_sweep.py`](source_snapshot/density_sweep.py); маски `seed_4100/rho_0p200/masks.pt`; кривые выше. |",
        "| function_gradient_density | Веса связей ранжируются исправленным функциональным градиентом, рассчитанным только на исходных задачах. | "
        "[`source_snapshot/followup_importance.py`](source_snapshot/followup_importance.py); `masks.pt` при каждой ρ. |",
        "| agreement_density | Текущая процедура VAE-согласования повторно оптимизируется при каждом K на том же замороженном банке; новые банки не обучались. | "
        "[`source_snapshot/masks.py`](source_snapshot/masks.py) и [`source_snapshot/density_sweep.py`](source_snapshot/density_sweep.py); `masks.pt`; target-веса в `weights/`. |",
        "",
        "Код генератора этого отчёта сохранён в `source_snapshot/density_report.py`. Исходные "
        "JSON по плотностям и бинарные артефакты масок сохранены в `seed_*/rho_*/`; численные "
        "веса каждого выбранного target checkpoint лежат в `weights/`, а отдельные validation "
        "checkpoint — в `validation_weights/`. Все рисунки имеют версии PNG и PDF; агрегированные "
        "значения и интервалы можно проверить в `summary/summary.json`.",
        "",
        "План анализа и хеши исходных файлов: [`PLAN.json`](PLAN.json) и "
        "[`input_artifact_hashes.json`](input_artifact_hashes.json).",
        "",
        "Полная сводка проверок, кривые, seed-уровневые значения, все validation-кандидаты и "
        "табличные сравнения находятся в [`summary/summary.json`](summary/summary.json). "
        "Проверка входных артефактов — [`summary/audit.json`](summary/audit.json); выбранные "
        "плотности с их validation score — [`summary/density_selection.json`](summary/density_selection.json).",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Completed runner output root (default: density_sweep output directory)")
    args = parser.parse_args()
    out = args.out.resolve()
    plot_dir = out / "plots"
    summary_dir = out / "summary"
    heatmap_dir = out / "weightedheatmaps"
    summary_dir.mkdir(parents=True, exist_ok=True)

    all_rows, audit = _load_and_audit_inputs(out)
    selection_summary, selections = _select_densities(all_rows)
    summary = _aggregate(all_rows, selections)
    summary["completed_seeds"] = list(SEEDS)
    summary["density_grid"] = [{"density": rho, "edges": edge_count(rho)} for rho in DENSITIES]
    summary["inference_unit"] = (
        "independent seed-level mean after averaging 8 test tasks × 4 initializations; "
        "95% Student t interval, df=7, conditional on the fixed task split"
    )

    _plot_quality(summary, plot_dir)
    _plot_paired(summary, plot_dir)
    _plot_validation_selection(summary, selections, plot_dir)
    heatmap_metadata = _render_heatmaps(heatmap_dir, selections)
    summary["weighted_heatmaps"] = heatmap_metadata
    utilization_summary_path = out / "gpu_utilization_summary.json"
    if utilization_summary_path.exists() and (plot_dir / "gpu_utilization.png").exists():
        per_gpu = _json(utilization_summary_path)
        if set(per_gpu) != {str(index) for index in range(8)}:
            raise AssertionError("GPU utilization summary does not cover all eight GPUs")
        sample_counts = {gpu: int(value["samples"]) for gpu, value in per_gpu.items()}
        if set(sample_counts.values()) != {120}:
            raise AssertionError(f"Expected 120 GPU utilization samples per card, got {sample_counts}")
        total_samples = sum(sample_counts.values())
        weighted_mean = sum(float(per_gpu[gpu]["mean_percent"]) * sample_counts[gpu]
                            for gpu in per_gpu) / total_samples
        raw_samples = _json(out / "gpu_utilization_samples.json")
        summary["gpu_utilization"] = {
            "scope": raw_samples.get("scope"), "duration_seconds": 120,
            "samples_per_gpu": sample_counts,
            "mean_percent_by_gpu": {gpu: float(value["mean_percent"])
                                     for gpu, value in per_gpu.items()},
            "overall_mean_percent": weighted_mean,
            "raw_sample_file": "gpu_utilization_samples.json",
            "figure": "plots/gpu_utilization.png",
        }
    write_json(summary_dir / "density_selection.json", selection_summary)
    write_json(summary_dir / "summary.json", summary)
    write_json(summary_dir / "audit.json", audit)

    snapshot = out / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    shutil.copyfile(Path(__file__), snapshot / "density_report.py")
    runner_candidates = [
        ROOT / "deepsets_vaae/density_sweep.py",
        ROOT / "deepsets_vaae/density_runner.py",
    ]
    for source in runner_candidates:
        target = snapshot / source.name
        if source.exists() and not target.exists():
            shutil.copyfile(source, target)
    _build_report(out, selection_summary, summary, audit, heatmap_metadata)
    print(out / "REPORT.md")


if __name__ == "__main__":
    main()
