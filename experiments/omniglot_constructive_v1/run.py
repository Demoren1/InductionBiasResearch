"""Apply saturated-pair recovery and cycle LPs to saved source weights."""

import argparse
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numba import njit

from experiments.constructive_dense_recovery_v1.recovery import (
    _fit_log_values,
    recover_dense_structure,
)
from experiments.omniglot_single_task_v1.extract import masks_for
from experiments.paper_structure_benchmark_v1.protocol.real_boundary import (
    StudyBoundary,
)
from experiments.sharing_structure_research_v1.blind_contract import digest

REFERENCE = Path("artifacts/omniglot_single_task_2026-11-14")
PRIMARY_BUDGETS = (8, 16, 32, 64)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default)
        + "\n"
    )


@njit
def _cover_count(values, radius):
    count, covered = 0, -np.inf
    for value in values:
        if value > covered:
            count += 1
            covered = value + radius
    return count


def local_error_lower_bound(logs, neurons, codes):
    """Minimal local interval radius, before imposing any shared codebook."""
    lower_bound = 0.0
    for neuron in np.unique(neurons):
        values = np.sort(logs[neurons == neuron])
        if len(np.unique(values)) <= codes:
            continue
        lower, upper = 0.0, float(np.ptp(values)) / 2
        for _ in range(55):
            middle = (lower + upper) / 2
            if _cover_count(values, 2 * middle) > codes:
                lower = middle
            else:
                upper = middle
        lower_bound = max(lower_bound, lower)
    return lower_bound


def sorted_contrasts(logs, neurons):
    differences = []
    for neuron in np.unique(neurons):
        local = np.sort(logs[neurons == neuron])
        left, right = np.triu_indices(len(local), 1)
        differences.append(local[right] - local[left])
    values = np.r_[0.0, np.sort(np.concatenate(differences))]
    gaps = np.diff(values)
    keep = min(len(gaps), 64 * 63 // 2 + 1)
    positions = np.argpartition(gaps, len(gaps) - keep)[-keep:]
    positions = positions[np.argsort(gaps[positions])[::-1]]
    return values, gaps, positions


def saturation_interval(values, gaps, positions, codes):
    """Check the entire half-open scalar tolerance interval for saturation."""
    required = codes * (codes - 1) // 2
    positive_gaps = int(np.count_nonzero(gaps > 0))
    result = {"codes": codes, "saturation_exists": False}
    if required > positive_gaps:
        return {**result, "reason": "count_below_saturation_at_zero"}
    upper_radius = float(gaps[positions[required - 1]])
    lower_radius = (
        0.0 if required == positive_gaps else float(gaps[positions[required]])
    )
    if lower_radius == upper_radius:
        return {**result, "reason": "gap_ties_skip_saturated_count"}
    cuts = np.sort(positions[gaps[positions] > lower_radius])
    if len(cuts) != required:
        raise RuntimeError("Scalar component count and gap interval disagree")
    starts = np.r_[0, cuts + 1]
    ends = np.r_[cuts, len(values) - 1]
    zero_radius = float(values[ends[0]])
    diameter = max(
        float(np.max(values[ends] - values[starts])), 2 * zero_radius
    )
    lower_epsilon, upper_epsilon = lower_radius / 4, upper_radius / 4
    minimum_epsilon = max(lower_epsilon, diameter / 4, zero_radius / 2)
    return {
        **result,
        "saturation_exists": True,
        "epsilon_lower_inclusive": lower_epsilon,
        "epsilon_upper_exclusive": upper_epsilon,
        "full_contrast_components": 2 * required + 1,
        "max_component_diameter": diameter,
        "zero_component_radius": zero_radius,
        "minimum_epsilon_for_width_checks": minimum_epsilon,
        "width_conditions_possible": minimum_epsilon < upper_epsilon,
        "width_violation_factor_at_upper_limit": (
            max(diameter / 4, zero_radius / 2) / upper_epsilon
        ),
        "reason": (
            "width_checks_allow_further_recovery"
            if minimum_epsilon < upper_epsilon
            else "width_checks_fail_on_entire_saturation_interval"
        ),
    }


def run(source_root, output):
    output.mkdir(parents=True, exist_ok=False)
    boundary = StudyBoundary(source_root)
    boundary.install()
    descriptor_path = REFERENCE / "extraction/descriptors.npz"
    descriptors = dict(np.load(descriptor_path))
    module = Path("experiments/constructive_dense_recovery_v1/recovery.py")
    save(
        output / "registration.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "source_root": str(source_root),
            "descriptors_sha256": digest(descriptor_path),
            "recovery_code_sha256": digest(module),
            "protocol_sha256": digest(Path(__file__).with_name("PROTOCOL.md")),
            "primary_budgets": PRIMARY_BUDGETS,
            "automatic_budgets": list(range(2, 65)),
            "source_only": True,
            "prior_target_results_already_known": True,
        },
    )
    snapshot = output / "code_snapshot"
    snapshot.mkdir()
    for path in Path(__file__).parent.iterdir():
        if path.is_file():
            shutil.copy2(path, snapshot / path.name)
    shutil.copy2(module, snapshot / "recovery_executed.py")
    started = time.perf_counter()
    records, candidates, metadata = [], {}, []
    for anchor in (60, 107):
        normal = descriptors[f"a{anchor}_normal"]
        uncertainty = descriptors[f"a{anchor}_uncertainty"]
        for mask_name, mask in masks_for(normal, uncertainty):
            rows, neurons = np.where(mask)
            logs = np.log(np.abs(normal[rows, neurons]))
            signs = np.sign(normal[rows, neurons])
            values, gaps, positions = sorted_contrasts(logs, neurons)
            intervals, native_calls = [], []
            for codes in range(2, 65):
                interval = saturation_interval(values, gaps, positions, codes)
                if codes in PRIMARY_BUDGETS:
                    interval["local_log_error_lower_bound"] = (
                        local_error_lower_bound(logs, neurons, codes)
                    )
                intervals.append(interval)
                if not interval["saturation_exists"]:
                    continue
                upper = np.nextafter(
                    interval["epsilon_upper_exclusive"], -np.inf
                )
                if interval["width_conditions_possible"]:
                    lower = interval["minimum_epsilon_for_width_checks"]
                    tolerances = sorted(
                        set((lower, (lower + upper) / 2, upper))
                    )
                elif codes in PRIMARY_BUDGETS:
                    tolerances = [upper]
                else:
                    continue
                for epsilon in tolerances:
                    result = recover_dense_structure(
                        logs, neurons, epsilon, max_codes=codes, signs=signs
                    )
                    native_calls.append(result)
                    if result["status"] != "complete":
                        continue
                    for family in result["families"]:
                        index = np.zeros(normal.shape, dtype=np.int16)
                        index[rows, neurons] = (
                            family["indexator"] + 1
                        ) * family["fixed_signs"]
                        name = f"candidate_{len(metadata):03d}"
                        candidates[name] = index
                        metadata.append(
                            {
                                "name": name,
                                "codes": codes,
                                "anchor": anchor,
                                "mask": mask_name,
                                "epsilon": epsilon,
                                "max_log_residual": family["max_log_residual"],
                                "free_neurons": family["free_neurons"],
                            }
                        )
            record = {
                "anchor": anchor,
                "mask": mask_name,
                "active": len(rows),
                "positive_contrast_observations": len(gaps),
                "omitted_squared_weight_fraction": float(
                    np.sum(normal[~mask] ** 2) / np.sum(normal**2)
                ),
                "intervals": intervals,
                "native_calls": native_calls,
            }
            records.append(record)
            save(output / "progress.json", {"masks": records})
            print(
                json.dumps(
                    {
                        "anchor": anchor,
                        "mask": mask_name,
                        "active": len(rows),
                        "native_calls": len(native_calls),
                        "width_eligible_budgets": [
                            r["codes"]
                            for r in intervals
                            if r.get("width_conditions_possible")
                        ],
                        "candidate_count": len(metadata),
                        "seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    np.savez_compressed(output / "candidates.npz", **candidates)
    save(output / "candidates.json", {"candidates": metadata})
    save(output / "recognition.json", {"masks": records})

    previous = json.loads(
        (REFERENCE / "extraction/candidates.json").read_text()
    )
    with np.load(REFERENCE / "extraction/candidates.npz") as indices:
        validations = []
        for row in previous["candidates"]:
            index = indices[row["name"]]
            normal = descriptors[f"a{row['anchor']}_normal"]
            rows, neurons = np.where(index)
            logs = np.log(np.abs(normal[rows, neurons]))
            labels = np.abs(index[rows, neurons]) - 1
            fit = _fit_log_values(
                logs[:, None], neurons, labels, row["codes"], 0
            )
            worst_cycle = 0.0
            for neuron in np.unique(neurons):
                for label in np.unique(labels[neurons == neuron]):
                    local = logs[(neurons == neuron) & (labels == label)]
                    worst_cycle = max(worst_cycle, float(np.ptp(local)) / 2)
            validations.append(
                {
                    "name": row["name"],
                    "codes": row["codes"],
                    "anchor": row["anchor"],
                    "mask": row["mask"],
                    "active": row["active"],
                    "optimal_max_log_error": fit.get("minimax_objective", 0.0),
                    "primal_max_log_residual": fit["max_log_residual"],
                    "two_edge_cycle_lower_bound": worst_cycle,
                    "multiplicative_log_error_factor": float(
                        np.exp(fit["max_log_residual"])
                    ),
                    "fit_method": fit["fit_method"],
                }
            )
    save(output / "previous_candidate_minimax.json", validations)
    save(
        output / "summary.json",
        {
            "status": "complete",
            "wall_seconds": time.perf_counter() - started,
            "masks": len(records),
            "budget_mask_combinations": 63 * len(records),
            "native_calls": sum(len(r["native_calls"]) for r in records),
            "complete_candidates": len(metadata),
            "width_eligible_combinations": sum(
                r.get("width_conditions_possible", False)
                for item in records
                for r in item["intervals"]
            ),
            "no_new_training_or_target_scoring": True,
        },
    )
    boundary.save(output / "access_receipt.json", "complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
