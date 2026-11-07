"""Attribute frozen prefix errors to codebook drift, label noise, or search."""

import argparse
import itertools
import json
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from experiments.darts_structured_tasks_v1.data import digest, save_json

from .shared_structure_data import load_source, teacher


def align_group_names(index, truth, codes):
    """Choose the same minimum-error group/sign gauge as the primary audit."""
    best = None
    for permutation in itertools.permutations(range(1, codes + 1)):
        for signs in itertools.product((-1, 1), repeat=codes):
            names = np.r_[0, np.asarray(permutation) * np.asarray(signs)]
            renamed = names[np.abs(index)] * np.sign(index)
            costs = (renamed[:, :, None] != truth[:, None, :]).sum(0)
            source, target = linear_sum_assignment(costs)
            errors = int(costs[source, target].sum())
            if best is None or errors < best[0]:
                best = errors, names
    return best


def rename(index, code, names):
    translated = names[np.abs(index)] * np.sign(index)
    values = np.empty_like(code)
    for old, new in enumerate(names[1:]):
        values[..., abs(new) - 1] = np.sign(new) * code[..., old]
    return translated, values


def hidden_errors(index, truth):
    costs = (index[:, :, None] != truth[:, None, :]).sum(0)
    first, second = linear_sum_assignment(costs)
    return int(costs[first, second].sum())


def predict(x, index, code, plan):
    weight = np.r_[0.0, code][np.abs(index)] * np.sign(index)
    return np.maximum(x @ weight + plan["bias"], 0).sum(-1) / math.sqrt(plan["hidden"])


def decompose(x, observed, wrong, truth, estimated, actual, plan):
    clean = predict(x, truth, actual, plan)
    true_estimated = predict(x, truth, estimated, plan)
    wrong_estimated = predict(x, wrong, estimated, plan)
    difference = wrong_estimated - true_estimated
    drift = true_estimated - clean
    noise = observed - clean
    energy = float(np.mean(difference**2))
    code_term = float(2 * np.mean(drift * difference))
    noise_term = float(-2 * np.mean(noise * difference))
    direct = float(
        np.mean((wrong_estimated - observed) ** 2)
        - np.mean((true_estimated - observed) ** 2)
    )
    oracle_wrong = predict(x, wrong, actual, plan)
    oracle_gap = float(np.mean((oracle_wrong - observed) ** 2) - np.mean(noise**2))
    return {
        "structure_energy": energy,
        "codebook_drift_term": code_term,
        "noise_term": noise_term,
        "observed_mse_gap_wrong_minus_true": direct,
        "decomposition_absolute_error": abs(direct - energy - code_term - noise_term),
        "fixed_true_C_counterfactual_gap": oracle_gap,
        "wrong_training_mse": float(np.mean((wrong_estimated - observed) ** 2)),
        "true_training_mse_same_estimated_C": float(
            np.mean((true_estimated - observed) ** 2)
        ),
        "C_max_absolute_error_in_audit_gauge": float(np.max(abs(estimated - actual))),
    }


def diagnose(root):
    destination = root / "failure_diagnosis_v2"
    destination.mkdir(exist_ok=False)
    shutil.copy2(Path(__file__), destination / "executed_code.py")
    plan = json.loads((root / "protocol.json").read_text())
    audit = json.loads((root / "common/audit.json").read_text())
    if audit["selection_freeze_sha256"] != digest(
        root / "common/selection_frozen.json"
    ):
        raise RuntimeError("Primary audit refers to another frozen selection")
    # This opened diagnostic examines the one remaining prefix sharing failure.
    family_id, sigma = 4, 2.0
    truth, actual_codes = teacher(plan, plan["families"][family_id])
    records, panel_cache = [], {}
    for job_id, job in enumerate(plan["extract_jobs"]):
        if (
            job["family_id"] != family_id
            or job["noise"] != sigma
            or job["method"] != "prefix"
        ):
            continue
        data = load_source(root, plan, family_id, job["task"], sigma, "cpu")
        panel_cache[job["task"]] = data
        path = root / "extract" / f"job_{job_id:04d}"
        metadata = json.loads(path.with_suffix(".json").read_text())
        order = data["order"]
        with np.load(path.with_suffix(".npz")) as arrays:
            for origin in metadata["records"]:
                trace = arrays[origin["key"] + "__trace_indices"]
                row = {
                    "source_job": job_id,
                    "task": job["task"],
                    "setting": origin["setting"],
                    "complete": origin["complete"],
                    "last_search_reason": origin["history"][-1]["reason"],
                    "accepted_prefixes": len(trace),
                }
                if not len(trace):
                    records.append(row)
                    continue
                terminal_truth = truth.copy()
                terminal_truth[order[len(trace) :]] = 0
                terminal_errors, names = align_group_names(
                    trace[-1], terminal_truth, plan["codes"]
                )
                row["last_prefix_categorical_errors"] = terminal_errors
                structural_errors = []
                for step, index in enumerate(trace, 1):
                    expected = truth.copy()
                    expected[order[step:]] = 0
                    errors, _ = align_group_names(index, expected, plan["codes"])
                    structural_errors.append(errors)
                    if not errors:
                        continue
                    code = np.asarray(origin["history"][step - 1]["codebook_before"])
                    axis = 2 * plan["inputs"] * plan["axis_points"]
                    begin = axis + (step - 1) * plan["slice_points"]
                    end = begin + plan["slice_points"]
                    x = data["train_x"][begin:end].numpy()
                    y = data[f"train_y_{sigma}"][begin:end].numpy()
                    previous = trace[step - 2] if step > 1 else np.zeros_like(index)
                    parent_truth = truth.copy()
                    parent_truth[order[step - 1 :]] = 0
                    completions = []
                    for permutation in itertools.permutations(
                        range(1, plan["codes"] + 1)
                    ):
                        for signs in itertools.product((-1, 1), repeat=plan["codes"]):
                            labels = np.r_[
                                0, np.asarray(permutation) * np.asarray(signs)
                            ]
                            candidate_parent, translated_code = rename(
                                previous, code, labels
                            )
                            if hidden_errors(candidate_parent, parent_truth):
                                continue
                            reference = predict(x, expected, translated_code, plan)
                            completions.append(
                                (float(np.mean((reference - y) ** 2)), labels)
                            )
                    if not completions:
                        raise RuntimeError(
                            "First structural failure has no correct parent gauge"
                        )
                    reference_mse, names = min(completions, key=lambda item: item[0])
                    renamed, fitted = rename(index, code, names)
                    terms = decompose(
                        x, y, renamed, expected, fitted, actual_codes[job["task"]], plan
                    )
                    native_mse = origin["history"][step - 1][
                        "slice_mse_integer_before_refit"
                    ]
                    row.update(
                        {
                            "first_wrong_prefix": step,
                            "first_wrong_input_coordinate": int(order[step - 1]),
                            "first_wrong_errors": errors,
                            "correct_parent_in_chosen_group_gauge": True,
                            "parent_compatible_group_gauges": len(completions),
                            "best_true_completion_mse": reference_mse,
                            "worst_true_completion_mse": max(
                                value[0] for value in completions
                            ),
                            "structural_errors_up_to_first_failure": structural_errors,
                            "categorical_errors_in_chosen_parent_gauge": hidden_errors(
                                renamed, expected
                            ),
                            "terms": terms,
                            "native_mse_reproduction_absolute_error": abs(
                                terms["wrong_training_mse"] - native_mse
                            ),
                        }
                    )
                    break
                records.append(row)

    selected_path = root / "common/selected/job_009"
    metadata = json.loads(selected_path.with_suffix(".json").read_text())
    chosen = next(
        row
        for row in metadata["records"]
        if row["method"] == "prefix" and row["scope"] == "common4"
    )
    with np.load(selected_path.with_suffix(".npz")) as arrays:
        index = arrays["prefix__common4__index"]
        code = arrays["prefix__common4__source_codes"]
    errors, names = align_group_names(index, truth, plan["codes"])
    renamed, fitted = rename(index, code, names)
    joint = []
    for task, data in sorted(panel_cache.items()):
        joint.append(
            {
                "task": task,
                **decompose(
                    data["train_x"].numpy(),
                    data[f"train_y_{sigma}"].numpy(),
                    renamed,
                    truth,
                    fitted[task],
                    actual_codes[task],
                    plan,
                ),
            }
        )
    result = {
        "kind": "Post-audit diagnosis on frozen G23 sharing failure; no new search/fits",
        "utc": datetime.now(UTC).isoformat(),
        "family": plan["families"][family_id],
        "noise": sigma,
        "source_protocol_sha256": digest(root / "protocol.json"),
        "source_observations_sha256": digest(root / "data/family_004_source.npz"),
        "source_selection_freeze_sha256": digest(root / "common/selection_frozen.json"),
        "code_sha256": digest(Path(__file__)),
        "theory_sha256": digest(Path(__file__).with_name("THEORY.md")),
        "prefix_diagnoses": records,
        "selected_errors": errors,
        "selected_candidate": chosen["selected_candidate"],
        "fixed_selected_C_joint_counterfactual": joint,
        "mean_joint_observed_gap": float(
            np.mean([r["observed_mse_gap_wrong_minus_true"] for r in joint])
        ),
        "limits": "Teacher I/C used only after all primary choices frozen. V2 first failure uses minimum errors over all group/sign/hidden permutations at each prefix. At that first structural failure, enumerate every group gauge consistent with the correct parent; compare to the feasible true completion with lowest observed loss at actual pre-refit C. Gauge/completion choice is an opened oracle diagnosis, never learner information. Joint full-I counterfactual uses terminal Hamming gauge as V1. No inference of global optimality from one pairwise comparison. No new labels, model updates or meta/test labels. V1 retained separately because its terminal group gauge could flag an earlier pure naming mismatch.",
    }
    save_json(destination / "decomposition.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.root)


if __name__ == "__main__":
    main()
