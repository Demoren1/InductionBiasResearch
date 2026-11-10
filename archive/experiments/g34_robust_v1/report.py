"""Freeze the ablation before inspecting its structural or query answers."""

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from experiments.paper_structure_benchmark_v1.reporting.evaluate import (
    adjusted_rand,
    align_index,
    write_csv,
)
from experiments.sharing_structure_research_v1.blind_contract import (
    canonical_index,
    digest,
    save_json,
)

from .run import PROJECT


METHODS = ("guard", "irls", "relative")
ALL_METHODS = ("current", *METHODS, "dense")
LABELS = {
    "current": "G34",
    "guard": "Q guard, unit weights",
    "irls": "IRLS, floor 0.01",
    "relative": "IRLS, relative residual",
    "dense": "Dense",
}


def model_folder(base, reference, case, condition, method):
    root = reference if method in ("current", "dense") else base / method
    key = f"{case}_c{0 if method == 'dense' else condition}"
    return root / "targets" / key / method


def freeze(base):
    output = base / "report"
    output.mkdir(exist_ok=True)
    manifest = output / "frozen_results.json"
    if manifest.exists():
        raise RuntimeError("This experiment has already been frozen")
    registration = json.loads((base / "registration.json").read_text())
    reference = Path(registration["reference"])
    previous_freeze = reference.parent / "report/frozen_results.json"
    if digest(previous_freeze) != registration["reference_freeze_sha256"]:
        raise RuntimeError("Reference prediction freeze changed")
    if digest(base / "PROTOCOL.md") != registration["protocol_sha256"]:
        raise RuntimeError("Registered protocol changed")
    for name, value in {
        **registration["core_sha256"],
        **registration["learner_sha256"],
    }.items():
        if digest(PROJECT / name) != value:
            raise RuntimeError(f"Registered learner changed: {name}")
    amendment = json.loads(
        (base / "relative_amendment.json").read_text()
    )
    for name, value in amendment["code_sha256"].items():
        if digest(PROJECT / name) != value:
            raise RuntimeError(f"Relative variant changed: {name}")
    for name in ("source", "target"):
        result = json.loads(
            (base / f"{name}_plan.finished.json").read_text()
        )
        if result["completed"] != result["total"] or result["failures"]:
            raise RuntimeError(f"Incomplete experiment plan: {name}")
    protocol = json.loads((reference / "protocol.json").read_text())
    files, external, predictions, rules = set(), set(), set(), set()
    for method in METHODS:
        root = base / method
        files.update(root.rglob("*.json"))
        files.update(root.rglob("*.npz"))
    for sizes in protocol["cases"]:
        case = sizes["name"]
        for condition in protocol["conditions"]:
            key = f"{case}_c{condition}"
            for method in ALL_METHODS:
                folder = model_folder(
                    base, reference, case, condition, method
                )
                prediction = folder / "frozen_predictions.npz"
                predictions.add(prediction)
                numeric = json.loads(
                    (folder / "registered.json").read_text()
                )
                rules.add(
                    (
                        numeric["code_sha256"],
                        numeric["adapter_sha256"],
                        numeric["numeric_sha256"],
                        numeric["adam_steps"],
                        numeric["restarts"],
                        numeric["gauss_newton_starts"],
                        numeric["gauss_newton_steps"],
                        tuple(numeric["budgets"]),
                    )
                )
                metric = json.loads((folder / "metrics.json").read_text())
                if metric["status"] != "complete":
                    raise RuntimeError(f"Incomplete target: {folder}")
                if metric["predictions_sha256"] != digest(prediction):
                    raise RuntimeError(f"Prediction changed: {prediction}")
                if method in ("current", "dense"):
                    external.update(folder.glob("*.json"))
                    external.update(folder.glob("*.npz"))
            external.update(
                reference / "fits" / key / name
                for name in (
                    "joint_bank.npz", "joint_bank.json",
                    "joint_fitted.npz", "joint_fitted.json",
                    "joint_selected.npz", "joint_selected.json",
                )
            )
            external.update(
                reference / "metrics" / f"{key}_{phase}.json"
                for phase in ("source", "seeds", "proposals", "fit", "select")
            )
    if len(rules) != 1:
        raise RuntimeError("Target numerical protocols differ")
    external.update(
        reference / name
        for name in ("data_commitments.json", "protocol.json")
    )
    external.add(previous_freeze)
    files.update(
        base / name
        for name in (
            "registration.json", "resource_amendment.json",
            "relative_amendment.json", "PROTOCOL.md",
            "source_plan.json", "source_plan.finished.json",
            "target_plan.json", "target_plan.finished.json",
        )
    )
    files.update((base / "code_snapshot").rglob("*.py"))
    files.update(
        output / name
        for name in ("calibration_states.csv", "calibration_states.json")
    )
    receipts = []
    for path in sorted(files | external):
        if (
            path.parent.name == "receipts"
            or path.name == "access_receipt.json"
        ):
            record = json.loads(path.read_text())
            if record["violations"] or record["status"] != "complete":
                raise RuntimeError(f"Observation boundary failed: {path}")
            receipts.append(str(path))
    hashes = {str(path.relative_to(base)): digest(path) for path in files}
    for method in METHODS:
        root = base / method
        copied = json.loads((root / "source_export.json").read_text())
        for name, expected in copied["copied_sha256"].items():
            key = str((root / name).relative_to(base))
            if hashes[key] != expected:
                raise RuntimeError(f"Observation export changed: {key}")
    evaluation_files = [
        Path(__file__),
        Path(__file__).with_name("diagnostic.py"),
        PROJECT / (
            "experiments/paper_structure_benchmark_v1/reporting/evaluate.py"
        ),
    ]
    for path in evaluation_files:
        target = (
            base / "evaluation_code_snapshot" / path.relative_to(PROJECT)
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        files.add(target)
        hashes[str(target.relative_to(base))] = digest(target)
    save_json(
        base / "evaluation_registration.json",
        {
            "utc": datetime.now(UTC).isoformat(),
            "code_sha256": {
                str(path.relative_to(PROJECT)): digest(path)
                for path in evaluation_files
            },
            "primary": registration["primary"],
            "retrospective_data": True,
            "new_target_answers_opened": False,
        },
    )
    hashes["evaluation_registration.json"] = digest(
        base / "evaluation_registration.json"
    )
    save_json(
        manifest,
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "files_sha256": hashes,
            "external_files_sha256": {
                str(path): digest(path) for path in sorted(external)
            },
            "prediction_count": len(predictions),
            "new_prediction_count": 12 * len(METHODS),
            "receipts": receipts,
            "violations": 0,
            "new_target_answers_opened": False,
            "new_structural_scores_opened": False,
        },
    )
    print(
        json.dumps(
            {
                "frozen": len(files) + len(external),
                "predictions": len(predictions),
                "new_predictions": 12 * len(METHODS),
                "receipts": len(receipts),
            }
        )
    )


def score(base):
    output = base / "report"
    frozen = json.loads((output / "frozen_results.json").read_text())
    for name, expected in frozen["files_sha256"].items():
        if digest(base / name) != expected:
            raise RuntimeError(f"Frozen result changed: {name}")
    for name, expected in frozen["external_files_sha256"].items():
        if digest(Path(name)) != expected:
            raise RuntimeError(f"Frozen baseline changed: {name}")
    registration = json.loads((base / "registration.json").read_text())
    reference = Path(registration["reference"])
    commitments = json.loads(
        (reference / "data_commitments.json").read_text()
    )["private_sha256"]
    protocol = json.loads((reference / "protocol.json").read_text())
    generation_path = reference / "private" / "generation.json"
    if digest(generation_path) != commitments["generation.json"]:
        raise RuntimeError("Private generation metadata changed")
    generation = json.loads(generation_path.read_text())
    metadata = {row["name"]: row for row in generation["cases"]}
    transfer, recovery, costs, geometry, paired = [], [], [], [], []
    index_audit = {}
    for sizes in protocol["cases"]:
        case, codes = sizes["name"], sizes["codes"]
        for name in (f"{case}.npz", f"{case}_targets.npz"):
            if digest(reference / "private" / name) != commitments[name]:
                raise RuntimeError(f"Private answer changed: {name}")
        with np.load(reference / "private" / f"{case}_targets.npz") as data:
            answer = np.stack([data[f"query_y_{t}"] for t in range(8)])
        truth = np.load(reference / "private" / f"{case}.npz")["index"]
        canonical = canonical_index(truth, codes)
        for condition in protocol["conditions"]:
            key = f"{case}_c{condition}"
            observed = np.load(
                base / "guard" / "source" / f"{key}_t0" / "learned.npz"
            )["dense_parameters"]
            index_audit[key] = {
                "truth": truth.tolist(),
                "dense_source_t0_weight": observed[
                    : sizes["inputs"] * sizes["hidden"]
                ].reshape(sizes["inputs"], sizes["hidden"]).tolist(),
            }
            common = {
                "case": case,
                "variant": metadata[case]["variant"],
                "condition": condition,
            }
            condition_runs = {}
            for method in ALL_METHODS:
                folder = model_folder(
                    base, reference, case, condition, method
                )
                prediction = np.load(folder / "frozen_predictions.npz")
                for budget in (64, 256, 1024):
                    for stage, prefix in (
                        ("Adam", "adam_prediction"),
                        ("Adam+GN", "prediction"),
                    ):
                        error = np.mean(
                            (prediction[f"{prefix}_{budget}"] - answer) ** 2,
                            axis=(1, 2),
                        )
                        variance = np.maximum(
                            np.var(answer, axis=(1, 2)), 1e-12
                        )
                        for task, value in enumerate(error):
                            transfer.append(
                                {
                                    **common, "method": method,
                                    "stage": stage, "support": budget,
                                    "task": task, "mse": float(value),
                                    "nmse": float(value / variance[task]),
                                }
                            )
                if method == "dense":
                    continue
                root = reference if method == "current" else base / method
                fitted = root / "fits" / key
                selected = np.load(fitted / "joint_selected.npz")
                index = selected["joint_functional_index"]
                aligned, metrics = align_index(index, truth, codes)
                both = (truth != 0) & (aligned != 0)
                metrics["sharing_ari"] = adjusted_rand(
                    np.abs(truth[both]), np.abs(aligned[both])
                )
                index_audit[key][method] = aligned.tolist()
                selected_record = json.loads(
                    (fitted / "joint_selected.json").read_text()
                )
                selected_json = selected_record["methods"][
                    "joint_functional"
                ]
                bank = np.load(fitted / "joint_bank.npz")
                refitted = np.load(fitted / "joint_fitted.npz")["indices"]
                true_ids = [
                    i for i, candidate in enumerate(refitted)
                    if np.array_equal(candidate, canonical)
                ]
                requested_ids = set(json.loads(
                    (fitted / "joint_fitted.json").read_text()
                )["new_numeric_candidates"])
                exact_ids = [
                    i for i, candidate in enumerate(bank["indices"])
                    if np.array_equal(candidate, canonical)
                ]
                metrics.update(
                    true_in_proposal_pool=bool(exact_ids),
                    first_true_pool_rank=(
                        exact_ids[0] + 1 if exact_ids else None
                    ),
                    true_in_refitted_bank=bool(true_ids),
                    true_joint_refit_requested=any(
                        i in requested_ids for i in true_ids
                    ),
                    true_meta_mse=(
                        min(
                            selected_record["all_meta_mse"][i]
                            for i in true_ids
                        )
                        if true_ids else None
                    ),
                    selected_candidate=selected_json["candidate"],
                    proposal_count=len(bank["indices"]),
                    refitted_bank_size=len(refitted),
                )
                recovery.append(
                    {
                        **common, "method": method,
                        "source_meta_mse": selected_json["meta_mse"],
                        **metrics,
                    }
                )
                shared_cost = sum(
                    json.loads(
                        (reference / "metrics" / f"{key}_{phase}.json")
                        .read_text()
                    )["wall_seconds"]
                    for phase in ("source", "seeds")
                )
                if method == "current":
                    stages = {
                        phase: json.loads(
                            (reference / "metrics" / f"{key}_{phase}.json")
                            .read_text()
                        )["wall_seconds"]
                        for phase in ("proposals", "fit", "select")
                    }
                    extraction = sum(stages.values())
                else:
                    metric = json.loads(
                        (root / "metrics" / f"{key}_pipeline.json")
                        .read_text()
                    )
                    stages = metric["stage_seconds"]
                    extraction = metric["wall_seconds"]
                    runs = json.loads(
                        (fitted / "optimization.json").read_text()
                    )["runs"]
                    condition_runs[method] = {
                        (run["anchor"], run["votes"], run["seed"]): run
                        for run in runs
                    }
                    differences = [
                        value
                        for run in runs
                        for value in np.diff(run["q"])
                    ]
                    geometry.append(
                        {
                            **common, "method": method, "runs": len(runs),
                            "q_steps": len(differences),
                            "q_increases_above_1e-10": int(
                                np.sum(np.asarray(differences) > 1e-10)
                            ),
                            "largest_q_increase": float(
                                max([0.0, *differences])
                            ),
                            "median_initial_q": float(np.median(
                                [run["q"][0] for run in runs]
                            )),
                            "median_final_q": float(np.median(
                                [run["q"][-1] for run in runs]
                            )),
                            "best_final_q": float(min(
                                run["q"][-1] for run in runs
                            )),
                            **{
                                field: sum(run[field] for run in runs)
                                for field in (
                                    "linear_solves",
                                    "full_q_evaluations",
                                    "accepted_numeric",
                                    "frozen_direction_rejected",
                                    "jacobian_fallback_accepted",
                                    "numeric_no_progress",
                                    "gauge_assignment_rejected",
                                )
                            },
                        }
                    )
                target_metric = json.loads(
                    (folder / "metrics.json").read_text()
                )
                costs.append(
                    {
                        **common, "method": method,
                        "shared_source_and_seeds_seconds": shared_cost,
                        "geometry_seconds": stages["proposals"],
                        "functional_fit_seconds": stages["fit"],
                        "extraction_seconds": extraction,
                        "source_total_seconds": shared_cost + extraction,
                        "target_seconds": target_metric["wall_seconds"],
                    }
                )
            guard, irls = [condition_runs[m] for m in ("guard", "irls")]
            if guard.keys() != irls.keys():
                raise RuntimeError(f"Unmatched projection starts: {key}")
            for identity, old in guard.items():
                new = irls[identity]
                if old["q"][0] != new["q"][0]:
                    raise RuntimeError("Calibration initialization differs")
                paired.append(
                    {
                        **common, "anchor": identity[0],
                        "votes": identity[1], "seed": identity[2],
                        "initial_q": old["q"][0],
                        "guard_q": old["q"][-1],
                        "irls_q": new["q"][-1],
                        "irls_minus_guard_q": new["q"][-1] - old["q"][-1],
                    }
                )
    result = {
        "scored_utc": datetime.now(UTC).isoformat(),
        "metadata": metadata, "noise": generation["noise"],
        "transfer": transfer, "recovery": recovery,
        "costs": costs, "geometry": geometry,
    }
    for name, rows in (
        ("transfer", transfer), ("recovery", recovery), ("costs", costs),
        ("geometry", geometry), ("paired_geometry", paired),
    ):
        write_csv(output / f"{name}.csv", rows)
    save_json(output / "index_audit.json", index_audit)
    save_json(output / "scoring.json", result)
    print(
        json.dumps(
            {
                "transfer_rows": len(transfer),
                "recovery_rows": len(recovery),
                "paired_geometry_runs": len(paired),
                "q_increases": sum(
                    row["q_increases_above_1e-10"] for row in geometry
                ),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--phase", choices=("freeze", "score"), required=True)
    args = parser.parse_args()
    operation = freeze if args.phase == "freeze" else score
    operation(args.base.resolve())


if __name__ == "__main__":
    main()
