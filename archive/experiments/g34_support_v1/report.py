"""Freeze first, score held-out answers second, then render the support study."""

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


SOURCE_METHODS = ("free",)
TARGET_METHODS = ("free",)
METHODS = ("current", "free", "dense")
LABELS = {
    "current": "G34 fixed mask",
    "free": "Joint zero / free mask",
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
        raise RuntimeError("Support study already frozen")
    registration = json.loads((base / "registration.json").read_text())
    reference = Path(registration["reference"])
    previous = reference.parent / "report/frozen_results.json"
    if digest(previous) != registration["reference_freeze_sha256"]:
        raise RuntimeError("Reference freeze changed")
    if digest(base / "PROTOCOL.md") != registration["protocol_sha256"]:
        raise RuntimeError("Registered protocol changed")
    for name, expected in registration["code_sha256"].items():
        if digest(PROJECT / name) != expected:
            raise RuntimeError(f"Registered code changed: {name}")
    amendment = json.loads((base / "scope_amendment.json").read_text())
    for name, expected in amendment["active_code_sha256"].items():
        if digest(PROJECT / name) != expected:
            raise RuntimeError(f"Active evaluation code changed: {name}")
    if digest(base / "ACTIVE_PROTOCOL.md") != amendment["active_protocol_sha256"]:
        raise RuntimeError("Active protocol changed")
    files = {
        base / "registration.json", base / "PROTOCOL.md",
        base / "scope_amendment.json", base / "ACTIVE_PROTOCOL.md",
    }
    external, predictions, rules = {previous}, set(), set()
    for kind in ("source", "target"):
        finished = base / f"{kind}_plan.finished.json"
        state = json.loads(finished.read_text())
        if state["completed"] != state["total"] or state["failures"]:
            raise RuntimeError(f"Incomplete plan: {kind}")
        files.update((finished, base / f"{kind}_plan.json"))
    for method in TARGET_METHODS:
        root = base / method
        files.update(root.rglob("*.json"))
        files.update(root.rglob("*.npz"))
        if method in SOURCE_METHODS:
            copied = json.loads((root / "source_export.json").read_text())
            for name, expected in copied["copied_sha256"].items():
                if digest(root / name) != expected:
                    raise RuntimeError(f"Source observation changed: {name}")
    protocol = json.loads((reference / "protocol.json").read_text())
    for sizes in protocol["cases"]:
        case = sizes["name"]
        for condition in protocol["conditions"]:
            key = f"{case}_c{condition}"
            for method in METHODS:
                folder = model_folder(base, reference, case, condition, method)
                path = folder / "frozen_predictions.npz"
                metric = json.loads((folder / "metrics.json").read_text())
                if (
                    metric["status"] != "complete"
                    or metric["predictions_sha256"] != digest(path)
                ):
                    raise RuntimeError(f"Incomplete/changed target: {folder}")
                predictions.add(path)
                record = json.loads((folder / "registered.json").read_text())
                rules.add((
                    record["code_sha256"], record["adapter_sha256"],
                    record["numeric_sha256"], record["adam_steps"],
                    record["restarts"], record["gauss_newton_starts"],
                    record["gauss_newton_steps"], tuple(record["budgets"]),
                ))
                # A changed parameter count must not change target examples.
                if method in TARGET_METHODS:
                    for name in ("support.npz", "selection.npz", "query_x.npz"):
                        original = reference / "targets" / key / name
                        if digest(folder.parent / name) != digest(original):
                            raise RuntimeError(f"Target observations differ: {name}")
                else:
                    external.update(folder.glob("*.json"))
                    external.update(folder.glob("*.npz"))
            external.update(
                reference / "fits" / key / name
                for name in (
                    "fitted.npz", "joint_bank.npz", "joint_bank.json",
                    "joint_fitted.npz", "joint_fitted.json",
                    "joint_selected.npz", "joint_selected.json",
                )
            )
            external.add(reference / "source" / f"{key}_t0" / "learned.npz")
            external.update(
                reference / "metrics" / f"{key}_{phase}.json"
                for phase in ("source", "seeds", "proposals", "fit", "select")
            )
    if len(rules) != 1:
        raise RuntimeError("Target adaptation protocols differ")
    external.update(reference / name for name in (
        "protocol.json", "data_commitments.json"
    ))
    files.update((base / "code_snapshot").rglob("*.py"))
    files.update((base / "scope_code_snapshot").rglob("*.*"))
    receipts = []
    for path in sorted(files | external):
        if path.parent.name == "receipts" or path.name == "access_receipt.json":
            record = json.loads(path.read_text())
            if record["violations"] or record["status"] != "complete":
                raise RuntimeError(f"Observation boundary failed: {path}")
            receipts.append(str(path))
    evaluation = [
        Path(__file__),
        PROJECT / "experiments/paper_structure_benchmark_v1/reporting/evaluate.py",
    ]
    for path in evaluation:
        target = base / "evaluation_code_snapshot" / path.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        files.add(target)
    save_json(manifest, {
        "frozen_utc": datetime.now(UTC).isoformat(),
        "files_sha256": {
            str(p.relative_to(base)): digest(p) for p in sorted(files)
        },
        "external_files_sha256": {str(p): digest(p) for p in sorted(external)},
        "prediction_count": len(predictions), "new_prediction_count": 12,
        "access_receipts": receipts, "violations": 0,
        "new_query_answers_opened": False, "new_structure_scores_opened": False,
    })
    print(json.dumps({
        "frozen_predictions": len(predictions), "receipts": len(receipts)
    }))


def mask_key(index):
    mask = (index != 0).astype(np.uint8)
    return mask[:, np.lexsort(mask[::-1])].tobytes()


def score(base):
    output = base / "report"
    frozen = json.loads((output / "frozen_results.json").read_text())
    for name, expected in frozen["files_sha256"].items():
        if digest(base / name) != expected:
            raise RuntimeError(f"Frozen file changed: {name}")
    for name, expected in frozen["external_files_sha256"].items():
        if digest(Path(name)) != expected:
            raise RuntimeError(f"Frozen reference changed: {name}")
    registration = json.loads((base / "registration.json").read_text())
    reference = Path(registration["reference"])
    protocol = json.loads((reference / "protocol.json").read_text())
    commitments = json.loads(
        (reference / "data_commitments.json").read_text()
    )["private_sha256"]
    for name, expected in commitments.items():
        if digest(reference / "private" / name) != expected:
            raise RuntimeError(f"Committed private answer changed: {name}")
    generation = json.loads(
        (reference / "private/generation.json").read_text()
    )
    metadata = {row["name"]: row for row in generation["cases"]}
    transfer, recovery, costs, geometry = [], [], [], []
    index_audit = {}
    for sizes in protocol["cases"]:
        case, codes = sizes["name"], sizes["codes"]
        truth = np.load(reference / "private" / f"{case}.npz")["index"]
        canonical = canonical_index(truth, codes)
        answer_file = np.load(reference / "private" / f"{case}_targets.npz")
        answer = np.stack([answer_file[f"query_y_{t}"] for t in range(8)])
        variance = np.maximum(np.var(answer, axis=(1, 2)), 1e-12)
        for condition in protocol["conditions"]:
            key = f"{case}_c{condition}"
            common = {
                "case": case, "variant": metadata[case]["variant"],
                "condition": condition,
            }
            dense_source = np.load(
                reference / "source" / f"{key}_t0" / "learned.npz"
            )["dense_parameters"]
            index_audit[key] = {
                "truth": truth.tolist(),
                "dense_source_t0_weight": dense_source[
                    :sizes["inputs"] * sizes["hidden"]
                ].reshape(sizes["inputs"], sizes["hidden"]).tolist(),
            }
            for method in METHODS:
                folder = model_folder(base, reference, case, condition, method)
                prediction = np.load(folder / "frozen_predictions.npz")
                model_sizes = json.loads((folder.parent / "sizes.json").read_text())
                model_codes = (
                    sizes["inputs"] * sizes["hidden"]
                    if method == "dense" else model_sizes["codes"]
                )
                for budget in (64, 256, 1024):
                    for stage, prefix in (
                        ("Adam", "adam_prediction"), ("Adam+GN", "prediction")
                    ):
                        errors = np.mean(
                            (prediction[f"{prefix}_{budget}"] - answer) ** 2,
                            axis=(1, 2),
                        )
                        for task, value in enumerate(errors):
                            transfer.append({
                                **common, "method": method, "stage": stage,
                                "support": budget, "task": task,
                                "mse": float(value),
                                "nmse": float(value / variance[task]),
                            })
                target_metric = json.loads((folder / "metrics.json").read_text())
                source_stages = {}
                if method == "current":
                    source_stages = {
                        phase: json.loads((
                            reference / "metrics" / f"{key}_{phase}.json"
                        ).read_text())["wall_seconds"]
                        for phase in ("proposals", "fit", "select")
                    }
                elif method in SOURCE_METHODS:
                    source_stages = json.loads((
                        base / method / "metrics" / f"{key}_pipeline.json"
                    ).read_text())["stage_seconds"]
                costs.append({
                    **common, "method": method,
                    "first_layer_parameters": model_codes,
                    "all_parameters": (
                        model_codes + sizes["hidden"]
                        + sizes["hidden"] * sizes["outputs"] + sizes["outputs"]
                    ),
                    "new_extraction_seconds": sum(source_stages.values()),
                    "target_seconds_all_budgets": target_metric["wall_seconds"],
                    "target_torch_operator_flops": target_metric[
                        "torch_operator_flops"
                    ],
                    "target_peak_allocated_bytes": target_metric[
                        "cuda_peak_allocated_bytes"
                    ],
                })
                if method not in ("current", *SOURCE_METHODS):
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
                proposals = np.load(fitted / "joint_bank.npz")["indices"]
                fitted_indices = np.load(fitted / "joint_fitted.npz")["indices"]
                initial_indices = np.load(fitted / "fitted.npz")["indices"]
                selected_record = json.loads(
                    (fitted / "joint_selected.json").read_text()
                )
                record = selected_record["methods"]["joint_functional"]
                refit_ids = set(json.loads(
                    (fitted / "joint_fitted.json").read_text()
                )["new_numeric_candidates"])
                true_pool = [
                    i for i, candidate in enumerate(proposals)
                    if np.array_equal(candidate, canonical)
                ]
                true_fit = [
                    i for i, candidate in enumerate(fitted_indices)
                    if np.array_equal(candidate, canonical)
                ]
                metrics.update({
                    "source_meta_mse": record["meta_mse"],
                    "proposal_count": len(proposals),
                    "true_in_proposal_pool": bool(true_pool),
                    "first_true_pool_rank": true_pool[0] + 1 if true_pool else None,
                    "true_mask_in_proposals": any(
                        mask_key(candidate) == mask_key(truth)
                        for candidate in proposals
                    ),
                    "true_in_available_refit_bank": bool(true_fit),
                    "true_joint_refit_requested": any(i in refit_ids for i in true_fit),
                    "selected_initial_anchor": any(
                        np.array_equal(index, candidate) for candidate in initial_indices
                    ),
                    "selected_in_proposal_pool": any(
                        np.array_equal(index, candidate) for candidate in proposals
                    ),
                    "selected_candidate": record["candidate"],
                    "true_active_connections": int((truth != 0).sum()),
                })
                recovery.append({**common, "method": method, **metrics})
                if method in SOURCE_METHODS:
                    runs = json.loads(
                        (fitted / "optimization.json").read_text()
                    )["runs"]
                    geometry.append({
                        **common, "method": method, "runs": len(runs),
                        "fraction_changed_support": float(np.mean([
                            r["final_support_changes"] > 0 for r in runs
                        ])),
                        "mean_initial_active": float(np.mean([
                            r["initial_active"] for r in runs
                        ])),
                        "mean_final_active": float(np.mean([
                            r["final_active"] for r in runs
                        ])),
                        "mean_support_flip_events": float(np.mean([
                            r["support_flip_events"] for r in runs
                        ])),
                    })
    payload = {
        "transfer": transfer, "recovery": recovery, "costs": costs,
        "geometry": geometry, "metadata": metadata,
        "scored_utc": datetime.now(UTC).isoformat(),
    }
    save_json(output / "scoring.json", payload)
    save_json(output / "index_audit.json", index_audit)
    for name, rows in (
        ("transfer", transfer), ("recovery", recovery), ("costs", costs),
        ("geometry", geometry),
    ):
        write_csv(output / f"{name}.csv", rows)
    print(json.dumps({name: len(payload[name]) for name in (
        "transfer", "recovery", "geometry"
    )}))


def render(base):
    # Presentation is kept separate from the frozen scoring implementation.
    from .render import render as render_report

    render_report(base)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--phase", choices=("freeze", "score", "render"), required=True)
    args = parser.parse_args()
    {"freeze": freeze, "score": score, "render": render}[args.phase](
        args.base.resolve()
    )


if __name__ == "__main__":
    main()
