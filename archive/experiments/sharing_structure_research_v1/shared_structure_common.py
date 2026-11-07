"""G23 common-I banks, fresh signed C, and strictly later meta selection."""

import argparse
import hashlib
import itertools
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.darts_structured_tasks_v1.evaluate import align_indices

from .shared_structure_data import load_source, teacher
from .shared_structure_numeric import (
    canonical_index,
    code_prediction,
    features,
    fit_codes,
    fresh_codes,
)

BASE = Path(__file__).resolve().parent


def register(root):
    parent = json.loads((root / "protocol.json").read_text())
    if not (root / "extract_frozen.json").exists():
        raise RuntimeError("All source extractors must be frozen before common fits")
    common = root / "common"
    common.mkdir(exist_ok=False)
    names = ("shared_structure_common.py", "shared_structure_numeric.py")
    plan = {
        "registered_utc": datetime.now(UTC).isoformat(),
        "stage": "G23 common-bank/fresh-C/meta-selection; source extractors and source protocol unchanged",
        "source_protocol_sha256": digest(root / "protocol.json"),
        "source_extract_freeze_sha256": digest(root / "extract_frozen.json"),
        "data_freeze_sha256": digest(root / "data_frozen.json"),
        "settings": {
            key: parent[key]
            for key in (
                "seed",
                "numeric_code_steps",
                "numeric_ridge",
                "numeric_line_search",
                "bank_per_source",
                "meta_alpha",
            )
        },
        "numeric_fit": "Every unique canonicalI is fitted from32freshsignedstarts independently on4source tasks. Uniformmean-query trainingloss, numericrestart/checkpoint onfit-validation. No sourceC ormetaY. EquivalentI numericfits reusedacrossmethods withsameobservations/initials.",
        "selection": "Freeze all fittedpredictors beforeopeningmetaY. One commonI/method byaverage4taskmetaMSE. Source1ablation restrictsorigins andscores totask0. PairwiseGaussianriskintervals conditional onfrozenpredictors/metaX, not trueIcertificates.",
        "code_sha256": {name: digest(BASE / name) for name in names},
        "theory_sha256": digest(BASE / "THEORY.md"),
    }
    save_json(common / "protocol.json", plan)
    shutil.copy2(BASE / "THEORY.md", common / "theory_at_registration.md")
    snapshot = common / "executed_code_snapshot"
    snapshot.mkdir()
    for name in names:
        shutil.copy2(BASE / name, snapshot / name)
    for name in ("banks", "fits", "selected", "logs"):
        (common / name).mkdir()


def build_banks(root, plan):
    common = root / "common"
    phase = json.loads((common / "protocol.json").read_text())
    if phase["source_extract_freeze_sha256"] != digest(root / "extract_frozen.json"):
        raise RuntimeError("Source extraction freeze changed")
    frozen = json.loads((root / "extract_frozen.json").read_text())
    for relative, expected in frozen["sha256"].items():
        if digest(root / relative) != expected:
            raise RuntimeError("Frozen source extractor changed")
    jobs = {
        (job["family_id"], job["noise"], job["method"], job["task"]): number
        for number, job in enumerate(plan["extract_jobs"])
    }
    for family_id in range(len(plan["families"])):
        for noise_id, sigma in enumerate(plan["noise"]):
            job_id = family_id * len(plan["noise"]) + noise_id
            destination = common / "banks" / f"job_{job_id:03d}"
            if destination.with_suffix(".json").exists():
                raise RuntimeError("Candidate bank already assembled")
            indices, candidates, lookup = [], [], {}
            methods = {method: [] for method in plan["methods"]}
            for method in plan["methods"]:
                for task in range(plan["sources"]):
                    source_id = jobs[(family_id, sigma, method, task)]
                    source = root / "extract" / f"job_{source_id:04d}"
                    meta = json.loads(source.with_suffix(".json").read_text())
                    with np.load(source.with_suffix(".npz")) as arrays:
                        for record_id in meta["selected_records"]:
                            record = meta["records"][record_id]
                            index = arrays[record["key"] + "__canonical_I"]
                            signature = hashlib.sha256(index.tobytes()).hexdigest()
                            if signature not in lookup:
                                lookup[signature] = len(indices)
                                indices.append(index)
                                candidates.append(
                                    {"signature": signature, "origins": []}
                                )
                            position = lookup[signature]
                            candidates[position]["origins"].append(
                                {
                                    "method": method,
                                    "task": task,
                                    "source_job": source_id,
                                    "record": record_id,
                                    "fit_validation_mse": record["validation_mse"],
                                }
                            )
                            if position not in methods[method]:
                                methods[method].append(position)
            values = (
                np.stack(indices)
                if indices
                else np.zeros((0, plan["inputs"], plan["hidden"]), dtype=np.int16)
            )
            np.savez_compressed(destination.with_suffix(".npz"), indices=values)
            save_json(
                destination.with_suffix(".json"),
                {
                    "job_id": job_id,
                    "family_id": family_id,
                    "noise": sigma,
                    "candidates": candidates,
                    "methods": methods,
                    "meta_labels_opened": False,
                    "source_extract_freeze_sha256": phase[
                        "source_extract_freeze_sha256"
                    ],
                },
            )


@torch.no_grad()
def fit_job(root, plan, job_id, device):
    common = root / "common"
    phase_hash = digest(common / "protocol.json")
    path = common / "banks" / f"job_{job_id:03d}"
    bank = json.loads(path.with_suffix(".json").read_text())
    destination = common / "fits" / f"job_{job_id:03d}"
    if destination.with_suffix(".json").exists():
        raise RuntimeError("Common fit job already completed")
    destination.mkdir(exist_ok=True)
    with np.load(path.with_suffix(".npz")) as arrays:
        indices = arrays["indices"]
    family_id, sigma = bank["family_id"], bank["noise"]
    family = plan["families"][family_id]["family"]
    observed = [
        load_source(root, plan, family_id, task, sigma, device)
        for task in range(plan["sources"])
    ]
    artefacts = []
    for number, index in enumerate(indices):
        signature = bank["candidates"][number]["signature"]
        output = destination / signature
        if output.with_suffix(".json").exists():
            previous = json.loads(output.with_suffix(".json").read_text())
            if previous["phase_protocol_sha256"] != phase_hash or previous[
                "arrays_sha256"
            ] != digest(output.with_suffix(".npz")):
                raise RuntimeError(
                    "Existing numeric fit belongs to another phase or changed"
                )
            artefacts.append(str(output.relative_to(common)))
            continue
        started = time.monotonic()
        codes, histories = [], []
        for task, data in enumerate(observed):
            code, history = fit_codes(
                index,
                data["train_x"],
                data[f"train_y_{sigma}"],
                data["fit_validation_x"],
                data[f"fit_validation_y_{sigma}"],
                fresh_codes(plan, family, task, device),
                plan,
            )
            codes.append(code.cpu().numpy())
            histories.append(history)
        np.savez_compressed(
            output.with_suffix(".npz"), index=index, codes=np.stack(codes)
        )
        save_json(
            output.with_suffix(".json"),
            {
                "job_id": job_id,
                "candidate": number,
                "signature": signature,
                "numeric_fits": histories,
                "meta_labels_opened": False,
                "phase_protocol_sha256": phase_hash,
                "arrays_sha256": digest(output.with_suffix(".npz")),
                "seconds": time.monotonic() - started,
            },
        )
        artefacts.append(str(output.relative_to(common)))
        print(
            json.dumps(
                {
                    "common_job": job_id,
                    "candidate": number,
                    "of": len(indices),
                    "seconds": time.monotonic() - started,
                    "source_fit_validation": [r["validation_mse"] for r in histories],
                }
            ),
            flush=True,
        )
    save_json(
        destination.with_suffix(".json"),
        {
            "job_id": job_id,
            "bank_sha256": digest(path.with_suffix(".json")),
            "fit_artefacts": artefacts,
            "phase_protocol_sha256": phase_hash,
            "meta_labels_opened": False,
        },
    )


def freeze(root, plan, stage):
    common = root / "common"
    destination = common / (stage + "_frozen.json")
    if destination.exists():
        raise RuntimeError("Common stage already frozen")
    paths = []
    for job_id in range(len(plan["families"]) * len(plan["noise"])):
        if stage == "fit":
            bank = common / "banks" / f"job_{job_id:03d}"
            fitted = common / "fits" / f"job_{job_id:03d}.json"
            paths.extend((bank.with_suffix(".json"), bank.with_suffix(".npz"), fitted))
            meta = json.loads(fitted.read_text())
            for relative in meta["fit_artefacts"]:
                paths.extend(
                    (
                        (common / relative).with_suffix(".json"),
                        (common / relative).with_suffix(".npz"),
                    )
                )
        else:
            selected = common / "selected" / f"job_{job_id:03d}"
            paths.extend((selected.with_suffix(".json"), selected.with_suffix(".npz")))
    save_json(
        destination,
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "sha256": {str(path.relative_to(common)): digest(path) for path in paths},
            "phase_protocol_sha256": digest(common / "protocol.json"),
            "teacher_audit_opened": False,
            "meta_labels_opened": stage != "fit",
        },
    )


def risk_intervals(predictions, y, chosen, sigma, repeats, alpha):
    count, tasks, points = predictions.shape
    scores = (predictions - y[None]).square().mean((1, 2))
    if count == 1:
        return {
            "one_candidate": True,
            "strictly_best_design_risk_certified": False,
            "pairs": [],
        }
    quantile = float(norm.ppf(1 - alpha / (count * (count - 1))))
    rows = []
    for candidate in range(count):
        if candidate == chosen:
            continue
        delta = predictions[candidate] - predictions[chosen]
        standard_error = float(
            (4 * sigma**2 / repeats * delta.square().sum()).sqrt() / (tasks * points)
        )
        difference = float(scores[candidate] - scores[chosen])
        rows.append(
            {
                "candidate": candidate,
                "observed_risk_difference": difference,
                "standard_error": standard_error,
                "lower": difference - quantile * standard_error,
                "upper": difference + quantile * standard_error,
            }
        )
    return {
        "one_candidate": False,
        "alpha_scope": "one fixed method/family/noise bank, all unordered candidate pairs; conditional design risk of fitted predictors only, not true-I",
        "quantile": quantile,
        "strictly_best_design_risk_certified": all(r["lower"] > 0 for r in rows),
        "pairs": rows,
    }


@torch.no_grad()
def select_job(root, plan, job_id, device):
    common = root / "common"
    frozen = json.loads((common / "fit_frozen.json").read_text())
    destination = common / "selected" / f"job_{job_id:03d}"
    if destination.with_suffix(".json").exists():
        raise RuntimeError("Meta selection already completed")
    path = common / "banks" / f"job_{job_id:03d}"
    for suffix in (".json", ".npz"):
        current = path.with_suffix(suffix)
        if digest(current) != frozen["sha256"][str(current.relative_to(common))]:
            raise RuntimeError("Candidate bank changed after predictor freeze")
    bank = json.loads(path.with_suffix(".json").read_text())
    family_id, sigma = bank["family_id"], bank["noise"]
    data_path = root / "data" / f"family_{family_id:03d}_meta.npz"
    observations = json.loads((root / "data_frozen.json").read_text())
    if digest(data_path) != observations["sha256"][str(data_path.relative_to(root))]:
        raise RuntimeError("Frozen meta observations changed")
    with np.load(data_path) as arrays:
        x = torch.tensor(arrays["meta_x"], device=device)
        y = torch.tensor(arrays[f"meta_y_{sigma}"], device=device)
    with np.load(path.with_suffix(".npz")) as arrays:
        indices = arrays["indices"]
    codes, predictions = [], []
    for number, index in enumerate(indices):
        fitted = (
            common
            / "fits"
            / f"job_{job_id:03d}"
            / (bank["candidates"][number]["signature"] + ".npz")
        )
        if digest(fitted) != frozen["sha256"][str(fitted.relative_to(common))]:
            raise RuntimeError("Numeric predictor changed after freeze")
        with np.load(fitted) as arrays:
            code = torch.tensor(arrays["codes"], device=device)
        codes.append(code.cpu().numpy())
        predictions.append(
            code_prediction(features(x, index, plan["codes"]), code, plan)
        )
    predictions = torch.stack(predictions)
    task_scores = (predictions - y[None]).square().mean(-1)
    records, selected = [], {}
    for method in plan["methods"]:
        available = bank["methods"][method]
        for scope in ("source1", "common4"):
            candidates = [
                n
                for n in available
                if scope == "common4"
                or any(
                    origin["method"] == method and origin["task"] == 0
                    for origin in bank["candidates"][n]["origins"]
                )
            ]
            record = {
                "method": method,
                "scope": scope,
                "available_candidates": candidates,
                "complete": bool(candidates),
            }
            if candidates:
                objective = (
                    task_scores[candidates].mean(1)
                    if scope == "common4"
                    else task_scores[candidates, 0]
                )
                position = int(objective.argmin())
                winner = candidates[position]
                record.update(
                    {
                        "selected_candidate": winner,
                        "meta_mse": float(objective[position]),
                        "source_meta_mse": task_scores[winner].tolist(),
                    }
                )
                current_predictions = (
                    predictions[candidates]
                    if scope == "common4"
                    else predictions[candidates, :1]
                )
                current_y = y if scope == "common4" else y[:1]
                record["risk_intervals"] = risk_intervals(
                    current_predictions,
                    current_y,
                    position,
                    sigma,
                    plan["slice_repeats"],
                    plan["meta_alpha"],
                )
                selected[f"{method}__{scope}__index"] = indices[winner]
                selected[f"{method}__{scope}__source_codes"] = codes[winner]
            records.append(record)
    np.savez_compressed(
        destination.with_suffix(".npz"),
        **selected,
        candidate_indices=indices,
        candidate_codes=np.stack(codes),
        candidate_meta_predictions=predictions.cpu().numpy(),
        task_scores=task_scores.cpu().numpy(),
    )
    save_json(
        destination.with_suffix(".json"),
        {
            "job_id": job_id,
            "family_id": family_id,
            "noise": sigma,
            "records": records,
            "fit_freeze_sha256": digest(common / "fit_frozen.json"),
            "meta_observation_sha256": digest(data_path),
            "teacher_audit_opened": False,
        },
    )


def structural_metrics(index, truth, codes):
    """Separate mask, unsigned sharing, and full signed structure in late audit."""
    active, expected = index != 0, truth != 0
    mask_costs = (active[:, :, None] != expected[:, None, :]).sum(0)
    source, target = linear_sum_assignment(mask_costs)
    mask_errors = int(mask_costs[source, target].sum())
    best_unsigned = index.size
    for permutation in itertools.permutations(range(1, codes + 1)):
        renamed = np.r_[0, permutation][np.abs(index)]
        costs = (renamed[:, :, None] != np.abs(truth)[:, None, :]).sum(0)
        source, target = linear_sum_assignment(costs)
        best_unsigned = min(best_unsigned, int(costs[source, target].sum()))
    metric, _, _ = align_indices(index, truth, codes)
    full_exact = metric["categorical_errors"] == 0
    stage = (
        "exact"
        if full_exact
        else "mask"
        if mask_errors
        else "sharing"
        if best_unsigned
        else "signs"
    )
    return {
        **metric,
        "exact_I": full_exact,
        "exact_mask": mask_errors == 0,
        "mask_errors_best_hidden_permutation": mask_errors,
        "exact_unsigned_I": best_unsigned == 0,
        "unsigned_categorical_errors": best_unsigned,
        "structural_failure_stage": stage,
    }


def audit(root, plan):
    common = root / "common"
    frozen = json.loads((common / "selection_frozen.json").read_text())
    for relative, expected in frozen["sha256"].items():
        if digest(common / relative) != expected:
            raise RuntimeError("Frozen selected model changed")
    records, truth_arrays = [], {}
    for family_id, family in enumerate(plan["families"]):
        truth, _ = teacher(plan, family)
        canonical = canonical_index(truth, plan["codes"])
        truth_arrays[f"family_{family_id}__truth_I"] = truth
        for noise_id, sigma in enumerate(plan["noise"]):
            job_id = family_id * len(plan["noise"]) + noise_id
            path = common / "selected" / f"job_{job_id:03d}"
            meta = json.loads(path.with_suffix(".json").read_text())
            bank = json.loads((common / "banks" / f"job_{job_id:03d}.json").read_text())
            with np.load(path.with_suffix(".npz")) as arrays:
                for selection in meta["records"]:
                    method, scope = selection["method"], selection["scope"]
                    available = selection["available_candidates"]
                    record = {
                        "family_id": family_id,
                        "family": family,
                        "noise": sigma,
                        "method": method,
                        "scope": scope,
                        "complete": selection["complete"],
                        "exact_I": False,
                        "exact_mask": False,
                        "exact_unsigned_I": False,
                        "structural_failure_stage": "no_candidate",
                        "bank_contains_true_I": any(
                            np.array_equal(arrays["candidate_indices"][n], canonical)
                            for n in available
                        ),
                    }
                    if selection["complete"]:
                        index = arrays[f"{method}__{scope}__index"]
                        record.update(structural_metrics(index, truth, plan["codes"]))
                        record["meta_mse"] = selection["meta_mse"]
                        record["selected_origins"] = bank["candidates"][
                            selection["selected_candidate"]
                        ]["origins"]
                    records.append(record)
    np.savez_compressed(common / "audit_only.npz", **truth_arrays)
    save_json(
        common / "audit.json",
        {
            "records": records,
            "selection_freeze_sha256": digest(common / "selection_frozen.json"),
            "limits": "Source structural audit only; fresh-target adaptation and independent confirmation still required",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "register",
            "banks",
            "fit",
            "freeze_fit",
            "select",
            "freeze_selection",
            "audit",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = json.loads((args.root / "protocol.json").read_text())
    if args.action == "register":
        register(args.root)
    elif args.action == "banks":
        build_banks(args.root, plan)
    elif args.action in ("freeze_fit", "freeze_selection"):
        freeze(args.root, plan, args.action.removeprefix("freeze_"))
    elif args.action == "audit":
        if (args.root / "common/audit.json").exists():
            raise RuntimeError("Common-I audit already completed")
        audit(args.root, plan)
    else:
        for job_id in range(
            args.shard, len(plan["families"]) * len(plan["noise"]), args.shards
        ):
            (fit_job if args.action == "fit" else select_job)(
                args.root, plan, job_id, args.device
            )


if __name__ == "__main__":
    main()
