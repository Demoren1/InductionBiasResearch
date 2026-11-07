"""Registered G24 diagnosis: aligned dictionaries and common integer prefixes."""

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json

from .count_projection import census as single_census
from .count_qp_active import function_qp
from .joint_prefix_search import census
from .shared_structure_common import audit, freeze, select_job
from .shared_structure_data import predict, teacher, weights_from_index
from .shared_structure_numeric import canonical_index

BASE = Path(__file__).resolve().parent
METHODS = [
    "bootstrap",
    "axis_single_refit",
    "bootstrap_single_fixed",
    "bootstrap_single_refit",
    "bootstrap_joint_fixed",
    "bootstrap_joint_refit",
]


def registration(root, parent):
    root.mkdir(exist_ok=False)
    source = json.loads((parent / "protocol.json").read_text())
    if not (parent / "common/fit_frozen.json").exists():
        raise RuntimeError("Bootstrap numeric fits are not frozen")
    names = (
        "joint_prefix_search.py",
        "joint_prefix_benchmark.py",
        "joint_prefix_pipeline.py",
        "joint_prefix_performance.py",
        "joint_prefix_report.py",
        "count_projection.py",
        "count_qp_active.py",
        "count_exchanges.py",
        "prefix_census.py",
        "shared_structure_numeric.py",
        "shared_structure_numeric_batch.py",
        "shared_structure_data.py",
        "shared_structure_common.py",
        "shared_structure_pipeline.py",
        "functional_observability.py",
    )
    jobs = []
    for family in range(len(source["families"])):
        for noise in source["noise"]:
            shared = {"family_id": family, "noise": noise}
            jobs.append(
                {
                    **shared,
                    "method": "bootstrap",
                    "tasks": list(range(4)),
                    "refit": False,
                    "solver": "none",
                }
            )
            for task in range(4):
                jobs.append(
                    {
                        **shared,
                        "method": "axis_single_refit",
                        "tasks": [task],
                        "refit": True,
                        "solver": "unit",
                    }
                )
            for joint in (False, True):
                for refit in (False, True):
                    method = f"bootstrap_{'joint' if joint else 'single'}_{'refit' if refit else 'fixed'}"
                    for solver in ("unit", "parent_pair"):
                        for tasks in (
                            [list(range(4))] if joint else [[task] for task in range(4)]
                        ):
                            jobs.append(
                                {
                                    **shared,
                                    "method": method,
                                    "tasks": tasks,
                                    "refit": refit,
                                    "solver": solver,
                                }
                            )
    plan = {
        **source,
        "registered_utc": datetime.now(UTC).isoformat(),
        "stage": "G24 opened diagnosis: bootstrap gauge, fresh independent noisy prefix panels, common counts",
        "parent": str(parent),
        "query_seed": 26110524017,
        "minimum_gpu_search_speedup": 2.0,
        "opened_geometries": True,
        "methods": METHODS,
        "extract_jobs": jobs,
        "parent_protocol_sha256": digest(parent / "protocol.json"),
        "parent_fit_freeze_sha256": digest(parent / "common/fit_frozen.json"),
        "code_sha256": {name: digest(BASE / name) for name in names},
        "theory_sha256": digest(BASE / "THEORY.md"),
        "bootstrap_policy": "Lowest mean old H among prefix-bank candidates; no old meta selection or teacher I/C. Freeze I/C before fresh query noise. New prefix search starts from empty and permits every0/±K connection/group.",
        "numeric_policy": "Original source train objective; final4C reset32signedstarts/GN24 with new GaussianH. Wholepredictorfreeze before freshmeta; wholeIselectionfreeze before lateaudit. Joint/frozen bootstrap have no source1 assessment.",
        "performance_policy": "Paired whole-census CPU/GPU timings per tasks/refit/solver and legacy-axis control on case9, prewarmed composition cache/context. GPU only with at least2x whole-census speedup and discrete/numeric parity; otherwise CPU reference. Only free GPU0--3; never6/7. Diagnostic results are not bank candidates.",
    }
    save_json(root / "protocol.json", plan)
    snapshot = root / "executed_code_snapshot"
    snapshot.mkdir()
    for name in names:
        shutil.copy2(BASE / name, snapshot / name)
    shutil.copy2(BASE / "THEORY.md", root / "theory_at_registration.md")
    for directory in ("bootstrap", "data", "extract", "logs"):
        (root / directory).mkdir()


def bootstrap(root, plan):
    parent = Path(plan["parent"])
    frozen = json.loads((parent / "common/fit_frozen.json").read_text())
    source_frozen = json.loads((parent / "extract_frozen.json").read_text())
    if digest(parent / "common/fit_frozen.json") != plan["parent_fit_freeze_sha256"]:
        raise RuntimeError("Old numeric freeze changed")
    lookup = {
        (job["family_id"], job["noise"], job["task"]): number
        for number, job in enumerate(
            json.loads((parent / "protocol.json").read_text())["extract_jobs"]
        )
        if job["method"] == "prefix"
    }
    outputs = []
    for family in range(len(plan["families"])):
        for noise_id, noise in enumerate(plan["noise"]):
            case = family * len(plan["noise"]) + noise_id
            destination = root / "bootstrap" / f"case_{case:03d}"
            if destination.with_suffix(".json").exists():
                raise RuntimeError("Bootstrap already assembled")
            bank_path = parent / "common/banks" / f"job_{case:03d}.json"
            if (
                digest(bank_path)
                != frozen["sha256"][str(bank_path.relative_to(parent / "common"))]
            ):
                raise RuntimeError("Frozen bootstrap bank changed")
            bank = json.loads(bank_path.read_text())
            scores = []
            for candidate in bank["methods"]["prefix"]:
                signature = bank["candidates"][candidate]["signature"]
                fitted = parent / "common/fits" / f"job_{case:03d}" / signature
                for suffix in (".json", ".npz"):
                    path = fitted.with_suffix(suffix)
                    if (
                        digest(path)
                        != frozen["sha256"][str(path.relative_to(parent / "common"))]
                    ):
                        raise RuntimeError("Frozen bootstrap candidate changed")
                meta = json.loads(fitted.with_suffix(".json").read_text())
                score = float(
                    np.mean(
                        [record["validation_mse"] for record in meta["numeric_fits"]]
                    )
                )
                scores.append((score, candidate, signature))
            if not scores:
                raise RuntimeError(
                    "No bootstrap candidate; record failure rather than use truth"
                )
            score, candidate, signature = min(scores)
            fitted = parent / "common/fits" / f"job_{case:03d}" / signature
            with np.load(fitted.with_suffix(".npz")) as arrays:
                index, codes = arrays["index"], arrays["codes"]
            axis = []
            for task in range(plan["sources"]):
                job = lookup[family, noise, task]
                source_path = parent / "extract" / f"job_{job:04d}.json"
                if (
                    digest(source_path)
                    != source_frozen["sha256"][str(source_path.relative_to(parent))]
                ):
                    raise RuntimeError("Axis-fit provenance changed")
                metadata = json.loads(source_path.read_text())
                axis.append(metadata["records"][0]["axis_fit"]["selected"]["codes"])
            np.savez_compressed(
                destination.with_suffix(".npz"),
                index=index,
                codes=codes,
                axis_codes=np.asarray(axis),
            )
            save_json(
                destination.with_suffix(".json"),
                {
                    "case": case,
                    "family_id": family,
                    "noise": noise,
                    "selected_candidate": candidate,
                    "signature": signature,
                    "mean_old_H_mse": score,
                    "selection_used_old_meta": False,
                    "scores": [
                        {"mean_H_mse": s, "candidate": n, "signature": sig}
                        for s, n, sig in scores
                    ],
                    "numeric_fit_metadata_sha256": digest(fitted.with_suffix(".json")),
                    "numeric_fit_arrays_sha256": digest(fitted.with_suffix(".npz")),
                    "arrays_sha256": digest(destination.with_suffix(".npz")),
                },
            )
            outputs.extend(
                (destination.with_suffix(".json"), destination.with_suffix(".npz"))
            )
    save_json(
        root / "bootstrap_frozen.json",
        {
            "utc": datetime.now(UTC).isoformat(),
            "sha256": {str(path.relative_to(root)): digest(path) for path in outputs},
            "fresh_query_labels_created": False,
            "old_meta_or_teacher_opened": False,
        },
    )


@torch.no_grad()
def prepare(root, plan, family):
    parent = Path(plan["parent"])
    old = parent / "data" / f"family_{family:03d}_source.npz"
    frozen = json.loads((parent / "data_frozen.json").read_text())
    if digest(old) != frozen["sha256"][str(old.relative_to(parent))]:
        raise RuntimeError("Original source observations changed")
    stem = root / "data" / f"family_{family:03d}"
    if stem.with_suffix(".json").exists():
        raise RuntimeError("Fresh queries already prepared")
    with np.load(old) as arrays:
        observed = {
            name: arrays[name] for name in ("train_x", "order", "train_repeats")
        }
        for sigma in plan["noise"]:
            observed[f"train_y_{sigma}"] = arrays[f"train_y_{sigma}"]
        axes = 2 * plan["inputs"] * plan["axis_points"]
        observed["search_train_x"] = arrays["train_x"][axes:]
        observed["prefix_validation_x"] = arrays["prefix_validation_x"]
    index, codes = teacher(plan, plan["families"][family])
    weight = weights_from_index(index, codes, "cpu")
    generator = torch.Generator().manual_seed(
        plan["query_seed"] + plan["families"][family]["family"] * 193
    )
    private = {}
    for name in ("search_train", "prefix_validation", "fit_validation", "meta"):
        if name + "_x" in observed:
            x = torch.tensor(observed[name + "_x"])
        else:
            count = (
                plan["fit_validation"]
                if name == "fit_validation"
                else plan["meta_points"]
            )
            x = torch.randn(
                (count, plan["inputs"]), dtype=torch.float64, generator=generator
            )
        clean = predict(x, weight, plan)
        noise = torch.randn(clean.shape, dtype=clean.dtype, generator=generator)
        destination = private if name == "meta" else observed
        destination[name + "_x"] = x.numpy()
        for sigma in plan["noise"]:
            destination[f"{name}_y_{sigma}"] = (
                clean + sigma / math.sqrt(plan["slice_repeats"]) * noise
            ).numpy()
    np.savez_compressed(stem.with_name(stem.name + "_source.npz"), **observed)
    np.savez_compressed(stem.with_name(stem.name + "_meta.npz"), **private)
    save_json(
        stem.with_suffix(".json"),
        {
            "family": plan["families"][family],
            "query_seed": plan["query_seed"],
            "teacher_seed_inherited": plan["seed"],
            "bootstrap_freeze_sha256": digest(root / "bootstrap_frozen.json"),
            "original_train_reused": True,
            "new_physical_labels_per_source_task_condition": plan["slice_repeats"]
            * (
                plan["inputs"] * (plan["slice_points"] + plan["prefix_validation"])
                + plan["fit_validation"]
                + plan["meta_points"]
            ),
            "new_noise_independent_of_bootstrap": True,
            "noise_conditions_paired": True,
        },
    )


def load_search(root, plan, job):
    case = job["family_id"] * len(plan["noise"]) + plan["noise"].index(job["noise"])
    source = root / "data" / f"family_{job['family_id']:03d}_source.npz"
    frozen = json.loads((root / "data_frozen.json").read_text())
    if digest(source) != frozen["sha256"][str(source.relative_to(root))]:
        raise RuntimeError("Fresh source observations changed")
    initial = root / "bootstrap" / f"case_{case:03d}.npz"
    expected = json.loads((root / "bootstrap_frozen.json").read_text())
    if digest(initial) != expected["sha256"][str(initial.relative_to(root))]:
        raise RuntimeError("Bootstrap changed after fresh queries")
    with np.load(initial) as arrays:
        index, codes, axis = arrays["index"], arrays["codes"], arrays["axis_codes"]
    with np.load(source) as arrays:
        observed = {
            name: arrays[name]
            for name in ("order", "search_train_x", "prefix_validation_x")
        }
        for name in ("search_train", "prefix_validation"):
            observed[name + "_y"] = arrays[f"{name}_y_{job['noise']}"]
    return observed, index, codes, axis


def compute_candidate(data, bootstrap_index, codes, axis, job, plan, device):
    if job["method"] == "bootstrap":
        result = {
            "complete": True,
            "indices": bootstrap_index[None],
            "codebooks": codes[None],
            "trace": [],
            "seconds": 0.0,
        }
    elif job["method"] == "axis_single_refit":
        started = time.monotonic()
        task = job["tasks"][0]
        x, vx, y, vy = [
            torch.tensor(data[name], device=device)
            for name in (
                "search_train_x",
                "prefix_validation_x",
                "search_train_y",
                "prefix_validation_y",
            )
        ]
        result = single_census(
            list(x.reshape(plan["inputs"], plan["slice_points"], plan["inputs"])),
            y[task].reshape(plan["inputs"], plan["slice_points"]),
            list(vx.reshape(plan["inputs"], plan["prefix_validation"], plan["inputs"])),
            vy[task].reshape(plan["inputs"], plan["prefix_validation"]),
            data["order"],
            axis[task],
            "qp_integer_refit",
            plan,
            qp_solver=function_qp,
        )
        result["seconds"] = time.monotonic() - started
    else:
        result = census(
            data, codes, job["tasks"], job["refit"], job["solver"], plan, device
        )
    return result


def search_job(root, plan, number, device):
    destination = root / "extract" / f"job_{number:04d}"
    if destination.with_suffix(".json").exists():
        raise RuntimeError("Scientific search job already complete")
    job = plan["extract_jobs"][number]
    data, bootstrap_index, codes, axis = load_search(root, plan, job)
    result = compute_candidate(data, bootstrap_index, codes, axis, job, plan, device)
    arrays = {"trace_indices": result["indices"], "trace_C": result["codebooks"]}
    if result["complete"]:
        arrays["index"] = canonical_index(result["indices"][-1], plan["codes"])
    np.savez_compressed(destination.with_suffix(".npz"), **arrays)
    record = {
        key: value
        for key, value in result.items()
        if key not in ("indices", "codebooks", "centers")
    }
    save_json(
        destination.with_suffix(".json"),
        {
            **record,
            "job": job,
            "job_id": number,
            "search_started_from_empty": job["method"] != "bootstrap",
            "frozen_bootstrap_control": job["method"] == "bootstrap",
            "protocol_sha256": digest(root / "protocol.json"),
            "arrays_sha256": digest(destination.with_suffix(".npz")),
            "teacher_or_meta_opened": False,
        },
    )
    print(
        json.dumps(
            {
                "job": number,
                "method": job["method"],
                "complete": result["complete"],
                "seconds": result["seconds"],
            }
        ),
        flush=True,
    )


def freeze_stage(root, plan, stage):
    destination = root / (
        "extract_frozen.json" if stage == "search" else "data_frozen.json"
    )
    if destination.exists():
        raise RuntimeError("Stage already frozen")
    paths = []
    if stage == "search":
        for job in range(len(plan["extract_jobs"])):
            path = root / "extract" / f"job_{job:04d}"
            paths.extend((path.with_suffix(".json"), path.with_suffix(".npz")))
    else:
        for family in range(len(plan["families"])):
            paths.extend(
                root / "data" / f"family_{family:03d}{suffix}"
                for suffix in (".json", "_source.npz", "_meta.npz")
            )
    save_json(
        destination,
        {
            "utc": datetime.now(UTC).isoformat(),
            "sha256": {str(path.relative_to(root)): digest(path) for path in paths},
            "source_protocol_sha256": digest(root / "protocol.json"),
            "meta_or_teacher_opened": False,
        },
    )


def banks(root, plan):
    for family in range(len(plan["families"])):
        for noise_id, noise in enumerate(plan["noise"]):
            case = family * len(plan["noise"]) + noise_id
            values, candidates, lookup = [], [], {}
            methods = {method: [] for method in METHODS}
            for number, job in enumerate(plan["extract_jobs"]):
                if job["family_id"] != family or job["noise"] != noise:
                    continue
                source = root / "extract" / f"job_{number:04d}"
                metadata = json.loads(source.with_suffix(".json").read_text())
                if not metadata["complete"]:
                    continue
                with np.load(source.with_suffix(".npz")) as arrays:
                    index = arrays["index"]
                signature = hashlib.sha256(index.tobytes()).hexdigest()
                if signature not in lookup:
                    lookup[signature] = len(values)
                    values.append(index)
                    candidates.append({"signature": signature, "origins": []})
                selected = lookup[signature]
                candidates[selected]["origins"].append(
                    {
                        "method": job["method"],
                        "task": job["tasks"][0] if len(job["tasks"]) == 1 else -1,
                        "source_job": number,
                        "record": 0,
                    }
                )
                if selected not in methods[job["method"]]:
                    methods[job["method"]].append(selected)
            path = root / "common/banks" / f"job_{case:03d}"
            if path.with_suffix(".json").exists():
                raise RuntimeError("Bank already assembled")
            np.savez_compressed(path.with_suffix(".npz"), indices=np.stack(values))
            save_json(
                path.with_suffix(".json"),
                {
                    "job_id": case,
                    "family_id": family,
                    "noise": noise,
                    "candidates": candidates,
                    "methods": methods,
                },
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "register",
            "bootstrap",
            "data",
            "freeze_data",
            "search",
            "freeze_search",
            "banks",
            "freeze_fit",
            "select",
            "freeze_selection",
            "audit",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", choices=("cpu", "cuda"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "register":
        registration(args.root, args.parent)
        return
    plan = json.loads((args.root / "protocol.json").read_text())
    for name, expected in plan["code_sha256"].items():
        if digest(BASE / name) != expected:
            raise RuntimeError(f"Registered code changed: {name}")
    if args.device.startswith("cuda") and os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    ) not in {"0", "1", "2", "3"}:
        raise RuntimeError("Only explicitly chosen GPU0--3 is permitted")
    if args.action == "bootstrap":
        bootstrap(args.root, plan)
    elif args.action == "data":
        for family in range(args.shard, len(plan["families"]), args.shards):
            prepare(args.root, plan, family)
    elif args.action in ("freeze_data", "freeze_search"):
        freeze_stage(args.root, plan, args.action.removeprefix("freeze_"))
    elif args.action == "search":
        performance = json.loads((args.root / "execution_measurement.json").read_text())
        eligible = []
        for number, job in enumerate(plan["extract_jobs"]):
            key = (
                "axis"
                if job["method"] == "axis_single_refit"
                else f"{len(job['tasks'])}/{int(job['refit'])}/{job['solver']}"
            )
            backend = (
                "cpu"
                if job["method"] == "bootstrap"
                else performance["cases"][key]["backend"]
            )
            if backend == args.backend:
                eligible.append(number)
        for number in eligible[args.shard :: args.shards]:
            search_job(args.root, plan, number, args.device)
    elif args.action == "banks":
        banks(args.root, plan)
    elif args.action in ("freeze_fit", "freeze_selection"):
        freeze(args.root, plan, args.action.removeprefix("freeze_"))
    elif args.action == "select":
        for case in range(
            args.shard, len(plan["families"]) * len(plan["noise"]), args.shards
        ):
            select_job(args.root, plan, case, args.device)
            path = args.root / "common/selected" / f"job_{case:03d}.json"
            result = json.loads(path.read_text())
            result["records"] = [
                record
                for record in result["records"]
                if record["scope"] != "source1"
                or record["method"]
                in (
                    "axis_single_refit",
                    "bootstrap_single_fixed",
                    "bootstrap_single_refit",
                )
            ]
            save_json(path, result)
    else:
        if (args.root / "common/audit.json").exists():
            raise RuntimeError("Teacher audit already completed")
        audit(args.root, plan)


if __name__ == "__main__":
    main()
