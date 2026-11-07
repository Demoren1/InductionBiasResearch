"""One registered independent confirmation of the frozen G24 algorithms."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json

from . import joint_prefix_benchmark as joint
from . import shared_structure_common as common
from . import shared_structure_transfer as transfer
from .shared_structure_data import prepare as prepare_source
from .shared_structure_extract import run as extract_source
from .shared_structure_numeric import fit_codes as serial_fit
from .shared_structure_numeric import fresh_codes
from .shared_structure_numeric_batch import fit_codes as batch_fit
from .shared_structure_pipeline import check_code, pid_identity
from .shared_structure_target_batch import fit_job as fit_target
from .shared_structure_target_batch import profile as profile_target

BASE = Path(__file__).resolve().parent
PACKAGE = "experiments.sharing_structure_research_v1."
METHODS = ("bootstrap", "bootstrap_single_fixed", "bootstrap_joint_fixed")


def status(root, stage, **details):
    record = {"utc": datetime.now(UTC).isoformat(), "stage": stage, **details}
    save_json(root / "confirmation_state.json", record)
    print(json.dumps(record), flush=True)


def register(root, pilot):
    root.mkdir(parents=True, exist_ok=False)
    old = json.loads((pilot / "protocol.json").read_text())
    original = json.loads((Path(old["parent"]) / "protocol.json").read_text())
    families = [
        {"family": number, "variant": variant}
        for variant, numbers in (
            ("local", range(11001, 11005)),
            ("quartet_mix", range(11011, 11015)),
            ("random", range(11021, 11025)),
        )
        for number in numbers
    ]
    names = (
        "joint_prefix_confirmation.py",
        "joint_prefix_confirmation_report.py",
        "joint_prefix_benchmark.py",
        "joint_prefix_search.py",
        "shared_structure_data.py",
        "shared_structure_extract.py",
        "shared_structure_native.py",
        "shared_structure_common.py",
        "shared_structure_numeric.py",
        "shared_structure_numeric_batch.py",
        "shared_structure_pipeline.py",
        "shared_structure_transfer.py",
        "shared_structure_target_batch.py",
        "codebook_counts.py",
        "prefix_census.py",
        "count_projection.py",
        "count_qp_active.py",
        "count_exchanges.py",
        "functional_observability.py",
    )
    hashes = {name: digest(BASE / name) for name in names}
    bootstrap = root / "bootstrap_stage"
    bootstrap.mkdir()
    initial_plan = {
        **original,
        "stage": "G25 confirmation bootstrap: frozen G23 prefix settings only",
        "registered_utc": datetime.now(UTC).isoformat(),
        "seed": 26110525017,
        "families": families,
        "methods": ["prefix"],
        "code_sha256": hashes,
        "theory_sha256": digest(BASE / "THEORY.md"),
        "limits": "Independent 12-family confirmation, unchanged known-bias/readout model and G23 prefix settings; no new HPO",
        "extract_jobs": [
            {"family_id": family, "task": task, "noise": noise, "method": "prefix"}
            for family in range(len(families))
            for noise in original["noise"]
            for task in range(original["sources"])
        ],
    }
    save_json(bootstrap / "protocol.json", initial_plan)
    jobs = []
    for family in range(len(families)):
        for noise in old["noise"]:
            jobs.append(
                {
                    "family_id": family,
                    "noise": noise,
                    "method": "bootstrap",
                    "tasks": list(range(old["sources"])),
                    "refit": False,
                    "solver": "frozen",
                }
            )
            for method, task_sets in (
                ("bootstrap_single_fixed", [[t] for t in range(old["sources"])]),
                ("bootstrap_joint_fixed", [list(range(old["sources"]))]),
            ):
                for solver in ("unit", "parent_pair"):
                    for tasks in task_sets:
                        jobs.append(
                            {
                                "family_id": family,
                                "noise": noise,
                                "method": method,
                                "tasks": tasks,
                                "refit": False,
                                "solver": solver,
                            }
                        )
    plan = {
        **old,
        "stage": "G25 independent confirmation of frozen G24, no new algorithm/HPO",
        "registered_utc": datetime.now(UTC).isoformat(),
        "seed": initial_plan["seed"],
        "query_seed": 26110525019,
        "families": families,
        "parent": str(bootstrap),
        "pilot": str(pilot),
        "pilot_protocol_sha256": digest(pilot / "protocol.json"),
        "pilot_selection_freeze_sha256": digest(pilot / "common/selection_frozen.json"),
        "parent_protocol_sha256": digest(bootstrap / "protocol.json"),
        "parent_fit_freeze_sha256": None,
        "methods": list(METHODS),
        "extract_jobs": jobs,
        "primary_method": "bootstrap_single_fixed",
        "secondary_methods": ["bootstrap", "bootstrap_joint_fixed"],
        "opened_geometries": False,
        "limits": "Independent confirmation of G24 on12newI, known commonbias/readout and costly queries; no claim of priority or new competitor superiority",
        "performance_policy": "CPU prefix search; measured compatible GPU batch C-fit. Idle MPS-only Default devices allowed. Never6/7, never alter foreign work.",
        "confirmation": {
            "criterion": "Primary exact signed I12/12 separately at each noise; all failures retained",
            "no_replacement_or_retuning": True,
            "new_codebooks_and_geometry_seed": True,
            "target_policy": "8newC/I, all32fresh signed starts, whole target-fit freeze before any target-test audit",
            "scope": "Known common bias/readout class only; no new competitor superiority or priority claim",
        },
        "execution": {
            "cpu_workers": 8,
            "gpu_allowlist": [0, 1, 2, 3],
            "reserved_gpus": [6, 7],
            "source_index_batch": 2,
            "target_index_batch": 4,
            "poll_seconds": 10,
            "idle_mps_policy": "Default compute mode, only MPS-server entries, <=64MiB and <=5% util are available; never modify foreign processes or GPU settings",
        },
        "code_sha256": hashes,
        "theory_sha256": digest(BASE / "THEORY.md"),
        "versions": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "supervisor": pid_identity(os.getpid()),
    }
    save_json(root / "protocol.json", plan)
    snapshot = root / "executed_code_snapshot"
    snapshot.mkdir()
    for name in names:
        shutil.copy2(BASE / name, snapshot / name)
    shutil.copy2(BASE / "THEORY.md", root / "theory_at_registration.md")
    for phase in (root, bootstrap):
        for name in ("data", "extract", "logs"):
            (phase / name).mkdir()
    (root / "bootstrap").mkdir()
    return plan


def gpu_inventory():
    available, records = [], []
    for gpu in (0, 1, 2, 3):
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-gpu=compute_mode,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        mode, utilization, memory = next(csv.reader(result.stdout.splitlines()))
        applications = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-compute-apps=process_name",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        idle_mps = all("nvidia-cuda-mps-server" in name for name in applications)
        allowed = not applications or (
            mode.strip() == "Default"
            and idle_mps
            and float(utilization) <= 5
            and float(memory) <= 64
        )
        if allowed:
            available.append(gpu)
        records.append(
            {
                "gpu": gpu,
                "mode": mode.strip(),
                "utilization": utilization.strip(),
                "memory_MiB": memory.strip(),
                "applications": applications,
                "available": allowed,
            }
        )
    return available, records


def run_workers(root, phase, count, *, gpu=False, device_count=8):
    plan = json.loads((root / "protocol.json").read_text())
    check_code(plan)
    devices, inventory = gpu_inventory() if gpu else ([], [])
    if gpu:
        save_json(
            root / f"inventory_{phase}.json",
            {"utc": datetime.now(UTC).isoformat(), "devices": inventory},
        )
        if not devices:
            raise RuntimeError("No allowed idle GPU; completed work retained")
    workers = min(count, len(devices) if gpu else device_count)
    processes = []
    with ExitStack() as stack:
        for shard in range(workers):
            environment = {
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "MPLCONFIGDIR": "/tmp/induction-bias-matplotlib",
            }
            if gpu:
                environment["CUDA_VISIBLE_DEVICES"] = str(devices[shard])
            log = stack.enter_context(
                (root / "logs" / f"{phase}_{shard}.log").open("w")
            )
            command = [
                sys.executable,
                "-u",
                "-m",
                PACKAGE + "joint_prefix_confirmation",
                "worker",
                "--root",
                str(root),
                "--phase",
                phase,
                "--shard",
                str(shard),
                "--shards",
                str(workers),
                "--device",
                "cuda:0" if gpu else "cpu",
            ]
            process = subprocess.Popen(
                command, env=environment, stdout=log, stderr=subprocess.STDOUT
            )
            processes.append((process, devices[shard] if gpu else None))
        status(root, phase, workers=[{"pid": p.pid, "gpu": g} for p, g in processes])
        while any(p.poll() is None for p, _ in processes):
            time.sleep(plan["execution"]["poll_seconds"])
        terminal = [
            {"pid": p.pid, "gpu": g, "exit_code": p.returncode} for p, g in processes
        ]
        status(root, phase + "_terminal", workers=terminal)
        if any(row["exit_code"] != 0 for row in terminal):
            raise RuntimeError(
                "A worker failed; inspect its log and preserve completed work"
            )


def load_numeric(root, plan, case, device):
    bank_path = root / "common/banks" / f"job_{case:03d}"
    bank = json.loads(bank_path.with_suffix(".json").read_text())
    source = root / "data" / f"family_{bank['family_id']:03d}_source.npz"
    frozen = json.loads((root / "data_frozen.json").read_text())
    if digest(source) != frozen["sha256"][str(source.relative_to(root))]:
        raise RuntimeError("Frozen numerical observations changed")
    with np.load(source) as arrays:
        data = tuple(
            torch.tensor(arrays[name], device=device)
            for name in (
                "train_x",
                f"train_y_{bank['noise']}",
                "fit_validation_x",
                f"fit_validation_y_{bank['noise']}",
            )
        )
    with np.load(bank_path.with_suffix(".npz")) as arrays:
        indices = arrays["indices"]
    initial = torch.stack(
        [
            fresh_codes(
                plan, plan["families"][bank["family_id"]]["family"], task, device
            )
            for task in range(plan["sources"])
        ]
    )
    return bank, indices, (*data, initial)


def profile_source(root, plan, device):
    cases = len(plan["families"]) * len(plan["noise"])
    for case in range(cases):
        bank = json.loads((root / "common/banks" / f"job_{case:03d}.json").read_text())
        if bank["candidates"]:
            break
    else:
        raise RuntimeError("No bootstrap numerical case for execution measurement")
    _, indices, data = load_numeric(root, plan, case, device)
    x, y, vx, vy, initial = data
    serial_fit(indices[0], x, y[0], vx, vy[0], initial[0], plan, max_steps=1)
    batch_fit(indices[:1], x, y, vx, vy, initial, plan, max_steps=1)
    torch.cuda.synchronize()
    started = time.perf_counter()
    values = [
        serial_fit(indices[0], x, y[t], vx, vy[t], initial[t], plan)
        for t in range(plan["sources"])
    ]
    torch.cuda.synchronize()
    serial_seconds = time.perf_counter() - started
    started = time.perf_counter()
    codes, histories, _ = batch_fit(indices[:1], x, y, vx, vy, initial, plan)
    torch.cuda.synchronize()
    batch_seconds = time.perf_counter() - started
    code_error = float((torch.stack([v[0] for v in values]) - codes[0]).abs().max())
    score_error = max(
        float(
            np.max(
                np.abs(
                    np.array(a[1]["best_validation_per_start"])
                    - np.array(b["best_validation_per_start"])
                )
                / (1 + np.abs(a[1]["best_validation_per_start"]))
            )
        )
        for a, b in zip(values, histories[0], strict=True)
    )
    result = {
        "case": case,
        "serial_seconds": serial_seconds,
        "batch_seconds": batch_seconds,
        "speedup": serial_seconds / batch_seconds,
        "C_difference": code_error,
        "per_start_validation_relative_difference": score_error,
        "parity_passed": code_error < 1e-6 and score_error < 1e-8,
        "backend": "batch" if batch_seconds < serial_seconds else "serial",
        "candidate_retained": False,
        "meta_or_teacher_opened": False,
    }
    save_json(root / "source_execution.json", result)
    print(json.dumps(result), flush=True)
    if not result["parity_passed"]:
        raise ArithmeticError("Numerical parity failed; no backend adopted")


def fit_case(root, plan, case, device, measurement):
    bank, indices, data = load_numeric(root, plan, case, device)
    x, y, vx, vy, initial = data
    phase_hash = digest(root / "common/protocol.json")
    destination = root / "common/fits" / f"job_{case:03d}"
    if destination.with_suffix(".json").exists():
        raise RuntimeError("Numerical case already completed")
    destination.mkdir()
    artifacts = []
    batch_size = 2 if measurement["backend"] == "batch" else 1
    for begin in range(0, len(indices), batch_size):
        selected = indices[begin : begin + batch_size]
        started = time.monotonic()
        if measurement["backend"] == "batch":
            codes, histories, _ = batch_fit(selected, x, y, vx, vy, initial, plan)
        else:
            values = [
                [
                    serial_fit(index, x, y[t], vx, vy[t], initial[t], plan)
                    for t in range(plan["sources"])
                ]
                for index in selected
            ]
            codes = torch.stack([torch.stack([v[0] for v in row]) for row in values])
            histories = [[v[1] for v in row] for row in values]
        for offset, index in enumerate(selected):
            number = begin + offset
            candidate = bank["candidates"][number]
            output = destination / candidate["signature"]
            np.savez_compressed(
                output.with_suffix(".npz"),
                index=index,
                codes=codes[offset].cpu().numpy(),
            )
            save_json(
                output.with_suffix(".json"),
                {
                    "job_id": case,
                    "candidate": number,
                    "signature": candidate["signature"],
                    "numeric_fits": histories[offset],
                    "phase_protocol_sha256": phase_hash,
                    "arrays_sha256": digest(output.with_suffix(".npz")),
                    "meta_labels_opened": False,
                    "seconds_per_batch": time.monotonic() - started,
                    "execution": measurement["backend"],
                },
            )
            artifacts.append(str(output.relative_to(root / "common")))
        print(
            json.dumps(
                {
                    "case": case,
                    "completed_candidates": min(begin + batch_size, len(indices)),
                    "total": len(indices),
                    "batch_seconds": time.monotonic() - started,
                }
            ),
            flush=True,
        )
    save_json(
        destination.with_suffix(".json"),
        {
            "job_id": case,
            "fit_artefacts": artifacts,
            "bank_sha256": digest(root / "common/banks" / f"job_{case:03d}.json"),
            "phase_protocol_sha256": phase_hash,
            "meta_labels_opened": False,
        },
    )


def bootstrap(root, plan):
    parent = Path(plan["parent"])
    previous = json.loads((parent / "protocol.json").read_text())
    lookup = {
        (j["family_id"], j["noise"], j["task"]): n
        for n, j in enumerate(previous["extract_jobs"])
    }
    outputs = []
    for family in range(len(plan["families"])):
        for noise_id, noise in enumerate(plan["noise"]):
            case = family * len(plan["noise"]) + noise_id
            bank_path = parent / "common/banks" / f"job_{case:03d}.json"
            bank = json.loads(bank_path.read_text())
            scores = []
            for candidate in bank["methods"]["prefix"]:
                signature = bank["candidates"][candidate]["signature"]
                fitted = parent / "common/fits" / f"job_{case:03d}" / signature
                meta = json.loads(fitted.with_suffix(".json").read_text())
                scores.append(
                    (
                        float(
                            np.mean([r["validation_mse"] for r in meta["numeric_fits"]])
                        ),
                        candidate,
                        signature,
                    )
                )
            axis = []
            for task in range(plan["sources"]):
                number = lookup[family, noise, task]
                record = json.loads(
                    (parent / "extract" / f"job_{number:04d}.json").read_text()
                )
                axis.append(record["records"][0]["axis_fit"]["selected"]["codes"])
            output = root / "bootstrap" / f"case_{case:03d}"
            metadata = {
                "case": case,
                "available": bool(scores),
                "selection_used_meta_or_teacher": False,
                "bank_sha256": digest(bank_path),
                "parent_fit_freeze_sha256": digest(parent / "common/fit_frozen.json"),
            }
            if scores:
                score, candidate, signature = min(scores)
                fitted = parent / "common/fits" / f"job_{case:03d}" / signature
                with np.load(fitted.with_suffix(".npz")) as arrays:
                    index, codes = arrays["index"], arrays["codes"]
                metadata.update(
                    {
                        "selected_candidate": candidate,
                        "signature": signature,
                        "mean_old_H_mse": score,
                        "numeric_fit_sha256": digest(fitted.with_suffix(".npz")),
                    }
                )
            else:
                index = np.zeros((0, plan["inputs"], plan["hidden"]), dtype=np.int16)
                codes = np.full((plan["sources"], plan["codes"]), np.nan)
            np.savez_compressed(
                output.with_suffix(".npz"),
                index=index,
                codes=codes,
                axis_codes=np.asarray(axis),
            )
            save_json(output.with_suffix(".json"), metadata)
            outputs.extend((output.with_suffix(".json"), output.with_suffix(".npz")))
    save_json(
        root / "bootstrap_frozen.json",
        {
            "utc": datetime.now(UTC).isoformat(),
            "sha256": {str(p.relative_to(root)): digest(p) for p in outputs},
            "parent_fit_freeze_sha256": digest(parent / "common/fit_frozen.json"),
            "fresh_queries_opened": False,
        },
    )


def build_final_banks(root, plan):
    for family in range(len(plan["families"])):
        for noise_id, noise in enumerate(plan["noise"]):
            case = family * len(plan["noise"]) + noise_id
            values, candidates, lookup = [], [], {}
            methods = {method: [] for method in plan["methods"]}
            for number, job in enumerate(plan["extract_jobs"]):
                if job["family_id"] != family or job["noise"] != noise:
                    continue
                path = root / "extract" / f"job_{number:04d}"
                if not json.loads(path.with_suffix(".json").read_text())["complete"]:
                    continue
                with np.load(path.with_suffix(".npz")) as arrays:
                    index = arrays["index"]
                signature = hashlib.sha256(index.tobytes()).hexdigest()
                if signature not in lookup:
                    lookup[signature] = len(values)
                    values.append(index)
                    candidates.append({"signature": signature, "origins": []})
                position = lookup[signature]
                candidates[position]["origins"].append(
                    {
                        "method": job["method"],
                        "task": job["tasks"][0] if len(job["tasks"]) == 1 else -1,
                        "source_job": number,
                        "record": 0,
                    }
                )
                if position not in methods[job["method"]]:
                    methods[job["method"]].append(position)
            path = root / "common/banks" / f"job_{case:03d}"
            indices = (
                np.stack(values)
                if values
                else np.zeros((0, plan["inputs"], plan["hidden"]), dtype=np.int16)
            )
            np.savez_compressed(path.with_suffix(".npz"), indices=indices)
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


def select(root, plan, case):
    bank = json.loads((root / "common/banks" / f"job_{case:03d}.json").read_text())
    path = root / "common/selected" / f"job_{case:03d}"
    if bank["candidates"]:
        common.select_job(root, plan, case, "cpu")
        metadata = json.loads(path.with_suffix(".json").read_text())
        metadata["records"] = [
            r for r in metadata["records"] if r["scope"] == "common4"
        ]
    else:
        np.savez_compressed(path.with_suffix(".npz"))
        metadata = {
            "job_id": case,
            "family_id": bank["family_id"],
            "noise": bank["noise"],
            "records": [
                {
                    "method": method,
                    "scope": "common4",
                    "complete": False,
                    "available_candidates": [],
                }
                for method in plan["methods"]
            ],
            "fit_freeze_sha256": digest(root / "common/fit_frozen.json"),
            "teacher_audit_opened": False,
        }
    save_json(path.with_suffix(".json"), metadata)


def worker(root, phase, shard, shards, device):
    torch.set_num_threads(1)
    plan = json.loads((root / "protocol.json").read_text())
    check_code(plan)
    parent = Path(plan["parent"])
    initial = json.loads((parent / "protocol.json").read_text())
    families, cases = len(plan["families"]), len(plan["families"]) * len(plan["noise"])
    if phase == "source_profile":
        profile_source(parent, initial, device)
        return
    if phase == "target_profile":
        profile_target(root, plan, device)
        return
    counts = {
        "source_prepare": families,
        "source_extract": len(initial["extract_jobs"]),
        "source_fit": cases,
        "fresh_prepare": families,
        "fresh_search": len(plan["extract_jobs"]),
        "fresh_fit": cases,
        "select": cases,
        "target_prepare": families,
        "target_fit": cases,
    }
    for number in range(shard, counts[phase], shards):
        if phase == "source_prepare":
            prepare_source(parent, initial, number, "cpu")
        elif phase == "source_extract":
            extract_source(parent, initial, number, "cpu")
        elif phase in ("source_fit", "fresh_fit"):
            measured = json.loads((parent / "source_execution.json").read_text())
            fit_case(
                parent if phase == "source_fit" else root,
                initial if phase == "source_fit" else plan,
                number,
                device,
                measured,
            )
        elif phase == "fresh_prepare":
            joint.prepare(root, plan, number)
        elif phase == "fresh_search":
            job = plan["extract_jobs"][number]
            case = job["family_id"] * len(plan["noise"]) + plan["noise"].index(
                job["noise"]
            )
            available = json.loads(
                (root / "bootstrap" / f"case_{case:03d}.json").read_text()
            )["available"]
            if available:
                joint.search_job(root, plan, number, "cpu")
            else:
                path = root / "extract" / f"job_{number:04d}"
                np.savez_compressed(path.with_suffix(".npz"))
                save_json(
                    path.with_suffix(".json"),
                    {
                        "job": job,
                        "job_id": number,
                        "complete": False,
                        "reason": "no_bootstrap_candidate",
                        "seconds": 0.0,
                        "protocol_sha256": digest(root / "protocol.json"),
                    },
                )
        elif phase == "select":
            select(root, plan, number)
        elif phase == "target_prepare":
            transfer.prepare(root, plan, number, "cpu")
        elif phase == "target_fit":
            fit_target(
                root,
                plan,
                number,
                device,
                plan["execution"]["target_index_batch"],
                None,
            )
        print(json.dumps({"phase": phase, "completed": number}), flush=True)


def run(root, pilot):
    torch.set_num_threads(1)
    plan = register(root, pilot)
    status(root, "registered_before_new_data")
    parent = Path(plan["parent"])
    initial = json.loads((parent / "protocol.json").read_text())
    cases = len(plan["families"]) * len(plan["noise"])
    run_workers(root, "source_prepare", len(plan["families"]))
    joint.freeze_stage(parent, initial, "data")
    run_workers(root, "source_extract", len(initial["extract_jobs"]))
    joint.freeze_stage(parent, initial, "search")
    common.register(parent)
    common.build_banks(parent, initial)
    run_workers(root, "source_profile", 1, gpu=True)
    run_workers(root, "source_fit", cases, gpu=True)
    common.freeze(parent, initial, "fit")
    bootstrap(root, plan)
    status(root, "bootstrap_frozen_before_fresh_queries")
    run_workers(root, "fresh_prepare", len(plan["families"]))
    joint.freeze_stage(root, plan, "data")
    run_workers(root, "fresh_search", len(plan["extract_jobs"]))
    joint.freeze_stage(root, plan, "search")
    common.register(root)
    build_final_banks(root, plan)
    run_workers(root, "fresh_fit", cases, gpu=True)
    common.freeze(root, plan, "fit")
    run_workers(root, "select", cases)
    common.freeze(root, plan, "selection")
    common.audit(root, plan)
    status(root, "source_audit_complete_no_retuning")
    transfer.register(root)
    phase_path = root / "transfer/protocol.json"
    phase = json.loads(phase_path.read_text())
    for name in (
        "shared_structure_numeric_batch.py",
        "shared_structure_target_batch.py",
    ):
        phase["code_sha256"][name] = plan["code_sha256"][name]
        shutil.copy2(BASE / name, root / "transfer/executed_code_snapshot" / name)
    phase["confirmation_protocol_sha256"] = digest(root / "protocol.json")
    save_json(phase_path, phase)
    run_workers(root, "target_prepare", len(plan["families"]))
    transfer.banks(root, plan)
    transfer.freeze(root, plan, "data")
    run_workers(root, "target_profile", 1, gpu=True)
    run_workers(root, "target_fit", cases, gpu=True)
    transfer.freeze(root, plan, "fit")
    transfer.audit(root, plan, "cpu")
    status(root, "target_audit_complete")
    from .joint_prefix_confirmation_report import write_report

    write_report(root)
    status(root, "confirmation_and_report_complete", new_experiments_pending=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "worker"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--phase")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.action == "run":
        if args.pilot is None:
            parser.error("--pilot is required for registration")
        run(args.root.resolve(), args.pilot.resolve())
    else:
        worker(args.root.resolve(), args.phase, args.shard, args.shards, args.device)


if __name__ == "__main__":
    main()
