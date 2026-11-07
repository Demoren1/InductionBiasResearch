"""Preserve G24 science and use CPU while foreign MPS occupies GPU contexts."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json

from .shared_structure_common import fit_job as serial_job
from .shared_structure_data import load_source
from .shared_structure_numeric import fit_codes as serial_fit
from .shared_structure_numeric import fresh_codes
from .shared_structure_numeric_batch import fit_codes as batch_fit
from .shared_structure_numeric_batch import fit_job as batch_job
from .shared_structure_pipeline import (
    PACKAGE,
    check_code,
    pid_identity,
    run_stage,
    state,
)


def cpu_workers(root, protocol, module, action=None, extra=()):
    check_code(protocol)
    command = [sys.executable, "-u", "-m", PACKAGE + module]
    if action:
        command.append(action)
    command.extend(("--root", str(root), *extra))
    processes = []
    with ExitStack() as stack:
        for shard in range(protocol["cpu_workers"]):
            handle = stack.enter_context(
                (root / "logs" / f"cpu_{module}_{action}_{shard}.log").open("w")
            )
            process = subprocess.Popen(
                command
                + ["--shard", str(shard), "--shards", str(protocol["cpu_workers"])],
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": "",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                },
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            processes.append(process)
        state(
            root,
            "cpu_" + (action or module),
            workers=[process.pid for process in processes],
        )
        while any(process.poll() is None for process in processes):
            time.sleep(protocol["poll_seconds"])
        results = [
            {"pid": process.pid, "exit_code": process.returncode}
            for process in processes
        ]
        state(root, "cpu_" + (action or module) + "_terminal", workers=results)
        if any(row["exit_code"] for row in results):
            raise RuntimeError(
                "CPU worker stopped; inspect without repeating completed jobs"
            )


def numeric_measurement(root, plan):
    source = root / "common/banks/job_009.npz"
    with np.load(source) as arrays:
        index = arrays["indices"][0]
    data = [
        load_source(root, plan, 4, task, 2.0, "cpu") for task in range(plan["sources"])
    ]
    x, vx = data[0]["train_x"], data[0]["fit_validation_x"]
    y = torch.stack([row["train_y_2.0"] for row in data])
    vy = torch.stack([row["fit_validation_y_2.0"] for row in data])
    initial = torch.stack(
        [
            fresh_codes(plan, plan["families"][4]["family"], task, "cpu")
            for task in range(plan["sources"])
        ]
    )
    serial_fit(index, x, y[0], vx, vy[0], initial[0], plan, max_steps=1)
    batch_fit(index[None], x, y, vx, vy, initial, plan, max_steps=1)
    started = time.perf_counter()
    serial = [
        serial_fit(index, x, y[task], vx, vy[task], initial[task], plan)
        for task in range(plan["sources"])
    ]
    serial_seconds = time.perf_counter() - started
    started = time.perf_counter()
    codes, histories, _ = batch_fit(index[None], x, y, vx, vy, initial, plan)
    batch_seconds = time.perf_counter() - started
    difference = float(
        (torch.stack([value[0] for value in serial]) - codes[0]).abs().max()
    )
    score_error = max(
        float(
            np.max(
                np.abs(
                    np.array(serial[task][1]["best_validation_per_start"])
                    - np.array(histories[0][task]["best_validation_per_start"])
                )
                / (1 + np.abs(serial[task][1]["best_validation_per_start"]))
            )
        )
        for task in range(plan["sources"])
    )
    parity = difference < 1e-6 and score_error < 1e-8
    measurement = {
        "device": "cpu",
        "indexators": 1,
        "tasks": plan["sources"],
        "serial_seconds": serial_seconds,
        "batch_seconds": batch_seconds,
        "serial_over_batch": serial_seconds / batch_seconds,
        "maximum_C_difference": difference,
        "maximum_relative_per_start_validation_difference": score_error,
        "parity_passed": parity,
        "backend": "batch" if parity and batch_seconds < serial_seconds else "serial",
        "source_protocol_sha256": digest(root / "protocol.json"),
        "new_meta_or_teacher_opened": False,
        "candidate_retained": False,
    }
    save_json(root / "cpu_numeric_measurement.json", measurement)
    print(json.dumps(measurement), flush=True)


def run(root):
    if (root / "cpu_continuation.json").exists():
        raise RuntimeError("CPU continuation already registered; inspect its session")
    source = json.loads((root / "protocol.json").read_text())
    check_code(source)
    previous = json.loads((root / "pipeline.json").read_text())["supervisor"]
    stage = json.loads((root / "comparison_pipeline_state.json").read_text())["stage"]
    if (
        pid_identity(previous["pid"]) != previous
        or stage != "waiting_for_free_GPU_for_paired_measurement"
    ):
        raise RuntimeError("Only the exact waiting CPU supervisor may be replaced")
    if (
        list((root / "extract").glob("job_*.json"))
        or (root / "execution_measurement.json").exists()
    ):
        raise RuntimeError("Scientific work already started; do not replace it")
    protocol = {
        "registered_utc": datetime.now(UTC).isoformat(),
        "source_protocol_sha256": digest(root / "protocol.json"),
        "code_sha256": {
            **source["code_sha256"],
            Path(__file__).name: digest(Path(__file__)),
        },
        "superseded_waiting_supervisor": previous,
        "supervisor": pid_identity(os.getpid()),
        "cpu_workers": 8,
        "poll_seconds": 15,
        "requested_gpus": [],
        "reserved_gpus": [6, 7],
        "reason": "Host inspection found foreign-user MPS PID702767 on all GPU0--3. No foreign service/settings will be changed. CPU reference was already the fallback in G24; run it without waiting for optional GPU timing. Preserve all scientific code/data/starts/settings; register this execution exception before any scientific outputs.",
        "numeric_policy": "Compare serial and batch1 actual CPU fixed-I fits on one bank case, then adopt fastest numerically compatible backend. Whole fit/predictor/selection freezes and independent new meta/lateaudit preserved. No GPU use or automatic retries.",
    }
    save_json(root / "cpu_continuation.json", protocol)
    shutil.copy2(Path(__file__), root / "cpu_continuation_snapshot.py")
    os.kill(previous["pid"], signal.SIGINT)
    deadline = time.monotonic() + 10
    while pid_identity(previous["pid"]) == previous:
        if time.monotonic() > deadline:
            raise RuntimeError(
                "Waiting supervisor did not terminate; no worker stopped"
            )
        time.sleep(0.1)
    save_json(
        root / "execution_measurement.json",
        {
            "cases": {
                key: {"backend": "cpu", "cpu_gpu_parity_measured": False}
                for key in [
                    "axis",
                    *(
                        f"{tasks}/{refit}/{solver}"
                        for tasks in (1, 4)
                        for refit in (0, 1)
                        for solver in ("unit", "parent_pair")
                    ),
                ]
            },
            "scope": "CPU reference only; no GPU speed/parity claim. Whole scientific runtimes measured per actual job. Foreign MPS prevented certification of free GPU0--3. Optional paired timing skipped by preregistered CPU-only continuation, before any outputs.",
            "source_protocol_sha256": digest(root / "protocol.json"),
            "cpu_continuation_sha256": digest(root / "cpu_continuation.json"),
            "candidate_retained": False,
        },
    )
    cpu_workers(
        root,
        protocol,
        "joint_prefix_benchmark",
        "search",
        ("--backend", "cpu", "--device", "cpu"),
    )
    run_stage(root, protocol, "joint_prefix_benchmark", "freeze_search")
    run_stage(root, protocol, "shared_structure_common", "register")
    run_stage(root, protocol, "joint_prefix_benchmark", "banks")
    torch.set_num_threads(1)
    numeric_measurement(root, source)
    path = root / "common/protocol.json"
    phase = json.loads(path.read_text())
    phase["cpu_numeric_execution"] = {
        "measurement_sha256": digest(root / "cpu_numeric_measurement.json"),
        "continuation_sha256": digest(root / "cpu_continuation.json"),
        "indexator_batch_size": 1,
        "scientific_settings_unchanged": True,
    }
    save_json(path, phase)
    cpu_workers(root, protocol, "joint_prefix_cpu_continuation", "fit")
    run_stage(root, protocol, "joint_prefix_benchmark", "freeze_fit")
    cpu_workers(root, protocol, "joint_prefix_benchmark", "select", ("--device", "cpu"))
    for action in ("freeze_selection", "audit"):
        run_stage(root, protocol, "joint_prefix_benchmark", action)
    run_stage(root, protocol, "joint_prefix_report")
    state(
        root,
        "g24_cpu_diagnosis_report_ready",
        report=str(root / "report/REPORT.md"),
        overall_goal_complete=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "fit"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU execution must hide all CUDA devices")
    root = args.root.resolve()
    if args.action == "run":
        try:
            run(root)
        except BaseException as error:
            state(
                root,
                "g24_cpu_continuation_stopped",
                error=repr(error),
                automatically_restarted=False,
            )
            raise
    else:
        torch.set_num_threads(1)
        protocol = json.loads((root / "cpu_continuation.json").read_text())
        check_code(protocol)
        source = json.loads((root / "protocol.json").read_text())
        measured = json.loads((root / "cpu_numeric_measurement.json").read_text())
        for job in range(
            args.shard, len(source["families"]) * len(source["noise"]), args.shards
        ):
            if measured["backend"] == "batch":
                batch_job(root, source, job, "cpu", 1)
            else:
                serial_job(root, source, job, "cpu")


if __name__ == "__main__":
    main()
