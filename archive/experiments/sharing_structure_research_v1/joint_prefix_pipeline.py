"""Run frozen G24 search on measured backends, then independent late selection."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from experiments.darts_structured_tasks_v1.data import digest, save_json

from .shared_structure_pipeline import (
    BASE,
    PACKAGE,
    check_code,
    free_gpus,
    pid_identity,
    run_stage,
    state,
)


def paired_measurement(root, protocol):
    while not (devices := free_gpus(protocol["requested_gpus"])):
        state(root, "waiting_for_free_GPU_for_paired_measurement")
        time.sleep(protocol["poll_seconds"])
    gpu = devices[0]
    state(root, "paired_whole_prefix_measurement", gpu=gpu)
    with (root / "logs/execution_measurement.log").open("w") as handle:
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                PACKAGE + "joint_prefix_performance",
                "--root",
                str(root),
            ],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def search(root, protocol):
    measurement = json.loads((root / "execution_measurement.json").read_text())
    use_gpu = any(row["backend"] == "cuda" for row in measurement["cases"].values())
    devices = []
    if use_gpu:
        while not (devices := free_gpus(protocol["requested_gpus"])):
            state(root, "waiting_for_free_GPU_for_measured_search")
            time.sleep(protocol["poll_seconds"])
    processes = []
    arguments = [
        sys.executable,
        "-u",
        "-m",
        PACKAGE + "joint_prefix_benchmark",
        "search",
        "--root",
        str(root),
    ]
    with ExitStack() as stack:
        for backend, workers in (("cpu", list(range(8))), ("cuda", devices)):
            for shard, worker in enumerate(workers):
                handle = stack.enter_context(
                    (root / "logs" / f"search_{backend}_{worker}.log").open("w")
                )
                environment = {
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": "" if backend == "cpu" else str(worker),
                }
                process = subprocess.Popen(
                    arguments
                    + [
                        "--backend",
                        backend,
                        "--device",
                        "cpu" if backend == "cpu" else "cuda:0",
                        "--shard",
                        str(shard),
                        "--shards",
                        str(len(workers)),
                    ],
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
                processes.append((backend, worker, process))
        state(
            root,
            "search_on_measured_backends",
            workers=[
                {"backend": b, "worker": w, "pid": p.pid} for b, w, p in processes
            ],
        )
        while any(process.poll() is None for _, _, process in processes):
            time.sleep(protocol["poll_seconds"])
        terminal = [
            {"backend": b, "worker": w, "exit_code": p.returncode}
            for b, w, p in processes
        ]
        state(root, "search_terminal", workers=terminal)
        if any(row["exit_code"] for row in terminal):
            raise RuntimeError("Inspect stopped search workers; no automatic repeat")


def run(root):
    source = json.loads((root / "protocol.json").read_text())
    check_code(source)
    if (root / "pipeline.json").exists():
        raise RuntimeError("Pipeline already registered; inspect its actual session")
    protocol = {
        "source_protocol_sha256": digest(root / "protocol.json"),
        "code_sha256": source["code_sha256"],
        "supervisor": pid_identity(os.getpid()),
        "requested_gpus": [0, 1, 2, 3],
        "reserved_gpus": [6, 7],
        "poll_seconds": 15,
    }
    save_json(root / "pipeline.json", protocol)
    paired_measurement(root, protocol)
    search(root, protocol)
    run_stage(root, protocol, "joint_prefix_benchmark", "freeze_search")
    run_stage(root, protocol, "shared_structure_common", "register")
    run_stage(root, protocol, "joint_prefix_benchmark", "banks")
    phase_path = root / "common/protocol.json"
    phase = json.loads(phase_path.read_text())
    phase["batched_numeric_execution"] = {
        "code_sha256": source["code_sha256"]["shared_structure_numeric_batch.py"],
        "indexators_per_batch": 4,
        "prior_same_train_shape_performance_sha256": digest(
            Path(source["parent"]) / "batched_numeric_performance.json"
        ),
        "scope": "Same scientific GN and32signedstarts; same original train, new same-shaped H. Prior local parity/timing evidence, not universal bitwise guarantee.",
    }
    phase["code_sha256"]["shared_structure_numeric_batch.py"] = source["code_sha256"][
        "shared_structure_numeric_batch.py"
    ]
    save_json(phase_path, phase)
    shutil.copy2(
        BASE / "shared_structure_numeric_batch.py",
        root / "common/executed_code_snapshot/shared_structure_numeric_batch.py",
    )
    run_stage(
        root,
        protocol,
        "shared_structure_numeric_batch",
        parallel=True,
        extra=("--batch-size", "4"),
    )
    for action in ("freeze_fit", "select", "freeze_selection", "audit"):
        run_stage(
            root,
            protocol,
            "joint_prefix_benchmark",
            action,
            parallel=action == "select",
        )
    run_stage(root, protocol, "joint_prefix_report")
    state(
        root,
        "g24_diagnosis_report_ready",
        report=str(root / "report/REPORT.md"),
        overall_goal_complete=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.root.resolve())
    except BaseException as error:
        state(
            args.root,
            "g24_pipeline_stopped",
            error=repr(error),
            automatically_restarted=False,
        )
        raise


if __name__ == "__main__":
    main()
