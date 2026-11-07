"""Continue G23 on GPUs 0--3 after the user reserved GPUs 6 and 7."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json

from .shared_structure_extract import run as extract
from .shared_structure_pipeline import (
    PACKAGE,
    check_code,
    pid_identity,
    run_stage,
    state,
)

GPU_ALLOWLIST = (0, 1, 2, 3)
PLAN_NAME = "gpu_restriction_continuation.json"


def load_plan(root):
    plan = json.loads((root / PLAN_NAME).read_text())
    check_code(plan)
    if digest(Path(__file__)) != plan["continuation_code_sha256"]:
        raise RuntimeError("Registered scheduling continuation changed")
    if plan["requested_gpus"] != list(GPU_ALLOWLIST):
        raise RuntimeError("Only the registered GPUs 0--3 are permitted")
    return plan


def source_worker(root, worker_id):
    plan = load_plan(root)
    assignment = plan["rescheduled_workers"][worker_id]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(assignment["gpu"]):
        raise RuntimeError("Worker CUDA visibility differs from its GPU assignment")
    torch.set_num_threads(1)
    source = json.loads((root / "protocol.json").read_text())
    for job_id in assignment["job_ids"]:
        extract(root, source, job_id, "cuda:0")


def register(root):
    destination = root / PLAN_NAME
    if destination.exists():
        raise RuntimeError("Continuation already registered; inspect its live handle")
    previous = json.loads((root / "comparison_pipeline.json").read_text())
    check_code(previous)
    source = json.loads((root / "protocol.json").read_text())
    if digest(root / "protocol.json") != previous["source_protocol_sha256"]:
        raise RuntimeError("Source protocol changed")
    if (root / "extract_frozen.json").exists() or (root / "common").exists():
        raise RuntimeError("This scheduling continuation precedes the common stage")
    identities = previous["source_processes"]
    released = identities[4:]
    if any(pid_identity(row["pid"]) == row for row in released):
        raise RuntimeError("A source process on a reserved GPU is still running")
    if pid_identity(403536) is not None:
        raise RuntimeError("The superseded comparison supervisor is still present")
    complete = {
        number
        for number in range(len(source["extract_jobs"]))
        if (root / "extract" / f"job_{number:04d}.json").exists()
    }
    pending = [
        number
        for number in range(len(source["extract_jobs"]))
        if number % 6 in (4, 5) and number not in complete
    ]
    survivors = []
    for shard, identity in enumerate(identities[:4]):
        current = pid_identity(identity["pid"])
        if current == identity:
            survivors.append(identity)
        elif any(
            number not in complete
            for number in range(shard, len(source["extract_jobs"]), 6)
        ):
            raise RuntimeError("A surviving original source shard has missing jobs")
    # Each allowed card already has our original source shard. Do not share a
    # card with any process outside those verified original source workers.
    allowed_pids = {str(row["pid"]) for row in survivors}
    for gpu in GPU_ALLOWLIST:
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if set(result.stdout.split()) - allowed_pids:
            raise RuntimeError(f"GPU {gpu} has an unrelated compute process")
    plan = {
        **previous,
        "registered_utc": datetime.now(UTC).isoformat(),
        "requested_gpus": list(GPU_ALLOWLIST),
        "reserved_gpus": [6, 7],
        "superseded_pipeline_sha256": digest(root / "comparison_pipeline.json"),
        "superseded_supervisor_pid": 403536,
        "source_processes": survivors,
        "released_source_processes": released,
        "complete_at_registration": sorted(complete),
        "rescheduled_workers": [
            {"gpu": gpu, "job_ids": pending[worker_id :: len(GPU_ALLOWLIST)]}
            for worker_id, gpu in enumerate(GPU_ALLOWLIST)
        ],
        "continuation_code_sha256": digest(Path(__file__)),
        "policy": (
            "Human instruction reserves GPUs 6 and 7. Their two original source "
            "workers and the old waiting supervisor were interrupted and verified "
            "terminal. Preserve all completed jobs. Restart only unsaved original "
            "shard-4/5 jobs, with unchanged observations, settings and seeds, on "
            "GPUs 0--3 alongside our surviving source workers. No teacher/meta "
            "labels are opened by scheduling. Later stages use only free GPUs "
            "from 0--3. GPUs 4/5 are occupied by unrelated work."
        ),
    }
    save_json(destination, plan)
    shutil.copy2(Path(__file__), root / "gpu_restriction_continuation_snapshot.py")
    return plan


def continue_comparison(root):
    plan = register(root)
    processes = []
    with ExitStack() as stack:
        for worker_id, assignment in enumerate(plan["rescheduled_workers"]):
            gpu = assignment["gpu"]
            log = stack.enter_context(
                (root / "logs" / f"extract_continuation_gpu{gpu}.log").open("w")
            )
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    PACKAGE + "shared_structure_gpu_continuation",
                    "source_worker",
                    "--root",
                    str(root),
                    "--worker-id",
                    str(worker_id),
                ],
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                },
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            processes.append((gpu, process))
        save_json(
            root / "gpu_restriction_processes.json",
            {
                "supervisor": pid_identity(os.getpid()),
                "source_workers": [
                    {"gpu": gpu, **pid_identity(process.pid)}
                    for gpu, process in processes
                ],
            },
        )
        expected = len(json.loads((root / "protocol.json").read_text())["extract_jobs"])
        while True:
            live = [
                row["pid"]
                for row in plan["source_processes"]
                if pid_identity(row["pid"]) == row
            ]
            completed = len(list((root / "extract").glob("job_*.json")))
            workers = [
                {"gpu": gpu, "pid": process.pid, "exit_code": process.poll()}
                for gpu, process in processes
            ]
            state(
                root,
                "waiting_for_source_on_GPU_0_to_3",
                completed_jobs=completed,
                total_jobs=expected,
                live_original_source_pids=live,
                continuation_workers=workers,
                reserved_gpus=[6, 7],
            )
            if not live and all(row["exit_code"] is not None for row in workers):
                if completed != expected or any(row["exit_code"] for row in workers):
                    raise RuntimeError(
                        "Source stage incomplete; inspect existing handles"
                    )
                break
            time.sleep(plan["poll_seconds"])
    load_plan(root)
    run_stage(root, plan, "shared_structure_benchmark", "freeze_extract")
    for action in (
        "register",
        "banks",
        "fit",
        "freeze_fit",
        "select",
        "freeze_selection",
        "audit",
    ):
        run_stage(
            root,
            plan,
            "shared_structure_common",
            action,
            parallel=action in ("fit", "select"),
        )
    run_stage(root, plan, "shared_structure_report", extra=("--source-only",))
    state(
        root,
        "structural_comparison_report_ready",
        report=str(root / "structure_report/REPORT.md"),
        overall_goal_complete=False,
        reserved_gpus=[6, 7],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "source_worker"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worker-id", type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.action == "run":
            continue_comparison(root)
        else:
            source_worker(root, args.worker_id)
    except BaseException as error:
        if args.action == "run":
            state(root, "GPU_restriction_continuation_stopped", error=repr(error))
        raise


if __name__ == "__main__":
    main()
