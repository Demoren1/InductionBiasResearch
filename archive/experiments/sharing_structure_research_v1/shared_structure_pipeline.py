"""Finish the registered G23 structural comparison after live source workers."""

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

from experiments.darts_structured_tasks_v1.data import digest, save_json

BASE = Path(__file__).resolve().parent
PACKAGE = "experiments.sharing_structure_research_v1."


def pid_identity(pid):
    path = Path("/proc") / str(pid)
    try:
        fields = (path / "stat").read_text().split()
        if fields[2] == "Z":
            return None
        return {
            "pid": pid,
            "start_time_ticks": fields[21],
            "command_sha256": digest(path / "cmdline"),
        }
    except FileNotFoundError:
        return None


def state(root, stage, **details):
    value = {"utc": datetime.now(UTC).isoformat(), "stage": stage, **details}
    save_json(root / "comparison_pipeline_state.json", value)
    print(json.dumps(value), flush=True)


def check_code(protocol):
    for name, expected in protocol["code_sha256"].items():
        if digest(BASE / name) != expected:
            raise RuntimeError(f"Registered comparison code changed: {name}")


def free_gpus(requested):
    available = []
    for gpu in requested:
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
        if not result.stdout.strip():
            available.append(gpu)
    return available


def run_stage(root, protocol, module, action=None, *, parallel=False, extra=()):
    check_code(protocol)
    arguments = [sys.executable, "-u", "-m", PACKAGE + module]
    if action:
        arguments.append(action)
    arguments.extend(("--root", str(root), *extra))
    label = module + ("_" + action if action else "_source_report")
    if not parallel:
        state(root, label)
        with (root / "logs" / (label + ".log")).open("w") as handle:
            subprocess.run(
                arguments, check=True, stdout=handle, stderr=subprocess.STDOUT
            )
        return
    while not (devices := free_gpus(protocol["requested_gpus"])):
        state(root, "waiting_for_free_GPU", pending_stage=label)
        time.sleep(protocol["poll_seconds"])
    processes = []
    with ExitStack() as stack:
        for shard, gpu in enumerate(devices):
            environment = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
            handle = stack.enter_context(
                (root / "logs" / f"{label}_gpu{gpu}.log").open("w")
            )
            process = subprocess.Popen(
                arguments
                + [
                    "--shard",
                    str(shard),
                    "--shards",
                    str(len(devices)),
                    "--device",
                    "cuda:0",
                ],
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            processes.append((gpu, process))
        state(root, label, workers=[{"gpu": g, "pid": p.pid} for g, p in processes])
        while any(p.poll() is None for _, p in processes):
            time.sleep(protocol["poll_seconds"])
        results = [
            {"gpu": g, "pid": p.pid, "exit_code": p.returncode} for g, p in processes
        ]
        state(root, label + "_terminal", workers=results)
    if any(row["exit_code"] for row in results):
        raise RuntimeError(
            f"Comparison workers failed; inspect logs without repeating completed fits: {results}"
        )


def run(root, source_pids, requested_gpus, poll_seconds):
    path = root / "comparison_pipeline.json"
    if path.exists():
        raise RuntimeError(
            "A comparison pipeline was already registered; inspect its handle/state"
        )
    source = json.loads((root / "protocol.json").read_text())
    for name, expected in source["code_sha256"].items():
        if digest(BASE / name) != expected:
            raise RuntimeError(
                f"Source code differs from the executed source snapshot: {name}"
            )
    names = set(source["code_sha256"]) | {
        "shared_structure_common.py",
        "shared_structure_report.py",
        "shared_structure_pipeline.py",
    }
    identities = [pid_identity(pid) for pid in source_pids]
    if any(value is None for value in identities):
        raise RuntimeError(
            "A supplied source PID is already terminal; revalidate its session before registration"
        )
    protocol = {
        "registered_utc": datetime.now(UTC).isoformat(),
        "source_protocol_sha256": digest(root / "protocol.json"),
        "code_sha256": {name: digest(BASE / name) for name in sorted(names)},
        "source_processes": identities,
        "requested_gpus": requested_gpus,
        "poll_seconds": poll_seconds,
        "policy": "No source restart/termination or algorithm changes. Wait for all 240 source artifacts and terminal registered source processes, then source freeze; common register/banks/fresh-C fits/freeze/meta selection/freeze/late teacher audit/source-only report. Only GPUs without existing compute processes. No target fits before primary structural report.",
    }
    save_json(path, protocol)
    snapshot = root / "comparison_pipeline_snapshot"
    snapshot.mkdir()
    for name in sorted(names):
        shutil.copy2(BASE / name, snapshot / name)
    expected = [
        root / "extract" / f"job_{number:04d}.json"
        for number in range(len(source["extract_jobs"]))
    ]
    while True:
        live = [
            previous
            for previous in identities
            if pid_identity(previous["pid"]) == previous
        ]
        completed = sum(path.exists() for path in expected)
        state(
            root,
            "waiting_for_source",
            completed_jobs=completed,
            total_jobs=len(expected),
            live_source_pids=[p["pid"] for p in live],
        )
        if not live:
            if completed != len(expected):
                raise RuntimeError(
                    "Source workers are terminal with missing results; inspect their session/logs"
                )
            break
        time.sleep(poll_seconds)
    run_stage(root, protocol, "shared_structure_benchmark", "freeze_extract")
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
            protocol,
            "shared_structure_common",
            action,
            parallel=action in ("fit", "select"),
        )
    run_stage(root, protocol, "shared_structure_report", extra=("--source-only",))
    state(
        root,
        "structural_comparison_report_ready",
        report=str(root / "structure_report/REPORT.md"),
        overall_goal_complete=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-pids", type=int, nargs="+", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    try:
        run(args.root.resolve(), args.source_pids, args.gpus, args.poll_seconds)
    except Exception as error:
        state(args.root, "failed", error=str(error), automatically_restarted=False)
        raise


if __name__ == "__main__":
    main()
