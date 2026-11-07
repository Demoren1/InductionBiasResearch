"""Preserve live G23 source work and use measured faster fixed-I execution."""

import argparse
import json
import os
import shutil
import signal
import time
from datetime import UTC, datetime
from pathlib import Path

from experiments.darts_structured_tasks_v1.data import digest, save_json

from .shared_structure_pipeline import BASE, check_code, pid_identity, run_stage, state


def register(root, supplement):
    destination = root / "performance_continuation.json"
    if destination.exists():
        raise RuntimeError(
            "Performance continuation already registered; inspect its handle"
        )
    parent = json.loads((root / "gpu_restriction_continuation.json").read_text())
    code = json.loads((supplement / "protocol.json").read_text())
    check_code(parent)
    check_code(code)
    timing = json.loads((root / "batched_numeric_performance.json").read_text())
    if not timing["parity_passed"] or timing["speedup"] <= 1:
        raise RuntimeError("No measured faster numerically compatible execution")
    if timing["batched_code_sha256"] != digest(
        BASE / "shared_structure_numeric_batch.py"
    ):
        raise RuntimeError(
            "Batch code changed after its performance/parity measurement"
        )
    if (root / "common").exists():
        raise RuntimeError(
            "Common numerical phase already started; do not interrupt its fits"
        )
    parent_processes = json.loads((root / "gpu_restriction_processes.json").read_text())
    queued = json.loads((supplement / "queued_pipeline.json").read_text())
    superseded = [
        {
            "root": str(supplement),
            "identity": queued["supervisor"],
            "waiting_stage": "waiting_for_primary_structure_report",
        },
        {
            "root": str(root),
            "identity": parent_processes["supervisor"],
            "waiting_stage": "waiting_for_source_on_GPU_0_to_3",
        },
    ]
    for previous in superseded:
        identity = previous["identity"]
        current = json.loads(
            (Path(previous["root"]) / "comparison_pipeline_state.json").read_text()
        )
        if (
            pid_identity(identity["pid"]) != identity
            or current["stage"] != previous["waiting_stage"]
        ):
            raise RuntimeError(
                "Only the two verified waiting CPU supervisors may be replaced"
            )
    sources = list(parent["source_processes"])
    sources.extend(
        {key: row[key] for key in ("pid", "start_time_ticks", "command_sha256")}
        for row in parent_processes["source_workers"]
    )
    names = (
        set(parent["code_sha256"])
        | set(code["code_sha256"])
        | {
            "shared_structure_numeric_batch.py",
            "numeric_batch_diagnostic.py",
            "shared_structure_performance_pipeline.py",
        }
    )
    protocol = {
        "registered_utc": datetime.now(UTC).isoformat(),
        "source_protocol_sha256": digest(root / "protocol.json"),
        "supplement": str(supplement),
        "supplement_protocol_sha256": digest(supplement / "protocol.json"),
        "performance_measurement_sha256": digest(
            root / "batched_numeric_performance.json"
        ),
        "code_sha256": {name: digest(BASE / name) for name in sorted(names)},
        "source_processes": sources,
        "superseded_waiting_supervisors": superseded,
        "supervisor": pid_identity(os.getpid()),
        "requested_gpus": [0, 1, 2, 3],
        "reserved_gpus": [6, 7],
        "poll_seconds": 30,
        "numeric_indexator_batch_size": 4,
        "policy": "Replace only two verified waiting CPU supervisors. Never stop/restart a live source worker. Preserve all completed data/results and source code/seeds/settings. Run same common objective/fresh signed starts/validation choices with independent vectorized indexator/task/start axes and active-model compaction. Register numerical execution before any common fits or meta-Y. Then unchanged meta/freeze/late-audit/report, followed by already registered T/V supplement. No teacher-driven settings or automatic repeats. GPU0--3 only.",
    }
    save_json(destination, protocol)
    snapshot = root / "performance_continuation_snapshot"
    snapshot.mkdir()
    for name in (
        "shared_structure_numeric_batch.py",
        "numeric_batch_diagnostic.py",
        "shared_structure_performance_pipeline.py",
    ):
        shutil.copy2(BASE / name, snapshot / name)
    shutil.copy2(
        root / "batched_numeric_performance.json",
        snapshot / "performance_measurement.json",
    )
    return protocol


def stop_waiting_supervisors(protocol):
    for previous in protocol["superseded_waiting_supervisors"]:
        identity = previous["identity"]
        current = json.loads(
            (Path(previous["root"]) / "comparison_pipeline_state.json").read_text()
        )
        if (
            pid_identity(identity["pid"]) != identity
            or current["stage"] != previous["waiting_stage"]
        ):
            raise RuntimeError(
                "Supervisor identity/state changed; no process was signaled"
            )
        os.kill(identity["pid"], signal.SIGINT)
        deadline = time.monotonic() + 10
        while pid_identity(identity["pid"]) == identity:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    "Waiting supervisor has not terminated; no worker was stopped"
                )
            time.sleep(0.1)


def register_batched_execution(root, protocol):
    path = root / "common/protocol.json"
    common = json.loads(path.read_text())
    if list((root / "common/fits").glob("job_*")):
        raise RuntimeError("Do not change numeric registration after fits have started")
    common["batched_numeric_execution"] = {
        "code_sha256": protocol["code_sha256"]["shared_structure_numeric_batch.py"],
        "indexators_per_batch": protocol["numeric_indexator_batch_size"],
        "measurement_sha256": protocol["performance_measurement_sha256"],
        "policy": "Same uniform query loss, data, fresh signed starts, code cap, GN/ridge, per-start backtracking and validation checkpoint. Only vectorization/active compaction and delayed CPU history copies differ. Floating summation differs; measured selected C within5.33e-15, identical selected starts, same training trajectories within2.25e-14 on the declared diagnosis. This is local numerical parity evidence, not a guarantee of bitwise equality on every case.",
    }
    common["code_sha256"]["shared_structure_numeric_batch.py"] = protocol[
        "code_sha256"
    ]["shared_structure_numeric_batch.py"]
    save_json(path, common)
    shutil.copy2(
        BASE / "shared_structure_numeric_batch.py",
        root / "common/executed_code_snapshot/shared_structure_numeric_batch.py",
    )


def finish_common(root, protocol, module):
    if module == "shared_structure_common":
        for action in ("register", "banks"):
            run_stage(root, protocol, module, action)
    else:
        run_stage(root, protocol, module, "banks")
    register_batched_execution(root, protocol)
    run_stage(
        root,
        protocol,
        "shared_structure_numeric_batch",
        parallel=True,
        extra=("--batch-size", str(protocol["numeric_indexator_batch_size"])),
    )
    for action in ("freeze_fit", "select", "freeze_selection", "audit"):
        run_stage(root, protocol, module, action, parallel=action == "select")


def run(root, supplement):
    protocol = register(root, supplement)
    stop_waiting_supervisors(protocol)
    count = len(json.loads((root / "protocol.json").read_text())["extract_jobs"])
    while True:
        live = [
            row["pid"]
            for row in protocol["source_processes"]
            if pid_identity(row["pid"]) == row
        ]
        complete = len(list((root / "extract").glob("job_*.json")))
        state(
            root,
            "waiting_for_source_with_batched_numeric_continuation",
            live_source_pids=live,
            completed_jobs=complete,
            total_jobs=count,
            reserved_gpus=[6, 7],
        )
        if not live:
            if complete != count:
                raise RuntimeError(
                    "Source workers terminal with missing results; inspect existing logs"
                )
            break
        time.sleep(protocol["poll_seconds"])
    run_stage(root, protocol, "shared_structure_benchmark", "freeze_extract")
    finish_common(root, protocol, "shared_structure_common")
    run_stage(root, protocol, "shared_structure_report", extra=("--source-only",))
    state(
        root,
        "structural_comparison_report_ready",
        report=str(root / "structure_report/REPORT.md"),
        overall_goal_complete=False,
        reserved_gpus=[6, 7],
    )
    state(
        supplement,
        "starting_under_batched_performance_continuation",
        supervisor=protocol["supervisor"],
        reserved_gpus=[6, 7],
    )
    run_stage(
        supplement, protocol, "shared_structure_bilevel", "extract", parallel=True
    )
    run_stage(supplement, protocol, "shared_structure_bilevel", "freeze_extract")
    finish_common(supplement, protocol, "shared_structure_bilevel")
    run_stage(supplement, protocol, "shared_structure_bilevel_report")
    state(
        supplement,
        "supplemental_structural_report_ready",
        report=str(supplement / "report/REPORT.md"),
        overall_goal_complete=False,
        reserved_gpus=[6, 7],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    args = parser.parse_args()
    root, supplement = args.root.resolve(), args.supplement.resolve()
    try:
        run(root, supplement)
    except BaseException as error:
        state(
            root,
            "performance_continuation_stopped",
            error=repr(error),
            automatically_restarted=False,
        )
        raise


if __name__ == "__main__":
    main()
