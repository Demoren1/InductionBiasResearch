"""Run the frozen T/V supplement after the primary G23 structural report."""

import argparse
import json
import os
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

from experiments.darts_structured_tasks_v1.data import digest, save_json

from .shared_structure_pipeline import check_code, pid_identity, run_stage, state


def run(root):
    registered = root / "queued_pipeline.json"
    if registered.exists():
        raise RuntimeError("Supplement runner already registered; inspect its handle")
    source = json.loads((root / "protocol.json").read_text())
    check_code(source)
    parent = Path(source["parent"])
    diagnostic = json.loads((root / "local_derivative_diagnostic.json").read_text())
    if diagnostic["protocol_sha256"] != digest(root / "protocol.json"):
        raise RuntimeError("Derivative diagnostic does not match source registration")
    original = json.loads((parent / "gpu_restriction_processes.json").read_text())[
        "supervisor"
    ]
    if pid_identity(original["pid"]) != original:
        raise RuntimeError(
            "Parent runner is not live; inspect its existing handle first"
        )
    protocol = {
        "registered_utc": datetime.now(UTC).isoformat(),
        "source_protocol_sha256": digest(root / "protocol.json"),
        "code_sha256": source["code_sha256"],
        "parent_supervisor": original,
        "supervisor": pid_identity(os.getpid()),
        "requested_gpus": [0, 1, 2, 3],
        "reserved_gpus": [6, 7],
        "poll_seconds": 30,
        "policy": "Do not delay primary structure table. Wait for the exact registered parent supervisor to terminate with frozen structural report. Then supplemental native extraction, source freeze, fresh-C bank fits, fit freeze, independent meta selection, whole-I freeze, late audit/report. Only free GPUs0--3, never6/7 or occupied4/5. No automatic restart or repeat completed jobs.",
    }
    save_json(registered, protocol)
    shutil.copy2(Path(__file__), root / "queued_pipeline_snapshot.py")
    while True:
        live = pid_identity(original["pid"]) == original
        ready = (parent / "structure_report/REPORT.md").exists()
        state(
            root,
            "waiting_for_primary_structure_report",
            parent_supervisor_live=live,
            parent_report_exists=ready,
            reserved_gpus=[6, 7],
        )
        if not live:
            if not ready or not (parent / "common/selection_frozen.json").exists():
                raise RuntimeError(
                    "Parent terminal without primary report; inspect its existing session"
                )
            break
        time.sleep(protocol["poll_seconds"])
    run_stage(root, protocol, "shared_structure_bilevel", "extract", parallel=True)
    for action in (
        "freeze_extract",
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
            "shared_structure_bilevel",
            action,
            parallel=action in ("fit", "select"),
        )
    run_stage(root, protocol, "shared_structure_bilevel_report")
    state(
        root,
        "supplemental_structural_report_ready",
        report=str(root / "report/REPORT.md"),
        overall_goal_complete=False,
        reserved_gpus=[6, 7],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        run(root)
    except BaseException as error:
        state(
            root,
            "supplemental_pipeline_stopped",
            error=repr(error),
            automatically_restarted=False,
        )
        raise


if __name__ == "__main__":
    main()
