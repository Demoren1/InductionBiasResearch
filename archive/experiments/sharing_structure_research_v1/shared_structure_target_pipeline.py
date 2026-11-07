"""Adapt frozen G23 indexators to fresh target tasks after both source reports."""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json

from .shared_structure_bilevel_report import LABELS, read_frozen_audit
from .shared_structure_pipeline import (
    BASE,
    PACKAGE,
    check_code,
    free_gpus,
    pid_identity,
    run_stage,
)
from .shared_structure_transfer import audit, banks, freeze, prepare, register


def status(root, stage, **details):
    record = {"utc": datetime.now(UTC).isoformat(), "stage": stage, **details}
    save_json(root / "target_pipeline_state.json", record)
    print(json.dumps(record), flush=True)


def register_queue(root, supplement):
    destination = root / "target_pipeline.json"
    if destination.exists():
        raise RuntimeError("Target queue already registered; inspect its session")
    previous = json.loads((root / "performance_continuation.json").read_text())
    check_code(previous)
    names = set(previous["code_sha256"]) | {
        "shared_structure_transfer.py",
        "shared_structure_target_batch.py",
        "shared_structure_target_pipeline.py",
    }
    protocol = {
        "registered_utc": datetime.now(UTC).isoformat(),
        "parent_supervisor": previous["supervisor"],
        "supervisor": pid_identity(os.getpid()),
        "primary": str(root),
        "supplement": str(supplement),
        "code_sha256": {name: digest(BASE / name) for name in sorted(names)},
        "requested_gpus": [0, 1, 2, 3],
        "reserved_gpus": [6, 7],
        "poll_seconds": 15,
        "policy": (
            "Wait for exact parent termination and both frozen source reports. "
            "Keep every selected I fixed. Generate eight new target tasks per I "
            "on CPU, with independently reset signed C. Same observed target "
            "panels for all methods. Measure serial versus batch on identical "
            "data/initials before adopting the faster compatible backend. Only "
            "free GPUs0--3. Reuse exact same-I/same-target numerical fits across "
            "reports only with frozen data/settings/initials/hash provenance. "
            "Freeze BOTH complete target fit sets before opening ANY test "
            "labels. No source update, teacher C, target-test HPO, automatic "
            "restart or repeat of completed stages. Overall goal remains active."
        ),
    }
    save_json(destination, protocol)
    snapshot = root / "target_pipeline_snapshot"
    snapshot.mkdir()
    for name in names:
        shutil.copy2(BASE / name, snapshot / name)
    return protocol


def register_phase(root, protocol, data_origin=None):
    register(root)
    phase = root / "transfer"
    path = phase / "protocol.json"
    plan = json.loads(path.read_text())
    names = (
        "shared_structure_numeric_batch.py",
        "shared_structure_target_batch.py",
        "shared_structure_target_pipeline.py",
    )
    plan["code_sha256"].update({name: protocol["code_sha256"][name] for name in names})
    plan["execution_policy"] = {
        "queue_sha256": digest(Path(protocol["primary"]) / "target_pipeline.json"),
        "target_generation_device": "cpu",
        "backend_selection": "Lowest synchronized runtime among numerically compatible serial/batch, without changing scientific settings",
        "indexators_per_batch": 4,
        "all_comparison_fits_frozen_before_either_test_audit": True,
    }
    if data_origin is not None:
        plan["same_target_observations"] = {
            "root": str(data_origin),
            "data_freeze_sha256": digest(data_origin / "transfer/data_frozen.json"),
        }
    save_json(path, plan)
    for name in names:
        shutil.copy2(BASE / name, phase / "executed_code_snapshot" / name)


def execution_measurement(root, protocol):
    while not (devices := free_gpus(protocol["requested_gpus"])):
        status(root, "waiting_for_free_GPU_for_target_measurement")
        time.sleep(protocol["poll_seconds"])
    gpu = devices[0]
    status(root, "target_execution_measurement", gpu=gpu)
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    with (root / "transfer/logs/execution_measurement.log").open("w") as handle:
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                PACKAGE + "shared_structure_target_batch",
                "profile",
                "--root",
                str(root),
                "--device",
                "cuda:0",
            ],
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def combined_report(root, supplement, plan):
    structures = read_frozen_audit(root) + read_frozen_audit(supplement)
    targets = []
    for origin in (root, supplement):
        value = json.loads((origin / "transfer/audit.json").read_text())
        if value["target_fit_freeze_sha256"] != digest(
            origin / "transfer/fit_frozen.json"
        ):
            raise RuntimeError("Target audit does not match its frozen fits")
        targets.extend(value["records"])
    rows = []
    for noise in plan["noise"]:
        for method in LABELS:
            cases = [
                row
                for row in structures
                if row["noise"] == noise and row["method"] == method
            ]
            adapted = [
                row
                for row in targets
                if row["noise"] == noise
                and row["method"] == method
                and row["scope"] == "common4"
            ]
            if len(cases) != len(plan["families"]) or len(adapted) != len(cases):
                raise RuntimeError("Incomplete structure/transfer comparison")
            complete = [row for row in adapted if row["complete"]]
            rows.append(
                {
                    "noise": noise,
                    "method": method,
                    "exact_I": sum(row["exact_I"] for row in cases),
                    "independent_I": len(cases),
                    "completed_target_tasks": len(complete) * plan["targets"],
                    "mean_target_clean_mse_complete_only": float(
                        np.mean([row["mean_target_clean_mse"] for row in complete])
                    )
                    if complete
                    else None,
                    "mean_target_excess_over_true_I_complete_only": float(
                        np.mean(
                            [row["mean_excess_mse_over_true_I"] for row in complete]
                        )
                    )
                    if complete
                    else None,
                    "families_with_negative_excess_over_true_I": sum(
                        row["mean_excess_mse_over_true_I"] < -1e-8 for row in complete
                    ),
                }
            )
    report = root / "transfer/combined_report"
    report.mkdir(exist_ok=False)
    with (report / "comparison.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = [
        "# Frozen-I transfer on eight fresh target codebooks",
        "",
        (
            "All source indexators were selected and frozen before target work. "
            "Every target C was reset; support4096, validation1024, 32signed starts, "
            "GN24 and clean test8192. Both comparison fit sets were frozen before "
            "either test audit. True-I reference uses the same numerical procedure "
            "and is not a certified global C optimum. Exact-I denominators are six "
            "independent geometries; the eight targets within each I are dependent "
            "for assessing generalization of the structure search."
        ),
        "",
        "|SD|Method|Exact I|Target tasks|Clean MSE|Excess over true I|",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        risk = row["mean_target_clean_mse_complete_only"]
        excess = row["mean_target_excess_over_true_I_complete_only"]
        risk_text = "NA" if risk is None else f"{risk:.6g}"
        excess_text = "NA" if excess is None else f"{excess:.6g}"
        text.append(
            f"|{row['noise']}|{LABELS[row['method']]}|"
            f"{row['exact_I']}/{row['independent_I']}|"
            f"{row['completed_target_tasks']}|{risk_text}|{excess_text}|"
        )
    text.extend(
        [
            "",
            (
                "Same-G23 pilot; known bias/readout, signed competitor adaptations "
                "and high source query cost remain limitations. Compute is not "
                "matched. New target tasks do not constitute independent new-I "
                "confirmation. Numerical parity measurement and all source/target "
                "protocols, freezes, actual optimizer trajectories and cache "
                "provenance are retained. Overall research goal remains active."
            ),
        ]
    )
    (report / "REPORT.md").write_text("\n".join(text) + "\n")


def run(root, supplement):
    protocol = register_queue(root, supplement)
    parent = protocol["parent_supervisor"]
    while True:
        live = pid_identity(parent["pid"]) == parent
        ready = (supplement / "report/REPORT.md").exists()
        status(root, "waiting_for_bilevel_report", parent_live=live, report_ready=ready)
        if not live:
            if not ready or not (root / "structure_report/REPORT.md").exists():
                raise RuntimeError("Parent terminal without both source reports")
            break
        time.sleep(protocol["poll_seconds"])
    check_code(protocol)
    torch.set_num_threads(1)
    plan = json.loads((root / "protocol.json").read_text())
    register_phase(root, protocol)
    status(root, "preparing_fresh_target_observations_cpu")
    for family in range(len(plan["families"])):
        prepare(root, plan, family, "cpu")
    freeze(root, plan, "data")
    banks(root, plan)
    register_phase(supplement, protocol, data_origin=root)
    (supplement / "transfer/data").rmdir()
    (supplement / "transfer/data").symlink_to(
        root / "transfer/data", target_is_directory=True
    )
    supplement_plan = json.loads((supplement / "protocol.json").read_text())
    freeze(supplement, supplement_plan, "data")
    banks(supplement, supplement_plan)
    execution_measurement(root, protocol)
    measurement = json.loads((root / "transfer/execution_measurement.json").read_text())
    save_json(
        supplement / "transfer/execution_measurement.json",
        {
            **measurement,
            "phase_protocol_sha256": digest(supplement / "transfer/protocol.json"),
            "reused_execution_measurement_sha256": digest(
                root / "transfer/execution_measurement.json"
            ),
            "timing_origin": str(root),
        },
    )
    for origin, extra in (
        (root, ()),
        (supplement, ("--reuse-root", str(root))),
    ):
        run_stage(
            origin,
            protocol,
            "shared_structure_target_batch",
            "fit",
            parallel=True,
            extra=("--batch-size", "4", *extra),
        )
        origin_plan = json.loads((origin / "protocol.json").read_text())
        freeze(origin, origin_plan, "fit")
    status(root, "both_target_fit_sets_frozen_before_test_audit")
    for origin, origin_plan in ((root, plan), (supplement, supplement_plan)):
        audit(origin, origin_plan, "cpu")
    run_stage(root, protocol, "shared_structure_report")
    combined_report(root, supplement, plan)
    status(
        root,
        "fresh_target_comparison_ready",
        report=str(root / "transfer/combined_report/REPORT.md"),
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
        status(
            root,
            "target_pipeline_stopped",
            error=repr(error),
            automatically_restarted=False,
        )
        raise


if __name__ == "__main__":
    main()
