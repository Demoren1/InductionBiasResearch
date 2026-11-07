"""Continue only frozen G21 states with an unfinished continuous QP."""

import argparse
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.darts_structured_tasks_v1.evaluate import align_indices

from .count_metric import freeze, oracle_case, source_data
from .count_projection import census
from .count_qp_active import function_qp
from .prefix_census import forward, make_teacher, numeric_prefix

BASE = Path(__file__).resolve().parent


def register(root, parent):
    original = json.loads((parent / "protocol.json").read_text())
    if not (parent / "models_frozen.json").exists():
        raise RuntimeError("All initial G21 states must be frozen")
    if (parent / "audit.json").exists():
        raise RuntimeError("Register numerical continuation before G21 teacher audit")
    root.mkdir(parents=True, exist_ok=False)
    plan = {
        **original,
        "stage": "G21 numerical continuation of QP-gap failures only, before teacher audit",
        "registered_utc": datetime.now(UTC).isoformat(),
        "root_primary": str(parent.resolve()),
        "primary_freeze_sha256": digest(parent / "models_frozen.json"),
        "active_iterations": 1000,
        "active_zero_tolerance": 1e-12,
        "active_direction_tolerance": 1e-12,
        "face_stationarity": "Check reduced gradient <= qp_gap_absolute/(4*hidden) before multiplier releases; final unchanged full FW-gap is mandatory.",
        "continuation": "Only last_reason qp_not_converged; preserve all accepted prefixes/C/history, restart numerical QP from feasible integer initializer at the first failed prefix. Keep full/rank-failed states unchanged. Same observations/loss/gap; more solver compute. No teacher metrics opened before registration.",
        "theory_sha256": digest(BASE / "THEORY.md"),
        "code_sha256": {
            name: digest(BASE / name)
            for name in (
                "count_numeric_continuation.py",
                "count_qp_active.py",
                "count_projection.py",
                "count_metric.py",
            )
        },
    }
    selected = {}
    for job_id in range(len(plan["jobs"])):
        path = parent / "models" / f"job_{job_id:03d}.json"
        meta = json.loads(path.read_text())
        selected[str(job_id)] = [
            row["key"]
            for row in meta["records"]
            if row["trace"][-1]["reason"] == "qp_not_converged"
        ]
    plan["selected_source_failures"] = selected
    save_json(root / "protocol.json", plan)
    shutil.copy2(BASE / "THEORY.md", root / "theory_at_registration.md")
    snapshot = root / "executed_code_snapshot"
    snapshot.mkdir()
    for name in plan["code_sha256"]:
        shutil.copy2(BASE / name, snapshot / name)
    (root / "models").mkdir()


@torch.no_grad()
def run_case(root, plan, job_id, device):
    started = time.monotonic()
    parent = Path(plan["root_primary"])
    if digest(parent / "models_frozen.json") != plan["primary_freeze_sha256"]:
        raise RuntimeError("Initial G21 freeze changed")
    frozen = json.loads((parent / "models_frozen.json").read_text())
    path = parent / "models" / f"job_{job_id:03d}"
    for suffix in (".json", ".npz"):
        current = path.with_suffix(suffix)
        if digest(current) != frozen["sha256"][str(current.relative_to(parent))]:
            raise RuntimeError("Initial G21 source changed")
    meta = json.loads(path.with_suffix(".json").read_text())
    old, targets, vx, vy, _ = source_data(plan, job_id, device)
    records, arrays = [], {}
    with np.load(path.with_suffix(".npz")) as previous:
        for source in meta["records"]:
            if source["key"] not in plan["selected_source_failures"][str(job_id)]:
                continue
            sigma = source["panel"]["noise"]
            resume = {
                "indices": previous[source["key"] + "__indices"],
                "codebooks": previous[source["key"] + "__codebooks"],
                "trace": source["trace"][:-1],
                "centers": source["final_C"],
            }
            local_started = time.monotonic()
            output = census(
                old["slices"],
                targets[sigma],
                vx,
                vy[sigma],
                old["order"],
                source["initial_C"],
                source["method"],
                plan,
                qp_solver=function_qp,
                resume=resume,
            )
            record = {
                **source,
                "final_C": output["centers"].tolist(),
                "complete": output["complete"],
                "trace": output["trace"],
                "resumed_prefix_length": len(resume["indices"]),
                "original_failure": source["trace"][-1],
                "additional_seconds": time.monotonic() - local_started,
                "original_seconds": source["seconds"],
            }
            record["seconds"] = (
                record["original_seconds"] + record["additional_seconds"]
            )
            if output["complete"]:
                weights = numeric_prefix(
                    output["indices"][-1], output["centers"], device
                )
                prediction = forward(old["validation_x"], weights, plan)
                observed = old["validation_clean"] + sigma * old["validation_noise"]
                record["iid_validation_mse"] = float(
                    (prediction - observed).square().mean()
                )
            arrays[source["key"] + "__indices"] = output["indices"]
            arrays[source["key"] + "__codebooks"] = output["codebooks"]
            records.append(record)
            print(
                json.dumps(
                    {
                        "job": job_id,
                        "key": source["key"],
                        "from_prefix": len(resume["indices"]),
                        "to_prefix": len(output["indices"]),
                        "complete": output["complete"],
                        "reason": output["trace"][-1]["reason"],
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    destination = root / "models" / f"job_{job_id:03d}"
    np.savez_compressed(destination.with_suffix(".npz"), **arrays)
    save_json(
        destination.with_suffix(".json"),
        {
            "job": plan["jobs"][job_id],
            "records": records,
            "seconds": time.monotonic() - started,
            "protocol_sha256": digest(root / "protocol.json"),
            "primary_freeze_sha256": plan["primary_freeze_sha256"],
        },
    )


def audit(root, plan):
    parent = Path(plan["root_primary"])
    original = json.loads((parent / "audit.json").read_text())
    records = {(row["job_id"], row["key"]): row for row in original["records"]}
    frozen = json.loads((root / "models_frozen.json").read_text())
    arrays = {}
    for job_id, job in enumerate(plan["jobs"]):
        path = root / "models" / f"job_{job_id:03d}"
        for suffix in (".json", ".npz"):
            current = path.with_suffix(suffix)
            if digest(current) != frozen["sha256"][str(current.relative_to(root))]:
                raise RuntimeError("Frozen continuation changed")
        meta = json.loads(path.with_suffix(".json").read_text())
        _, truth, true_C = make_teacher(job, plan["g19_plan"], "cpu")
        expected_C = np.sort(np.abs(true_C))
        with np.load(path.with_suffix(".npz")) as data:
            for source in meta["records"]:
                indices = data[source["key"] + "__indices"]
                record = {
                    "job_id": job_id,
                    "job": job,
                    "key": source["key"],
                    "panel": source["panel"],
                    "method": source["method"],
                    "complete": source["complete"],
                    "exact_I": False,
                    "completed_prefixes": len(indices),
                    "initial_C_max_error": float(
                        np.max(np.abs(np.sort(source["initial_C"]) - expected_C))
                    ),
                    "final_C_max_error": float(
                        np.max(np.abs(np.sort(source["final_C"]) - expected_C))
                    ),
                    "full_available_source_labels": source[
                        "full_available_source_labels"
                    ],
                    "iid_validation_mse": source.get("iid_validation_mse"),
                    "last_reason": source["trace"][-1]["reason"],
                    "numerical_continuation": True,
                    "resumed_prefix_length": source["resumed_prefix_length"],
                }
                if source["complete"]:
                    metric, aligned, _ = align_indices(
                        indices[-1], truth, plan["codes"]
                    )
                    record.update(metric)
                    record["exact_I"] = metric["categorical_errors"] == 0
                    arrays[f"job_{job_id}__{source['key']}__aligned"] = aligned
                records[(job_id, source["key"])] = record
    ordered = [records[(row["job_id"], row["key"])] for row in original["records"]]
    np.savez_compressed(root / "audit_only.npz", **arrays)
    save_json(
        root / "audit.json",
        {
            "records": ordered,
            "replaced_states": sum(
                row.get("numerical_continuation", False) for row in ordered
            ),
            "models_frozen_sha256": digest(root / "models_frozen.json"),
            "primary_audit_sha256": digest(parent / "audit.json"),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("register", "run", "freeze", "audit", "oracle")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "register":
        register(args.root, args.parent)
        return
    plan = json.loads((args.root / "protocol.json").read_text())
    if args.action in ("freeze", "audit"):
        destination = args.root / (
            "models_frozen.json" if args.action == "freeze" else "audit.json"
        )
        if destination.exists():
            raise RuntimeError("Continuation phase already completed")
        if args.action == "freeze":
            freeze(args.root, plan)
        else:
            audit(args.root, plan)
        return
    for job_id in range(args.shard, len(plan["jobs"]), args.shards):
        suffix = "_oracle.json" if args.action == "oracle" else ".json"
        if (args.root / "models" / (f"job_{job_id:03d}" + suffix)).exists():
            raise RuntimeError("Refusing to repeat completed continuation")
        if args.action == "run":
            run_case(args.root, plan, job_id, args.device)
        else:
            oracle_case(args.root, plan, job_id, args.device, qp_solver=function_qp)


if __name__ == "__main__":
    main()
