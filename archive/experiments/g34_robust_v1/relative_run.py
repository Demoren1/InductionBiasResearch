"""Add a relative-residual variant before opening any new query scores."""

import argparse
import json
import os
import resource
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from experiments.sharing_structure_research_v1 import blind_joint_bank
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)
from experiments.sharing_structure_research_v1.blind_runtime import (
    ObservationBoundary,
)

from . import relative_bank
from .run import PROJECT


def setup(base):
    root = base / "relative"
    root.mkdir(exist_ok=False)
    registration = json.loads((base / "registration.json").read_text())
    files = (Path(__file__), Path(__file__).with_name("relative_bank.py"))
    amendment = {
        "utc": datetime.now(UTC).isoformat(),
        "reason": (
            "A positive denominator floor can suppress entire neurons when "
            "code and scale approach zero. This additional variant uses the "
            "relative residual of the original geometry ranking instead."
        ),
        "status": "Registered before any new target fitting or scoring",
        "earlier_source_results_seen": True,
        "new_target_results_seen": False,
        "code_sha256": {
            str(path.relative_to(PROJECT)): digest(path) for path in files
        },
        "scale_domain": "abs(a)>=1e-6",
        "rho": "4 log(1+r^2/4), unchanged",
        "role": "Additional prescoring comparison; original variants retained",
    }
    save_json(base / "relative_amendment.json", amendment)
    save_json(root / "relative_amendment.json", amendment)
    save_json(root / "registration.json", registration)
    for path in files:
        target = base / "code_snapshot" / path.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    source = base / "irls"
    export = json.loads((source / "source_export.json").read_text())
    for name in export["copied_sha256"]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    save_json(root / "source_export.json", export)
    shutil.copy2(source / "protocol.json", root / "protocol.json")
    for name in ("metrics", "logs", "receipts", "targets"):
        (root / name).mkdir()
    protocol = json.loads((root / "protocol.json").read_text())
    plan = json.loads((base / "source_plan.json").read_text())
    for sizes in protocol["cases"]:
        for condition in protocol["conditions"]:
            case = sizes["name"]
            key = f"{case}_c{condition}"
            plan["jobs"].append({
                "id": f"{key}_relative",
                "module": "experiments.g34_robust_v1.relative_run",
                "args": [
                    "--base", str(root), "--case", case,
                    "--condition", str(condition),
                    "--method", "relative", "--phase", "source",
                ],
                "registration": str(root / "fits" / key / "started.json"),
                "metric": str(root / "metrics" / f"{key}_pipeline.json"),
                "log": str(root / "logs" / f"{key}_pipeline.log"),
            })
    save_json(base / "source_plan.json", plan)
    print(json.dumps({"source_jobs": len(plan["jobs"])}))


def source(root, case, condition):
    torch.set_num_threads(2)
    registration = json.loads((root / "registration.json").read_text())
    amendment = json.loads((root / "relative_amendment.json").read_text())
    for name, expected in {
        **registration["core_sha256"],
        **registration["learner_sha256"],
        **amendment["code_sha256"],
    }.items():
        if digest(PROJECT / name) != expected:
            raise RuntimeError(f"Registered code changed: {name}")
    boundary = ObservationBoundary(root, case, condition)
    boundary.install()
    key = f"{case}_c{condition}"
    save_json(root / "fits" / key / "started.json", {
        "utc": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "method": "relative",
    })
    device = torch.device("cuda:0")
    torch.zeros(1, device=device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    status, stages = "failed", {}
    counter = FlopCounterMode(display=False)
    try:
        for phase in ("proposals", "fit", "select"):
            boundary.phase = phase
            boundary.role = "select" if phase == "select" else "train"
            tick = time.perf_counter()
            if phase == "proposals":
                relative_bank.propose(root, case, condition, robust=True)
            elif phase == "fit":
                with counter:
                    blind_joint_bank.fit(root, case, condition, device)
            else:
                blind_joint_bank.select(root, case, condition)
            torch.cuda.synchronize(device)
            stages[phase] = time.perf_counter() - tick
        status = "complete"
    finally:
        save_json(root / "metrics" / f"{key}_pipeline.json", {
            "status": status,
            "method": "relative",
            "wall_seconds": time.perf_counter() - started,
            "stage_seconds": stages,
            "torch_operator_flops": counter.get_total_flops(),
            "cpu_peak_rss_bytes": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss * 1024,
            "cuda_peak_allocated_bytes": (
                torch.cuda.max_memory_allocated(device)
            ),
        })
        boundary.save(root / "receipts" / f"{key}.json", status)


def export_targets(base):
    root = base / "relative"
    registration = json.loads((base / "registration.json").read_text())
    reference = Path(registration["reference"])
    protocol = json.loads((root / "protocol.json").read_text())
    plan = json.loads((base / "target_plan.json").read_text())
    for sizes in protocol["cases"]:
        for condition in protocol["conditions"]:
            case, key = sizes["name"], f"{sizes['name']}_c{condition}"
            selected = root / "fits" / key / "joint_selected.npz"
            folder = root / "targets" / key
            folder.mkdir(exist_ok=False)
            np.savez_compressed(
                folder / "structures.npz",
                relative=np.load(selected)["joint_functional_index"],
            )
            for name in (
                "support.npz", "selection.npz", "query_x.npz", "sizes.json"
            ):
                shutil.copy2(
                    reference / "targets" / key / name, folder / name
                )
            save_json(folder / "ready.json", {
                "utc": datetime.now(UTC).isoformat(),
                "source_sha256": digest(selected),
                "target_query_answers_exported": False,
                "structural_truth_accessed": False,
            })
            plan["jobs"].append({
                "id": f"{key}_relative",
                "module": (
                    "experiments.paper_structure_benchmark_v1"
                    ".fitting.target_fit"
                ),
                "args": [
                    "--root", str(root), "--case", case,
                    "--condition", str(condition), "--method", "relative",
                ],
                "registration": str(
                    folder / "relative" / "registered.json"
                ),
                "metric": str(folder / "relative" / "metrics.json"),
                "dependencies": [str(folder / "ready.json")],
                "log": str(root / "logs" / f"{key}_target.log"),
            })
    save_json(base / "target_plan.json", plan)
    print(json.dumps({"target_jobs": len(plan["jobs"])}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--case")
    parser.add_argument("--condition", type=int, default=0)
    parser.add_argument("--method", choices=("relative",), default="relative")
    parser.add_argument(
        "--phase", choices=("setup", "source", "export"), required=True
    )
    args = parser.parse_args()
    if args.phase == "setup":
        setup(args.base.resolve())
    elif args.phase == "export":
        export_targets(args.base.resolve())
    else:
        source(args.base.resolve(), args.case, args.condition)


if __name__ == "__main__":
    main()
