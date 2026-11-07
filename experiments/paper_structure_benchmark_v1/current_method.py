"""Run the unchanged G34 method with measured cost and access boundaries."""

import argparse
import json
import os
import resource
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)
from experiments.sharing_structure_research_v1.blind_runtime import (
    ObservationBoundary,
)


def run(root, case, condition, phase, device):
    torch.set_num_threads(2)
    print(
        json.dumps(
            {
                "pid": os.getpid(),
                "case": case,
                "phase": phase,
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            }
        ),
        flush=True,
    )
    project = Path(__file__).resolve().parents[2]
    registration = json.loads((root / "registered_sources.json").read_text())
    for name, expected in registration["code_sha256"].items():
        learner = "/sharing_structure_research_v1/blind_" in name
        learner |= name.endswith("/current_method.py")
        if learner and digest(project / name) != expected:
            raise ValueError(f"Registered code changed: {name}")
    boundary = ObservationBoundary(root, case, condition)
    boundary.phase = phase
    boundary.role = "select" if phase == "select" else "train"
    boundary.install()
    if device.type == "cuda":
        torch.zeros(1, device=device)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    counter = FlopCounterMode(display=False)
    status = "failed"
    try:
        with counter:
            if phase == "source":
                from experiments.sharing_structure_research_v1 import (
                    blind_dense,
                    blind_dense_extract,
                    blind_dense_repair,
                )

                for name, operation in (
                    (
                        "adam",
                        lambda: blind_dense.run(root, case, condition, device, 2500),
                    ),
                    (
                        "polish",
                        lambda: blind_dense_extract.run(
                            root, case, condition, device, 32, polish_only=True
                        ),
                    ),
                    (
                        "revival",
                        lambda: blind_dense_repair.run(
                            root, case, condition, device, 2
                        ),
                    ),
                    (
                        "descriptor",
                        lambda: blind_dense_extract.run(
                            root, case, condition, device, 0, from_repair=True
                        ),
                    ),
                ):
                    boundary.phase = name
                    stage_started = time.perf_counter()
                    operation()
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    print(
                        json.dumps(
                            {
                                "case": case,
                                "phase": name,
                                "seconds": time.perf_counter() - stage_started,
                            }
                        ),
                        flush=True,
                    )
            else:
                from experiments.sharing_structure_research_v1 import blind_joint_bank

                operations = {
                    "seeds": lambda: blind_joint_bank.initialize_bank(
                        root, case, condition, device
                    ),
                    "proposals": lambda: blind_joint_bank.propose(
                        root, case, condition
                    ),
                    "fit": lambda: blind_joint_bank.fit(root, case, condition, device),
                    "select": lambda: blind_joint_bank.select(root, case, condition),
                }
                operations[phase]()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        status = "complete"
    finally:
        metrics = {
            "status": status,
            "case": case,
            "condition": condition,
            "phase": phase,
            "completed_utc": datetime.now(UTC).isoformat(),
            "wall_seconds": time.perf_counter() - started,
            "torch_operator_flops": counter.get_total_flops(),
            "flop_scope": "PyTorch registered operators; excludes NumPy/SciPy and unsupported operations",
            "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
            if device.type == "cuda"
            else 0,
        }
        save_json(root / "metrics" / f"{case}_c{condition}_{phase}.json", metrics)
        boundary.save(
            root / "receipts" / f"paper_{case}_c{condition}_{phase}.json", status
        )
        print(json.dumps(metrics), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--condition", type=int, default=0)
    parser.add_argument(
        "--phase",
        choices=("source", "seeds", "proposals", "fit", "select"),
        required=True,
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(
        args.root.resolve(),
        args.case,
        args.condition,
        args.phase,
        torch.device(args.device),
    )


if __name__ == "__main__":
    main()
