"""Run the original dense preparation and G34 on exactly one source task."""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from experiments.sharing_structure_research_v1 import (
    blind_dense,
    blind_dense_extract,
    blind_dense_repair,
    blind_joint_bank,
)
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)
from experiments.sharing_structure_research_v1.blind_runtime import (
    ObservationBoundary,
)


def run(root, case, phase, device):
    torch.set_num_threads(2)
    boundary = ObservationBoundary(root, case, 0)
    boundary.install()
    started = time.perf_counter()
    torch.zeros(1, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    if phase == "source":
        operations = [
            ("adam", lambda: blind_dense.run(root, case, 0, device, 2500)),
            (
                "polish",
                lambda: blind_dense_extract.run(
                    root, case, 0, device, 32, polish_only=True
                ),
            ),
            (
                "repair",
                lambda: blind_dense_repair.run(root, case, 0, device, 2),
            ),
            (
                "descriptor",
                lambda: blind_dense_extract.run(
                    root, case, 0, device, 0, from_repair=True
                ),
            ),
        ]
    else:
        operations = [
            (
                "initialize",
                lambda: blind_joint_bank.initialize_bank(
                    root, case, 0, device
                ),
            ),
            ("propose", lambda: blind_joint_bank.propose(root, case, 0)),
            ("fit", lambda: blind_joint_bank.fit(root, case, 0, device)),
            ("select", lambda: blind_joint_bank.select(root, case, 0)),
        ]
    status = "failed"
    stages = []
    try:
        for name, operation in operations:
            boundary.phase = name
            boundary.role = "select" if name == "select" else "train"
            tick = time.perf_counter()
            operation()
            torch.cuda.synchronize(device)
            stages.append(
                {"stage": name, "seconds": time.perf_counter() - tick}
            )
            print(
                json.dumps({"case": case, "phase": phase, **stages[-1]}),
                flush=True,
            )
        status = "complete"
    finally:
        save_json(
            root / "metrics" / f"{case}_{phase}.json",
            {
                "status": status,
                "completed_utc": datetime.now(UTC).isoformat(),
                "seconds": time.perf_counter() - started,
                "stages": stages,
                "sources": 1,
                "runner_sha256": digest(Path(__file__)),
                "peak_cuda_allocated": torch.cuda.max_memory_allocated(device),
            },
        )
        boundary.save(root / "receipts" / f"{case}_{phase}.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--case", choices=("case_04", "case_05"), required=True
    )
    parser.add_argument("--phase", choices=("source", "g34"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.root.resolve(), args.case, args.phase, torch.device(args.device))


if __name__ == "__main__":
    main()
