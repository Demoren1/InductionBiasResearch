"""Measure the previously uninstrumented dense warm-start component once per shape."""

import argparse
import resource
import time
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

from experiments.sharing_structure_research_v1 import blind_dense
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from experiments.sharing_structure_research_v1.blind_runtime import ObservationBoundary


def run(root, case, device):
    boundary = ObservationBoundary(root, case, 0)
    boundary.install()
    key = f"{case}_c0"
    registration = root / "metrics" / f"{key}_registered.json"
    if registration.exists():
        raise RuntimeError("Cost replay already registered")
    save_json(
        registration,
        {
            "purpose": "One-time operation/memory measurement of unchanged source DenseAdam; outputs cannot enter model selection",
            "code_sha256": digest(Path(__file__)),
            "dense_code_sha256": digest(Path(blind_dense.__file__)),
            "steps": 2500,
            "starts": 12,
        },
    )
    torch.zeros(1, device=device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    counter = FlopCounterMode(display=False)
    started = time.perf_counter()
    status = "failed"
    try:
        with counter:
            blind_dense.run(root, case, 0, device, 2500)
        torch.cuda.synchronize(device)
        status = "complete"
    finally:
        save_json(
            root / "metrics" / f"{key}_replay.json",
            {
                "status": status,
                "wall_seconds": time.perf_counter() - started,
                "torch_operator_flops": counter.get_total_flops(),
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
                "measurement_only": True,
            },
        )
        boundary.save(root / "receipts" / f"{key}_access_receipt.json", status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.root.resolve(), args.case, torch.device(args.device))


if __name__ == "__main__":
    main()
