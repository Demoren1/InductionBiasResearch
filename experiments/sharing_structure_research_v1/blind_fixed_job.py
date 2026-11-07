"""Run one fixed-example experiment under an enforced teacher-data boundary."""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from .blind_contract import digest
from .blind_runtime import ObservationBoundary


def run(root, case, condition, device):
    registration = json.loads((root / "method_frozen.json").read_text())
    for name, expected in registration["code_sha256"].items():
        if digest(Path(__file__).resolve().parent / name) != expected:
            raise ValueError(f"Frozen learner source changed: {name}")
    boundary = ObservationBoundary(root, case, condition)
    boundary.install()
    receipt = root / "receipts" / f"{case}_c{condition}.json"
    status = "failed"
    started = time.monotonic()
    try:
        import torch

        from .blind_dense import run as train_dense
        from .blind_dense_extract import run as extract
        from .blind_dense_repair import run as repair
        from .blind_pipeline import fit_bank, select

        torch_device = torch.device(device)
        stages = (
            (
                "dense_adam",
                lambda: train_dense(root, case, condition, torch_device, 2500),
            ),
            (
                "dense_polish",
                lambda: extract(
                    root, case, condition, torch_device, 32, polish_only=True
                ),
            ),
            ("residual_repair", lambda: repair(root, case, condition, torch_device, 2)),
            (
                "candidate_bank",
                lambda: extract(
                    root, case, condition, torch_device, 0, from_repair=True
                ),
            ),
            (
                "indexed_fits",
                lambda: fit_bank(root, case, condition, 4, torch_device, 32),
            ),
        )
        for phase, operation in stages:
            boundary.phase = phase
            print(
                json.dumps(
                    {
                        "case": case,
                        "condition": condition,
                        "phase": phase,
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
            operation()
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()
        boundary.role, boundary.phase = "select", "meta_selection"
        select(root, case, condition)
        complete = root / "fits" / f"{case}_c{condition}" / "job_complete.json"
        complete.write_text(
            json.dumps(
                {
                    "completed_utc": datetime.now(UTC).isoformat(),
                    "case": case,
                    "condition": condition,
                    "seconds": time.monotonic() - started,
                    "teacher_access": False,
                }
            )
            + "\n"
        )
        status = "complete"
    finally:
        boundary.save(receipt, status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--condition", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(args.root.resolve(), args.case, args.condition, args.device)


if __name__ == "__main__":
    main()
