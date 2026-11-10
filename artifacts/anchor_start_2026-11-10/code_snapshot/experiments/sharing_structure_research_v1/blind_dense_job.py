"""Prepare learned dense observations without opening meta or teacher data."""

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path

from .blind_contract import save_json
from .blind_runtime import ObservationBoundary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--condition", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    root = args.root.resolve()
    boundary = ObservationBoundary(root, args.case, args.condition)
    boundary.install()
    status, started = "failed", time.monotonic()
    try:
        import torch

        from .blind_dense import run as train
        from .blind_dense_extract import run as extract
        from .blind_dense_repair import run as repair

        device = torch.device(args.device)
        stages = (
            (
                "dense_adam",
                lambda: train(root, args.case, args.condition, device, 2500),
            ),
            (
                "dense_polish",
                lambda: extract(
                    root, args.case, args.condition, device, 32, polish_only=True
                ),
            ),
            (
                "dense_repair",
                lambda: repair(root, args.case, args.condition, device, 2),
            ),
            (
                "dense_observations",
                lambda: extract(
                    root, args.case, args.condition, device, 0, from_repair=True
                ),
            ),
        )
        for phase, operation in stages:
            boundary.phase = phase
            operation()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        save_json(
            root / "dense" / f"{args.case}_c{args.condition}" / "prepared.json",
            {
                "completed_utc": datetime.now(UTC).isoformat(),
                "seconds": time.monotonic() - started,
                "teacher_access": False,
                "meta_access": False,
            },
        )
        status = "complete"
    finally:
        boundary.save(
            root / "receipts" / f"dense_{args.case}_c{args.condition}.json", status
        )


if __name__ == "__main__":
    main()
