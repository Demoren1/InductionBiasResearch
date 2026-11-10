"""Enforce the observation boundary around the task-rank selection control."""

import argparse
import json
from pathlib import Path

from .blind_contract import digest
from .blind_runtime import ObservationBoundary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--condition", type=int, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    registration = root / "rank_registered.json"
    if registration.exists():
        record = json.loads(registration.read_text())
        for name, expected in record["code_sha256"].items():
            if digest(Path(__file__).resolve().parent / name) != expected:
                raise ValueError(f"Registered task-rank selector changed: {name}")
    boundary = ObservationBoundary(root, args.case, args.condition)
    boundary.role, boundary.phase = "select", "task_rank_selection"
    boundary.install()
    status = "failed"
    try:
        from .blind_rank_core import select

        select(root, args.case, args.condition)
        status = "complete"
    finally:
        boundary.save(
            root / "receipts" / f"task_rank_{args.case}_c{args.condition}.json", status
        )


if __name__ == "__main__":
    main()
