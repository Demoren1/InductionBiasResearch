"""Run joint proposal/refinement under the enforced observation boundary."""

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
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--phase",
        choices=("seed", "bank", "fit", "criteria", "select", "all"),
        default="all",
    )
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args()
    root = args.root.resolve()
    registration = root / "joint_registered.json"
    if registration.exists():
        hashes = json.loads(registration.read_text())["code_sha256"]
        for name, expected in hashes.items():
            if digest(Path(__file__).resolve().parent / name) != expected:
                raise ValueError(f"Registered joint learner changed: {name}")
    boundary = ObservationBoundary(root, args.case, args.condition)
    boundary.install()
    status = "failed"
    try:
        import torch

        from .blind_joint_bank import (
            fit,
            information_criterion,
            initialize_bank,
            propose,
            select,
        )

        if args.phase in ("all", "seed"):
            boundary.phase = "joint_source_seeds"
            initialize_bank(root, args.case, args.condition, torch.device(args.device))

        if args.phase in ("all", "bank"):
            boundary.phase = "joint_proposals"
            propose(root, args.case, args.condition)
        if args.phase in ("all", "fit"):
            boundary.phase = "joint_numeric"
            fit(root, args.case, args.condition, torch.device(args.device), args.limit)
        if args.phase in ("all", "criteria"):
            boundary.phase = "joint_information_criterion"
            information_criterion(root, args.case, args.condition)
        if args.phase in ("all", "select"):
            boundary.role, boundary.phase = "select", "joint_meta"
            select(root, args.case, args.condition)
        status = "complete"
    finally:
        boundary.save(
            root
            / "receipts"
            / f"joint_{args.case}_c{args.condition}_{args.phase}.json",
            status,
        )


if __name__ == "__main__":
    main()
