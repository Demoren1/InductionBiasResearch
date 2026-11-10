"""Resume completed scientific phases without recomputing frozen artifacts."""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from .blind_contract import digest, save_json


def run(root, case, condition, device, phase):
    folder = root / "fits" / f"{case}_c{condition}"
    stages = {
        "prepare": (
            ("seed", "fitted.npz", "seed_initialization.json"),
            ("bank", "joint_bank.npz", "joint_bank.json"),
        ),
        "finish": (
            ("fit", "joint_fitted.npz", "joint_fitted.json"),
            ("criteria", None, "joint_bic.json"),
            ("select", "joint_selected.npz", "joint_selected.json"),
        ),
    }[phase]
    for operation, data_name, metadata_name in stages:
        metadata = folder / metadata_name
        if metadata.exists():
            if data_name is not None:
                data = folder / data_name
                content = json.loads(metadata.read_text())
                if not data.exists() or digest(data) != content["sha256"]:
                    raise ValueError(f"Frozen phase data changed: {data}")
            continue
        if data_name is not None and (folder / data_name).exists():
            raise ValueError(f"Partial phase needs explicit recovery: {operation}")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.sharing_structure_research_v1.blind_joint_job",
                "--root",
                str(root),
                "--case",
                case,
                "--condition",
                str(condition),
                "--device",
                device,
                "--phase",
                operation,
            ],
            check=True,
        )
    save_json(
        folder / f"joint_resume_{phase}.json",
        {
            "completed_utc": datetime.now(UTC).isoformat(),
            "phase": phase,
            "case": case,
            "condition": condition,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--condition", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--phase", choices=("prepare", "finish"), required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.case, args.condition, args.device, args.phase)


if __name__ == "__main__":
    main()
