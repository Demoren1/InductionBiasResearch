"""Prepare a fresh confirmation set and freeze selections before generating it."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROTOCOL = Path("experiments/dynamic_signal_v1/protocol_confirmation.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "freeze"], required=True)
    args = parser.parse_args()
    plan = json.loads(PROTOCOL.read_text())
    root = Path(plan["run"])
    development = root.parent / "v1"
    root.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        for name in ["source", "calibration", "banks", "banks_large_tuned"]:
            link = root / name
            if not link.exists():
                link.symlink_to(
                    (development / name).resolve(), target_is_directory=True
                )
        selection = json.loads(
            (development / "bank_selection/selection.json").read_text()
        )
        bank = Path(selection["chosen_bank"])
        (root / "banks_relations").mkdir(exist_ok=True)
        for noise in range(3):
            link = root / "banks_relations" / f"noise_{noise}"
            if not link.exists():
                link.symlink_to(bank.resolve(), target_is_directory=True)
            prior = root / "priors/relations" / f"noise_{noise}"
            prior.mkdir(parents=True, exist_ok=True)
            for filename in ["made.pt", "frequencies.npz", "training.json"]:
                shutil.copyfile(
                    development / "priors/relations" / bank.name / filename,
                    prior / filename,
                )
        (root / "preparation.json").write_text(
            json.dumps(
                {
                    "source_selection": selection,
                    "confirmation_protocol": plan,
                    "status": "Target data not generated. Same learned source generator reused across noise levels.",
                },
                indent=2,
            )
            + "\n"
        )
        return
    expected = [
        root / directory / f"noise_{noise}" / f"calibrate_n{size}" / "complete.json"
        for directory in ["priors/relations", "competitors"]
        for noise in range(3)
        for size in plan["target_sizes"]
    ]
    expected += [
        root / "calibration_fit" / f"noise_{noise}_n{size}" / "complete.json"
        for noise in range(3)
        for size in plan["target_sizes"]
    ]
    expected += [
        root
        / "long_control"
        / f"noise_{noise}"
        / f"calibrate_n{size}"
        / "complete.json"
        for noise in range(3)
        for size in [2048, 8192]
    ]
    expected += [
        root
        / "weight_sharing"
        / f"noise_{noise}"
        / f"calibrate_n{size}"
        / "complete.json"
        for noise in range(3)
        for size in plan["target_sizes"]
    ]
    expected += [
        root
        / "weight_sharing"
        / f"noise_{noise}"
        / f"calibrate_n{size}"
        / "refinement/complete.json"
        for noise in range(3)
        for size in plan["target_sizes"]
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(f"Unfinished confirmation calibration: {missing}")
    from experiments.dynamic_signal_v1.report import freeze_primary

    freeze_primary(root, plan)
    files = (
        [
            PROTOCOL,
            root / "preparation.json",
            root / "primary_selection.json",
            development / "bank_selection/selection.json",
            development / "data_manifest.json",
            Path("pattern/autoregressive/model.py"),
        ]
        + list(root.rglob("selection.json"))
        + list(root.rglob("made.pt"))
        + list(Path("experiments/dynamic_signal_v1").glob("*.py"))
    )
    frozen = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    }
    (root / "frozen_selection.json").write_text(json.dumps(frozen, indent=2) + "\n")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.dynamic_signal_v1.benchmark",
            "--protocol",
            str(PROTOCOL),
            "--only-role",
            "target",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
