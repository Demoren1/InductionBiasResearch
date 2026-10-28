"""Run continuous-codebook extensions and a paired noise-free control."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=["tanh", "noisy_relu", "clean_relu", "all"], default="all"
    )
    args = parser.parse_args()
    for version, activation, noise in [
        ("tanh_continuous_v1", "tanh", "0"),
        ("noisy_relu_continuous_v1", "relu", "0.1"),
        ("clean_relu_continuous_v1", "relu", "0"),
    ]:
        if args.case != "all" and not version.startswith(args.case + "_"):
            continue
        commands = [
            [
                sys.executable,
                str(ROOT / "source.py"),
                "--version",
                version,
                "--activation",
                activation,
                "--noise",
                noise,
                "--continuous-source",
                "--steps",
                "8192",
            ],
            [
                sys.executable,
                str(ROOT / "run.py"),
                "--version",
                version,
                "--stage",
                "prior",
            ],
            [
                sys.executable,
                str(ROOT / "run.py"),
                "--version",
                version,
                "--stage",
                "calibrate",
                "--steps",
                "2048",
                "--starts",
                "4",
                "--tasks",
                "8",
                "--oracle-control",
            ],
            [
                sys.executable,
                str(ROOT / "run.py"),
                "--version",
                version,
                "--stage",
                "evaluate",
            ],
        ]
        for command in commands:
            print("RUN", " ".join(command), flush=True)
            subprocess.run(command, check=True, cwd=ROOT.parent.parent)


if __name__ == "__main__":
    main()
