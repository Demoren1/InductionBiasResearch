"""Locations and fixed settings for the frequency-competition experiment."""

import hashlib
import os
import shutil
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
RUN = (
    Path(os.environ.get("SINE_MASK_RUN", DIRECTORY / "runs/v1"))
    .expanduser()
    .resolve()
)
CONTROL = RUN / "control"
DEVICE = os.environ.get("SINE_MASK_DEVICE", "cuda:1")
OPTIONS = {"steps": 1024, "lr": 0.05, "scale": 2.0, "batch": 64}
SEEDS = [4213, 4214, 4215]


def families(split):
    """Return the structural split fixed before generator training."""
    remainder = {"train": (0, 1), "validation": (2,), "test": (3,)}[split]
    return [
        (a, b) for a in range(4) for b in range(4) if (b - a) % 4 in remainder
    ]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def implementation_hashes():
    return {path.name: digest(path) for path in sorted(DIRECTORY.glob("*.py"))}


def snapshot_sources():
    """Keep the first source snapshot and reject mixed-code resumptions."""
    destination = RUN / "source"
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(DIRECTORY.glob("*.py")):
        target = destination / path.name
        if target.exists() and digest(target) != digest(path):
            raise RuntimeError(
                f"Changed implementation: {path.name}; use a new run directory"
            )
        if not target.exists():
            shutil.copy2(path, target)
