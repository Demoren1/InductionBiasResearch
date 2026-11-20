"""Paths and fixed settings for the six-task transfer comparison."""

import json
import os
from pathlib import Path

from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

PLAN = Path(__file__).with_name("protocol.json")
CONFIG = json.loads(PLAN.read_text())
SEED = CONFIG["seed"]
OUT = Path("artifacts/paper_transfer_2026-11-20")
WORK = OUT.parent / "local" / OUT.name
METHODS = ("dense", "wsr", "lutq", "yeh_map", "yeh_soft", "msr")


def register():
    """Record the shared validation grid before any test predictions."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "protocol.json"
    if path.exists() and digest(path) != digest(PLAN):
        raise ValueError("The registered experimental protocol has changed")
    if not path.exists():
        unique = path.with_name(f"protocol.{os.getpid()}.json")
        unique.write_bytes(PLAN.read_bytes())
        unique.replace(path)


def freeze(path, record):
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError(f"Frozen data or choices changed at {path}")
    if not path.exists():
        save_json(path, record)
