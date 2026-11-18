"""Shared source initialization for aSRR."""

from pathlib import Path

OUTPUT = Path("artifacts/asrr_anchor_init_2026-11-18")
ROOT = Path("artifacts/local/asrr_anchor_init_2026-11-18")
PARENT = Path("artifacts/local/paper_asrr_2026-11-17/mnist")
DENSE = PARENT / "source/dense/models.npz"
ANCHORS = tuple(range(10))
