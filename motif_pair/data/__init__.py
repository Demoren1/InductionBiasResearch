"""Data generation for circular motif-pair tasks."""

from .generate import (
    circular_windows,
    gold_first_layer,
    gold_mask,
    is_task_feasible,
    make_dataset,
    make_task_bank,
    sample_balanced,
    valid_pairs_for_gap,
)

__all__ = [
    "circular_windows",
    "gold_first_layer",
    "gold_mask",
    "is_task_feasible",
    "make_dataset",
    "make_task_bank",
    "sample_balanced",
    "valid_pairs_for_gap",
]
