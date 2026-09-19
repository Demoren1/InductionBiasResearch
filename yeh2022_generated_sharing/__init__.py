"""Generated assignment matrices for the Yeh et al. (2022) benchmarks."""

from .core import (
    CoordinateAssignmentGenerator,
    DirectAssignment,
    assignment_from_logits,
    partition_distance,
    solve_shared_mean,
)

__all__ = [
    "CoordinateAssignmentGenerator",
    "DirectAssignment",
    "assignment_from_logits",
    "partition_distance",
    "solve_shared_mean",
]
