"""Shared task-level objective reductions for generator and pool scoring."""
from __future__ import annotations

import torch
from torch import Tensor


QUALITY_OBJECTIVES = ("average", "mean", "worst", "mean_positive_worst")


def validate_quality_objective(objective: str) -> str:
    """Return a supported objective name or raise a clear validation error."""
    if not isinstance(objective, str) or objective not in QUALITY_OBJECTIVES:
        choices = ", ".join(repr(choice) for choice in QUALITY_OBJECTIVES)
        raise ValueError(f"quality_objective must be one of {choices}")
    return objective


def quality_objective_cost(
    delta: Tensor,
    objective: str = "average",
    *,
    task_dim: int = -1,
) -> Tensor:
    """Reduce finite real task deltas to per-item costs.

    ``average`` (also ``mean``) returns the arithmetic mean task delta.
    ``worst`` returns the maximum task delta. ``mean_positive_worst`` returns
    the mean task delta plus any positive worst-task regression. The task axis
    is removed, while every other leading dimension is retained. The operation
    stays differentiable for floating tensors.
    """
    validate_quality_objective(objective)
    if not isinstance(delta, Tensor):
        raise TypeError("delta must be a torch.Tensor")
    if delta.ndim < 1 or not delta.is_floating_point():
        raise ValueError("delta must be a nonempty-rank real floating tensor")
    if isinstance(task_dim, bool) or not isinstance(task_dim, int):
        raise TypeError("task_dim must be an integer axis")
    if task_dim < -delta.ndim or task_dim >= delta.ndim:
        raise ValueError("task_dim is outside delta dimensions")
    task_dim %= delta.ndim
    if delta.shape[task_dim] < 1:
        raise ValueError("delta must contain at least one task")
    if not bool(torch.isfinite(delta).all()):
        raise ValueError("delta must contain only finite real values")

    if objective in ("average", "mean"):
        cost = delta.mean(dim=task_dim)
    elif objective == "worst":
        cost = delta.max(dim=task_dim).values
    else:
        cost = delta.mean(dim=task_dim) + torch.relu(delta.max(dim=task_dim).values)
    if not bool(torch.isfinite(cost).all()):
        raise ValueError("quality objective cost must be finite")
    return cost
