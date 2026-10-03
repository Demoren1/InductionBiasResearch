"""Compatibility façade for full-batch vectorised pattern measurements.

The solver lives in :mod:`generator_evaluator.pattern_fit`; this module keeps
the established public function and its deliberate full-batch contract.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor

from .data import InnerProtocol, TaskData
from .pattern_fit import PatternFitEngine, validate_pattern_fits


def fit_pattern_batch(masks: Tensor, tasks: Sequence[TaskData], protocol: InnerProtocol,
                      device: str = "cpu", *,
                      initialization_seeds: Sequence[int] | None = None) -> list[dict[str, Any]]:
    """Fit homogeneous full-batch jobs, retaining the historical result schema."""
    clean = torch.as_tensor(masks, dtype=torch.float32).detach().cpu().contiguous()
    validate_pattern_fits(clean, tasks, protocol)
    if protocol.batch_size is not None and protocol.batch_size < tasks[0].x_support.shape[0]:
        raise ValueError("batched pattern fitting supports full-batch protocol only")
    results = PatternFitEngine(protocol, device).fit(
        clean, tasks, initialization_seeds=initialization_seeds)
    seeds = ([protocol.seed] * len(tasks) if initialization_seeds is None
             else list(map(int, initialization_seeds)))
    for result, task, seed in zip(results, tasks, seeds):
        result.update(label_source="fresh_terminal_query", fixed_horizon=True,
                      protocol_id=protocol.fingerprint, task_id=task.task_id,
                      actual_initialization_seed=seed,
                      solver_protocol_seed=protocol.seed, minibatch_seed_base=None)
    return results


__all__ = ["fit_pattern_batch"]
