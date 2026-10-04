"""Source-only functional consensus controls for ordered pattern banks.

This module turns each bank's ``diagnostics['aligned_q_abs']`` maps into a
normalized cross-bank importance map. It builds generic exact-cardinality
controls from that importance and does not inspect labels or task partitions.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class FunctionalConsensusProposals:
    """Detached CPU outputs from :func:`build_functional_consensus_proposals`.

    ``importance`` is float64. The proposals are float32 binary masks.
    """

    importance: Tensor
    global_topk: Tensor
    balanced_per_column: Tensor


def _bank_maps(bank: Any, index: int) -> Tensor:
    if isinstance(bank, Mapping):
        diagnostics = bank.get("diagnostics")
    else:
        diagnostics = getattr(bank, "diagnostics", None)
    if not isinstance(diagnostics, Mapping):
        raise ValueError(f"bank {index} must expose a diagnostics mapping")
    if "aligned_q_abs" not in diagnostics:
        raise ValueError(f"bank {index} diagnostics must contain aligned_q_abs")
    try:
        maps = torch.as_tensor(diagnostics["aligned_q_abs"])
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"bank {index} aligned_q_abs must be numeric") from error
    if maps.dtype == torch.bool or maps.is_complex() or maps.is_quantized:
        raise ValueError(f"bank {index} aligned_q_abs must be numeric real values")
    if not (maps.is_floating_point() or maps.dtype in {
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}):
        raise ValueError(f"bank {index} aligned_q_abs must be numeric real values")
    if maps.ndim != 3 or maps.shape[0] < 1 or maps.shape[1] < 1 or maps.shape[2] < 1:
        raise ValueError(f"bank {index} aligned_q_abs must have shape [maps, features, hidden]")

    maps = maps.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(maps).all()):
        raise ValueError(f"bank {index} aligned_q_abs must be finite")
    if bool((maps < 0).any()):
        raise ValueError(f"bank {index} aligned_q_abs must be non-negative")
    return maps


def _global_topk(importance: Tensor, k: int) -> Tensor:
    features, hidden = importance.shape
    flat = importance.reshape(-1)
    # Python's secondary index makes exact ties stable in row-major order.
    order = sorted(range(flat.numel()), key=lambda index: (-float(flat[index]), index))
    proposal = torch.zeros((features, hidden), dtype=torch.float32)
    if k:
        selected = torch.tensor(order[:k], dtype=torch.long)
        proposal.reshape(-1)[selected] = 1.
    return proposal


def build_spatial_jittered_anchor(importance: Tensor, k: int, jitter: float,
                                 seed: int) -> Tensor:
    """Build a stable exact-K anchor from per-column ranks and tiled spatial jitter."""
    if (not isinstance(importance, Tensor) or importance.ndim != 2 or
            importance.shape[0] != 784 or importance.shape[1] < 1):
        raise ValueError("spatial anchor jitter requires importance with shape [784, hidden]")
    if not importance.is_floating_point() or not bool(torch.isfinite(importance).all()):
        raise ValueError("spatial anchor importance must be finite floating-point values")
    if type(k) is not int or not 0 < k < importance.numel():
        raise ValueError("spatial anchor k must be a positive exact-K budget")
    if (isinstance(jitter, bool) or not isinstance(jitter, (int, float)) or
            not math.isfinite(jitter) or jitter <= 0):
        raise ValueError("spatial anchor jitter must be finite and positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("spatial anchor seed must be an integer")

    scores = importance.detach().to(device="cpu", dtype=torch.float64).contiguous().numpy()
    ranks = np.empty_like(scores)
    feature_count, hidden = scores.shape
    positions = (feature_count - np.arange(feature_count, dtype=np.float64)) / feature_count
    for column in range(hidden):
        order = np.argsort(-scores[:, column], kind="stable")
        ranks[order, column] = positions

    spatial_rng_seed = seed
    if spatial_rng_seed < 0:
        spatial_rng_seed %= 1 << 128
    tile = np.random.default_rng(spatial_rng_seed).uniform(
        -jitter, jitter, size=(7, 7, hidden)
    )
    tile_noise = np.repeat(np.repeat(tile, 4, axis=0), 4, axis=1).reshape(
        feature_count, hidden
    )
    adjusted = ranks + tile_noise
    order = np.argsort(-adjusted.reshape(-1), kind="stable")[:k]
    mask = np.zeros(feature_count * hidden, dtype=np.float32)
    mask[order] = 1.
    return torch.from_numpy(mask.reshape(feature_count, hidden)).contiguous()


def _balanced_per_column(importance: Tensor, k: int) -> Tensor:
    features, hidden = importance.shape
    base, remainder = divmod(k, hidden)
    proposal = torch.zeros((features, hidden), dtype=torch.float32)
    next_choices: list[tuple[float, int, int]] = []

    for column in range(hidden):
        order = sorted(range(features), key=lambda row: (-float(importance[row, column]), row))
        for row in order[:base]:
            proposal[row, column] = 1.
        if base < features:
            row = order[base]
            next_choices.append((-float(importance[row, column]), column, row))

    # The remainder is smaller than the number of columns, so each column can
    # receive at most one extra edge. Ties go to the lower hidden-column index.
    next_choices.sort()
    for _, column, row in next_choices[:remainder]:
        proposal[row, column] = 1.
    return proposal


def build_functional_consensus_proposals(
    banks: Sequence[Any], k: int,
) -> FunctionalConsensusProposals:
    """Build equal-bank functional importance and two exact-K controls.

    Each ``aligned_q_abs`` tensor must have shape ``[R, F, H]``. Every map is
    normalized independently so each nonzero hidden column sums to one;
    zero columns stay zero. Maps are averaged within each bank first, then
    banks are averaged equally so a bank with more retained teachers receives
    no extra weight.

    Global top-K ties use row-major order. Balanced-per-column first selects
    ``floor(K / H)`` highest entries per column, with feature-index tie breaks,
    then assigns the remaining edges to columns with the largest next entry;
    ties go to the lower hidden-column index.
    """
    if type(k) is not int:
        raise ValueError("k must be an integer")
    if not isinstance(banks, Sequence) or isinstance(banks, (str, bytes)) or not banks:
        raise ValueError("banks must be a non-empty sequence")

    bank_means: list[Tensor] = []
    expected_shape: tuple[int, int] | None = None
    for index, bank in enumerate(banks):
        maps = _bank_maps(bank, index)
        shape = (maps.shape[1], maps.shape[2])
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError("all banks must have compatible feature and hidden dimensions")

        column_sums = maps.sum(dim=1, keepdim=True)
        denominators = torch.where(column_sums > 0, column_sums, torch.ones_like(column_sums))
        normalized = maps / denominators
        bank_means.append(normalized.mean(dim=0))

    assert expected_shape is not None
    features, hidden = expected_shape
    if not 0 <= k <= features * hidden:
        raise ValueError(f"k must be between 0 and {features * hidden}")
    importance = torch.stack(bank_means).mean(dim=0).detach().cpu().contiguous()
    return FunctionalConsensusProposals(
        importance=importance,
        global_topk=_global_topk(importance, k).detach().cpu().contiguous(),
        balanced_per_column=_balanced_per_column(importance, k).detach().cpu().contiguous(),
    )
