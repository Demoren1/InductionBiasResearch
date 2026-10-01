"""Post-hoc support and signed-weight audits for length-11 task masks.

The gold pattern is used only by these reporting functions.  In particular,
``align_mask_to_gold`` is not a mask-construction or model-selection helper.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def gold_toeplitz_mask(
    seq_len: int = 11, hidden_width: int = 8, kernel_size: int = 4
) -> np.ndarray:
    """Return the canonical ``i - h in [0, kernel_size)`` support mask."""
    if seq_len < 1 or hidden_width < 1 or kernel_size < 1:
        raise ValueError("dimensions and kernel_size must be positive")
    rows = np.arange(seq_len)[:, None]
    columns = np.arange(hidden_width)[None, :]
    return ((rows - columns) >= 0) & ((rows - columns) < kernel_size)


def exact_topk_mask(scores: np.ndarray, k: int) -> np.ndarray:
    """Select exactly ``k`` entries, breaking ties by row-major index."""
    values = np.asarray(scores)
    if values.ndim != 2:
        raise ValueError("scores must be a two-dimensional matrix")
    if not 0 <= k <= values.size:
        raise ValueError(f"k must be between zero and {values.size}")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    order = np.argsort(-values.ravel(), kind="stable")
    result = np.zeros(values.size, dtype=bool)
    result[order[:k]] = True
    return result.reshape(values.shape)


def _as_support(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("mask must be a non-empty two-dimensional array")
    if values.dtype != np.bool_:
        if not np.isfinite(values).all():
            raise ValueError("mask values must be finite")
        values = values != 0
    return values.astype(bool, copy=False)


def align_mask_to_gold(mask: np.ndarray, gold: np.ndarray | None = None) -> dict[str, Any]:
    """Align predicted hidden columns to canonical gold columns by support.

    The Hungarian assignment is computed only from binary support overlap.
    The returned permutation arrays use explicit directions:

    * ``candidate_to_gold[c]`` is the gold column assigned to candidate c.
    * ``gold_to_candidate[g]`` is the candidate assigned to gold column g.

    Callers should reuse ``gold_to_candidate`` to align weights.  Weight values
    never influence this assignment.
    """
    found = _as_support(mask)
    target = gold_toeplitz_mask(*found.shape) if gold is None else _as_support(gold)
    if found.shape != target.shape:
        raise ValueError(f"mask and gold shapes differ: {found.shape} != {target.shape}")

    overlap = found.astype(np.int64).T @ target.astype(np.int64)
    candidate_columns, gold_columns = linear_sum_assignment(-overlap)
    candidate_to_gold = np.full(found.shape[1], -1, dtype=np.int64)
    gold_to_candidate = np.full(target.shape[1], -1, dtype=np.int64)
    candidate_to_gold[candidate_columns] = gold_columns
    gold_to_candidate[gold_columns] = candidate_columns
    if np.any(candidate_to_gold < 0) or np.any(gold_to_candidate < 0):
        raise ValueError("Hungarian assignment requires equal hidden widths")

    aligned = found[:, gold_to_candidate]
    intersection = int(np.logical_and(aligned, target).sum())
    union = int(np.logical_or(aligned, target).sum())
    predicted = int(found.sum())
    gold_count = int(target.sum())

    gold_column_counts = np.logical_and(aligned, target).sum(axis=0).astype(int)
    exact_gold_matches = np.all(aligned == target, axis=0)
    # Per-candidate coverage asks how much of its best-matching canonical
    # four-edge window is represented; it is independent of assignment ties.
    candidate_overlap = overlap.max(axis=1)
    gold_column_size = target.sum(axis=0)
    if not np.all(gold_column_size == gold_column_size[0]):
        raise ValueError("window coverage currently expects equal-size gold supports")
    window_size = int(gold_column_size[0])
    exact_candidate_window = np.zeros(found.shape[1], dtype=bool)
    for candidate in range(found.shape[1]):
        exact_candidate_window[candidate] = any(
            np.array_equal(found[:, candidate], target[:, gold_column])
            for gold_column in range(target.shape[1])
        )
    # Every gold window must be present in exactly one candidate column.
    exact_match_multiplicity = np.array(
        [int(np.sum(np.all(found == target[:, gold_column, None], axis=0)))
         for gold_column in range(target.shape[1])],
        dtype=np.int64,
    )

    spans: list[int] = []
    edge_ranges: list[list[int] | None] = []
    for column in range(found.shape[1]):
        active = np.flatnonzero(found[:, column])
        if active.size:
            spans.append(int(active[-1] - active[0] + 1))
            edge_ranges.append([int(active[0]), int(active[-1])])
        else:
            spans.append(0)
            edge_ranges.append(None)

    offsets = np.arange(found.shape[0])[:, None] - np.arange(found.shape[1])[None, :]
    offset_values, offset_counts = np.unique(offsets[aligned], return_counts=True)
    offset_histogram = {str(int(value)): int(count)
                        for value, count in zip(offset_values, offset_counts)}

    return {
        "shape": list(found.shape),
        "active_edges": predicted,
        "gold_active_edges": gold_count,
        "intersection": intersection,
        "union": union,
        "iou": intersection / union if union else 1.0,
        "precision": intersection / predicted if predicted else 0.0,
        "recall": intersection / gold_count if gold_count else 0.0,
        "hamming_matches": int(np.equal(aligned, target).sum()),
        "candidate_to_gold": candidate_to_gold.tolist(),
        "gold_to_candidate": gold_to_candidate.tolist(),
        "aligned_mask": aligned.astype(np.uint8),
        "gold_mask": target.astype(np.uint8),
        "gold_column_overlap": gold_column_counts.tolist(),
        "gold_column_exact_match": exact_gold_matches.tolist(),
        "window_coverage_by_candidate": (candidate_overlap / window_size).tolist(),
        "candidate_best_window_overlap": candidate_overlap.astype(int).tolist(),
        "candidate_exact_local_window": exact_candidate_window.tolist(),
        "exact_recover_all_windows_once": bool(np.all(exact_match_multiplicity == 1)),
        "exact_window_multiplicity_by_gold": exact_match_multiplicity.tolist(),
        "column_edge_span": spans,
        "column_edge_range": edge_ranges,
        "mean_column_edge_span": float(np.mean(spans)),
        "max_column_edge_span": int(max(spans, default=0)),
        "relative_offset_histogram": offset_histogram,
    }


def _diagonal_projection(matrix: np.ndarray, offsets: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """Project onto matrices constant on each selected diagonal."""
    result = np.zeros_like(matrix, dtype=np.float64)
    for offset in np.unique(offsets[selected]):
        positions = selected & (offsets == offset)
        result[positions] = float(np.mean(matrix[positions]))
    return result


def _projection_energy(matrix: np.ndarray, projected: np.ndarray) -> float:
    denominator = float(np.square(matrix).sum())
    return float(np.square(projected).sum() / denominator) if denominator > 0 else 0.0


def signed_weight_audit(
    weight: np.ndarray,
    mask: np.ndarray,
    *,
    readout: np.ndarray | None = None,
    bias: np.ndarray | None = None,
    kernel_size: int = 4,
) -> dict[str, Any]:
    """Measure diagonal structure in the actual signed ``W * mask`` matrix.

    ``weight`` is the child's raw first-layer weight with shape ``[L, H]``.
    The active matrix is formed as ``weight * mask`` and then aligned using the
    mask-only Hungarian permutation.  If a readout vector is supplied, a
    separate ``W * mask * readout`` analysis is returned because the child
    activation can include positive hidden-unit rescaling gauge.  Negative
    readout signs are preserved and are never absorbed into a column norm.
    """
    raw = np.asarray(weight, dtype=np.float64)
    support = _as_support(mask)
    if raw.shape != support.shape:
        raise ValueError(f"weight and mask shapes differ: {raw.shape} != {support.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("weight must be finite")
    audit = align_mask_to_gold(support)
    order = np.asarray(audit["gold_to_candidate"], dtype=np.int64)
    active = raw * support
    aligned_raw = raw[:, order]
    aligned_active = active[:, order]
    offsets = np.arange(raw.shape[0])[:, None] - np.arange(raw.shape[1])[None, :]
    all_diagonals = np.ones_like(offsets, dtype=bool)
    oracle_band = (offsets >= 0) & (offsets < kernel_size)

    def metrics(matrix: np.ndarray) -> dict[str, Any]:
        all_projection = _diagonal_projection(matrix, offsets, all_diagonals)
        band_projection = _diagonal_projection(matrix, offsets, oracle_band)
        return {
            "all_offset_toeplitz_explained_energy": _projection_energy(matrix, all_projection),
            "oracle_band_0_to_k_minus_1_explained_energy": _projection_energy(matrix, band_projection),
            "all_offset_diagonal_means": {
                str(int(offset)): float(np.mean(matrix[offsets == offset]))
                for offset in np.unique(offsets)
            },
        }

    result: dict[str, Any] = {
        "alignment_source": "support_only_hungarian",
        "weight_shape": list(raw.shape),
        "raw_weight_l2": float(np.linalg.norm(aligned_raw)),
        "masked_effective_weight_l2": float(np.linalg.norm(aligned_active)),
        "raw_weight": aligned_raw,
        "masked_effective_weight": aligned_active,
        "raw_weight_structure": metrics(aligned_raw),
        "masked_effective_weight_structure": metrics(aligned_active),
        "masked_effective_column_l2": np.linalg.norm(aligned_active, axis=0),
        "masked_effective_column_normalized_structure": metrics(
            aligned_active / np.maximum(np.linalg.norm(aligned_active, axis=0, keepdims=True), 1e-12)
        ),
    }
    if readout is not None:
        coefficients = np.asarray(readout, dtype=np.float64)
        if coefficients.shape != (raw.shape[1],):
            raise ValueError(f"readout must have shape {(raw.shape[1],)}, got {coefficients.shape}")
        if not np.isfinite(coefficients).all():
            raise ValueError("readout must be finite")
        aligned_readout = coefficients[order]
        readout_scaled = aligned_active * aligned_readout[None, :]
        normalized_readout_scaled = readout_scaled / np.maximum(
            np.linalg.norm(readout_scaled, axis=0, keepdims=True), 1e-12
        )
        result.update({
            "aligned_readout": aligned_readout,
            "readout_scaled_effective_weight": readout_scaled,
            "readout_scaled_structure": metrics(readout_scaled),
            "readout_column_normalized_structure": metrics(normalized_readout_scaled),
            "readout_signs": np.sign(aligned_readout).astype(int),
        })
    if bias is not None:
        bias_values = np.asarray(bias, dtype=np.float64)
        if bias_values.shape != (raw.shape[1],) or not np.isfinite(bias_values).all():
            raise ValueError(f"bias must be finite with shape {(raw.shape[1],)}")
        aligned_bias = bias_values[order]
        affine_norms = np.sqrt(np.square(aligned_active).sum(axis=0) + np.square(aligned_bias))
        affine_normalized_weight = aligned_active / np.maximum(affine_norms[None, :], 1e-12)
        result["aligned_bias"] = aligned_bias
        result["affine_column_l2"] = affine_norms
        result["affine_column_normalized_weight"] = affine_normalized_weight
        result["affine_column_normalized_bias"] = aligned_bias / np.maximum(affine_norms, 1e-12)
        result["affine_column_normalized_structure"] = metrics(affine_normalized_weight)
    return result


def summarize_structure(mask: np.ndarray, weight: np.ndarray | None = None,
                        *, readout: np.ndarray | None = None,
                        bias: np.ndarray | None = None) -> dict[str, Any]:
    """Return JSON-safe support metrics and optional signed-weight metrics."""
    audit = align_mask_to_gold(mask)
    output = {key: value for key, value in audit.items()
              if not isinstance(value, np.ndarray)}
    output["aligned_mask"] = audit["aligned_mask"].tolist()
    output["gold_mask"] = audit["gold_mask"].tolist()
    if weight is not None:
        weights = signed_weight_audit(weight, mask, readout=readout, bias=bias)
        for key, value in weights.items():
            if isinstance(value, np.ndarray):
                output[key] = value.tolist()
            elif isinstance(value, dict):
                output[key] = {
                    subkey: (subvalue.tolist() if isinstance(subvalue, np.ndarray)
                             else subvalue)
                    for subkey, subvalue in value.items()
                }
            else:
                output[key] = value
    return output


def validate_toy_cases() -> dict[str, float | bool]:
    """Fast CPU self-checks for alignment, exact-K and signed projections."""
    gold = gold_toeplitz_mask()
    perm = np.array([5, 2, 7, 0, 6, 1, 4, 3])
    reversed_columns = gold[:, perm]
    aligned = align_mask_to_gold(reversed_columns)
    if aligned["iou"] != 1.0 or not aligned["exact_recover_all_windows_once"]:
        raise AssertionError("Hungarian alignment failed on a permuted gold mask")
    random_scores = np.random.default_rng(8100).normal(size=gold.shape)
    random_mask = exact_topk_mask(random_scores, 32)
    if random_mask.sum() != 32:
        raise AssertionError("exact_topk_mask did not return exactly K entries")
    random_audit = align_mask_to_gold(random_mask)
    if not random_audit["iou"] < 1.0:
        raise AssertionError("random exact-K mask unexpectedly recovered the full gold support")
    toeplitz_weight = gold.astype(np.float64) * np.array([1., 2., 3., 4., 5., 6., 7., 8.])[None, :]
    weight_audit = signed_weight_audit(toeplitz_weight[:, perm], reversed_columns)
    if weight_audit["masked_effective_weight"].shape != (11, 8):
        raise AssertionError("weight alignment returned the wrong shape")
    if not np.isfinite(weight_audit["masked_effective_weight_structure"]
                       ["oracle_band_0_to_k_minus_1_explained_energy"]):
        raise AssertionError("weight projection returned a non-finite value")
    return {
        "gold_iou_after_permutation": float(aligned["iou"]),
        "random_exact_k": int(random_mask.sum()) == 32,
        "random_iou": float(random_audit["iou"]),
        "weight_shape_ok": True,
    }
