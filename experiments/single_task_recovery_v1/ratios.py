"""Direct pair recovery from one vector of scale-ambiguous weights."""

import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from experiments.sharing_structure_research_v1.blind_contract import (
    canonical_index,
)
from experiments.theorem_pair_recovery_v1.recover import score_magnitudes

THRESHOLDS = np.r_[0.0, np.geomspace(1e-12, 0.2, 48)]


def pair_partition(normal, mask, zero_tolerance, pair_tolerance):
    """Merge local zeros and matching nonzero ordered log differences."""
    rows, columns = np.where(mask)
    logs = np.log(np.maximum(np.abs(normal[rows, columns]), 1e-300))
    first, second = [], []
    for neuron in np.unique(columns):
        positions = np.flatnonzero(columns == neuron)
        left, right = np.meshgrid(positions, positions, indexing="ij")
        keep = left != right
        first.extend(left[keep])
        second.extend(right[keep])
    first, second = np.asarray(first, dtype=int), np.asarray(second, dtype=int)
    difference = logs[first] - logs[second]
    adjacency = np.eye(len(rows), dtype=bool)
    zero = np.abs(difference) <= zero_tolerance
    adjacency[first[zero], second[zero]] = True
    nonzero = np.flatnonzero(~zero)
    if len(nonzero):
        matches = cKDTree(difference[nonzero, None]).query_pairs(
            pair_tolerance, output_type="ndarray"
        )
        if len(matches):
            left, right = nonzero[matches[:, 0]], nonzero[matches[:, 1]]
            adjacency[first[left], first[right]] = True
            adjacency[second[left], second[right]] = True
    count, labels = connected_components(csr_matrix(adjacency), directed=False)
    index = np.zeros(mask.shape, dtype=np.int16)
    index[rows, columns] = labels + 1
    return index, count


def ratio_proposals(source, codes, limit=32):
    """Generate and rank complete partitions without observing a true mask."""
    started = time.perf_counter()
    normal, uncertainty = source["normal"], source["uncertainty"]
    masks = {}
    for multiplier in (1, 3, 5):
        for floor in (0.0, 0.002):
            mask = np.abs(normal) > np.maximum(
                multiplier * np.maximum(uncertainty[None], 0.0003), floor
            )
            masks.setdefault(
                mask.tobytes(),
                (
                    mask,
                    {
                        "kind": "direction_uncertainty",
                        "multiplier": multiplier,
                        "floor": floor,
                    },
                ),
            )
    for name in [key for key in source if key.startswith("index_")][:2]:
        mask = source[name] != 0
        masks.setdefault(mask.tobytes(), (mask, {"kind": name}))
    bank, native_seen, diagnostics = {}, set(), []
    for mask, mask_origin in masks.values():
        if np.count_nonzero(mask) < codes:
            continue
        counts = []
        for threshold in THRESHOLDS:
            index, count = pair_partition(
                normal, mask, threshold, 2 * threshold
            )
            counts.append(int(count))
            if count != codes:
                continue
            index *= np.where(normal >= 0, 1, -1).astype(np.int16)
            native_key = index.tobytes()
            if native_key in native_seen:
                continue
            native_seen.add(native_key)
            canonical_key = canonical_index(index, codes).tobytes()
            active_score, guesses = score_magnitudes(
                index, normal[None], uncertainty[None], codes
            )
            inactive_error = (
                normal[~mask]
                / np.broadcast_to(
                    np.maximum(uncertainty, 0.0003), normal.shape
                )[~mask]
            )
            score = float(
                (
                    np.count_nonzero(mask) * active_score
                    + np.sum(4 * np.log1p(inactive_error**2 / 4))
                )
                / normal.size
            )
            record = {
                "index": index,
                "score": score,
                "code_guesses": guesses,
                "origin": {"mask": mask_origin, "threshold": float(threshold)},
            }
            if (
                canonical_key not in bank
                or score < bank[canonical_key]["score"]
            ):
                bank[canonical_key] = record
        diagnostics.append(
            {
                "mask": mask_origin,
                "active": int(np.count_nonzero(mask)),
                "group_counts": counts,
            }
        )
    ordered = sorted(bank.values(), key=lambda item: item["score"])
    return ordered[:limit], {
        "seconds": time.perf_counter() - started,
        "thresholds": THRESHOLDS.tolist(),
        "equal_pair_threshold_multiplier": 2,
        "masks": diagnostics,
        "unique_complete_partitions": len(ordered),
        "retained": min(limit, len(ordered)),
        "incomplete_partitions_completed": False,
    }
