"""Source-only signature recovery; numerical calibration only scores proposals."""

import itertools

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


THRESHOLDS = np.r_[0.0, np.geomspace(1e-10, 0.2, 32)]


def task_precision_weights(uncertainty):
    """Scale task columns by inverse RMS direction uncertainty."""
    weights = 1 / np.sqrt(np.mean(np.maximum(uncertainty, 0.0003) ** 2, axis=1))
    return weights / weights.max()


def canonical_partition(index, codes):
    """Canonicalize unsigned groups and interchangeable hidden columns."""
    best, result = None, None
    for permutation in itertools.permutations(range(1, codes + 1)):
        renamed = np.r_[0, permutation][index]
        renamed = renamed[:, np.lexsort(renamed[::-1])].astype(np.int16)
        key = renamed.tobytes()
        if best is None or key < best:
            best, result = key, renamed
    return result


def contrast_rank(index, codes):
    """Compute m minus the number of code-neuron graph components."""
    rows, columns = np.where(index)
    labels = np.abs(index[rows, columns]) - 1
    edges = np.zeros((codes + index.shape[1],) * 2, dtype=bool)
    edges[labels, codes + columns] = True
    _, components = connected_components(csr_matrix(edges), directed=False)
    return codes - len(np.unique(components[:codes]))


def signature_partitions(normal, mask, codes, task_weights=None):
    """Apply the theorem's two merge rules for fixed ranks and thresholds.

    The zero threshold is t and the equal-pair threshold is 2t, matching
    the ratio in the stability proposition. No incomplete partition is
    forcibly merged to the prescribed code count.
    """
    rows, columns = np.where(mask)
    if len(rows) < codes:
        return [], {"active": len(rows), "reason": "too_few_positions"}
    logs = np.log(np.maximum(np.abs(normal[:, rows, columns]), 1e-300)).T
    centered = logs.copy()
    for neuron in np.unique(columns):
        positions = columns == neuron
        centered[positions] -= centered[positions].mean(axis=0)
    if task_weights is not None:
        centered *= task_weights[None]
    basis, singular, _ = np.linalg.svd(centered, full_matrices=False)
    first, second = [], []
    for neuron in np.unique(columns):
        positions = np.flatnonzero(columns == neuron)
        left, right = np.meshgrid(positions, positions, indexing="ij")
        keep = left != right
        first.extend(left[keep])
        second.extend(right[keep])
    first, second = np.asarray(first, dtype=int), np.asarray(second, dtype=int)
    proposals = []
    seen = {}
    counts = []
    maximum_rank = min(codes - 1, basis.shape[1])
    for rank in range(maximum_rank + 1):
        signatures = basis[first, :rank] - basis[second, :rank]
        norms = np.linalg.norm(signatures, axis=1)
        for threshold in THRESHOLDS:
            adjacency = np.eye(len(rows), dtype=bool)
            zero = norms <= threshold
            adjacency[first[zero], second[zero]] = True
            nonzero = np.flatnonzero(~zero)
            if rank and len(nonzero):
                matches = cKDTree(signatures[nonzero]).query_pairs(
                    2 * threshold, output_type="ndarray"
                )
                if len(matches):
                    left, right = nonzero[matches[:, 0]], nonzero[matches[:, 1]]
                    adjacency[first[left], first[right]] = True
                    adjacency[second[left], second[right]] = True
            count, labels = connected_components(csr_matrix(adjacency), directed=False)
            counts.append(count)
            if count != codes:
                continue
            index = np.zeros(mask.shape, dtype=np.int16)
            index[rows, columns] = labels + 1
            if contrast_rank(index, codes) != rank:
                continue
            key = index.tobytes()
            if key in seen:
                proposals[seen[key]]["threshold_last"] = float(threshold)
                proposals[seen[key]]["threshold_values"].append(float(threshold))
                continue
            seen[key] = len(proposals)
            proposals.append(
                {
                    "index": index,
                    "rank": rank,
                    "threshold": float(threshold),
                    "threshold_last": float(threshold),
                    "threshold_values": [float(threshold)],
                }
            )
    return proposals, {
        "active": len(rows),
        "singular_values": singular.tolist(),
        "ranks": list(range(maximum_rank + 1)),
        "minimum_groups": min(counts, default=len(rows)),
        "maximum_groups": max(counts, default=len(rows)),
        "complete_partitions": len(proposals),
    }


def score_magnitudes(index, normal, uncertainty, codes):
    """Fit log codes/scales for a fixed partition, without changing labels."""
    rows, columns = np.where(index)
    labels = np.abs(index[rows, columns]) - 1
    magnitude = np.maximum(np.abs(normal[:, rows, columns]), 1e-300)
    logs = np.log(magnitude)
    deviation = np.clip(
        np.maximum(uncertainty[:, columns], 0.0003) / magnitude, 0.03, 3.0
    )
    matrix = np.c_[np.eye(codes)[labels], -np.eye(index.shape[1])[columns]]
    residuals, guesses = [], []
    for task in range(len(normal)):
        weight = 1 / deviation[task]
        solution = np.linalg.lstsq(
            matrix * weight[:, None], logs[task] * weight, rcond=1e-10
        )[0]
        for _ in range(3):
            residual = (matrix @ solution - logs[task]) / deviation[task]
            robust = weight / np.sqrt(1 + residual**2 / 4)
            solution = np.linalg.lstsq(
                matrix * robust[:, None], logs[task] * robust, rcond=1e-10
            )[0]
        residuals.append((matrix @ solution - logs[task]) / deviation[task])
        values = solution[:codes]
        guesses.append(np.exp(np.clip(values - values.mean(), -12, 12)))
    score = float(np.mean(4 * np.log1p(np.asarray(residuals) ** 2 / 4)))
    return score, np.asarray(guesses)


def signed_proposals(index, normal, uncertainty, codes):
    """Recover relative signs using observed task profiles and weighted votes."""
    rows, columns = np.where(index)
    labels = index[rows, columns]
    observed = np.where(normal[:, rows, columns] >= 0, 1, -1)
    weight = np.clip(
        np.abs(normal[:, rows, columns]) / np.maximum(uncertainty[:, columns], 0.0003),
        0.01,
        10.0,
    )
    seen = set()
    for reference in range(len(normal)):
        signs = observed[reference].copy()
        for refinement in range(2):
            if refinement:
                for _ in range(8):
                    previous = signs.copy()
                    for group in range(1, codes + 1):
                        active = labels == group
                        vote = (
                            weight[:, active] * observed[:, active] * signs[active]
                        ).sum(1)
                        profile = np.where(vote >= 0, 1, -1)
                        vote = (
                            weight[:, active] * observed[:, active] * profile[:, None]
                        ).sum(0)
                        signs[active] = np.where(vote >= 0, 1, -1)
                    if np.array_equal(previous, signs):
                        break
            key = signs.tobytes()
            if key in seen:
                continue
            seen.add(key)
            error = 0.0
            for group in range(1, codes + 1):
                active = labels == group
                vote = (weight[:, active] * observed[:, active] * signs[active]).sum(1)
                error += float((weight[:, active].sum(1) - np.abs(vote)).sum() / 2)
            result = np.zeros_like(index)
            result[rows, columns] = labels * signs
            yield (
                result,
                error / float(weight.sum()),
                {
                    "sign_reference": reference,
                    "sign_refined": bool(refinement),
                },
            )
