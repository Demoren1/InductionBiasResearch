"""A signed group score and its interpolation with source directions."""

import numpy as np


def unit_frobenius(matrix):
    return matrix / max(np.linalg.norm(matrix), 1e-12)


def signed_groups(kernel, rank):
    """Greedily maximize tr(G.T @ kernel @ G) at each group count."""
    size = len(kernel)
    sums = kernel.copy()
    sizes = np.ones(size)
    scores = np.diag(sums).copy()
    active = np.ones(size, dtype=bool)
    labels = np.arange(size)
    signs = np.ones(size)
    gains = (scores[:, None] + scores + 2 * np.abs(sums)) / 2
    gains -= scores[:, None] + scores
    np.fill_diagonal(gains, -np.inf)

    for _ in range(size - rank):
        first, second = divmod(np.argmax(gains), size)
        phase = -1 if sums[first, second] < 0 else 1
        combined = sums[first] + phase * sums[second]
        combined[first] += sums[second, second] + phase * sums[first, second]
        sums[first] = combined
        sums[:, first] = combined
        sizes[first] += sizes[second]
        scores[first] = sums[first, first] / sizes[first]
        active[second] = False
        joined = labels == second
        signs[joined] *= phase
        labels[joined] = first

        candidate = (
            (sums[first, first] + np.diag(sums) + 2 * np.abs(combined))
            / (sizes[first] + sizes)
            - scores[first]
            - scores
        )
        candidate[~active] = -np.inf
        candidate[first] = -np.inf
        gains[first] = candidate
        gains[:, first] = candidate
        gains[second] = -np.inf
        gains[:, second] = -np.inf

    _, labels = np.unique(labels, return_inverse=True)
    groups = np.zeros((size, rank))
    groups[np.arange(size), labels] = signs / np.sqrt(
        np.bincount(labels)[labels]
    )
    return groups


def spectral_basis(groups, guide, group_weight):
    """Compute the leading eigenvectors without forming a square matrix."""
    if group_weight == 1:
        return groups
    if group_weight == 0:
        return guide
    rank = groups.shape[1]
    ordering = np.sqrt(np.linspace(1, 0.9, rank))
    factors = np.column_stack(
        (
            np.sqrt(1 - group_weight) * guide * ordering,
            np.sqrt(group_weight) * groups * ordering,
        )
    )
    left, _, _ = np.linalg.svd(factors, full_matrices=False)
    return left[:, :rank]
