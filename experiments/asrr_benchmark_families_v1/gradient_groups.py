"""Fixed-budget partitions scored by paired support and query gradients."""

import numpy as np


def gradient_kernel(support, query):
    """Arrays have shape (inputs, independent gradient coordinates)."""
    cross = support @ query.T / support.shape[-1]
    return (cross + cross.T) / 2


def membership(labels):
    return np.eye(int(labels.max()) + 1, dtype=np.float64)[labels]


def orthogonal_groups(labels):
    matrix = membership(labels)
    return matrix / np.sqrt(matrix.sum(axis=0))


def group_score(kernel, labels):
    basis = orthogonal_groups(labels)
    return float(((kernel @ basis) * basis).sum())


def gradient_tree(kernel, rank):
    """At every merge, maximize the change in the sum of group scores."""
    n = len(kernel)
    sums = kernel.copy()
    sizes = np.ones(n)
    scores = np.diag(sums).copy()
    active = np.ones(n, dtype=bool)
    labels = np.arange(n)
    gains = (scores[:, None] + scores[None, :] + 2 * sums) / 2
    gains -= scores[:, None] + scores[None, :]
    np.fill_diagonal(gains, -np.inf)
    for _ in range(n - rank):
        first, second = divmod(np.argmax(gains), n)
        sums[first] += sums[second]
        sums[:, first] += sums[:, second]
        sizes[first] += sizes[second]
        scores[first] = sums[first, first] / sizes[first]
        active[second] = False
        labels[labels == second] = first
        candidate = (
            (sums[first, first] + np.diag(sums) + 2 * sums[first])
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
    return np.unique(labels, return_inverse=True)[1]


def gradient_refine(kernel, labels, passes=5):
    """Move individual inputs only when the exact score increases."""
    labels = labels.copy()
    matrix = membership(labels)
    sizes = matrix.sum(axis=0)
    cross = kernel @ matrix
    totals = (cross * matrix).sum(axis=0)
    diagonal = np.diag(kernel)
    for _ in range(passes):
        moved = False
        for i, value in enumerate(diagonal):
            source = labels[i]
            if sizes[source] == 1:
                continue
            remove = totals[source] - 2 * cross[i, source] + value
            gain = remove / (sizes[source] - 1) - totals[source] / sizes[source]
            additions = totals + 2 * cross[i] + value
            changes = gain + additions / (sizes + 1) - totals / sizes
            changes[source] = 0
            target = int(changes.argmax())
            if changes[target] <= 1e-12:
                continue
            totals[source] = remove
            totals[target] = additions[target]
            sizes[source] -= 1
            sizes[target] += 1
            cross[:, source] -= kernel[:, i]
            cross[:, target] += kernel[:, i]
            labels[i] = target
            moved = True
        if not moved:
            break
    return labels
