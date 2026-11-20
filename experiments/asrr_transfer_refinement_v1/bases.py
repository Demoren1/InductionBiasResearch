"""Learn input partitions and local subspaces under the same rank budget."""

import numpy as np
from scipy.cluster.hierarchy import cut_tree, linkage

from experiments.asrr_benchmark_families_v1.gradient_groups import (
    membership,
    orthogonal_groups,
)


def ward_profiles(profiles, rank):
    return cut_tree(linkage(profiles, method="ward"), rank).ravel()


def local_basis(signal, labels, rank):
    """Maximize the trace of a symmetric signal over group-local subspaces."""
    basis = np.zeros((len(labels), rank))
    components = []
    groups = int(labels.max()) + 1
    for group in range(groups):
        rows = np.flatnonzero(labels == group)
        values, vectors = np.linalg.eigh(signal[np.ix_(rows, rows)])
        basis[rows, group] = vectors[:, -1]
        components.extend(
            (value, group, i, rows, vectors[:, i])
            for i, value in enumerate(values[:-1])
        )
    components.sort(key=lambda x: (-x[0], x[1], x[2]))
    for column, (_, _, _, rows, directions) in zip(
        range(groups, rank), components
    ):
        basis[rows, column] = directions
    for column in range(rank):
        if basis[np.argmax(abs(basis[:, column])), column] < 0:
            basis[:, column] *= -1
    return basis


def factor_model(bases, layout, *, initial_scales=None):
    count = len(bases)
    if 1 < count < layout.sites:
        repeats = layout.sites // count
        bases = np.repeat(bases, repeats, axis=0)
    model = {"factor": np.asarray(bases, dtype=np.float32)}
    if initial_scales is not None:
        if count < layout.sites:
            initial_scales = np.repeat(
                initial_scales, layout.sites // count, axis=0
            )
        width = layout.hidden // len(initial_scales)
        model["initial_scale"] = np.repeat(
            initial_scales, width, axis=1
        ).ravel()
    return model


def partition_model(groups, layout, *, normalized):
    if normalized:
        bases = [orthogonal_groups(row) for row in groups]
        scales = np.stack(
            [np.sqrt(2 * np.bincount(row) / layout.fan_in) for row in groups]
        ).astype(np.float32)
    else:
        bases = [membership(row) for row in groups]
        scales = np.full(
            (len(groups), bases[0].shape[1]),
            np.sqrt(2 / layout.fan_in),
            dtype=np.float32,
        )
    return factor_model(bases, layout, initial_scales=scales)
