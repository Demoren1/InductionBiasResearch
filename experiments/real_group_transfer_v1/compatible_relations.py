"""Exact compatibility of a small bank of complete signed indexators."""

from math import ceil

import numpy as np

from experiments.srr_approx_v1.compatible import common_index
from experiments.srr_approx_v1.core import canonical


def relation_intersections(indices):
    """Enumerate distinct intersections of nonempty subsets of the finite bank."""
    complete = np.stack([canonical(index) for index in indices])
    known = {}
    for model in complete:
        extra = {model.tobytes(): model}
        for partial in known.values():
            shared = common_index(partial, model).astype(complete.dtype)
            extra[shared.tobytes()] = shared
        known.update(extra)

    maps = np.stack(list(known.values()))
    flat = complete.reshape(len(complete), -1)
    extends = np.empty((len(maps), len(complete)), dtype=bool)
    for i, row in enumerate(maps.reshape(len(maps), -1)):
        active = row != 0
        _, first, inv = np.unique(
            abs(row[active]), return_index=True, return_inverse=True
        )
        oriented = flat[:, active] * np.sign(row[active])
        extends[i] = (flat[:, ~active] == 0).all(axis=1) & (
            oriented == oriented[:, first[inv]]
        ).all(axis=1)
    return maps, extends


def maximal_supported(good, extends, coverage):
    """A smaller completion set means stronger constraints for an intersection."""
    witnesses = good @ extends.T
    accepted = np.flatnonzero(witnesses.sum(axis=0) >= ceil(coverage * len(good)))
    order = accepted[np.argsort(extends[accepted].sum(axis=1), kind="stable")]
    maximal = []
    for i in order:
        if not any(np.all(~extends[j] | extends[i]) for j in maximal):
            maximal.append(int(i))
    return maximal


def pair_counts(maps):
    """Number of distinct signed ties. The two real benchmarks use full support."""
    counts = []
    for index in maps:
        if np.any(index == 0):
            raise ValueError("The benchmark pair count assumes full support")
        sizes = np.bincount(abs(index).ravel())[1:]
        counts.append(int(sizes @ (sizes - 1) // 2))
    return np.asarray(counts)
