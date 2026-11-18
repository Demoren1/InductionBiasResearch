"""Exact one-dimensional implementation of the two ratio merge rules."""

import numpy as np
from numba import njit


@njit
def _find(parent, point):
    while parent[point] != point:
        parent[point] = parent[parent[point]]
        point = parent[point]
    return point


@njit
def _join(parent, sizes, first, second):
    left, right = _find(parent, first), _find(parent, second)
    if left == right:
        return 0
    if sizes[left] < sizes[right]:
        left, right = right, left
    parent[right] = left
    sizes[left] += sizes[right]
    return 1


@njit
def _partition(size, first, second, delta, zfirst, zsecond, zgap, threshold):
    parent = np.arange(size, dtype=np.int32)
    sizes = np.ones(size, dtype=np.int32)
    count = size
    for position in range(len(zgap)):
        if zgap[position] <= threshold:
            count -= _join(parent, sizes, zfirst[position], zsecond[position])
    begin = np.searchsorted(delta, threshold, side="right")
    for position in range(begin, len(delta) - 1):
        if delta[position + 1] - delta[position] <= 2 * threshold:
            count -= _join(parent, sizes, first[position], first[position + 1])
            count -= _join(
                parent, sizes, second[position], second[position + 1]
            )
            if count == 1:
                break
    for position in range(size):
        parent[position] = _find(parent, position)
    return count, parent


class ScalarPairs:
    """Sort oriented scalar pairs once and recover each threshold exactly.

    Positive differences suffice: negative differences are their mirrors.
    After differences with magnitude <=t are excluded, opposite signs are
    separated by more than 2t. Within each sorted positive cloud component,
    adjacent radius edges span the same endpoint identifications as all
    radius edges. The zero rule likewise has an adjacent-point forest in
    each neuron. No transitive merge from a previous threshold is reused.
    """

    def __init__(self, normal, mask):
        self.rows, self.columns = np.where(mask)
        self.shape = mask.shape
        logs = np.log(np.abs(normal[self.rows, self.columns]))
        first, second, delta = [], [], []
        zfirst, zsecond, zgap = [], [], []
        for neuron in np.unique(self.columns):
            positions = np.flatnonzero(self.columns == neuron)
            positions = positions[np.argsort(logs[positions], kind="stable")]
            left, right = np.triu_indices(len(positions), 1)
            first.append(positions[right].astype(np.int32))
            second.append(positions[left].astype(np.int32))
            delta.append(logs[positions[right]] - logs[positions[left]])
            zfirst.extend(positions[:-1])
            zsecond.extend(positions[1:])
            zgap.extend(np.diff(logs[positions]))
        self.delta = np.concatenate(delta)
        order = np.argsort(self.delta, kind="stable")
        self.delta = np.ascontiguousarray(self.delta[order])
        self.first = np.ascontiguousarray(np.concatenate(first)[order])
        self.second = np.ascontiguousarray(np.concatenate(second)[order])
        self.zfirst = np.asarray(zfirst, dtype=np.int32)
        self.zsecond = np.asarray(zsecond, dtype=np.int32)
        self.zgap = np.asarray(zgap)

    def partition(self, threshold):
        count, labels = _partition(
            len(self.rows),
            self.first,
            self.second,
            self.delta,
            self.zfirst,
            self.zsecond,
            self.zgap,
            threshold,
        )
        _, first, inverse = np.unique(
            labels, return_index=True, return_inverse=True
        )
        label_dtype = np.int16 if count <= np.iinfo(np.int16).max else np.int32
        numbering = np.empty(count, dtype=label_dtype)
        numbering[np.argsort(first)] = np.arange(1, count + 1)
        index = np.zeros(self.shape, dtype=label_dtype)
        index[self.rows, self.columns] = numbering[inverse]
        return index, int(count)
