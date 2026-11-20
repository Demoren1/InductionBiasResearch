"""Propose full and partial indexators from signed weight profiles."""

import json
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import cut_tree, fcluster, linkage

from experiments.srr_approx_v1.core import orient, projective_distances

ROOT = Path("artifacts/asrr_benchmark_families_2026-11-19")
LOCAL = ROOT.parent / "local" / ROOT.name
PLAN_PATH = Path(__file__).with_name("plan.json")
PLAN = json.loads(PLAN_PATH.read_text())


def consistency_distances(halves, signed):
    """Maximum relative discrepancy on two source halves with one common sign."""
    count = len(halves[0])
    distances = np.empty(count * (count - 1) // 2)
    energies = [np.square(z).sum(1) for z in halves]
    for start in range(0, count, 128):
        end = min(start + 128, count)
        agreement = []
        for z, e in zip(halves, energies, strict=True):
            denom = np.maximum(e[start:end, None] + e[None, :], 1e-15)
            agreement.append(2 * (z[start:end] @ z.T) / denom)
        value = np.minimum(*agreement)
        if signed:
            value = np.maximum(value, -np.maximum(*agreement))
        block = np.clip(1 - value, 0, 2)
        for row in range(start, end):
            begin = row * (2 * count - row - 1) // 2
            distances[begin : begin + count - row - 1] = block[row - start, row + 1 :]
    return distances


def label_tree(profiles, values, *, partial, signed, halves=None):
    if len(profiles) < 2:
        return {v: np.zeros(len(profiles), dtype=int) for v in values}
    if partial:
        if halves is None:
            halves = (profiles[:, ::2], profiles[:, 1::2])
        distances = consistency_distances(halves, signed)
        tree = linkage(distances, method="complete")
        return {v: fcluster(tree, v, criterion="distance") - 1 for v in values}
    tree = linkage(
        projective_distances(profiles) if signed else profiles, method="ward"
    )
    budgets = sorted({min(int(v), len(profiles)) for v in values})
    cuts = dict(zip(budgets, cut_tree(tree, n_clusters=budgets).T, strict=True))
    return {v: cuts[min(v, len(profiles))] for v in values}


def signed_bank(weights, mask, values, *, partial):
    profiles = weights[:, mask].T
    labels = label_tree(profiles, values, partial=partial, signed=True)
    maps = {}
    for value, groups in labels.items():
        index = np.zeros(mask.shape, dtype=np.int32)
        index[mask] = orient(profiles, groups)
        maps[value] = index
    return maps


def input_bank(weights, blocks, values, *, partial):
    tasks, inputs, hidden = weights.shape
    all_groups = []
    for slab in np.split(weights, blocks, axis=2):
        profiles = slab.transpose(1, 0, 2).reshape(inputs, tasks, -1)
        halves = tuple(profiles[:, k::2].reshape(inputs, -1) for k in (0, 1))
        all_groups.append(
            label_tree(
                profiles.reshape(inputs, -1),
                values,
                partial=partial,
                signed=False,
                halves=halves,
            )
        )
    result = {}
    for v in values:
        compiled, offset = [], 0
        width = hidden // blocks
        for groups in all_groups:
            labels = groups[v][:, None] * width + np.arange(1, width + 1) + offset
            compiled.append(labels)
            offset = int(labels.max())
        result[v] = np.concatenate(compiled, axis=1).astype(np.int32)
    return result


def shuffled(index, rng, *, input_groups=False):
    if input_groups:
        return index[rng.permutation(len(index))]
    copy = index.copy()
    for source, dest in zip(index.T, copy.T, strict=True):
        active = source != 0
        dest[active] = np.sign(source[active]) * rng.permutation(np.abs(source[active]))
    return copy


def untied(mask):
    index = np.zeros(mask.shape, dtype=np.int32)
    index[mask] = np.arange(1, mask.sum() + 1)
    return index
