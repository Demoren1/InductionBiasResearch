"""Approximate SRR: signed-profile geometry and partial-sharing hierarchies."""

import time

import numpy as np
import torch
from scipy.cluster.hierarchy import cut_tree, linkage

from experiments.omniglot_prior_v1.model import calibration


def canonical(index):
    """Remove label and code-sign gauges, retaining zero support explicitly."""
    flat = np.asarray(index).ravel()
    active = np.flatnonzero(flat)
    labels, first = np.unique(np.abs(flat[active]), return_index=True)
    order = np.argsort(first)
    mapping = np.zeros(int(labels.max(initial=0)) + 1, dtype=np.int16)
    for number, group in enumerate(order, 1):
        label = labels[group]
        mapping[label] = number * np.sign(flat[active[first[group]]])
    return (mapping[np.abs(index)] * np.sign(index)).astype(np.int16)


def projective_distances(profiles, *, device="cpu", block=256):
    """Euclidean distances of zz.T / ||z||, without storing lifted matrices."""
    count = len(profiles)
    result = np.empty(count * (count - 1) // 2, dtype=np.float64)
    norms = np.linalg.norm(profiles, axis=1)
    square = norms**2
    tensor = None
    if device != "cpu":
        tensor = torch.as_tensor(profiles, dtype=torch.float64, device=device)
    for begin in range(0, count, block):
        stop = min(begin + block, count)
        if tensor is None:
            product = profiles[begin:stop] @ profiles.T
        else:
            product = (tensor[begin:stop] @ tensor.T).cpu().numpy()
        denominator = np.maximum(norms[begin:stop, None] * norms[None], 1e-30)
        distance = (
            square[begin:stop, None] + square[None] - 2 * product**2 / denominator
        )
        np.sqrt(np.maximum(distance, 0), out=distance)
        for local, row in enumerate(range(begin, stop)):
            offset = (2 * count - row - 1) * row // 2
            result[offset : offset + count - row - 1] = distance[local, row + 1 :]
    return result


def orient(profiles, labels):
    """Infer signs inside groups using a representative and raw-profile means."""
    groups = int(labels.max()) + 1
    norms = np.sum(profiles**2, axis=1)
    order = np.lexsort((-norms, labels))
    first = np.r_[True, labels[order[1:]] != labels[order[:-1]]]
    representatives = order[first]
    center = profiles[representatives]
    signs = np.ones(len(profiles), dtype=np.int16)
    for _ in range(4):
        updated = np.where(np.sum(profiles * center[labels], axis=1) >= 0, 1, -1)
        signs = updated.astype(np.int16)
        center = np.zeros((groups, profiles.shape[1]))
        np.add.at(center, labels, profiles * signs[:, None])
        center /= np.bincount(labels, minlength=groups)[:, None]
    return (labels + 1).astype(np.int16) * signs


def signed_forest(weights, budgets, *, device="cpu"):
    """Construct all prescribed cuts of one global signed-profile hierarchy."""
    profiles = weights.reshape(len(weights), -1).T
    started = time.perf_counter()
    distances = projective_distances(profiles, device=device)
    tree = linkage(distances, method="ward")
    del distances
    cuts = cut_tree(tree, n_clusters=list(budgets))
    bank = {}
    for column, budget in enumerate(budgets):
        index = orient(profiles, cuts[:, column]).reshape(weights.shape[1:])
        if budget == profiles.shape[0]:
            index = np.arange(1, budget + 1, dtype=np.int16).reshape(index.shape)
        bank[int(budget)] = canonical(index)
    return bank, {"seconds": time.perf_counter() - started, "tree": tree}


def local_forest(weights, budgets):
    """Keep different neurons separate; choose merge counts by Ward costs."""
    _, dimensions, hidden = weights.shape
    trees = []
    for neuron in range(hidden):
        profile = weights[:, :, neuron].T
        trees.append(linkage(projective_distances(profile), method="ward"))
    # Ward heights are monotone within each Euclidean tree. A global prefix
    # therefore respects every local tree's order of merges.
    costs = np.concatenate([tree[:, 2] for tree in trees])
    neurons = np.repeat(np.arange(hidden), dimensions - 1)
    order = np.argsort(costs, kind="stable")
    bank = {}
    for budget in budgets:
        if budget < hidden:
            continue
        counts = np.bincount(
            neurons[order[: dimensions * hidden - budget]], minlength=hidden
        )
        index, offset = np.zeros((dimensions, hidden), np.int16), 0
        for neuron in range(hidden):
            groups = dimensions - int(counts[neuron])
            labels = cut_tree(trees[neuron], n_clusters=[groups])[:, 0]
            local = orient(weights[:, :, neuron].T, labels)
            index[:, neuron] = np.sign(local) * (np.abs(local) + offset)
            offset += groups
        bank[int(budget)] = canonical(index)
    return bank


def input_forest(weights, budgets):
    """Learn common input groups without matching neurons between tasks.

    Different neurons receive independent codes. This is a structured
    candidate family, not a consequence of unrestricted global sharing.
    No pixel coordinates or spatial-neighbour graph are supplied.
    """
    _, dimensions, hidden = weights.shape
    profiles = weights.transpose(1, 0, 2).reshape(dimensions, -1)
    tree = linkage(profiles, method="ward")
    budgets = [
        budget
        for budget in budgets
        if hidden <= budget <= dimensions * hidden and budget % hidden == 0
    ]
    cuts = cut_tree(tree, n_clusters=[budget // hidden for budget in budgets])
    bank = {}
    for column, budget in enumerate(budgets):
        labels = cuts[:, column]
        index = labels[:, None] * hidden + np.arange(1, hidden + 1)[None]
        bank[int(budget)] = canonical(index.astype(np.int16))
    return bank


def synchronize_profiles(index, weights, uncertainty):
    """One positive-scale calibration; retain the independently built seed."""
    expanded = np.broadcast_to(index, weights.shape)
    _, scales = calibration(expanded, weights, uncertainty)
    calibrated = weights * scales[:, None]
    # One common factor per task preserves comparisons between neurons.
    norm = np.sqrt(np.mean(calibrated**2, axis=(1, 2))).clip(1e-12)
    return calibrated / norm[:, None, None]


def source_fit(index, weights):
    """Signed least-squares projection in the current calibrated coordinates."""
    labels = np.abs(index).ravel() - 1
    signs = np.sign(index).ravel()
    groups = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=groups)
    profiles = weights.reshape(len(weights), -1)
    code = np.stack(
        [
            np.bincount(labels, row * signs, minlength=groups) / counts
            for row in profiles
        ]
    )
    prediction = code[:, labels] * signs[None]
    return prediction.reshape(weights.shape), float(
        np.mean((prediction - profiles) ** 2)
    )
