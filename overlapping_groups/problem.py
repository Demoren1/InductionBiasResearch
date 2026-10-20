"""Independent inputs and families sharing half of their first-layer edges."""

import math

import numpy as np
import torch
from torch.nn import functional as F

INPUTS, HIDDEN, EDGES = 48, 8, 32


def families(config):
    """Generate distinct groups, with 16–20 aligned common edges per pair."""
    rng = np.random.default_rng(config["seed"])
    accepted, assignments = [], []
    used_pairs = [set() for _ in range(HIDDEN)]
    required = sum(config["families"].values())
    attempts = 0
    while len(accepted) < required:
        attempts += 1
        if attempts > 1000000:
            raise RuntimeError(
                "Family constraints exhausted the candidate budget"
            )
        groups = rng.permutation(16).reshape(HIDDEN, 2)
        pairs = [tuple(sorted(group)) for group in groups]
        if any(pair in used for pair, used in zip(pairs, used_pairs)):
            continue
        assignment = np.empty(16, dtype=int)
        for column, group in enumerate(groups):
            assignment[group] = column
        if any(np.count_nonzero(assignment == old) > 4 for old in assignments):
            continue
        accepted.append((groups + 16).tolist())
        assignments.append(assignment)
        for pair, used in zip(pairs, used_pairs):
            used.add(pair)
    result = []
    index = 0
    for split, count in config["families"].items():
        for _ in range(count):
            result.append(
                {"id": index, "split": split, "groups": accepted[index]}
            )
            index += 1
    intersections = [
        16 + int(np.count_nonzero(left == right))
        for i, left in enumerate(assignments)
        for right in assignments[i + 1 :]
    ]
    return result, {
        "candidate_draws": attempts,
        "common_edges": 16,
        "pair_intersection_min": min(intersections),
        "pair_intersection_mean": float(np.mean(intersections)),
        "pair_intersection_max": max(intersections),
        "indices": "zero-based; each groups entry lists two variable inputs",
    }


def reference(family, device):
    mask = torch.zeros(INPUTS, HIDDEN, device=device)
    for hidden, pair in enumerate(family["groups"]):
        mask[[2 * hidden, 2 * hidden + 1, *pair], hidden] = 1
    return mask


def teacher(config, family, function, device):
    seed = config["seed"] + 100003 * family["id"] + 1009 * function + 1
    generator = torch.Generator().manual_seed(seed)
    mask = reference(family, "cpu")
    weights = (0.5 + torch.rand(INPUTS, HIDDEN, generator=generator)) * mask
    weights /= weights.square().sum(0).sqrt()
    thresholds = torch.rand(HIDDEN, generator=generator) - 0.5
    signs = torch.randint(2, (HIDDEN,), generator=generator) * 2 - 1
    readout = signs * (0.8 + 0.4 * torch.rand(HIDDEN, generator=generator))
    readout /= math.sqrt(HIDDEN)
    expected_hidden = torch.exp(-0.5 * thresholds.square()) / math.sqrt(
        2 * math.pi
    )
    expected_hidden -= thresholds * 0.5 * torch.erfc(thresholds / math.sqrt(2))
    state = {
        "w": weights,
        "b": -thresholds,
        "a": readout,
        "c": -(expected_hidden * readout).sum(),
    }
    return {key: value.to(device) for key, value in state.items()}


def logits(state, mask, x, negative_slope=0.0, packed=False):
    weights = state["w"] * mask
    if packed and weights.ndim == 3 and x.ndim == 2:
        networks, inputs, hidden_size = weights.shape
        wide_weights = weights.permute(1, 0, 2).reshape(inputs, -1)
        preactivation = (
            (x @ wide_weights)
            .reshape(len(x), networks, hidden_size)
            .permute(1, 0, 2)
        )
    elif packed and weights.ndim == 4 and x.ndim == 3:
        tasks, networks, inputs, hidden_size = weights.shape
        wide_weights = weights.permute(0, 2, 1, 3).reshape(
            tasks, inputs, networks * hidden_size
        )
        preactivation = (
            torch.bmm(x, wide_weights)
            .reshape(tasks, x.shape[1], networks, hidden_size)
            .permute(0, 2, 1, 3)
        )
    elif weights.ndim == 4 and x.ndim == 3:
        preactivation = torch.einsum("tni,tbih->tbnh", x, weights)
    else:
        preactivation = x @ weights
    hidden = F.leaky_relu(
        preactivation + state["b"].unsqueeze(-2),
        negative_slope=negative_slope,
    )
    return (hidden * state["a"].unsqueeze(-2)).sum(-1) + state["c"].unsqueeze(
        -1
    )


def observations(config, family, function, state, device, splits):
    result = {}
    for name in splits:
        offset = {"support": 11, "query": 23, "test": 37}[name]
        seed = (
            config["seed"] + 100003 * family["id"] + 1009 * function + offset
        )
        generator = torch.Generator().manual_seed(seed)
        x = torch.randn(config["data"][name], INPUTS, generator=generator).to(
            device
        )
        with torch.no_grad():
            y = logits(state, torch.ones_like(state["w"]), x).sigmoid()
        result[name] = (x, y)
    return result
