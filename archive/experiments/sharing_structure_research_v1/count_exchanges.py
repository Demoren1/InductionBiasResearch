"""Known exact parent and two-transfer neighbourhoods for integer counts."""

import math
import time
from functools import cache

import numpy as np
import torch


@cache
def compositions(total, choices):
    if choices == 1:
        return np.array([[total]], dtype=np.int16)
    output = np.empty(
        (math.comb(total + choices - 1, choices - 1), choices), dtype=np.int16
    )
    cursor = 0
    for first in range(total + 1):
        tail = compositions(total - first, choices - 1)
        output[cursor : cursor + len(tail), 0] = first
        output[cursor : cursor + len(tail), 1:] = tail
        cursor += len(tail)
    return output


def pair_candidate(value, gradient, gram, choices):
    source = np.repeat(np.flatnonzero(value > 0), choices)
    target = source // choices * choices + np.tile(
        np.arange(choices), len(source) // choices
    )
    keep = source != target
    source, target = source[keep], target[keep]
    diagonal = np.diag(gram)
    single = (
        gradient[target]
        - gradient[source]
        + 0.5 * (diagonal[target] + diagonal[source] - 2 * gram[source, target])
    )
    cross = (
        gram[np.ix_(target, target)]
        - gram[np.ix_(target, source)]
        - gram[np.ix_(source, target)]
        + gram[np.ix_(source, source)]
    )
    changes = single[:, None] + single[None, :] + cross
    invalid = (source[:, None] == source[None, :]) & (value[source[:, None]] < 2)
    changes[invalid] = math.inf
    changes[np.tril_indices(len(source), -1)] = math.inf
    best_single = int(single.argmin())
    first, second = np.unravel_index(int(changes.argmin()), changes.shape)
    selected = (
        [best_single]
        if single[best_single] <= changes[first, second]
        else [first, second]
    )
    delta = np.zeros_like(value)
    for item in selected:
        delta[source[item]] -= 1
        delta[target[item]] += 1
    return (
        delta,
        float(gradient @ delta + 0.5 * delta @ gram @ delta),
        len(source) + len(source) * (len(source) + 1) // 2,
    )


@torch.no_grad()
def parent_candidate(value, gradient, gram, totals, choices, device, chunk):
    best = 0.0
    delta = np.zeros_like(value)
    evaluated = 0
    for row, total in enumerate(totals):
        begin, end = row * choices, (row + 1) * choices
        block = torch.as_tensor(gram[begin:end, begin:end], device=device)
        derivative = torch.as_tensor(gradient[begin:end], device=device)
        current = torch.as_tensor(value[begin:end], device=device)
        table = compositions(int(total), choices)
        evaluated += len(table)
        for offset in range(0, len(table), chunk):
            candidates = torch.as_tensor(
                table[offset : offset + chunk], device=device, dtype=current.dtype
            )
            changes = candidates - current
            objective = changes @ derivative + 0.5 * ((changes @ block) * changes).sum(
                1
            )
            position = int(objective.argmin())
            improvement = float(objective[position])
            if improvement < best:
                best = improvement
                delta.fill(0)
                delta[begin:end] = changes[position].cpu().numpy()
    return delta, best, evaluated


def search_counts(counts, gram, cross, plan, *, kind, device):
    started = time.monotonic()
    groups, choices = counts.shape
    width = groups * choices
    keep = np.arange(width).reshape(groups, choices)[:, 1:].flatten()
    full_gram = np.zeros((width, width))
    full_gram[np.ix_(keep, keep)] = gram
    full_cross = np.zeros(width)
    full_cross[keep] = cross
    value = counts.flatten().astype(float)
    totals = counts.sum(1)
    updates = evaluations = parent_updates = pair_updates = 0
    objective_change = 0.0
    history = []
    for iteration in range(plan["exchange_iterations"] + 1):
        gradient = full_gram @ value - full_cross
        best, delta, origin = 0.0, np.zeros_like(value), None
        if kind in ("pair", "parent_pair"):
            direction, change, evaluated = pair_candidate(
                value, gradient, full_gram, choices
            )
            evaluations += evaluated
            if change < best:
                best, delta, origin = change, direction, "pair"
        if kind in ("parent", "parent_pair"):
            direction, change, evaluated = parent_candidate(
                value,
                gradient,
                full_gram,
                totals,
                choices,
                device,
                plan["composition_chunk"],
            )
            evaluations += evaluated
            if change < best:
                best, delta, origin = change, direction, "parent"
        if (
            best >= -plan["integer_decrease"]
            or iteration == plan["exchange_iterations"]
        ):
            break
        before = value.copy()
        value += delta
        if np.any(value < 0) or not np.array_equal(
            value.reshape(groups, choices).sum(1), totals
        ):
            raise RuntimeError("Integer exchange violated count feasibility")
        actual = float(
            delta @ (full_gram @ before - full_cross) + 0.5 * delta @ full_gram @ delta
        )
        if actual >= -plan["integer_decrease"]:
            raise RuntimeError(
                "Proposed exchange did not improve the original task quadratic"
            )
        objective_change += actual
        updates += 1
        parent_updates += origin == "parent"
        pair_updates += origin == "pair"
        history.append(
            {
                "update": updates,
                "origin": origin,
                "units": int(np.abs(delta).sum() // 2),
                "half_mse_change": actual,
                "cumulative_half_mse_change": objective_change,
            }
        )
    return value.reshape(groups, choices).astype(np.int16), {
        "neighbourhood": kind,
        "moves": updates,
        "parent_updates": parent_updates,
        "pair_updates": pair_updates,
        "candidates_evaluated": evaluations,
        "half_mse_change": objective_change,
        "minimum_next_change": best,
        "local_stationary": best >= -plan["integer_decrease"],
        "history": history,
        "seconds": time.monotonic() - started,
    }
