"""Global compatibility certificate for scalar signed indexators.

Given simultaneous weight intervals and exactly K signed groups, return an I
only when all assignments are forced in the K-witness branch. Interval piercing
is a classical subroutine; THEORY.md section17 states the specific certificate.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def bounded_cover(
    lower, upper, range_lower, range_upper, forced, force_lower, force_upper
):
    cursor, previous = 0, -np.inf
    count = len(lower)
    for group in range(len(range_lower)):
        while cursor < count and lower[cursor] <= previous:
            cursor += 1
        deadline = upper[cursor] if cursor < count else np.inf
        left, right = range_lower[group], range_upper[group]
        if group == forced:
            left = max(left, force_lower)
            right = min(right, force_upper)
        point = min(right, deadline)
        if point < left:
            return False
        previous = point
    while cursor < count and lower[cursor] <= previous:
        cursor += 1
    return cursor == count


@njit(cache=True)
def feasible_assignments(
    lower, upper, required_lower, required_upper, range_lower, range_upper
):
    codes = len(range_lower)
    allowed = np.zeros((len(lower), 2 * codes + 1), dtype=np.bool_)
    checks = 0
    for entry in range(len(lower)):
        allowed[entry, 0] = lower[entry] <= 0 <= upper[entry]
        for sign in (1, -1):
            if sign == 1:
                left, right = max(lower[entry], 0.0), upper[entry]
            else:
                left, right = max(-upper[entry], 0.0), -lower[entry]
            if right <= 0:
                continue
            for group in range(codes):
                if max(range_lower[group], left) > min(range_upper[group], right):
                    continue
                checks += 1
                possible = bounded_cover(
                    required_lower,
                    required_upper,
                    range_lower,
                    range_upper,
                    group,
                    left,
                    right,
                )
                column = group + 1 if sign == 1 else codes + group + 1
                allowed[entry, column] = possible
    return allowed, checks


def interval_indexator(lower, upper, codes):
    """Return the unique compatible signed assignment, or an abstention."""
    lower, upper = (
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
    )
    if lower.shape != upper.shape or lower.size == 0:
        raise ValueError("Intervals must have matching, nonempty shapes")
    if (
        not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or np.any(lower > upper)
    ):
        raise ValueError("Intervals must be finite and ordered")
    if not isinstance(codes, (int, np.integer)) or codes < 1:
        raise ValueError("The exact code count must be a positive integer")
    shape = lower.shape
    low, high = lower.ravel(), upper.ravel()
    required = (low > 0) | (high < 0)
    positive_lower = np.where(low > 0, low, -high)[required]
    positive_upper = np.where(low > 0, high, -low)[required]
    positions = np.flatnonzero(required)
    order = np.argsort(positive_upper, kind="stable")
    positive_lower, positive_upper, positions = (
        positive_lower[order],
        positive_upper[order],
        positions[order],
    )
    witnesses, last = [], -np.inf
    for i, left in enumerate(positive_lower):
        if left > last:
            witnesses.append(i)
            last = positive_upper[i]
    metadata = {
        "minimum_nonzero_codes": len(witnesses),
        "known_nonzero_codes": int(codes),
        "required_positions": int(required.sum()),
        "returned_index": False,
        "witness_positions": positions[witnesses].tolist(),
        "witness_ranges": [
            [float(positive_lower[i]), float(positive_upper[i])] for i in witnesses
        ],
    }
    if len(witnesses) != codes:
        metadata["reason"] = (
            "Minimum interval piercing count differs from exactK; sufficient certificate abstains"
        )
        return None, metadata
    range_lower, range_upper = positive_lower[witnesses], positive_upper[witnesses]
    if not bounded_cover(
        positive_lower, positive_upper, range_lower, range_upper, -1, 0.0, 0.0
    ):
        raise ArithmeticError("Greedy witness intervals should admit a cover")
    allowed, checks = feasible_assignments(
        low, high, positive_lower, positive_upper, range_lower, range_upper
    )
    counts = allowed.sum(-1)
    selected = allowed.argmax(-1)
    indices = np.where(selected > codes, codes - selected, selected).astype(np.int16)
    metadata.update(
        {
            "feasibility_checks": int(checks),
            "ambiguous_positions": int((counts > 1).sum()),
            "infeasible_positions": int((counts == 0).sum()),
            "assignment_counts": counts.tolist(),
            "forced_mask": (counts == 1).tolist(),
            "forced_indices": np.where(counts == 1, indices, 0).tolist(),
        }
    )
    if np.any(counts != 1):
        metadata["reason"] = (
            "Several signed assignments remain compatible with the full interval system"
        )
        return None, metadata
    if len(np.unique(np.abs(indices[indices != 0]))) != codes:
        raise ArithmeticError("Every disjoint witness must force a nonempty group")
    metadata["returned_index"] = True
    return indices.reshape(shape), metadata
