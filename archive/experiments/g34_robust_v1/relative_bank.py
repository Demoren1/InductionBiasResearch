"""Consistent IRLS on relative residuals, with nonzero scale constraints."""

import json
import time
from datetime import UTC, datetime

import numpy as np
from scipy.optimize import linear_sum_assignment

from experiments.sharing_structure_research_v1.blind_contract import (
    canonical_index,
    digest,
    load_sizes,
    save_json,
)
from experiments.sharing_structure_research_v1.blind_joint_bank import (
    align_source,
    scale_projection,
)


NU = 4.0
MIN_ABS_SCALE = 1e-6
UNCERTAINTY_FLOOR = 0.0003
INNER_STEPS = 3
OUTER_STEPS = 40
BACKTRACK_STEPS = 12
RELATIVE_TOLERANCE = 1e-7


def residual(index, code, scales, normal, uncertainty):
    weight = np.sign(index)[None] * np.c_[np.zeros(len(code)), code][
        :, np.abs(index)
    ]
    deviation = np.maximum(uncertainty, UNCERTAINTY_FLOOR)
    deviation = deviation * np.abs(scales)
    return (weight - scales[:, None] * normal) / deviation[:, None]


def objective(index, code, scales, normal, uncertainty):
    if np.any(np.abs(scales) < MIN_ABS_SCALE):
        return np.full(len(code), np.inf)
    error = residual(index, code, scales, normal, uncertainty)
    return NU * np.log1p(error * error / NU).mean(axis=(1, 2))


def components(index, codes):
    parent = np.arange(codes + index.shape[1])

    def root(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for d, h in zip(*np.where(index)):
        parent[root(abs(index[d, h]) - 1)] = root(codes + h)
    groups = {}
    for node in range(len(parent)):
        groups.setdefault(root(node), []).append(node)
    return [
        (
            [node for node in nodes if node < codes],
            [node - codes for node in nodes if node >= codes],
        )
        for nodes in groups.values()
        if any(node >= codes for node in nodes)
    ]


def regauge(index, code, scales, uncertainty):
    """Enforce the original per-component anchor constraints."""
    code, scales = code.copy(), scales.copy()
    for labels, neurons in components(index, code.shape[1]):
        for task in range(len(code)):
            anchor = min(neurons, key=lambda h: uncertainty[task, h])
            factor = scales[task, anchor]
            if not np.isfinite(factor) or abs(factor) < 1e-12:
                return None
            code[task, labels] /= factor
            scales[task, neurons] /= factor
    if np.any(np.abs(scales) < MIN_ABS_SCALE):
        return None
    return code, scales


def constrained_assignment(costs, mask, codes):
    """Exact nonempty-group assignment, including the optimal signs.

    Give every code one distinct representative position. Other positions take
    their unconstrained minimum. The rectangular Hungarian problem minimizes
    the extra cost of those representatives, so this enforces lower quotas
    without a heuristic reseeding step.
    """
    symbols = np.r_[np.arange(1, codes + 1), -np.arange(1, codes + 1)]
    local = costs[mask]
    best = local.argmin(1)
    best_cost = local[np.arange(len(local)), best]
    unsigned = np.minimum(local[:, :codes], local[:, codes:])
    labels, positions = linear_sum_assignment(
        (unsigned - best_cost[:, None]).T
    )
    for label, position in zip(labels, positions):
        best[position] = label + (
            codes
            if local[position, label + codes] < local[position, label]
            else 0
        )
    index = np.zeros(mask.shape, dtype=np.int16)
    index[mask] = symbols[best]
    return index


def complete_seed(initial, mask, codes):
    index = initial.copy()
    index[~mask] = 0
    positions = np.flatnonzero(mask)
    missing = index.ravel()[positions] == 0
    index.ravel()[positions[missing]] = np.resize(
        np.arange(1, codes + 1), missing.sum()
    )
    for label in range(1, codes + 1):
        if np.any(np.abs(index) == label):
            continue
        counts = np.bincount(np.abs(index).ravel(), minlength=codes + 1)
        available = positions[counts[np.abs(index).ravel()[positions]] > 1]
        index.ravel()[available[0]] = label
    return index


def numerical_step(index, code, scales, normal, uncertainty, robust, stats):
    """Frozen-scale IRLS, with full-objective acceptance and a full Jacobian.

    If the frozen-denominator direction cannot lower Q, try the Jacobian of
    the actual normalized residual, including its scale derivative. Both
    directions receive the same chosen robust weights and full-Q safeguard.
    """
    tasks, dimensions, hidden = normal.shape
    codes = code.shape[1]
    groups = components(index, codes)
    values = objective(index, code, scales, normal, uncertainty)
    for task in range(tasks):
        anchors = {
            min(neurons, key=lambda h: uncertainty[task, h])
            for _, neurons in groups
        }
        free = [h for h in range(hidden) if h not in anchors]
        matrix = np.zeros((dimensions * hidden, codes + len(free)))
        target = np.zeros(dimensions * hidden)
        rows = np.arange(dimensions * hidden).reshape(dimensions, hidden)
        active = index != 0
        matrix[rows[active], np.abs(index[active]) - 1] = np.sign(
            index[active]
        )
        for column, h in enumerate(free, start=codes):
            matrix[rows[:, h], column] = -normal[task, :, h]
        for h in anchors:
            target[rows[:, h]] = normal[task, :, h]
        current = np.r_[code[task], scales[task, free]]
        deviation = np.maximum(uncertainty[task], UNCERTAINTY_FLOOR)
        deviation = deviation * np.abs(scales[task])
        deviation = np.tile(deviation, dimensions)
        error = (matrix @ current - target) / deviation
        omega = 1 / (1 + error * error / NU) if robust else np.ones_like(error)
        whitening = np.sqrt(omega) / deviation
        solution = np.linalg.lstsq(
            matrix * whitening[:, None], target * whitening, rcond=1e-10
        )[0]
        stats["linear_solves"] += 1

        def accept(direction):
            start = min(
                1.0,
                10 * (np.linalg.norm(current) + 1) / (
                    np.linalg.norm(direction) + 1e-30
                ),
            )
            for backtrack in range(BACKTRACK_STEPS):
                candidate = current + (start / 2**backtrack) * direction
                trial_scale = np.ones(hidden)
                trial_scale[free] = candidate[codes:]
                if np.any(np.abs(trial_scale) < MIN_ABS_SCALE):
                    continue
                value = objective(
                    index,
                    candidate[None, :codes],
                    trial_scale[None],
                    normal[task : task + 1],
                    uncertainty[task : task + 1],
                )[0]
                stats["full_q_evaluations"] += 1
                if np.isfinite(value) and value < values[task] - 1e-12:
                    code[task] = candidate[:codes]
                    scales[task] = trial_scale
                    values[task] = value
                    stats["accepted_numeric"] += 1
                    stats["backtracks"] += backtrack
                    return True
            return False

        if accept(solution - current):
            continue
        stats["frozen_direction_rejected"] += 1
        jacobian = matrix / deviation[:, None]
        for column, h in enumerate(free, start=codes):
            jacobian[rows[:, h], column] -= (
                error[rows[:, h]] / scales[task, h]
            )
        delta = np.linalg.lstsq(
            jacobian * np.sqrt(omega)[:, None],
            -error * np.sqrt(omega),
            rcond=1e-10,
        )[0]
        stats["linear_solves"] += 1
        if accept(delta):
            stats["jacobian_fallback_accepted"] += 1
        else:
            stats["numeric_no_progress"] += 1
    return code, scales


def joint_assign(initial, mask, normal, uncertainty, codes, robust):
    if int(mask.sum()) < codes:
        return None
    index = complete_seed(initial, mask, codes)
    code, scales = scale_projection(index, normal, uncertainty, codes)
    scales = np.where(scales >= 0, 1, -1) * np.maximum(
        np.abs(scales), MIN_ABS_SCALE
    )
    stats = {
        name: 0
        for name in (
            "linear_solves",
            "full_q_evaluations",
            "accepted_numeric",
            "backtracks",
            "frozen_direction_rejected",
            "jacobian_fallback_accepted",
            "numeric_no_progress",
            "assignment_accepted",
            "gauge_assignment_rejected",
        )
    }
    curve = [float(objective(index, code, scales, normal, uncertainty).mean())]
    for _ in range(OUTER_STEPS):
        previous = curve[-1]
        for _ in range(INNER_STEPS):
            before = curve[-1]
            code, scales = numerical_step(
                index, code, scales, normal, uncertainty, robust, stats
            )
            curve.append(
                float(
                    objective(index, code, scales, normal, uncertainty).mean()
                )
            )
            if before - curve[-1] <= RELATIVE_TOLERANCE * max(1.0, before):
                break
        centers = np.c_[code, -code]
        target = scales[:, None] * normal
        deviation = np.maximum(uncertainty, UNCERTAINTY_FLOOR)
        deviation = deviation * np.abs(scales)
        error = (target[..., None] - centers[:, None, None]) / (
            deviation[:, None, :, None]
        )
        costs = (NU * np.log1p(error * error / NU)).mean(0)
        updated = constrained_assignment(costs, mask, codes)
        changed = not np.array_equal(index, updated)
        if changed:
            gauged = regauge(updated, code, scales, uncertainty)
            value = (
                float(objective(updated, *gauged, normal, uncertainty).mean())
                if gauged is not None
                else np.inf
            )
            if value <= curve[-1]:
                index, (code, scales) = updated, gauged
                stats["assignment_accepted"] += 1
                curve.append(value)
            else:
                stats["gauge_assignment_rejected"] += 1
                changed = False
        if not changed and previous - curve[-1] <= (
            RELATIVE_TOLERANCE * max(1.0, previous)
        ):
            break
    return index, code, scales, curve[-1], {
        "q": curve,
        "minimum_abs_scale": float(np.min(np.abs(scales))),
        "scales_below_0_01": int(np.sum(np.abs(scales) < 0.01)),
        "scales_at_lower_bound": int(
            np.sum(np.abs(scales) <= MIN_ABS_SCALE * (1 + 1e-8))
        ),
        **stats,
    }


def propose(root, case, condition, robust, anchors=4, starts=16):
    sizes = load_sizes(root, case)
    codes = sizes["codes"]
    folder = root / "fits" / f"{case}_c{condition}"
    base = dict(np.load(folder / "fitted.npz"))
    tasks = base["parameters"].shape[1]
    source = [
        dict(
            np.load(
                root / "source" / f"{case}_c{condition}_t{t}" / "learned.npz"
            )
        )
        for t in range(tasks)
    ]
    ranked = base["validation_mse"].mean(1).argsort()[:anchors]
    rng = np.random.default_rng(33670019)
    bank, origin, traces = {}, {}, []
    started = time.monotonic()
    for number in ranked:
        anchor = base["indices"][number]
        mappings = [
            align_source(
                anchor,
                base["parameters"][number, t, :codes],
                source[t]["normal"],
                source[t]["uncertainty"],
            )
            for t in range(tasks)
        ]
        normal = np.stack(
            [s["normal"][:, m] for s, m in zip(source, mappings)]
        )
        uncertainty = np.stack(
            [s["uncertainty"][m] for s, m in zip(source, mappings)]
        )
        evidence = np.abs(normal) > np.maximum(3 * uncertainty[:, None], 0.002)
        for votes in (0, 1, 2, 3):
            mask = (
                anchor != 0
                if votes == 0
                else evidence.sum(0) >= min(votes, tasks)
            )
            seeds = [anchor]
            for observed, mapping in zip(source, mappings):
                for key in [k for k in observed if k.startswith("index_")][:2]:
                    seeds.append(observed[key][:, mapping])
            while len(seeds) < starts:
                label = rng.integers(1, codes + 1, size=mask.shape)
                sign = np.where(normal[len(seeds) % tasks] >= 0, 1, -1)
                seeds.append((label * sign).astype(np.int16))
            for seed_number, seed in enumerate(seeds):
                result = joint_assign(
                    seed, mask, normal, uncertainty, codes, robust
                )
                if result is None:
                    continue
                index, code, scales, score, trace = result
                traces.append(
                    {
                        "anchor": int(number),
                        "votes": votes,
                        "seed": seed_number,
                        **trace,
                    }
                )
                if not np.isfinite(score):
                    continue
                for reference in range(tasks):
                    oriented = index * np.where(
                        scales[reference] >= 0, 1, -1
                    )[None]
                    canonical = canonical_index(
                        oriented.astype(np.int16), codes
                    )
                    key = canonical.tobytes()
                    if key in bank and bank[key][0] <= score:
                        continue
                    bank[key] = (score, canonical, np.abs(code))
                    origin[key] = {
                        "anchor": int(number),
                        "votes": votes,
                        "reference": reference,
                        "trace": len(traces) - 1,
                    }
            print(
                json.dumps(
                    {
                        "case": case,
                        "phase": "consistent_candidates",
                        "robust": robust,
                        "anchor": int(number),
                        "votes": votes,
                        "unique": len(bank),
                        "runs": len(traces),
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    if not bank:
        raise ValueError("No complete consistent candidates")
    keys = sorted(bank, key=lambda key: bank[key][0])
    np.savez_compressed(
        folder / "joint_bank.npz",
        indices=np.stack([bank[k][1] for k in keys]),
        score=np.array([bank[k][0] for k in keys]),
        code_guesses=np.stack([bank[k][2] for k in keys]),
    )
    save_json(
        folder / "joint_bank.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "seconds": time.monotonic() - started,
            "anchors": ranked.tolist(),
            "starts": starts,
            "projection_loss": "4 log(1+r^2/4), relative residuals",
            "scale_domain": "abs(a)>=1e-6; no 0.01 denominator floor",
            "robust_weights": robust,
            "origins": [origin[k] for k in keys],
            "teacher_access": False,
            "sha256": digest(folder / "joint_bank.npz"),
        },
    )
    save_json(folder / "optimization.json", {"runs": traces})
