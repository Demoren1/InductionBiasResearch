"""Known feasible primal active-set QP for the G21 numerical continuation."""

import time

import numpy as np

from .count_projection import cap_projection, qp_gap


def function_qp(gram, cross, initial, totals, plan):
    started = time.monotonic()
    groups = len(totals)
    width = len(initial) // groups
    constraints = np.kron(np.eye(groups), np.ones((1, width)))
    value = cap_projection(initial.astype(float), totals)
    zero = value <= plan["active_zero_tolerance"]
    saturated = totals - constraints @ value <= plan["active_zero_tolerance"]
    releases = blockers = fallbacks = stationary_faces = 0
    face_error = None
    message = "iteration_limit"
    for iteration in range(plan["active_iterations"]):
        gap, gradient = qp_gap(value, gram, cross, totals)
        if abs(gap) <= plan["qp_gap_absolute"]:
            message = "fw_gap_satisfied"
            break
        free = ~zero
        active = constraints[saturated][:, free]
        direction = np.zeros_like(value)
        multiplier = np.zeros(groups)
        if free.any():
            if active.size:
                kkt = np.block(
                    [
                        [gram[np.ix_(free, free)], active.T],
                        [active, np.zeros((len(active), len(active)))],
                    ]
                )
                right = np.r_[-gradient[free], np.zeros(len(active))]
            else:
                kkt, right = gram[np.ix_(free, free)], -gradient[free]
            solution = np.linalg.solve(kkt, right)
            direction[free] = solution[: int(free.sum())]
            if active.size:
                multiplier[saturated] = solution[int(free.sum()) :]
        reduced_gradient = gradient + constraints.T @ multiplier
        face_error = (
            float(np.max(np.abs(reduced_gradient[free]))) if free.any() else 0.0
        )
        stationarity_tolerance = plan["qp_gap_absolute"] / (4 * plan["hidden"])
        if (
            face_error > stationarity_tolerance
            and np.max(np.abs(direction)) > plan["active_direction_tolerance"]
        ):
            fraction = 1.0
            blocked_variable = blocked_group = None
            negative = (direction < -plan["active_direction_tolerance"]) & free
            if negative.any():
                ids = np.flatnonzero(negative)
                ratios = -value[ids] / direction[ids]
                selected = int(ratios.argmin())
                if ratios[selected] < fraction:
                    fraction = max(0.0, float(ratios[selected]))
                    blocked_variable = int(ids[selected])
            increase = constraints @ direction
            positive = (increase > plan["active_direction_tolerance"]) & ~saturated
            if positive.any():
                ids = np.flatnonzero(positive)
                ratios = (totals - constraints @ value)[ids] / increase[ids]
                selected = int(ratios.argmin())
                if ratios[selected] < fraction:
                    fraction = max(0.0, float(ratios[selected]))
                    blocked_group, blocked_variable = int(ids[selected]), None
            value = cap_projection(value + fraction * direction, totals)
            zero |= value <= plan["active_zero_tolerance"]
            saturated |= totals - constraints @ value <= plan["active_zero_tolerance"]
            if blocked_variable is not None:
                zero[blocked_variable] = True
                blockers += 1
            if blocked_group is not None:
                saturated[blocked_group] = True
                blockers += 1
            continue
        stationary_faces += 1
        zero_multiplier = reduced_gradient
        zero_candidates = np.where(zero, zero_multiplier, np.inf)
        cap_candidates = np.where(saturated, multiplier, np.inf)
        lowest_zero = float(zero_candidates.min())
        lowest_cap = float(cap_candidates.min())
        tolerance = plan["qp_gap_absolute"] / (4 * plan["hidden"])
        if min(lowest_zero, lowest_cap) < -tolerance:
            if lowest_zero <= lowest_cap:
                zero[int(zero_candidates.argmin())] = False
            else:
                saturated[int(cap_candidates.argmin())] = False
            releases += 1
            continue
        vertex = np.zeros((groups, width))
        grouped = gradient.reshape(groups, width)
        for row, total in enumerate(totals):
            column = int(grouped[row].argmin())
            if grouped[row, column] < 0:
                vertex[row, column] = total
        direction = vertex.flatten() - value
        denominator = float(direction @ gram @ direction)
        fraction = min(1.0, max(0.0, gap / max(denominator, 1e-30)))
        value = cap_projection(value + fraction * direction, totals)
        zero = value <= plan["active_zero_tolerance"]
        saturated = totals - constraints @ value <= plan["active_zero_tolerance"]
        fallbacks += 1
    gap, _ = qp_gap(value, gram, cross, totals)
    violation = max(
        float(np.maximum(-value, 0).max()),
        float(np.maximum(constraints @ value - totals, 0).max()),
    )
    converged = (
        abs(gap) <= plan["qp_gap_absolute"] and violation <= plan["qp_primal_tolerance"]
    )
    return value, {
        "solver": "feasible_primal_active_set",
        "success_flag": converged,
        "message": message,
        "iterations": iteration + 1,
        "releases": releases,
        "blockers": blockers,
        "fw_fallbacks": fallbacks,
        "stationary_faces": stationary_faces,
        "face_stationarity_error": face_error,
        "fw_gap": gap,
        "primal_violation": violation,
        "converged": converged,
        "seconds": time.monotonic() - started,
    }
