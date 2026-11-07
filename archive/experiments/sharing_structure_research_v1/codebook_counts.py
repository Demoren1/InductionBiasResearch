"""Integer-intercept segmentation is a relaxed DP, not an exact spline solver."""

import math
import time

import numpy as np
import torch
from scipy.optimize import least_squares, nnls

from .prefix_census import segmented_start


@torch.no_grad()
def integer_segments(x, y, plan, device):
    started = time.monotonic()
    count, maximum = len(x), plan["inputs"] * plan["hidden"]
    beta = plan["bias"] / math.sqrt(plan["hidden"])
    tail = float(np.polyfit(x[-16:], y[-16:], 1)[0])
    residual_y = y - tail * x
    t = torch.tensor(x, device=device, dtype=torch.float64)
    target = torch.tensor(residual_y, device=device, dtype=torch.float64)
    statistics = [
        torch.cat((t.new_zeros(1), value.cumsum(0)))
        for value in (t, t.square(), target, target.square(), t * target)
    ]
    mass = torch.arange(maximum + 1, device=device, dtype=torch.float64)
    starts = torch.arange(count + 1, device=device)
    previous = t.new_full((count + 1, maximum + 1), math.inf)
    original_square = np.r_[0.0, np.cumsum(y * y)]
    previous[2:, 0] = torch.tensor(original_square[2:], device=device)
    parents = []
    for _ in range(plan["codes"]):
        prefix_value, prefix_index = torch.cummin(previous, dim=1)
        prefix_value = torch.cat(
            (previous.new_full((count + 1, 1), math.inf), prefix_value[:, :-1]), dim=1
        )
        prefix_index = torch.cat(
            (prefix_index.new_zeros((count + 1, 1)), prefix_index[:, :-1]), dim=1
        )
        current = torch.full_like(previous, math.inf)
        left_parent = torch.zeros_like(previous, dtype=torch.int16)
        mass_parent = torch.zeros_like(left_parent)
        for begin in range(1, count + 1, plan["dp_chunk"]):
            ends = torch.arange(
                begin, min(count + 1, begin + plan["dp_chunk"]), device=device
            )
            n = (ends[:, None] - starts).clamp_min(1).to(t.dtype)
            sx, sxx, sy, syy, sxy = [
                value[ends, None] - value[starts] for value in statistics
            ]
            inverse = sxx.clamp_min(1e-30).reciprocal()
            constant = syy - sxy.square() * inverse
            linear = 2 * beta * (sy - sxy * sx * inverse)
            quadratic = beta**2 * (n - sx.square() * inverse)
            cost = (
                constant[..., None]
                + linear[..., None] * mass
                + quadratic[..., None] * mass.square()
            )
            cost.clamp_min_(0)
            cost[(ends[:, None] - starts) < 2] = math.inf
            candidate = cost + prefix_value[None]
            best, where = candidate.min(dim=1)
            current[ends] = best
            left_parent[ends] = where.to(torch.int16)
            previous_mass = prefix_index[
                where, torch.arange(maximum + 1, device=device)[None]
            ]
            mass_parent[ends] = previous_mass.to(torch.int16)
        parents.append((left_parent.cpu().numpy(), mass_parent.cpu().numpy()))
        previous = current
    total = int(previous[-1].argmin())
    bound = float(previous[-1, total])
    end, current_mass, segments = count, total, []
    for left_parent, mass_parent in reversed(parents):
        left = int(left_parent[end, current_mass])
        old_mass = int(mass_parent[end, current_mass])
        segments.append((left, end, current_mass))
        end, current_mass = left, old_mass
    segments.append((0, end, 0))
    segments.reverse()
    slopes, direct_sse = [0.0], float(np.dot(y[:end], y[:end]))
    for left, right, cumulative in segments[1:]:
        local_x, local_y = x[left:right], y[left:right]
        slope = float(local_x @ (local_y + beta * cumulative) / (local_x @ local_x))
        slopes.append(slope)
        error = local_y - slope * local_x + beta * cumulative
        direct_sse += float(error @ error)
    amounts = np.diff([segment[2] for segment in segments])
    increments = np.diff(slopes)
    raw_codes = increments * math.sqrt(plan["hidden"]) / amounts
    knots = np.divide(
        beta * amounts,
        increments,
        out=np.full_like(increments, math.inf),
        where=increments > 0,
    )
    physical = bool((increments > 0).all())
    for knot, (left, _, _) in zip(knots, segments[1:], strict=True):
        physical &= bool(x[left - 1] <= knot <= x[left])
    initial = np.sort(
        np.clip(raw_codes, plan["axis_code_min"] * 1.01, plan["axis_code_max"] / 1.01)
    )
    return (
        initial,
        amounts[::-1].copy(),
        {
            "relaxed_dp_sse": bound,
            "direct_path_sse": direct_sse,
            "path_numeric_difference": direct_sse - bound,
            "raw_physical": physical,
            "segments": [list(map(int, value)) for value in segments],
            "raw_descending_codes": raw_codes.tolist(),
            "masses_ascending": amounts[::-1].tolist(),
            "seconds": time.monotonic() - started,
        },
    )


def curve_fit(x, y, vx, vy, plan, seed, method, dp_initial, masses):
    first = (
        plan["bias"] / segmented_start(x, y, plan["codes"])
        if method == "continuous"
        else dp_initial
    )
    starts = [first, [0.15, 0.4, 1.0, 2.5], [0.3, 0.6, 1.0, 1.7]]
    rng = np.random.default_rng(seed)
    while len(starts) < plan["axis_starts"]:
        starts.append(np.exp(rng.uniform(np.log(0.03), np.log(4), plan["codes"])))

    def residual(log_codes, return_amplitude=False):
        codes = np.sort(np.exp(log_codes))
        matrix = np.maximum(x[:, None] * codes - plan["bias"], 0)
        if method == "integer_mass":
            amplitude = masses / math.sqrt(plan["hidden"])
        else:
            scale = np.maximum(np.linalg.norm(matrix, axis=0), 1e-30)
            amplitude, _ = nnls(matrix / scale, y, maxiter=100)
            amplitude /= scale
        error = matrix @ amplitude - y
        return (error, amplitude) if return_amplitude else error

    attempts = []
    for initial in starts:
        initial = np.clip(
            initial, plan["axis_code_min"] * 1.001, plan["axis_code_max"] / 1.001
        )
        result = least_squares(
            residual,
            np.log(initial),
            bounds=(math.log(plan["axis_code_min"]), math.log(plan["axis_code_max"])),
            max_nfev=plan["axis_solver_evaluations"],
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-11,
        )
        codes = np.sort(np.exp(result.x))
        error, amplitude = residual(result.x, return_amplitude=True)
        validation_error = (
            np.maximum(vx[:, None] * codes - plan["bias"], 0) @ amplitude - vy
        )
        region = np.searchsorted(x, np.sort(plan["bias"] / codes))
        counts = np.diff(np.r_[0, region, len(x)])
        attempts.append(
            {
                "codes": codes.tolist(),
                "amplitude": amplitude.tolist(),
                "train_mse": float(np.mean(error**2)),
                "validation_mse": float(np.mean(validation_error**2)),
                "evaluations": result.nfev,
                "success": bool(result.success),
                "min_points_per_region": int(counts.min()),
            }
        )
    selected = int(np.argmin([row["validation_mse"] for row in attempts]))
    best = attempts[selected]
    return np.asarray(best["codes"]), {
        "selected_validation_restart": selected,
        "selected_training_restart": int(
            np.argmin([row["train_mse"] for row in attempts])
        ),
        "selected": best,
        "attempts": attempts,
    }
