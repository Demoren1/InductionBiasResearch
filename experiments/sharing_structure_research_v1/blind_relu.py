"""Recover candidate structures using output observations and public sizes only."""

import math
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks


def estimate_noise(observe, dimensions, rng):
    x = rng.normal(size=(32, dimensions))
    repeated = observe(x, repeats=1, replicas=32)
    variance = np.var(repeated, axis=0, ddof=1).mean(axis=0)
    mean = repeated.mean(axis=0)
    signal = max(float(np.var(mean) - variance.mean() / 32), 1e-12)
    return (
        variance,
        signal,
        {
            "variance_estimate": variance.tolist(),
            "signal_variance_estimate": signal,
            "points": 32,
            "independent_replicates": 32,
        },
    )


def averaging_budget(variance, standard_error, maximum=2**26):
    required = max(float(np.max(variance)) / max(standard_error**2, 1e-30), 1)
    return min(2 ** math.ceil(math.log2(required)), maximum)


def fit_hinge(t, y, low, high):
    def residual(knot, output=False):
        design = np.c_[np.ones(len(t)), t, np.maximum(t - knot, 0)]
        coefficient = np.linalg.lstsq(design, y, rcond=1e-12)[0]
        error = float(np.mean((design @ coefficient - y) ** 2))
        return (error, coefficient) if output else error

    result = minimize_scalar(
        residual, bounds=(low, high), method="bounded", options={"xatol": 1e-12}
    )
    error, coefficient = residual(result.x, output=True)
    return float(result.x), coefficient[-1], error


def local_jacobian(observe, point, radius, repeats, variance):
    dimensions = len(point)
    perturbation = radius * np.eye(dimensions)
    x = np.r_[point[None], point + perturbation, point - perturbation]
    y = observe(x, repeats=repeats, replicas=1)[0]
    positive, negative = y[1 : 1 + dimensions], y[1 + dimensions :]
    jacobian = ((positive - negative) / (2 * radius)).T
    curvature = positive + negative - 2 * y[0]
    standardized = np.mean(curvature**2) / max(
        6 * float(np.mean(variance)) / repeats, 1e-28
    )
    return jacobian, float(standardized)


def recover_planes(
    observe, sizes, variance, signal, seed, *, rays=32, precision_factor=1
):
    rng = np.random.default_rng(seed)
    dimensions, hidden = sizes["inputs"], sizes["hidden"]
    repeats = averaging_budget(
        variance,
        math.sqrt(signal) * 1e-4 * precision_factor,
        max(2**22 // precision_factor**2, 1),
    )
    sem = math.sqrt(float(np.max(variance)) / repeats)
    grid = np.linspace(-12, 12, 769)
    spacing = grid[1] - grid[0]
    clusters, trace = [], []
    check_x = (
        rng.normal(size=(768, dimensions))
        * np.resize(np.array([0.5, 1, 2, 4]), 768)[:, None]
    )
    check_y = observe(check_x, repeats=repeats, replicas=1)[0]
    started = time.monotonic()
    for ray in range(rays):
        center = rng.normal(size=dimensions) * (0.5, 1.0, 2.0, 4.0)[ray % 4]
        direction = rng.normal(size=dimensions)
        direction /= np.linalg.norm(direction)
        y = observe(center + grid[:, None] * direction, repeats=repeats, replicas=1)[0]
        slopes = np.diff(y, axis=0) / spacing
        curvature = gaussian_filter1d(np.diff(slopes, axis=0), 1.0, axis=0)
        energy = np.linalg.norm(curvature, axis=1)
        threshold = max(4 * sem / spacing, float(energy.max()) * 1e-5, 1e-12)
        peaks, _ = find_peaks(energy, height=threshold, distance=5)
        peaks = peaks[np.argsort(energy[peaks])[::-1]][: 2 * hidden]
        locations = np.sort(peaks + 1)
        accepted = 0
        for position in locations:
            neighbors = locations[locations != position]
            separation = min(
                float(np.min(np.abs(grid[neighbors] - grid[position])))
                if len(neighbors)
                else 2,
                2,
            )
            half_window = max(min(int(separation / spacing / 3), 12), 3)
            begin, end = (
                max(position - half_window, 0),
                min(position + half_window + 1, len(grid)),
            )
            knot, jump, hinge_error = fit_hinge(
                grid[begin:end],
                y[begin:end],
                grid[max(position - 2, 0)],
                grid[min(position + 2, len(grid) - 1)],
            )
            if np.linalg.norm(jump) < max(sem / spacing, 1e-10):
                continue
            distance = min(separation / 4, 0.4)
            radius = distance / (2 * math.sqrt(dimensions))
            target_error = max(
                float(np.linalg.norm(jump)) * 3e-4 * precision_factor, 1e-8
            )
            left = center + (knot - distance) * direction
            right = center + (knot + distance) * direction
            best_probe = None
            for _ in range(7):
                gradient_repeats = averaging_budget(
                    variance,
                    radius * target_error,
                    max(2**24 // precision_factor**2, 1),
                )
                jl, cl = local_jacobian(
                    observe, left, radius, gradient_repeats, variance
                )
                jr, cr = local_jacobian(
                    observe, right, radius, gradient_repeats, variance
                )
                if max(cl, cr) >= 6:
                    if best_probe is not None:
                        break
                    radius /= 2
                else:
                    best_probe = jl, jr, radius, gradient_repeats
                    radius *= 2
                    if radius > 2 * distance:
                        break
            if best_probe is None:
                continue
            jl, jr, radius, gradient_repeats = best_probe
            difference = jr - jl
            _, singular, vh = np.linalg.svd(difference, full_matrices=False)
            if singular[0] < 1e-10 or (
                len(singular) > 1 and singular[1] > 0.08 * singular[0]
            ):
                continue
            normal = vh[0]
            significant = np.flatnonzero(np.abs(normal) > 0.5 * np.abs(normal).max())
            normal *= np.sign(normal[significant[0]])
            intercept = -float((center + knot * direction) @ normal)
            uncertainty = (
                math.sqrt(2 * float(np.max(variance)) / gradient_repeats)
                / radius
                / singular[0]
            )
            intercept_uncertainty = np.linalg.norm(
                center + knot * direction
            ) * uncertainty + abs(float(normal @ direction)) * math.sqrt(
                hinge_error
            ) / max(float(np.linalg.norm(jump)), 1e-12)
            matches, orientations = [], []
            for row in clusters:
                angular_tolerance = max(
                    0.012,
                    min(
                        0.08,
                        5
                        * math.sqrt(dimensions)
                        * math.hypot(uncertainty, row["uncertainty"]),
                    ),
                )
                intercept_tolerance = max(
                    0.015,
                    5 * math.hypot(intercept_uncertainty, row["intercept_uncertainty"]),
                )
                distances = [
                    np.linalg.norm(row["normal"] - sign * normal) / angular_tolerance
                    + abs(row["intercept"] - sign * intercept) / intercept_tolerance
                    for sign in (1, -1)
                ]
                matches.append(min(distances))
                orientations.append(1 if distances[0] <= distances[1] else -1)
            quality = 1 / max(uncertainty**2, 1e-14)
            if matches and min(matches) < 1:
                chosen = int(np.argmin(matches))
                normal *= orientations[chosen]
                intercept *= orientations[chosen]
                row = clusters[chosen]
                total = row["quality"] + quality
                row["normal"] = (
                    row["normal"] * row["quality"] + normal * quality
                ) / total
                row["intercept"] = (
                    row["intercept"] * row["quality"] + intercept * quality
                ) / total
                row["quality"] = total
                row["count"] += 1
                row["uncertainty"] = min(row["uncertainty"], uncertainty)
                row["intercept_uncertainty"] = min(
                    row["intercept_uncertainty"], intercept_uncertainty
                )
            else:
                clusters.append(
                    {
                        "normal": normal,
                        "intercept": intercept,
                        "count": 1,
                        "quality": quality,
                        "uncertainty": uncertainty,
                        "intercept_uncertainty": intercept_uncertainty,
                    }
                )
            accepted += 1
        selected = sorted(
            clusters, key=lambda r: (r["count"], r["quality"]), reverse=True
        )
        check_mse = None
        if len(selected) >= hidden:
            probe_normal = np.stack(
                [row["normal"] for row in selected[:hidden]], axis=1
            )
            probe_bias = np.array([row["intercept"] for row in selected[:hidden]])
            design = np.c_[
                np.maximum(check_x @ probe_normal + probe_bias, 0),
                check_x,
                np.ones(len(check_x)),
            ]
            fitted = np.linalg.lstsq(design[:512], check_y[:512], rcond=1e-12)[0]
            check_mse = float(np.mean((design[512:] @ fitted - check_y[512:]) ** 2))
        trace.append(
            {
                "ray": ray,
                "detected_kinks": len(locations),
                "accepted_planes": accepted,
                "clusters": len(clusters),
                "repeated_clusters": sum(r["count"] >= 2 for r in clusters),
                "observed_check_mse": check_mse,
                "seconds": time.monotonic() - started,
            }
        )
        if (
            ray >= 7
            and len(selected) >= hidden
            and all(row["count"] >= 3 for row in selected[:hidden])
            and check_mse <= max(100 * sem**2, signal * 1e-5 * precision_factor**2)
        ):
            break
    selected = sorted(clusters, key=lambda r: (r["count"], r["quality"]), reverse=True)
    selected = selected[:hidden]
    return selected, {
        "trace": trace,
        "ray_repeats": repeats,
        "seconds": time.monotonic() - started,
    }


def orient_planes(planes, x, y):
    normal = np.stack([row["normal"] for row in planes], axis=1)
    intercept = np.array([row["intercept"] for row in planes])
    hidden = normal.shape[1]
    hinge = np.maximum(x @ normal + intercept, 0)
    design = np.c_[hinge, x, np.ones(len(x))]
    coefficient = np.linalg.lstsq(design, y, rcond=1e-12)[0]
    head, linear = coefficient[:hidden], coefficient[hidden:-1]
    mapping = np.einsum("dh,ho->doh", normal, head).reshape(-1, hidden)
    target = -linear.ravel()
    gram, cross = mapping.T @ mapping, mapping.T @ target
    best = (math.inf, None)
    if hidden > 20:
        raise ValueError("Registered binary orientation budget supports H <= 20")
    for start in range(0, 2**hidden, 4096):
        number = np.arange(start, min(start + 4096, 2**hidden), dtype=np.uint32)
        bits = ((number[:, None] >> np.arange(hidden, dtype=np.uint32)) & 1).astype(
            float
        )
        error = np.sum((bits @ gram) * bits, axis=1) - 2 * bits @ cross
        chosen = int(error.argmin())
        if error[chosen] < best[0]:
            best = float(error[chosen]), bits[chosen]
    signs = 1 - 2 * best[1]
    normal, intercept = normal * signs, intercept * signs
    design = np.c_[np.maximum(x @ normal + intercept, 0), np.ones(len(x))]
    coefficient = np.linalg.lstsq(design, y, rcond=1e-12)[0]
    return (
        normal,
        intercept,
        coefficient,
        {
            "orientation_residual": float(np.mean((mapping @ best[1] - target) ** 2)),
            "training_mse": float(np.mean((design @ coefficient - y) ** 2)),
            "negative_orientations": int(best[1].sum()),
        },
    )


def ordered_centers(values, count):
    ordered = np.sort(values)
    size = len(ordered)
    sums, squares = np.r_[0, np.cumsum(ordered)], np.r_[0, np.cumsum(ordered**2)]
    cost = np.full((count + 1, size + 1), np.inf)
    back = np.zeros((count + 1, size + 1), dtype=int)
    cost[0, 0] = 0
    for groups in range(1, count + 1):
        for end in range(groups, size + 1):
            for begin in range(groups - 1, end):
                error = (
                    squares[end]
                    - squares[begin]
                    - (sums[end] - sums[begin]) ** 2 / (end - begin)
                )
                candidate = cost[groups - 1, begin] + error
                if candidate < cost[groups, end]:
                    cost[groups, end], back[groups, end] = candidate, begin
    centers, end = [], size
    for groups in range(count, 0, -1):
        begin = back[groups, end]
        centers.append(float(np.mean(ordered[begin:end])))
        end = begin
    return np.array(centers[::-1])


def scale_quantizations(normal, uncertainty, codes, seed, starts=12, zero_multiplier=5):
    rng = np.random.default_rng(seed)
    dimensions, hidden = normal.shape
    cutoff = np.maximum(zero_multiplier * uncertainty[None], 0.002)
    active = np.abs(normal) > cutoff
    rows, columns = np.where(active)
    logs = np.log(np.abs(normal[rows, columns]))
    if len(logs) < codes:
        return []
    candidates, initial = [], [np.quantile(logs, np.linspace(0, 1, codes))]
    for column in range(hidden):
        values = np.sort(np.abs(normal[active[:, column], column]))
        if len(values) >= codes:
            initial.append(ordered_centers(np.log(values), codes))
        tolerance = max(8 * uncertainty[column], 0.002)
        boundaries = np.flatnonzero(np.diff(values) > tolerance) + 1
        groups = np.split(values, boundaries)
        if len(groups) == codes:
            initial.append(np.log([np.mean(group) for group in groups]))
    while len(initial) < starts:
        initial.append(np.sort(rng.choice(logs, codes, replace=False)))
    for center in initial:
        scales = np.zeros(hidden)
        for column in range(hidden):
            local = logs[columns == column]
            if not len(local):
                continue
            possible = (center[:, None] - local).ravel()
            error = np.min(
                (local[:, None, None] + possible[None, :, None] - center) ** 2,
                axis=2,
            ).mean(axis=0)
            scales[column] = possible[error.argmin()]
        previous = None
        for _ in range(80):
            labels = np.abs(logs[:, None] + scales[columns, None] - center).argmin(1)
            if previous is not None and np.array_equal(labels, previous):
                break
            previous = labels.copy()
            matrix = np.zeros((len(rows) + 1, codes + hidden))
            matrix[np.arange(len(rows)), labels] = 1
            matrix[np.arange(len(rows)), codes + columns] = -1
            matrix[-1, codes] = 1
            solution = np.linalg.lstsq(matrix, np.r_[logs, 0], rcond=1e-12)[0]
            center, scales = solution[:codes], solution[codes:]
        index = np.zeros((dimensions, hidden), dtype=np.int16)
        index[rows, columns] = np.sign(normal[rows, columns]).astype(np.int16) * (
            labels + 1
        )
        candidates.append(
            {
                "index": index,
                "codes": np.exp(center),
                "scales": np.exp(scales),
                "projection_mse": float(
                    np.mean((center[labels] - scales[columns] - logs) ** 2)
                ),
            }
        )
    unique = {}
    for row in candidates:
        key = row["index"].tobytes()
        if key not in unique or row["projection_mse"] < unique[key]["projection_mse"]:
            unique[key] = row
    return sorted(unique.values(), key=lambda row: row["projection_mse"])
