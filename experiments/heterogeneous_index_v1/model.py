"""Signed identity-descriptor instance of Algorithm 1 and exact small alignment.

This diagnostic intentionally removes neuron calibration, support estimation,
and nonlinear optimization. The numerical block is a Cauchy IRLS projection;
the categorical block is the usual common signed lookup assignment.
"""

import itertools
import time

import numpy as np


def fit_dense(x, y):
    """Fit an actual dense layer; estimate uncertainty from its residuals."""
    weight = np.linalg.lstsq(x, y, rcond=1e-10)[0]
    residual = y - x @ weight
    variance = (residual**2).sum(0) / max(len(x) - x.shape[1], 1)
    floor = 0.03 * np.sqrt(np.mean(y**2, axis=0))
    sigma = np.maximum(np.sqrt(variance), floor)
    diagonal = np.diag(np.linalg.pinv(x.T @ x)).clip(1e-12)
    uncertainty = np.sqrt(diagonal)[:, None] * sigma[None]
    uncertainty = np.maximum(uncertainty, 0.015 * np.sqrt(np.mean(weight**2)))
    return weight, uncertainty


def scalar_seed(weight, codes):
    """Deterministic signed Lloyd seed, without any generating parameters."""
    magnitude = np.abs(weight).ravel()
    centers = np.quantile(magnitude, (np.arange(codes) + 0.5) / codes)
    labels = np.zeros(len(magnitude), dtype=int)
    for _ in range(20):
        labels = np.abs(magnitude[:, None] - centers).argmin(1)
        for group in range(codes):
            active = labels == group
            if active.any():
                centers[group] = magnitude[active].mean()
        distance = np.abs(magnitude - centers[labels])
        for group in range(codes):
            if np.any(labels == group):
                continue
            counts = np.bincount(labels, minlength=codes)
            position = np.where(counts[labels] > 1, distance, -1).argmax()
            labels[position] = group
            centers[group] = magnitude[position]
    return ((labels.reshape(weight.shape) + 1) * np.where(weight >= 0, 1, -1)).astype(
        np.int16
    )


def code_fit(index, weights, uncertainty, robust=True):
    """Fit one independent code vector per task at the fixed indexator."""
    codes = int(np.abs(index).max())
    labels = np.abs(index).ravel() - 1
    signs = np.sign(index).ravel()
    data = weights.reshape(len(weights), -1)
    deviation = uncertainty.reshape(data.shape)
    precision = deviation**-2
    values = np.zeros((len(data), codes))
    for iteration in range(4 if robust else 1):
        if iteration:
            residual = (data - signs * values[:, labels]) / deviation
            local = precision / (1 + residual**2 / 4)
        else:
            local = precision
        for group in range(codes):
            active = labels == group
            values[:, group] = (local[:, active] * data[:, active] * signs[active]).sum(
                1
            ) / local[:, active].sum(1).clip(1e-30)
    return values


def project(initial, weights, uncertainty, updates):
    """Apply Algorithm 1 with the identity descriptor and a signed codebook."""
    index = initial.copy()
    codes = int(np.abs(index).max())
    symbols = np.r_[np.arange(1, codes + 1), -np.arange(1, codes + 1)]
    for _ in range(updates):
        values = code_fit(index, weights, uncertainty)
        centers = np.c_[values, -values]
        residual = (weights[..., None] - centers[:, None, None]) / uncertainty[
            ..., None
        ]
        cost = np.mean(4 * np.log1p(residual**2 / 4), axis=0)
        updated = symbols[cost.argmin(-1)].astype(np.int16)
        for group in range(1, codes + 1):
            if np.any(np.abs(updated) == group):
                continue
            counts = np.bincount(np.abs(updated).ravel(), minlength=codes + 1)
            allowed = counts[np.abs(updated)] > 1
            group_cost = np.minimum(cost[..., group - 1], cost[..., group - 1 + codes])
            excess = group_cost - cost.min(-1)
            position = np.where(allowed, excess, np.inf).argmin()
            updated.ravel()[position] = (
                group
                if cost.reshape(-1, 2 * codes)[position, group - 1]
                <= cost.reshape(-1, 2 * codes)[position, group - 1 + codes]
                else -group
            )
        if np.array_equal(updated, index):
            break
        index = updated
    return index


def align(index, weight, uncertainty):
    """Enumerate all 24 column permutations, profiling out code values."""
    permutations = np.array(list(itertools.permutations(range(index.shape[1]))))
    proposals = np.stack([index[:, order] for order in permutations])
    labels = np.abs(proposals).reshape(len(proposals), -1) - 1
    signs = np.sign(proposals).reshape(len(proposals), -1)
    data = weight.ravel()
    precision = uncertainty.ravel() ** -2
    values = np.zeros((len(proposals), int(np.abs(index).max())))
    for group in range(values.shape[1]):
        active = labels == group
        values[:, group] = (active * precision * data * signs).sum(1) / (
            active * precision
        ).sum(1).clip(1e-30)
    prediction = signs * np.take_along_axis(values, labels, axis=1)
    costs = ((prediction - data) ** 2 * precision).sum(1)
    selected = int(costs.argmin())
    return proposals[selected], permutations[selected]


def projection_score(index, weights, uncertainty):
    values = code_fit(index, weights, uncertainty)
    labels = np.abs(index) - 1
    predicted = np.sign(index)[None] * values[:, labels]
    return float(np.mean(4 * np.log1p(((predicted - weights) / uncertainty) ** 2 / 4)))


def common_projection(initial, weights, uncertainty):
    """Alternate estimated column alignment with the multi-task projection."""
    index = initial.copy()
    for _ in range(4):
        aligned, deviations = [], []
        for weight, error in zip(weights, uncertainty):
            _, order = align(index, weight, error)
            inverse = np.argsort(order)
            aligned.append(weight[:, inverse])
            deviations.append(error[:, inverse])
        aligned, deviations = np.asarray(aligned), np.asarray(deviations)
        updated = project(index, aligned, deviations, 8)
        if np.array_equal(updated, index):
            break
        index = updated
    return index, projection_score(index, aligned, deviations)


def fit_index(index, x, y, validation_x, validation_y, ridges):
    """Fresh functional code fitting and validation, without hypergradients."""
    codes = int(np.abs(index).max())
    basis = np.eye(codes)[np.abs(index) - 1] * np.sign(index)[..., None]
    design = np.einsum("nd,dhk->nhk", x, basis).reshape(-1, codes)
    left, singular, right = np.linalg.svd(design, full_matrices=False)
    cross = left.T @ y.ravel()
    scale = max(float(np.mean(singular**2)), 1e-15)
    options = []
    for ridge in ridges:
        denominator = singular**2 + ridge * scale
        inverse = np.divide(
            singular,
            denominator,
            out=np.zeros_like(singular),
            where=denominator > max(float(singular.max(initial=0)) ** 2 * 1e-20, 1e-30),
        )
        code = right.T @ (inverse * cross)
        weight = np.einsum("dhk,k->dh", basis, code)
        loss = float(np.mean((validation_x @ weight - validation_y) ** 2))
        options.append((loss, ridge, weight))
    return min(options, key=lambda item: (item[0], item[1]))


def target_fit(bank, family, mode, x, y, validation_x, validation_y, ridges, depths):
    """Adapt from shared structures, or independently from this task alone."""
    started = time.perf_counter()
    dense, deviation = fit_dense(x, y)
    identity = np.arange(1, dense.size + 1).reshape(dense.shape).astype(np.int16)
    if family == "dense":
        candidates = [(identity, {"proposal": "unshared", "depth": 0})]
    elif family == "single":
        candidates = []
        for codes in (2, 4, 8):
            seed = scalar_seed(dense, codes)
            for depth in depths:
                candidates.append(
                    (
                        project(seed, dense[None], deviation[None], depth),
                        {"proposal": "task_only", "depth": depth},
                    )
                )
        candidates.append((identity, {"proposal": "unshared", "depth": 0}))
    else:
        candidates = []
        for number, index in enumerate(bank[family]):
            if int(np.abs(index).max()) == index.size:
                candidates.append((identity, {"proposal": number, "depth": 0}))
                continue
            initial = align(index, dense, deviation)[0] if mode != "fixed" else index
            for depth in depths if mode == "adapt" else (0,):
                adapted = project(initial, dense[None], deviation[None], depth)
                candidates.append((adapted, {"proposal": number, "depth": depth}))
    best, best_key, seen = None, None, set()
    for index, metadata in candidates:
        key = index.tobytes()
        if key in seen:
            continue
        seen.add(key)
        loss, ridge, weight = fit_index(index, x, y, validation_x, validation_y, ridges)
        actual_codes = int(np.abs(index).max())
        criterion = (loss, actual_codes, metadata["depth"], ridge)
        if best_key is None or criterion < best_key:
            best_key = criterion
            best = {
                "weight": weight,
                "index": index,
                "validation_mse": loss,
                "ridge": ridge,
                "codes": actual_codes,
                "candidates": len(candidates),
                **metadata,
            }
    best["seconds"] = time.perf_counter() - started
    return best
