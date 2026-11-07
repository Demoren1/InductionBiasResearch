"""Prefix-support diagnostics and conservative candidate-pool alternatives."""

import math

import numpy as np

from .weighted_prefix_search import support_path


def paired_choice(records, matrix, y, variance):
    best = min(records, key=lambda row: row["validation_mse"])
    prediction = matrix[:, best["active"]] @ best["coefficient"]
    best_loss = ((prediction - y) ** 2).mean(axis=1)
    accepted = []
    floor = 1e-12 * max(float(np.var(y)), 1)
    for row in records:
        fitted = matrix[:, row["active"]] @ row["coefficient"]
        difference = ((fitted - y) ** 2).mean(axis=1) - best_loss
        standard_error = float(np.std(difference, ddof=1) / math.sqrt(len(y)))
        if difference.mean() <= 2 * standard_error + floor:
            accepted.append(row)
    selected = min(accepted, key=lambda row: len(row["active"]))
    selected["paired_floor"] = floor
    selected["noise_variance"] = variance
    return selected


def candidate_support(matrix, y, vmatrix, vy, hidden, variance, mode, final):
    if mode == "old":
        result, trace = support_path(matrix, y, vmatrix, vy, hidden, variance)
        return None if result is None else result["active"], trace
    count = len(matrix)
    gram = matrix.T @ matrix / count
    cross = matrix.T @ y / count
    eigenvalues, vectors = np.linalg.eigh(gram)
    rank = int(np.sum(eigenvalues > max(eigenvalues[-1], 1e-30) * 1e-12))
    pooled = mode.startswith("keep")
    if not pooled and rank != len(gram):
        return None, {
            "complete": False,
            "reason": "rank_deficient_dictionary",
            "rank": rank,
            "columns": len(gram),
        }
    penalty = max(variance / count, eigenvalues[-1] * 1e-14) if pooled else 0.0
    spectrum = np.maximum(eigenvalues, 0) + penalty
    inverse = (vectors / spectrum) @ vectors.T
    coefficient = inverse @ cross
    active = np.arange(len(gram))
    multiplier = int(mode.removeprefix("keep")) if pooled else 1
    capacity = hidden if final else hidden * multiplier
    records = []
    while len(active):
        if len(active) <= capacity:
            if pooled:
                fitted = np.linalg.solve(
                    gram[np.ix_(active, active)] + penalty * np.eye(len(active)),
                    cross[active],
                )
            else:
                fitted = np.linalg.lstsq(matrix[:, active], y, rcond=1e-12)[0]
            records.append(
                {
                    "active": active.copy(),
                    "coefficient": fitted,
                    "training_mse": float(
                        np.mean((matrix[:, active] @ fitted - y) ** 2)
                    ),
                    "validation_mse": float(
                        np.mean((vmatrix[:, active] @ fitted - vy) ** 2)
                    ),
                }
            )
            if pooled:
                break
        if len(active) == 1:
            break
        increase = np.sum(coefficient**2, axis=1) / np.maximum(
            inverse.diagonal(), 1e-30
        )
        removed = int(increase.argmin())
        keep = np.arange(len(active)) != removed
        column, diagonal = inverse[keep, removed], inverse[removed, removed]
        coefficient = (
            coefficient[keep] - column[:, None] * coefficient[removed] / diagonal
        )
        inverse = inverse[np.ix_(keep, keep)] - np.outer(column, column) / diagonal
        active = active[keep]
    if not pooled:
        records.append(
            {
                "active": np.empty(0, dtype=int),
                "coefficient": np.empty((0, y.shape[1])),
                "training_mse": float(np.mean(y**2)),
                "validation_mse": float(np.mean(vy**2)),
            }
        )
    selected = (
        paired_choice(records, vmatrix, vy, variance)
        if mode == "paired"
        else min(records, key=lambda row: row["validation_mse"])
    )
    return selected["active"], {
        "complete": True,
        "rank": rank,
        "columns": len(gram),
        "penalty": penalty,
        "selected_atoms": len(selected["active"]),
        "training_mse": selected["training_mse"],
        "validation_mse": selected["validation_mse"],
        "condition": float(eigenvalues[-1] / max(eigenvalues[0], 1e-300)),
        "path": [
            {
                "atoms": len(row["active"]),
                "train_mse": row["training_mse"],
                "validation_mse": row["validation_mse"],
            }
            for row in records
        ],
    }


def extract(data, code, spec, task, phase, sigma, mode):
    order, hidden = data["order"], spec["hidden"]
    choices = np.r_[
        0, np.arange(1, spec["codes"] + 1), -np.arange(1, spec["codes"] + 1)
    ]
    codebook = np.r_[0, code]
    parents = np.empty((1, 0), dtype=np.int16)
    trace, prefixes_by_step = [], []
    index = np.zeros((spec["inputs"], hidden), dtype=np.int16)
    for step in range(1, spec["inputs"] + 1):
        candidates = np.c_[
            np.repeat(parents, len(choices), axis=0), np.tile(choices, len(parents))
        ].astype(np.int16)
        candidates = np.unique(candidates, axis=0)
        candidates = candidates[np.any(candidates != 0, axis=1)]
        weight = np.sign(candidates) * codebook[np.abs(candidates)]
        block = slice((step - 1) * spec["slice_points"], step * spec["slice_points"])
        vblock = slice(
            (step - 1) * spec["prefix_validation"], step * spec["prefix_validation"]
        )
        x = data[f"prefix_x_{phase}"][block][:, order[:step]]
        vx = data[f"prefix_vx_{phase}"][vblock][:, order[:step]]
        matrix = (
            np.maximum(x @ weight.T + spec["bias"], 0) - max(spec["bias"], 0)
        ) / math.sqrt(hidden)
        vmatrix = (
            np.maximum(vx @ weight.T + spec["bias"], 0) - max(spec["bias"], 0)
        ) / math.sqrt(hidden)
        selected, record = candidate_support(
            matrix,
            data[f"prefix_y_{phase}"][task, block],
            vmatrix,
            data[f"prefix_vy_{phase}"][task, vblock],
            hidden,
            sigma**2 / spec["slice_repeats"],
            mode,
            step == spec["inputs"],
        )
        trace.append({"step": step, **record})
        if selected is None:
            break
        prefixes = candidates[selected]
        prefixes_by_step.append(prefixes.copy())
        # The intermediate pool may exceed H. Only full supports define a network.
        if step == spec["inputs"]:
            index[np.ix_(order, np.arange(len(prefixes)))] = prefixes.T
        parents = np.unique(
            np.r_[prefixes, np.zeros((1, step), dtype=np.int16)], axis=0
        )
    return (
        index,
        {
            "complete": len(trace) == spec["inputs"] and trace[-1]["complete"],
            "trace": trace,
            "mode": mode,
        },
        prefixes_by_step,
    )
