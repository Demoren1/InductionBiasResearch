"""Unknown-readout prefix extraction with function-based support selection."""

import itertools
import math
import time

import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from .joint_readout_control import feature


def axis_dictionary(s, code, bias):
    values = np.r_[code, -code]
    return np.maximum(bias + s[:, None] * values, 0) - max(bias, 0)


def estimate_codes(s, values, vs, validation, spec, seed):
    started = time.monotonic()
    count = spec["codes"]
    y = values.reshape(len(s), -1)
    vy = validation.reshape(len(vs), -1)
    positive = s > 0
    slopes = np.diff(y[positive], axis=0) / np.diff(s[positive])[:, None]
    energy = np.linalg.norm(np.diff(slopes, axis=0), axis=1)
    locations = s[positive][1:-1]
    peaks, _ = find_peaks(energy)
    peaks = peaks[np.argsort(energy[peaks])[::-1]]
    initial = []
    for peak in peaks:
        candidate = abs(spec["bias"]) / locations[peak]
        if 0.03 < candidate < 8 and all(
            abs(math.log(candidate / old)) > 0.08 for old in initial
        ):
            initial.append(candidate)
        if len(initial) == count:
            break
    rng = np.random.default_rng(seed)
    starts = [np.geomspace(0.2, 2, count), np.geomspace(0.4, 1.5, count)]
    if len(initial) == count:
        starts.insert(0, np.sort(initial))
    while len(starts) < 7:
        starts.append(np.exp(rng.uniform(np.log(0.15), np.log(2.5), count)))

    def residual(logcode):
        matrix = axis_dictionary(s, np.sort(np.exp(logcode)), spec["bias"])
        coefficient = np.linalg.lstsq(matrix, y, rcond=1e-12)[0]
        return (matrix @ coefficient - y).ravel()

    records = []
    for start in starts:
        fit = least_squares(
            residual,
            np.log(start),
            bounds=(np.log(0.03), np.log(8)),
            max_nfev=160,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        code = np.sort(np.exp(fit.x))
        matrix = axis_dictionary(s, code, spec["bias"])
        coefficient = np.linalg.lstsq(matrix, y, rcond=1e-12)[0]
        score = float(
            ((axis_dictionary(vs, code, spec["bias"]) @ coefficient - vy) ** 2).mean()
        )
        records.append(
            {
                "code": code.tolist(),
                "validation_mse": score,
                "training_mse": float(np.mean(fit.fun**2)),
                "evaluations": fit.nfev,
                "success": bool(fit.success),
            }
        )
    winner = min(records, key=lambda row: row["validation_mse"])
    return np.array(winner["code"]), {
        "starts": records,
        "selected": winner,
        "seconds": time.monotonic() - started,
    }


def support_path(matrix, y, vmatrix, vy, hidden, noise_variance):
    count, outputs = y.shape
    gram = matrix.T @ matrix / count
    cross = matrix.T @ y / count
    vgram = vmatrix.T @ vmatrix / len(vy)
    vcross = vmatrix.T @ vy / len(vy)
    eigenvalues, vectors = np.linalg.eigh(gram)
    cutoff = max(eigenvalues[-1], 1e-30) * 1e-12
    rank = int((eigenvalues > cutoff).sum())
    if rank < len(gram):
        return None, {
            "complete": False,
            "reason": "rank_deficient_dictionary",
            "rank": rank,
            "columns": len(gram),
            "minimum_eigenvalue": float(eigenvalues[0]),
        }
    inverse = (vectors / eigenvalues) @ vectors.T
    coefficient = inverse @ cross
    active = np.arange(len(gram))
    y2, vy2 = float(np.mean(y**2)), float(np.mean(vy**2))
    records = []
    while len(active):
        if len(active) <= hidden:
            g, c = gram[np.ix_(active, active)], cross[active]
            vg, vc = vgram[np.ix_(active, active)], vcross[active]
            train = (
                y2 + float(np.sum(coefficient * (g @ coefficient - 2 * c))) / outputs
            )
            validation = (
                vy2 + float(np.sum(coefficient * (vg @ coefficient - 2 * vc))) / outputs
            )
            # Direct residuals avoid cancellation in the numerical selection floor.
            train = float(np.mean((matrix[:, active] @ coefficient - y) ** 2))
            validation = float(np.mean((vmatrix[:, active] @ coefficient - vy) ** 2))
            records.append(
                {
                    "active": active.copy(),
                    "coefficient": coefficient.copy(),
                    "training_mse": train,
                    "validation_mse": validation,
                }
            )
        if len(active) == 1:
            break
        increase = np.sum(coefficient**2, axis=1) / np.maximum(
            inverse.diagonal(), 1e-30
        )
        removed = int(increase.argmin())
        keep = np.arange(len(active)) != removed
        column = inverse[keep, removed]
        diagonal = inverse[removed, removed]
        coefficient = (
            coefficient[keep] - column[:, None] * coefficient[removed] / diagonal
        )
        inverse = inverse[np.ix_(keep, keep)] - np.outer(column, column) / diagonal
        active = active[keep]
    records.append(
        {
            "active": np.empty(0, dtype=int),
            "coefficient": np.empty((0, outputs)),
            "training_mse": y2,
            "validation_mse": vy2,
        }
    )
    best = min(row["validation_mse"] for row in records)
    tolerance = 2 * noise_variance * math.sqrt(2 / (len(vy) * outputs)) + 1e-12 * max(
        float(np.var(vy)), 1
    )
    accepted = [row for row in records if row["validation_mse"] <= best + tolerance]
    selected = min(accepted, key=lambda row: len(row["active"]))
    # Refit the selected hard support rather than retain accumulated inverse error.
    active = selected["active"]
    coefficient = np.linalg.lstsq(matrix[:, active], y, rcond=1e-12)[0]
    result = {
        "active": active,
        "coefficient": coefficient,
        "training_mse": float(np.mean((matrix[:, active] @ coefficient - y) ** 2)),
        "validation_mse": float(np.mean((vmatrix[:, active] @ coefficient - vy) ** 2)),
    }
    trace = {
        "complete": True,
        "rank": rank,
        "columns": len(gram),
        "condition": float(eigenvalues[-1] / eigenvalues[0]),
        "selected_atoms": len(active),
        "selection_tolerance": tolerance,
        "training_mse": result["training_mse"],
        "validation_mse": result["validation_mse"],
        "path": [
            {
                "atoms": len(row["active"]),
                "train_mse": row["training_mse"],
                "validation_mse": row["validation_mse"],
            }
            for row in records
        ],
    }
    return result, trace


def extract(data, code, spec, task, phase, sigma):
    started = time.monotonic()
    order = data["order"]
    dimensions, hidden = spec["inputs"], spec["hidden"]
    choices = np.r_[
        0, np.arange(1, spec["codes"] + 1), -np.arange(1, spec["codes"] + 1)
    ]
    codebook = np.r_[0, code]
    parents = np.empty((1, 0), dtype=np.int16)
    trace = []
    index = np.zeros((dimensions, hidden), dtype=np.int16)
    for step in range(1, dimensions + 1):
        candidates = np.c_[
            np.repeat(parents, len(choices), axis=0), np.tile(choices, len(parents))
        ].astype(np.int16)
        candidates = np.unique(candidates, axis=0)
        nonzero = np.any(candidates != 0, axis=1)
        candidates = candidates[nonzero]
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
        y = data[f"prefix_y_{phase}"][task, block]
        vy = data[f"prefix_vy_{phase}"][task, vblock]
        result, record = support_path(
            matrix, y, vmatrix, vy, hidden, sigma**2 / spec["slice_repeats"]
        )
        trace.append({"step": step, **record})
        if result is None:
            break
        prefixes = candidates[result["active"]]
        index = np.zeros((dimensions, hidden), dtype=np.int16)
        index[np.ix_(order[:step], np.arange(len(prefixes)))] = prefixes.T
        # Zero prefixes remain available for neurons inactive on earlier coordinates.
        parents = np.unique(
            np.r_[prefixes, np.zeros((1, step), dtype=np.int16)], axis=0
        )
    return index, {
        "complete": len(trace) == dimensions and trace[-1]["complete"],
        "trace": trace,
        "codebook": code.tolist(),
        "seconds": time.monotonic() - started,
    }


def values(feat, parameters, spec, outputs, jacobian=False):
    codes, hidden = spec["codes"], spec["hidden"]
    activation = torch.einsum("nhk,bk->bnh", feat, parameters[:, :codes]) + spec["bias"]
    centered = activation.relu() - max(spec["bias"], 0)
    head = parameters[:, codes:].reshape(-1, hidden, outputs) / math.sqrt(hidden)
    prediction = torch.einsum("bnh,bho->bno", centered, head)
    if not jacobian:
        return prediction
    cj = torch.einsum("nhk,bnh,bho->bnok", feat, (activation > 0).to(feat.dtype), head)
    hj = torch.einsum(
        "bnh,oq->bnohq",
        centered / math.sqrt(hidden),
        torch.eye(outputs, device=feat.device, dtype=feat.dtype),
    ).flatten(-2)
    return prediction, torch.cat([cj, hj], -1).flatten(1, 2)


@torch.no_grad()
def fit_numeric(index, x, y, vx, vy, spec, seed, device):
    started = time.monotonic()
    tasks, _, outputs = y.shape
    starts, codes = spec["numeric_starts"], spec["codes"]
    rng = np.random.default_rng(seed)
    signs = np.tile(
        np.array(list(itertools.product((-1, 1), repeat=codes))),
        (starts // 2**codes, 1),
    )
    initial_c = signs[None] * np.exp(rng.normal(-0.3, 0.7, (tasks, starts, codes)))
    head = rng.normal(0, 1, (tasks, starts, spec["hidden"] * outputs))
    head[:, : starts // 2] = 1
    parameters = torch.tensor(
        np.concatenate([initial_c, head], -1).reshape(tasks * starts, -1),
        device=device,
        dtype=torch.float64,
    ).clamp(-16, 16)
    feat = feature(x, index, codes, device, torch.float64)
    vf = feature(vx, index, codes, device, torch.float64)
    target = torch.tensor(
        np.repeat(y, starts, axis=0), device=device, dtype=torch.float64
    )
    validation = torch.tensor(
        np.repeat(vy, starts, axis=0), device=device, dtype=torch.float64
    )
    best = parameters.new_full((len(parameters),), torch.inf)
    best_parameters, best_step = (
        parameters.clone(),
        torch.zeros(len(parameters), dtype=torch.long, device=device),
    )
    identity = torch.eye(parameters.shape[1], device=device, dtype=parameters.dtype)
    history = []
    train_history, validation_history = [], []
    for step in range(spec["numeric_steps"] + 1):
        prediction, jac = values(feat, parameters, spec, outputs, jacobian=True)
        residual = (target - prediction).flatten(1)
        train = residual.square().mean(1)
        if step % 4 == 0 or step == spec["numeric_steps"]:
            score = (
                (values(vf, parameters, spec, outputs) - validation)
                .square()
                .mean((1, 2))
            )
            better = score < best
            best[better], best_parameters[better], best_step[better] = (
                score[better],
                parameters[better],
                step,
            )
            history.append(step)
            train_history.append(train.cpu().numpy())
            validation_history.append(score.cpu().numpy())
        if step == spec["numeric_steps"]:
            break
        gram = jac.transpose(1, 2) @ jac / residual.shape[1]
        cross = jac.transpose(1, 2) @ residual[..., None] / residual.shape[1]
        scale = gram.diagonal(dim1=1, dim2=2).amax(1).clamp_min(1e-12)
        direction = torch.linalg.solve(
            gram + 1e-6 * scale[:, None, None] * identity, cross
        )[..., 0]
        accepted = train < 1e-24
        new = parameters.clone()
        for exponent in range(12):
            proposal = parameters + 2.0 ** (-exponent) * direction
            loss = (
                (values(feat, proposal, spec, outputs) - target).square().mean((1, 2))
            )
            take = (~accepted) & (proposal.abs().amax(1) <= 16) & (loss < train)
            new[take] = proposal[take]
            accepted |= take
            if accepted.all():
                break
        parameters = new
    winner = best.reshape(tasks, starts).argmin(1)
    ids = torch.arange(tasks, device=device) * starts + winner
    return (
        best_parameters[ids].cpu().numpy(),
        {
            "validation_mse": best[ids].cpu().tolist(),
            "selected_start": winner.cpu().tolist(),
            "best_step": best_step[ids].cpu().tolist(),
            "seconds": time.monotonic() - started,
        },
        {
            "steps": np.array(history),
            "train": np.stack(train_history),
            "validation": np.stack(validation_history),
        },
    )
