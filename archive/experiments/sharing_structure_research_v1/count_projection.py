"""G21 count constraints and task metric; QP and local moves are known tools."""

import math
import time

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, minimize
from torch.nn import functional as F

from .prefix_census import numeric_prefix


def integer_counts(estimate, totals):
    result = np.zeros(estimate.shape, dtype=np.int16)
    for row, total in enumerate(totals):
        for _ in range(int(total)):
            column = int(np.argmin(2 * (result[row] - estimate[row]) + 1))
            result[row, column] += 1
    return result


def cap_projection(value, totals):
    rows = value.reshape(len(totals), -1).copy()
    for row, total in enumerate(totals):
        positive = np.maximum(rows[row], 0)
        if positive.sum() <= total:
            rows[row] = positive
            continue
        ordered = np.sort(rows[row])[::-1]
        shifts = (np.cumsum(ordered) - total) / np.arange(1, len(ordered) + 1)
        active = np.flatnonzero(ordered > shifts)[-1]
        rows[row] = np.maximum(rows[row] - shifts[active], 0)
    return rows.flatten()


def qp_gap(value, gram, cross, totals):
    gradient = gram @ value - cross
    minimum = np.minimum(0, gradient.reshape(len(totals), -1).min(1))
    return float(gradient @ value - totals @ minimum), gradient


def function_qp(gram, cross, initial, totals, plan):
    started = time.monotonic()
    width = len(initial) // len(totals)
    constraints = np.kron(np.eye(len(totals)), np.ones((1, width)))
    scale = max(float(np.linalg.norm(gram, ord=np.inf)), 1e-30)

    def objective(value):
        return float(value @ (gram @ value) / 2 - cross @ value) / scale

    def gradient(value):
        return (gram @ value - cross) / scale

    result = minimize(
        objective,
        initial.astype(float),
        jac=gradient,
        method="SLSQP",
        bounds=Bounds(0, np.repeat(totals, width)),
        constraints=LinearConstraint(constraints, -np.inf, totals),
        options={"maxiter": plan["qp_iterations"], "ftol": plan["qp_ftol"]},
    )
    value = cap_projection(result.x, totals)
    polishing = 0
    for _ in range(plan["qp_polish_rounds"]):
        gap, derivative = qp_gap(value, gram, cross, totals)
        if gap <= plan["qp_gap_absolute"]:
            break
        polishing += 1
        free = value > 1e-9
        binding = totals - constraints @ value < 1e-8
        if free.any():
            active = constraints[binding][:, free]
            if active.size:
                kkt = np.block(
                    [
                        [gram[np.ix_(free, free)], active.T],
                        [active, np.zeros((len(active), len(active)))],
                    ]
                )
                right = np.r_[cross[free], totals[binding]]
            else:
                kkt, right = gram[np.ix_(free, free)], cross[free]
            solution = np.linalg.solve(kkt, right)[: int(free.sum())]
            candidate = np.zeros_like(value)
            candidate[free] = solution
            direction = candidate - value
            fraction = 1.0
            negative = direction < 0
            if negative.any():
                fraction = min(
                    fraction, float(np.min(-value[negative] / direction[negative]))
                )
            increase = constraints @ direction
            violated = increase > 1e-14
            if violated.any():
                fraction = min(
                    fraction,
                    float(
                        np.min(
                            (totals - constraints @ value)[violated]
                            / increase[violated]
                        )
                    ),
                )
            candidate = cap_projection(value + max(0.0, fraction) * direction, totals)
            if objective(candidate) <= objective(value) + 1e-14:
                value = candidate
        gap, derivative = qp_gap(value, gram, cross, totals)
        if gap <= plan["qp_gap_absolute"]:
            break
        vertex = np.zeros((len(totals), width))
        grouped = derivative.reshape(len(totals), width)
        for row, total in enumerate(totals):
            column = int(grouped[row].argmin())
            if grouped[row, column] < 0:
                vertex[row, column] = total
        direction = vertex.flatten() - value
        curvature = float(direction @ gram @ direction)
        fraction = min(1.0, max(0.0, gap / max(curvature, 1e-30)))
        value = cap_projection(value + fraction * direction, totals)
    gap, _ = qp_gap(value, gram, cross, totals)
    violation = max(
        float(np.maximum(-value, 0).max()),
        float(np.maximum(constraints @ value - totals, 0).max()),
    )
    return value, {
        "success_flag": bool(result.success),
        "message": str(result.message),
        "iterations": result.nit,
        "face_polish_rounds": polishing,
        "fw_gap": gap,
        "primal_violation": violation,
        "converged": abs(gap) <= plan["qp_gap_absolute"]
        and violation <= plan["qp_primal_tolerance"],
        "seconds": time.monotonic() - started,
    }


def integer_moves(counts, gram, cross, plan):
    groups, choices = counts.shape
    width = groups * choices
    keep = np.arange(width).reshape(groups, choices)[:, 1:].flatten()
    full_gram = np.zeros((width, width))
    full_gram[np.ix_(keep, keep)] = gram
    full_cross = np.zeros(width)
    full_cross[keep] = cross
    value = counts.flatten().astype(float)
    gradient = full_gram @ value - full_cross
    blocks = np.stack(
        [
            full_gram[
                row * choices : (row + 1) * choices, row * choices : (row + 1) * choices
            ]
            for row in range(groups)
        ]
    )
    diagonal = np.diagonal(blocks, axis1=1, axis2=2)
    difference = 0.0
    moves = 0
    for _ in range(plan["integer_moves"]):
        local = gradient.reshape(groups, choices)
        change = (
            local[:, None, :]
            - local[:, :, None]
            + (diagonal[:, None, :] + diagonal[:, :, None] - 2 * blocks) / 2
        )
        change[value.reshape(groups, choices) <= 0] = math.inf
        change[:, np.arange(choices), np.arange(choices)] = math.inf
        flat = int(change.argmin())
        row, old, new = np.unravel_index(flat, change.shape)
        best = float(change[row, old, new])
        if best >= -plan["integer_decrease"]:
            break
        source, target = row * choices + old, row * choices + new
        value[source] -= 1
        value[target] += 1
        gradient += full_gram[:, target] - full_gram[:, source]
        difference += best
        moves += 1
    return value.reshape(groups, choices).astype(np.int16), {
        "moves": moves,
        "half_mse_change": difference,
        "minimum_next_change": best,
        "local_stationary": best >= -plan["integer_decrease"],
    }


@torch.no_grad()
def refit_codes(index, centers, train_x, train_y, validation_x, validation_y, plan):
    started = time.monotonic()
    device = train_x.device
    code = torch.tensor(centers, device=device, dtype=train_x.dtype)
    index = torch.tensor(index, device=device).long()
    basis = (
        F.one_hot(index.abs(), plan["codes"] + 1)[..., 1:].to(train_x.dtype)
        * index.sign()[..., None]
    )
    features = torch.einsum("ni,ihk->nhk", train_x, basis)
    validation_features = torch.einsum("ni,ihk->nhk", validation_x, basis)
    records, candidates = [], []

    def predict(feature, value):
        return (feature @ value + plan["bias"]).relu().sum(1) / math.sqrt(
            plan["hidden"]
        )

    for step in range(plan["code_refit_steps"] + 1):
        activation = features @ code + plan["bias"]
        prediction = activation.relu().sum(1) / math.sqrt(plan["hidden"])
        residual = train_y - prediction
        validation_mse = float(
            (predict(validation_features, code) - validation_y).square().mean()
        )
        records.append(
            {
                "step": step,
                "train_mse": float(residual.square().mean()),
                "validation_mse": validation_mse,
            }
        )
        candidates.append(code.clone())
        if step == plan["code_refit_steps"]:
            break
        jacobian = (features * (activation > 0)[..., None]).sum(1) / math.sqrt(
            plan["hidden"]
        )
        gram = jacobian.T @ jacobian / len(train_y)
        score = jacobian.T @ residual / len(train_y)
        scale = torch.linalg.eigvalsh(gram)[-1].clamp_min(1e-30)
        direction = torch.linalg.solve(
            gram
            + scale
            * plan["code_ridge_relative"]
            * torch.eye(plan["codes"], device=device, dtype=code.dtype),
            score,
        )
        accepted = False
        for exponent in range(plan["code_line_search"]):
            fraction = 2.0**-exponent
            candidate = code + fraction * direction
            if (
                candidate.min() < plan["axis_code_min"]
                or candidate.max() > plan["axis_code_max"]
            ):
                continue
            candidate_mse = (predict(features, candidate) - train_y).square().mean()
            if candidate_mse < residual.square().mean():
                code = candidate
                records[-1]["fraction"] = fraction
                accepted = True
                break
        if not accepted:
            break
    selected = int(np.argmin([row["validation_mse"] for row in records]))
    return candidates[selected].cpu().numpy(), {
        "records": records,
        "selected": selected,
        "seconds": time.monotonic() - started,
    }


@torch.no_grad()
def census(
    slices,
    targets,
    validation_slices,
    validation_targets,
    order,
    centers,
    method,
    plan,
    *,
    qp_solver=None,
    integer_solver=None,
    resume=None,
):
    device = slices[0].device
    prefix = np.zeros((1, 0), dtype=np.int16)
    totals = np.array([plan["hidden"]], dtype=np.int16)
    choices = np.r_[
        0, np.arange(1, plan["codes"] + 1), -np.arange(1, plan["codes"] + 1)
    ]
    trace, indices, codebooks = [], [], []
    centers = np.asarray(centers).copy()
    start = 0
    if resume is not None:
        indices = list(resume["indices"])
        codebooks = list(resume["codebooks"])
        trace = list(resume["trace"])
        centers = np.asarray(resume["centers"]).copy()
        start = len(indices)
        if start:
            columns = indices[-1][order[:start]].T
            unique, first, counts = np.unique(
                columns, axis=0, return_index=True, return_counts=True
            )
            appearance = np.argsort(first)
            prefix, totals = unique[appearance], counts[appearance]
    solver = function_qp if qp_solver is None else qp_solver
    count_optimizer = integer_moves if integer_solver is None else integer_solver
    for step, (x, y) in enumerate(
        zip(slices[start:], targets[start:], strict=True), start + 1
    ):
        numeric = numeric_prefix(prefix, centers, device)
        base = x[:, order[: step - 1]] @ numeric.T + plan["bias"]
        atom0 = base.relu() / math.sqrt(plan["hidden"])
        values = numeric_prefix(choices[1:], centers, device)
        matrix = (base[..., None] + x[:, order[step - 1], None, None] * values).relu()
        matrix = (matrix - base.relu()[..., None]).flatten(1) / math.sqrt(
            plan["hidden"]
        )
        target = y - atom0 @ y.new_tensor(totals)
        gram = matrix.T @ matrix / len(y)
        cross = matrix.T @ target / len(y)
        eigenvalues, vectors = torch.linalg.eigh(gram)
        valid = eigenvalues > eigenvalues[-1].clamp_min(1e-30) * plan["rank_relative"]
        record = {
            "prefix_length": step,
            "parameters": matrix.shape[1],
            "rank": int(valid.sum()),
            "condition": float(eigenvalues[-1] / eigenvalues[0])
            if eigenvalues[0] > 0
            else None,
            "codebook_before": centers.tolist(),
        }
        if not valid.all():
            trace.append(
                {**record, "accepted": False, "reason": "rank_deficient_dictionary"}
            )
            break
        coefficient = vectors @ ((vectors.T @ cross) / eigenvalues)
        other = coefficient.reshape(len(prefix), len(choices) - 1).cpu().numpy()
        estimated = np.c_[totals - other.sum(1), other]
        continuous = matrix @ coefficient - target
        record["slice_mse_unconstrained"] = float(continuous.square().mean())
        rounded = integer_counts(estimated, totals)
        g, s = gram.cpu().numpy(), cross.cpu().numpy()
        if method != "euclidean":
            constrained, qp = solver(g, s, rounded[:, 1:].flatten(), totals, plan)
            record["qp"] = qp
            if not qp["converged"]:
                trace.append(
                    {**record, "accepted": False, "reason": "qp_not_converged"}
                )
                break
            nonzero = constrained.reshape(len(prefix), len(choices) - 1)
            rounded = integer_counts(np.c_[totals - nonzero.sum(1), nonzero], totals)
        if method in ("qp_integer", "qp_integer_refit"):
            rounded, numerical = count_optimizer(rounded, g, s, plan)
            record["integer"] = numerical
        rounded_other = y.new_tensor(rounded[:, 1:].flatten())
        record["slice_mse_integer_before_refit"] = float(
            (matrix @ rounded_other - target).square().mean()
        )
        extended = np.c_[
            np.repeat(prefix, len(choices), axis=0), np.tile(choices, len(prefix))
        ]
        keep = rounded.flatten() > 0
        prefix = extended[keep]
        totals = rounded.flatten()[keep]
        index = np.zeros((plan["inputs"], plan["hidden"]), dtype=np.int16)
        index[order[:step]] = np.repeat(prefix, totals, axis=0).T
        if method == "qp_integer_refit":
            centers, fitted = refit_codes(
                index,
                centers,
                torch.cat(slices[:step]),
                targets[:step].flatten(),
                torch.cat(validation_slices[:step]),
                validation_targets[:step].flatten(),
                plan,
            )
            record["code_refit"] = fitted
        weights = numeric_prefix(index, centers, device)
        source_prediction = (x @ weights + plan["bias"]).relu().sum(1) / math.sqrt(
            plan["hidden"]
        )
        validation_prediction = (
            validation_slices[step - 1] @ weights + plan["bias"]
        ).relu().sum(1) / math.sqrt(plan["hidden"])
        record.update(
            {
                "accepted": True,
                "reason": "integer_prefix",
                "distinct_prefixes": len(prefix),
                "codebook_after": centers.tolist(),
                "slice_mse_integer": float((source_prediction - y).square().mean()),
                "prefix_validation_mse": float(
                    (validation_prediction - validation_targets[step - 1])
                    .square()
                    .mean()
                ),
            }
        )
        indices.append(index)
        codebooks.append(centers.copy())
        trace.append(record)
    return {
        "trace": trace,
        "indices": np.stack(indices)
        if indices
        else np.zeros((0, plan["inputs"], plan["hidden"]), dtype=np.int16),
        "codebooks": np.stack(codebooks) if codebooks else np.zeros((0, plan["codes"])),
        "complete": len(indices) == plan["inputs"],
        "centers": centers,
    }
