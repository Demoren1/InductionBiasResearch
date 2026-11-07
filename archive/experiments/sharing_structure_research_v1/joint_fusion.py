"""Joint signed profile fusion: explicit PACS/convex-clustering adaptation."""

import time

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .competitor_models import signed_basis, stats_loss, validation_loss


def signed_incidence(entries, device):
    """Zero, differences and sums; B'B=(2E-1)I."""
    left, right = np.triu_indices(entries, 1)
    pairs = len(left)
    rows = np.r_[
        np.arange(entries),
        entries + np.arange(pairs).repeat(2),
        entries + pairs + np.arange(pairs).repeat(2),
    ]
    columns = np.r_[
        np.arange(entries),
        np.stack((left, right), 1).ravel(),
        np.stack((left, right), 1).ravel(),
    ]
    values = np.r_[np.ones(entries), np.tile((1.0, -1.0), pairs), np.ones(2 * pairs)]
    matrix = torch.sparse_coo_tensor(
        torch.tensor(np.stack((rows, columns)), device=device),
        torch.tensor(values, device=device),
        (entries**2, entries),
    ).coalesce()
    return matrix, left, right


def active_index(active, left, right, shape):
    """Native proximal equality graph, including sign cycles forcing zero."""
    entries = int(np.prod(shape))
    pairs = len(left)
    anchor = 2 * entries
    adjacency = np.eye(anchor + 1, dtype=bool)
    zero = np.flatnonzero(active[:entries])
    for positions in (zero, zero + entries):
        adjacency[positions, anchor] = True
        adjacency[anchor, positions] = True
    for sign, selected in (
        (1, active[entries : entries + pairs]),
        (-1, active[entries + pairs :]),
    ):
        first, second = left[selected], right[selected]
        if sign == -1:
            second = second + entries
        for a, b in (
            (first, second),
            ((first + entries) % (2 * entries), (second + entries) % (2 * entries)),
        ):
            adjacency[a, b] = True
            adjacency[b, a] = True
    _, labels = connected_components(csr_matrix(adjacency), directed=False)
    zero_label = labels[anchor]
    cyclic = np.unique(labels[:entries][labels[:entries] == labels[entries:anchor]])
    labels[np.isin(labels, cyclic)] = zero_label
    mapping, following = {zero_label: 0}, 1
    result = np.zeros(entries, dtype=np.int16)
    for entry in range(entries):
        positive, negative = labels[entry], labels[entry + entries]
        if positive not in mapping:
            mapping[positive], mapping[negative] = following, -following
            following += 1
        result[entry] = mapping[positive]
    return result.reshape(shape), following - 1


@torch.no_grad()
def train_joint(rows, lambdas, stats, validation, plan, solver, weight_metric="raw"):
    """Warm lambda paths; eligible readout requires both ADMM residuals small."""
    gram, cross = stats["gram"] / plan["n"], stats["cross"] / plan["n"]
    initial = torch.linalg.solve(gram, cross.transpose(-1, -2)).transpose(-1, -2)
    tasks, outputs, inputs = initial.shape
    entries, count = outputs * inputs, len(rows)
    matrix, left, right = signed_incidence(entries, gram.device)
    transpose = matrix.transpose(0, 1).coalesce()

    def differences(weights):
        profiles = weights.flatten(2).permute(2, 0, 1).reshape(entries, count * tasks)
        return (
            torch.sparse.mm(matrix, profiles)
            .reshape(entries**2, count, tasks)
            .permute(1, 0, 2)
            .contiguous()
        )

    def adjoint(edges):
        values = edges.permute(1, 0, 2).reshape(entries**2, count * tasks)
        return (
            torch.sparse.mm(transpose, values)
            .reshape(entries, count, tasks)
            .permute(1, 2, 0)
            .reshape(count, tasks, outputs, inputs)
        )

    current = initial[None].expand(count, -1, -1, -1).clone()
    dense_residual = stats["square_y"] - (initial * stats["cross"]).sum(-1)
    degrees = tasks * outputs * (plan["n"] - inputs)
    variance = dense_residual.sum() / degrees
    floor = (
        (variance * torch.linalg.inv(stats["gram"]).diagonal(dim1=-2, dim2=-1))
        .mean(0)
        .sqrt()
        .median()
    )
    initial_differences = differences(current)
    rms = initial_differences.square().mean(-1).sqrt()
    weights = torch.stack(
        [(rms[i] + floor).pow(-row["gamma"]) for i, row in enumerate(rows)]
    )
    if weight_metric != "raw":
        from .fusion_weights import calibrated_weights

        weights, floor = calibrated_weights(
            initial_differences[0],
            torch.linalg.inv(stats["gram"]),
            variance,
            rows,
            outputs,
            weight_metric,
        )
    weights /= weights.median(-1, keepdim=True).values
    weights[:, entries:] /= entries - 1
    weights[:, :entries] *= weights.new_tensor([row["zero_ratio"] for row in rows])[
        :, None
    ]
    primal = initial_differences.clone()
    dual = torch.zeros_like(primal)
    rho = gram.new_full((count,), solver["rho_shift"] / (2 * entries - 1))
    shape = (count, len(lambdas), outputs, inputs)
    saved_indices = np.zeros(shape, dtype=np.int16)
    saved_weights = np.zeros((count, len(lambdas), tasks, outputs, inputs))
    best_validation = np.full((count, len(lambdas)), np.inf)
    selected_steps = np.full((count, len(lambdas)), -1, dtype=int)
    history, masks = [], []
    cumulative_steps, started = 0, time.perf_counter()
    identity = torch.eye(inputs, device=gram.device, dtype=gram.dtype)
    smooth_factor = torch.linalg.cholesky(gram)
    for path, regularization in enumerate(lambdas):
        factor = torch.linalg.cholesky(
            gram[None] + (rho * (2 * entries - 1))[:, None, None, None] * identity
        )
        for iteration in range(1, solver["maximum_steps"] + 1):
            right_hand = cross[None] + rho[:, None, None, None] * adjoint(primal - dual)
            current = torch.cholesky_solve(
                right_hand.transpose(-1, -2), factor
            ).transpose(-1, -2)
            raw = differences(current)
            relaxed = (
                solver["over_relaxation"] * raw
                + (1 - solver["over_relaxation"]) * primal
            )
            proposal = relaxed + dual
            norms = torch.linalg.vector_norm(proposal, dim=-1)
            cutoff = regularization * np.sqrt(tasks) * weights / rho[:, None]
            following = (
                proposal
                * (1 - cutoff / norms.clamp_min(1e-300)).clamp_min(0)[..., None]
            )
            dual += relaxed - following
            check = (
                iteration % solver["checkpoint_every"] == 0
                or iteration == solver["maximum_steps"]
            )
            if check:
                p_norm = (raw - following).square().mean((1, 2)).sqrt()
                d_norm = (
                    (rho[:, None, None, None] * adjoint(following - primal))
                    .square()
                    .mean((1, 2, 3))
                    .sqrt()
                )
                p_scale = (
                    1
                    + raw.square().mean((1, 2)).sqrt()
                    + following.square().mean((1, 2)).sqrt()
                )
                d_scale = 1 + cross.square().mean().sqrt()
                p_ratio, d_ratio = p_norm / p_scale, d_norm / d_scale
                converged = (p_ratio <= solver["tolerance"]) & (
                    d_ratio <= solver["tolerance"]
                )
                edge_dual = rho[:, None, None] * dual
                dual_norm = torch.linalg.vector_norm(edge_dual, dim=-1)
                ball = regularization * np.sqrt(tasks) * weights
                edge_dual = (
                    edge_dual
                    * (ball / dual_norm.clamp_min(1e-300)).clamp_max(1)[..., None]
                )
                dual_right = cross[None] - adjoint(edge_dual)
                dual_solution = torch.cholesky_solve(
                    dual_right.transpose(-1, -2), smooth_factor[None]
                ).transpose(-1, -2)
                penalty = (
                    regularization
                    * np.sqrt(tasks)
                    * (weights * raw.square().sum(-1).sqrt()).sum(-1)
                    / (tasks * outputs)
                )
                primal_value = (
                    0.5 * (current @ gram[None] * current).sum((1, 2, 3))
                    - (current * cross[None]).sum((1, 2, 3))
                ) / (tasks * outputs) + penalty
                dual_gap = primal_value + 0.5 * (dual_solution * dual_right).sum(
                    (1, 2, 3)
                ) / (tasks * outputs)
                relative_gap = dual_gap / (1 + primal_value.abs())
                if bool((relative_gap < -1e-9).any()):
                    raise ArithmeticError(
                        "A feasible convex dual bound exceeded the primal objective"
                    )
                converged &= relative_gap <= solver["tolerance"]
                active = (following.square().sum(-1) == 0).cpu().numpy()
                indexators, groups = zip(
                    *(active_index(a, left, right, (outputs, inputs)) for a in active)
                )
                indexators = np.stack(indexators)
                masks.append(indexators != 0)
                eligible = [
                    i
                    for i in range(count)
                    if bool(converged[i]) and groups[i] <= plan["codes"]
                ]
                scores = np.full(count, np.inf)
                if eligible:
                    assigned = signed_basis(
                        torch.tensor(indexators[eligible], device=gram.device),
                        plan["codes"],
                    )
                    measured = validation_loss(assigned, validation).cpu().numpy()
                    for i, value in zip(eligible, measured):
                        scores[i] = value
                        if value < best_validation[i, path]:
                            best_validation[i, path] = value
                            saved_indices[i, path] = indexators[i]
                            saved_weights[i, path] = current[i].cpu().numpy()
                            selected_steps[i, path] = cumulative_steps + iteration
                mse = stats_loss(current, stats, plan["n"])
                history.append(
                    {
                        "step": cumulative_steps + iteration,
                        "lambda_index": path,
                        "lambda": regularization,
                        "iteration": iteration,
                        "source_mse": mse.tolist(),
                        "objective_per_output_task": (0.5 * mse + penalty).tolist(),
                        "primal_residual": p_ratio.tolist(),
                        "dual_residual": d_ratio.tolist(),
                        "relative_primal_dual_gap": relative_gap.tolist(),
                        "native_groups": list(groups),
                        "converged": converged.tolist(),
                        "validation_mse": [
                            None if not np.isfinite(v) else float(v) for v in scores
                        ],
                        "seconds": time.perf_counter() - started,
                    }
                )
                if bool(converged.all()):
                    primal = following
                    break
                if solver["adaptive_rho"]:
                    multiplier = torch.where(
                        p_ratio > 10 * d_ratio,
                        2.0,
                        torch.where(d_ratio > 10 * p_ratio, 0.5, 1.0),
                    )
                    updated = (rho * multiplier).clamp(
                        1e-6 / (2 * entries - 1), 1e4 / (2 * entries - 1)
                    )
                    dual *= (rho / updated)[:, None, None]
                    rho = updated
                    factor = torch.linalg.cholesky(
                        gram[None]
                        + (rho * (2 * entries - 1))[:, None, None, None] * identity
                    )
            primal = following
        cumulative_steps += iteration
        print(
            {
                "lambda": regularization,
                "steps": iteration,
                "converged": converged.tolist(),
                "groups": list(groups),
                "seconds": time.perf_counter() - started,
            },
            flush=True,
        )
    return {
        "indices": saved_indices,
        "weights": saved_weights,
        "selection_mse": best_validation,
        "selected_step": selected_steps,
        "trace_masks": np.stack(masks),
        "history": history,
        "seconds": time.perf_counter() - started,
        "adaptive_floor": float(floor),
        "weight_metric": weight_metric,
    }
