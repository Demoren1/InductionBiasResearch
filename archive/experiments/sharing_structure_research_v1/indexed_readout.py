"""Task-loss-based native signed indexators for an optional shared readout."""

import itertools
import math

import numpy as np
import torch

from .joint_readout_control import feature
from .weighted_prefix_search import values


def code_gauges(axis):
    codes = axis.shape[1]
    permutations = np.array(list(itertools.permutations(range(codes))))
    signs = np.array(list(itertools.product((-1, 1), repeat=codes)))
    return (axis[:, permutations][:, :, None, :] * signs[None, None]).reshape(
        len(axis), -1, codes
    )


@torch.no_grad()
def profile_codes(index, axis, data, spec, device):
    """Choose a signed/permuted axis estimate by independent H0 loss."""
    grid = code_gauges(axis)
    tasks, starts, codes = grid.shape
    flat_codes = torch.tensor(
        grid.reshape(-1, codes), device=device, dtype=torch.float64
    )
    train_feature = feature(data["fit_x"], index, codes, device, torch.float64)
    validation_feature = feature(data["h0_x"], index, codes, device, torch.float64)
    train_y = torch.tensor(data["fit_y"], device=device, dtype=torch.float64)
    validation_y = torch.tensor(data["h0_y"], device=device, dtype=torch.float64)
    scores, heads, losses = [], [], []
    identity = torch.eye(spec["hidden"], device=device, dtype=torch.float64)
    ids = torch.arange(len(flat_codes), device=device) // starts
    # Several dozen mappings in each batch keep the small models efficient.
    for begin in range(0, len(flat_codes), 192):
        end = min(begin + 192, len(flat_codes))
        code = flat_codes[begin:end]
        phi = (
            (torch.einsum("nhk,bk->bnh", train_feature, code) + spec["bias"]).relu()
            - max(spec["bias"], 0)
        ) / math.sqrt(spec["hidden"])
        gram = phi.transpose(1, 2) @ phi / phi.shape[1]
        cross = phi.transpose(1, 2) @ train_y[ids[begin:end]] / phi.shape[1]
        scale = gram.diagonal(dim1=1, dim2=2).amax(1).clamp_min(1e-12)
        head = torch.linalg.solve(gram + 1e-10 * scale[:, None, None] * identity, cross)
        vphi = (
            (
                torch.einsum("nhk,bk->bnh", validation_feature, code) + spec["bias"]
            ).relu()
            - max(spec["bias"], 0)
        ) / math.sqrt(spec["hidden"])
        scores.append(
            ((vphi @ head - validation_y[ids[begin:end]]) ** 2)
            .mean((1, 2))
            .cpu()
            .numpy()
        )
        losses.append(
            ((phi @ head - train_y[ids[begin:end]]) ** 2).mean((1, 2)).cpu().numpy()
        )
        heads.append(head.cpu().numpy())
    scores = np.concatenate(scores).reshape(tasks, starts)
    losses = np.concatenate(losses).reshape(tasks, starts)
    heads = np.concatenate(heads).reshape(
        tasks, starts, spec["hidden"], spec["outputs"]
    )
    winners = scores.argmin(axis=1)
    return (
        grid[np.arange(tasks), winners],
        heads[np.arange(tasks), winners],
        {
            "mappings": starts,
            "selected_mapping": winners.tolist(),
            "validation_mse": scores[np.arange(tasks), winners].tolist(),
            "grid_train": losses,
            "grid_validation": scores,
        },
    )


def design(x, index, codes, bias):
    values = np.c_[np.zeros(len(codes)), codes]
    weight = np.sign(index)[None] * values[:, np.abs(index)]
    return (
        np.maximum(np.einsum("nd,tdh->tnh", x, weight) + bias, 0) - max(bias, 0)
    ) / math.sqrt(index.shape[1])


@torch.no_grad()
def polish_codes(index, code, head, data, spec, device):
    parameters = torch.tensor(
        np.c_[code, head.reshape(len(code), -1)], device=device, dtype=torch.float64
    )
    train = feature(data["fit_x"], index, spec["codes"], device, torch.float64)
    validation = feature(data["h0_x"], index, spec["codes"], device, torch.float64)
    y = torch.tensor(data["fit_y"], device=device, dtype=torch.float64)
    vy = torch.tensor(data["h0_y"], device=device, dtype=torch.float64)
    best = torch.full((len(code),), torch.inf, device=device, dtype=torch.float64)
    selected = parameters.clone()
    identity = torch.eye(parameters.shape[1], device=device, dtype=torch.float64)
    history = []
    for step in range(17):
        prediction, jacobian = values(
            train, parameters, spec, spec["outputs"], jacobian=True
        )
        residual = (y - prediction).flatten(1)
        loss = residual.square().mean(1)
        score = (
            (values(validation, parameters, spec, spec["outputs"]) - vy)
            .square()
            .mean((1, 2))
        )
        take = score < best
        best[take], selected[take] = score[take], parameters[take]
        history.append(
            {
                "step": step,
                "train_mse": loss.cpu().tolist(),
                "validation_mse": score.cpu().tolist(),
            }
        )
        if step == 16:
            break
        gram = jacobian.transpose(1, 2) @ jacobian / residual.shape[1]
        cross = jacobian.transpose(1, 2) @ residual[..., None] / residual.shape[1]
        scale = gram.diagonal(dim1=1, dim2=2).amax(1).clamp_min(1e-12)
        direction = torch.linalg.solve(
            gram + 1e-6 * scale[:, None, None] * identity, cross
        )[..., 0]
        accepted = loss < 1e-24
        updated = parameters.clone()
        for exponent in range(12):
            proposal = parameters + 2.0 ** (-exponent) * direction
            proposal_loss = (
                (values(train, proposal, spec, spec["outputs"]) - y)
                .square()
                .mean((1, 2))
            )
            take = (~accepted) & (proposal.abs().amax(1) <= 16) & (proposal_loss < loss)
            updated[take], accepted[take] = proposal[take], True
            if accepted.all():
                break
        parameters = updated
    result = selected.cpu().numpy()
    return (
        result[:, : spec["codes"]],
        result[:, spec["codes"] :].reshape(head.shape),
        history,
    )


def canonical_head(index):
    mapping = {}
    result = np.zeros(index.shape, dtype=np.int16)
    for position in np.ndindex(index.shape):
        value = int(index[position])
        if value == 0:
            continue
        if abs(value) not in mapping:
            mapping[abs(value)] = (len(mapping) + 1) * (1 if value > 0 else -1)
        result[position] = mapping[abs(value)] * (1 if value > 0 else -1)
    return result


def head_basis(index):
    count = int(np.max(np.abs(index), initial=0))
    return (
        np.eye(count + 1)[np.abs(index).ravel(), 1:] * np.sign(index).ravel()[:, None]
    )


def fit_head(index, gram, cross):
    basis = head_basis(index)
    if not basis.shape[1]:
        return (
            np.zeros((len(gram),) + index.shape),
            np.empty((len(gram), 0)),
            np.empty((len(gram), 0, 0)),
        )
    group_gram = np.einsum("ek,tef,fl->tkl", basis, gram, basis)
    group_cross = np.einsum("ek,te->tk", basis, cross)
    scale = np.maximum(np.max(np.diagonal(group_gram, axis1=1, axis2=2), axis=1), 1e-12)
    inverse = np.linalg.inv(
        group_gram + 1e-12 * scale[:, None, None] * np.eye(basis.shape[1])
    )
    centers = np.einsum("tkl,tl->tk", inverse, group_cross)
    head = np.einsum("ek,tk->te", basis, centers).reshape((len(gram),) + index.shape)
    return head, centers, inverse


def head_path(phi, y, validation_phi, validation_y):
    tasks, count, hidden = phi.shape
    outputs = y.shape[-1]
    raw_gram = np.einsum("tnh,tnj->thj", phi, phi) / count
    gram = (
        np.einsum("thj,oq->thojq", raw_gram, np.eye(outputs)).reshape(
            tasks, hidden * outputs, hidden * outputs
        )
        / outputs
    )
    cross = np.einsum("tnh,tno->tho", phi, y).reshape(tasks, -1) / (count * outputs)
    index = np.arange(1, hidden * outputs + 1, dtype=np.int16).reshape(hidden, outputs)
    rows = []
    while True:
        head, centers, inverse = fit_head(index, gram, cross)
        prediction = np.einsum("tnh,tho->tno", phi, head)
        validation = np.einsum("tnh,tho->tno", validation_phi, head)
        groups = centers.shape[1]
        rows.append(
            {
                "index": index.copy(),
                "head": head,
                "centers": centers,
                "groups": groups,
                "train_mse": float(np.mean((prediction - y) ** 2)),
                "validation_mse": float(np.mean((validation - validation_y) ** 2)),
                "validation_prediction": validation,
            }
        )
        if not groups:
            break
        best = None
        diagonal = np.diagonal(inverse, axis1=1, axis2=2)
        for i in range(groups):
            deletion = float(
                np.mean(centers[:, i] ** 2 / np.maximum(diagonal[:, i], 1e-30))
            )
            move = (deletion, i + 1, 0, 0)
            if best is None or move < best:
                best = move
            for j in range(i + 1, groups):
                for sign in (-1, 1):
                    denominator = (
                        diagonal[:, i] + diagonal[:, j] - 2 * sign * inverse[:, i, j]
                    )
                    cost = float(
                        np.mean(
                            (centers[:, i] - sign * centers[:, j]) ** 2
                            / np.maximum(denominator, 1e-30)
                        )
                    )
                    move = (cost, j + 1, i + 1, sign)
                    best = min(best, move)
        _, source, destination, sign = best
        changed = index.copy()
        mask = np.abs(index) == source
        changed[mask] = np.sign(index[mask]) * destination * sign
        index = canonical_head(changed)
    best = min(rows, key=lambda row: row["validation_mse"])
    best_loss = (
        ((best["validation_prediction"] - validation_y) ** 2).mean(axis=2).ravel()
    )
    acceptable = []
    floor = 1e-12 * max(float(np.var(validation_y)), 1)
    for row in rows:
        delta = ((row["validation_prediction"] - validation_y) ** 2).mean(
            axis=2
        ).ravel() - best_loss
        error = float(np.std(delta, ddof=1) / math.sqrt(len(delta)))
        if delta.mean() <= 2 * error + floor:
            acceptable.append(row)
    selected = min(acceptable, key=lambda row: row["groups"])
    return rows, selected


def signed_relation(index):
    flat = index.ravel()
    return (
        (
            (np.abs(flat[:, None]) == np.abs(flat[None, :]))
            & (flat[:, None] != 0)
            & (flat[None, :] != 0)
        )
        * np.sign(flat[:, None])
        * np.sign(flat[None, :])
    )


def head_metrics(index, truth):
    relation, expected = signed_relation(index), signed_relation(truth)
    off_diagonal = np.triu(np.ones(relation.shape, dtype=bool), 1)
    difference = relation != expected
    pair_errors = int(np.sum(difference & off_diagonal))
    mask_errors = int(np.sum((index == 0) != (truth == 0)))
    return {
        "exact": pair_errors == 0 and mask_errors == 0,
        "pair_errors": pair_errors,
        "mask_errors": mask_errors,
        "groups": len(np.unique(np.abs(index[index != 0]))),
        "true_groups": len(np.unique(np.abs(truth[truth != 0]))),
    }
