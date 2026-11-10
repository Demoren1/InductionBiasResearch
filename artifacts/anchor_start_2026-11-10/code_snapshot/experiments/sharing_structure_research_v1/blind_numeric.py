"""Free-bias, free-readout fitting of data-derived integer structures."""

import itertools
import math
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def calibrate_geometry(index, normal, uncertainty, mapping, initial):
    dimensions, hidden = index.shape
    codes = len(initial)
    parent = np.arange(codes + hidden)

    def root(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for d, h in zip(*np.where(index != 0)):
        left, right = root(abs(index[d, h]) - 1), root(codes + h)
        parent[left] = right
    components = {}
    for h in range(hidden):
        components.setdefault(root(codes + h), []).append(h)
    anchors = {
        min(columns, key=lambda h: uncertainty[mapping[h]])
        for columns in components.values()
        if all(mapping[h] >= 0 for h in columns)
    }
    free = [h for h in range(hidden) if h not in anchors]
    location = {h: codes + j for j, h in enumerate(free)}
    matrix = np.zeros((dimensions * hidden, codes + len(free)))
    target = np.zeros(dimensions * hidden)
    for row, (d, h) in enumerate(np.ndindex(index.shape)):
        if index[d, h]:
            matrix[row, abs(index[d, h]) - 1] = np.sign(index[d, h])
        if mapping[h] < 0:
            continue
        if h in anchors:
            target[row] = normal[d, mapping[h]]
        else:
            matrix[row, location[h]] = -normal[d, mapping[h]]
    deviation = np.array(
        [uncertainty[mapping[h]] if mapping[h] >= 0 else 1 for h in range(hidden)]
    )
    weights = 1 / np.maximum(deviation, 0.001) ** 2
    scales = np.ones(hidden)
    code = initial.copy()
    for _ in range(3):
        row_weight = np.tile(np.sqrt(weights), dimensions)
        solution = np.linalg.lstsq(
            matrix * row_weight[:, None], target * row_weight, rcond=1e-10
        )[0]
        proposed = solution[codes:]
        if np.any(proposed <= 0) or not np.isfinite(solution).all():
            break
        code = solution[:codes]
        scales[free] = proposed
        residual = (matrix @ solution - target).reshape(dimensions, hidden)
        error = np.sqrt(np.mean(residual**2, axis=0))
        robust = np.minimum(
            1, 3 * max(float(np.median(error)), 1e-6) / np.maximum(error, 1e-12)
        )
        weights = robust / np.maximum(deviation * scales, 0.001) ** 2
    return code / max(float(np.linalg.norm(code) / math.sqrt(codes)), 1e-12)


@torch.no_grad()
def initial_states(index, source, codes, x, y, vx, vy, keep=2, device=None):
    hidden, outputs = index.shape[1], y.shape[1]
    permutations = list(itertools.permutations(range(codes)))
    signs = list(itertools.product((-1, 1), repeat=codes))
    available = [name for name in source if name.startswith("codes_")][:2]
    choices = []
    normal, estimated_bias = source["normal"], source["bias"]
    for name in available:
        original = source[name]
        for permutation in permutations:
            for sign in signs:
                c = original[list(permutation)] * sign
                c /= max(float(np.linalg.norm(c) / math.sqrt(codes)), 1e-12)
                weight = np.sign(index) * np.r_[0, c][np.abs(index)]
                norm = np.linalg.norm(weight, axis=0)
                unit = weight / np.maximum(norm, 1e-12)
                cost = np.sum((unit[:, :, None] - normal[:, None, :]) ** 2, axis=0)
                left, right = linear_sum_assignment(cost)
                mapping = np.full(hidden, -1, dtype=int)
                mapping[left] = right
                choices.append((float(cost[left, right].sum()), c, norm, mapping))
    choices.sort(key=lambda row: row[0])
    records, seen, calibrated = [], set(), set()
    for _, c, norm, mapping in choices[:16]:
        if c.tobytes() in seen:
            continue
        seen.add(c.tobytes())
        for _ in range(2):
            c = calibrate_geometry(index, normal, source["uncertainty"], mapping, c)
            weight = np.sign(index) * np.r_[0, c][np.abs(index)]
            norm = np.linalg.norm(weight, axis=0)
            unit = weight / np.maximum(norm, 1e-12)
            cost = np.sum((unit[:, :, None] - normal[:, None, :]) ** 2, axis=0)
            left, right = linear_sum_assignment(cost)
            mapping = np.full(hidden, -1, dtype=int)
            mapping[left] = right
        geometry_key = (c.tobytes(), tuple(mapping))
        if geometry_key in calibrated:
            continue
        calibrated.add(geometry_key)
        bias = np.zeros(hidden)
        matched = mapping >= 0
        bias[matched] = estimated_bias[mapping[matched]] * norm[matched]
        if device is not None:
            parameters = np.r_[c, bias, np.zeros(hidden * outputs + outputs)]
            records.append((0, parameters))
            continue
        weight = np.sign(index) * np.r_[0, c][np.abs(index)]
        design = np.c_[
            np.maximum(x @ weight + bias, 0) / math.sqrt(hidden), np.ones(len(x))
        ]
        coefficient = np.linalg.lstsq(design, y, rcond=1e-12)[0]
        validation = (
            np.c_[
                np.maximum(vx @ weight + bias, 0) / math.sqrt(hidden), np.ones(len(vx))
            ]
            @ coefficient
        )
        parameters = np.r_[c, bias, coefficient[:-1].ravel(), coefficient[-1]]
        records.append((float(np.mean((validation - vy) ** 2)), parameters))
    if not records:
        raise ValueError("No observed numerical state available")
    if device is not None:
        parameters = torch.tensor(
            np.stack([row[1] for row in records]), device=device, dtype=torch.float64
        )
        tx, target, tvx, validation = [
            torch.tensor(a, device=device, dtype=torch.float64) for a in (x, y, vx, vy)
        ]
        feat = feature(tx[None], index[None], codes, device).expand(
            len(records), -1, -1, -1
        )
        vf = feature(tvx[None], index[None], codes, device).expand(
            len(records), -1, -1, -1
        )
        parameters = profile_head(
            feat,
            parameters,
            target[None].expand(len(records), -1, -1),
            codes,
            hidden,
            outputs,
        )
        score = (
            (prediction(vf, parameters, codes, hidden, outputs) - validation[None])
            .square()
            .mean((1, 2))
        )
        records = list(zip(score.cpu().tolist(), parameters.cpu().numpy()))
    records.sort(key=lambda row: row[0])
    while len(records) < keep:
        records.append(records[0])
    return np.stack([row[1] for row in records[:keep]])


def feature(x, index, codes, device):
    labels = torch.as_tensor(index, device=device, dtype=torch.long)
    basis = torch.nn.functional.one_hot(labels.abs(), codes + 1)[..., 1:]
    basis = basis.to(torch.float64) * labels.sign()[..., None]
    return torch.einsum("bnd,bdhk->bnhk", x, basis)


def prediction(feat, parameters, codes, hidden, outputs, *, jacobian=False):
    c = parameters[:, :codes]
    bias = parameters[:, codes : codes + hidden]
    head = parameters[:, codes + hidden : -outputs].reshape(-1, hidden, outputs)
    offset = parameters[:, -outputs:]
    preactivation = torch.einsum("bnhk,bk->bnh", feat, c) + bias[:, None]
    activation = preactivation.relu() / math.sqrt(hidden)
    value = torch.einsum("bnh,bho->bno", activation, head) + offset[:, None]
    if not jacobian:
        return value
    gate = (preactivation > 0).to(feat.dtype) / math.sqrt(hidden)
    code_jacobian = torch.einsum("bnhk,bnh,bho->bnok", feat, gate, head)
    bias_jacobian = torch.einsum("bnh,bho->bnoh", gate, head)
    eye = torch.eye(outputs, device=feat.device, dtype=feat.dtype)
    head_jacobian = torch.einsum("bnh,oq->bnohq", activation, eye).flatten(-2)
    offset_jacobian = eye[None, None].expand(len(feat), feat.shape[1], -1, -1)
    jac = torch.cat(
        [code_jacobian, bias_jacobian, head_jacobian, offset_jacobian], dim=-1
    )
    return value, jac.flatten(1, 2)


def normalize_gauge(parameters, codes, hidden, outputs):
    scale = parameters[:, :codes].square().mean(1).sqrt().clamp_min(1e-12)
    normalized = parameters.clone()
    normalized[:, : codes + hidden] /= scale[:, None]
    normalized[:, codes + hidden : -outputs] *= scale[:, None]
    return normalized


def profile_head(feat, parameters, target, codes, hidden, outputs):
    preactivation = torch.einsum("bnhk,bk->bnh", feat, parameters[:, :codes])
    preactivation += parameters[:, None, codes : codes + hidden]
    design = torch.cat(
        [
            preactivation.relu() / math.sqrt(hidden),
            torch.ones(
                (*preactivation.shape[:2], 1), device=feat.device, dtype=feat.dtype
            ),
        ],
        dim=-1,
    )
    gram = design.transpose(1, 2) @ design / design.shape[1]
    cross = design.transpose(1, 2) @ target / design.shape[1]
    scale = gram.diagonal(dim1=1, dim2=2).amax(1).clamp_min(1e-12)
    eye = torch.eye(hidden + 1, device=feat.device, dtype=feat.dtype)
    coefficient = torch.linalg.solve(gram + 1e-12 * scale[:, None, None] * eye, cross)
    result = parameters.clone()
    result[:, codes + hidden : -outputs] = coefficient[:, :-1].flatten(1)
    result[:, -outputs:] = coefficient[:, -1]
    return result


@torch.no_grad()
def fit_batch(indices, initial, x, y, vx, vy, codes, device, *, steps=32):
    started = time.monotonic()
    hidden, outputs = indices.shape[-1], y.shape[-1]
    dtype = torch.float64
    parameters = torch.as_tensor(initial, device=device, dtype=dtype)
    parameters = normalize_gauge(parameters, codes, hidden, outputs)
    tx, tvx = [torch.as_tensor(v, device=device, dtype=dtype) for v in (x, vx)]
    target, validation = [
        torch.as_tensor(v, device=device, dtype=dtype) for v in (y, vy)
    ]
    feat, vf = feature(tx, indices, codes, device), feature(tvx, indices, codes, device)
    best = parameters.new_full((len(parameters),), torch.inf)
    selected, selected_step = (
        parameters.clone(),
        torch.zeros(len(parameters), device=device, dtype=torch.long),
    )
    identity = torch.eye(parameters.shape[1], device=device, dtype=dtype)
    history_train, history_validation = [], []
    for step in range(steps + 1):
        parameters = profile_head(feat, parameters, target, codes, hidden, outputs)
        fitted, jac = prediction(
            feat, parameters, codes, hidden, outputs, jacobian=True
        )
        residual = (target - fitted).flatten(1)
        train = residual.square().mean(1)
        score = (
            (prediction(vf, parameters, codes, hidden, outputs) - validation)
            .square()
            .mean((1, 2))
        )
        better = score < best
        best[better], selected[better], selected_step[better] = (
            score[better],
            parameters[better],
            step,
        )
        history_train.append(train.cpu().numpy())
        history_validation.append(score.cpu().numpy())
        if step == steps:
            break
        gram = jac.transpose(1, 2) @ jac / residual.shape[1]
        cross = jac.transpose(1, 2) @ residual[..., None] / residual.shape[1]
        damping = gram.diagonal(dim1=1, dim2=2).amax(1).clamp_min(1e-12) * 1e-6
        direction = torch.linalg.solve(gram + damping[:, None, None] * identity, cross)[
            ..., 0
        ]
        accepted = train < 1e-24
        updated = parameters.clone()
        for exponent in range(12):
            proposal = normalize_gauge(
                parameters + 2.0 ** (-exponent) * direction, codes, hidden, outputs
            )
            proposal = profile_head(feat, proposal, target, codes, hidden, outputs)
            finite = torch.isfinite(proposal).all(1)
            loss = (
                (prediction(feat, proposal, codes, hidden, outputs) - target)
                .square()
                .mean((1, 2))
            )
            take = (~accepted) & finite & (loss < train)
            updated[take] = proposal[take]
            accepted |= take
            if accepted.all():
                break
        parameters = updated
    return selected.cpu().numpy(), {
        "validation_mse": best.cpu().numpy(),
        "selected_step": selected_step.cpu().numpy(),
        "train_history": np.stack(history_train),
        "validation_history": np.stack(history_validation),
        "seconds": time.monotonic() - started,
        "device": str(device),
    }


def numpy_predict(x, index, parameters, codes, outputs):
    hidden = index.shape[1]
    c, bias = parameters[:codes], parameters[codes : codes + hidden]
    head = parameters[codes + hidden : -outputs].reshape(hidden, outputs)
    weight = np.sign(index) * np.r_[0, c][np.abs(index)]
    return (
        np.maximum(x @ weight + bias, 0) @ head / math.sqrt(hidden)
        + parameters[-outputs:]
    )
