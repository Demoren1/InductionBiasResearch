"""Numerical ReLU fits and scale-aware discrete adaptation of dense weights."""

import math

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment, minimize

INPUTS, HIDDEN, OUTPUTS = 784, 16, 5
LEARNING_RATES = (0.003, 0.01, 0.03)
STEPS = 200


def numerical_fit(
    panels, index=None, basis=None, *, precompute=True, device="cpu", seed=3511158401
):
    """Fresh numerical parameters; select states only on numerical validation."""
    x, y, vx, vy, qx = [
        torch.as_tensor(panels[key], dtype=torch.float32, device=device)
        for key in ("train_x", "train_y", "valid_x", "valid_y", "query_x")
    ]
    tasks = len(x)
    codes = (
        INPUTS * HIDDEN
        if index is None and basis is None
        else (basis.shape[-1] if basis is not None else int(np.abs(index).max()))
    )
    if index is not None:
        index = np.broadcast_to(index, (tasks, INPUTS, HIDDEN))
        labels = torch.as_tensor(np.abs(index).copy(), dtype=torch.long, device=device)
        signs = torch.as_tensor(
            np.sign(index).copy(), dtype=torch.float32, device=device
        )

        def features(data):
            result = data.new_zeros((tasks, data.shape[1], HIDDEN, codes + 1))
            for column in range(HIDDEN):
                result[:, :, column].scatter_add_(
                    2,
                    labels[:, None, :, column].expand(-1, data.shape[1], -1),
                    data * signs[:, None, :, column],
                )
            return result[..., 1:].contiguous()

        if precompute:
            x, vx, qx = [features(data) for data in (x, vx, qx)]
        energy = float(np.mean(index != 0))
    elif basis is not None:
        tensor = torch.as_tensor(basis, dtype=torch.float32, device=device)
        if precompute:
            x, vx, qx = [
                torch.einsum("bnd,dhk->bnhk", data, tensor).contiguous()
                for data in (x, vx, qx)
            ]
        energy = float(tensor.square().sum(-1).mean())
    else:
        energy = 1.0
    scenarios = len(LEARNING_RATES) * 2
    generator = torch.Generator().manual_seed(seed)
    length = codes + HIDDEN + HIDDEN * OUTPUTS + OUTPUTS
    parameter = torch.randn((scenarios, tasks, length), generator=generator)
    parameter[..., :codes] *= math.sqrt(2 / INPUTS / max(energy, 1e-12))
    parameter[..., codes : codes + HIDDEN] *= 0.1
    parameter[..., -OUTPUTS:] = 0
    parameter = parameter.to(device)
    parameter.requires_grad_()
    first, second = torch.zeros_like(parameter), torch.zeros_like(parameter)
    rates = parameter.new_tensor(np.repeat(LEARNING_RATES, 2))[:, None, None]
    best = parameter.new_full((scenarios, tasks), torch.inf)
    selected = parameter.detach().clone()
    selected_step = torch.zeros((scenarios, tasks), dtype=torch.int64, device=device)
    trace = {key: [] for key in ("step", "train", "validation", "recall")}

    def predict(data, values):
        if index is None and basis is None:
            weight = values[..., :codes].reshape(scenarios, tasks, INPUTS, HIDDEN)
            preactivation = torch.einsum("bnd,rbdh->rbnh", data, weight)
        elif index is not None and not precompute:
            dictionary = torch.cat((values[..., :1] * 0, values[..., :codes]), -1)
            positions = labels.flatten(1)[None].expand(scenarios, -1, -1)
            weight = (
                dictionary.gather(2, positions).reshape(
                    scenarios, tasks, INPUTS, HIDDEN
                )
                * signs[None]
            )
            preactivation = torch.einsum("bnd,rbdh->rbnh", data, weight)
        elif basis is not None and not precompute:
            weight = torch.einsum("dhk,rbk->rbdh", tensor, values[..., :codes])
            preactivation = torch.einsum("bnd,rbdh->rbnh", data, weight)
        else:
            preactivation = torch.einsum("bnhk,rbk->rbnh", data, values[..., :codes])
        activation = (preactivation + values[:, :, None, codes : codes + HIDDEN]).relu()
        head = values[..., codes + HIDDEN : -OUTPUTS].reshape(
            scenarios, tasks, HIDDEN, OUTPUTS
        )
        return (
            torch.einsum("rbnh,rbho->rbno", activation, head) / math.sqrt(HIDDEN)
            + values[:, :, None, -OUTPUTS:]
        )

    for step in range(STEPS + 1):
        if step % 25 == 0:
            with torch.no_grad():
                train_logits, valid_logits = (
                    predict(x, parameter),
                    predict(vx, parameter),
                )
                train_loss = -(train_logits.log_softmax(-1) * y[None]).sum(-1).mean(-1)
                valid_loss = -(valid_logits.log_softmax(-1) * vy[None]).sum(-1).mean(-1)
                recall = (
                    (valid_logits.argmax(-1) == vy.argmax(-1)[None]).float().mean(-1)
                )
                improved = valid_loss < best
                best[improved] = valid_loss[improved]
                selected[improved] = parameter[improved]
                selected_step[improved] = step
                trace["step"].append(step)
                trace["train"].append(train_loss.mean().item())
                trace["validation"].append(valid_loss.mean().item())
                trace["recall"].append(recall.mean().item())
        if step == STEPS:
            break
        loss = -(predict(x, parameter).log_softmax(-1) * y[None]).sum(-1).mean(-1).sum()
        gradient = torch.autograd.grad(loss, parameter)[0]
        with torch.no_grad():
            first.mul_(0.9).add_(gradient, alpha=0.1)
            second.mul_(0.999).addcmul_(gradient, gradient, value=0.001)
            parameter.add_(
                -rates
                * (first / (1 - 0.9 ** (step + 1)))
                / ((second / (1 - 0.999 ** (step + 1))).sqrt() + 1e-8)
            )
    with torch.no_grad():
        chosen = best.argmin(0)
        task = torch.arange(tasks, device=device)
        prediction = predict(qx, selected)[chosen, task].cpu().numpy()
        train_prediction = predict(x, selected)[chosen, task].cpu().numpy()
        valid_prediction = predict(vx, selected)[chosen, task].cpu().numpy()
        weights = selected[chosen, task, :codes].cpu().numpy()
        ensemble = selected[..., :codes].permute(1, 0, 2).cpu().numpy()
    return {
        "prediction": prediction,
        "train_prediction": train_prediction,
        "valid_prediction": valid_prediction,
        "weight_parameters": weights,
        "ensemble_parameters": ensemble,
        "selected_step": selected_step[chosen, task].cpu().numpy(),
        "scenario": chosen.cpu().numpy(),
        "trace": trace,
        "codes": codes,
    }


def dense_descriptor(result):
    weights = result["weight_parameters"].reshape(-1, INPUTS, HIDDEN)
    normal = weights / np.maximum(np.linalg.norm(weights, axis=1, keepdims=True), 1e-12)
    ensemble = result["ensemble_parameters"].reshape(-1, 6, INPUTS, HIDDEN)
    uncertainty = []
    for reference, choices in zip(normal, ensemble):
        aligned = []
        for weight in choices:
            unit = weight / np.maximum(np.linalg.norm(weight, axis=0), 1e-12)
            left, right = linear_sum_assignment(-reference.T @ unit)
            aligned.append(unit[:, right[np.argsort(left)]])
        uncertainty.append(
            np.sqrt(np.mean((np.stack(aligned) - reference) ** 2, axis=(0, 1)))
        )
    return normal, np.maximum(uncertainty, 0.0003)


def calibration(index, normal, uncertainty):
    """Eliminate code values and solve the small positive-scale quadratic."""
    batch, _dimensions, hidden = index.shape
    codes = int(np.abs(index).max())
    labels, signs = np.abs(index), np.sign(index)
    coordinates = (
        np.arange(batch)[:, None, None] * (codes + 1) + labels
    ) * hidden + np.arange(hidden)[None, None]
    shape = (batch, codes + 1, hidden)
    counts0 = np.bincount(coordinates.ravel(), minlength=np.prod(shape)).reshape(shape)[
        :, 1:
    ]
    cross0 = np.bincount(
        coordinates.ravel(),
        weights=(normal * signs).ravel(),
        minlength=np.prod(shape),
    ).reshape(shape)[:, 1:]
    anchors = np.zeros((batch, hidden), dtype=bool)
    for task, counts in enumerate(counts0):
        incidence = (counts > 0).astype(np.int32)
        graph = incidence.T @ incidence > 0
        remaining = set(range(hidden))
        while remaining:
            group, frontier = set(), {min(remaining)}
            while frontier:
                group |= frontier
                frontier = set(np.flatnonzero(graph[list(frontier)].any(0))) - group
            columns = np.array(sorted(group), dtype=int)
            anchors[task, columns[uncertainty[task, columns].argmin()]] = True
            remaining -= group
    scales = np.ones((batch, hidden))
    norm = np.sum(normal**2, axis=1)
    for _ in range(3):
        deviation = uncertainty * np.maximum(scales, 0.01)
        precision = 1 / deviation**2
        cross = cross0 * precision[:, None]
        counts = np.maximum(np.einsum("bkh,bh->bk", counts0, precision), 1e-30)
        schur = -np.einsum("bkh,bkj->bhj", cross / counts[..., None], cross)
        diagonal = np.arange(hidden)
        schur[:, diagonal, diagonal] += norm * precision
        fixed = anchors.astype(float)
        rhs = -np.einsum("bhj,bj->bh", schur, fixed)
        system = schur.copy()
        ridge = np.maximum(np.abs(system[:, diagonal, diagonal]).max(1), 1) * 1e-10
        system[:, diagonal, diagonal] += (~anchors) * ridge[:, None]
        system[anchors] = 0
        system[:, diagonal, diagonal] = np.where(
            anchors, 1, system[:, diagonal, diagonal]
        )
        rhs[anchors] = 0
        scales = np.linalg.solve(system, rhs[..., None])[..., 0] + fixed
        for task in np.flatnonzero((scales <= 1e-6).any(1)):
            free = np.flatnonzero(~anchors[task])
            local = schur[task][np.ix_(free, free)] + np.eye(len(free)) * ridge[task]
            cross_term = schur[task][np.ix_(free, np.flatnonzero(anchors[task]))].sum(1)
            answer = minimize(
                lambda value, matrix, vector: (
                    0.5 * value @ matrix @ value + vector @ value,
                    matrix @ value + vector,
                ),
                np.maximum(scales[task, free], 1e-6),
                args=(local, cross_term),
                jac=True,
                bounds=[(1e-6, None)] * len(free),
                method="L-BFGS-B",
            )
            scales[task, free] = answer.x
        code = np.einsum("bkh,bh->bk", cross, scales) / counts
    return code, scales


def align(index, normal, uncertainty):
    batch = len(normal)
    template = np.sign(index).astype(float)
    template /= np.maximum(np.linalg.norm(template, axis=0), 1e-12)
    proxies = np.broadcast_to(template, normal.shape).copy()
    expanded = np.broadcast_to(index, normal.shape).copy()
    for _ in range(3):
        aligned, deviation = [], []
        for proxy, observed, error in zip(proxies, normal, uncertainty):
            left, right = linear_sum_assignment(-proxy.T @ observed)
            order = right[np.argsort(left)]
            aligned.append(observed[:, order])
            deviation.append(error[order])
        aligned, deviation = np.stack(aligned), np.stack(deviation)
        codes, _ = calibration(expanded, aligned, deviation)
        dictionary = np.concatenate((np.zeros((batch, 1)), codes), 1)
        proxies = np.take_along_axis(
            dictionary[:, None], np.abs(expanded), 2
        ) * np.sign(expanded)
        proxies /= np.maximum(np.linalg.norm(proxies, axis=1, keepdims=True), 1e-12)
    return aligned, deviation


def project(initial, normal, uncertainty, penalty, steps):
    """Cauchy assignment updates, including zero, with a fixed aligned anchor."""
    anchor = np.broadcast_to(initial, normal.shape).copy()
    index = anchor.copy()
    codes = int(np.abs(initial).max())
    symbols = np.r_[0, np.arange(1, codes + 1), -np.arange(1, codes + 1)]
    for _ in range(steps):
        values, scales = calibration(index, normal, uncertainty)
        centers = np.concatenate((np.zeros((len(normal), 1)), values, -values), 1)
        updated = np.empty_like(index)
        # Bound the temporary categorical tensor on CPU.
        for begin in range(0, len(normal), 4):
            local = slice(begin, begin + 4)
            target = scales[local, None] * normal[local]
            deviation = uncertainty[local] * np.maximum(scales[local], 0.01)
            residual = (target[..., None] - centers[local, None, None]) / deviation[
                :, None, :, None
            ]
            cost = 4 * np.log1p(residual**2 / 4)
            cost += penalty * (symbols != anchor[local, ..., None])
            chosen = symbols[cost.argmin(-1)].astype(np.int16)
            minimum = cost.min(-1)
            for task in range(len(chosen)):
                for group in range(1, codes + 1):
                    if np.any(np.abs(chosen[task]) == group):
                        continue
                    counts = np.bincount(
                        np.abs(chosen[task]).ravel(), minlength=codes + 1
                    )
                    allowed = (chosen[task] == 0) | (counts[np.abs(chosen[task])] > 1)
                    excess = (
                        np.minimum(
                            cost[task, ..., group],
                            cost[task, ..., group + codes],
                        )
                        - minimum[task]
                    )
                    position = int(np.where(allowed, excess, np.inf).argmin())
                    flat = cost[task].reshape(-1, 2 * codes + 1)
                    chosen[task].ravel()[position] = (
                        group
                        if flat[position, group] <= flat[position, group + codes]
                        else -group
                    )
            updated[local] = chosen
        if np.array_equal(updated, index):
            break
        index = updated
    return index
