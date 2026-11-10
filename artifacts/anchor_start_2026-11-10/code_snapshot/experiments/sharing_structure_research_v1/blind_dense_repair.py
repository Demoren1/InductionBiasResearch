"""Revive unhelpful hidden units by profiled residual fitting on observed data."""

import argparse
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from .blind_contract import digest, load_sizes, save_json
from .blind_dense_extract import polish, values


def replacements(parameters, x, y, vx, vy, dimensions, hidden, outputs, device, seed):
    rng = np.random.default_rng(seed)
    removals, starts = min(3, hidden), 32
    fixed, validation, residual, inverse, bases, removed, task_ids = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for task, p in enumerate(parameters):
        count = dimensions * hidden
        weight = p[:count].reshape(dimensions, hidden)
        bias = p[count : count + hidden]
        activation = np.maximum(x[task] @ weight + bias, 0) / math.sqrt(hidden)
        vactivation = np.maximum(vx[task] @ weight + bias, 0) / math.sqrt(hidden)
        costs, states = [], []
        for h in range(hidden):
            columns = np.arange(hidden) != h
            design = np.c_[activation[:, columns], np.ones(len(activation))]
            vdesign = np.c_[vactivation[:, columns], np.ones(len(vactivation))]
            coefficient = np.linalg.lstsq(design, y[task], rcond=1e-12)[0]
            error = y[task] - design @ coefficient
            costs.append(float(np.mean(error**2)))
            states.append((design, vdesign, coefficient, error))
        for h in np.argsort(costs)[:removals]:
            design, vdesign, coefficient, error = states[h]
            fixed.append(design)
            validation.append(vdesign)
            residual.append(error)
            inverse.append(np.linalg.pinv(design.T @ design, rcond=1e-10))
            bases.append(coefficient)
            removed.append(int(h))
            task_ids.append(task)
    groups = len(fixed)
    tensors = [
        torch.tensor(np.stack(a), device=device, dtype=torch.float32)
        for a in (fixed, validation, residual, inverse, bases)
    ]
    tf, tvf, target_residual, ginverse, base = tensors
    tx, tvx, tvy = [
        torch.tensor(a[np.array(task_ids)], device=device, dtype=torch.float32)
        for a in (x, vx, vy)
    ]
    initial_weight = rng.normal(size=(groups, dimensions, starts)) * math.sqrt(
        2 / dimensions
    )
    initial_bias = rng.normal(0, 0.5, (groups, starts))
    for group, (task, h) in enumerate(zip(task_ids, removed)):
        p = parameters[task]
        w = p[: dimensions * hidden].reshape(dimensions, hidden)[:, h]
        b = p[dimensions * hidden + h]
        initial_weight[group, :, :8] = w[:, None] + rng.normal(
            size=(dimensions, 8)
        ) * max(float(np.std(w)), 0.1)
        initial_bias[group, :8] = b + rng.normal(0, 0.5, 8)
    weight = torch.tensor(
        initial_weight, device=device, dtype=torch.float32, requires_grad=True
    )
    bias = torch.tensor(
        initial_bias, device=device, dtype=torch.float32, requires_grad=True
    )
    optimizer = torch.optim.Adam([weight, bias], lr=0.03)
    best = torch.full((groups,), torch.inf, device=device)
    best_w, best_b = weight[:, :, 0].detach().clone(), bias[:, 0].detach().clone()
    best_amp = torch.zeros((groups, outputs), device=device)
    best_base = base.clone()
    variance = target_residual.square().mean((1, 2)).clamp_min(1e-12)
    for step in range(301):
        activation = (torch.bmm(tx, weight) + bias[:, None]).relu() / math.sqrt(hidden)
        projection = torch.bmm(ginverse, torch.bmm(tf.transpose(1, 2), activation))
        orthogonal = activation - torch.bmm(tf, projection)
        denominator = orthogonal.square().sum(1).clamp_min(1e-10)
        cross = torch.bmm(orthogonal.transpose(1, 2), target_residual)
        amplitude = cross / denominator[:, :, None]
        gain = cross.square().sum(2) / denominator / (tx.shape[1] * outputs)
        if step % 20 == 0 or step == 300:
            with torch.no_grad():
                vactivation = (
                    torch.bmm(tvx, weight) + bias[:, None]
                ).relu() / math.sqrt(hidden)
                orthogonal_v = vactivation - torch.bmm(tvf, projection)
                base_prediction = torch.bmm(tvf, base)
                prediction = (
                    base_prediction[:, :, None]
                    + orthogonal_v[..., None] * amplitude[:, None]
                )
                score = (prediction - tvy[:, :, None]).square().mean((1, 3))
                error, selected = score.min(1)
                changed = error < best
                ids = torch.arange(groups, device=device)
                old_coefficient = (
                    base[:, None]
                    - projection.transpose(1, 2)[..., None] * amplitude[:, :, None]
                )
                best[changed] = error[changed]
                best_w[changed] = weight[ids, :, selected][changed]
                best_b[changed] = bias[ids, selected][changed]
                best_amp[changed] = amplitude[ids, selected][changed]
                best_base[changed] = old_coefficient[ids, selected][changed]
        if step == 300:
            break
        optimizer.zero_grad(set_to_none=True)
        loss = -(gain / variance[:, None]).mean()
        loss.backward()
        optimizer.step()
    weight, bias, amplitude, coefficient = [
        a.detach().cpu().numpy() for a in (best_w, best_b, best_amp, best_base)
    ]
    proposals = [[p.copy()] for p in parameters]
    for group, (task, h) in enumerate(zip(task_ids, removed)):
        p = parameters[task].copy()
        p[: dimensions * hidden].reshape(dimensions, hidden)[:, h] = weight[group]
        p[dimensions * hidden + h] = bias[group]
        head = p[dimensions * hidden + hidden : -outputs].reshape(hidden, outputs)
        head[np.arange(hidden) != h] = coefficient[group, :-1]
        head[h] = amplitude[group]
        p[-outputs:] = coefficient[group, -1]
        proposals[task].append(p)
    return np.stack(proposals), {
        "removed": removed,
        "validation_mse": best.cpu().tolist(),
    }


def run(root, case, condition, device, rounds):
    torch.set_num_threads(2)
    started = time.monotonic()
    sizes = load_sizes(root, case)
    dimensions, hidden, outputs = [sizes[k] for k in ("inputs", "hidden", "outputs")]
    folder = root / "dense" / f"{case}_c{condition}"
    source = dict(np.load(folder / "polished.npz"))
    scores = source["validation_mse"]
    parameters = source["parameters"][np.arange(len(scores)), scores.argmin(1)]
    fixed = dict(np.load(root / "public" / f"{case}_c{condition}" / "fixed_train.npz"))
    selection = dict(
        np.load(root / "public" / f"{case}_c{condition}" / "selection.npz")
    )
    tasks = len(parameters)
    x, y = [
        np.stack([fixed[f"{name}_{t}"] for t in range(tasks)]) for name in ("x", "y")
    ]
    vx, vy = [
        np.stack([selection[f"{name}_{t}"] for t in range(tasks)])
        for name in ("x", "y")
    ]
    trace, all_parameters = [], [parameters.copy()]
    train_histories, validation_histories = [], []
    for iteration in range(rounds):
        proposed, search_trace = replacements(
            parameters,
            x,
            y,
            vx,
            vy,
            dimensions,
            hidden,
            outputs,
            device,
            3360007 + iteration * 1009,
        )
        keep = proposed.shape[1]
        packed = [np.repeat(a, keep, axis=0) for a in (x, y, vx, vy)]
        fitted, score, training, validation = polish(
            proposed.reshape(tasks * keep, -1),
            *packed,
            dimensions,
            hidden,
            outputs,
            device,
            32,
        )
        train_histories.append(training)
        validation_histories.append(validation)
        fitted = fitted.reshape(tasks, keep, -1)
        score = score.reshape(tasks, keep)
        chosen = fitted[np.arange(tasks), score.argmin(1)]
        accepted, gains, errors = [], [], []
        with torch.no_grad():
            tvx = torch.tensor(vx, device=device, dtype=torch.float64)
            old_pred = (
                values(
                    tvx,
                    torch.tensor(parameters, device=device),
                    dimensions,
                    hidden,
                    outputs,
                )
                .cpu()
                .numpy()
            )
            new_pred = (
                values(
                    tvx,
                    torch.tensor(chosen, device=device),
                    dimensions,
                    hidden,
                    outputs,
                )
                .cpu()
                .numpy()
            )
        for task in range(tasks):
            difference = (
                (vy[task] - old_pred[task]) ** 2 - (vy[task] - new_pred[task]) ** 2
            ).mean(1)
            gain = float(difference.mean())
            standard_error = float(
                np.std(difference, ddof=1) / math.sqrt(len(difference))
            )
            accept = gain > 2 * standard_error + 1e-12 * max(float(np.var(vy[task])), 1)
            accepted.append(accept)
            gains.append(gain)
            errors.append(standard_error)
            if accept:
                parameters[task] = chosen[task]
        trace.append(
            {
                "round": iteration,
                "accepted": accepted,
                "gain": gains,
                "paired_standard_error": errors,
                "search": search_trace,
                "dense_validation_mse": score.min(1).tolist(),
            }
        )
        all_parameters.append(parameters.copy())
        print(
            json.dumps(
                {
                    "case": case,
                    "round": iteration,
                    "accepted": accepted,
                    "validation_mse": score.min(1).tolist(),
                }
            ),
            flush=True,
        )
    np.savez_compressed(
        folder / "repaired.npz",
        parameters=parameters,
        round_parameters=np.stack(all_parameters),
        train_history=np.stack(train_histories),
        validation_history=np.stack(validation_histories),
    )
    save_json(
        folder / "repaired.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "seconds": time.monotonic() - started,
            "trace": trace,
            "teacher_access": False,
            "extra_observations": 0,
            "sha256": digest(folder / "repaired.npz"),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--condition", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    run(
        args.root.resolve(),
        args.case,
        args.condition,
        torch.device(args.device),
        args.rounds,
    )


if __name__ == "__main__":
    main()
