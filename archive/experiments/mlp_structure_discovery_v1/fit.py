"""Batched MLP fitting with arbitrary, externally supplied weight partitions."""

import gzip
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def read_observations(path, task_ids, support_size=None):
    """Only read learner-visible support/query observations from an archive."""
    with np.load(path) as archive:
        data = {
            name: archive[name][task_ids]
            for name in ("support_x", "support_y", "query_x", "query_y")
        }
    if support_size is not None:
        data["support_x"] = data["support_x"][:, :support_size]
        data["support_y"] = data["support_y"][:, :support_size]
    return data


def untied_indices(mask):
    shape = mask.shape
    labels = np.arange(1, shape[-2] * shape[-1] + 1).reshape(shape[-2:])
    return (mask * labels).astype(np.int64)


def predict(inputs, parameters):
    weights, bias, readout, offset = parameters
    hidden = (torch.bmm(inputs, weights) + bias[:, None]).relu()
    return (hidden * readout[:, None]).sum(-1) + offset[:, None]


def evaluate(data_x, data_y, parameters, task_ids, device):
    loss = torch.zeros(len(task_ids), device=device)
    accuracy = torch.zeros_like(loss)
    samples = data_y.shape[1]
    with torch.no_grad():
        for start in range(0, samples, 512):
            x = torch.as_tensor(data_x[:, start : start + 512], device=device)
            y = torch.as_tensor(data_y[:, start : start + 512], device=device)
            logits = predict(x[task_ids], parameters)
            target = y[task_ids]
            loss += F.binary_cross_entropy_with_logits(
                logits, target, reduction="none"
            ).sum(-1)
            accuracy += ((logits >= 0) == target.bool()).sum(-1)
    return loss / samples, accuracy / samples


def fit_models(
    data,
    indices,
    rows,
    plan,
    output,
    device,
    steps=None,
    initial=None,
    save_parameters=True,
):
    """Each row specifies task, restart, learning rate and weight decay."""
    started = time.perf_counter()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count, dimensions, hidden = indices.shape
    if count != len(rows) or dimensions != data["support_x"].shape[-1]:
        raise ValueError("Model rows and observed input dimensions must match")
    index = torch.as_tensor(indices, device=device, dtype=torch.long)
    task_ids = torch.tensor([row["task"] for row in rows], device=device)
    total_steps = plan["transfer_steps"] if steps is None else steps
    codes = int(index.max())
    values, biases, readouts, offsets = [], [], [], []
    for item, row in zip(indices, rows):
        rng = np.random.default_rng(
            plan["seed"] + 1009 * row["task"] + 100003 * row["restart"]
        )
        fan_in = max(1.0, np.count_nonzero(item) / hidden)
        values.append(rng.normal(size=codes) * math.sqrt(2.0 / fan_in))
        biases.append(np.zeros(hidden))
        readouts.append(rng.normal(size=hidden) / math.sqrt(hidden))
        offsets.append(0.0)
    if initial is not None:
        old_weights, old_bias, old_readout, old_offset = initial
        values = np.zeros((count, codes), dtype=np.float32)
        for row_id, item in enumerate(indices):
            sums = np.bincount(
                item.ravel(), weights=old_weights[row_id].ravel(), minlength=codes + 1
            )
            sizes = np.bincount(item.ravel(), minlength=codes + 1)
            values[row_id] = sums[1:] / np.maximum(sizes[1:], 1)
        biases, readouts, offsets = old_bias, old_readout, old_offset
    parameters = [
        torch.tensor(
            np.asarray(value), dtype=torch.float32, device=device
        ).requires_grad_()
        for value in (values, biases, readouts, offsets)
    ]
    first = [torch.zeros_like(value) for value in parameters]
    second = [torch.zeros_like(value) for value in parameters]
    rates = torch.tensor([row["lr"] for row in rows], device=device)
    decays = torch.tensor([row["weight_decay"] for row in rows], device=device)
    support_x = torch.as_tensor(data["support_x"], device=device)
    support_y = torch.as_tensor(data["support_y"], device=device)
    generator = torch.Generator(device=device).manual_seed(plan["seed"] + 701)
    best_loss = torch.full((count,), torch.inf, device=device)
    best_step = torch.zeros(count, device=device, dtype=torch.long)
    best = None
    history = []

    def materialize():
        weights = F.pad(parameters[0], (1, 0)).gather(1, index.flatten(1))
        return [weights.reshape(count, dimensions, hidden), *parameters[1:]]

    for step in range(total_steps + 1):
        if step % plan["checkpoint_every"] == 0 or step == total_steps:
            with torch.no_grad():
                current = materialize()
                query_loss, query_accuracy = evaluate(
                    data["query_x"], data["query_y"], current, task_ids, device
                )
                if best is None:
                    best = [value.detach().clone() for value in current]
                improved = query_loss < best_loss
                best_loss[improved] = query_loss[improved]
                best_step[improved] = step
                for stored, value in zip(best, current):
                    stored[improved] = value[improved]
        if step == total_steps:
            break
        samples = torch.randint(
            support_y.shape[1],
            (len(support_y), plan["batch_size"]),
            generator=generator,
            device=device,
        )
        x = support_x[task_ids[:, None], samples[task_ids]]
        y = support_y[task_ids[:, None], samples[task_ids]]
        logits = predict(x, materialize())
        training = F.binary_cross_entropy_with_logits(logits, y, reduction="none").mean(
            -1
        )
        if step % plan["log_every"] == 0:
            history.append(
                {
                    "step": step,
                    "train_loss": training.detach().cpu().tolist(),
                    "query_loss": query_loss.cpu().tolist(),
                    "query_accuracy": query_accuracy.cpu().tolist(),
                }
            )
            print(
                output.name,
                step,
                "best_query",
                round(float(best_loss.mean()), 5),
                "seconds",
                round(time.perf_counter() - started, 1),
                flush=True,
            )
        training.sum().backward()
        with torch.no_grad():
            for parameter, mean, variance in zip(parameters, first, second):
                shape = (-1,) + (1,) * (parameter.ndim - 1)
                gradient = parameter.grad + decays.reshape(shape) * parameter
                mean.mul_(0.9).add_(gradient, alpha=0.1)
                variance.mul_(0.999).addcmul_(gradient, gradient, value=0.001)
                corrected = mean / (1 - 0.9 ** (step + 1))
                scale = (variance / (1 - 0.999 ** (step + 1))).sqrt() + 1e-8
                parameter.add_(-rates.reshape(shape) * corrected / scale)
                parameter.grad = None
    query_loss, query_accuracy = evaluate(
        data["query_x"], data["query_y"], best, task_ids, device
    )
    arrays = {
        "query_loss": query_loss.cpu().numpy(),
        "query_accuracy": query_accuracy.cpu().numpy(),
        "step": best_step.cpu().numpy(),
    }
    if save_parameters:
        arrays.update(
            {f"parameter_{i}": value.cpu().numpy() for i, value in enumerate(best)}
        )
        arrays["indices"] = indices.astype(np.int32)
    np.savez_compressed(output.with_suffix(".npz"), **arrays)
    with gzip.open(output.parent / (output.name + "_history.json.gz"), "wt") as stream:
        json.dump(history, stream)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "rows": rows,
                "steps": total_steps,
                "seconds": time.perf_counter() - started,
            },
            indent=2,
        )
        + "\n"
    )
    return arrays
