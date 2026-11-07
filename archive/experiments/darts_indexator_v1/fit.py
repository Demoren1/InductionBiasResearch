"""Fresh numerical fits for fixed assignments and explicitly separate dense controls."""

import time

import numpy as np
import torch
from torch.nn import functional as F

from .model import BatchedAdam, balanced_loss, clipped, metrics


def predict(inputs, indices, parameters, full_head):
    weights = F.pad(parameters[:, :88], (1, 0)).gather(1, indices.flatten(1))
    weights = weights.reshape(-1, 11, 8)
    bias = torch.where(full_head[:, None], parameters[:, 88:96], parameters[:, 88:89])
    readout = torch.where(
        full_head[:, None], parameters[:, 96:104], parameters[:, 96:97]
    )
    hidden = (torch.einsum("pi,nih->nph", inputs, weights) + bias[:, None]).relu()
    return (hidden * readout[:, None]).sum(-1) + parameters[:, 104, None]


def initialization(rows, seed):
    values = []
    for row in rows:
        rng = np.random.default_rng(seed + 100003 * row["task"] + 1009 * row["restart"])
        codebook = rng.normal(size=88)
        if row["kind"] == "shared":
            codebook[:4] *= row["norm"] / max(np.linalg.norm(codebook[:4]), 1e-8)
        else:
            codebook *= row["norm"]
        bias = np.zeros(8)
        readout = rng.normal(size=8) * 0.3
        if row["kind"] != "dense_full":
            readout[0] = 1.0
        values.append(np.r_[codebook, bias, readout, 0.0])
    return np.stack(values).astype(np.float32)


@torch.no_grad()
def evaluate(inputs, target, indices, parameters, full_head, task_ids, chunk=512):
    result = []
    for left in range(0, len(parameters), chunk):
        right = left + chunk
        logits = predict(
            inputs, indices[left:right], parameters[left:right], full_head[left:right]
        )
        result.append(metrics(logits, target[task_ids[left:right]]))
    return torch.cat(result)


def fit_fixed(data, indices, rows, spec, seed, device, initial_parameters=None):
    started = time.perf_counter()
    inputs = {
        split: torch.tensor(data[split + "_x"], device=device)
        for split in ("support", "query", "selection")
    }
    target = {
        split: torch.tensor(data[split + "_y"], device=device) for split in inputs
    }
    index = torch.tensor(indices, dtype=torch.long, device=device)
    tasks = torch.tensor([row["task"] for row in rows], device=device)
    full_head = torch.tensor(
        [row["kind"] == "dense_full" for row in rows], device=device
    )
    initial = (
        initialization(rows, seed)
        if initial_parameters is None
        else np.asarray(initial_parameters, dtype=np.float32)
    )
    if initial.shape != (len(rows), 105):
        raise ValueError("Initial numerical parameters must match the fit rows")
    parameters = torch.tensor(initial, device=device).requires_grad_()
    optimizer = BatchedAdam(parameters, [row["lr"] for row in rows])
    best_parameters = parameters.detach().clone()
    best_loss = torch.full((len(rows),), torch.inf, device=device)
    best_step = torch.zeros(len(rows), dtype=torch.long, device=device)
    history_steps, training_history, query_history, recall_history = [], [], [], []
    rng = torch.Generator(device=device).manual_seed(seed + 701)
    for step in range(spec["steps"] + 1):
        if step % spec["checkpoint_every"] == 0 or step == spec["steps"]:
            with torch.no_grad():
                query = evaluate(
                    inputs["query"],
                    target["query"],
                    index,
                    parameters,
                    full_head,
                    tasks,
                )
                training = evaluate(
                    inputs["support"],
                    target["support"],
                    index,
                    parameters,
                    full_head,
                    tasks,
                )
                better = query[:, 0] < best_loss
                best_loss[better] = query[better, 0]
                best_step[better] = step
                best_parameters[better] = parameters[better]
                history_steps.append(step)
                training_history.append(training[:, 0].cpu().numpy())
                query_history.append(query[:, 0].cpu().numpy())
                recall_history.append(query[:, 2].cpu().numpy())
                if not torch.isfinite(query).all():
                    raise FloatingPointError("Non-finite fixed-structure fit")
        if step == spec["steps"]:
            break
        ids = torch.randint(
            len(inputs["support"]), (spec["batch"],), generator=rng, device=device
        )
        logits = predict(inputs["support"][ids], index, parameters, full_head)
        loss = balanced_loss(logits, target["support"][tasks[:, None], ids])
        loss = loss + spec["numeric_l2"] * parameters.square().sum(-1) / 2
        gradient = torch.autograd.grad(loss.sum(), parameters)[0]
        optimizer.step(clipped(gradient, spec["gradient_clip"], (-1,)))
    result = {
        "indices": np.asarray(indices, dtype=np.int16),
        "parameters": best_parameters.cpu().numpy(),
        "best_step": best_step.cpu().numpy(),
        "history_steps": np.array(history_steps),
        "train_history": np.stack(training_history),
        "query_history": np.stack(query_history),
        "recall_history": np.stack(recall_history),
        "seconds": np.array(time.perf_counter() - started),
    }
    with torch.no_grad():
        for split, split_inputs in inputs.items():
            result[split + "_metrics"] = (
                evaluate(
                    split_inputs,
                    target[split],
                    index,
                    best_parameters,
                    full_head,
                    tasks,
                )
                .cpu()
                .numpy()
            )
    return result
