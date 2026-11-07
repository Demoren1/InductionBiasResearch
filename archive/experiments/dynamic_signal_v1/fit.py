import gzip
import json
import math
import time

import numpy as np
import torch
from torch.nn import functional as F


def model_indices(permutation, blocks, width, device):
    indices = torch.zeros(blocks * width, 2 * blocks, dtype=torch.long, device=device)
    for block in range(blocks):
        rows = torch.tensor(
            permutation[block * width : (block + 1) * width], device=device
        )
        indices[rows, 2 * block] = torch.arange(1, width + 1, device=device)
        indices[rows, 2 * block + 1] = torch.arange(
            width + 1, 2 * width + 1, device=device
        )
    return indices


def predictions(parameters, inputs):
    weights, bias, readout, offset = parameters
    return ((torch.bmm(inputs, weights) + bias[:, None]).relu() * readout[:, None]).sum(
        -1
    ) + offset[:, None]


def metrics(logits, labels):
    positive = labels.bool()
    decision = logits >= 0
    return {
        "positive_count": positive.sum(-1),
        "negative_count": (~positive).sum(-1),
        "loss": F.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        ).mean(-1),
        "accuracy": (decision == positive).float().mean(-1),
        "recall_positive": (decision & positive).sum(-1)
        / positive.sum(-1).clamp_min(1),
        "recall_negative": (~decision & ~positive).sum(-1)
        / (~positive).sum(-1).clamp_min(1),
    }


def combine_metrics(parts):
    result = {}
    for name in parts[0]:
        values = torch.stack([part[name] for part in parts])
        if name.endswith("_count"):
            result[name] = values.sum(0)
        elif name.startswith("recall_"):
            count_name = (
                "positive_count" if name == "recall_positive" else "negative_count"
            )
            counts = torch.stack([part[count_name] for part in parts])
            result[name] = (values * counts).sum(0) / counts.sum(0).clamp_min(1)
        else:
            counts = torch.stack(
                [part["positive_count"] + part["negative_count"] for part in parts]
            )
            result[name] = (values * counts).sum(0) / counts.sum(0).clamp_min(1)
    return result


class BatchedOptimizer:
    def __init__(self, parameters, options, device, total_steps):
        self.parameters = parameters
        self.rate = torch.tensor([row["lr"] for row in options], device=device)
        self.decay = torch.tensor(
            [row["weight_decay"] for row in options], device=device
        )
        self.momentum = torch.tensor(
            [row["momentum"] for row in options], device=device
        )
        self.kind = options[0]["optimizer"]
        self.first = [torch.zeros_like(value) for value in parameters]
        self.second = [torch.zeros_like(value) for value in parameters]
        self.step_id = 0
        self.total_steps = total_steps
        self.cosine = torch.tensor(
            [row.get("schedule", "constant") == "cosine" for row in options],
            device=device,
        )

    @torch.no_grad()
    def step(self):
        self.step_id += 1
        for parameter, first, second in zip(self.parameters, self.first, self.second):
            if parameter.grad is None:
                continue
            shape = (-1,) + (1,) * (parameter.ndim - 1)
            gradient = parameter.grad + self.decay.reshape(shape) * parameter
            fraction = (self.step_id - 1) / self.total_steps
            multiplier = torch.where(
                self.cosine,
                0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * fraction)),
                1.0,
            )
            rate = (self.rate * multiplier).reshape(shape)
            if self.kind == "adam":
                first.mul_(0.9).add_(gradient, alpha=0.1)
                second.mul_(0.999).addcmul_(gradient, gradient, value=0.001)
                corrected = first / (1 - 0.9**self.step_id)
                variance = second / (1 - 0.999**self.step_id)
                parameter.add_(-rate * corrected / (variance.sqrt() + 1e-8))
            else:
                first.mul_(self.momentum.reshape(shape)).add_(gradient)
                parameter.add_(-rate * first)
            parameter.grad = None


def fit(
    plan,
    data,
    method,
    configs,
    sizes,
    restarts,
    device,
    output,
    indices=None,
    initial=None,
    steps=None,
    mask=None,
):
    started = time.perf_counter()
    tasks = len(data["support_y"])
    rows = [
        (config_id, task, restart)
        for config_id in range(len(configs))
        for task in range(tasks)
        for restart in range(restarts)
    ]
    task_ids = torch.tensor([row[1] for row in rows], device=device)
    options = [configs[row[0]] for row in rows]
    count = len(rows)
    dimensions, hidden = plan["blocks"] * plan["block_size"], plan["hidden"]
    tensor = {key: torch.tensor(value, device=device) for key, value in data.items()}
    torch.manual_seed(plan["seed"] + 400)
    scales = torch.tensor(
        [option["initial_scale"] for option in options], device=device
    )
    if method == "oracle" or indices is not None:
        code = torch.randn(count, 2 * plan["block_size"], device=device)
        code = (
            F.normalize(code.reshape(count, 2, -1), dim=-1).reshape(count, -1)
            * scales[:, None]
        )
        bias = torch.zeros(count, hidden, device=device)
        readout = torch.randn(count, hidden, device=device) / math.sqrt(hidden)
        offset = torch.zeros(count, device=device)
        parameters = [code, bias, readout, offset]
        if indices.ndim == 2:
            indices = indices.expand(count, -1, -1)
    elif method == "linear":
        parameters = [
            torch.randn(count, dimensions, 1, device=device)
            * scales[:, None, None]
            / math.sqrt(dimensions),
            torch.zeros(count, 1, device=device),
            torch.ones(count, 1, device=device),
            torch.zeros(count, device=device),
        ]
    else:
        parameters = [
            torch.randn(count, dimensions, hidden, device=device)
            * math.sqrt(2 / dimensions)
            * scales[:, None, None],
            torch.zeros(count, hidden, device=device),
            torch.randn(count, hidden, device=device) / math.sqrt(hidden),
            torch.zeros(count, device=device),
        ]
    if initial is not None:
        parameters = [torch.tensor(value, device=device).clone() for value in initial]
    for value in parameters:
        value.requires_grad_()
    total_steps = plan["steps"] if steps is None else steps
    optimizer = BatchedOptimizer(parameters, options, device, total_steps)

    def current():
        if indices is not None:
            weights = (
                F.pad(parameters[0], (1, 0))
                .gather(1, indices.flatten(1))
                .reshape(count, dimensions, hidden)
            )
        elif mask is not None:
            weights = parameters[0] * mask
        else:
            weights = parameters[0]
        return [weights, *parameters[1:]]

    def predict(values, inputs):
        if method == "linear":
            return torch.bmm(inputs, values[0]).squeeze(-1) + values[3][:, None]
        return predictions(values, inputs)

    best_loss = torch.full((count,), torch.inf, device=device)
    best = [value.detach().clone() for value in current()]
    best_step = torch.zeros(count, dtype=torch.long, device=device)
    history = []
    sharing = torch.tensor(
        [option.get("sharing_penalty", 0.0) for option in options], device=device
    )
    sparse_penalty = torch.tensor(
        [option.get("l1_penalty", 0.0) for option in options], device=device
    )
    regularized = bool(sharing.any() or sparse_penalty.any())
    dimensions_first = dimensions * hidden
    sharing_rank = (
        2 * torch.arange(dimensions_first, device=device) - dimensions_first + 1
    ) / (dimensions_first - 1)
    bias_rank = (2 * torch.arange(hidden, device=device) - hidden + 1) / (hidden - 1)
    generator = torch.Generator(device=device).manual_seed(plan["seed"] + 401)
    for step in range(total_steps + 1):
        if step % plan["checkpoint_every"] == 0:
            with torch.no_grad():
                values = current()
                query_parts = []
                for start in range(0, tensor["query_y"].shape[1], 512):
                    ids = slice(start, start + 512)
                    query_parts.append(
                        metrics(
                            predict(values, tensor["query_x"][task_ids, ids]),
                            tensor["query_y"][task_ids, ids],
                        )
                    )
                query = combine_metrics(query_parts)
                better = query["loss"] < best_loss
                best_loss[better] = query["loss"][better]
                best_step[better] = step
                for stored, value in zip(best, values):
                    stored[better] = value[better]
        ids = torch.randint(
            sizes, (tasks, plan["batch_size"]), generator=generator, device=device
        )
        inputs = tensor["support_x"][task_ids[:, None], ids[task_ids]]
        target = tensor["support_y"][task_ids[:, None], ids[task_ids]]
        training = metrics(predict(current(), inputs), target)
        if step % plan["log_every"] == 0:
            history.append(
                {
                    "step": step,
                    "train": {
                        key: value.detach().cpu().tolist()
                        for key, value in training.items()
                    },
                    "query": {
                        key: value.cpu().tolist() for key, value in query.items()
                    },
                }
            )
            print("FIT", method, step, "query", float(best_loss.mean()), flush=True)
        if step == total_steps:
            break
        objective = training["loss"]
        if regularized:
            # Exact sorted expression for sum_{i>j}|w_i-w_j|/(d-1).
            # Same subgradient objective as the authors' FC1 baseline, applied
            # separately to first-layer weights and biases.
            ordered = parameters[0].flatten(1).sort(-1).values
            ordered_bias = parameters[1].sort(-1).values
            objective = objective + sharing * (
                (ordered * sharing_rank).sum(-1) + (ordered_bias * bias_rank).sum(-1)
            )
            objective = objective + sparse_penalty * (
                parameters[0].abs().flatten(1).sum(-1) + parameters[1].abs().sum(-1)
            )
        objective.sum().backward()
        optimizer.step()
    arrays = {f"parameter_{i}": value.cpu().numpy() for i, value in enumerate(best)}
    arrays.update(
        config_ids=np.array([row[0] for row in rows]),
        task_ids=np.array([row[1] for row in rows]),
        restart_ids=np.array([row[2] for row in rows]),
        step=best_step.cpu().numpy(),
    )
    with torch.no_grad():
        for split in ["query", "test"]:
            parts = []
            for start in range(0, tensor[split + "_y"].shape[1], 512):
                ids = slice(start, start + 512)
                parts.append(
                    metrics(
                        predict(best, tensor[split + "_x"][task_ids, ids]),
                        tensor[split + "_y"][task_ids, ids],
                    )
                )
            arrays.update(
                {
                    split + "_" + key: value.cpu().numpy()
                    for key, value in combine_metrics(parts).items()
                }
            )
    if indices is not None:
        arrays["indices"] = indices.cpu().numpy()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix(".npz"), **arrays)
    with gzip.open(output.parent / (output.name + "_history.json.gz"), "wt") as stream:
        json.dump(history, stream)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "configs": configs,
                "rows": rows,
                "support_size": sizes,
                "steps": total_steps,
                "seconds": time.perf_counter() - started,
                "model_count": count,
                "gpu": torch.cuda.get_device_name(device),
            },
            indent=2,
        )
        + "\n"
    )
    return arrays
