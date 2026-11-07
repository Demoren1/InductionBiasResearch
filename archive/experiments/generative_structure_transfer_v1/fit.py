"""Fresh numeric fitting with a fixed indexator for every attempt."""

import time

import numpy as np
import torch


def fit_numeric(
    plan,
    data,
    indices,
    rates,
    optimizer,
    initial_std,
    seed,
    steps,
    pretrained=None,
):
    """Fit code values, or dense weights for an explicitly separate control."""
    device = data["train_x"].device
    tasks, starts = len(data["train_x"]), plan["starts"]
    dimensions, hidden = plan["inputs"], plan["hidden"]
    positions = dimensions * hidden
    rate_count = len(rates)
    replicas = rate_count * starts
    count = tasks * replicas
    groups = positions if indices is None else plan["groups"]
    torch.manual_seed(seed)
    initial = torch.randn(tasks, starts, groups, device=device) * initial_std
    if pretrained is not None:
        if indices is not None:
            raise ValueError("Source numeric transfer is a dense control only")
        chosen = torch.randint(len(pretrained), (tasks, starts), device=device)
        initial = pretrained[chosen].flatten(2).clone()
    values = initial[:, None].expand(-1, rate_count, -1, -1)
    values = values.reshape(count, groups).clone()
    assignment = None
    if indices is not None:
        assignment = indices[:, None].expand(-1, rate_count, -1, -1)
        assignment = assignment.reshape(count, positions).clone()
    x = data["train_x"].repeat_interleave(replicas, 0)
    y = data["train_y"].repeat_interleave(replicas, 0)
    qx = data["query_x"].repeat_interleave(replicas, 0)
    qy = data["query_y"].repeat_interleave(replicas, 0)
    scale = y.var(1, unbiased=False).clamp_min(1e-8)
    rate = torch.tensor(rates, device=device)[None, :, None]
    rate = rate.expand(tasks, -1, starts).reshape(count, 1)
    velocity, second = torch.zeros_like(values), torch.zeros_like(values)
    best_query = torch.full((count,), torch.inf, device=device)
    best_train = best_query.clone()
    best_values = values.clone()
    best_step = torch.zeros(count, dtype=torch.long, device=device)
    history = []
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(steps + 1):
        values = values.detach().requires_grad_()
        flat_weights = (
            values if assignment is None else values.gather(1, assignment)
        )
        weights = flat_weights.reshape(count, dimensions, hidden)
        prediction = torch.bmm(x, weights).tanh().sum(2) / hidden**0.5
        losses = (prediction - y).square().mean(1) / scale
        if step % plan["log_every"] == 0 or step == steps:
            with torch.no_grad():
                query_prediction = torch.bmm(qx, weights).tanh().sum(2)
                query_prediction /= hidden**0.5
                query_loss = (query_prediction - qy).square().mean(1) / scale
                positive = qy > 0
                recall = ((query_prediction > 0) & positive).sum(1)
                recall = recall / positive.sum(1).clamp_min(1)
                better = query_loss < best_query
                best_query[better] = query_loss[better]
                best_train[better] = losses[better]
                best_values[better] = values[better]
                best_step[better] = step
                history.append(
                    {
                        "step": step,
                        "train_nmse": losses.detach().cpu().tolist(),
                        "query_nmse": query_loss.cpu().tolist(),
                        "query_positive_recall": recall.cpu().tolist(),
                    }
                )
        if step == steps:
            break
        gradient = torch.autograd.grad(losses.sum(), values)[0]
        with torch.no_grad():
            if optimizer == "adam":
                velocity.mul_(0.9).add_(gradient, alpha=0.1)
                second.mul_(0.999).addcmul_(gradient, gradient, value=0.001)
                corrected = velocity / (1 - 0.9 ** (step + 1))
                denominator = (second / (1 - 0.999 ** (step + 1))).sqrt()
                values -= rate * corrected / (denominator + 1e-8)
            elif optimizer == "sgdm":
                velocity.mul_(plan["momentum"]).add_(gradient)
                values -= rate * velocity
            else:
                raise ValueError(f"Unknown numeric optimizer: {optimizer}")
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    with torch.no_grad():
        flat_weights = (
            best_values
            if assignment is None
            else best_values.gather(1, assignment)
        )
        best_weights = flat_weights.reshape(count, dimensions, hidden)
    shape = (tasks, rate_count, starts)
    result = {
        "query_nmse": best_query.cpu().numpy().reshape(shape),
        "train_nmse": best_train.cpu().numpy().reshape(shape),
        "step": best_step.cpu().numpy().reshape(shape),
        "values": best_values.cpu().numpy().reshape(*shape, groups),
        "weights": best_weights.cpu().numpy().reshape(
            *shape, dimensions, hidden
        ),
        "initial_values": initial.cpu().numpy(),
        "rates": np.array(rates),
        "initial_std": np.array(initial_std),
    }
    if indices is not None:
        result["indices"] = indices.cpu().numpy()
    timing = {
        "seconds": elapsed,
        "models": count,
        "updates_per_model": steps,
        "nonfinite_final_models": int((~torch.isfinite(values).all(1)).sum()),
        "optimizer": optimizer,
        "trainable_values_per_model": groups,
        "numeric_initialization": (
            "source_weights" if pretrained is not None else "fresh"
        ),
        "indexator_fixed": indices is not None,
    }
    return result, history, timing


@torch.no_grad()
def evaluate_attempts(plan, result, data):
    """Evaluate saved attempts only; selection continues to use query loss."""
    device = data["test_x"].device
    weights = torch.as_tensor(result["weights"][:, 0], device=device)
    tasks, starts, dimensions, hidden = weights.shape
    weights = weights.reshape(-1, dimensions, hidden)
    squared_error = torch.zeros(tasks * starts, device=device)
    correct_positive = torch.zeros_like(squared_error)
    positives = torch.zeros_like(squared_error)
    for first in range(0, plan["test"], 512):
        x = data["test_x"][:, first : first + 512].repeat_interleave(starts, 0)
        y = data["test_y"][:, first : first + 512].repeat_interleave(starts, 0)
        prediction = torch.bmm(x, weights).tanh().sum(2) / hidden**0.5
        squared_error += (prediction - y).square().sum(1)
        correct_positive += ((prediction > 0) & (y > 0)).sum(1)
        positives += (y > 0).sum(1)
    variance = data["test_y"].var(1, unbiased=False).clamp_min(1e-8)
    nmse = squared_error.reshape(tasks, starts) / plan["test"]
    nmse /= variance[:, None]
    recall = (correct_positive / positives.clamp_min(1)).reshape(tasks, starts)
    return {
        "test_nmse": nmse.cpu().numpy(),
        "test_positive_recall": recall.cpu().numpy(),
    }
