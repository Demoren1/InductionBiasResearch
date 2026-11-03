"""Fit independent dense source banks and extract their sharing patterns."""

import json
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from experiments.shared_codebook_mlp_v1.model import predict


def scalar_kmeans(weights, groups, restarts=8, iterations=100):
    values = weights.flatten(1).repeat_interleave(restarts, 0)
    count, positions = values.shape
    rows = torch.arange(count, device=values.device)
    centers = torch.empty(count, groups, device=values.device)
    first = torch.randint(positions, (count,), device=values.device)
    centers[:, 0] = values[rows, first]
    nearest = (values - centers[:, :1]).square()
    for label in range(1, groups):
        choice = torch.multinomial(nearest.clamp_min(1e-20), 1)[:, 0]
        centers[:, label] = values[rows, choice]
        nearest = torch.minimum(
            nearest, (values - centers[:, label : label + 1]).square()
        )
    for _ in range(iterations):
        labels = (values[:, :, None] - centers[:, None]).square().argmin(2)
        sums = torch.zeros_like(centers).scatter_add_(1, labels, values)
        sizes = torch.zeros_like(centers).scatter_add_(
            1, labels, torch.ones_like(values)
        )
        centers = torch.where(sizes > 0, sums / sizes.clamp_min(1), centers)
    distances = (values[:, :, None] - centers[:, None]).square()
    labels = distances.argmin(2)
    error = distances.min(2).values.sum(1).reshape(-1, restarts)
    choice = error.argmin(1)
    return labels.reshape(-1, restarts, positions)[
        torch.arange(len(weights), device=weights.device), choice
    ].reshape(weights.shape)


def align_indices(indices, reference, groups):
    dimensions, hidden = reference.shape
    signature = (indices[:, None] == indices[None, :]).reshape(-1, hidden)
    target = (reference[:, None] == reference[None, :]).reshape(-1, hidden)
    cost = np.not_equal(signature[:, :, None], target[:, None, :]).sum(0)
    old, new = linear_sum_assignment(cost)
    order = np.empty(hidden, dtype=int)
    order[new] = old
    rearranged = indices[:, order]
    votes = np.zeros((groups, groups), dtype=int)
    np.add.at(votes, (rearranged.flatten(), reference.flatten()), 1)
    old, new = linear_sum_assignment(-votes)
    labels = np.empty(groups, dtype=int)
    labels[old] = new
    return labels[rearranged], order


def fit_source(plan, seed, output, device):
    started = time.perf_counter()
    output.mkdir()
    torch.manual_seed(seed)
    dimensions, hidden, groups = (
        plan["inputs"], plan["hidden"], plan["groups"]
    )
    tasks, starts = plan["source_tasks"], plan["source_starts"]
    support, query = plan["source_support"], plan["source_query"]
    teacher_i = torch.randperm(dimensions * hidden, device=device)
    teacher_i = (teacher_i % groups).reshape(dimensions, hidden)
    codes = torch.rand(tasks, groups, device=device) * 2 - 1
    inputs = torch.randn(support + query, dimensions, device=device)
    labels = predict(inputs, codes[:, teacher_i], "tanh")
    target = labels.repeat_interleave(starts, 0)
    scale = target[:, :support].var(1, unbiased=False).clamp_min(1e-8)
    torch.manual_seed(seed + 1)
    weights = (
        torch.randn(tasks * starts, dimensions, hidden, device=device)
        * plan["initial_std"]
    ).requires_grad_()
    optimizer = torch.optim.Adam([weights], lr=plan["source_lr"])
    best_weights = weights.detach().clone()
    best_loss = torch.full((tasks * starts,), torch.inf, device=device)
    history = []
    for step in range(plan["source_steps"] + 1):
        if step % 128 == 0 or step == plan["source_steps"]:
            with torch.no_grad():
                query_prediction = predict(inputs[support:], weights, "tanh")
                error = (
                    (query_prediction - target[:, support:]).square().mean(1)
                    / scale
                )
                train_prediction = predict(inputs[:1024], weights, "tanh")
                train_error = (
                    (train_prediction - target[:, :1024]).square().mean(1)
                    / scale
                )
                positive = target[:, support:] > 0
                recall = ((query_prediction > 0) & positive).sum(1)
                recall = recall / positive.sum(1).clamp_min(1)
                better = error < best_loss
                best_loss[better] = error[better]
                best_weights[better] = weights[better]
                history.append({
                    "step": step,
                    "train_nmse_first_1024": train_error.cpu().tolist(),
                    "query_nmse": error.cpu().tolist(),
                    "query_positive_recall": recall.cpu().tolist(),
                })
            if step % 2048 == 0:
                print(json.dumps({
                    "stage": "source", "bank": output.name, "step": step,
                    "best_query_nmse": float(
                        best_loss.reshape(tasks, starts).min(1).values.mean()
                    ),
                }), flush=True)
        if step == plan["source_steps"]:
            break
        ids = torch.randint(support, (plan["source_batch"],), device=device)
        prediction = predict(inputs[ids], weights, "tanh")
        loss = ((prediction - target[:, ids]).square().mean(1) / scale).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.manual_seed(seed + 2)
    raw = scalar_kmeans(best_weights, groups).cpu().numpy()
    quality = best_loss.cpu().numpy()
    reference = raw[int(quality.argmin())]
    aligned, orders = zip(*[
        align_indices(matrix, reference, groups) for matrix in raw
    ])
    aligned = np.stack(aligned)
    selected = np.arange(tasks) * starts
    selected += quality.reshape(tasks, starts).argmin(1)
    votes = np.stack([
        (aligned[selected] == label).sum(0) for label in range(groups)
    ])
    consensus = votes.argmax(0)
    if len(np.unique(consensus)) != groups:
        raise RuntimeError("Source consensus lost a group; protocol needs review")
    np.savez_compressed(
        output / "bank.npz", weights=best_weights.cpu().numpy(),
        quality=quality, raw_indices=raw, aligned_indices=aligned,
        selected=selected, consensus=consensus, column_orders=np.stack(orders),
    )
    np.savez_compressed(
        output / "source_data.npz", inputs=inputs.cpu().numpy(),
        labels=labels.cpu().numpy(), codes=codes.cpu().numpy(),
        teacher_i=teacher_i.cpu().numpy(),
    )
    (output / "source_history.json").write_text(json.dumps(history))
    # Teacher access below is diagnostic, after consensus has been saved.
    truth = teacher_i.cpu().numpy()
    recovered, _ = align_indices(consensus, truth, groups)
    diagnostic = {
        "seed": seed,
        "source_best_mean_nmse": float(quality[selected].mean()),
        "correct_assignments": int((recovered == truth).sum()),
        "positions": dimensions * hidden,
        "occupied_groups": int(len(np.unique(consensus))),
        "group_sizes": np.bincount(consensus.flatten()).tolist(),
        "seconds": time.perf_counter() - started,
    }
    (output / "source_diagnostic.json").write_text(
        json.dumps(diagnostic, indent=2)
    )
    print(json.dumps({"stage": "source_done", **diagnostic}), flush=True)
    return consensus, teacher_i


def make_tasks(plan, teacher_i, tasks, seed, include_test, device):
    torch.manual_seed(seed)
    dimensions, hidden = plan["inputs"], plan["hidden"]
    positions = dimensions * hidden
    codes = torch.rand(tasks, plan["groups"], device=device) * 2 - 1
    base_weights = codes[:, teacher_i].flatten(1)
    residual = torch.randn_like(base_weights)
    assignment = teacher_i.flatten()[None].expand(tasks, -1)
    sizes = torch.bincount(teacher_i.flatten()).to(base_weights.dtype)
    means = torch.zeros_like(codes).scatter_add_(1, assignment, residual)
    means /= sizes
    residual -= means.gather(1, assignment)
    residual *= base_weights.norm(dim=1, keepdim=True)
    residual /= residual.norm(dim=1, keepdim=True)
    ratios = torch.tensor(plan["residual_ratios"], device=device)
    weights = base_weights[None] + ratios[:, None, None] * residual[None]
    weights = weights.reshape(-1, dimensions, hidden)
    samples = plan["support"] + plan["query"]
    if include_test:
        samples += plan["test"]
    inputs = torch.randn(tasks, samples, dimensions, device=device)
    inputs = inputs.repeat(len(ratios), 1, 1)
    labels = torch.bmm(inputs, weights).tanh().sum(2) / hidden**0.5
    support_end = plan["support"]
    query_end = support_end + plan["query"]
    data = {
        "train_x": inputs[:, :support_end],
        "train_y": labels[:, :support_end],
        "query_x": inputs[:, support_end:query_end],
        "query_y": labels[:, support_end:query_end],
        "ratios": ratios.repeat_interleave(tasks),
        "task_ids": torch.arange(tasks, device=device).repeat(len(ratios)),
        "teacher_weights": weights,
        "base_codes": codes,
        "residual_direction": residual.reshape(tasks, dimensions, hidden),
    }
    if include_test:
        data["test_x"] = inputs[:, query_end:]
        data["test_y"] = labels[:, query_end:]
    return data
