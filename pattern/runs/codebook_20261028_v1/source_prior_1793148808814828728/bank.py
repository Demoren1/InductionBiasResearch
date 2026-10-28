"""Collect full Pattern solutions and infer tying from their weight profiles."""

import math

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional


def predictions(parameters, x):
    weights, bias, readout, offset = parameters
    hidden = (x @ weights + bias[:, None]).relu()
    return (hidden * readout[:, None]).sum(-1) + offset[:, None]


def scores(predicted, target, balanced=False):
    loss = functional.binary_cross_entropy_with_logits(
        predicted, target, reduction="none"
    )
    if balanced:
        p = target.mean(1, keepdim=True).clamp(1e-6, 1 - 1e-6)
        loss *= target * 0.5 / p + (1 - target) * 0.5 / (1 - p)
    positive = target.bool()
    decision = predicted >= 0
    return {
        "loss": loss.mean(1),
        "accuracy": (decision == positive).float().mean(1),
        "recall_positive": (decision & positive).sum(1) / positive.sum(1).clamp_min(1),
        "recall_negative": (~decision & ~positive).sum(1)
        / (~positive).sum(1).clamp_min(1),
    }


def collect(plan, x, y, splits, output):
    spec = plan["bank"]
    networks, tasks = spec["networks_per_task"], len(plan["source_patterns"])
    count = networks * tasks
    torch.manual_seed(plan["seed"])
    initial = [
        torch.randn(count, 11, 8, device=x.device) * 0.1,
        torch.zeros(count, 8, device=x.device),
        torch.randn(count, 8, device=x.device) * 0.1,
        torch.zeros(count, device=x.device),
    ]
    labels = y.repeat_interleave(networks, 0)
    support_y = labels[:, splits["support"]]
    query_y = labels[:, splits["query"]]
    mask = torch.ones_like(initial[0])
    histories = []
    for round_id, keep in enumerate(spec["active_edges"]):
        if round_id:
            ranking = fitted[0].abs().masked_fill(mask == 0, -torch.inf).flatten(1)
            selected = ranking.topk(keep, dim=1).indices
            mask = torch.zeros_like(ranking).scatter_(1, selected, 1).reshape_as(mask)
        parameters = [p.detach().clone().requires_grad_() for p in initial]
        with torch.no_grad():
            parameters[0].mul_(mask)
        optimizer = torch.optim.Adam(parameters, lr=spec["learning_rate"])
        best = [p.detach().clone() for p in parameters]
        best_loss = torch.full((count,), float("inf"), device=x.device)
        for step in range(spec["steps_per_round"] + 1):
            if step % spec["checkpoint_every"] == 0:
                with torch.no_grad():
                    query = scores(
                        predictions(parameters, x[splits["query"]]),
                        query_y,
                        balanced=True,
                    )
                    improved = query["loss"] < best_loss
                    best_loss[improved] = query["loss"][improved]
                    for stored, current in zip(best, parameters):
                        stored[improved] = current[improved]
            if step % spec["log_every"] == 0:
                with torch.no_grad():
                    training = scores(
                        predictions(parameters, x[splits["support"]]),
                        support_y,
                        balanced=True,
                    )
                histories.append(
                    {
                        "round": round_id,
                        "active_edges": keep,
                        "step": step,
                        "training": {
                            key: value.cpu().tolist() for key, value in training.items()
                        },
                        "query": {
                            key: value.cpu().tolist() for key, value in query.items()
                        },
                    }
                )
            if step == spec["steps_per_round"]:
                break
            predicted = predictions(parameters, x[splits["support"]])
            loss = scores(predicted, support_y, balanced=True)["loss"]
            l2 = sum(
                p.square().flatten(1).sum(1) if p.ndim > 1 else p.square()
                for p in parameters
            )
            objective = loss + spec["l2"] * 0.5 * l2
            optimizer.zero_grad(set_to_none=True)
            objective.sum().backward()
            parameters[0].grad *= mask
            optimizer.step()
            with torch.no_grad():
                parameters[0].mul_(mask)
        fitted = best
        print(
            f"BANK round={round_id} edges={keep} mean_balanced_query_bce={float(best_loss.mean()):.4f}",
            flush=True,
        )
    weight = fitted[0].clone().requires_grad_()
    gradient = (
        torch.autograd.grad(
            scores(
                predictions([weight, *fitted[1:]], x[splits["support"]]),
                support_y,
                balanced=True,
            )["loss"].sum(),
            weight,
        )[0]
        * mask
    )
    with torch.no_grad():
        activated = (
            (x[splits["probe"]] @ fitted[0] + fitted[1][:, None] > 0).float().mean(1)
        )
        functional_map = (fitted[0] * fitted[2][:, None]).abs() * activated[:, None]
        fallback = fitted[0].abs()
        mass = torch.where(
            functional_map.sum(1, keepdim=True) > 1e-8, functional_map, fallback
        )
        centers = (mass * torch.arange(11, device=x.device)[None, :, None]).sum(
            1
        ) / mass.sum(1).clamp_min(1e-8)
        order = centers.argsort(dim=1, stable=True)
        canonical = [
            fitted[0].gather(2, order[:, None].expand(-1, 11, -1)),
            fitted[1].gather(1, order),
            fitted[2].gather(1, order),
            fitted[3],
        ]
        mask = mask.gather(2, order[:, None].expand(-1, 11, -1))
        gradient = gradient.gather(2, order[:, None].expand(-1, 11, -1))
        functional_map = functional_map.gather(2, order[:, None].expand(-1, 11, -1))
        features = torch.stack([canonical[0], gradient, functional_map], -1)
        features /= features.abs().amax((1, 2), keepdim=True).clamp_min(1e-8)
        final = scores(
            predictions(canonical, x[splits["query"]]), query_y, balanced=True
        )
    arrays = {
        f"parameter_{i}": value.cpu().numpy() for i, value in enumerate(canonical)
    }
    arrays.update(
        mask=mask.cpu().numpy(),
        features=features.cpu().numpy(),
        column_order=order.cpu().numpy(),
        query_loss=final["loss"].cpu().numpy(),
        query_accuracy=final["accuracy"].cpu().numpy(),
    )
    np.savez_compressed(output / "bank.npz", **arrays)
    return arrays, histories


def infer_indices(plan, bank):
    tasks, networks = len(plan["source_patterns"]), plan["bank"]["networks_per_task"]
    train = plan["bank"]["train_networks"]
    weights = bank["parameter_0"].reshape(tasks, networks, 11, 8)
    readout = bank["parameter_2"].reshape(tasks, networks, 8)
    masks = bank["mask"].reshape(tasks, networks, 11, 8)
    quality = np.maximum(0, math.log(2) - bank["query_loss"].reshape(tasks, networks))
    quality /= np.maximum(quality[:, :train].sum(1, keepdims=True), 1e-8)
    support = (masks[:, :train] * quality[:, :train, None, None]).sum((0, 1)) / tasks
    active = np.argsort(-support.flatten(), kind="stable")[:32]
    folded = weights * np.sign(readout[:, :, None])
    folded /= np.maximum(np.linalg.norm(weights, axis=2, keepdims=True), 1e-8)
    profiles = (
        (folded[:, :train] * quality[:, :train, None, None]).sum(1).reshape(tasks, 88).T
    )
    profiles /= np.maximum(np.linalg.norm(profiles, axis=1, keepdims=True), 1e-8)
    selected = profiles[active]
    rng = np.random.default_rng(plan["seed"] + 1)
    best = None
    for _ in range(plan["extraction"]["cluster_starts"]):
        centers = selected[rng.choice(32, 4, replace=False)].copy()
        for _ in range(plan["extraction"]["cluster_iterations"]):
            distance = ((selected[:, None] - centers[None]) ** 2).sum(-1)
            rows, slots = linear_sum_assignment(np.repeat(distance, 8, axis=1))
            labels = np.empty(32, dtype=int)
            labels[rows] = slots // 8
            centers = np.array([selected[labels == k].mean(0) for k in range(4)])
            centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-8)
        error = float(((selected - centers[labels]) ** 2).sum())
        if best is None or error < best[0]:
            best = error, labels.copy(), centers.copy()
    _, labels, centers = best
    inferred = np.zeros(88, dtype=np.int64)
    inferred[active] = labels + 1
    all_labels = ((profiles[:, None] - centers[None]) ** 2).sum(-1).argmin(1) + 1
    targets = (
        bank["mask"].astype(np.int64).reshape(-1, 88) * all_labels[None]
    ).reshape(-1, 11, 8)
    return (
        inferred.reshape(11, 8),
        targets,
        {"support_frequency": support, "profiles": profiles, "centers": centers},
    )
