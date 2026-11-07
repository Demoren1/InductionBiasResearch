"""Matched-budget numeric fitting with gradient and random group splits."""

import time

import numpy as np
import torch


def group_statistics(indices, gradient, capacity):
    sums = torch.zeros(
        len(indices), capacity, device=gradient.device, dtype=gradient.dtype
    ).scatter_add_(1, indices, gradient)
    sizes = torch.zeros_like(sums).scatter_add_(
        1, indices, torch.ones_like(gradient)
    )
    return sums, sizes


def gradient_split(indices, gradient, groups):
    count, positions = indices.shape
    order = gradient.argsort(dim=1, stable=True)
    by_group = indices.gather(1, order).argsort(dim=1, stable=True)
    order = order.gather(1, by_group)
    labels = indices.gather(1, order)
    ordered = gradient.gather(1, order)
    sums, sizes = group_statistics(indices, gradient, groups)
    starts = sizes.cumsum(1) - sizes
    rank = torch.arange(positions, device=indices.device)[None]
    rank = rank - starts.gather(1, labels) + 1
    prefix = torch.cat(
        [torch.zeros(count, 1, device=gradient.device), ordered.cumsum(1)], 1
    )
    before = prefix.gather(1, starts.long()).gather(1, labels)
    left_sum = prefix[:, 1:] - before
    size = sizes.gather(1, labels)
    total = sums.gather(1, labels)
    contrast = left_sum - rank * total / size
    score = contrast.square() * size / (rank * (size - rank)).clamp_min(1)
    score[rank >= size] = -torch.inf
    best_score, cut = score.max(1)
    parent = labels.gather(1, cut[:, None])[:, 0]
    left_size = rank.gather(1, cut[:, None])[:, 0].long()
    move_sorted = (labels == parent[:, None]) & (rank > left_size[:, None])
    move = torch.zeros_like(move_sorted).scatter_(1, order, move_sorted)
    return parent, left_size, move, best_score


def random_split(indices, gradient, parent, left_size, generator):
    eligible = indices == parent[:, None]
    keys = torch.rand(
        indices.shape, device=indices.device, generator=generator
    )
    keys[~eligible] = 2
    order = keys.argsort(1)
    rank = torch.empty_like(order).scatter_(
        1,
        order,
        torch.arange(indices.shape[1], device=indices.device)[None].expand_as(
            order
        ),
    )
    move = eligible & (rank >= left_size[:, None])
    right_size = move.sum(1)
    total = (gradient * eligible).sum(1)
    right = (gradient * move).sum(1)
    score = (
        (right / right_size - (total - right) / left_size).square()
        * right_size
        * left_size
        / (right_size + left_size)
    )
    return move, score


def fit(
    plan,
    inputs,
    targets,
    query_x,
    query_y,
    consensus,
    method,
    rates,
    seed,
    mirrored_events=None,
):
    started = time.perf_counter()
    device = inputs.device
    tasks, starts = len(inputs), plan["starts"]
    dimensions, hidden = plan["inputs"], plan["hidden"]
    positions = dimensions * hidden
    rate_count = len(rates)
    replicas = starts * rate_count
    count = tasks * replicas
    dense = method.startswith("dense")
    groups = positions if dense else plan["groups"]
    normalized = method in {"fixed_normalized", "adaptive", "random_matched"}
    torch.manual_seed(seed)
    initial = torch.randn(tasks, starts, groups, device=device)
    initial *= plan["initial_std"]
    initial = initial[:, None].expand(-1, rate_count, -1, -1)
    values = torch.zeros(count, positions, device=device)
    values[:, :groups] = initial.reshape(count, groups)
    if dense:
        indices = torch.arange(positions, device=device)[None].repeat(count, 1)
    else:
        indices = torch.as_tensor(consensus, device=device).flatten()[None]
        indices = indices.repeat(count, 1)
    x = inputs.repeat_interleave(replicas, 0)
    y = targets.repeat_interleave(replicas, 0)
    qx = query_x.repeat_interleave(replicas, 0)
    qy = query_y.repeat_interleave(replicas, 0)
    scale = y.var(1, unbiased=False).clamp_min(1e-8)
    rate = torch.tensor(rates, device=device)[None, :, None]
    rate = rate.expand(tasks, -1, starts).reshape(count, 1)
    velocity, second = torch.zeros_like(values), torch.zeros_like(values)
    optimizer_step = 0
    schedule = dict(plan["split_schedule"])
    generator = torch.Generator(device=device).manual_seed(seed + 17)
    best = {}
    history, events = [], []
    split_seconds = 0.0
    for step in range(plan["steps"] + 1):
        weights = values.gather(1, indices).reshape(count, dimensions, hidden)
        weights = weights.detach().requires_grad_()
        prediction = torch.bmm(x, weights).tanh().sum(2) / hidden**0.5
        losses = (prediction - y).square().mean(1) / scale
        gradient = torch.autograd.grad(losses.sum(), weights)[0].flatten(1)
        sums, sizes = group_statistics(indices, gradient, positions)
        means = sums / sizes.clamp_min(1)
        if step % plan["log_every"] == 0 or step == plan["steps"]:
            with torch.no_grad():
                query_prediction = (
                    torch.bmm(qx, weights).tanh().sum(2) / hidden**0.5
                )
                query_loss = (query_prediction - qy).square().mean(1) / scale
                positive = qy > 0
                recall = ((query_prediction > 0) & positive).sum(1)
                recall = recall / positive.sum(1).clamp_min(1)
                train_positive = y > 0
                train_recall = ((prediction > 0) & train_positive).sum(1)
                train_recall = train_recall / train_positive.sum(1).clamp_min(
                    1
                )
                removed = (gradient - means.gather(1, indices)).square().sum(1)
                energy = gradient.square().sum(1)
                history.append(
                    {
                        "step": step,
                        "groups": groups,
                        "train_nmse": losses.detach().cpu().tolist(),
                        "query_nmse": query_loss.cpu().tolist(),
                        "train_positive_recall": train_recall.cpu().tolist(),
                        "query_positive_recall": recall.cpu().tolist(),
                        "gradient_energy": energy.cpu().tolist(),
                        "conflict_energy": removed.cpu().tolist(),
                        "conflict_fraction": (
                            removed / energy.clamp_min(1e-30)
                        )
                        .cpu()
                        .tolist(),
                    }
                )
                if groups not in best:
                    best[groups] = {
                        "query_nmse": torch.full_like(query_loss, torch.inf),
                        "weights": weights.detach().clone(),
                        "indices": indices.clone(),
                        "step": torch.zeros(
                            count, device=device, dtype=torch.long
                        ),
                        "train_nmse": torch.full_like(query_loss, torch.inf),
                    }
                saved = best[groups]
                good = query_loss < saved["query_nmse"]
                saved["query_nmse"][good] = query_loss[good]
                saved["weights"][good] = weights[good]
                saved["indices"][good] = indices[good]
                saved["step"][good] = step
                saved["train_nmse"][good] = losses[good]
        if step == plan["steps"]:
            break
        if step in schedule:
            if method in {"adaptive", "random_matched"}:
                torch.cuda.synchronize(device)
                split_started = time.perf_counter()
                source_event = None
                if method == "random_matched":
                    source_event = next(
                        event
                        for event in mirrored_events
                        if event["step"] == step
                    )
                decisions = []
                while groups < schedule[step]:
                    if method == "adaptive":
                        parent, left_size, move, score = gradient_split(
                            indices, gradient, groups
                        )
                    else:
                        decision = source_event["decisions"][len(decisions)]
                        parent = torch.tensor(
                            decision["parent"], device=device
                        )
                        left_size = torch.tensor(
                            decision["left_size"], device=device
                        )
                        move, score = random_split(
                            indices, gradient, parent, left_size, generator
                        )
                    values[:, groups] = values.gather(1, parent[:, None])[:, 0]
                    velocity[:, groups] = velocity.gather(1, parent[:, None])[
                        :, 0
                    ]
                    indices[move] = groups
                    decisions.append(
                        {
                            "new_group": groups,
                            "parent": parent.cpu().tolist(),
                            "left_size": left_size.cpu().tolist(),
                            "contrast_squared": score.cpu().tolist(),
                        }
                    )
                    groups += 1
                sums, sizes = group_statistics(indices, gradient, positions)
                means = sums / sizes.clamp_min(1)
                torch.cuda.synchronize(device)
                split_seconds += time.perf_counter() - split_started
                events.append({"step": step, "decisions": decisions})
        update = means if normalized else sums
        optimizer_step += 1
        if method == "dense_adam":
            velocity.mul_(0.9).add_(update, alpha=0.1)
            second.mul_(0.999).addcmul_(update, update, value=0.001)
            corrected = velocity / (1 - 0.9**optimizer_step)
            denominator = (second / (1 - 0.999**optimizer_step)).sqrt() + 1e-8
            values -= rate * corrected / denominator
        else:
            velocity.mul_(plan["momentum"]).add_(update)
            values -= rate * velocity
    torch.cuda.synchronize(device)
    result = {
        "phase_groups": np.array(list(best)),
        **{
            key: torch.stack([saved[key] for saved in best.values()])
            .cpu()
            .numpy()
            for key in [
                "query_nmse",
                "weights",
                "indices",
                "step",
                "train_nmse",
            ]
        },
        "rates": np.array(rates),
        "tasks": np.array(tasks),
        "starts": np.array(starts),
    }
    timing = {
        "seconds": time.perf_counter() - started,
        "split_seconds": split_seconds,
        "fitted_models": count,
        "gradient_updates_per_model": plan["steps"],
        "gradient_evaluations_per_model": plan["steps"] + 1,
    }
    return result, history, events, timing


def selected_models(result):
    tasks = int(result["tasks"])
    replicas = int(result["starts"]) * len(result["rates"])
    query = result["query_nmse"].reshape(
        len(result["phase_groups"]), tasks, replicas
    )
    best_flat = query.transpose(1, 0, 2).reshape(tasks, -1).argmin(1)
    phase, replica = np.divmod(best_flat, replicas)
    model = np.arange(tasks) * replicas + replica
    return phase, model


def evaluate_selected(plan, result, test_x, test_y):
    device = test_x.device
    tasks = int(result["tasks"])
    replicas = int(result["starts"]) * len(result["rates"])
    phase_ids, model_ids = selected_models(result)
    choices = {"selected": (phase_ids, model_ids)}
    for phase, groups in enumerate(result["phase_groups"]):
        query = result["query_nmse"][phase].reshape(tasks, replicas)
        ids = np.arange(tasks) * replicas + query.argmin(1)
        choices[f"k_{groups}"] = (np.full(tasks, phase), ids)
    metrics = []
    for selection, (phases, models) in choices.items():
        weights = torch.tensor(
            result["weights"][phases, models], device=device
        )
        with torch.no_grad():
            prediction = torch.bmm(test_x, weights).tanh().sum(2)
            prediction /= plan["hidden"] ** 0.5
            mse = (prediction - test_y).square().mean(1)
            variance = test_y.var(1, unbiased=False).clamp_min(1e-8)
            # Independent test variance is used only for reporting, not selection.
            nmse = mse / variance
            positive = test_y > 0
            recall = ((prediction > 0) & positive).sum(1)
            recall = recall / positive.sum(1).clamp_min(1)
        for task in range(tasks):
            phase, model = phases[task], models[task]
            metrics.append(
                {
                    "selection": selection,
                    "task_row": task,
                    "groups": int(result["phase_groups"][phase]),
                    "selected_start": int(model % replicas),
                    "selected_step": int(result["step"][phase, model]),
                    "query_nmse": float(result["query_nmse"][phase, model]),
                    "train_nmse": float(result["train_nmse"][phase, model]),
                    "test_mse": float(mse[task]),
                    "test_nmse": float(nmse[task]),
                    "test_positive_recall": float(recall[task]),
                }
            )
    return metrics
