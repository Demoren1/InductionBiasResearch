"""Budgeted WSR and LUT-Q source proposals for a fixed active layer."""

import numpy as np
import torch

from experiments.wsr_omniglot_v1.source import proximal_groups


@torch.no_grad()
def lookup(weight, centers):
    ordered, ids = centers.sort(dim=-1, stable=True)
    high = torch.searchsorted(ordered, weight.contiguous()).clamp_max(
        ordered.shape[-1] - 1
    )
    low = (high - 1).clamp_min(0)
    left, right = ordered.gather(1, low), ordered.gather(1, high)
    position = torch.where(
        (weight - left).abs() <= (weight - right).abs(), low, high
    )
    return ids.gather(1, position), ordered.gather(1, position)


@torch.no_grad()
def refill_dictionary(weight, centers):
    labels, projected = lookup(weight, centers)
    count = torch.zeros_like(centers).scatter_add_(
        1, labels, torch.ones_like(weight)
    )
    for i in (count == 0).any(1).nonzero().flatten().tolist():
        absent = (count[i] == 0).nonzero().flatten()
        error = (weight[i] - projected[i]).abs()
        # Reserve one distinct value for every unused dictionary entry.
        order = error.argsort(descending=True)
        candidates = weight[i, order]
        values, inverse = torch.unique(candidates, return_inverse=True)
        first = order.new_full((len(values),), len(order))
        first.scatter_reduce_(
            0,
            inverse,
            torch.arange(len(order), device=order.device),
            reduce="amin",
            include_self=True,
        )
        if len(values) < len(centers[i]):
            raise ValueError("Too few distinct weights for the LUT-Q budget")
        values = values[first.argsort()]
        present = (count[i] != 0).nonzero().flatten()
        nearest = (weight[i] - projected[i]).abs().argsort(descending=True)
        representatives = nearest.new_full((len(centers[i]),), len(nearest))
        representatives.scatter_reduce_(
            0,
            labels[i, nearest],
            torch.arange(len(nearest), device=nearest.device),
            reduce="amax",
            include_self=False,
        )
        centers[i, present] = weight[i, nearest[representatives[present]]]
        used = torch.isin(values, centers[i, present])
        centers[i, absent] = values[~used][: len(absent)]
    return lookup(weight, centers)


def fit_indexators(method, initial, width, codes, loss, config):
    """Fit each source task's dictionary and keep its best numerical checkpoint."""
    if method not in ("lutq", "wsr"):
        raise ValueError(f"Unknown discrete source learner {method}")
    parameter = initial.detach().clone().requires_grad_()
    optimizer = torch.optim.Adam([parameter], lr=config["rate"])
    strengths = [0.001] * len(parameter)
    if method == "lutq":
        ticks = torch.linspace(0, width - 1, codes, device=parameter.device)
        centers = (
            parameter.detach()[:, :width]
            .sort(dim=1)
            .values[:, ticks.round().long()]
        )
    best = np.full(len(parameter), np.inf)
    selected = np.zeros((len(parameter), width), np.int32)
    selected_steps = np.zeros(len(parameter), dtype=int)
    for step in range(config["steps"] + 1):
        weight, tail = parameter[:, :width], parameter[:, width:]
        with torch.no_grad():
            if method == "wsr":
                rows = [
                    proximal_groups(w, codes, c, 0.98, break_ties=True)
                    for w, c in zip(
                        weight.detach().cpu().numpy(), strengths, strict=True
                    )
                ]
                strengths = [row[2] for row in rows]
                labels = np.stack([row[1] for row in rows])
                weight.copy_(
                    torch.as_tensor(
                        np.stack([row[0] for row in rows]),
                        device=weight.device,
                    )
                )
            else:
                assignment, projected = lookup(weight, centers)
            if step in config["checkpoints"]:
                if method == "lutq":
                    assignment, projected = refill_dictionary(weight, centers)
                    labels = assignment.cpu().numpy() + 1
                effective = weight if method == "wsr" else projected
                scores = loss(effective, tail, validation=True).cpu().numpy()
                improved = scores < best
                best[improved] = scores[improved]
                selected[improved] = labels[improved]
                selected_steps[improved] = step
        if step == config["steps"]:
            break
        effective = (
            weight
            if method == "wsr"
            else weight + (projected - weight).detach()
        )
        optimizer.zero_grad(set_to_none=True)
        loss(effective, tail, validation=False).sum().backward()
        optimizer.step()
        if method == "lutq":
            with torch.no_grad():
                weight = parameter[:, :width]
                assignment, _ = lookup(weight, centers)
                count = torch.zeros_like(centers).scatter_add_(
                    1, assignment, torch.ones_like(weight)
                )
                totals = torch.zeros_like(centers).scatter_add_(
                    1, assignment, weight
                )
                centers.copy_(
                    torch.where(
                        count > 0, totals / count.clamp_min(1), centers
                    )
                )
    if any(len(np.unique(row)) != codes or row.min() < 1 for row in selected):
        raise ValueError(f"{method} did not produce {codes} occupied codes")
    return selected, best, selected_steps
