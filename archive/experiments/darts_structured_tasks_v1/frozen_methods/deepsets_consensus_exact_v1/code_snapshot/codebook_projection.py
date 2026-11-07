"""Globally solve scalar signed-code projection with a charged zero option."""

import torch
from torch.nn import functional as F


@torch.no_grad()
def exact_signed_projection(values, codes, charge, rho):
    shape = values.shape
    flattened = values.flatten(1)
    magnitudes, order = flattened.abs().double().sort(-1)
    roots, length = magnitudes.shape
    endpoints = torch.arange(length + 1, device=values.device)
    prefix = F.pad(magnitudes.cumsum(-1), (1, 0))
    squared = F.pad(magnitudes.square().cumsum(-1), (1, 0))
    count = endpoints[None, :] - endpoints[:, None]
    totals = prefix[:, None, :] - prefix[:, :, None]
    intervals = squared[:, None, :] - squared[:, :, None]
    intervals -= totals.square() / count.clamp_min(1)
    intervals.clamp_min_(0)
    intervals.masked_fill_(count[None] <= 0, torch.inf)
    penalty = (2 * charge / rho).double()
    # A zero prefix followed by any positive clusters pays once for the tail.
    cost = squared + penalty[:, None] * (length - endpoints)
    finishes, back = [cost[:, -1]], []
    for _ in range(codes):
        cost, previous = (cost[:, :, None] + intervals).min(1)
        back.append(previous)
        finishes.append(cost[:, -1])
    groups = torch.stack(finishes).argmin(0)
    pointers = torch.stack(back)
    end = torch.full((roots,), length, device=values.device, dtype=torch.long)
    root = torch.arange(roots, device=values.device)
    positions = torch.arange(length, device=values.device)[None]
    assignments = torch.zeros_like(magnitudes, dtype=torch.long)
    for _ in range(codes):
        active = groups > 0
        start = pointers[groups.clamp_min(1) - 1, root, end]
        block = (
            active[:, None] & (positions >= start[:, None]) & (positions < end[:, None])
        )
        assignments = torch.where(block, groups[:, None], assignments)
        end = torch.where(active, start, end)
        groups = (groups - 1).clamp_min(0)
    centers = torch.zeros(roots, codes, dtype=torch.float64, device=values.device)
    for code in range(1, codes + 1):
        active = assignments == code
        centers[:, code - 1] = (magnitudes * active).sum(-1) / active.sum(-1).clamp_min(
            1
        )
    negative = (flattened.gather(1, order) < 0) & (assignments != 0)
    signed = torch.where(negative, assignments + codes, assignments)
    labels = torch.zeros_like(signed).scatter(1, order, signed)
    centers = centers.to(values.dtype)
    dictionary = torch.cat((centers[:, :1] * 0, centers, -centers), 1)
    quantized = dictionary.gather(1, labels).reshape(shape)
    return labels.reshape(shape), centers, quantized
