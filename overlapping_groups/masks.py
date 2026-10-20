"""Column canonicalization, exact top-k masks and structural metrics."""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .problem import EDGES, HIDDEN, INPUTS


def column_order(mask):
    """Sort hard column profiles without using a reference or observations."""
    hard = mask.detach().to(torch.float32).round().to(torch.int64)
    powers = 1 << torch.arange(INPUTS - 1, -1, -1, device=mask.device)
    codes = (hard * powers[:, None]).sum(-2)
    return codes.argsort(dim=-1, descending=True, stable=True)


def canonical_mask(mask):
    order = column_order(mask)
    return mask.gather(-1, order[..., None, :].expand_as(mask))


def topk(scores, straight_through=False):
    flat = scores.detach().flatten(-2)
    indices = flat.argsort(dim=-1, descending=True, stable=True)[..., :EDGES]
    hard = torch.zeros_like(flat).scatter_(-1, indices, 1).reshape_as(scores)
    if straight_through:
        soft = scores.sigmoid()
        return hard + (soft - soft.detach())
    return hard


def assignments(cost):
    """Solve discrete 8x8 assignments; all differentiable work stays on CUDA."""
    costs = cost.detach().reshape(-1, HIDDEN, HIDDEN).cpu().numpy()
    orders = np.stack([linear_sum_assignment(value)[1] for value in costs])
    return torch.as_tensor(orders, device=cost.device, dtype=torch.long)


def structural_metrics(masks, target):
    """Align hidden columns only; retain noise edges in the full-mask metric."""
    target = target.cpu().numpy()
    core = target.copy()
    core[16:] = 0
    values, orders = [], []
    for mask in masks.detach().cpu().numpy():
        overlap = mask.T @ target
        source, destination = linear_sum_assignment(-overlap)
        order = np.empty(HIDDEN, dtype=int)
        order[destination] = source
        aligned = mask[:, order]
        intersection = float((aligned * target).sum())
        variable_intersection = float((aligned[16:32] * target[16:32]).sum())
        variable_union = float(
            aligned[16:32].sum() + 16 - variable_intersection
        )
        values.append(
            {
                "iou": intersection
                / (float(mask.sum()) + EDGES - intersection),
                "core_recall": float((aligned * core).sum() / 16),
                "variable_recall": variable_intersection / 16,
                "variable_iou": variable_intersection / max(1, variable_union),
                "exact_groups": int(np.all(aligned == target, axis=0).sum()),
                "noise_edges": int(aligned[32:].sum()),
                "exact_mask": bool(np.array_equal(aligned, target)),
            }
        )
        orders.append(order.tolist())
    return values, orders
