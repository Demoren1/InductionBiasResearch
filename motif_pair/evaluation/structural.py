"""Permutation-invariant structural diagnostics for first-layer masks."""

from __future__ import annotations

import torch


def _assignment_max(scores: torch.Tensor) -> torch.Tensor:
    """Maximum-weight square assignment, with SciPy optional not required."""
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment((-scores.detach().cpu().numpy()))
        result = torch.empty(len(rows), dtype=torch.long)
        result[torch.as_tensor(rows)] = torch.as_tensor(cols)
        return result
    except ImportError:
        # Kuhn--Munkres for a minimisation cost matrix, O(n^3).  A constant
        # shift turns the maximum score problem into a non-negative minimum.
        a = (scores.max() - scores).detach().cpu().double().tolist()
        n = len(a); u = [0.] * (n + 1); v = [0.] * (n + 1)
        p = [0] * (n + 1); way = [0] * (n + 1)
        for i in range(1, n + 1):
            p[0] = i; j0 = 0; minv = [float("inf")] * (n + 1); used = [False] * (n + 1)
            while True:
                used[j0] = True; i0 = p[j0]; delta = float("inf"); j1 = 0
                for j in range(1, n + 1):
                    if not used[j]:
                        cur = a[i0 - 1][j - 1] - u[i0] - v[j]
                        if cur < minv[j]: minv[j] = cur; way[j] = j0
                        if minv[j] < delta: delta = minv[j]; j1 = j
                for j in range(n + 1):
                    if used[j]: u[p[j]] += delta; v[j] -= delta
                    else: minv[j] -= delta
                j0 = j1
                if p[j0] == 0: break
            while True:
                j1 = way[j0]; p[j0] = p[j1]; j0 = j1
                if j0 == 0: break
        assignment = torch.empty(n, dtype=torch.long)
        for j in range(1, n + 1): assignment[p[j] - 1] = j - 1
        return assignment


def best_permutation_iou(mask: torch.Tensor, gold: torch.Tensor) -> dict:
    """IoU after optimally permuting hidden units (the mask's columns).

    Since masks and gold masks have fixed support cardinalities, maximizing
    column-wise intersection also maximizes global IoU.
    """
    if mask.ndim == 1:
        side = int(mask.numel() ** .5)
        mask = mask.reshape(side, side)
    if gold.ndim == 1:
        gold = gold.reshape(mask.shape)
    if mask.shape != gold.shape or mask.ndim != 2:
        raise ValueError(f"expected equally shaped 2D masks, got {mask.shape} and {gold.shape}")
    m, g = mask.bool(), gold.bool()
    # score[i,j]: overlap when generated hidden unit i is aligned to gold j.
    scores = (m.T[:, None, :] & g.T[None, :, :]).sum(-1).float()
    perm = _assignment_max(scores)
    aligned = m[:, torch.argsort(perm)]
    intersection = int((aligned & g).sum())
    union = int((aligned | g).sum())
    return {"iou": intersection / union if union else 1.0,
            "intersection": intersection, "union": union,
            "permutation": perm.tolist()}
