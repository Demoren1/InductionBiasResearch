"""Bounded-memory positive-scale Algorithm1 proposals for approximate SRR."""

import numpy as np
import torch

from experiments.omniglot_prior_v1.model import calibration

from .core import canonical


def project_chunked(initial, weights, uncertainty, *, steps, device):
    """Evaluate the full Student assignment cost in position/code blocks.

    Positive calibration uses the existing small Schur solve. Empty codes
    are restored greedily by minimum incremental cost; support is fixed.
    This is a practical positive-scale instantiation, not the original
    exhaustive signed-orientation search. Keep its initial seed separately.
    """
    index = initial.copy()
    shape = index.shape
    mask = index.ravel() != 0
    codes = int(np.abs(index).max())
    symbols = np.r_[np.arange(1, codes + 1), -np.arange(1, codes + 1)]
    normal = torch.as_tensor(
        weights.reshape(len(weights), -1), dtype=torch.float64, device=device
    )
    errors = np.broadcast_to(uncertainty[:, None], weights.shape).reshape(
        len(weights), -1
    )
    records = []
    for step in range(steps):
        values, scales = calibration(
            np.broadcast_to(index, weights.shape), weights, uncertainty
        )
        factor = np.broadcast_to(scales[:, None], weights.shape).reshape(
            len(weights), -1
        )
        target = normal * torch.as_tensor(factor, device=device)
        deviation = torch.as_tensor(
            np.maximum(errors, 0.0003) * np.maximum(np.abs(factor), 0.01), device=device
        )
        centers = torch.as_tensor(np.c_[values, -values], device=device)
        cost = np.empty((index.size, 2 * codes), dtype=np.float32)
        for start in range(0, index.size, 256):
            stop = min(start + 256, index.size)
            for begin in range(0, 2 * codes, 512):
                end = min(begin + 512, 2 * codes)
                residual = (
                    target[:, start:stop, None] - centers[:, None, begin:end]
                ) / deviation[:, start:stop, None]
                local = (4 * torch.log1p(residual.square() / 4)).mean(0)
                cost[start:stop, begin:end] = local.cpu().numpy()
        selected = cost.argmin(1)
        updated = symbols[selected].astype(np.int16)
        updated[~mask] = 0
        counts = np.bincount(np.abs(updated), minlength=codes + 1)
        minimum = cost[np.arange(index.size), selected]
        missing = np.flatnonzero(counts[1:] == 0) + 1
        for group in missing:
            positive, negative = cost[:, group - 1], cost[:, group - 1 + codes]
            extra = np.minimum(positive, negative) - minimum
            allowed = mask & (counts[np.abs(updated)] > 1)
            position = int(np.where(allowed, extra, np.inf).argmin())
            counts[abs(int(updated[position]))] -= 1
            updated[position] = (
                group if positive[position] <= negative[position] else -group
            )
            counts[group] += 1
            minimum[position] = min(positive[position], negative[position])
        changed = float(np.mean(updated != index.ravel()))
        records.append(
            {
                "step": step + 1,
                "changed_fraction": changed,
                "restored_codes": len(missing),
                "mean_cost": float(np.mean(minimum[mask])),
            }
        )
        index = updated.reshape(shape)
        if changed == 0:
            break
    return canonical(index), records
