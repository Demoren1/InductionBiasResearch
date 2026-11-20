"""Batched dense and tied MLPs, support-only projection and fine-tuning."""

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .posthoc_source import C, D, H


def start(batch, blocks, groups, width, seed, device):
    gen = torch.Generator(device=device).manual_seed(seed)

    def normal(*dims):
        return torch.randn(batch, *dims, device=device, generator=gen)

    return (
        normal(blocks, groups, width // blocks) * math.sqrt(2 / D),
        normal(1, width) * 0.1,
        normal(width, C),
        torch.zeros(batch, 1, C, device=device),
    )


def indicators(labels, device):
    index = torch.tensor(labels.astype(np.int64), device=device)
    return F.one_hot(index, int(labels.max()) + 1).float()


def inputs(x, mapping=None):
    if mapping is None:
        return x[:, None]
    return torch.stack([x @ m for m in mapping], dim=1)


def features(x, w, b):
    hidden = (x @ w).transpose(1, 2).flatten(2)
    return (hidden + b).relu() / math.sqrt(hidden.shape[-1])


def predict(x, state):
    w, b, v, a = state
    return features(x, w, b) @ v + a


def fit(state, sx, sy, lr, checkpoints, record, soft_target=None):
    state = tuple(t.detach().clone().requires_grad_() for t in state)
    optimizer = torch.optim.Adam(state, lr=lr, fused=True)
    checkpoints = set(checkpoints)
    last = max(checkpoints)
    if soft_target is not None:
        prob = (soft_target / 2).softmax(-1)
    for step in range(last + 1):
        if step in checkpoints:
            with torch.no_grad():
                record(step, state)
        if step == last:
            break
        logits = predict(sx, state)
        loss = F.cross_entropy(logits.flatten(0, 1), sy.flatten())
        if soft_target is not None:
            # Cross-entropy has the same gradient as teacher KL up to a constant.
            soft = -(prob * (logits / 2).log_softmax(-1)).sum(-1).mean()
            loss = 0.5 * loss + 2 * soft
        optimizer.zero_grad(set_to_none=True)
        (loss * len(sx)).backward()
        optimizer.step()
    return tuple(t.detach() for t in state)


def align_teacher(state, anchor):
    w, b, v, a = state
    cosines = anchor.T @ F.normalize(w[:, 0], dim=1)
    order = np.stack([linear_sum_assignment(-m)[1] for m in cosines.cpu().numpy()])
    order = torch.as_tensor(order, device=w.device)
    return (
        w.gather(3, order[:, None, None].expand(-1, 1, D, -1)),
        b.gather(2, order[:, None]),
        v.gather(1, order[:, :, None].expand(-1, -1, C)),
        a,
    )


def project(state, mapping):
    w, b, v, a = state
    blocks, _, groups = mapping.shape
    per_block = H // blocks
    raw = w[:, 0].reshape(-1, D, blocks, per_block).transpose(1, 2)
    sums = mapping.transpose(1, 2) @ raw
    codes = sums / mapping.sum(1).view(1, blocks, groups, 1)
    return codes, b, v, a


def prune(state, sx, width):
    w, b, v, a = state
    rms = features(sx, w, b).square().mean(1).sqrt()
    saliency = rms * (v - v.mean(-1, keepdim=True)).norm(dim=-1)
    order = saliency.topk(width).indices
    return (
        w.gather(3, order[:, None, None].expand(-1, 1, D, -1)),
        b.gather(2, order[:, None]),
        v.gather(1, order[:, :, None].expand(-1, -1, C)) * math.sqrt(width / H),
        a,
    )


def repair(state, sx, targets, coefficient=1e-3):
    w, b, v, _ = state
    z = features(sx, w, b).double()
    y, v = targets.double(), v.double()
    z_bar, y_bar = z.mean(1, keepdim=True), y.mean(1, keepdim=True)
    z, y = z - z_bar, y - y_bar
    ridge = (coefficient * z.square().sum((1, 2)) / z.shape[-1]).clamp_min(1e-6)
    zt = z.transpose(1, 2)
    resid = y - z @ v
    if z.shape[1] <= z.shape[2]:
        gram = z @ zt
        gram.diagonal(dim1=1, dim2=2).add_(ridge[:, None])
        delta = zt @ torch.linalg.solve(gram, resid)
    else:
        gram = zt @ z
        gram.diagonal(dim1=1, dim2=2).add_(ridge[:, None])
        delta = torch.linalg.solve(gram, zt @ resid)
    v = v + delta
    return w, b, v.float(), (y_bar - z_bar @ v).float()
