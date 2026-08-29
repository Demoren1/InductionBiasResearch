"""Leakage-safe mask generators used in motif-pair OOD evaluation."""

from __future__ import annotations

from pathlib import Path

import torch

import config
from models.cvae import load_top_importance, task_condition


def _shape():
    return int(config.SEQ_LEN), int(config.H)


def random_bernoulli(n: int, *, p: float = .375, generator=None, device=None):
    """Independent Bernoulli baseline; expected cardinality is 96."""
    return (torch.rand(n, config.MASK_DIM, generator=generator, device=device) < p).float()


def random_exact(n: int, *, k_active: int | None = None, generator=None, device=None):
    """Uniform fixed-cardinality baseline; this is the primary comparator."""
    k = int(config.K_ACTIVE if k_active is None else k_active)
    scores = torch.rand(n, config.MASK_DIM, generator=generator, device=device)
    top = scores.topk(k, dim=-1).indices
    masks = torch.zeros_like(scores)
    return masks.scatter_(1, top, 1.0)


class ConditionalMean:
    """Train-only mean importance baseline conditioned strictly on the gap.

    A held-out task may use its gap's pool, but all maps in the pool originate
    from ``train_tasks``.  This is intentionally not a per-held-out-task mean.
    """

    def __init__(self, train_tasks, *, ckpt_root: Path | None = None,
                 importance_name: str = "importance.pt", top_frac: float = .1,
                 device: torch.device | str = "cpu", split_path: Path | None = None,
                 expected_provenance: dict | None = None):
        self.train_tasks = list(train_tasks)
        x, c, self.provenance = load_top_importance(self.train_tasks, ckpt_root,
                                                    importance_name, top_frac, device=device,
                                                    split_path=split_path,
                                                    expected_provenance=expected_provenance)
        self.by_condition = {}
        for key in torch.unique(c, dim=0):
            matching = (c == key).all(dim=1)
            self.by_condition[tuple(key.tolist())] = x[matching].mean(dim=0)
        self.global_mean = x.mean(dim=0)

    def __call__(self, task, k_active: int | None = None):
        c = task_condition([task])[0]
        # Every evaluation gap should be represented in meta-train.  The
        # fallback keeps the error readable for deliberately sparse smoke data.
        mean = self.by_condition.get(tuple(c.tolist()), self.global_mean)
        k = int(config.K_ACTIVE if k_active is None else k_active)
        top = mean.topk(k).indices
        mask = torch.zeros_like(mean)
        return mask.scatter_(0, top, 1.0)


def gold_mask(task):
    """Resolve the known synthetic gold support without exposing it to models."""
    from data.generate import ideal_mask
    return ideal_mask(task).reshape(-1).float()
