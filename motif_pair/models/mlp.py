"""Batched masked MLP used to build the motif-pair candidate bank.

Each batch dimension of the parameters represents an independent model.  A
binary mask is applied to the input-to-hidden weights only, so a candidate is
fully described by a 16 x H connectivity matrix and its trained weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from data.generate import make_dataset  # noqa: E402


def generate_masks(n_mlps: int, in_dim: int, hidden: int, p: float,
                   seed: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Generate seeded, independent Bernoulli connectivity masks."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    return (torch.rand(n_mlps, in_dim, hidden, generator=generator,
                       device=device) < p).float()


class BatchedMaskedMLP(nn.Module):
    """A bank of independent one-hidden-layer binary classifiers."""

    def __init__(self, n_mlps: int, in_dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(n_mlps, in_dim, hidden) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(n_mlps, hidden))
        self.w2 = nn.Parameter(torch.randn(n_mlps, hidden, 1) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(n_mlps, 1))
        self.register_buffer("mask", torch.ones(n_mlps, in_dim, hidden))

    def load_masks(self, masks: torch.Tensor) -> None:
        if tuple(masks.shape) != tuple(self.mask.shape):
            raise ValueError(f"expected masks of shape {tuple(self.mask.shape)}, "
                             f"got {tuple(masks.shape)}")
        self.mask.copy_(masks.to(device=self.mask.device, dtype=self.mask.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape ``(batch, n_mlps)`` for ``(batch, in_dim)``."""
        if x.ndim != 2 or x.size(1) != self.w1.size(1):
            raise ValueError(f"expected x shaped (batch, {self.w1.size(1)})")
        hidden = torch.einsum("bi,mih->bmh", x, self.w1 * self.mask)
        hidden = F.relu(hidden + self.b1)
        return torch.einsum("bmh,mh->bm", hidden, self.w2.squeeze(-1)) + self.b2.squeeze(-1)

    def val_loss(self, x_val: torch.Tensor, y_val: torch.Tensor,
                 val_batch: int) -> torch.Tensor:
        """Compute a validation BCE for every candidate, without gradients."""
        was_training = self.training
        self.eval()
        losses = []
        with torch.no_grad():
            for start in range(0, x_val.size(0), val_batch):
                x = x_val[start:start + val_batch]
                y = y_val[start:start + val_batch]
                logits = self(x)
                losses.append(F.binary_cross_entropy_with_logits(
                    logits, y.unsqueeze(1).expand_as(logits), reduction="none"
                ).mean(dim=0))
        self.train(was_training)
        return torch.stack(losses).mean(dim=0)

    def val_acc(self, x_val: torch.Tensor, y_val: torch.Tensor,
                val_batch: int) -> torch.Tensor:
        """Compute validation accuracy for every candidate, without gradients."""
        was_training = self.training
        self.eval()
        correct, total = [], 0
        with torch.no_grad():
            for start in range(0, x_val.size(0), val_batch):
                x = x_val[start:start + val_batch]
                y = y_val[start:start + val_batch]
                prediction = (self(x) > 0).to(y.dtype)
                correct.append((prediction == y.unsqueeze(1)).sum(dim=0))
                total += x.size(0)
        self.train(was_training)
        return torch.stack(correct).sum(dim=0) / total

    @staticmethod
    def state_as_dict(w1: torch.Tensor, b1: torch.Tensor, w2: torch.Tensor,
                      b2: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"w1": w1.detach().cpu(), "b1": b1.detach().cpu(),
                "w2": w2.detach().cpu(), "b2": b2.detach().cpu()}


def get_train_batch(task: str, batch_size: int, seed: int,
                    device: torch.device | str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Create one deterministic fresh train batch for a motif-pair task."""
    data = make_dataset(task, batch_size, seed, config.POS_FRACTION)
    if isinstance(data, dict):
        x, y = data["x"], data["y"]
        return ((x.to(device), y.to(device)) if device is not None else (x, y))
    x, y = data
    return ((x.to(device), y.to(device)) if device is not None else (x, y))
