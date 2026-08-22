"""Batched masked MLP for the pattern-in-sequence task.

Trains n_mlps independent single-hidden-layer MLPs in one pass:

    logits = W2^T ReLU((W1 * M)^T x + b1) + b2

M is a fixed binary mask applied only to the first-layer weights.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from data.generate import make_dataset  # noqa: E402


def generate_masks(n_mlps: int, in_dim: int, hidden: int, p: float,
                   seed: int) -> torch.Tensor:
    """Return seeded Bernoulli masks with activation probability p."""
    g = torch.Generator().manual_seed(seed)
    m = (torch.rand(n_mlps, in_dim, hidden, generator=g) < p).float()
    return m


class BatchedMaskedMLP(nn.Module):
    """n_mlps MLPs trained simultaneously."""

    def __init__(self, n_mlps: int, in_dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(n_mlps, in_dim, hidden) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(n_mlps, hidden))
        self.w2 = nn.Parameter(torch.randn(n_mlps, hidden, 1) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(n_mlps, 1))
        self.register_buffer("mask", torch.ones(n_mlps, in_dim, hidden))

    def load_masks(self, masks: torch.Tensor) -> None:
        self.mask.copy_(masks.to(self.mask.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (n_batch, in_dim) -> logits: (n_batch, n_mlps)."""
        w1_m = self.w1 * self.mask
        h = torch.einsum("bl,mlh->bmh", x, w1_m)
        h = F.relu(h + self.b1)
        out = torch.einsum("bmh,mh->bm", h, self.w2.squeeze(-1))
        out = out + self.b2.squeeze(-1)
        return out

    def val_loss(self, x_val: torch.Tensor, y_val: torch.Tensor,
                 val_batch: int) -> torch.Tensor:
        """Per-MLP BCE-with-logits over the validation set -> (n_mlps,)."""
        was_training = self.training
        self.eval()
        losses = []
        with torch.no_grad():
            for i in range(0, x_val.size(0), val_batch):
                xb = x_val[i:i + val_batch]
                yb = y_val[i:i + val_batch]
                pred = self.forward(xb)
                target = yb.unsqueeze(1).expand_as(pred)
                l = F.binary_cross_entropy_with_logits(
                    pred, target, reduction="none").mean(dim=0)
                losses.append(l)
        self.train(was_training)
        return torch.stack(losses, dim=0).mean(dim=0)

    def val_acc(self, x_val: torch.Tensor, y_val: torch.Tensor,
                val_batch: int) -> torch.Tensor:
        """Per-MLP accuracy (logits > 0) -> (n_mlps,)."""
        was_training = self.training
        self.eval()
        correct = []
        total = 0
        with torch.no_grad():
            for i in range(0, x_val.size(0), val_batch):
                xb = x_val[i:i + val_batch]
                yb = y_val[i:i + val_batch]
                pred = (self.forward(xb) > 0).float()
                target = yb.unsqueeze(1).expand_as(pred)
                correct.append((pred == target).float().sum(dim=0))
                total += xb.size(0)
        self.train(was_training)
        return torch.stack(correct, dim=0).sum(dim=0) / total

    @staticmethod
    def state_as_dict(w1, b1, w2, b2):
        return {"w1": w1.detach().cpu(), "b1": b1.detach().cpu(),
                "w2": w2.detach().cpu(), "b2": b2.detach().cpu()}


def get_train_batch(pat: str, batch_size: int, seed: int) -> tuple:
    """Return one seeded on-the-fly training batch."""
    data = make_dataset(pat, batch_size, seed, config.POS_FRACTION)
    return data["x"], data["y"]
