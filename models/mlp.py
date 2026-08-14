"""Batched masked MLP.

Trains ``n_mlps`` independent single-hidden-layer MLPs in one pass:

    y = W2^T ReLU(W1^T x + b1) + b2 ,   W1 effective = W1 * M

``M`` is a fixed binary mask (p * 100% active entries) applied only to the
weights of the first linear layer. All MLPs share the same forward pass,
so one forward/backward updates every MLP in the batch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_masks(n_mlps: int, in_dim: int, hidden: int, p: float,
                   seed: int) -> torch.Tensor:
    """Return a binary tensor (n_mlps, in_dim, hidden) with == ``p`` ones."""
    g = torch.Generator().manual_seed(seed)
    m = (torch.rand(n_mlps, in_dim, hidden, generator=g) < p).float()
    return m


class BatchedMaskedMLP(nn.Module):
    """``n_mlps`` MLPs trained simultaneously.

    Args:
        n_mlps: number of MLPs in this batch.
        in_dim: input size (L).
        hidden: hidden size (H).
    """

    def __init__(self, n_mlps: int, in_dim: int, hidden: int):
        super().__init__()
        # First layer, masked elementwise by M.
        self.w1 = nn.Parameter(torch.randn(n_mlps, in_dim, hidden) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(n_mlps, hidden))
        # Second layer, dense.
        self.w2 = nn.Parameter(torch.randn(n_mlps, hidden, 1) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(n_mlps, 1))
        # Fixed binary mask (20% active by default). Registered as a buffer
        # so it is *not* updated during training.
        self.register_buffer("mask", torch.ones(n_mlps, in_dim, hidden))

    def load_masks(self, masks: torch.Tensor) -> None:
        """Copy binary masks into the buffer (replaces the current ones)."""
        self.mask.copy_(masks.to(self.mask.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (n_batch, in_dim) -> out: (n_batch, n_mlps)."""
        w1_m = self.w1 * self.mask                       # (M, L, H)
        h = torch.einsum("bl,mlh->bmh", x, w1_m)         # (B, M, H)
        h = F.relu(h + self.b1)
        out = torch.einsum("bmh,mh->bm", h, self.w2.squeeze(-1))
        out = out + self.b2.squeeze(-1)                  # (B, M)
        return out

    def val_loss(self, x_val: torch.Tensor, y_val: torch.Tensor,
                 val_batch: int) -> torch.Tensor:
        """Per-MLP MSE over the validation set -> (n_mlps,)."""
        self.eval()
        losses = []
        with torch.no_grad():
            for i in range(0, x_val.size(0), val_batch):
                xb = x_val[i:i + val_batch]
                yb = y_val[i:i + val_batch]
                pred = self.forward(xb)                      # (B', M)
                l = F.mse_loss(
                    pred, yb.unsqueeze(1).expand_as(pred), reduction="none"
                ).mean(dim=0)                                # (M,)
                losses.append(l)
        self.train()
        return torch.stack(losses, dim=0).mean(dim=0)        # (M,)

    @staticmethod
    def state_as_dict(w1, b1, w2, b2):
        """Pack trained tensors into a small checkpoint dict."""
        return {"w1": w1.detach().cpu(), "b1": b1.detach().cpu(),
                "w2": w2.detach().cpu(), "b2": b2.detach().cpu()}


def get_train_batch(kernel: int, offset: int, batch_size: int, in_dim: int,
                    seed: int) -> tuple:
    """Fresh (x, y) for one training step: L i.i.d. N(0,1),
    y = shifted MA over the window ending at in_dim - offset."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch_size, in_dim, generator=g)
    y = x[:, in_dim - offset - kernel: in_dim - offset].mean(dim=1)
    return x, y