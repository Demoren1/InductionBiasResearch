"""Batched, independently trained masked MLPs used to build the CVAE bank.

The bank deliberately keeps one ordinary MLP per row.  Its random fixed-K
masks impose no locality and no weights are shared between models.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


def ideal_mask(pattern: str, *, seq_len: int = 32, hidden: int = 32) -> torch.Tensor:
    """Return the exact-``hidden * len(pattern)`` cyclic-window mask.

    Hidden unit ``h`` receives the contiguous window beginning at
    ``h % (seq_len - k + 1)``.  Thus every hidden unit is active even when the
    number of valid windows is smaller than ``hidden``.
    """
    if not pattern or any(bit not in "01" for bit in pattern):
        raise ValueError("pattern must be a nonempty binary string")
    k = len(pattern)
    if not 1 <= k <= seq_len or hidden < 1:
        raise ValueError("invalid sequence, hidden, or pattern length")
    n_windows = seq_len - k + 1
    mask = torch.zeros(seq_len, hidden, dtype=torch.float32)
    for h in range(hidden):
        start = h % n_windows
        mask[start : start + k, h] = 1.0
    expected = hidden * k
    if int(mask.sum().item()) != expected:  # guards future changes to indexing
        raise AssertionError("ideal mask does not have the required cardinality")
    return mask


class BatchedMaskedMLP(nn.Module):
    """A batch of independent one-hidden-layer MLPs sharing input minibatches.

    ``masks`` has shape ``[models, seq_len, hidden]``.  ``forward(x)`` returns
    logits with shape ``[examples, models]``.
    """

    def __init__(self, masks: torch.Tensor, seed: int, *, weight_scale: float = 0.1):
        super().__init__()
        if masks.ndim != 3 or masks.size(0) < 1 or masks.size(1) < 1 or masks.size(2) < 1:
            raise ValueError("masks must have shape [models, seq_len, hidden]")
        if weight_scale < 0:
            raise ValueError("weight_scale must be nonnegative")
        n_models, seq_len, hidden = masks.shape
        # A local CPU RNG avoids changing the caller's global RNG state.  The
        # module can subsequently be moved to any accelerator.
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.w1 = nn.Parameter(torch.randn(n_models, seq_len, hidden, generator=generator) * weight_scale)
        self.b1 = nn.Parameter(torch.zeros(n_models, hidden))
        self.w2 = nn.Parameter(torch.randn(n_models, hidden, generator=generator) * weight_scale)
        self.b2 = nn.Parameter(torch.zeros(n_models))
        self.register_buffer("masks", masks.detach().to(dtype=torch.float32, device="cpu").clone())

    @property
    def n_models(self) -> int:
        return self.w1.size(0)

    @property
    def seq_len(self) -> int:
        return self.w1.size(1)

    @property
    def hidden(self) -> int:
        return self.w1.size(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.size(1) != self.seq_len:
            raise ValueError(f"x must have shape [batch, {self.seq_len}]")
        if x.device != self.w1.device:
            raise ValueError("x and model must be on the same device")
        hidden = torch.einsum("bi,mih->bmh", x, self.w1 * self.masks)
        hidden = F.relu(hidden + self.b1.unsqueeze(0))
        return torch.einsum("bmh,mh->bm", hidden, self.w2) + self.b2.unsqueeze(0)

    @torch.no_grad()
    def validation(self, x: torch.Tensor, y: torch.Tensor, *, batch_size: int = 2048) -> dict[str, torch.Tensor]:
        """Per-model BCE and accuracy, evaluated without constructing B×M targets."""
        if x.ndim != 2 or y.ndim != 1 or x.size(0) != y.numel():
            raise ValueError("x and y must be aligned rank-2/rank-1 tensors")
        losses = torch.zeros(self.n_models, device=x.device)
        correct = torch.zeros(self.n_models, device=x.device)
        for start in range(0, x.size(0), batch_size):
            xb, yb = x[start : start + batch_size], y[start : start + batch_size]
            logits = self(xb)
            losses += F.binary_cross_entropy_with_logits(
                logits, yb[:, None].expand_as(logits), reduction="none"
            ).sum(dim=0)
            correct += ((logits > 0) == yb.bool()[:, None]).sum(dim=0)
        return {"bce": losses / x.size(0), "accuracy": correct / x.size(0)}

    def cpu_state(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name).detach().cpu().clone() for name in ("w1", "b1", "w2", "b2")}


def fit_mlp(
    model: BatchedMaskedMLP,
    x_support: torch.Tensor,
    y_support: torch.Tensor,
    *,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> dict[str, Any]:
    """Fit every MLP on the same support pool.

    The per-model mean BCE is *summed* before ``backward``.  Averaging across
    models would make an Adam update depend on bank size via epsilon and is
    not the independent-model objective we intend to reproduce.
    """
    if steps < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("steps, batch_size, and lr must be positive")
    if x_support.ndim != 2 or y_support.ndim != 1 or x_support.size(0) != y_support.numel():
        raise ValueError("support x and y must be aligned rank-2/rank-1 tensors")
    if x_support.size(0) < 1 or x_support.size(1) != model.seq_len:
        raise ValueError("invalid support pool shape")
    if x_support.device != model.w1.device or y_support.device != model.w1.device:
        raise ValueError("support pool and model must share a device")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Index sampling uses a private CPU generator for run-order independence.
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    last_loss = float("nan")
    model.train()
    for _ in range(steps):
        index = torch.randint(x_support.size(0), (batch_size,), generator=generator, device="cpu")
        index = index.to(x_support.device, non_blocking=True)
        logits = model(x_support.index_select(0, index))
        labels = y_support.index_select(0, index)
        per_model = F.binary_cross_entropy_with_logits(
            logits, labels[:, None].expand_as(logits), reduction="none"
        ).mean(dim=0)
        loss = per_model.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    return {"steps": steps, "last_sum_bce": last_loss, "n_models": model.n_models}
