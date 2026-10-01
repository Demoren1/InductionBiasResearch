"""Empirical loss-plateau checks; a step cap is never convergence."""

from __future__ import annotations

import torch


def loss_plateau(history: list[torch.Tensor] | torch.Tensor, *, width: int = 8,
                 tolerance: float = 0.01, denominator_floor: float = 0.01
                 ) -> torch.Tensor:
    """Check two adjacent windows and the slope of the most recent window.

    Time is axis zero; all remaining axes are independent runs. Histories must
    use the same deterministic evaluation data at every checkpoint. This is an
    empirical stopping diagnostic, not proof of optimality or stationarity.
    """
    if width < 2 or tolerance <= 0 or denominator_floor <= 0:
        raise ValueError("invalid loss-plateau configuration")
    values = torch.stack(history) if isinstance(history, list) else history
    if values.ndim < 1 or not values.shape[0]:
        raise ValueError("a nonempty time-indexed loss history is required")
    if values.shape[0] < 2 * width:
        return torch.zeros_like(values[-1], dtype=torch.bool)
    values = values[-2 * width:].detach().to(dtype=torch.float64)
    before, after = values[:width], values[width:]
    denominator = torch.maximum(before.mean(0).abs(), after.mean(0).abs())
    denominator = denominator.clamp_min(denominator_floor)
    change = (after.mean(0) - before.mean(0)).abs() / denominator
    index = torch.arange(width, device=values.device, dtype=values.dtype)
    index -= (width - 1) / 2
    index = index.reshape((width,) + (1,) * (values.ndim - 1))
    slope = (after * index).sum(0) / index.square().sum(0)
    trend = slope.abs() * width / denominator
    finite = torch.isfinite(values).all(0)
    return finite & (change <= tolerance) & (trend <= tolerance)
