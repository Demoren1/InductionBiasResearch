"""Agreement-only latent optimization for three or more frozen decoders.

The search aligns every decoder mask to the first decoder and minimizes the
variance around their common mean.  Gold/Toeplitz masks and task labels are
deliberately absent from this module; they are post-hoc evaluation data.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from evaluation.decoder_agreement import (
    _decode, _freeze, _matrix_shape, _project_ball_, align_columns, hard_topk,
    soft_topk,
)


def _aligned_stack(masks: Sequence[torch.Tensor]) -> torch.Tensor:
    """Return ``(models, starts, rows, cols)`` masks in one column ordering."""
    if len(masks) < 2:
        raise ValueError("at least two masks are required")
    reference = masks[0]
    return torch.stack([reference, *[align_columns(reference, x) for x in masks[1:]]])


def _ensemble_loss(aligned: torch.Tensor) -> torch.Tensor:
    """Per-start variance of aligned masks around their ensemble mean."""
    center = aligned.mean(dim=0, keepdim=True)
    return (aligned - center).square().mean(dim=(0, 2, 3))


def optimize_multi_agreement(
    models: Sequence[torch.nn.Module], n_starts: int = 64, steps: int = 2000,
    lr: float = .03, seed: int = 20260919, temperature: float = .5,
    radius: float = 12., device: str | torch.device = "cpu", k: int = 32,
    progress_every: int = 100,
) -> dict[str, Any]:
    """Optimize one independent latent per decoder using ensemble agreement.

    The best joint state is retained independently for every start.  Selection
    uses only ensemble soft-mask variance, including the initial state.
    """
    if len(models) < 3:
        raise ValueError("multi-decoder agreement requires at least three models")
    if n_starts <= 0 or steps < 0 or lr <= 0:
        raise ValueError("n_starts and lr must be positive; steps must be nonnegative")
    device = torch.device(device)
    for model in models:
        _freeze(model, device)
    mask_dims = [int(model.mask_dim) for model in models]
    if len(set(mask_dims)) != 1:
        raise ValueError("decoders must have equal mask_dim")
    rows, cols = _matrix_shape(mask_dims[0])
    if not 0 < k < mask_dims[0]:
        raise ValueError(f"k must satisfy 0 < k < {mask_dims[0]}")

    generator = torch.Generator(device=device).manual_seed(seed)
    z = [torch.nn.Parameter(torch.randn(
        n_starts, int(model.latent_dim), generator=generator, device=device,
    )) for model in models]
    for latent in z:
        _project_ball_(latent, radius)
    initial_z = torch.stack([latent.detach().clone() for latent in z])

    def masks_and_loss() -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        soft = [soft_topk(_decode(model, latent), k, temperature).reshape(
            n_starts, rows, cols,
        ) for model, latent in zip(models, z)]
        aligned = _aligned_stack(soft)
        return soft, aligned, _ensemble_loss(aligned)

    with torch.no_grad():
        initial_soft, initial_aligned, initial_loss = masks_and_loss()
        best_loss = initial_loss.clone()
        best_z = torch.stack([latent.detach().clone() for latent in z])
    history = [initial_loss.detach().cpu()]
    optimizer = torch.optim.Adam(z, lr=lr)
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        _, _, loss = masks_and_loss()
        loss.sum().backward()
        optimizer.step()
        for latent in z:
            _project_ball_(latent, radius)
        with torch.no_grad():
            _, _, after_loss = masks_and_loss()
            improved = after_loss < best_loss
            best_loss = torch.where(improved, after_loss, best_loss)
            current = torch.stack([latent.detach() for latent in z])
            best_z = torch.where(improved[None, :, None], current, best_z)
        history.append(after_loss.detach().cpu())
        if progress_every and ((step + 1) % progress_every == 0 or step + 1 == steps):
            print(f"[multi-agreement] models={len(models)} step={step + 1}/{steps} "
                  f"mean_loss={after_loss.mean().item():.6f} "
                  f"best={best_loss.mean().item():.6f}", flush=True)

    with torch.no_grad():
        for latent, best in zip(z, best_z):
            latent.copy_(best)
        final_soft, final_aligned, final_loss = masks_and_loss()
        initial_hard = torch.stack([
            hard_topk(_decode(model, latent), k).reshape(n_starts, rows, cols)
            for model, latent in zip(models, initial_z)
        ])
        final_hard = torch.stack([
            hard_topk(_decode(model, latent), k).reshape(n_starts, rows, cols)
            for model, latent in zip(models, best_z)
        ])
        initial_hard_loss = _ensemble_loss(_aligned_stack(list(initial_hard)))
        final_hard_loss = _ensemble_loss(_aligned_stack(list(final_hard)))

    def cpu(value: torch.Tensor) -> torch.Tensor:
        return value.detach().cpu()

    return {
        "initial_z": cpu(initial_z), "final_z": cpu(best_z),
        "initial_masks": cpu(initial_hard), "final_masks": cpu(final_hard),
        "initial_soft_masks": cpu(torch.stack(initial_soft)),
        "final_soft_masks": cpu(torch.stack(final_soft)),
        "initial_aligned_soft_masks": cpu(initial_aligned),
        "final_aligned_soft_masks": cpu(final_aligned),
        "initial_loss": cpu(initial_loss), "final_loss": cpu(final_loss),
        "initial_hard_loss": cpu(initial_hard_loss),
        "final_hard_loss": cpu(final_hard_loss),
        "history": [row.tolist() for row in history],
    }
