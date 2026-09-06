"""Agreement-only latent optimization for two frozen unconditional decoders.

The objective is to find independent latent codes ``z1`` and ``z2`` whose
fixed-cardinality *soft* masks agree after a permutation of hidden columns:
``mean((soft_topk(D1(z1)) - P soft_topk(D2(z2))) ** 2)``.  The Hungarian
assignment is computed from detached masks, while gradients still flow
through the gathered second mask.  This module deliberately does not import
training data, task labels, or a gold/Toeplitz mask: those are evaluation
quantities and must not influence the latent search.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from scipy.optimize import linear_sum_assignment


DEFAULT_K_ACTIVE = 32


class _SoftTopK(torch.autograd.Function):
    """Fixed-sum sigmoid projection with the implicit-gradient backward."""

    @staticmethod
    def forward(ctx: Any, logits: torch.Tensor, k: int,
                temperature: float) -> torch.Tensor:
        if logits.ndim < 1:
            raise ValueError("logits must have at least one dimension")
        width = logits.size(-1)
        if not 0 < k < width:
            raise ValueError(f"k must satisfy 0 < k < {width}, got {k}")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        # tau is solved independently for every leading index.  A +/-40T
        # bracket makes the residual below float32 precision at either end.
        lo = logits.detach().amin(dim=-1, keepdim=True) - 40.0 * temperature
        hi = logits.detach().amax(dim=-1, keepdim=True) + 40.0 * temperature
        target = torch.full_like(lo, float(k))
        for _ in range(64):
            tau = (lo + hi) * 0.5
            count = torch.sigmoid((logits.detach() - tau) / temperature).sum(
                dim=-1, keepdim=True)
            lo = torch.where(count > target, tau, lo)
            hi = torch.where(count > target, hi, tau)
        out = torch.sigmoid((logits - (lo + hi) * 0.5) / temperature)
        ctx.save_for_backward(out)
        ctx.temperature = temperature
        return out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        (out,) = ctx.saved_tensors
        # d y_i / d x_j with sum_i y_i constrained to k.  This is the exact
        # implicit derivative of the solved threshold, not a straight-through
        # approximation.
        weight = out * (1.0 - out)
        weighted_mean = (grad_output * weight).sum(dim=-1, keepdim=True)
        weighted_mean = weighted_mean / weight.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(out.dtype).tiny)
        grad_logits = weight * (grad_output - weighted_mean) / ctx.temperature
        return grad_logits, None, None


def soft_topk(logits: torch.Tensor, k: int, temperature: float) -> torch.Tensor:
    """Return a differentiable mask with every row summing exactly to ``k``.

    The last axis is the mask axis; all preceding axes are independent masks.
    """
    return _SoftTopK.apply(logits, int(k), float(temperature))


def hard_topk(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Return a binary top-``k`` mask along the last (including flat) axis."""
    if not 0 < k <= logits.size(-1):
        raise ValueError(f"k must satisfy 0 < k <= {logits.size(-1)}, got {k}")
    indices = logits.topk(k, dim=-1).indices
    result = torch.zeros_like(logits)
    return result.scatter(-1, indices, 1.0)


def align_columns(reference: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """Hungarian-align columns of ``other`` to ``reference``.

    Inputs are ``(N, L, H)`` masks, or one ``(L, H)`` mask.  The discrete
    assignment uses detached squared-L2 costs.  The returned tensor remains a
    gather from ``other``, so its ordinary autograd path is intact.
    """
    if reference.shape != other.shape or reference.ndim not in (2, 3):
        raise ValueError("reference and other must have equal (N,L,H) or (L,H) shapes")
    singleton = reference.ndim == 2
    if singleton:
        reference, other = reference.unsqueeze(0), other.unsqueeze(0)
    aligned = []
    for ref_i, other_i in zip(reference, other):
        diff = ref_i.detach().transpose(0, 1)[:, None, :] - other_i.detach().transpose(0, 1)[None, :, :]
        cost = diff.square().sum(dim=-1).cpu().numpy()
        rows, cols = linear_sum_assignment(cost)
        order = torch.empty(other_i.size(-1), dtype=torch.long, device=other_i.device)
        order[torch.as_tensor(rows, device=other_i.device)] = torch.as_tensor(cols, device=other_i.device)
        aligned.append(other_i.index_select(-1, order))
    result = torch.stack(aligned)
    return result.squeeze(0) if singleton else result


def _matrix_shape(mask_dim: int) -> tuple[int, int]:
    side = math.isqrt(mask_dim)
    if side * side != mask_dim:
        raise ValueError("decoder mask_dim must be square to align hidden columns")
    return side, side


def _freeze(model: torch.nn.Module, device: torch.device) -> None:
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _decode(model: torch.nn.Module, z: torch.Tensor) -> torch.Tensor:
    cond_dim = int(getattr(model, "cond_dim", 0))
    condition = z.new_zeros(z.size(0), cond_dim)
    return model.decode(z, condition)


def _project_ball_(z: torch.Tensor, radius: float) -> None:
    if radius <= 0:
        raise ValueError("radius must be positive")
    with torch.no_grad():
        norm = z.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(z.dtype).tiny)
        z.mul_((radius / norm).clamp(max=1.0))


def optimize_agreement(model1: torch.nn.Module, model2: torch.nn.Module,
                       n_starts: int = 64, steps: int = 1000, lr: float = .03,
                       seed: int = 20260906, temperature: float = .5,
                       radius: float = 8., device: str | torch.device = "cpu",
                       k: int = DEFAULT_K_ACTIVE) -> dict[str, Any]:
    """Optimize independent latents using only decoder-mask agreement.

    Models are put in eval mode and frozen.  Each restart is independent in a
    batch; its mean-squared mask loss is summed (not averaged) before Adam so
    the effective learning rate does not vary with ``n_starts``.  The returned
    ``final_*`` tensors are the best soft-agreement iterate per restart,
    including the initial draw.  No gold mask, task data, or downstream score
    participates in choosing them.
    """
    if n_starts <= 0 or steps < 0 or lr <= 0:
        raise ValueError("n_starts and lr must be positive; steps must be nonnegative")
    device = torch.device(device)
    _freeze(model1, device)
    _freeze(model2, device)
    latent1, latent2 = int(model1.latent_dim), int(model2.latent_dim)
    mask_dim1, mask_dim2 = int(model1.mask_dim), int(model2.mask_dim)
    if mask_dim1 != mask_dim2:
        raise ValueError("decoders must have equal mask_dim")
    rows, cols = _matrix_shape(mask_dim1)
    if not 0 < k < mask_dim1:
        raise ValueError(f"k must satisfy 0 < k < {mask_dim1}")
    generator = torch.Generator(device=device).manual_seed(seed)
    z1 = torch.nn.Parameter(torch.randn(n_starts, latent1, generator=generator, device=device))
    z2 = torch.nn.Parameter(torch.randn(n_starts, latent2, generator=generator, device=device))
    _project_ball_(z1, radius)
    _project_ball_(z2, radius)
    initial_z1, initial_z2 = z1.detach().clone(), z2.detach().clone()

    def masks_and_loss() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits1, logits2 = _decode(model1, z1), _decode(model2, z2)
        soft1 = soft_topk(logits1, k, temperature).reshape(n_starts, rows, cols)
        soft2 = soft_topk(logits2, k, temperature).reshape(n_starts, rows, cols)
        aligned2 = align_columns(soft1, soft2)
        return soft1, soft2, aligned2, (soft1 - aligned2).square().mean(dim=(1, 2))

    with torch.no_grad():
        initial_soft1, initial_soft2, initial_aligned2, initial_loss = masks_and_loss()
        best_loss = initial_loss.clone()
        best_z1, best_z2 = z1.detach().clone(), z2.detach().clone()
    history = [initial_loss.detach().cpu()]
    optimizer = torch.optim.Adam([z1, z2], lr=lr)
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        _, _, _, loss = masks_and_loss()
        loss.sum().backward()
        optimizer.step()
        _project_ball_(z1, radius)
        _project_ball_(z2, radius)
        with torch.no_grad():
            _, _, _, after_loss = masks_and_loss()
            improved = after_loss < best_loss
            best_loss = torch.where(improved, after_loss, best_loss)
            best_z1 = torch.where(improved[:, None], z1.detach(), best_z1)
            best_z2 = torch.where(improved[:, None], z2.detach(), best_z2)
        history.append(after_loss.detach().cpu())
        if (step + 1) % 100 == 0 or step + 1 == steps:
            print(f"[decoder-agreement] step={step + 1}/{steps} "
                  f"mean_loss={after_loss.mean().item():.6f} "
                  f"best={best_loss.mean().item():.6f}", flush=True)

    with torch.no_grad():
        z1.copy_(best_z1)
        z2.copy_(best_z2)
        final_soft1, final_soft2, final_aligned2, final_loss = masks_and_loss()
        initial_hard1 = hard_topk(_decode(model1, initial_z1), k).reshape(n_starts, rows, cols)
        initial_hard2 = hard_topk(_decode(model2, initial_z2), k).reshape(n_starts, rows, cols)
        final_hard1 = hard_topk(_decode(model1, best_z1), k).reshape(n_starts, rows, cols)
        final_hard2 = hard_topk(_decode(model2, best_z2), k).reshape(n_starts, rows, cols)
        initial_hard_loss = (initial_hard1 - align_columns(initial_hard1, initial_hard2)).square().mean(dim=(1, 2))
        final_hard_loss = (final_hard1 - align_columns(final_hard1, final_hard2)).square().mean(dim=(1, 2))

    def cpu(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().cpu()

    return {
        "initial_z1": cpu(initial_z1), "initial_z2": cpu(initial_z2),
        "final_z1": cpu(best_z1), "final_z2": cpu(best_z2),
        "initial_masks1": cpu(initial_hard1), "initial_masks2": cpu(initial_hard2),
        "final_masks1": cpu(final_hard1), "final_masks2": cpu(final_hard2),
        "initial_soft_masks1": cpu(initial_soft1), "initial_soft_masks2": cpu(initial_soft2),
        "soft_final_masks1": cpu(final_soft1), "soft_final_masks2": cpu(final_soft2),
        # Short aliases are convenient for downstream diagnostics such as
        # softness S(1-S); the longer names above make their mask role clear.
        "initial_soft1": cpu(initial_soft1), "initial_soft2": cpu(initial_soft2),
        "final_soft1": cpu(final_soft1), "final_soft2": cpu(final_soft2),
        "initial_loss": cpu(initial_loss), "final_loss": cpu(final_loss),
        "per_start_initial_loss": cpu(initial_loss), "per_start_final_loss": cpu(final_loss),
        "per_start_initial_hard_loss": cpu(initial_hard_loss),
        "per_start_final_hard_loss": cpu(final_hard_loss),
        "losses": {"initial": cpu(initial_loss), "final": cpu(final_loss),
                   "initial_hard": cpu(initial_hard_loss), "final_hard": cpu(final_hard_loss)},
        "history": [row.tolist() for row in history],
    }
