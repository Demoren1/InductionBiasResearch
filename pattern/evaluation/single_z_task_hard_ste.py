"""Task-z search with binary masks in forward and a soft-top-k STE backward."""

from __future__ import annotations

from typing import Any

import torch

import config
from data.generate import make_dataset
from evaluation.single_z_task import (
    _GRAD_CLIP_NORM,
    _assert_unchanged,
    _decode_soft,
    _freeze_and_snapshot,
    _fresh_mlp,
    _hard_from_logits,
    _mlp_forward,
    _per_network_bce,
    _project_ball_,
)


class _HardForwardSoftBackward(torch.autograd.Function):
    """Return hard values exactly while routing the backward pass to soft."""

    @staticmethod
    def forward(ctx, hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
        return hard

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[None, torch.Tensor]:
        return None, grad_output


def hard_ste(logits: torch.Tensor, soft: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (hard, STE): bit-exact hard forward, soft derivative backward."""
    hard = _hard_from_logits(logits)
    return hard, _HardForwardSoftBackward.apply(hard, soft)


def optimize_task_z_hard_ste(
    model: torch.nn.Module,
    initial_z: torch.Tensor,
    pattern: str,
    device: str | torch.device,
    *,
    outer_steps: int = 30,
    warmup_steps: int = 300,
    grad_steps: int = 100,
    z_lr: float = .05,
    radius: float = 4.,
    temperature: float = .5,
    seed: int = 20260912,
    val_seed_base: int = 80000,
) -> dict[str, Any]:
    """Adapt z while every MLP forward pass sees an exact binary top-k mask."""
    if outer_steps <= 0 or warmup_steps < 0 or grad_steps < 0:
        raise ValueError("invalid optimization budget")
    if z_lr <= 0 or radius <= 0 or temperature <= 0:
        raise ValueError("learning rates, radius, and temperature must be positive")
    if initial_z.ndim != 2 or initial_z.size(0) <= 0:
        raise ValueError("initial_z must have shape (n_starts, latent_dim)")
    if int(initial_z.size(1)) != int(getattr(model, "latent_dim")):
        raise ValueError("initial_z latent width differs from decoder")
    if int(getattr(model, "mask_dim")) != config.MASK_DIM:
        raise ValueError("decoder mask_dim differs from pattern mask dimension")

    device = torch.device(device)
    model_before = _freeze_and_snapshot(model, device)
    n_starts = int(initial_z.size(0))
    pat_int = int(pattern, 2)
    z = torch.nn.Parameter(initial_z.detach().to(device).clone())
    _project_ball_(z, radius)
    initial_z_cpu = z.detach().cpu().clone()
    outer_optimizer = torch.optim.Adam([z], lr=z_lr)

    validation = make_dataset(pattern, 1024, val_seed_base + pat_int, config.POS_FRACTION)
    x_val, y_val = validation["x"].to(device), validation["y"].to(device)
    with torch.no_grad():
        initial_logits, initial_soft = _decode_soft(model, z, temperature)
        initial_masks = _hard_from_logits(initial_logits)

    best_val_loss = z.new_full((n_starts,), float("inf"))
    best_val_z = z.detach().clone()
    best_val_soft = initial_soft.detach().clone()
    best_val_masks = initial_masks.detach().clone()
    best_val_outer = torch.full((n_starts,), -1, dtype=torch.long, device=device)
    history: list[dict[str, Any]] = []
    last_post_update_val: torch.Tensor | None = None

    for outer in range(outer_steps):
        outer_optimizer.zero_grad(set_to_none=True)
        initial_logits, initial_outer_soft = _decode_soft(model, z, temperature)
        detached_hard = _hard_from_logits(initial_logits).detach()
        mlp_seed = seed + pat_int * 1000 + outer
        params = _fresh_mlp(n_starts, mlp_seed, device)
        inner_optimizer = torch.optim.Adam(params, lr=config.LR)
        train_losses: list[float] = []

        # The warm-up is deliberately binary and detached from z.
        for step in range(warmup_steps):
            train_seed = 90_000_000 + pat_int * 1_000_000 + outer * 1000 + step
            data = make_dataset(pattern, config.TRAIN_BATCH_SIZE, train_seed, config.POS_FRACTION)
            xb, yb = data["x"].to(device), data["y"].to(device)
            loss_per_network = _per_network_bce(
                _mlp_forward(xb, *params, detached_hard), yb)
            inner_optimizer.zero_grad(set_to_none=True)
            loss_per_network.sum().backward()
            inner_optimizer.step()
            train_losses.append(float(loss_per_network.detach().mean()))

        # Forward values are hard; d(mask)/d(logits) is borrowed from soft top-k.
        for step in range(grad_steps):
            train_seed = (90_000_000 + pat_int * 1_000_000 + outer * 1000
                          + warmup_steps + step)
            data = make_dataset(pattern, config.TRAIN_BATCH_SIZE, train_seed, config.POS_FRACTION)
            xb, yb = data["x"].to(device), data["y"].to(device)
            logits, soft = _decode_soft(model, z, temperature)
            _, mask = hard_ste(logits, soft)
            loss_per_network = _per_network_bce(_mlp_forward(xb, *params, mask), yb)
            inner_optimizer.zero_grad(set_to_none=True)
            loss_per_network.sum().backward()
            inner_optimizer.step()
            train_losses.append(float(loss_per_network.detach().mean()))

        logits, soft = _decode_soft(model, z, temperature)
        hard, mask = hard_ste(logits, soft)
        pre_update_val = _per_network_bce(_mlp_forward(x_val, *params, mask), y_val)
        with torch.no_grad():
            improved = pre_update_val.detach() < best_val_loss
            best_val_loss = torch.where(improved, pre_update_val.detach(), best_val_loss)
            best_val_z[improved] = z.detach()[improved]
            best_val_soft[improved] = soft.detach()[improved]
            best_val_masks[improved] = hard.detach()[improved]
            best_val_outer[improved] = outer

        pre_update_val.sum().backward()
        if z.grad is None:
            raise AssertionError("STE objective did not produce a z gradient")
        raw_grad_norm = z.grad.detach().norm(dim=1)
        with torch.no_grad():
            scale = (_GRAD_CLIP_NORM / raw_grad_norm.clamp_min(
                torch.finfo(z.dtype).tiny)).clamp(max=1.0)
            z.grad.mul_(scale.unsqueeze(1))
        outer_optimizer.step()
        _project_ball_(z, radius)

        with torch.no_grad():
            post_logits, post_soft = _decode_soft(model, z, temperature)
            _, post_mask = hard_ste(post_logits, post_soft)
            post_update_val = _per_network_bce(
                _mlp_forward(x_val, *params, post_mask), y_val)
            last_post_update_val = post_update_val.detach().clone()
        history.append({
            "outer": outer,
            "train_loss_mean": train_losses,
            "pre_update_val_loss": pre_update_val.detach().cpu().tolist(),
            "post_update_val_loss": post_update_val.detach().cpu().tolist(),
            "raw_z_grad_norm": raw_grad_norm.cpu().tolist(),
            "z_norm_after_update": z.detach().norm(dim=1).cpu().tolist(),
        })
        if (outer + 1) % 5 == 0 or outer + 1 == outer_steps:
            print(f"[hard-ste] pattern={pattern} outer={outer+1}/{outer_steps} "
                  f"search_val={float(pre_update_val.mean()):.5f} "
                  f"mean_z_norm={float(z.detach().norm(dim=1).mean()):.3f}", flush=True)

    with torch.no_grad():
        final_logits, final_soft = _decode_soft(model, z, temperature)
        final_masks = _hard_from_logits(final_logits)
    _assert_unchanged(model, model_before)
    return {
        "initial_z": initial_z_cpu,
        "final_z": z.detach().cpu(),
        "initial_masks": initial_masks.detach().cpu(),
        "final_masks": final_masks.detach().cpu(),
        "final_soft": final_soft.detach().cpu(),
        "best_val_z": best_val_z.detach().cpu(),
        "best_val_masks": best_val_masks.detach().cpu(),
        "best_val_soft": best_val_soft.detach().cpu(),
        "best_val_loss": best_val_loss.detach().cpu(),
        "best_val_outer": best_val_outer.detach().cpu(),
        "final_post_update_val_loss": last_post_update_val.detach().cpu(),
        "history": history,
        "decoder_unchanged": True,
        "settings": {
            "n_starts": n_starts, "outer_steps": outer_steps,
            "warmup_steps": warmup_steps, "grad_steps": grad_steps,
            "z_lr": z_lr, "radius": radius, "temperature": temperature,
            "seed": seed, "val_seed_base": val_seed_base, "k_active": 32,
            "mask_semantics": "binary hard top-k forward; soft-top-k straight-through backward",
            "gradient_semantics": (
                "direct accumulated gradients; inner Adam updates are not unrolled"
            ),
        },
    }
