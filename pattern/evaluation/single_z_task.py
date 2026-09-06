"""Task-loss latent search through one frozen pattern VAE decoder.

This is the matched single-decoder baseline for ``decoder_agreement``.  A
batch contains independent latent codes and independent downstream MLPs, but
all of them see the same seeded training stream at a given inner step.  The
outer update accumulates *direct* derivatives through the live soft mask from
the last inner training steps and a validation BCE.  Optimizer updates of the
inner MLP happen in place, so this is deliberately not an unrolled/bilevel
hypergradient.

The module does not import an ideal mask or downstream test data.  It only
uses task samples required by the task-loss search itself.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

import config
from data.generate import make_dataset
from evaluation.decoder_agreement import hard_topk, soft_topk


_K_ACTIVE = 32
_GRAD_CLIP_NORM = 10.0


def _freeze_and_snapshot(model: torch.nn.Module, device: torch.device) -> dict[str, torch.Tensor]:
    """Freeze ``model`` and take a bit-exact state snapshot for the assertion."""
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _assert_unchanged(model: torch.nn.Module, before: dict[str, torch.Tensor]) -> None:
    for name, value in model.state_dict().items():
        if not torch.equal(before[name], value):
            raise AssertionError(f"Frozen decoder state changed: {name}")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError("Frozen decoder received gradients")


def _project_ball_(z: torch.Tensor, radius: float) -> None:
    with torch.no_grad():
        norm = z.norm(dim=1, keepdim=True).clamp_min(torch.finfo(z.dtype).tiny)
        z.mul_((radius / norm).clamp(max=1.0))


def _fresh_mlp(n_mlps: int, seed: int, device: torch.device) -> list[torch.nn.Parameter]:
    """Create one independent first/second-layer MLP per latent code."""
    generator = torch.Generator(device=device).manual_seed(seed)
    w1 = torch.nn.Parameter(torch.randn(
        n_mlps, config.SEQ_LEN, config.H, generator=generator, device=device) * .1)
    b1 = torch.nn.Parameter(torch.zeros(n_mlps, config.H, device=device))
    w2 = torch.nn.Parameter(torch.randn(
        n_mlps, config.H, 1, generator=generator, device=device) * .1)
    b2 = torch.nn.Parameter(torch.zeros(n_mlps, 1, device=device))
    return [w1, b1, w2, b2]


def _mlp_forward(x: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor,
                 w2: torch.Tensor, b2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return logits shaped ``(batch, n_mlps)`` for a per-MLP soft mask."""
    hidden = F.relu(torch.einsum("bl,nlh->bnh", x, w1 * mask) + b1)
    return torch.einsum("bnh,nh->bn", hidden, w2.squeeze(-1)) + b2.squeeze(-1)


def _per_network_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    expanded = targets.unsqueeze(1).expand_as(logits)
    return F.binary_cross_entropy_with_logits(logits, expanded, reduction="none").mean(dim=0)


def _decode_soft(model: torch.nn.Module, z: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    condition = z.new_zeros(z.size(0), int(getattr(model, "cond_dim", 0)))
    logits = model.decode(z, condition)
    soft = soft_topk(logits, _K_ACTIVE, temperature).reshape(z.size(0), config.SEQ_LEN, config.H)
    return logits, soft


def _hard_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return hard_topk(logits, _K_ACTIVE).reshape(-1, config.SEQ_LEN, config.H)


def optimize_task_z(model: torch.nn.Module, initial_z: torch.Tensor, pattern: str,
                    device: str | torch.device, *, outer_steps: int = 30,
                    warmup_steps: int = 300, grad_steps: int = 100,
                    z_lr: float = .05, radius: float = 8., temperature: float = .5,
                    seed: int = 20260906, val_seed_base: int = 80000) -> dict[str, Any]:
    """Adapt independent latents using direct task-loss gradients only.

    The final code is the legacy recipe's last outer iterate.  ``best_val_*``
    is supplied separately as a validation-loss witness from the pre-update
    outer iterates; neither masks nor validation values are replaced using a
    gold structure or a downstream test result.
    """
    if outer_steps <= 0 or warmup_steps < 0 or grad_steps < 0:
        raise ValueError("outer_steps must be positive; warmup_steps and grad_steps nonnegative")
    if z_lr <= 0 or radius <= 0 or temperature <= 0:
        raise ValueError("z_lr, radius, and temperature must be positive")
    if initial_z.ndim != 2 or initial_z.size(0) <= 0:
        raise ValueError("initial_z must have shape (n_starts, latent_dim)")
    if int(initial_z.size(1)) != int(getattr(model, "latent_dim")):
        raise ValueError("initial_z latent width differs from the decoder")
    if int(getattr(model, "mask_dim")) != config.MASK_DIM:
        raise ValueError("decoder mask_dim differs from the pattern mask dimension")

    device = torch.device(device)
    model_before = _freeze_and_snapshot(model, device)
    n_starts = int(initial_z.size(0))
    pat_int = int(pattern, 2)
    z = torch.nn.Parameter(initial_z.detach().to(device).clone())
    _project_ball_(z, radius)
    initial_z_cpu = z.detach().cpu().clone()
    outer_optimizer = torch.optim.Adam([z], lr=z_lr)

    # This validation set is part of the search objective, never a final
    # downstream evaluation.  It is fixed across outer iterations.
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
        _, initial_outer_soft = _decode_soft(model, z, temperature)
        detached_mask = initial_outer_soft.detach()
        mlp_seed = seed + pat_int * 1000 + outer
        params = _fresh_mlp(n_starts, mlp_seed, device)
        inner_optimizer = torch.optim.Adam(params, lr=config.LR)
        train_losses: list[float] = []

        # Warm-up trains the MLP but cannot create a path to z.
        for step in range(warmup_steps):
            train_seed = 90_000_000 + pat_int * 1_000_000 + outer * 1000 + step
            data = make_dataset(pattern, config.TRAIN_BATCH_SIZE, train_seed, config.POS_FRACTION)
            xb, yb = data["x"].to(device), data["y"].to(device)
            loss_per_network = _per_network_bce(_mlp_forward(xb, *params, detached_mask), yb)
            inner_optimizer.zero_grad(set_to_none=True)
            # Sum gives every independently parameterized MLP the same update
            # it would receive in a standalone n=1 run.
            loss_per_network.sum().backward()
            inner_optimizer.step()
            train_losses.append(float(loss_per_network.detach().mean()))

        # Direct-path phase: inner optimizer steps are in-place and are not
        # differentiated through, while each live-mask loss contributes to z.
        for step in range(grad_steps):
            train_seed = (90_000_000 + pat_int * 1_000_000 + outer * 1000
                          + warmup_steps + step)
            data = make_dataset(pattern, config.TRAIN_BATCH_SIZE, train_seed, config.POS_FRACTION)
            xb, yb = data["x"].to(device), data["y"].to(device)
            _, live_soft = _decode_soft(model, z, temperature)
            loss_per_network = _per_network_bce(_mlp_forward(xb, *params, live_soft), yb)
            inner_optimizer.zero_grad(set_to_none=True)
            loss_per_network.sum().backward()
            inner_optimizer.step()
            train_losses.append(float(loss_per_network.detach().mean()))

        # This loss is captured immediately before the z update and is the
        # sole criterion for the auxiliary best-validation witness.
        logits, soft = _decode_soft(model, z, temperature)
        pre_update_val = _per_network_bce(_mlp_forward(x_val, *params, soft), y_val)
        hard = _hard_from_logits(logits)
        with torch.no_grad():
            improved = pre_update_val.detach() < best_val_loss
            best_val_loss = torch.where(improved, pre_update_val.detach(), best_val_loss)
            best_val_z[improved] = z.detach()[improved]
            best_val_soft[improved] = soft.detach()[improved]
            best_val_masks[improved] = hard.detach()[improved]
            best_val_outer[improved] = outer

        pre_update_val.sum().backward()
        if z.grad is None:
            raise AssertionError("Task-loss objective did not produce a z gradient")
        raw_grad_norm = z.grad.detach().norm(dim=1)
        with torch.no_grad():
            scale = (_GRAD_CLIP_NORM / raw_grad_norm.clamp_min(torch.finfo(z.dtype).tiny)).clamp(max=1.0)
            z.grad.mul_(scale.unsqueeze(1))
        outer_optimizer.step()
        _project_ball_(z, radius)

        # The last post-update value uses its final inner MLP state.  It is
        # reported but deliberately does not select the best-validation code.
        with torch.no_grad():
            _, post_soft = _decode_soft(model, z, temperature)
            post_update_val = _per_network_bce(_mlp_forward(x_val, *params, post_soft), y_val)
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
            print(f"[single-z] pattern={pattern} outer={outer+1}/{outer_steps} "
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
            "seed": seed, "val_seed_base": val_seed_base,
            "k_active": _K_ACTIVE, "gradient_semantics": (
                "direct gradients from live-mask inner losses plus pre-update validation BCE; "
                "inner Adam updates are in-place and not unrolled"),
            "selection": "final_z is the last outer iterate; best_val_z uses only pre-update search validation BCE",
        },
    }
