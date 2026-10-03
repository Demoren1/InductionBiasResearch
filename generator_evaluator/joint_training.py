"""Interleaved own-task updates for joint cooperative mask generators."""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from .cooperative_policy import align_elite_to_logits
from .generator_objectives import joint_mask_agreement, reconstruct_bank_masks
from .training import generator_update


def _distil_measured_targets(model, tokens: Tensor, quality: Tensor | None,
                             optimizer: torch.optim.Optimizer, targets: Tensor,
                             k: int, rng: torch.Generator | None, weight: float,
                             limit: int) -> tuple[float, int]:
    """Distil randomly sampled real TRAIN elites at fresh, independent noise."""
    if not len(targets) or weight == 0:
        return 0.0, 0
    parameter = next(model.parameters(), None)
    if parameter is None:
        raise ValueError("generator must have parameters")
    device, dtype = parameter.device, parameter.dtype
    tokens = tokens.to(device=device, dtype=dtype)
    quality = None if quality is None else quality.to(device=device, dtype=dtype)
    elites = torch.as_tensor(targets, device=device, dtype=dtype)
    elites = elites[elites.sum((1, 2)) == k][:limit]
    if not len(elites):
        return 0.0, 0

    draws = 2
    rng_device = device if rng is None else torch.device(rng.device)
    noise = torch.randn((draws, model.noise_dim), device=rng_device,
                        dtype=torch.float32, generator=rng).to(device=device, dtype=dtype)
    bank_tokens = tokens.unsqueeze(0) if tokens.ndim == 3 else tokens
    bank_quality = (None if quality is None else
                    quality.unsqueeze(0) if quality.ndim == 2 else quality)
    if bank_tokens.shape[0] != 1:
        raise ValueError("joint updates require one functional bank per generator")
    logits = model(
        bank_tokens.expand(draws, *bank_tokens.shape[1:]), noise,
        None if bank_quality is None else bank_quality.expand(draws, *bank_quality.shape[1:]),
    )
    selected = torch.randint(len(elites), (draws,), device=rng_device, generator=rng).to(device)
    aligned = torch.stack([
        align_elite_to_logits(elites[row], logits[index])
        for index, row in enumerate(selected.tolist())
    ])
    loss = F.binary_cross_entropy_with_logits(logits, aligned)
    optimizer.zero_grad(set_to_none=True)
    (weight * loss).backward()
    torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad], max_norm=10.0)
    optimizer.step()
    return float(loss.detach().cpu()), len(elites)


def joint_generator_update(*, models: Mapping[str, torch.nn.Module],
                           banks: Mapping[str, object],
                           optimizers: Mapping[str, torch.optim.Optimizer],
                           training_views: Mapping[str, object], ensemble: torch.nn.Module,
                           contexts: Tensor, dense_quality: Tensor,
                           targets: Tensor, output_k: int,
                           own_rngs: Mapping[str, torch.Generator],
                           agreement_rng: torch.Generator | None,
                           update_ordinal: int, updates_per_epoch: int,
                           agreement_weight: float, agreement_ramp_epochs: int,
                           elite_weight: float, elite_limit: int,
                           reconstruction_weight: float,
                           reconstruction_batch_size: int,
                           permutation_weight: float) -> dict[str, dict[str, float | str]]:
    """Run the original ordered quality, elite, reconstruction, agreement steps.

    Every policy update receives its own task context. Measured targets use two
    fresh random latent draws and are selected independently for each generator.
    Agreement then uses a new common set of random latent draws, with one
    optimizer step per generator. No trainable or fixed shared latent is used.
    """
    names = tuple(models)
    if (len(names) < 2 or set(names) != set(banks) or set(names) != set(optimizers) or
            set(names) != set(training_views) or set(names) != set(own_rngs)):
        raise ValueError("joint update models, banks, optimizers, views and RNGs must match")
    if contexts.ndim != 2 or dense_quality.shape != (len(names),):
        raise ValueError("joint update contexts and dense qualities must match the generators")
    if update_ordinal < 0 or updates_per_epoch < 1 or agreement_ramp_epochs < 1:
        raise ValueError("invalid joint update schedule position")

    rows: dict[str, dict[str, float | str]] = {}
    for index, name in enumerate(names):
        model, view = models[name], training_views[name]
        parameter = next(model.parameters(), None)
        if parameter is None:
            raise ValueError(f"generator {name!r} must have parameters")
        device, dtype = parameter.device, parameter.dtype
        view_tokens = view.tokens.to(device=device, dtype=dtype)
        view_quality = (None if view.quality is None else
                        view.quality.to(device=device, dtype=dtype))
        if hasattr(model, "set_budget"):
            model.set_budget(output_k)  # type: ignore[attr-defined]
        logs = generator_update(
            model, ensemble, view_tokens, view_quality,
            contexts[index:index + 1], dense_quality[index:index + 1],
            optimizers[name], output_k, own_rngs[name],
            permutation_weight=permutation_weight,
            # Policy learning is own-task raw delta. The configured composite
            # objective applies to global pool acquisition and selection.
            quality_objective="worst",
        )
        distillation_loss, target_count = _distil_measured_targets(
            model, view_tokens, view_quality, optimizers[name], targets, output_k,
            own_rngs[name], elite_weight, elite_limit)
        reconstruction = reconstruct_bank_masks(
            model, banks[name], optimizers[name], rng=own_rngs[name],
            batch_size=reconstruction_batch_size, weight=reconstruction_weight)
        rows[name] = dict(
            **logs,
            elite_distillation_loss=distillation_loss,
            elite_count=float(target_count),
            distillation_target_count=float(target_count),
            **reconstruction,
            output_k=float(output_k),
            ordinal=float(update_ordinal),
        )

    ramp = min(1.0, (update_ordinal + 1) /
               (agreement_ramp_epochs * updates_per_epoch))
    effective_agreement_weight = agreement_weight * ramp
    agreement = joint_mask_agreement(
        [models[name] for name in names], [banks[name] for name in names],
        [optimizers[name] for name in names], output_k,
        rng=agreement_rng, weight=effective_agreement_weight, sample_count=2,
    )
    for name in names:
        rows[name].update(agreement)
    return rows
