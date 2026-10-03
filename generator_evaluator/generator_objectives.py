"""Auxiliary objectives shared by cooperative mask generators."""
from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .cooperative_policy import _hungarian, align_elite_to_logits


def _weight(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _model_info(model: nn.Module) -> tuple[int, int, int, int | None]:
    try:
        features = int(model.features)  # type: ignore[attr-defined]
        hidden = int(model.hidden)  # type: ignore[attr-defined]
        noise_dim = int(model.noise_dim)  # type: ignore[attr-defined]
        token_dim = getattr(model, "token_dim", None)
        token_dim = None if token_dim is None else int(token_dim)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("generator must expose features, hidden, and noise_dim") from exc
    if min(features, hidden, noise_dim) < 1 or (token_dim is not None and token_dim < 1):
        raise ValueError("generator dimensions must be positive")
    return features, hidden, noise_dim, token_dim


def _device_dtype(model: nn.Module) -> tuple[torch.device, torch.dtype]:
    parameter = next(model.parameters(), None)
    if parameter is None or not parameter.is_floating_point():
        raise ValueError("generator must have floating-point parameters")
    return parameter.device, parameter.dtype


def _validate_optimizer(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch.optim.Optimizer")
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("generator has no trainable parameters")


def _bank_tensors(bank: Any, model: nn.Module) -> tuple[Tensor, Tensor | None, Tensor]:
    tokens = getattr(bank, "tokens", None)
    quality = getattr(bank, "quality", None)
    masks = getattr(bank, "masks", None)
    if not isinstance(tokens, Tensor) or tokens.ndim != 4 or tokens.shape[0] != 1:
        raise ValueError("bank.tokens must have shape [1, R, H, D]")
    if not tokens.is_floating_point() or not bool(torch.isfinite(tokens).all()):
        raise ValueError("bank.tokens must be finite floating-point values")
    if not isinstance(masks, Tensor) or masks.ndim != 3:
        raise ValueError("bank.masks must have shape [R, features, hidden]")
    if not masks.is_floating_point() or not bool(torch.isfinite(masks).all()):
        raise ValueError("bank.masks must be finite floating-point values")
    rows, features, hidden = masks.shape
    if rows < 1 or tokens.shape[1] != rows or tokens.shape[2] != hidden:
        raise ValueError("bank token and mask rows have incompatible dimensions")
    if not bool(((masks == 0) | (masks == 1)).all()):
        raise ValueError("bank masks must be binary")
    if bool((masks.sum((1, 2)) <= 0).any()):
        raise ValueError("each bank mask must have at least one active edge")
    expected_features, expected_hidden, _, token_dim = _model_info(model)
    if (features, hidden) != (expected_features, expected_hidden):
        raise ValueError("bank mask dimensions must match the generator output")
    if token_dim is not None and tokens.shape[-1] != token_dim:
        raise ValueError("bank token dimension must match the generator")
    if quality is not None:
        if (not isinstance(quality, Tensor) or quality.ndim != 3 or
                quality.shape[:2] != tokens.shape[:2] or not quality.is_floating_point() or
                not bool(torch.isfinite(quality).all())):
            raise ValueError("bank.quality must be finite [1, R, quality_dim] values")
    return tokens, quality, masks


def _rng_device(rng: torch.Generator | None, fallback: torch.device) -> torch.device:
    return fallback if rng is None else torch.device(rng.device)


def _indices(size: int, count: int, rng: torch.Generator | None,
             output_device: torch.device) -> Tensor:
    sample_device = _rng_device(rng, output_device)
    return torch.randint(size, (count,), generator=rng, device=sample_device).to(output_device)


def _noise(count: int, noise_dim: int, rng: torch.Generator | None,
           device: torch.device, dtype: torch.dtype) -> Tensor:
    sample_device = _rng_device(rng, device)
    values = torch.randn((count, noise_dim), generator=rng, device=sample_device, dtype=torch.float32)
    return values.to(device=device, dtype=dtype)


def _forward(model: nn.Module, tokens: Tensor, noise: Tensor,
             quality: Tensor | None, density: Tensor) -> Tensor:
    logits = model(tokens, noise, quality, density=density)
    expected = (tokens.shape[0], model.features, model.hidden)  # type: ignore[attr-defined]
    if not isinstance(logits, Tensor) or logits.shape != expected:
        raise ValueError("generator must return [batch, features, hidden] logits")
    if not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
        raise ValueError("generator logits must be finite floating-point values")
    return logits


def _balanced_bce(logits: Tensor, target: Tensor) -> Tensor:
    """Balance present positive/negative classes within each example."""
    element_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    example_losses: list[Tensor] = []
    for row in range(len(target)):
        positive = target[row] >= 0.5
        negative = ~positive
        class_losses = []
        if bool(positive.any()):
            class_losses.append(element_loss[row][positive].mean())
        if bool(negative.any()):
            class_losses.append(element_loss[row][negative].mean())
        # A dense mask has only the positive class and remains a valid target.
        example_losses.append(torch.stack(class_losses).mean())
    return torch.stack(example_losses).mean()


def _topk_overlap(logits: Tensor, targets: Tensor) -> float:
    overlaps = []
    for row in range(len(targets)):
        target = targets[row].detach().flatten()
        count = int(target.sum().item())
        prediction = logits[row].detach().flatten().topk(count).indices
        overlaps.append(target.index_select(0, prediction).sum() / count)
    return float(torch.stack(overlaps).mean().cpu())


def reconstruct_bank_masks(
    model: nn.Module,
    bank: Any,
    optimizer: torch.optim.Optimizer,
    *,
    rng: torch.Generator | None = None,
    batch_size: int = 8,
    weight: float = 1.0,
    accumulate: bool = False,
    bank_consensus: bool = False,
) -> dict[str, float | str]:
    """Reconstruct masks paired with randomly sampled profiles from ``bank``.

    Each sampled teacher is presented as its own one-map bank. Its mask density
    conditions that example, so sparse and dense teachers can share a batch
    without changing the generator's persistent budget.  With
    ``bank_consensus=True``, the complete bank is instead the input and its
    data-derived ``baseline_mask`` is the target.  This second mode bridges
    single-teacher pretraining to the full-bank input used during proposals.
    """
    count = _positive_int(batch_size, "batch_size")
    if not isinstance(bank_consensus, bool):
        raise TypeError("bank_consensus must be a bool")
    objective_weight = _weight(weight, "weight")
    _validate_optimizer(model, optimizer)
    source_tokens, source_quality, source_masks = _bank_tensors(bank, model)
    device, dtype = _device_dtype(model)
    if bank_consensus:
        count = min(count, 2)
        baseline = getattr(bank, "baseline_mask", None)
        if (not isinstance(baseline, Tensor) or baseline.shape != source_masks.shape[1:] or
                not baseline.is_floating_point() or not bool(torch.isfinite(baseline).all()) or
                not bool(((baseline == 0) | (baseline == 1)).all()) or
                not bool((baseline.sum() > 0))):
            raise ValueError("bank.baseline_mask must be a nonempty finite binary [features, hidden] mask")
        tokens = source_tokens.to(device=device, dtype=dtype).expand(count, *source_tokens.shape[1:])
        targets = baseline.to(device=device, dtype=dtype).unsqueeze(0).expand(count, -1, -1)
        quality = (None if source_quality is None else
                   source_quality.to(device=device, dtype=dtype).expand(count, *source_quality.shape[1:]))
    else:
        selected = _indices(len(source_masks), count, rng, source_tokens.device)
        tokens = source_tokens[0].index_select(0, selected).unsqueeze(1).to(device=device, dtype=dtype)
        targets = source_masks.index_select(0, selected).to(device=device, dtype=dtype)
        quality = None
        if source_quality is not None:
            quality = source_quality[0].index_select(0, selected).unsqueeze(1).to(device=device, dtype=dtype)
    total_edges = targets.shape[1] * targets.shape[2]
    density = targets.sum((1, 2)) / total_edges
    if bool((density <= 0).any()) or bool((density > 1).any()):
        raise ValueError("each bank mask must have density in (0, 1]")
    noise = _noise(count, model.noise_dim, rng, device, dtype)  # type: ignore[attr-defined]
    logits = _forward(model, tokens, noise, quality, density)
    aligned = torch.stack([align_elite_to_logits(targets[row], logits[row])
                           for row in range(count)])
    reconstruction = _balanced_bce(logits, aligned)
    overlap = _topk_overlap(logits, aligned)

    if objective_weight > 0:
        if not accumulate:
            optimizer.zero_grad(set_to_none=True)
        (objective_weight * reconstruction).backward()
        if not accumulate:
            parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=10.0)
            optimizer.step()
    return {
        "reconstruction_loss": float(reconstruction.detach().cpu()),
        "reconstruction_overlap": overlap,
        "reconstruction_scope": "bank_consensus" if bank_consensus else "teacher",
    }


def _soft_exact_mass(logits: Tensor, k: int) -> Tensor:
    """Sigmoid relaxation with approximately ``k`` mass per flattened row."""
    flat = logits.reshape(len(logits), -1)
    if k == flat.shape[1]:
        return torch.ones_like(logits) + logits * 0.0
    detached = flat.detach()
    temperature = detached.std(dim=1, unbiased=False).clamp_min(0.05) * 0.5
    low = detached.amin(dim=1) - 32.0 * temperature
    high = detached.amax(dim=1) + 32.0 * temperature
    for _ in range(40):
        threshold = (low + high) * 0.5
        mass = torch.sigmoid((detached - threshold[:, None]) / temperature[:, None]).sum(dim=1)
        too_much = mass > k
        low = torch.where(too_much, threshold, low)
        high = torch.where(too_much, high, threshold)
    threshold = ((low + high) * 0.5).detach()
    return torch.sigmoid((flat - threshold[:, None]) / temperature[:, None]).reshape_as(logits)


def _topk_straight_through(logits: Tensor, k: int) -> tuple[Tensor, Tensor]:
    detached = logits.detach().reshape(len(logits), -1)
    hard_flat = torch.zeros_like(detached)
    hard_flat.scatter_(1, detached.topk(k, dim=1).indices, 1.0)
    hard = hard_flat.reshape_as(logits)
    soft = _soft_exact_mass(logits, k)
    return hard - soft.detach() + soft, hard


def _generator_inputs(
    models: Sequence[nn.Module],
    banks: Sequence[Any],
    optimizers: Sequence[torch.optim.Optimizer],
) -> tuple[tuple[nn.Module, ...], tuple[Any, ...], tuple[torch.optim.Optimizer, ...]]:
    if not isinstance(models, Sequence) or len(models) < 2:
        raise ValueError("models must contain at least two items")
    count = len(models)
    if (not isinstance(banks, Sequence) or not isinstance(optimizers, Sequence) or
            len(banks) != count or len(optimizers) != count):
        raise ValueError("models, banks, and optimizers must contain the same number of items")
    return tuple(models), tuple(banks), tuple(optimizers)


def joint_mask_agreement(
    models: Sequence[nn.Module],
    banks: Sequence[Any],
    optimizers: Sequence[torch.optim.Optimizer],
    k: int,
    *,
    rng: torch.Generator | None = None,
    weight: float = 0.1,
    sample_count: int = 2,
    accumulate: bool = False,
    shared_noise: Tensor | None = None,
    latent_optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """Train generators toward direct agreement on aligned exact-K masks.

    Every generator sees its own full bank and the same noise draws. For each
    pair, detached hard supports choose the Hungarian column assignment. The
    objective and overlap are the means across all aligned pairs. The forward
    loss uses exact hard masks; a fixed-mass sigmoid supplies its surrogate
    gradient. No quality critic is called or updated by this objective.
    """
    models, banks, optimizers = _generator_inputs(models, banks, optimizers)
    objective_weight = _weight(weight, "weight")
    draws = _positive_int(sample_count, "sample_count")
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer")
    for model, optimizer in zip(models, optimizers):
        _validate_optimizer(model, optimizer)
    model_infos = [_model_info(model) for model in models]
    features_a, hidden_a, noise_dim_a, _ = model_infos[0]
    if any((features, hidden) != (features_a, hidden_a)
           for features, hidden, _, _ in model_infos[1:]):
        raise ValueError("all generators must have identical output mask dimensions")
    edge_count = features_a * hidden_a
    if not 1 <= k <= edge_count:
        raise ValueError("k must be within the output mask size")
    if any(noise_dim != noise_dim_a for _, _, noise_dim, _ in model_infos[1:]):
        raise ValueError("all generators must use the same noise dimension")

    bank_inputs = [_bank_tensors(bank, model) for model, bank in zip(models, banks)]
    devices_dtypes = [_device_dtype(model) for model in models]
    device_a, dtype_a = devices_dtypes[0]
    tokens_batch: list[Tensor] = []
    quality_batch: list[Tensor | None] = []
    for (tokens, quality, _), (device, dtype) in zip(bank_inputs, devices_dtypes):
        tokens_batch.append(tokens.expand(draws, *tokens.shape[1:]).to(device=device, dtype=dtype))
        quality_batch.append(None if quality is None else
                             quality.expand(draws, *quality.shape[1:]).to(device=device, dtype=dtype))
    latent_parameter = shared_noise
    if shared_noise is None:
        shared_noise = _noise(draws, noise_dim_a, rng, device_a, dtype_a)
        latent_parameter = None
    else:
        if (not isinstance(shared_noise, Tensor) or shared_noise.ndim not in (1, 2) or
                shared_noise.shape[-1] != noise_dim_a or not shared_noise.is_floating_point() or
                not bool(torch.isfinite(shared_noise).all())):
            raise ValueError("shared_noise must be a finite floating [noise_dim] or [draws, noise_dim] tensor")
        if shared_noise.ndim == 1:
            shared_noise = shared_noise.unsqueeze(0).expand(draws, -1)
        elif shared_noise.shape[0] == 1:
            shared_noise = shared_noise.expand(draws, -1)
        elif shared_noise.shape[0] != draws:
            raise ValueError("shared_noise must have one row or one row per agreement sample")
    loss_device = shared_noise.device
    logits = []
    for model, tokens, quality, (device, dtype) in zip(models, tokens_batch, quality_batch,
                                                       devices_dtypes):
        noise = shared_noise.to(device=device, dtype=dtype)
        density = torch.full((draws,), k / edge_count, device=device, dtype=dtype)
        logits.append(_forward(model, tokens, noise, quality, density))
    straight_and_hard = [_topk_straight_through(values, k) for values in logits]

    pair_agreements = []
    pair_overlaps = []
    for first in range(len(models)):
        straight_a, hard_a = straight_and_hard[first]
        for second in range(first + 1, len(models)):
            straight_b, hard_b = straight_and_hard[second]
            aligned_straight_b = []
            aligned_hard_b = []
            for row in range(draws):
                # Hungarian rows are the first model's columns; columns are
                # the second model's columns in this pair.
                columns_a = hard_a[row].detach().transpose(0, 1)
                columns_b = hard_b[row].detach().transpose(0, 1)
                cost = (columns_a[:, None, :] - columns_b.to(columns_a.device)[None, :, :]).square().sum(-1)
                permutation = torch.tensor(_hungarian(cost), device=hard_b.device, dtype=torch.long)
                aligned_straight_b.append(straight_b[row].index_select(1, permutation))
                aligned_hard_b.append(hard_b[row].index_select(1, permutation))
            aligned_straight = torch.stack(aligned_straight_b)
            aligned_hard = torch.stack(aligned_hard_b)
            aligned_straight = aligned_straight.to(straight_a.device)
            aligned_hard = aligned_hard.to(hard_a.device)
            pair_agreements.append((straight_a - aligned_straight).square().mean().to(loss_device))
            pair_overlaps.append((hard_a * aligned_hard).sum((1, 2)).div(k).mean().to(loss_device))
    agreement = torch.stack(pair_agreements).mean()
    overlap = float(torch.stack(pair_overlaps).mean().detach().cpu())

    if objective_weight > 0:
        if not accumulate:
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            if latent_optimizer is not None:
                latent_optimizer.zero_grad(set_to_none=True)
        (objective_weight * agreement).backward()
        if not accumulate:
            for model, optimizer in zip(models, optimizers):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                               max_norm=10.0)
                optimizer.step()
            if latent_optimizer is not None:
                if latent_parameter is None or latent_parameter.grad is None:
                    raise RuntimeError("shared noise received no agreement gradient")
                torch.nn.utils.clip_grad_norm_([latent_parameter], max_norm=10.0)
                latent_optimizer.step()
        loss_value = float(agreement.detach().cpu())
    else:
        # Weight zero is a true optimizer and parameter no-op; hard overlap is
        # still useful as a diagnostic.
        loss_value = 0.0
    return {
        "direct_agreement_loss": loss_value,
        "direct_agreement_overlap": overlap,
        "direct_agreement_weight": objective_weight,
    }
