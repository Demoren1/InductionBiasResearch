"""Single-objective updates for cooperative mask generators."""
from __future__ import annotations

from collections.abc import Mapping
import math

import torch
from torch import Tensor, nn
from scipy.optimize import linear_sum_assignment

from generator_evaluator.training.objectives import _topk_straight_through
from generator_evaluator.training.device_executor import PerDeviceGeneratorExecutor
from generator_evaluator.training.updates import _ensemble_predict_validated, _freeze, _restore


def _finite_nonnegative(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def _parameter_device_dtype(model: nn.Module) -> tuple[torch.device, torch.dtype]:
    parameter = next(model.parameters(), None)
    if parameter is None or not parameter.is_floating_point():
        raise ValueError("generator must have floating-point parameters")
    return parameter.device, parameter.dtype


def _column_costs(first: Tensor, second: Tensor) -> Tensor:
    """Return batched squared distances between columns without an F-sized broadcast."""
    # Inputs are [draws, hidden, features]. The result is [draws, H, H].
    first_norm = first.square().sum(-1, keepdim=True)
    second_norm = second.square().sum(-1).unsqueeze(1)
    cross = torch.matmul(first, second.transpose(-1, -2))
    return (first_norm + second_norm - 2.0 * cross).clamp_min_(0.0)


def _linear_assignments(costs: Tensor) -> Tensor:
    """Match batched square costs after one GPU-to-CPU transfer."""
    side = costs.shape[-1]
    leading_shape = costs.shape[:-2]
    cpu_costs = costs.detach().double().cpu().numpy().reshape(-1, side, side)
    assignments = []
    for cost in cpu_costs:
        rows, columns = linear_sum_assignment(cost)
        assignment = [0] * len(rows)
        for row, column in zip(rows, columns):
            assignment[int(row)] = int(column)
        assignments.append(assignment)
    return torch.tensor(assignments, dtype=torch.long, device=costs.device).reshape(
        *leading_shape, side
    )


def _elite_assignments(generated: Tensor, targets: Tensor) -> Tensor:
    """Match all generators/draws to all targets with one batched cost copy.

    ``generated`` has shape [N, draws, F, H], ``targets`` has shape [T, F, H].
    """
    generated_columns = generated.detach().transpose(2, 3)
    target_columns = targets.detach().transpose(1, 2)
    generated_norm = generated_columns.square().sum(-1)[:, :, None, :, None]
    target_norm = target_columns.square().sum(-1)[None, None, :, None, :]
    cross = torch.matmul(
        generated_columns[:, :, None],
        target_columns.transpose(-1, -2)[None, None, :],
    )
    costs = (generated_norm + target_norm - 2.0 * cross).clamp_min_(0.0)
    return _linear_assignments(costs)


def _targets_for_update(targets: Tensor | None, *, output_k: int, features: int,
                        hidden: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    if targets is None:
        return torch.empty((0, features, hidden), device=device, dtype=dtype)
    values = torch.as_tensor(targets, device=device, dtype=dtype)
    if values.ndim != 3 or values.shape[1:] != (features, hidden):
        raise ValueError("targets must have shape [N, features, hidden]")
    if not values.is_floating_point() or not bool(torch.isfinite(values).all()):
        raise ValueError("targets must be finite floating-point masks")
    if not bool(((values == 0) | (values == 1)).all()):
        raise ValueError("targets must be binary masks")
    # The archive may contain the configured K while an explicitly requested
    # auxiliary output budget is active. Keep every target at this update's K.
    return values[values.sum((1, 2)) == output_k].detach()


def joint_generator_update(*, models: Mapping[str, nn.Module],
                            banks: Mapping[str, object],
                            optimizers: Mapping[str, torch.optim.Optimizer],
                            training_views: Mapping[str, object], ensemble: nn.Module,
                            contexts: Tensor, dense_quality: Tensor,
                            targets: Tensor, output_k: int,
                            own_rngs: Mapping[str, torch.Generator],
                            agreement_rng: torch.Generator | None,
                            update_ordinal: int, updates_per_epoch: int,
                            agreement_weight: float, agreement_ramp_epochs: int,
                            elite_weight: float, elite_limit: int,
                            reconstruction_weight: float,
                            reconstruction_batch_size: int,
                            permutation_weight: float,
                            quality_scope: str = "own",
                            executor: PerDeviceGeneratorExecutor | None = None
                            ) -> dict[str, dict[str, float | str]]:
    """Apply one shared quality/agreement/distillation objective and one step.

    ``training_views`` and the reconstruction/permutation/ramp arguments stay
    in the signature for callers from older checkpoints. Joint updates always
    use the full cached ``banks`` and the configured agreement weight directly.
    """
    del training_views, own_rngs, agreement_ramp_epochs, elite_limit
    del reconstruction_weight, reconstruction_batch_size, permutation_weight

    names = tuple(models)
    if (len(names) < 2 or set(names) != set(banks) or set(names) != set(optimizers)):
        raise ValueError("joint update models, banks, and optimizers must match and include two generators")
    if contexts.ndim != 2 or len(contexts) != len(names) or dense_quality.shape != (len(names),):
        raise ValueError("joint update contexts and dense qualities must match the generators")
    if not isinstance(output_k, int) or isinstance(output_k, bool) or output_k < 1:
        raise ValueError("output_k must be a positive integer")
    if update_ordinal < 0 or updates_per_epoch < 1:
        raise ValueError("invalid joint update schedule position")
    if quality_scope not in ("own", "all_train"):
        raise ValueError("quality_scope must be own or all_train")
    agreement_weight = _finite_nonnegative(agreement_weight, "agreement_weight")
    elite_weight = _finite_nonnegative(elite_weight, "elite_weight")

    model_info = {}
    devices = []
    banks_by_name = {}
    for name in names:
        model = models[name]
        device, dtype = _parameter_device_dtype(model)
        features, hidden, noise_dim = (int(model.features), int(model.hidden), int(model.noise_dim))
        if not 1 <= output_k <= features * hidden:
            raise ValueError("output_k must fit every generator's output mask")
        bank = banks[name]
        tokens = getattr(bank, "tokens", None)
        quality = getattr(bank, "quality", None)
        if not isinstance(tokens, Tensor) or tokens.ndim not in (3, 4):
            raise ValueError(f"full bank for generator {name!r} must expose tensor tokens")
        if tokens.ndim == 3:
            tokens = tokens.unsqueeze(0)
        if tokens.shape[0] != 1 or not tokens.is_floating_point():
            raise ValueError("joint updates require one floating-point full bank per generator")
        if quality is not None:
            if not isinstance(quality, Tensor) or quality.ndim not in (2, 3):
                raise ValueError("bank quality must be [R,Q] or [1,R,Q]")
            if quality.ndim == 2:
                quality = quality.unsqueeze(0)
            if quality.shape[:2] != tokens.shape[:2]:
                raise ValueError("bank quality and tokens must contain the same full bank")
        if hasattr(model, "set_budget"):
            model.set_budget(output_k)  # type: ignore[attr-defined]
        model_info[name] = (device, dtype, features, hidden, noise_dim)
        devices.append(device)
        banks_by_name[name] = (tokens, quality)

    reference_features, reference_hidden, reference_noise = model_info[names[0]][2:]
    if any(model_info[name][2:] != (reference_features, reference_hidden, reference_noise)
           for name in names[1:]):
        raise ValueError("joint generators must have matching output and noise dimensions")

    critic_parameter = next(ensemble.parameters(), None)
    if critic_parameter is None:
        raise ValueError("quality ensemble must have parameters")
    critic_device, critic_dtype = critic_parameter.device, critic_parameter.dtype

    # The checkpointed shared RNG is CPU-backed in cooperative search. Draw
    # once, then copy the same fresh noise rows to every generator device.
    rng_device = torch.device("cpu") if agreement_rng is None else torch.device(agreement_rng.device)
    shared_noise = torch.randn((2, reference_noise), device=rng_device,
                               dtype=torch.float32, generator=agreement_rng)

    def forward_one(name: str) -> Tensor:
        model = models[name]
        device, dtype, _, _, _ = model_info[name]
        tokens, quality = banks_by_name[name]
        tokens = tokens.to(device=device, dtype=dtype)
        quality = None if quality is None else quality.to(device=device, dtype=dtype)
        tokens = tokens.expand(2, *tokens.shape[1:])
        quality = None if quality is None else quality.expand(2, *quality.shape[1:])
        noise = shared_noise.to(device=device, dtype=dtype)
        logits = model(tokens, noise, quality)
        expected = (2, reference_features, reference_hidden)
        if not isinstance(logits, Tensor) or logits.shape != expected:
            raise ValueError("generator must return [draws, features, hidden] logits")
        if not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
            raise ValueError("generator logits must be finite floating-point values")
        return logits

    if executor is None:
        with PerDeviceGeneratorExecutor(names, devices) as local_executor:
            logits_by_name = local_executor.map(forward_one)
    else:
        logits_by_name = executor.map(forward_one)

    straight_by_name: dict[str, Tensor] = {}
    hard_by_name: dict[str, Tensor] = {}
    for name in names:
        straight, hard = _topk_straight_through(logits_by_name[name], output_k)
        # Device copies preserve autograd from the central objective back to
        # generators on each device. Hard masks remain detached for matching.
        straight_by_name[name] = straight.to(device=critic_device, dtype=critic_dtype)
        hard_by_name[name] = hard.detach().to(device=critic_device, dtype=critic_dtype)

    dense = dense_quality.to(device=critic_device, dtype=critic_dtype).reshape(len(names), 1)

    ensemble.zero_grad(set_to_none=True)
    flags, was_training = _freeze(ensemble)
    try:
        if quality_scope == "own":
            own_masks = torch.cat([straight_by_name[name] for name in names], dim=0)
            own_contexts = torch.cat([
                contexts[index:index + 1].to(device=critic_device, dtype=critic_dtype).expand(2, -1)
                for index in range(len(names))
            ], dim=0)
            predicted, _uncertainty = _ensemble_predict_validated(ensemble, own_masks, own_contexts)
            predicted = predicted.reshape(len(names), 2)
            quality_delta = predicted - dense
            per_generator_quality = quality_delta.mean(dim=1)
        else:
            masks_by_generator = torch.stack([straight_by_name[name] for name in names], dim=0)
            all_masks = masks_by_generator.unsqueeze(2).expand(
                len(names), 2, len(names), reference_features, reference_hidden
            ).reshape(len(names) * 2 * len(names), reference_features, reference_hidden)
            all_contexts = contexts.to(device=critic_device, dtype=critic_dtype).reshape(
                1, 1, len(names), -1
            ).expand(len(names), 2, len(names), contexts.shape[1]).reshape(
                len(names) * 2 * len(names), contexts.shape[1]
            )
            predicted, _uncertainty = _ensemble_predict_validated(ensemble, all_masks, all_contexts)
            predicted = predicted.reshape(len(names), 2, len(names))
            quality_delta = predicted - dense.reshape(1, 1, len(names))
            per_generator_quality = quality_delta.mean(dim=(1, 2))
        quality_loss = quality_delta.mean()

        pair_indices: list[tuple[int, int]] = []
        pair_costs: list[Tensor] = []
        for first in range(len(names)):
            for second in range(first + 1, len(names)):
                first_hard = hard_by_name[names[first]].transpose(1, 2)
                second_hard = hard_by_name[names[second]].transpose(1, 2)
                pair_indices.append((first, second))
                pair_costs.append(_column_costs(first_hard, second_hard))
        # Stack every unordered generator pair before the single CPU transfer
        # and SciPy matching pass.
        pair_assignments = _linear_assignments(torch.stack(pair_costs))
        pair_losses: list[Tensor] = []
        pair_overlaps: list[Tensor] = []
        for pair_index, (first, second) in enumerate(pair_indices):
            assignments = pair_assignments[pair_index]
            second_masks = straight_by_name[names[second]]
            aligned_second = torch.stack([
                second_masks[row].index_select(1, assignments[row])
                for row in range(len(assignments))
            ])
            first_masks = straight_by_name[names[first]]
            pair_losses.append((first_masks - aligned_second).square().mean())
            hard_second = hard_by_name[names[second]]
            aligned_hard_second = torch.stack([
                hard_second[row].index_select(1, assignments[row])
                for row in range(len(assignments))
            ])
            pair_overlaps.append(
                (hard_by_name[names[first]] * aligned_hard_second).sum((1, 2)).div(output_k).mean()
            )
        agreement_loss = torch.stack(pair_losses).mean()
        agreement_overlap = torch.stack(pair_overlaps).mean()

        target_masks = _targets_for_update(
            targets, output_k=output_k, features=reference_features,
            hidden=reference_hidden, device=critic_device, dtype=critic_dtype,
        )
        distillation_terms = []
        if len(target_masks):
            generated_hard = torch.stack([hard_by_name[name] for name in names], dim=0)
            elite_assignments = _elite_assignments(generated_hard, target_masks)
            for generator_index, name in enumerate(names):
                assignments = elite_assignments[generator_index]
                target_count = len(target_masks)
                target_values = target_masks.unsqueeze(0).expand(2, -1, -1, -1)
                gather_index = assignments[:, :, None, :].expand(
                    2, target_count, reference_features, reference_hidden
                )
                aligned_targets = torch.gather(target_values, dim=3, index=gather_index)
                generated = straight_by_name[name][:, None, :, :]
                distillation_terms.append((generated - aligned_targets).square().mean())
            distillation_loss = torch.stack(distillation_terms).mean()
        else:
            distillation_terms = [quality_loss.new_zeros(()) for _ in names]
            distillation_loss = quality_loss.new_zeros(())
        total_loss = (quality_loss + agreement_weight * agreement_loss +
                      elite_weight * distillation_loss)
        if not bool(torch.isfinite(total_loss)):
            raise FloatingPointError("non-finite joint generator objective")

        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
    finally:
        _restore(ensemble, flags, was_training)

    # One clipped optimizer step per generator follows the single combined
    # backward. The frozen critic remains gradient-free.
    for name in names:
        parameters = [parameter for parameter in models[name].parameters()
                      if parameter.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=10.0)
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError(f"non-finite gradient for generator {name!r}")

    def step_one(name: str) -> None:
        optimizers[name].step()

    if executor is None:
        for name in names:
            step_one(name)
    else:
        executor.map(step_one)

    total_value = float(total_loss.detach().cpu())
    quality_value = float(quality_loss.detach().cpu())
    agreement_value = float(agreement_loss.detach().cpu())
    distillation_value = float(distillation_loss.detach().cpu())
    target_count_value = float(len(target_masks))
    rows: dict[str, dict[str, float | str]] = {}
    for index, name in enumerate(names):
        own_quality_value = float(per_generator_quality[index].detach().cpu())
        own_distillation_value = float(distillation_terms[index].detach().cpu())
        rows[name] = {
            "total_loss": total_value,
            "quality_loss": quality_value,
            "agreement_loss": agreement_value,
            "distillation_loss": distillation_value,
            "own_quality_loss": own_quality_value,
            "own_distillation_loss": own_distillation_value,
            # Compatibility fields used by existing history and plotting code.
            "loss": total_value,
            "predicted_cost": own_quality_value,
            "predicted_objective": own_quality_value,
            "direct_agreement_loss": agreement_value,
            "direct_agreement_overlap": float(agreement_overlap.detach().cpu()),
            "direct_agreement_weight": agreement_weight,
            "elite_distillation_loss": distillation_value,
            "elite_count": target_count_value,
            "distillation_target_count": target_count_value,
            "sample_count": 2.0,
            "output_k": float(output_k),
            "ordinal": float(update_ordinal),
            "quality_scope": quality_scope,
        }
    return rows
