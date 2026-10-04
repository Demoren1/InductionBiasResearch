"""Training and acquisition utilities for the measured-quality search loop.

The functions here deliberately keep the three sources of information apart:
the evaluator is fitted only on real ``train`` replay rows, its validation
rows are reporting-only, and a generator update treats evaluator predictions
as detached policy costs for exact ordered top-k sampling.
"""
from __future__ import annotations

import math
from copy import deepcopy
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from deepsets_vaae.permutation_utility_loss import quality_policy_loss, sample_ordered_topk

from generator_evaluator.data.types import RealReplay, topology_id
from generator_evaluator.models.transformer import QualityEnsemble, TransformerMaskGenerator, permute_bank
from generator_evaluator.storage.progress import progress
from generator_evaluator.search.quality import quality_objective_cost, validate_quality_objective


def _finite_float(value: Tensor | float) -> float:
    result = float(torch.as_tensor(value).detach().cpu())
    if not torch.isfinite(torch.tensor(result)):
        raise ValueError("computed metric must be finite")
    return result


def _random_device(rng: torch.Generator | None, fallback: torch.device) -> torch.device:
    return fallback if rng is None else torch.device(rng.device)


def _random_gumbels(shape: torch.Size | tuple[int, ...], rng: torch.Generator | None,
                    device: torch.device, dtype: torch.dtype) -> Tensor:
    """Draw Gumbels on the checkpointed RNG's device, then place them with logits."""
    source_device = _random_device(rng, device)
    uniform = torch.rand(shape, device=source_device, dtype=torch.float32, generator=rng)
    eps = torch.finfo(torch.float32).eps
    uniform = uniform.clamp(eps, 1.0 - eps)
    return (-torch.log(-torch.log(uniform))).to(device=device, dtype=dtype)


def _random_noise(shape: torch.Size | tuple[int, ...], rng: torch.Generator | None,
                  device: torch.device, dtype: torch.dtype) -> Tensor:
    source_device = _random_device(rng, device)
    return torch.randn(shape, device=source_device, dtype=torch.float32,
                       generator=rng).to(device=device, dtype=dtype)


def _require_float(value: Tensor, name: str, ndim: int) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != ndim or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating [{ndim}]-D tensor")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must be finite")


def _rankdata(values: Tensor) -> Tensor:
    """Average ranks, including tied values, without requiring scipy."""
    values = values.detach().flatten().double().cpu()
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty_like(values)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop].item() == sorted_values[start].item():
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def evaluator_metrics(
    predictions: Tensor,
    targets: Tensor,
    *,
    top_k: int = 1,
    task_ids: Sequence[str] | None = None,
) -> dict[str, float]:
    """Regression and ranking diagnostics, safe when either vector is tied."""
    _require_float(predictions, "predictions", 1)
    _require_float(targets, "targets", 1)
    if predictions.shape != targets.shape or len(predictions) == 0:
        raise ValueError("predictions and targets must be equal nonempty vectors")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    errors = predictions - targets
    pred_rank, target_rank = _rankdata(predictions), _rankdata(targets)
    pred_centered, target_centered = pred_rank - pred_rank.mean(), target_rank - target_rank.mean()
    denom = pred_centered.norm() * target_centered.norm()
    spearman = torch.tensor(0.0) if denom == 0 else (pred_centered * target_centered).sum() / denom
    count = min(top_k, len(predictions))
    # Lower quality is better.  Report the true cost of the predicted best set
    # relative to the true best set; this makes a tied rank useful to inspect.
    chosen = torch.argsort(predictions)[:count]
    actual = torch.argsort(targets)[:count]
    result = {
        "mse": _finite_float(errors.square().mean()),
        "mae": _finite_float(errors.abs().mean()),
        "spearman": _finite_float(spearman),
        "top_candidate_error": _finite_float(targets[chosen].mean() - targets[actual].mean()),
        "top_candidate_mae": _finite_float(errors[chosen].abs().mean()),
    }
    if task_ids is not None:
        if len(task_ids) != len(predictions):
            raise ValueError("task_ids must have one item for each prediction")
        per_task = []
        for task_id in dict.fromkeys(task_ids):
            rows = torch.tensor([index for index, item in enumerate(task_ids) if item == task_id])
            if len(rows) >= 2:
                per_task.append(evaluator_metrics(predictions[rows], targets[rows], top_k=top_k))
        # A single topology per task cannot define a within-task ranking.  A
        # neutral zero keeps every partition metric schema stable for early
        # replay collections.
        result["within_task_spearman"] = (
            sum(item["spearman"] for item in per_task) / len(per_task) if per_task else 0.0
        )
        result["within_task_top_candidate_error"] = (
            sum(item["top_candidate_error"] for item in per_task) / len(per_task) if per_task else 0.0
        )
    return result


def _ensemble_members(ensemble: QualityEnsemble) -> list[nn.Module]:
    members = getattr(ensemble, "evaluators", None)
    if members is None or len(members) == 0:
        raise TypeError("ensemble must expose a nonempty .evaluators collection")
    return list(members)


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Make a CPU-restored optimizer usable with members moved to ``device``."""
    for state in optimizer.state.values():
        for name, value in state.items():
            if isinstance(value, Tensor):
                state[name] = value.to(device)


def _partition_metrics(ensemble: QualityEnsemble, replay: RealReplay, split: str, device: torch.device) -> dict[str, float]:
    try:
        masks, contexts, targets = replay.tensors(split)
    except ValueError:
        return {}
    with torch.no_grad():
        mean, std = ensemble.predict(masks.to(device), contexts.to(device))
    records = getattr(replay, "records", None)
    task_ids = None if records is None else [row["task_id"] for row in records if row["split"] == split]
    if task_ids is not None and len(task_ids) != len(targets):
        raise ValueError("replay records and partition tensors disagree")
    out = evaluator_metrics(mean.cpu(), targets.float().cpu(), task_ids=task_ids)
    out["mean_std"] = _finite_float(std.mean())
    return out


def _evaluator_selection_data(replay: RealReplay, active_edges: int | None):
    rows = [row for row in replay.records if row["split"] == "mask_validation"]
    budget_rows = [row for row in rows if row["active_edges"] == active_edges]
    if budget_rows:
        rows = budget_rows
    if not rows:
        return None
    identities = list(dict.fromkeys(row["topology_id"] for row in rows))
    indices = {identity: index for index, identity in enumerate(identities)}
    return (torch.stack([replay.masks[row["mask_key"]] for row in rows]),
            torch.stack([replay.contexts[row["task_id"]] for row in rows]),
            torch.tensor([row["quality"] for row in rows], dtype=torch.float32),
            torch.tensor([indices[row["topology_id"]] for row in rows]),
            len(identities), bool(budget_rows))


def _evaluator_selection_metrics(ensemble, selection, device):
    masks, contexts, targets, groups, count, target_budget = selection
    with torch.no_grad():
        prediction = torch.cat([
            ensemble.predict(masks[start:start + 128].to(device),
                             contexts[start:start + 128].to(device))[0].cpu()
            for start in range(0, len(masks), 128)])
    group_counts = torch.bincount(groups, minlength=count).float()
    mean_prediction = torch.zeros(count).scatter_add_(0, groups, prediction) / group_counts
    mean_target = torch.zeros(count).scatter_add_(0, groups, targets) / group_counts
    metrics = evaluator_metrics(mean_prediction, mean_target)
    return dict(metrics, masks=count, target_budget=target_budget)


def train_evaluators(
    ensemble: QualityEnsemble,
    replay: RealReplay,
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 0,
    device: str | torch.device = "cpu",
    *,
    restore_best: bool = False,
    selection_active_edges: int | None = None,
) -> list[dict[str, Any]]:
    """Fit independent bootstrap evaluator members from replay's train rows.

    Validation partitions are evaluated after an epoch and can select the
    final checkpoint when requested.  In particular,
    ``mask_validation`` and ``meta_validation`` can never be sampled by an
    optimizer step.
    """
    if epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("epochs, batch_size and lr must be positive")
    device = torch.device(device)
    members = _ensemble_members(ensemble)
    masks, contexts, targets = replay.tensors("train")
    _require_float(masks, "train masks", 3)
    _require_float(contexts, "train contexts", 2)
    _require_float(targets, "train quality", 1)
    if not (len(masks) == len(contexts) == len(targets)):
        raise ValueError("train replay tensors must have equal row counts")
    ensemble.to(device)
    optimizers = [torch.optim.Adam(member.parameters(), lr=lr) for member in members]
    # The checkpointable state deliberately holds generators rather than an
    # old index array: after new real measurements arrive, each independently
    # seeded bootstrap is redrawn from both old and new train rows.
    saved_state = getattr(ensemble, "training_state", None)
    if saved_state is not None and not isinstance(saved_state, dict):
        raise TypeError("ensemble.training_state must be a plain dictionary")
    if saved_state and saved_state.get("member_count") == len(members):
        for optimizer, state in zip(optimizers, saved_state.get("optimizer_states", [])):
            optimizer.load_state_dict(state)
            _optimizer_to_device(optimizer, device)
    bootstrap_rngs: list[torch.Generator] = []
    shufflers: list[torch.Generator] = []
    for index in range(len(members)):
        sample_rng = torch.Generator(device="cpu").manual_seed(seed + 10_003 * (index + 1))
        shuffle_rng = torch.Generator(device="cpu").manual_seed(seed + 20_011 * (index + 1))
        if saved_state and saved_state.get("member_count") == len(members):
            bootstrap_states = saved_state.get("bootstrap_rng_states", [])
            shuffle_states = saved_state.get("shuffle_rng_states", [])
            if index < len(bootstrap_states):
                sample_rng.set_state(bootstrap_states[index].cpu())
            if index < len(shuffle_states):
                shuffle_rng.set_state(shuffle_states[index].cpu())
        bootstrap_rngs.append(sample_rng)
        shufflers.append(shuffle_rng)

    selection = _evaluator_selection_data(replay, selection_active_edges) if restore_best else None
    best_score, best_epoch, best_model, best_training_state = math.inf, 0, None, None

    def training_state():
        return {
            "member_count": len(members),
            "optimizer_states": [optimizer.state_dict() for optimizer in optimizers],
            "bootstrap_rng_states": [rng.get_state().cpu() for rng in bootstrap_rngs],
            "shuffle_rng_states": [rng.get_state().cpu() for rng in shufflers],
        }

    history: list[dict[str, Any]] = []
    epochs_bar = progress(range(epochs), desc="Quality evaluator", unit="epoch")
    for epoch in epochs_bar:
        train_losses: list[Tensor] = []
        for member, optimizer, sample_rng, shuffle_rng in zip(members, optimizers, bootstrap_rngs, shufflers):
            member.train()
            rows = torch.randint(len(masks), (len(masks),), generator=sample_rng)
            order = torch.randperm(len(rows), generator=shuffle_rng)
            for start in range(0, len(rows), batch_size):
                batch_rows = rows[order[start:start + batch_size]]
                prediction = member(masks[batch_rows].to(device), contexts[batch_rows].to(device))
                loss = (prediction - targets[batch_rows].to(device)).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite evaluator training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(member.parameters(), max_norm=10.0)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("non-finite evaluator gradient")
                optimizer.step()
                train_losses.append(loss.detach())
        ensemble.eval()
        with torch.no_grad():
            train_prediction, train_std = ensemble.predict(masks.to(device), contexts.to(device))
        records = getattr(replay, "records", None)
        train_task_ids = None if records is None else [row["task_id"] for row in records if row["split"] == "train"]
        if train_task_ids is not None and len(train_task_ids) != len(targets):
            raise ValueError("replay records and train tensors disagree")
        train_metrics = evaluator_metrics(train_prediction.cpu(), targets.float().cpu(), task_ids=train_task_ids)
        event: dict[str, Any] = {
            "epoch": epoch + 1,
            "train_mse": train_metrics["mse"],
            "train_bootstrap_mse": _finite_float(torch.stack(train_losses).mean()),
            "train_spearman": train_metrics["spearman"],
            "train_mean_std": _finite_float(train_std.mean()),
        }
        for split in ("mask_validation", "meta_validation", "joint_validation"):
            metrics = _partition_metrics(ensemble, replay, split, device)
            for key, value in metrics.items():
                event[f"{split}_{key}"] = value
        if selection is not None:
            selected = _evaluator_selection_metrics(ensemble, selection, device)
            event.update({f"selection_{key}": value for key, value in selected.items()})
            if selected["mse"] < best_score:
                best_score, best_epoch = selected["mse"], epoch + 1
                best_model = {key: value.detach().cpu().clone()
                              for key, value in ensemble.state_dict().items()}
                best_training_state = deepcopy(training_state())
        history.append(event)
        epochs_bar.set_postfix(train_mse=f"{event['train_mse']:.5f}", refresh=False)
    ensemble.training_state = training_state()
    if best_model is not None:
        ensemble.load_state_dict(best_model)
        ensemble.training_state = best_training_state
        ensemble.training_state["validation_selection"] = {
            "epoch": best_epoch, "epochs_evaluated": epochs,
            "metric": "mean_task_mask_validation_mse", "mse": best_score,
            "active_edges": selection_active_edges if selection[-1] else None,
        }
        history[-1].update(selected_epoch=best_epoch, selection_restored=True)
    return history


def _bank_inputs(
    generator: TransformerMaskGenerator,
    tokens: Tensor,
    quality: Tensor | None,
) -> tuple[Tensor, Tensor | None]:
    if tokens.ndim == 3:
        tokens = tokens.unsqueeze(0)
    _require_float(tokens, "bank_tokens", 4)
    if quality is not None and quality.ndim == 2:
        quality = quality.unsqueeze(0)
    if quality is not None:
        _require_float(quality, "bank_quality", 3)
    generator._validate_bank(tokens, quality)  # The model owns exact feature dimensions.
    return tokens, quality


def _generator_forward_validated(generator: TransformerMaskGenerator, tokens: Tensor,
                                 noise: Tensor, quality: Tensor | None) -> Tensor:
    """Use the model's no-validation path only after ``_bank_inputs`` validated it."""
    fast = getattr(generator, "_forward_validated", None)
    if callable(fast):
        return fast(tokens, noise, quality, tokens.shape[0])
    # Density-conditioned and test-double generators retain their public path.
    return generator(tokens, noise, quality)


def _ensemble_predict_validated(ensemble: QualityEnsemble, masks: Tensor,
                                contexts: Tensor) -> tuple[Tensor, Tensor]:
    """Keep overridden ``predict`` methods observable in tests and extensions."""
    fast = getattr(ensemble, "_predict_validated", None)
    if callable(fast) and "predict" not in getattr(ensemble, "__dict__", {}):
        return fast(masks, contexts)
    return ensemble.predict(masks, contexts)


def _freeze(module: nn.Module) -> tuple[list[bool], bool]:
    flags = [parameter.requires_grad for parameter in module.parameters()]
    was_training = module.training
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return flags, was_training


def _restore(module: nn.Module, flags: list[bool], was_training: bool) -> None:
    for parameter, flag in zip(module.parameters(), flags):
        parameter.requires_grad_(flag)
    module.train(was_training)


def _generator_samples(
    generator: TransformerMaskGenerator,
    bank_tokens: Tensor,
    bank_quality: Tensor | None,
    k: int,
    rng: torch.Generator | None,
    draws: int,
    shared_noise: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if draws < 2:
        raise ValueError("at least two independent policy draws are required")
    tokens, quality = _bank_inputs(generator, bank_tokens, bank_quality)
    if tokens.shape[0] != 1:
        raise ValueError("a shared-mask update requires one bank (batch size 1)")
    device, dtype = tokens.device, tokens.dtype
    expanded_tokens = tokens.expand(draws, *tokens.shape[1:])
    expanded_quality = None if quality is None else quality.expand(draws, *quality.shape[1:])
    if shared_noise is None:
        noise = _random_noise((draws, generator.noise_dim), rng, device, dtype)
    else:
        if (not isinstance(shared_noise, Tensor) or shared_noise.ndim not in (1, 2) or
                shared_noise.shape[-1] != generator.noise_dim or not shared_noise.is_floating_point() or
                not torch.isfinite(shared_noise).all().item()):
            raise ValueError("shared_noise must be a finite floating [noise_dim] or [draws, noise_dim] tensor")
        if shared_noise.ndim == 1:
            noise = shared_noise.to(device=device, dtype=dtype).unsqueeze(0).expand(draws, -1)
        elif shared_noise.shape[0] == 1:
            noise = shared_noise.to(device=device, dtype=dtype).expand(draws, -1)
        elif shared_noise.shape == (draws, generator.noise_dim):
            noise = shared_noise.to(device=device, dtype=dtype)
        else:
            raise ValueError("shared_noise must have one row or one row per policy draw")
    logits = _generator_forward_validated(generator, expanded_tokens, noise, expanded_quality)
    # Keep the caller's RNG checkpointable even when the generator and RNG
    # live on different devices.  The Gumbels remain independent per draw.
    gumbel = _random_gumbels(logits.shape, rng, logits.device, logits.dtype)
    masks, log_prob, _ = sample_ordered_topk(logits, k, gumbel=gumbel)
    return masks, log_prob, logits, noise


def _mask_diversity(masks: Tensor) -> Tensor:
    if len(masks) < 2:
        return masks.new_zeros(())
    flat = masks.flatten(1)
    return (flat[:, None] != flat[None, :]).float().mean()


def _permutation_loss(
    generator: TransformerMaskGenerator,
    tokens: Tensor,
    quality: Tensor | None,
    noise: Tensor,
    rng: torch.Generator | None,
) -> Tensor:
    # `permute_bank` intentionally follows tokens' device.  Draw indices on
    # the RNG's own device first so CPU checkpointed RNGs also work with CUDA
    # generators, then move only the discrete indices.
    rng_device = tokens.device if rng is None else torch.device(rng.device)
    solution_order = torch.randperm(tokens.shape[1], generator=rng, device=rng_device).to(tokens.device)
    neuron_order = torch.randperm(tokens.shape[2], generator=rng, device=rng_device).to(tokens.device)
    permuted_tokens = tokens.index_select(1, solution_order).index_select(2, neuron_order)
    permuted_quality = None if quality is None else quality.index_select(1, solution_order)
    draws = len(noise)
    original = _generator_forward_validated(
        generator, tokens.expand(draws, *tokens.shape[1:]), noise,
        None if quality is None else quality.expand(draws, *quality.shape[1:]))
    permuted = _generator_forward_validated(
        generator, permuted_tokens.expand(draws, *permuted_tokens.shape[1:]), noise,
        None if permuted_quality is None else permuted_quality.expand(draws, *permuted_quality.shape[1:]))
    return (original - permuted).square().mean()


def generator_update(
    generator: TransformerMaskGenerator,
    ensemble: QualityEnsemble,
    bank_tokens: Tensor,
    bank_quality: Tensor | None,
    contexts: Tensor,
    dense_quality: Tensor,
    optimizer: torch.optim.Optimizer,
    k: int,
    rng: torch.Generator | None = None,
    *,
    permutation_weight: float = 1.0,
    uncertainty_weight: float = 0.0,
    sample_count: int = 2,
    accumulate: bool = False,
    shared_noise: Tensor | None = None,
    loss_scale: float = 1.0,
    quality_objective: str = "worst",
) -> dict[str, float]:
    """One score-function update using a configured predicted task cost.

    The hard masks and evaluator costs are detached.  Gradients therefore use
    the exact ordered top-k log probability and cannot alter evaluator weights.
    """
    if permutation_weight < 0 or uncertainty_weight < 0:
        raise ValueError("loss weights must be nonnegative")
    validate_quality_objective(quality_objective)
    if (isinstance(loss_scale, bool) or not isinstance(loss_scale, (int, float)) or
            not math.isfinite(float(loss_scale)) or loss_scale < 0):
        raise ValueError("loss_scale must be a finite nonnegative number")
    _require_float(contexts, "contexts", 2)
    _require_float(dense_quality, "dense_quality", 1)
    if len(contexts) != len(dense_quality):
        raise ValueError("contexts and dense_quality must have the same task count")
    tokens, quality = _bank_inputs(generator, bank_tokens, bank_quality)
    masks, log_prob, _, noise = _generator_samples(generator, tokens, quality, k, rng, sample_count,
                                                    shared_noise=shared_noise)
    context_batch = contexts.to(device=masks.device, dtype=masks.dtype).repeat(sample_count, 1)
    mask_batch = masks[:, None].expand(-1, len(contexts), -1, -1).reshape(-1, *masks.shape[1:])
    # A global quality critic may live on the coordinator while generators
    # are spread over devices.  Evaluator parameters never receive gradients;
    # move the hard masks and contexts to its device for scoring, then return
    # detached costs to the policy device.
    critic_device = next(ensemble.parameters(), masks).device
    critic_masks = mask_batch.to(critic_device)
    critic_contexts = context_batch.to(critic_device)
    flags, was_training = _freeze(ensemble)
    try:
        with torch.no_grad():
            predicted, uncertainty = _ensemble_predict_validated(ensemble, critic_masks, critic_contexts)
            predicted = predicted.to(log_prob.device)
            uncertainty = uncertainty.to(log_prob.device)
            delta = predicted.reshape(sample_count, len(contexts)) - dense_quality.to(predicted).unsqueeze(0)
            risk = delta + uncertainty_weight * uncertainty.reshape(sample_count, len(contexts))
            objective_cost = quality_objective_cost(risk, quality_objective, task_dim=1).detach()
    finally:
        _restore(ensemble, flags, was_training)
    policy_loss, advantage = quality_policy_loss(log_prob.unsqueeze(0), objective_cost.unsqueeze(0))
    perm_loss = _permutation_loss(generator, tokens, quality, noise, rng)
    loss = policy_loss + permutation_weight * perm_loss
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite generator surrogate loss")
    if not accumulate:
        optimizer.zero_grad(set_to_none=True)
    (float(loss_scale) * loss).backward()
    if not accumulate:
        grad_norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=10.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("non-finite generator gradient")
        optimizer.step()
    return {
        "loss": _finite_float(loss),
        "gradient_surrogate": _finite_float(policy_loss),
        "policy_gradient_loss": _finite_float(policy_loss),
        "predicted_objective": _finite_float(objective_cost.mean()),
        "predicted_cost": _finite_float(objective_cost.mean()),
        "permutation_loss": _finite_float(perm_loss),
        "diversity": _finite_float(_mask_diversity(masks)),
        "advantage_abs_mean": _finite_float(advantage.abs().mean()),
        "sample_count": float(sample_count),
        "loss_scale": float(loss_scale),
    }


def direct_generator_update(
    generator: TransformerMaskGenerator,
    bank_tokens: Tensor,
    bank_quality: Tensor | None,
    optimizer: torch.optim.Optimizer,
    k: int,
    rng: torch.Generator | None,
    measure_cost: Callable[[Tensor], Tensor],
    *,
    permutation_weight: float = 1.0,
    sample_count: int = 2,
) -> dict[str, float]:
    """Score-function update from actual external measurements.

    ``measure_cost`` is called exactly once with ``[sample_count,F,H]`` hard masks;
    callers own the corresponding fresh-fit measurement budget and provenance.
    """
    if permutation_weight < 0:
        raise ValueError("permutation_weight must be nonnegative")
    tokens, quality = _bank_inputs(generator, bank_tokens, bank_quality)
    masks, log_prob, _, noise = _generator_samples(generator, tokens, quality, k, rng, sample_count)
    measured = measure_cost(masks.detach())
    _require_float(measured, "measure_cost result", 1)
    if measured.shape != (sample_count,):
        raise ValueError(f"measure_cost must return [{sample_count}] costs")
    if measured.device != log_prob.device:
        measured = measured.to(log_prob.device)
    policy_loss, advantage = quality_policy_loss(log_prob.unsqueeze(0), measured.unsqueeze(0))
    perm_loss = _permutation_loss(generator, tokens, quality, noise, rng)
    loss = policy_loss + permutation_weight * perm_loss
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite direct generator loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=10.0)
    if not torch.isfinite(grad_norm):
        raise FloatingPointError("non-finite generator gradient")
    optimizer.step()
    return {
        "loss": _finite_float(loss),
        "gradient_surrogate": _finite_float(policy_loss),
        "policy_gradient_loss": _finite_float(policy_loss),
        "measured_objective": _finite_float(measured.mean()),
        "predicted_cost": _finite_float(measured.mean()),
        "permutation_loss": _finite_float(perm_loss),
        "diversity": _finite_float(_mask_diversity(masks)),
        "advantage_abs_mean": _finite_float(advantage.abs().mean()),
        "measurements": float(sample_count),
        "sample_count": float(sample_count),
    }


def _proposal_logits_from_shared_bank(
    generator: nn.Module,
    tokens: Tensor,
    quality: Tensor | None,
    noise: Tensor,
) -> Tensor:
    """Decode proposal logits from one bank encoding and many noise draws.

    The production Transformer encodes each bank independently of the noise.
    Reusing its compact solution and neuron memories avoids expanding the raw
    functional profiles across proposal draws.  Other generators use a
    count-one forward fallback so they also never receive an expanded bank.
    """
    base = generator
    encode_tokens = tokens
    if isinstance(generator, TransformerMaskGenerator):
        # A subclass with a custom forward path may define different logits;
        # leave that behavior to its public forward method below.
        if (type(generator).forward is not TransformerMaskGenerator.forward or
                type(generator)._forward_validated is not TransformerMaskGenerator._forward_validated):
            base = None
    else:
        inner = getattr(generator, "inner", None)
        augment = getattr(generator, "_augment_validated", None)
        if isinstance(inner, TransformerMaskGenerator) and callable(augment):
            base = inner
            # DensityConditionedGenerator adds the requested target density
            # before encoding.  Its current budget is set from proposal k.
            encode_tokens = augment(tokens, None, tokens.shape[0])
        else:
            base = None

    encode_components = getattr(base, "_encode_bank_components_validated", None)
    if callable(encode_components):
        batch, solutions = encode_tokens.shape[:2]
        count = len(noise)
        solution_memory, neuron_memory = encode_components(
            encode_tokens, quality, batch, solutions
        )
        noise_memory = base.noise_projection(noise).unsqueeze(1)
        solution_memory = solution_memory + noise_memory
        neuron_memory = neuron_memory + solution_memory.unsqueeze(2)
        neuron_memory = neuron_memory.reshape(count, -1, base.width)
        # The encoded bank has batch size one.  Adding the count-shaped noise
        # expands only these width-sized memories across the proposal draws.
        memory = torch.cat((solution_memory, neuron_memory), dim=1)
        queries = base.output_queries.unsqueeze(0).expand(count, -1, -1)
        decoded = base.decoder(tgt=queries, memory=memory)
        hidden_terms = decoded.unsqueeze(1).expand(-1, base.features, -1, -1)
        feature_terms = base.feature_embeddings.unsqueeze(0).unsqueeze(2).expand(
            count, -1, base.hidden, -1
        )
        return base.output_head(torch.cat((hidden_terms, feature_terms), dim=-1)).squeeze(-1)

    return torch.cat([
        generator(tokens, noise[index:index + 1], quality)
        for index in range(len(noise))
    ], dim=0)


def propose_candidates(
    model: TransformerMaskGenerator,
    tokens: Tensor,
    quality: Tensor | None,
    k: int,
    count: int,
    rng: torch.Generator | None = None,
) -> Tensor:
    """Generate ``count`` independent exact-K hard masks for a single bank."""
    if count < 1:
        raise ValueError("count must be positive")
    tokens, quality = _bank_inputs(model, tokens, quality)
    if tokens.shape[0] != 1:
        raise ValueError("proposals require one bank (batch size 1)")
    if hasattr(model, "set_budget"):
        model.set_budget(k)  # type: ignore[attr-defined]
    with torch.no_grad():
        noise = _random_noise((count, model.noise_dim), rng, tokens.device, tokens.dtype)
        logits = _proposal_logits_from_shared_bank(model, tokens, quality, noise)
        gumbel = _random_gumbels(logits.shape, rng, logits.device, logits.dtype)
        masks, _, _ = sample_ordered_topk(logits, k, gumbel=gumbel)
    return masks


def select_acquisition(
    masks: Tensor,
    ensemble: QualityEnsemble,
    contexts: Tensor,
    dense_quality: Tensor,
    budget: int,
    rng: torch.Generator | None = None,
) -> tuple[Tensor, list[str]]:
    """Select distinct topologies from promising, uncertain and random pools."""
    _require_float(masks, "masks", 3)
    _require_float(contexts, "contexts", 2)
    _require_float(dense_quality, "dense_quality", 1)
    if budget < 1 or len(masks) == 0 or len(contexts) != len(dense_quality):
        raise ValueError("invalid acquisition budget or task inputs")
    context_batch = contexts.to(masks).repeat(len(masks), 1)
    mask_batch = masks[:, None].expand(-1, len(contexts), -1, -1).reshape(-1, *masks.shape[1:])
    with torch.no_grad():
        mean, std = ensemble.predict(mask_batch, context_batch)
    delta = mean.reshape(len(masks), len(contexts)) - dense_quality.to(mean).unsqueeze(0)
    uncertainty = std.reshape(len(masks), len(contexts)).max(dim=1).values
    promising = torch.argsort(delta.max(dim=1).values).tolist()
    uncertain = torch.argsort(uncertainty, descending=True).tolist()
    random_order = torch.randperm(len(masks), generator=rng, device=masks.device).tolist()
    pools = (("promising", promising), ("uncertain", uncertain), ("random", random_order))
    selected: list[int] = []
    origins: list[str] = []
    seen: set[str] = set()
    # Cycle sources, preserving the requested mixture whenever unique masks
    # exist.  Duplicate column permutations are never selected twice.
    positions = [0, 0, 0]
    while len(selected) < min(budget, len(masks)):
        made_progress = False
        for pool_index, (origin, order) in enumerate(pools):
            while positions[pool_index] < len(order):
                candidate = order[positions[pool_index]]
                positions[pool_index] += 1
                identity = topology_id(masks[candidate])
                if identity not in seen:
                    seen.add(identity)
                    selected.append(candidate)
                    origins.append(origin)
                    made_progress = True
                    break
            if len(selected) == min(budget, len(masks)):
                break
        if not made_progress:
            break
    return masks[torch.tensor(selected, device=masks.device)], origins
