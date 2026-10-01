"""Differentiable support-to-query meta-training and final child fitting."""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
import torch.nn.functional as F

from .core import (
    ACTIVE_EDGES,
    HIDDEN,
    SEQ_LEN,
    binary_metrics,
    build_experiment_data,
    sample_balanced,
)
from .convergence import loss_plateau
from .generator import FEATURE_DIM, Generator, generate, permute_hidden_columns


@dataclass(frozen=True)
class MetaConfig:
    method: str = "transformer_mask"
    meta_batch_tasks: int = 4
    replicas: int = 2
    validation_replicas: int = 4
    support_size: int = 128
    query_size: int = 128
    inner_steps: int = 64
    inner_learning_rate: float = 0.1
    inner_momentum: float = 0.9
    outer_learning_rate: float = 0.002
    lr_decay_every: int = 500
    lr_decay_factor: float = 0.5
    outer_minimum_lr: float = 0.0001
    min_steps: int = 400
    max_steps: int = 2000
    resume_max_steps: int = 12000
    eval_every: int = 20
    relative_improvement: float = 0.01
    patience_steps: int = 400
    plateau_passes: int = 3
    hidden_column_augmentation: bool = True
    map_order_augmentation: bool = True

    def validate(self) -> None:
        if self.method not in ("transformer_mask", "free_mask"):
            raise ValueError("method must be 'transformer_mask' or 'free_mask'")
        if min(self.meta_batch_tasks, self.replicas, self.validation_replicas,
               self.support_size, self.query_size, self.inner_steps) < 1:
            raise ValueError("meta batch, replica, sample, and horizon settings must be positive")
        if not self.min_steps <= self.max_steps <= self.resume_max_steps:
            raise ValueError("outer step caps must satisfy min <= max <= resume_max")
        if self.eval_every < 1:
            raise ValueError("eval_every must be positive")


def init_child(seed: int = 0, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    """Initialize one fresh ReLU child with the experiment's paired seed rule."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    w = torch.randn((SEQ_LEN, HIDDEN), generator=generator) * 0.1
    a = torch.randn((HIDDEN,), generator=generator) * 0.1
    return {
        "w": w.to(device).requires_grad_(),
        "b": torch.zeros(HIDDEN, device=device, requires_grad=True),
        "a": a.to(device).requires_grad_(),
        "c": torch.zeros((), device=device, requires_grad=True),
    }


def child_logits(x: torch.Tensor, mask: torch.Tensor,
                 params: dict[str, torch.Tensor]) -> torch.Tensor:
    """Evaluate the actual masked ReLU child: ``a*ReLU(x(W*M)+b)+c``."""
    if x.ndim != 2 or x.size(1) != SEQ_LEN:
        raise ValueError(f"x must have shape [batch, {SEQ_LEN}]")
    if mask.shape != (SEQ_LEN, HIDDEN):
        raise ValueError(f"mask must have shape [{SEQ_LEN}, {HIDDEN}]")
    preactivation = x @ (params["w"] * mask) + params["b"]
    return F.relu(preactivation) @ params["a"] + params["c"]


def child_logits_batch(
    x: torch.Tensor,
    masks: torch.Tensor,
    params: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Run C fixed-mask children together on shared or per-run inputs.

    Args:
        x: ``[N, 11]`` shared across runs or ``[C, N, 11]`` per run.
        masks: ``[C, 11, 8]``.
        params: ``w:[C,11,8], b/a:[C,8], c:[C]``.
    Returns:
        Logits with shape ``[C, N]``.
    """
    if masks.ndim != 3 or masks.shape[1:] != (SEQ_LEN, HIDDEN):
        raise ValueError(f"masks must have shape [runs, {SEQ_LEN}, {HIDDEN}]")
    count = masks.size(0)
    if x.ndim == 2:
        if x.size(1) != SEQ_LEN:
            raise ValueError(f"x must have shape [examples, {SEQ_LEN}]")
        preactivation = torch.einsum("ni,cih->cnh", x, params["w"] * masks)
    elif x.ndim == 3:
        if x.shape[0] != count or x.size(2) != SEQ_LEN:
            raise ValueError(f"per-run x must have shape [runs, examples, {SEQ_LEN}]")
        preactivation = torch.einsum("cni,cih->cnh", x, params["w"] * masks)
    else:
        raise ValueError("x must have rank two or three")
    hidden = F.relu(preactivation + params["b"][:, None, :])
    return torch.einsum("cnh,ch->cn", hidden, params["a"]) + params["c"][:, None]


def _init_children(seeds: list[int] | torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    if isinstance(seeds, torch.Tensor):
        seeds = [int(value) for value in seeds.detach().cpu().tolist()]
    initialized = [init_child(seed, "cpu") for seed in seeds]
    return {
        key: torch.stack([params[key].detach().cpu() for params in initialized])
        .to(device).requires_grad_()
        for key in ("w", "b", "a", "c")
    }


def fit_child_batch(
    masks: torch.Tensor,
    x_support: torch.Tensor,
    y_support: torch.Tensor,
    *,
    x_query: Optional[torch.Tensor] = None,
    y_query: Optional[torch.Tensor] = None,
    seeds: Optional[list[int] | torch.Tensor] = None,
    learning_rates: float | torch.Tensor = 0.001,
    device: str | torch.device = "cpu",
    max_steps: int = 12000,
    min_steps: int = 1000,
    batch_size: int = 128,
    momentum: float = 0.9,
    beta2: float = 0.999,
    decay_every: int = 2000,
    minimum_lr_factor: float = 1.0 / 16.0,
    eval_every: int = 100,
    patience_steps: int = 1000,
    plateau_relative: float = 0.01,
    checkpoint_path: Optional[str | Path] = None,
    resume: bool = False,
    extend: bool = False,
    max_updates: Optional[int] = None,
) -> dict[str, Any]:
    """Vectorized Adam fit for C children with query-selected snapshots.

    Runs share one vectorized model operation but keep independent Adam state,
    minibatch RNG streams, convergence status, and checkpoints. Inputs and
    labels may be shared or have a leading run dimension. Query data selects
    checkpoints and enters the convergence audit; it never supplies gradients.
    """
    device = torch.device(device)
    masks = masks.detach().to(device=device, dtype=torch.float32)
    if masks.ndim != 3 or masks.shape[1:] != (SEQ_LEN, HIDDEN):
        raise ValueError(f"masks must have shape [runs, {SEQ_LEN}, {HIDDEN}]")
    count = masks.size(0)

    def normalize_inputs(x: torch.Tensor, y: torch.Tensor, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.detach().to(device=device, dtype=torch.float32)
        y = y.detach().to(device=device, dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(0).expand(count, -1, -1)
        if y.ndim == 1:
            y = y.unsqueeze(0).expand(count, -1)
        if x.ndim != 3 or x.shape[0] != count or x.size(2) != SEQ_LEN:
            raise ValueError(f"{name} x must have shape [examples,11] or [runs,examples,11]")
        if y.shape != x.shape[:2]:
            raise ValueError(f"{name} y must align with its run and example dimensions")
        return x, y

    x_support, y_support = normalize_inputs(x_support, y_support, "support")
    if (x_query is None) != (y_query is None):
        raise ValueError("x_query and y_query must be supplied together")
    if x_query is not None:
        x_query, y_query = normalize_inputs(x_query, y_query, "query")
    if (min_steps < 0 or max_steps < 1 or max_steps > 48000 or min_steps > max_steps
            or batch_size < 2 or eval_every < 1):
        raise ValueError("invalid batched child fit schedule")
    if decay_every < 1 or not 0 < minimum_lr_factor <= 1:
        raise ValueError("invalid child learning-rate decay")
    if max_updates is not None and max_updates < 1:
        raise ValueError("max_updates must be positive")
    if seeds is None:
        seeds = list(range(count))
    elif isinstance(seeds, torch.Tensor):
        seeds = [int(value) for value in seeds.detach().cpu().tolist()]
    if len(seeds) != count:
        raise ValueError("one initialization seed is required per run")
    params = _init_children(seeds, device)
    first_moment = {key: torch.zeros_like(value) for key, value in params.items()}
    second_moment = {key: torch.zeros_like(value) for key, value in params.items()}
    time_count = torch.zeros(count, dtype=torch.int64, device=device)
    if isinstance(learning_rates, torch.Tensor):
        lr = learning_rates.detach().to(device=device, dtype=torch.float32).reshape(-1)
    else:
        lr = torch.full((count,), float(learning_rates), device=device)
    if lr.numel() == 1:
        lr = lr.expand(count)
    if lr.numel() != count or (lr <= 0).any():
        raise ValueError("learning_rates must be positive and scalar or one per run")

    pos_indices: list[torch.Tensor] = []
    neg_indices: list[torch.Tensor] = []
    for run in range(count):
        pos_indices.append(torch.nonzero(y_support[run].detach().cpu() > 0.5,
                                         as_tuple=False).flatten())
        neg_indices.append(torch.nonzero(y_support[run].detach().cpu() <= 0.5,
                                         as_tuple=False).flatten())
        if not pos_indices[-1].numel() or not neg_indices[-1].numel():
            raise ValueError("each support run must include both classes")
    # Independent streams make paired methods/LRs consume the same batches
    # whenever their run seed and support rows match.
    # Paired methods/LRs have identical sampling streams. Draw each unique
    # stream once per step rather than repeating CPU RNG calls for every row.
    samplers: list[torch.Generator] = []
    sampler_groups: list[int] = []
    group_representatives: list[int] = []
    group_by_key: dict[tuple, int] = {}
    for run, seed in enumerate(seeds):
        key = (int(seed), tuple(pos_indices[run].tolist()), tuple(neg_indices[run].tolist()))
        if key not in group_by_key:
            group_by_key[key] = len(group_representatives)
            group_representatives.append(run)
            samplers.append(torch.Generator(device="cpu").manual_seed(int(seed) + 70_001))
        sampler_groups.append(group_by_key[key])
    sampler_rows = torch.tensor(sampler_groups, dtype=torch.long, device=device)

    def metrics_batch(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        positive = labels > 0.5
        negative = ~positive
        pos_count = positive.sum(1).clamp_min(1)
        neg_count = negative.sum(1).clamp_min(1)
        correct = ((logits > 0) == positive).to(logits.dtype)
        return {
            "natural_bce": losses.mean(1),
            "balanced_bce": 0.5 * ((losses * positive).sum(1) / pos_count
                                    + (losses * negative).sum(1) / neg_count),
            "natural_accuracy": correct.mean(1),
            "balanced_accuracy": 0.5 * ((correct * positive).sum(1) / pos_count
                                         + (correct * negative).sum(1) / neg_count),
        }

    def signature(include_max_steps: bool = True) -> str:
        digest = hashlib.sha256()
        for tensor in (masks, x_support, y_support, x_query, y_query, lr):
            if tensor is not None:
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        digest.update(str(tuple(int(value) for value in seeds)).encode())
        options = (max_steps, min_steps, batch_size, momentum, beta2,
                   eval_every, patience_steps, plateau_relative,
                   decay_every, minimum_lr_factor)
        if not include_max_steps:
            options = options[1:]
        digest.update(str(options).encode())
        return digest.hexdigest()

    protocol_hash = signature()
    extension_hash = signature(include_max_steps=False)

    def evaluate(step_number: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        with torch.no_grad():
            support_logits = child_logits_batch(x_support, masks, params)
            support_metrics = metrics_batch(support_logits, y_support)
            if x_query is not None:
                query_logits = child_logits_batch(x_query, masks, params)
                query_metrics = metrics_batch(query_logits, y_query)
            else:
                query_metrics = support_metrics
        metric_tensors = {}
        for key, short_name in (("balanced_bce", "balanced_bce"),
                                ("natural_bce", "natural_bce"),
                                ("balanced_accuracy", "balanced_accuracy"),
                                ("natural_accuracy", "accuracy")):
            metric_tensors[f"query_{short_name}"] = query_metrics[key].detach().cpu()
            metric_tensors[f"support_{short_name}"] = support_metrics[key].detach().cpu()
        metric_tensors["step"] = torch.tensor(step_number, dtype=torch.int64)
        return {key: value.detach().clone() for key, value in params.items()}, metric_tensors

    best_params, first_metrics = evaluate(0)
    best_scores = first_metrics["query_balanced_bce"].clone()
    best_steps = torch.zeros(count, dtype=torch.int64)
    plateau_passes = torch.zeros(count, dtype=torch.int64)
    converged = torch.zeros(count, dtype=torch.bool)
    frozen_at = torch.zeros(count, dtype=torch.int64)
    history = [first_metrics]
    start_step = 0
    if resume and checkpoint_path is not None and Path(checkpoint_path).exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        exact_resume = saved.get("protocol_hash") == protocol_hash
        extension_resume = (extend and saved.get("extension_hash") == extension_hash
                            and saved.get("max_steps", 0) < max_steps
                            and saved.get("stop_reason") == "step_cap")
        if not (exact_resume or extension_resume):
            raise ValueError("batched child resume inputs or hyperparameters changed")
        start_step = int(saved["step"])
        params = {key: value.to(device).detach().requires_grad_()
                  for key, value in saved["last_params"].items()}
        best_params = {key: value.to(device) for key, value in saved["best_params"].items()}
        first_moment = {key: value.to(device) for key, value in saved["first_moment"].items()}
        second_moment = {key: value.to(device) for key, value in saved["second_moment"].items()}
        time_count = saved["time_count"].to(device)
        best_scores = saved["best_scores"]
        best_steps = saved["best_steps"]
        plateau_passes = saved["plateau_passes"]
        converged = saved["converged"]
        frozen_at = saved["frozen_at"]
        history = saved["history"]
        for group, run in enumerate(group_representatives):
            samplers[group].set_state(saved["sampler_states"][run])
    elif resume and checkpoint_path is not None:
        raise FileNotFoundError(f"child checkpoint not found: {checkpoint_path}")

    def save_checkpoint(step_number: int, current: dict[str, torch.Tensor]) -> None:
        if checkpoint_path is None:
            return
        checkpoint = {
            "protocol": "task_quality_child_batch_v1", "protocol_hash": protocol_hash,
            "extension_hash": extension_hash, "max_steps": int(max_steps),
            "stop_reason": (
                "empirical_train_query_plateau" if bool(converged.all())
                else ("step_cap" if step_number >= max_steps else "running")
            ),
            "step": step_number,
            "last_params": {key: value.detach().cpu() for key, value in current.items()},
            "best_params": {key: value.detach().cpu() for key, value in best_params.items()},
            "first_moment": {key: value.detach().cpu() for key, value in first_moment.items()},
            "second_moment": {key: value.detach().cpu() for key, value in second_moment.items()},
            "time_count": time_count.detach().cpu(), "best_scores": best_scores,
            "best_steps": best_steps, "plateau_passes": plateau_passes,
            "converged": converged, "frozen_at": frozen_at,
            "sampler_states": [samplers[group].get_state() for group in sampler_groups],
            "history": history,
        }
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, temp)
        temp.replace(path)

    step = start_step
    end_step = max_steps if max_updates is None else min(max_steps, start_step + max_updates)
    if bool(converged.all()):
        end_step = start_step
    for step in range(start_step + 1, end_step + 1):
        sampled = []
        for group, run in enumerate(group_representatives):
            pos = pos_indices[run]
            neg = neg_indices[run]
            pos_draw = pos[torch.randint(pos.numel(), (batch_size // 2,), generator=samplers[group])]
            neg_draw = neg[torch.randint(neg.numel(), (batch_size - batch_size // 2,),
                                        generator=samplers[group])]
            indices = torch.cat((pos_draw, neg_draw))
            sampled.append(indices[torch.randperm(indices.numel(), generator=samplers[group])])
        indices = torch.stack(sampled).to(device).index_select(0, sampler_rows)
        run_indices = torch.arange(count, device=device)[:, None]
        xb = x_support[run_indices, indices]
        yb = y_support[run_indices, indices]
        logits = child_logits_batch(xb, masks, params)
        loss_per = F.binary_cross_entropy_with_logits(logits, yb, reduction="none").mean(dim=1)
        gradients = torch.autograd.grad(loss_per.sum(), tuple(params.values()))
        active = ~converged.to(device)
        time_count += active.to(torch.int64)
        updated: dict[str, torch.Tensor] = {}
        for (key, value), gradient in zip(params.items(), gradients):
            row_shape = (count,) + (1,) * (value.ndim - 1)
            active_view = active.view(row_shape)
            gradient = torch.where(active_view, gradient, torch.zeros_like(gradient))
            first_moment[key] = momentum * first_moment[key] + (1.0 - momentum) * gradient
            second_moment[key] = beta2 * second_moment[key] + (1.0 - beta2) * gradient.square()
            first_moment[key] = torch.where(active_view, first_moment[key], torch.zeros_like(first_moment[key]))
            second_moment[key] = torch.where(active_view, second_moment[key], torch.zeros_like(second_moment[key]))
            t = time_count.clamp_min(1).to(value.dtype).view(row_shape)
            m_hat = first_moment[key] / (1.0 - momentum ** t)
            v_hat = second_moment[key] / (1.0 - beta2 ** t)
            decay_factor = max(minimum_lr_factor, 0.5 ** ((step - 1) // decay_every))
            lr_view = (lr * decay_factor).view(row_shape)
            candidate = value - lr_view * m_hat / (v_hat.sqrt() + 1e-8)
            updated[key] = torch.where(active_view, candidate, value).detach().requires_grad_()
        params = updated
        if step % eval_every:
            continue
        candidate_params, metrics = evaluate(step)
        history.append(metrics)
        scores = metrics["query_balanced_bce"]
        improved = scores < best_scores
        best_scores = torch.minimum(best_scores, scores)
        best_steps = torch.where(improved, torch.full_like(best_steps, step), best_steps)
        for key in best_params:
            shape = (count,) + (1,) * (best_params[key].ndim - 1)
            best_params[key] = torch.where(improved.to(device).view(shape),
                                           candidate_params[key], best_params[key])
        if step >= min_steps:
            train_hist = [entry["support_balanced_bce"] for entry in history]
            query_hist = [entry["query_balanced_bce"] for entry in history]
            train_plateau = loss_plateau(train_hist, width=8, tolerance=plateau_relative)
            query_plateau = loss_plateau(query_hist, width=8, tolerance=plateau_relative)
            plateau_now = train_plateau & query_plateau & ~converged
            plateau_passes = torch.where(plateau_now, plateau_passes + 1,
                                         torch.where(converged, plateau_passes,
                                                     torch.zeros_like(plateau_passes)))
            newly_converged = plateau_now & (plateau_passes >= 3)
            converged |= newly_converged
            frozen_at = torch.where(newly_converged,
                                    torch.full_like(frozen_at, step), frozen_at)
        save_checkpoint(step, params)
        if bool(converged.all()):
            break
    last_params, last_metrics = evaluate(step)
    if int(history[-1]["step"]) != step:
        history.append(last_metrics)
    final_improved = last_metrics["query_balanced_bce"] < best_scores
    best_scores = torch.minimum(best_scores, last_metrics["query_balanced_bce"])
    best_steps = torch.where(final_improved, torch.full_like(best_steps, step), best_steps)
    for key in best_params:
        shape = (count,) + (1,) * (best_params[key].ndim - 1)
        best_params[key] = torch.where(final_improved.to(device).view(shape),
                                       last_params[key], best_params[key])
    if step % eval_every and step >= min_steps and not bool(converged.all()):
        train_hist = [entry["support_balanced_bce"] for entry in history]
        query_hist = [entry["query_balanced_bce"] for entry in history]
        train_plateau = loss_plateau(train_hist, width=8, tolerance=plateau_relative)
        query_plateau = loss_plateau(query_hist, width=8, tolerance=plateau_relative)
        plateau_now = train_plateau & query_plateau & ~converged
        plateau_passes = torch.where(plateau_now, plateau_passes + 1,
                                     torch.where(converged, plateau_passes,
                                                 torch.zeros_like(plateau_passes)))
        newly_converged = plateau_now & (plateau_passes >= 3)
        converged |= newly_converged
        frozen_at = torch.where(newly_converged,
                                torch.full_like(frozen_at, step), frozen_at)
    save_checkpoint(step, last_params)
    stop_reasons = ["empirical_train_query_plateau" if value else "step_cap"
                    for value in converged.tolist()]
    if bool(converged.all()):
        termination_reason = "empirical_train_query_plateau"
    elif step >= max_steps:
        termination_reason = "step_cap"
    elif max_updates is not None:
        termination_reason = "checkpoint_chunk"
    else:
        termination_reason = "step_cap"
    per_run_steps = torch.where(converged, frozen_at,
                                torch.full_like(frozen_at, step))
    fit_status = [
        {"converged": bool(converged[index]), "stop_reason": stop_reasons[index],
         "steps": int(per_run_steps[index]), "max_steps": int(max_steps),
         "selection_complete": x_query is not None}
        for index in range(count)
    ]
    return {
        "params": best_params,
        "best_params": best_params,
        "last_params": last_params,
        "best_steps": best_steps,
        "best_query_balanced_bce": best_scores,
        "history": history,
        "steps": per_run_steps,
        "total_steps": int(step),
        "max_steps": int(max_steps),
        "converged": converged,
        "stop_reason": stop_reasons,
        "fit_status": fit_status,
        "selection_complete": torch.full((count,), x_query is not None, dtype=torch.bool),
        "termination_reason": termination_reason,
        "checkpoint_selected_on": "query" if x_query is not None else "support",
        "frozen_before_test": True,
        "lr": lr.detach().cpu(),
    }


def _batched_logits(x: torch.Tensor, mask: torch.Tensor,
                    params: dict[str, torch.Tensor]) -> torch.Tensor:
    """Vectorized child forward for task and initialization axes."""
    # x [tasks, examples, input], mask [tasks, input, hidden],
    # params [tasks, replicas, ...] -> logits [tasks, replicas, examples].
    hidden_pre = torch.einsum("tbi,trih->trbh", x, params["w"] * mask[:, None])
    hidden = F.relu(hidden_pre + params["b"][:, :, None, :])
    return torch.einsum("trbh,trh->trb", hidden, params["a"]) + params["c"][:, :, None]


def _init_child_batch(n_tasks: int, n_replicas: int, rng: torch.Generator,
                      device: torch.device) -> dict[str, torch.Tensor]:
    w = torch.randn((n_tasks, n_replicas, SEQ_LEN, HIDDEN), generator=rng) * 0.1
    a = torch.randn((n_tasks, n_replicas, HIDDEN), generator=rng) * 0.1
    return {
        "w": w.to(device).requires_grad_(),
        "b": torch.zeros((n_tasks, n_replicas, HIDDEN), device=device, requires_grad=True),
        "a": a.to(device).requires_grad_(),
        "c": torch.zeros((n_tasks, n_replicas), device=device, requires_grad=True),
    }


def _inner_adapt(
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    mask: torch.Tensor,
    params: dict[str, torch.Tensor],
    *,
    steps: int,
    learning_rate: float,
    momentum: float,
    create_graph: bool,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    velocities = {key: torch.zeros_like(value) for key, value in params.items()}
    targets = support_y[:, None, :].expand(-1, params["w"].size(1), -1)
    for _ in range(steps):
        logits = _batched_logits(support_x, mask, params)
        loss_per = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dim=-1)
        gradients = torch.autograd.grad(loss_per.sum(), tuple(params.values()),
                                        create_graph=create_graph)
        new_params: dict[str, torch.Tensor] = {}
        new_velocities: dict[str, torch.Tensor] = {}
        for (key, value), gradient in zip(params.items(), gradients):
            velocity = momentum * velocities[key] + gradient
            updated = value - learning_rate * velocity
            if create_graph:
                new_params[key] = updated
                new_velocities[key] = velocity
            else:
                new_params[key] = updated.detach().requires_grad_()
                new_velocities[key] = velocity.detach()
        params = new_params
        velocities = new_velocities
    final_logits = _batched_logits(support_x, mask, params)
    final_loss = F.binary_cross_entropy_with_logits(final_logits, targets, reduction="none").mean(dim=-1)
    return params, final_loss


def _stack_episode_samples(pools: list[dict[str, dict[str, torch.Tensor]]],
                           task_ids: list[str], n_support: int, n_query: int,
                           rng: torch.Generator) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    support = [sample_balanced(pools[task_id]["support"], n_support, rng) for task_id in task_ids]
    query = [sample_balanced(pools[task_id]["query"], n_query, rng) for task_id in task_ids]
    return (
        {key: torch.stack([item[key] for item in support]) for key in ("x", "y", "ids")},
        {key: torch.stack([item[key] for item in query]) for key in ("x", "y", "ids")},
    )


def _fixed_validation_episodes(data: dict[str, Any], config: MetaConfig,
                               seed: int) -> tuple[list[str], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    task_ids = [task.task_id for task in data["splits"]["val"]]
    rng = torch.Generator(device="cpu").manual_seed(seed)
    support, query = _stack_episode_samples(data["pools"], task_ids,
                                            config.support_size, config.query_size, rng)
    return task_ids, support, query


def _fixed_train_episodes(data: dict[str, Any], config: MetaConfig,
                          seed: int) -> tuple[list[str], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    task_ids = [task.task_id for task in data["splits"]["train"]]
    rng = torch.Generator(device="cpu").manual_seed(seed)
    support, query = _stack_episode_samples(data["pools"], task_ids,
                                            config.support_size, config.query_size, rng)
    return task_ids, support, query


def _run_validation(model: Generator, bank_feature: Optional[torch.Tensor],
                    support: dict[str, torch.Tensor], query: dict[str, torch.Tensor],
                    seed: int, config: MetaConfig, device: torch.device) -> dict[str, torch.Tensor]:
    support_x = support["x"].to(device)
    support_y = support["y"].to(device)
    query_x = query["x"].to(device)
    query_y = query["y"].to(device)
    local_rng = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        masks, _ = generate(model, bank_feature, support_x, support_y)
    masks = masks.detach()
    params = _init_child_batch(support_x.size(0), config.validation_replicas,
                               local_rng, device)
    with torch.enable_grad():
        params, _ = _inner_adapt(
            support_x, support_y, masks, params, steps=config.inner_steps,
            learning_rate=config.inner_learning_rate, momentum=config.inner_momentum,
            create_graph=False,
        )
    with torch.no_grad():
        query_logits = _batched_logits(query_x, masks, params)
        labels = query_y[:, None, :].expand_as(query_logits)
        losses = F.binary_cross_entropy_with_logits(query_logits, labels, reduction="none")
        positive = labels > 0.5
        negative = ~positive
        balanced_bce = 0.5 * (losses[positive].mean() + losses[negative].mean())
        correct = (query_logits > 0) == positive
        balanced_accuracy = 0.5 * (correct[positive].float().mean() + correct[negative].float().mean())
        per_task = 0.5 * (
            (losses * positive).sum(dim=(1, 2)) / positive.sum(dim=(1, 2)).clamp_min(1)
            + (losses * negative).sum(dim=(1, 2)) / negative.sum(dim=(1, 2)).clamp_min(1)
        )
    return {
        "query_balanced_bce": balanced_bce.detach().cpu(),
        "query_balanced_accuracy": balanced_accuracy.detach().cpu(),
        "query_balanced_bce_per_task": per_task.detach().cpu(),
        "masks": masks.detach().cpu(),
    }


def _cpu_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _cpu_optimizer_state(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    # ``Optimizer.state_dict`` may retain references to live Adam buffers.
    # Copy first so serializing a checkpoint cannot move an active GPU optimizer
    # state to CPU underneath the next update.
    state = copy.deepcopy(optimizer.state_dict())
    for item in state["state"].values():
        for key, value in item.items():
            if isinstance(value, torch.Tensor):
                item[key] = value.detach().cpu()
    return state


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_meta(
    bank_path: str | Path | None,
    out: str | Path,
    seed: int = 8100,
    method: str = "transformer_mask",
    device: str | torch.device = "cpu",
    config: Optional[MetaConfig] = None,
    resume: bool = True,
    extend: bool = False,
    max_updates: Optional[int] = None,
) -> Path:
    """Train one mask generator on train tasks and select on val tasks only."""
    config = config or MetaConfig(method=method)
    if config.method != method:
        raise ValueError("config.method and method argument differ")
    config.validate()
    bank_path = Path(bank_path) if bank_path is not None else None
    out = Path(out)
    device = torch.device(device)
    if method == "free_mask":
        bank_path = None
    if method == "transformer_mask" and (bank_path is None or not bank_path.exists()):
        raise FileNotFoundError(f"source feature bank not found: {bank_path}")
    bank_payload = (torch.load(bank_path, map_location="cpu", weights_only=False)
                    if method == "transformer_mask" else None)
    bank_sha256 = _file_sha256(bank_path) if bank_path is not None else None
    bank_feature = None if method == "free_mask" else bank_payload["feature"].to(device)
    if bank_feature is not None and bank_feature.shape[1:] != (HIDDEN, FEATURE_DIM):
        raise ValueError(f"unexpected bank feature shape: {tuple(bank_feature.shape)}")

    data = build_experiment_data(task_seed=42, split_seed=1729,
                                 probe_size=128, probe_seed=seed)
    train_task_ids = [task.task_id for task in data["splits"]["train"]]
    val_task_ids = [task.task_id for task in data["splits"]["val"]]
    if config.meta_batch_tasks > len(train_task_ids):
        raise ValueError("meta_batch_tasks exceeds the number of source tasks")
    train_monitor_ids, train_monitor_support, train_monitor_query = _fixed_train_episodes(
        data, config, seed + 601)
    val_ids, val_support, val_query = _fixed_validation_episodes(data, config, seed + 701)
    # Seed before constructing the generator so method/seed runs are repeatable.
    torch.manual_seed(int(seed) + 31)
    model = Generator(feature_dim=FEATURE_DIM, mode=method).to(device)
    # Method runs with the same seed begin from paired child initializations and
    # episode samples; only the generator architecture differs.
    episode_rng = torch.Generator(device="cpu").manual_seed(int(seed) + 1201)
    augmentation_rng = torch.Generator(device="cpu").manual_seed(int(seed) + 6201)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.outer_learning_rate)
    curves: dict[str, list[torch.Tensor]] = {
        "step": [], "stochastic_train_query_bce": [], "train_monitor_query_bce": [],
        "val_query_bce": [], "val_query_accuracy": [],
    }
    step = 0
    best_val = math.inf
    best_step = 0
    train_plateau_passes = 0
    val_plateau_passes = 0
    stop_reason = "step_cap"
    best_payload: Optional[dict[str, Any]] = None
    resume_path = out / "meta" / "checkpoint.pt"
    best_path = out / "meta" / "best.pt"
    last_path = out / "meta" / "last.pt"
    if resume and resume_path.exists():
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        old_config = saved.get("config", {})
        current_config = asdict(config)
        compatible = old_config == current_config
        extension_compatible = (
            extend and old_config.get("max_steps", 0) < config.max_steps
            and {key: value for key, value in old_config.items() if key != "max_steps"}
            == {key: value for key, value in current_config.items() if key != "max_steps"}
            and saved.get("stop_reason") == "step_cap"
        )
        if not (compatible or extension_compatible) or saved.get("seed") != int(seed):
            raise ValueError(f"meta resume protocol mismatch: {resume_path}")
        model.load_state_dict(saved["model_state"])
        model.to(device)
        optimizer.load_state_dict(saved["optimizer_state"])
        episode_rng.set_state(saved["episode_rng"])
        step = int(saved["step"])
        curves = saved["curves"]
        best_val = float(saved["best_val"])
        best_step = int(saved.get("best_step", 0))
        train_plateau_passes = int(saved["train_plateau_passes"])
        val_plateau_passes = int(saved["val_plateau_passes"])
        stop_reason = saved.get("stop_reason", "step_cap")
        if bank_path is not None and saved.get("bank_sha256") != bank_sha256:
            raise ValueError("source bank changed since this meta checkpoint was created")
        if "augmentation_rng" in saved:
            augmentation_rng.set_state(saved["augmentation_rng"])
        # Final resumable payloads keep compact stacked curves. Restore mutable
        # lists here so resumed runs can append new evaluation points.
        curves = {
            key: (list(value.unbind(0)) if isinstance(value, torch.Tensor) else list(value))
            for key, value in curves.items()
        }

    call_start_step = step

    task_lookup = data["pools"]
    while step < config.max_steps and (max_updates is None or step - call_start_step < max_updates):
        task_order = torch.randperm(len(train_task_ids), generator=episode_rng)
        chosen_task_indices = task_order[:config.meta_batch_tasks]
        batch_task_ids = [train_task_ids[int(index)] for index in chosen_task_indices]
        support, query = _stack_episode_samples(task_lookup, batch_task_ids,
                                                config.support_size, config.query_size, episode_rng)
        support_x = support["x"].to(device)
        support_y = support["y"].to(device)
        query_x = query["x"].to(device)
        query_y = query["y"].to(device)
        current_bank = bank_feature
        if current_bank is not None and config.hidden_column_augmentation:
            hidden_order = torch.stack([
                torch.randperm(HIDDEN, generator=augmentation_rng)
                for _ in range(current_bank.size(0))
            ])
            current_bank = permute_hidden_columns(current_bank, hidden_order)
        if current_bank is not None and config.map_order_augmentation:
            order = torch.randperm(current_bank.size(0), generator=augmentation_rng)
            current_bank = current_bank.index_select(0, order.to(device))

        masks, _ = generate(model, current_bank, support_x, support_y)
        params = _init_child_batch(config.meta_batch_tasks, config.replicas,
                                   episode_rng, device)
        params, _ = _inner_adapt(
            support_x, support_y, masks, params, steps=config.inner_steps,
            learning_rate=config.inner_learning_rate, momentum=config.inner_momentum,
            create_graph=True,
        )
        query_logits = _batched_logits(query_x, masks, params)
        query_targets = query_y[:, None, :].expand_as(query_logits)
        outer_loss = F.binary_cross_entropy_with_logits(query_logits, query_targets)
        optimizer.zero_grad(set_to_none=True)
        outer_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        step += 1
        curves["stochastic_train_query_bce"].append(outer_loss.detach().cpu())

        if step and step % config.lr_decay_every == 0:
            new_lr = max(config.outer_minimum_lr,
                         config.outer_learning_rate * config.lr_decay_factor **
                         (step // config.lr_decay_every))
            for group in optimizer.param_groups:
                group["lr"] = new_lr
        if step % config.eval_every:
            continue

        train_monitor_result = _run_validation(
            model, bank_feature, train_monitor_support, train_monitor_query,
            seed + 8000, config, device)
        val_result = _run_validation(model, bank_feature, val_support, val_query,
                                     seed + 9000, config, device)
        val_score = float(val_result["query_balanced_bce"])
        curves["step"].append(torch.tensor(step, dtype=torch.int64))
        curves["train_monitor_query_bce"].append(train_monitor_result["query_balanced_bce"])
        curves["val_query_bce"].append(val_result["query_balanced_bce"])
        curves["val_query_accuracy"].append(val_result["query_balanced_accuracy"])
        if val_score < best_val:
            best_val = val_score
            best_step = step
            best_payload = {
                "protocol": "task_quality_meta_v1", "method": method,
                "seed": int(seed), "step": step,
                "model_state": _cpu_state(model), "val": val_result,
                "best_val_query_balanced_bce": best_val,
                "config": asdict(config), "bank_path": str(bank_path) if bank_path else None,
                "bank_sha256": bank_sha256,
                "bank_protocol": bank_payload.get("protocol") if bank_payload else None,
                "feature_mean": bank_payload.get("feature_mean") if bank_payload else None,
                "feature_std": bank_payload.get("feature_std") if bank_payload else None,
                "train_task_ids": train_task_ids, "val_task_ids": val_task_ids,
                "train_monitor_task_ids": train_monitor_ids,
                "val_episode_task_ids": val_ids,
                "val_support_ids": val_support["ids"], "val_query_ids": val_query["ids"],
                "inner_horizon": config.inner_steps,
                "finite_horizon_objective": True,
                "inner_solver_converged_claim": False,
                "history": {key: torch.stack(value) if value else torch.empty(0)
                            for key, value in curves.items()},
            }
            _atomic_save(best_payload, best_path)

        if step >= config.min_steps:
            train_flat = bool(loss_plateau(curves["train_monitor_query_bce"], width=8,
                                           tolerance=config.relative_improvement))
            val_flat = bool(loss_plateau(curves["val_query_bce"], width=8,
                                         tolerance=config.relative_improvement))
            train_plateau_passes = train_plateau_passes + 1 if train_flat else 0
            val_plateau_passes = val_plateau_passes + 1 if val_flat else 0
        if (step >= config.min_steps
                and train_plateau_passes >= config.plateau_passes
                and val_plateau_passes >= config.plateau_passes):
            stop_reason = "empirical_train_val_outer_query_plateau"

        checkpoint_stop_reason = (
            stop_reason if stop_reason == "empirical_train_val_outer_query_plateau"
            else ("step_cap" if step >= config.max_steps else "running")
        )
        latest = {
            "protocol": "task_quality_meta_v1", "method": method,
            "seed": int(seed), "step": step, "model_state": _cpu_state(model),
            "optimizer_state": _cpu_optimizer_state(optimizer), "episode_rng": episode_rng.get_state(),
            "augmentation_rng": augmentation_rng.get_state(),
            "best_val": best_val, "best_step": best_step,
            "train_plateau_passes": train_plateau_passes,
            "val_plateau_passes": val_plateau_passes, "config": asdict(config),
            "bank_path": str(bank_path) if bank_path else None,
            "bank_sha256": bank_sha256,
            "train_task_ids": train_task_ids, "val_task_ids": val_task_ids,
            "train_monitor_task_ids": train_monitor_ids,
            "val_episode_task_ids": val_ids, "val_support_ids": val_support["ids"],
            "val_query_ids": val_query["ids"], "curves": curves,
            "stop_reason": checkpoint_stop_reason,
        }
        _atomic_save(latest, resume_path)
        _atomic_save(latest, last_path)
        if stop_reason == "empirical_train_val_outer_query_plateau":
            break

    if step >= config.max_steps and stop_reason != "empirical_train_val_outer_query_plateau":
        stop_reason = "step_cap"
    elif (stop_reason != "empirical_train_val_outer_query_plateau"
          and max_updates is not None and step - call_start_step >= max_updates):
        stop_reason = "checkpoint_chunk"
    last_payload = {
        "protocol": "task_quality_meta_v1", "method": method,
        "seed": int(seed), "step": step, "model_state": _cpu_state(model),
        "optimizer_state": _cpu_optimizer_state(optimizer), "episode_rng": episode_rng.get_state(),
        "augmentation_rng": augmentation_rng.get_state(),
        "best_val": best_val, "best_step": best_step,
        "train_plateau_passes": train_plateau_passes,
        "val_plateau_passes": val_plateau_passes, "config": asdict(config),
        "bank_path": str(bank_path) if bank_path else None,
        "bank_sha256": bank_sha256,
        "train_task_ids": train_task_ids, "val_task_ids": val_task_ids,
        "train_monitor_task_ids": train_monitor_ids,
        "val_episode_task_ids": val_ids, "val_support_ids": val_support["ids"],
        "val_query_ids": val_query["ids"],
        "curves": {key: torch.stack(value) if value else torch.empty(0)
                   for key, value in curves.items()},
        "stop_reason": stop_reason,
        "converged": stop_reason == "empirical_train_val_outer_query_plateau",
        "cap_hit": stop_reason == "step_cap",
    }
    _atomic_save(last_payload, last_path)
    if not best_path.exists():
        # A zero-step smoke configuration still produces a loadable checkpoint.
        _atomic_save({**last_payload, "best_val_query_balanced_bce": math.inf}, best_path)
    _atomic_save(last_payload, resume_path)
    if best_path.exists():
        best_existing = torch.load(best_path, map_location="cpu", weights_only=False)
        best_existing.update({
            "config": asdict(config),
            "history": last_payload["curves"],
            "stop_reason": stop_reason,
            "converged": last_payload["converged"],
            "cap_hit": last_payload["cap_hit"],
            "step_cap": config.max_steps,
            "inner_horizon": config.inner_steps,
            "finite_horizon_objective": True,
            "inner_solver_converged_claim": False,
        })
        _atomic_save(best_existing, best_path)
    return best_path


def fit_child(
    mask: torch.Tensor,
    x_support: torch.Tensor,
    y_support: torch.Tensor,
    *,
    x_query: Optional[torch.Tensor] = None,
    y_query: Optional[torch.Tensor] = None,
    seed: int = 0,
    device: str | torch.device = "cpu",
    max_steps: int = 12000,
    min_steps: int = 1000,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    momentum: float = 0.9,
    eval_every: int = 100,
    patience_steps: int = 1000,
    plateau_relative: float = 0.01,
    checkpoint_path: Optional[str | Path] = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Single-child wrapper around the vectorized Adam fitter."""
    result = fit_child_batch(
        mask.unsqueeze(0), x_support, y_support,
        x_query=x_query, y_query=y_query, seeds=[seed],
        learning_rates=learning_rate, device=device, max_steps=max_steps,
        min_steps=min_steps, batch_size=batch_size, momentum=momentum,
        eval_every=eval_every, patience_steps=patience_steps,
        plateau_relative=plateau_relative, checkpoint_path=checkpoint_path,
        resume=resume,
    )
    history = []
    for entry in result["history"]:
        row = {key: (int(value) if key == "step" else float(value[0]))
               for key, value in entry.items()}
        history.append(row)
    best_params = {key: value[0] for key, value in result["best_params"].items()}
    last_params = {key: value[0] for key, value in result["last_params"].items()}
    converged = bool(result["converged"][0])
    stop_reason = result["stop_reason"][0]
    return {
        "params": best_params,
        "best_params": best_params,
        "last_params": last_params,
        "best_step": int(result["best_steps"][0]),
        "best_query_balanced_bce": float(result["best_query_balanced_bce"][0]),
        "history": history,
        "steps": int(result["steps"][0]),
        "max_steps": int(max_steps),
        "converged": converged,
        "stop_reason": stop_reason,
        "selection_complete": True,
        "checkpoint_selected_on": result["checkpoint_selected_on"],
        "frozen_before_test": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=False)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=8100)
    parser.add_argument("--method", choices=("transformer_mask", "free_mask"),
                        default="transformer_mask")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--min-steps", type=int, default=MetaConfig.min_steps)
    parser.add_argument("--max-steps", type=int, default=MetaConfig.max_steps)
    parser.add_argument("--replicas", type=int, default=MetaConfig.replicas)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    bank_path = args.bank or args.out / "bank.pt"
    config = MetaConfig(method=args.method, min_steps=args.min_steps,
                        max_steps=args.max_steps, replicas=args.replicas)
    path = train_meta(bank_path, args.out, args.seed, args.method, args.device,
                      config, resume=not args.no_resume)
    print(path)


if __name__ == "__main__":
    main()
