"""Fixed-horizon batched fresh-child fits for utility-based mask comparison.

Support labels alone determine gradients and the optimizer horizon. Query
losses are computed only at checkpoints as diagnostics and once at termination;
they never select parameters or affect stopping. The model and evaluation
normalization reuse ``followup_batched_eval.BatchedConditions`` and ``losses``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor

from .core import MaskedDeepSets
from .followup_batched_eval import BatchedConditions, losses


def _exact_replica_initialization(
    model: BatchedConditions,
    masks: Tensor,
    condition_seeds: Sequence[int],
    replicas: Sequence[int],
    reference_models: int,
) -> None:
    """Use the numbered rows of the same reference initialization bank.

    ``BatchedConditions`` pairs replicas by first occurrence. That is useful
    for generic repeated labels, but these fits have stable reference IDs: a
    child labelled replica ``r`` must receive reference draw ``r`` even when
    another method's replica appeared earlier in the input list.
    """
    features, hidden = masks.shape[1:]
    reference_masks = torch.ones(
        reference_models, features, hidden, dtype=masks.dtype, device=masks.device
    )
    replica_index = torch.as_tensor(replicas, dtype=torch.long, device=masks.device)
    with torch.no_grad():
        for condition, seed in enumerate(condition_seeds):
            reference = MaskedDeepSets(
                reference_masks,
                seed=int(seed),
                initialization_reference_models=reference_models,
            )
            model.weight[condition].copy_(reference.weight[replica_index])
            model.readout[condition].copy_(reference.readout[replica_index])


def _l2_penalty(model: BatchedConditions, l2: float) -> Tensor:
    """Per-child half-scaled sum of squares on effective parameters."""
    effective_weight = model.weight * model.masks
    return 0.5 * l2 * (
        effective_weight.square().sum(dim=(2, 3))
        + model.bias.square().sum(dim=-1)
        + model.readout.square().sum(dim=-1)
        + model.per_image_offset.square()
    )


def _streamed_losses(
    model: BatchedConditions,
    x: Tensor,
    y: Tensor,
    set_size: int,
    chunk_size: int,
    device: torch.device,
) -> Tensor:
    """Evaluate the shared batched loss while moving query chunks as needed."""
    total: Tensor | None = None
    count = x.shape[1]
    for start in range(0, count, chunk_size):
        stop = min(count, start + chunk_size)
        x_chunk = x[:, start:stop].to(device=device, dtype=torch.float32)
        y_chunk = y[:, start:stop].to(device=device, dtype=torch.float32)
        chunk_mean = losses(model, x_chunk, y_chunk, set_size, chunk=chunk_size)
        weighted = chunk_mean * (stop - start)
        total = weighted if total is None else total + weighted
    assert total is not None
    return total / count


def _plateau_flags(
    history: list[Tensor],
    steps: list[int],
    checkpoint_every: int,
    *,
    tolerance: float = 0.01,
) -> Tensor:
    """Support-objective stability audit over the last 100 and 50 updates.

    The 100-step endpoint change and the range across the trailing 50-step
    window must both be within 1% of the corresponding objective scale. With
    insufficient history the flag is false. This flag is descriptive only.
    """
    current = history[-1]
    n100 = max(1, int(round(100 / checkpoint_every)))
    n50 = max(1, int(round(50 / checkpoint_every)))
    if len(history) <= n100 or steps[-1] - steps[-1 - n100] < 100:
        return torch.zeros_like(current, dtype=torch.bool)
    before = history[-1 - n100]
    scale100 = torch.maximum(before.abs(), current.abs()).clamp_min(1e-8)
    stable100 = (current - before).abs() / scale100 <= tolerance
    recent = torch.stack(history[-(n50 + 1):])
    recent_mean = recent.mean(dim=0)
    scale50 = recent_mean.abs().clamp_min(1e-8)
    stable50 = (recent.amax(dim=0) - recent.amin(dim=0)) / scale50 <= tolerance
    return stable100 & stable50


def fit_children(
    masks: Tensor,
    x_support: Tensor,
    y_support: Tensor,
    x_query: Tensor,
    y_query: Tensor,
    condition_seeds: Sequence[int],
    replicas: Sequence[int],
    steps: int,
    lr: float,
    l2: float,
    device: str | torch.device,
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
    *,
    reference_models: int = 20,
    chunk_size: int = 32,
    batch_size: int | None = None,
    seed: int = 0,
    lr_decay_every: int = 200,
    lr_floor: float = 1.0 / 64.0,
    support_sampler: Callable[[int], tuple[Tensor, Tensor]] | None = None,
    initial_state: dict[str, Tensor] | None = None,
    optimizer_state: dict[str, Any] | None = None,
    start_step: int = 0,
    checkpoint_every: int = 25,
    plateau_tolerance: float = 0.01,
) -> dict[str, Any]:
    """Fit fresh children for paired task conditions and candidate masks.

    Args:
        masks: Candidate masks ``[M, 784, H]``.
        x_support/y_support: Paired support sets ``[C,N,5,784]`` and ``[C,N]``.
        x_query/y_query: Diagnostic query sets ``[C,Q,5,784]`` and ``[C,Q]``.
        condition_seeds: One deterministic child initialization seed per task.
        replicas: Reference-model initialization IDs, one per mask row.
        steps: Fixed Adam update count. No early stopping is performed.
        lr: Constant Adam learning rate.
        l2: Coefficient for ``0.5*l2*(||W*M||² + ||b||² + ||a||² + o²)``.
        device: Training device.
        checkpoint_callback: Optional one-argument callback receiving a CPU
            metrics record and the current trainable parameter tensors at each
            checkpoint. It does not control the optimizer.
        reference_models: Size of the shared initialization bank (default 20).
        chunk_size: Number of support/query sets per forward chunk. Training
            accumulates scaled gradients over every chunk before each step.
        batch_size: Optional support-only minibatch size. ``None`` (default)
            uses every support set at every step; smaller values sample rows
            independently per condition using the fixed ``seed`` and transfer
            only that minibatch.
        seed: CPU RNG seed for reproducible support minibatch indices.
        lr_decay_every: Halve the learning rate at this step interval.
        lr_floor: Minimum learning-rate multiplier relative to ``lr``.
        checkpoint_every: Diagnostic cadence; the terminal step is always saved.
        plateau_tolerance: Relative tolerance for the support-objective audit.

    Returns CPU tensors. ``support_loss`` and ``query_loss`` are the per-child
    normalized MSE used by ``followup_batched_eval.losses``. The optimizer
    minimizes ``support_loss + l2_penalty``. Query metrics are diagnostics only.
    """
    device = torch.device(device)
    masks = torch.as_tensor(masks, dtype=torch.float32, device=device)
    # Keep large support/query tensors on their supplied device and stream
    # chunks. This also lets a GPU runner retain its source bank on host RAM.
    x_support = torch.as_tensor(x_support, dtype=torch.float32)
    y_support = torch.as_tensor(y_support, dtype=torch.float32)
    condition_seeds = [int(seed) for seed in condition_seeds]
    replicas = [int(replica) for replica in replicas]

    if masks.ndim != 3 or masks.shape[1] != 784 or masks.shape[0] < 1 or masks.shape[2] < 1:
        raise ValueError("masks must have shape [M, 784, H] with positive M and H")
    if x_support.ndim != 4 or x_support.shape[2:] != (5, 784):
        raise ValueError("x_support must have shape [C, N, 5, 784]")
    conditions, support_count = x_support.shape[:2]
    if support_count < 1 or y_support.shape != (conditions, support_count):
        raise ValueError("y_support must have shape [C, N] matching x_support")
    if x_query.ndim != 4 or x_query.shape[0] != conditions or x_query.shape[2:] != (5, 784):
        raise ValueError("x_query must have shape [C, Q, 5, 784]")
    query_count = x_query.shape[1]
    if query_count < 1 or y_query.shape != (conditions, query_count):
        raise ValueError("y_query must have shape [C, Q] matching x_query")
    if len(condition_seeds) != conditions or len(replicas) != masks.shape[0]:
        raise ValueError("condition_seeds must have length C and replicas length M")
    if min(steps, chunk_size, checkpoint_every, reference_models, lr_decay_every) < 1:
        raise ValueError("steps, chunk_size, checkpoint cadence, reference count, and lr decay must be positive")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive or None")
    if lr <= 0 or l2 < 0 or plateau_tolerance <= 0 or not 0 < lr_floor <= 1:
        raise ValueError("lr, plateau_tolerance, and lr_floor must be positive; l2 must be nonnegative")
    if any(replica < 0 or replica >= reference_models for replica in replicas):
        raise ValueError("replica IDs must index the reference initialization bank")

    # Query tensors remain where supplied and are transferred a chunk at a time.
    x_query = torch.as_tensor(x_query)
    y_query = torch.as_tensor(y_query)
    model = BatchedConditions(
        masks,
        condition_seeds,
        replicas,
        reference_models=reference_models,
        kernel_mode="bmm",
    ).to(device)
    _exact_replica_initialization(model, masks, condition_seeds, replicas, reference_models)
    if initial_state is not None:
        model.load_state_dict(initial_state)
        if not torch.equal(model.masks, masks[None].expand_as(model.masks)):
            raise ValueError('resume masks differ')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    callback_masks = model.masks.detach().cpu().clone() if checkpoint_callback is not None else None

    step_history: list[int] = []
    support_history: list[Tensor] = []
    objective_history: list[Tensor] = []
    query_history: list[Tensor] = []
    penalty_history: list[Tensor] = []
    plateau_history: list[Tensor] = []
    minibatch_generators = [
        torch.Generator(device="cpu").manual_seed(int(seed) + 1009 * (condition + 1))
        for condition in range(conditions)
    ]
    learning_rate_history: list[float] = []
    full_support_slices = tuple(
        slice(start, min(support_count, start + chunk_size))
        for start in range(0, support_count, chunk_size)
    )
    parameters = tuple(model.parameters())
    # Retain every gradient check on CUDA, but read the cumulative flag only
    # at bounded intervals. No callback or result can contain a failed fit.
    gradients_finite = (torch.ones((), dtype=torch.bool, device=device)
                        if device.type == "cuda" else None)

    @torch.no_grad()
    def record(step: int) -> None:
        support_nmse = _streamed_losses(
            model, x_support, y_support, 5, chunk_size, device
        )
        penalty = _l2_penalty(model, l2)
        objective = support_nmse + penalty
        query_nmse = _streamed_losses(
            model, x_query, y_query, 5, chunk_size, device
        )
        prospective_objectives = objective_history + [objective.detach().cpu().clone()]
        prospective_steps = step_history + [step]
        plateau = _plateau_flags(
            prospective_objectives,
            prospective_steps,
            checkpoint_every,
            tolerance=plateau_tolerance,
        )
        step_history.append(step)
        support_history.append(support_nmse.detach().cpu().clone())
        objective_history.append(objective.detach().cpu().clone())
        query_history.append(query_nmse.detach().cpu().clone())
        penalty_history.append(penalty.detach().cpu().clone())
        plateau_history.append(plateau.detach().cpu().clone())
        learning_rate_history.append(float(optimizer.param_groups[0]["lr"]))
        if checkpoint_callback is not None:
            callback_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.named_parameters()
            }
            callback_state["masks"] = callback_masks
            checkpoint_callback({
                "step": step,
                "support_loss": support_history[-1],
                "support_objective": objective_history[-1],
                "trainNMSE": support_history[-1],
                "queryNMSE": query_history[-1],
                "l2_penalty": penalty_history[-1],
                "plateau_flags": plateau_history[-1],
                "lr": learning_rate_history[-1],
                "state_dict": callback_state,
                "optimizer_state": optimizer.state_dict(),
            })

    # Step zero is a diagnostic baseline; it is not a candidate checkpoint.
    record(start_step)
    for step in range(start_step + 1, start_step + steps + 1):
        current_lr = lr * max(lr_floor, 0.5 ** ((step - 1) // lr_decay_every))
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.zero_grad(set_to_none=True)
        if support_sampler is not None:
            sampled_x, sampled_y = support_sampler(step)
            sampled_x = sampled_x.to(device); sampled_y = sampled_y.to(device)
            if sampled_x.shape[:1] != (conditions,) or sampled_x.shape[2:] != (5, 784):
                raise ValueError('support sampler must return [C,B,5,784] and [C,B]')
            if sampled_y.shape != sampled_x.shape[:2]:
                raise ValueError('support sampler labels must match batch dimensions')
            train_count = sampled_x.shape[1]
            chunk_indices = [None]
        elif batch_size is None or batch_size >= support_count:
            chunk_indices = full_support_slices
            train_count = support_count
        else:
            train_count = batch_size
            condition_indices = torch.stack([
                torch.randint(support_count, (train_count,), generator=generator)
                for generator in minibatch_generators
            ])
            chunk_indices = [condition_indices]
        for indices_cpu in chunk_indices:
            if support_sampler is not None:
                x_chunk, y_chunk = sampled_x, sampled_y
            elif batch_size is not None and batch_size < support_count:
                indices = indices_cpu.to(x_support.device)
                x_chunk = torch.stack([
                    x_support[condition].index_select(0, indices[condition])
                    for condition in range(conditions)
                ]).to(device=device, dtype=torch.float32)
                y_chunk = torch.stack([
                    y_support[condition].index_select(0, indices[condition])
                    for condition in range(conditions)
                ]).to(device=device, dtype=torch.float32)
            else:
                x_chunk = x_support[:, indices_cpu].to(device=device, dtype=torch.float32)
                y_chunk = y_support[:, indices_cpu].to(device=device, dtype=torch.float32)
            prediction = model(x_chunk)
            residual = prediction - y_chunk[:, None, :]
            per_model_chunk_loss = residual.square().sum(dim=-1) / (train_count * 5 * reference_models)
            per_model_chunk_loss.sum().backward()
        (_l2_penalty(model, l2).sum() / reference_models).backward()
        for parameter in parameters:
            if parameter.grad is not None:
                finite = torch.isfinite(parameter.grad).all()
                if gradients_finite is None:
                    if not finite:
                        raise RuntimeError("nonfinite fresh-child gradient")
                else:
                    gradients_finite.logical_and_(finite)
        is_checkpoint = step % checkpoint_every == 0 or step == start_step + steps
        if gradients_finite is not None and (step % 64 == 0 or is_checkpoint):
            if not gradients_finite:
                raise RuntimeError("nonfinite fresh-child gradient")
        optimizer.step()
        if is_checkpoint:
            record(step)

    with torch.no_grad():
        final_support = _streamed_losses(model, x_support, y_support, 5, chunk_size, device)
        final_penalty = _l2_penalty(model, l2)
        final_query = _streamed_losses(model, x_query, y_query, 5, chunk_size, device)
        final_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
    history = {
        "steps": torch.tensor(step_history, dtype=torch.long),
        "support_loss": torch.stack(support_history),
        "support_objective": torch.stack(objective_history),
        "trainNMSE": torch.stack(support_history),
        "queryNMSE": torch.stack(query_history),
        "l2_penalty": torch.stack(penalty_history),
        "plateau_flags": torch.stack(plateau_history),
        "learning_rate": torch.tensor(learning_rate_history, dtype=torch.float32),
    }
    return {
        "state_dict": final_state,
        "support_loss": final_support.detach().cpu(),
        "query_loss": final_query.detach().cpu(),
        "support_objective": (final_support + final_penalty).detach().cpu(),
        "l2_penalty": final_penalty.detach().cpu(),
        "plateau_flags": history["plateau_flags"][-1].clone(),
        "history": history,
        "steps_run": steps,
        "fixed_horizon": True,
        "stopping_source": "none; fixed step horizon",
        "plateau_source": "support objective only; diagnostic and non-stopping",
        "l2_definition": "0.5*l2*(sum((weight*masks)^2)+sum(bias^2)+sum(readout^2)+sum(offset^2)) per child",
        "gradient_normalizer": reference_models,
        "chunk_size": chunk_size,
        "batch_size": batch_size,
        "minibatch_seed": int(seed) if batch_size is not None else None,
        "minibatch_condition_seeds": [int(seed) + 1009 * (condition + 1)
                                       for condition in range(conditions)] if batch_size is not None else None,
        "base_lr": lr,
        "lr_decay_every": lr_decay_every,
        "lr_floor": lr_floor,
        "support_sampling": "independent fresh source sets per update; fixed support monitor" if support_sampler is not None else "fixed support dataset",
        "optimizer_state": optimizer.state_dict(),
        "terminal_step": start_step+steps,
    }
