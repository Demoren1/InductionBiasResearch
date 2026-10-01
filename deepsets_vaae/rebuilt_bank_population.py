"""Exact iid-five source-population fitting for rebuilt DeepSets masks.

For a per-image prediction error ``e = g(x) - c[digit]``, the expected
five-image set NMSE is ``E[e**2] + 4 * E[e]**2``. This module optimizes that
population objective over the complete source image pool, without sampling
sets or consulting query labels for optimization or stopping.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from .core import MaskedDeepSets


def iid5_population_nmse(residual: Tensor) -> Tensor:
    """Compute the exact expected set NMSE for iid sets of five.

    ``residual`` has shape ``[N]`` or ``[M,N]`` and contains the per-image
    error ``g(x)-c[digit]``. The returned scalar/vector is
    ``mean(e**2) + 4*mean(e)**2``.
    """
    if residual.ndim not in (1, 2) or residual.shape[-1] < 1:
        raise ValueError("residual must have shape [N] or [M,N] with N positive")
    return residual.square().mean(dim=-1) + 4.0 * residual.mean(dim=-1).square()


def _packed_per_image_predictions(model: MaskedDeepSets, images: Tensor) -> Tensor:
    """Predict each image for every child using one ``N×(F·M·H)`` GEMM."""
    if images.ndim != 2 or images.shape[1] != 784:
        raise ValueError("images must have shape [N,784]")
    models, features, hidden_dim = model.weight.shape
    effective = model.weight * model.masks
    packed_weight = effective.permute(1, 0, 2).reshape(features, models * hidden_dim)
    preactivation = (images @ packed_weight).reshape(-1, models, hidden_dim)
    hidden = torch.tanh(preactivation + model.bias[None])
    per_image = (hidden * model.readout[None]).sum(dim=-1) + model.per_image_offset[None]
    return per_image.transpose(0, 1)


def _population_loss(model: MaskedDeepSets, images: Tensor, image_targets: Tensor) -> Tensor:
    residual = _packed_per_image_predictions(model, images) - image_targets[None]
    return iid5_population_nmse(residual)


def _query_nmse(
    model: MaskedDeepSets,
    qx: Tensor,
    qy: Tensor,
    *,
    chunk_size: int = 128,
) -> Tensor:
    total: Tensor | None = None
    count = qx.shape[0]
    for start in range(0, count, chunk_size):
        stop = min(count, start + chunk_size)
        prediction = model(qx[start:stop])
        squared = (prediction - qy[None, start:stop]).square().sum(dim=-1)
        total = squared if total is None else total + squared
    assert total is not None
    return total / (count * 5.0)


def _l2_penalty(model: MaskedDeepSets, l2: float) -> Tensor:
    effective = model.weight * model.masks
    return 0.5 * l2 * (
        effective.square().sum(dim=(1, 2))
        + model.bias.square().sum(dim=1)
        + model.readout.square().sum(dim=1)
        + model.per_image_offset.square()
    )


def _cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _plateau_flags(
    objectives: list[Tensor],
    steps: list[int],
    current_step: int,
    *,
    tolerance: float,
) -> Tensor:
    """Audit objective drift over 100 steps and range over the latest 50."""
    current = objectives[-1]
    eligible = [i for i, step in enumerate(steps) if step <= current_step - 100]
    recent = [i for i, step in enumerate(steps) if step >= current_step - 50]
    if not eligible or len(recent) < 2:
        return torch.zeros_like(current, dtype=torch.bool)
    before = objectives[eligible[-1]]
    scale100 = torch.maximum(before.abs(), current.abs()).clamp_min(1e-8)
    stable100 = (current - before).abs() / scale100 <= tolerance
    values = torch.stack([objectives[i] for i in recent])
    scale50 = values.mean(dim=0).abs().clamp_min(1e-8)
    stable50 = (values.amax(dim=0) - values.amin(dim=0)) / scale50 <= tolerance
    return stable100 & stable50


def _reference_initialization(
    model: MaskedDeepSets,
    masks: Tensor,
    init_seed: int,
    replicas: Sequence[int],
    reference_models: int,
) -> None:
    features, hidden = masks.shape[1:]
    reference = MaskedDeepSets(
        torch.ones(reference_models, features, hidden, device=masks.device),
        seed=init_seed,
        initialization_reference_models=reference_models,
    )
    indices = torch.as_tensor(replicas, dtype=torch.long, device=masks.device)
    with torch.no_grad():
        model.weight.copy_(reference.weight[indices])
        model.readout.copy_(reference.readout[indices])


def fit_population(
    masks: Tensor,
    images: Tensor,
    image_targets: Tensor,
    qx: Tensor,
    qy: Tensor,
    init_seed: int,
    replicas: Sequence[int],
    lr: float = 0.002,
    l2: float = 0.001,
    cap: int = 4000,
    minimum: int = 1000,
    device: str | torch.device = "cpu",
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
    resume: Mapping[str, Any] | None = None,
    *,
    reference_models: int = 20,
    lr_decay_every: int = 500,
    lr_floor: float = 1.0 / 64.0,
    checkpoint_every: int = 50,
    plateau_tolerance: float = 0.01,
    plateau_patience: int = 3,
    query_chunk_size: int = 128,
) -> dict[str, Any]:
    """Fit candidate masks to the exact source population over iid 5-sets.

    Training images and per-image targets are consumed by one packed matrix
    multiply per update. The query pool is evaluated at checkpoint cadence and
    at termination; its values never affect gradients, learning rate, or
    stopping. Candidates freeze individually after ``plateau_patience``
    consecutive support-objective plateau checks after ``minimum`` updates.

    L2 is ``0.5*l2*(||W*M||²+||b||²+||a||²+o²)`` per child. Returned tensors
    and optimizer state are copied to CPU. ``cap`` is the maximum absolute
    update count, which also makes callback checkpoints directly resumable.
    """
    device = torch.device(device)
    masks = torch.as_tensor(masks, dtype=torch.float32, device=device)
    images = torch.as_tensor(images, dtype=torch.float32, device=device)
    image_targets = torch.as_tensor(image_targets, dtype=torch.float32, device=device)
    qx = torch.as_tensor(qx, dtype=torch.float32, device=device)
    qy = torch.as_tensor(qy, dtype=torch.float32, device=device)
    replicas = [int(replica) for replica in replicas]
    if masks.ndim != 3 or masks.shape[0] < 1 or masks.shape[1] != 784 or masks.shape[2] < 1:
        raise ValueError("masks must have shape [M,784,H] with positive M and H")
    models = masks.shape[0]
    if images.ndim != 2 or images.shape[0] < 1 or images.shape[1] != 784:
        raise ValueError("images must have shape [N,784] with N positive")
    if image_targets.shape != (images.shape[0],):
        raise ValueError("image_targets must have shape [N]")
    if qx.ndim != 3 or qx.shape[0] < 1 or qx.shape[1:] != (5, 784):
        raise ValueError("qx must have shape [Q,5,784] with Q positive")
    if qy.shape != (qx.shape[0],):
        raise ValueError("qy must have shape [Q]")
    if len(replicas) != models or any(r < 0 or r >= reference_models for r in replicas):
        raise ValueError("replicas must contain M valid reference-bank indices")
    if min(cap, minimum, reference_models, lr_decay_every, checkpoint_every,
           plateau_patience, query_chunk_size) < 1 or minimum > cap:
        raise ValueError("cap/minimum and cadence parameters must be positive with minimum <= cap")
    if lr <= 0 or l2 < 0 or not 0 < lr_floor <= 1 or plateau_tolerance <= 0:
        raise ValueError("lr and plateau_tolerance must be positive; l2 nonnegative")

    model = MaskedDeepSets(
        masks, seed=int(init_seed), initialization_reference_models=reference_models
    ).to(device)
    _reference_initialization(model, masks, int(init_seed), replicas, reference_models)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    active = torch.ones(models, dtype=torch.bool, device=device)
    stable_counts = torch.zeros(models, dtype=torch.long, device=device)
    stopping_steps = torch.full((models,), -1, dtype=torch.long, device=device)
    start_step = 0
    step_history: list[int] = []
    population_history: list[Tensor] = []
    objective_history: list[Tensor] = []
    query_history: list[Tensor] = []
    l2_history: list[Tensor] = []
    plateau_history: list[Tensor] = []
    lr_history: list[float] = []

    def load_history(history: Mapping[str, Any]) -> None:
        step_history.extend(int(x) for x in torch.as_tensor(history.get("steps", [])).tolist())
        population_history.extend(torch.as_tensor(history.get("population_loss", history.get("trainNMSE", []))).unbind(0))
        objective_history.extend(torch.as_tensor(history.get("population_objective", history.get("support_objective", []))).unbind(0))
        query_history.extend(torch.as_tensor(history.get("queryNMSE", [])).unbind(0))
        l2_history.extend(torch.as_tensor(history.get("l2_penalty", torch.zeros_like(torch.stack(objective_history)) if objective_history else [])).unbind(0))
        plateau_history.extend(torch.as_tensor(history.get("plateau_flags", torch.zeros_like(torch.stack(objective_history), dtype=torch.bool) if objective_history else [])).unbind(0))
        lr_history.extend(float(x) for x in torch.as_tensor(history.get("learning_rate", [])).tolist())

    if resume is not None:
        if "state_dict" not in resume or "optimizer_state" not in resume:
            raise ValueError("resume must contain state_dict and optimizer_state")
        model.load_state_dict(resume["state_dict"])
        if not torch.equal(model.masks, masks):
            raise ValueError("resume masks differ from requested masks")
        optimizer.load_state_dict(resume["optimizer_state"])
        start_step = int(resume.get("step", resume.get("terminal_step", 0)))
        if "active" in resume:
            active.copy_(torch.as_tensor(resume["active"], dtype=torch.bool, device=device))
        if "plateau_stable_counts" in resume:
            stable_counts.copy_(torch.as_tensor(resume["plateau_stable_counts"], dtype=torch.long, device=device))
        if "stopping_steps" in resume:
            stopping_steps.copy_(torch.as_tensor(resume["stopping_steps"], dtype=torch.long, device=device))
        if "history" in resume:
            load_history(resume["history"])

    @torch.no_grad()
    def record(step: int) -> Tensor:
        population = _population_loss(model, images, image_targets)
        penalty = _l2_penalty(model, l2)
        objective = population + penalty
        flags = _plateau_flags(
            objective_history + [objective.detach().cpu()],
            step_history + [step], step, tolerance=plateau_tolerance,
        )
        if step >= minimum:
            stable_counts.copy_(torch.where(active & flags.to(device), stable_counts + 1,
                                            torch.where(active, torch.zeros_like(stable_counts), stable_counts)))
            newly_stopped = active & (stable_counts >= plateau_patience)
            if newly_stopped.any():
                active[newly_stopped] = False
                stopping_steps[newly_stopped] = step
                # Clear moments so zeroed gradients cannot move a frozen child.
                for parameter in model.parameters():
                    state = optimizer.state.get(parameter, {})
                    for moment_name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                        moment = state.get(moment_name)
                        if moment is not None and moment.ndim > 0 and moment.shape[0] == models:
                            moment[newly_stopped] = 0
        step_history.append(step)
        population_history.append(population.detach().cpu().clone())
        objective_history.append(objective.detach().cpu().clone())
        query_value = _query_nmse(model, qx, qy, chunk_size=query_chunk_size)
        query_history.append(query_value.detach().cpu().clone())
        l2_history.append(penalty.detach().cpu().clone())
        plateau_history.append(flags.detach().cpu().clone())
        learning_rate = float(optimizer.param_groups[0]["lr"])
        lr_history.append(learning_rate)
        if checkpoint_callback is not None:
            history_snapshot = {
                "steps": torch.tensor(step_history, dtype=torch.long),
                "population_loss": torch.stack(population_history),
                "population_objective": torch.stack(objective_history),
                "objective": torch.stack(objective_history),
                "support_objective": torch.stack(objective_history),
                "trainNMSE": torch.stack(population_history),
                "queryNMSE": torch.stack(query_history),
                "l2_penalty": torch.stack(l2_history),
                "plateau_flags": torch.stack(plateau_history),
                "learning_rate": torch.tensor(lr_history, dtype=torch.float32),
            }
            checkpoint_callback({
                "step": step,
                "population_loss": population_history[-1].clone(),
                "population_objective": objective_history[-1].clone(),
                "objective": objective_history[-1].clone(),
                "support_objective": objective_history[-1].clone(),
                "trainNMSE": population_history[-1].clone(),
                "queryNMSE": query_history[-1].clone(),
                "l2_penalty": l2_history[-1].clone(),
                "plateau_flags": plateau_history[-1].clone(),
                "active": active.detach().cpu().clone(),
                "plateau_stable_counts": stable_counts.detach().cpu().clone(),
                "stopping_steps": stopping_steps.detach().cpu().clone(),
                "learning_rate": learning_rate,
                "state_dict": _cpu_tree(model.state_dict()),
                "optimizer_state": _cpu_tree(optimizer.state_dict()),
                "history": history_snapshot,
            })
        return flags

    if not step_history:
        record(start_step)
    elif step_history[-1] != start_step:
        record(start_step)

    last_flags = (plateau_history[-1].to(device) if plateau_history
                  else torch.zeros(models, dtype=torch.bool, device=device))
    for step in range(start_step + 1, cap + 1):
        if not active.any():
            break
        learning_rate = lr * max(lr_floor, 0.5 ** ((step - 1) // lr_decay_every))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        population = _population_loss(model, images, image_targets)
        objective = population + _l2_penalty(model, l2)
        (objective * active.to(objective.dtype)).sum().backward()
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            view = (models,) + (1,) * (parameter.ndim - 1)
            parameter.grad.mul_(active.reshape(view))
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError("nonfinite population-fit gradient")
        optimizer.step()
        if step % checkpoint_every == 0 or step == cap:
            last_flags = record(step)
            if not active.any():
                break

    terminal_step = step_history[-1] if step_history else start_step
    final_population = _population_loss(model, images, image_targets)
    final_penalty = _l2_penalty(model, l2)
    final_query = _query_nmse(model, qx, qy, chunk_size=query_chunk_size)
    capped = active.clone()
    final_stopping_steps = stopping_steps.clone()
    final_stopping_steps[capped & (final_stopping_steps < 0)] = terminal_step
    history_out = {
        "steps": torch.tensor(step_history, dtype=torch.long),
        "population_loss": torch.stack(population_history),
        "population_objective": torch.stack(objective_history),
        "objective": torch.stack(objective_history),
        "support_objective": torch.stack(objective_history),
        "trainNMSE": torch.stack(population_history),
        "queryNMSE": torch.stack(query_history),
        "l2_penalty": torch.stack(l2_history),
        "plateau_flags": torch.stack(plateau_history),
        "learning_rate": torch.tensor(lr_history, dtype=torch.float32),
    }
    return {
        "state_dict": _cpu_tree(model.state_dict()),
        "optimizer_state": _cpu_tree(optimizer.state_dict()),
        "population_loss": final_population.detach().cpu(),
        "population_objective": (final_population + final_penalty).detach().cpu(),
        "objective": (final_population + final_penalty).detach().cpu(),
        "support_objective": (final_population + final_penalty).detach().cpu(),
        "trainNMSE": final_population.detach().cpu(),
        "query_loss": final_query.detach().cpu(),
        "queryNMSE": final_query.detach().cpu(),
        "l2_penalty": final_penalty.detach().cpu(),
        "history": history_out,
        "plateau_flags": last_flags.detach().cpu(),
        "plateau_stopped": (~active).detach().cpu(),
        "capped": capped.detach().cpu(),
        "stopping_steps": final_stopping_steps.detach().cpu(),
        "terminal_step": terminal_step,
        "steps_run": terminal_step - start_step,
        "active": active.detach().cpu(),
        "plateau_stable_counts": stable_counts.detach().cpu(),
        "fixed_source_population": True,
        "iid_set_size": 5,
        "stopping_source": "support population objective only; query diagnostic only",
        "population_loss_definition": "mean(e^2)+4*mean(e)^2 for per-image e=g(x)-c[digit]; equals exact E[(sum_5 e)^2/5] for iid draws",
        "l2_definition": "0.5*l2*(sum((weight*masks)^2)+sum(bias^2)+sum(readout^2)+sum(offset^2)) per child",
        "base_lr": lr,
        "lr_decay_every": lr_decay_every,
        "lr_floor": lr_floor,
        "plateau_tolerance": plateau_tolerance,
        "plateau_patience": plateau_patience,
        "minimum_updates": minimum,
        "cap": cap,
        "reference_models": reference_models,
        "replicas": replicas,
    }
