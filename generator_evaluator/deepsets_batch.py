"""Packed same-task DeepSets measurements.

The generic measurement store asks for one child result per ``(mask, task)``.
For a fixed task, however, all candidates use the same support and query
sets.  This module packs their independent replica fits along the model axis
of :func:`deepsets_vaae.utility_graph_child.fit_children`, then restores the
ordinary per-mask child-result schema before artifacts are written.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

import torch
from torch import Tensor

from .data import InnerProtocol, TaskData


def _slice_model_axis(value: Any, start: int, stop: int) -> Any:
    """Copy a packed ``[condition, model, ...]`` result slice to CPU."""
    if not torch.is_tensor(value):
        return deepcopy(value)
    value = value.detach().cpu()
    if value.ndim >= 2 and value.shape[0] == 1:
        return value[:, start:stop].clone()
    return value.clone()


def _slice_history(value: Any, start: int, stop: int) -> Any:
    """Copy one candidate from a ``[checkpoint, condition, model]`` trace."""
    if not torch.is_tensor(value):
        return deepcopy(value)
    value = value.detach().cpu()
    if value.ndim >= 3 and value.shape[1] == 1:
        return value[:, :, start:stop].clone()
    return value.clone()


def _slice_optimizer_state(state: dict[str, Any], start: int, stop: int) -> dict[str, Any]:
    """Keep Adam moments for one candidate's replica range."""
    result = {"state": {}, "param_groups": deepcopy(state["param_groups"])}
    for parameter_id, values in state["state"].items():
        result["state"][parameter_id] = {
            name: _slice_model_axis(value, start, stop)
            for name, value in values.items()
        }
    return result


def _validate(masks: Tensor, task: TaskData, protocol: InnerProtocol) -> Tensor:
    masks = torch.as_tensor(masks, dtype=torch.float32).detach().cpu().contiguous()
    if protocol.metric != "nmse":
        raise ValueError("the DeepSets batch adapter requires protocol.metric='nmse'")
    if masks.ndim != 3 or masks.shape[0] < 1 or masks.shape[1] != 784 or masks.shape[2] < 1:
        raise ValueError("masks must have shape [count, 784, hidden]")
    if not torch.isfinite(masks).all() or not bool(((masks == 0) | (masks == 1)).all()):
        raise ValueError("DeepSets masks must be finite and binary")
    if task.x_support.ndim != 3 or task.x_support.shape[1:] != (5, 784):
        raise ValueError("DeepSets task inputs must have shape [sets, 5, 784]")
    return masks


def fit_deepsets_batch(
    masks: Tensor,
    task: TaskData,
    protocol: InnerProtocol,
    device: str = "cpu",
    *, initialization_seeds: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Fit many masks on one task and return ordinary individual results.

    Each candidate keeps the same numbered initialization replicas as a
    scalar adapter call.  Support and query data are uploaded once when the
    target device is CUDA; the fixed-horizon solver then performs one full
    support forward per update rather than seven small 32-set chunks.
    """
    from deepsets_vaae.utility_graph_child import fit_children

    masks = _validate(masks, task, protocol)
    replicas = protocol.replicas
    count = len(masks)
    target = torch.device(device)
    # A single target task is only a few MB.  Keeping it resident avoids a
    # host-to-device copy for each optimizer step while retaining CPU outputs.
    x_support = task.x_support.detach().float().unsqueeze(0).to(target)
    y_support = task.y_support.detach().float().unsqueeze(0).to(target)
    x_query = task.x_query.detach().float().unsqueeze(0).to(target)
    y_query = task.y_query.detach().float().unsqueeze(0).to(target)
    packed_masks = masks[:, None].expand(-1, replicas, -1, -1).reshape(
        count * replicas, *masks.shape[1:]
    )
    seeds = [protocol.seed] * count if initialization_seeds is None else list(map(int, initialization_seeds))
    if len(seeds) != count:
        raise ValueError("one initialization seed is required per mask")
    initial_state = None
    if initialization_seeds is not None:
        if protocol.batch_size is not None and protocol.batch_size < len(task.x_support):
            raise ValueError("per-candidate initialization seeds require full-support fits")
        from deepsets_vaae.core import MaskedDeepSets
        reference_masks = torch.ones(max(20, replicas), *masks.shape[1:], device=target)
        rows = []
        for seed in seeds:
            reference = MaskedDeepSets(reference_masks, seed=seed,
                                      initialization_reference_models=max(20, replicas))
            rows.append({name: value.detach()[:replicas].clone()
                         for name, value in reference.named_parameters()})
        initial_state = {name: torch.cat([row[name] for row in rows])[None] for name in rows[0]}
        initial_state["masks"] = packed_masks.to(target)[None]
    fitted = fit_children(
        packed_masks,
        x_support,
        y_support,
        x_query,
        y_query,
        [protocol.seed],
        list(range(replicas)) * count,
        protocol.steps,
        protocol.lr,
        protocol.l2,
        target,
        reference_models=max(20, replicas),
        chunk_size=len(task.x_support),
        batch_size=protocol.batch_size,
        lr_decay_every=protocol.lr_decay_every,
        lr_floor=protocol.lr_floor,
        checkpoint_every=protocol.checkpoint_every,
        plateau_tolerance=protocol.plateau_tolerance,
        initial_state=initial_state,
    )
    state = fitted["state_dict"]
    results: list[dict[str, Any]] = []
    for index, mask in enumerate(masks):
        actual_seed = seeds[index]
        start, stop = index * replicas, (index + 1) * replicas
        child_state = {name: _slice_model_axis(value, start, stop) for name, value in state.items()}
        history = {
            name: _slice_history(value, start, stop)
            for name, value in fitted["history"].items()
        }
        weights = (child_state["weight"] * mask[None, None]).squeeze(0)
        results.append({
            "label_source": "fresh_terminal_query",
            "fixed_horizon": True,
            "protocol_id": protocol.fingerprint,
            "task_id": task.task_id,
            "replica_losses": fitted["query_loss"][0, start:stop].detach().cpu().tolist(),
            "seeds": [f"{actual_seed}:{replica}" for replica in range(replicas)],
            "initialization": {"base_seed": actual_seed,
                               "reference_replica_ids": list(range(replicas))},
            "actual_initialization_seed": actual_seed,
            "solver_protocol_seed": protocol.seed,
            "minibatch_seed_base": protocol.seed if protocol.batch_size is not None else None,
            "plateau_flags": fitted["plateau_flags"][0, start:stop].detach().cpu().tolist(),
            "state_dict": child_state,
            "optimizer_state": _slice_optimizer_state(fitted["optimizer_state"], start, stop),
            "history": history,
            "effective_weights": weights.detach().cpu(),
        })
    return results
