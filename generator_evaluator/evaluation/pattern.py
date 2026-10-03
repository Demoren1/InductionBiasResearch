"""Vectorised, protocol-faithful fitting for the small pattern children.

``PatternFitEngine`` is the one implementation of the fixed-horizon child
solver.  It keeps a leading ``run`` axis for independent masks and replicas:
parameters, Adam moments, initialisation seeds, and minibatch streams remain
independent, while the expensive forward/backward work is one CUDA operation.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from generator_evaluator.data.types import InnerProtocol, TaskData
from generator_evaluator.storage.progress import progress


def validate_pattern_fits(masks: Tensor, tasks: Sequence[TaskData], protocol: InnerProtocol) -> None:
    """Validate a homogeneous batch without changing its public semantics."""
    if protocol.metric != "bce":
        raise ValueError("the binary pattern adapter requires protocol.metric='bce'")
    if masks.ndim != 3 or masks.shape[1:] != (11, 8) or masks.shape[0] != len(tasks):
        raise ValueError("masks must be [count, 11, 8] and match the task count")
    if not torch.isfinite(masks).all() or not ((masks == 0) | (masks == 1)).all():
        raise ValueError("pattern masks must be finite and binary")
    if not tasks:
        raise ValueError("at least one mask/task measurement is required")
    support_shape, query_shape = tasks[0].x_support.shape, tasks[0].x_query.shape
    if support_shape[-1] != 11 or query_shape[-1] != 11:
        raise ValueError("pattern task inputs must have 11 features")
    for task in tasks:
        if task.x_support.ndim != 2 or task.x_query.ndim != 2:
            raise ValueError("batched pattern fitting requires rank-two inputs")
        if task.x_support.shape != support_shape or task.x_query.shape != query_shape:
            raise ValueError("all batched pattern tasks must have equal support/query shapes")
        if task.y_support.shape != (support_shape[0],) or task.y_query.shape != (query_shape[0],):
            raise ValueError("pattern task labels must agree with their inputs")


def _per_run_bce(logits: Tensor, labels: Tensor) -> Tensor:
    return F.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(dim=1)


def _plateau(history: list[dict[str, Any]], protocol: InnerProtocol) -> bool:
    terminal_step = history[-1]["step"]
    prior = next((row for row in reversed(history) if row["step"] <= terminal_step - 100), None)
    trailing = [row["support_objective"] for row in history if row["step"] >= terminal_step - 50]
    if prior is None or len(trailing) < 2:
        return False
    current, earlier = history[-1]["support_objective"], prior["support_objective"]
    stable100 = abs(current - earlier) <= protocol.plateau_tolerance * max(abs(current), abs(earlier), 1e-8)
    stable50 = max(trailing) - min(trailing) <= protocol.plateau_tolerance * max(abs(sum(trailing) / len(trailing)), 1e-8)
    return bool(stable100 and stable50)


def _split_optimizer_states(optimizer: torch.optim.Optimizer, count: int) -> list[dict[str, Any]]:
    """Split packed Adam state after a single device-to-CPU copy per tensor."""
    packed = optimizer.state_dict()
    cpu_state: dict[Any, dict[str, Any]] = {}
    for parameter_id, state in packed["state"].items():
        cpu_state[parameter_id] = {
            key: value.detach().cpu() if torch.is_tensor(value) else deepcopy(value)
            for key, value in state.items()
        }
    results = [{"state": {}, "param_groups": deepcopy(packed["param_groups"])} for _ in range(count)]
    for run, result in enumerate(results):
        for parameter_id, state in cpu_state.items():
            result["state"][parameter_id] = {
                key: (value[run].clone() if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == count
                      else value.clone() if torch.is_tensor(value) else deepcopy(value))
                for key, value in state.items()
            }
    return results


class PatternFitEngine:
    """Fit independent ReLU pattern children in one packed Adam trajectory."""

    def __init__(self, protocol: InnerProtocol, device: str | torch.device = "cpu") -> None:
        self.protocol = protocol
        self.device = torch.device(device)

    def fit_one(self, mask: Tensor, task: TaskData) -> dict[str, Any]:
        """Scalar compatibility entry point used by ``adapters._fit_pattern``."""
        return self.fit(torch.as_tensor(mask).unsqueeze(0), [task])[0]

    def fit(self, masks: Tensor, tasks: Sequence[TaskData], *,
            initialization_seeds: Sequence[int] | None = None) -> list[dict[str, Any]]:
        masks_cpu = torch.as_tensor(masks, dtype=torch.float32).detach().cpu().contiguous()
        validate_pattern_fits(masks_cpu, tasks, self.protocol)
        from pattern.task_quality.meta import _init_children, child_logits_batch

        n_masks, replicas = len(tasks), self.protocol.replicas
        base_seeds = ([self.protocol.seed] * n_masks if initialization_seeds is None
                      else list(initialization_seeds))
        if len(base_seeds) != n_masks or any(type(seed) is not int for seed in base_seeds):
            raise ValueError("initialization_seeds must contain one integer per mask")
        count = n_masks * replicas
        run_masks = masks_cpu[:, None].expand(-1, replicas, -1, -1).reshape(count, 11, 8).to(self.device)
        x_support = torch.stack([task.x_support.detach().float() for task in tasks]).to(self.device)
        y_support = torch.stack([task.y_support.detach().float() for task in tasks]).to(self.device)
        x_query = torch.stack([task.x_query.detach().float() for task in tasks]).to(self.device)
        y_query = torch.stack([task.y_query.detach().float() for task in tasks]).to(self.device)
        run_x_support = x_support[:, None].expand(-1, replicas, -1, -1).reshape(count, *x_support.shape[1:])
        run_y_support = y_support[:, None].expand(-1, replicas, -1).reshape(count, y_support.shape[1])
        run_x_query = x_query[:, None].expand(-1, replicas, -1, -1).reshape(count, *x_query.shape[1:])
        run_y_query = y_query[:, None].expand(-1, replicas, -1).reshape(count, y_query.shape[1])
        # Repeat the replica schedule for every mask; it exactly matches scalar fitting.
        seeds = [base + 10_007 * replica for base in base_seeds for replica in range(replicas)]
        params = _init_children(seeds, self.device)
        optimizer = torch.optim.Adam(params.values(), lr=self.protocol.lr)
        histories: list[list[dict[str, Any]]] = [[] for _ in range(count)]
        zero_penalty = torch.zeros(count, device=self.device)
        minibatch_rngs = [torch.Generator(device="cpu").manual_seed(
            base_seeds[run // replicas] + 73_003 * ((run % replicas) + 1)) for run in range(count)]

        def penalty_per_run() -> Tensor:
            if self.protocol.l2 == 0:
                return zero_penalty
            penalty = torch.zeros(count, device=self.device)
            for name, value in params.items():
                component = value * run_masks if name == "w" else value
                penalty.add_(component.reshape(count, -1).square().sum(dim=1))
            return penalty.mul(.5 * self.protocol.l2)

        @torch.no_grad()
        def record(step: int) -> tuple[Tensor, Tensor]:
            support = _per_run_bce(child_logits_batch(run_x_support, run_masks, params), run_y_support)
            query = _per_run_bce(child_logits_batch(run_x_query, run_masks, params), run_y_query)
            objective = support + penalty_per_run()
            support_cpu, objective_cpu, query_cpu = (item.detach().cpu() for item in (support, objective, query))
            for run in range(count):
                histories[run].append({"step": step, "support_bce": float(support_cpu[run]),
                                       "support_objective": float(objective_cpu[run]), "query_bce": float(query_cpu[run])})
            return objective, query

        record(0)
        steps = progress(range(1, self.protocol.steps + 1),
                         desc=f"Pattern batch ({n_masks} masks × {replicas} init)",
                         position=1, leave=False, unit="step")
        for step in steps:
            for group in optimizer.param_groups:
                group["lr"] = self.protocol.lr * max(
                    self.protocol.lr_floor, .5 ** ((step - 1) // self.protocol.lr_decay_every))
            if self.protocol.batch_size is None or self.protocol.batch_size >= run_x_support.shape[1]:
                batch_x, batch_y = run_x_support, run_y_support
            else:
                # The CPU generators follow the scalar rule exactly, then only indices move to CUDA.
                indices = torch.stack([torch.randint(run_x_support.shape[1], (self.protocol.batch_size,), generator=rng)
                                       for rng in minibatch_rngs]).to(self.device)
                batch_x = run_x_support.gather(1, indices[..., None].expand(-1, -1, run_x_support.shape[-1]))
                batch_y = run_y_support.gather(1, indices)
            support = _per_run_bce(child_logits_batch(batch_x, run_masks, params), batch_y)
            objective = support.sum() + penalty_per_run().sum()
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
            if step % self.protocol.checkpoint_every == 0 or step == self.protocol.steps:
                support_objective, query = record(step)
                steps.set_postfix(support=f"{support_objective.mean():.4f}", query=f"{query.mean():.4f}", refresh=False)

        with torch.no_grad():
            terminal = _per_run_bce(child_logits_batch(run_x_query, run_masks, params), run_y_query).detach().cpu()
        states = [{name: value[run].detach().cpu().clone() for name, value in params.items()} for run in range(count)]
        optimizer_states = _split_optimizer_states(optimizer, count)
        result: list[dict[str, Any]] = []
        for index in range(n_masks):
            runs = [index * replicas + replica for replica in range(replicas)]
            child_states = [states[run] for run in runs]
            result.append({
                "replica_losses": [float(terminal[run]) for run in runs],
                "seeds": [seeds[run] for run in runs],
                "plateau_flags": [_plateau(histories[run], self.protocol) for run in runs],
                "state_dict": child_states,
                "optimizer_state": [optimizer_states[run] for run in runs],
                "history": [histories[run] for run in runs],
                "effective_weights": [state["w"] * masks_cpu[index] for state in child_states],
            })
        return result


def fit_pattern_batch(masks: Tensor, tasks: Sequence[TaskData], protocol: InnerProtocol,
                      device: str = "cpu", *,
                      initialization_seeds: Sequence[int] | None = None) -> list[dict[str, Any]]:
    """Fit homogeneous full-batch jobs, retaining the historical result schema."""
    clean = torch.as_tensor(masks, dtype=torch.float32).detach().cpu().contiguous()
    validate_pattern_fits(clean, tasks, protocol)
    if protocol.batch_size is not None and protocol.batch_size < tasks[0].x_support.shape[0]:
        raise ValueError("batched pattern fitting supports full-batch protocol only")
    results = PatternFitEngine(protocol, device).fit(
        clean, tasks, initialization_seeds=initialization_seeds)
    seeds = ([protocol.seed] * len(tasks) if initialization_seeds is None
             else list(map(int, initialization_seeds)))
    for result, task, seed in zip(results, tasks, seeds):
        result.update(label_source="fresh_terminal_query", fixed_horizon=True,
                      protocol_id=protocol.fingerprint, task_id=task.task_id,
                      actual_initialization_seed=seed,
                      solver_protocol_seed=protocol.seed, minibatch_seed_base=None)
    return results
