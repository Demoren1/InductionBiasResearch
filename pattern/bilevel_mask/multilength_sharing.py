"""Generated parameter sharing across several pattern lengths.

The sequence length and the maximum filter width are fixed.  A coordinate
generator produces a hard parameter-sharing tensor U.  Task-specific filter
values and readout weights are freshly adapted.  Two variants are supported:

* ``global``: one latent and one U for every pattern length;
* ``length_latent``: one optimized latent per length and a shared generator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from pattern.evaluation.decoder_agreement import align_columns
from .generated_sharing import (
    DenseWeights,
    SharedWeights,
    SharingGenerator,
    _DenseView,
    _dense_forward,
    _gather_restart,
    _network_losses,
    _project,
    _regularization,
    adapt_weights,
    assignments,
    sharing_forward,
)


def _seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


@dataclass(frozen=True)
class MultiLengthConfig:
    seed: int = 42
    seq_len: int = 32
    pattern_lengths: tuple[int, ...] = (3, 5, 7)
    hidden: int = 32
    filter_dim: int = 7
    latent_dim: int = 4
    generator_width: int = 16
    train_restarts: int = 8
    eval_restarts: int = 16
    tasks_per_length: int = 8
    outer_steps: int = 400
    temperature_anneal_steps: int = 160
    checkpoint_every: int = 10
    outer_patience: int = 15
    outer_min_delta: float = 1e-4
    checkpoint_task_chunk: int = 32
    inner_steps: int = 80
    final_refit_steps: int = 3000
    refit_validate_every: int = 20
    eval_task_chunk: int = 256
    inner_lr: float = 0.1
    final_refit_lr: float = 0.003
    latent_lr: float = 0.03
    generator_lr: float = 0.003
    z_radius: float = 4.0
    temperature_start: float = 1.0
    temperature_end: float = 0.1
    binary_penalty: float = 0.01
    category_entropy_penalty: float = 0.001
    category_balance_penalty: float = 0.01
    support_per_class: int = 256
    validation_per_class: int = 128
    query_per_class: int = 256
    evaluation_support_per_class: int = 512
    evaluation_validation_per_class: int = 256
    evaluation_query_per_class: int = 512

    def __post_init__(self) -> None:
        if not self.pattern_lengths:
            raise ValueError("pattern_lengths cannot be empty")
        if tuple(sorted(set(self.pattern_lengths))) != self.pattern_lengths:
            raise ValueError("pattern_lengths must be sorted and unique")
        if min(self.pattern_lengths) <= 0 or max(self.pattern_lengths) > self.seq_len:
            raise ValueError("pattern lengths must lie inside the sequence")
        if self.filter_dim < max(self.pattern_lengths):
            raise ValueError("filter_dim must cover the maximum pattern length")
        if self.hidden != self.seq_len:
            raise ValueError("the cyclic super-structure requires hidden == seq_len")
        integer_names = (
            "seq_len", "hidden", "filter_dim", "latent_dim", "generator_width",
            "train_restarts", "eval_restarts", "tasks_per_length", "outer_steps",
            "temperature_anneal_steps", "checkpoint_every", "outer_patience",
            "checkpoint_task_chunk", "inner_steps", "final_refit_steps",
            "refit_validate_every", "eval_task_chunk",
            "support_per_class", "validation_per_class", "query_per_class",
            "evaluation_support_per_class", "evaluation_validation_per_class",
            "evaluation_query_per_class",
        )
        if any(getattr(self, name) <= 0 for name in integer_names):
            raise ValueError("integer configuration values must be positive")

    @property
    def pattern_len(self) -> int:
        """Compatibility with the fixed-width assignment constructor."""
        return self.filter_dim

    @property
    def k_active(self) -> int:
        return self.hidden * self.filter_dim

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["pattern_lengths"] = list(self.pattern_lengths)
        return result


@dataclass(frozen=True)
class PatternTask:
    length: int
    pattern: str


@dataclass
class MultiLengthData:
    tasks: tuple[PatternTask, ...]
    support_x: torch.Tensor
    support_y: torch.Tensor
    validation_x: torch.Tensor
    validation_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor

    def subset(self, indices: torch.Tensor) -> "MultiLengthData":
        raw = indices.detach().cpu().tolist()
        return MultiLengthData(
            tuple(self.tasks[index] for index in raw),
            self.support_x[indices], self.support_y[indices],
            self.validation_x[indices], self.validation_y[indices],
            self.query_x[indices], self.query_y[indices],
        )


def all_tasks(lengths: Sequence[int]) -> tuple[PatternTask, ...]:
    return tuple(
        PatternTask(length, f"{value:0{length}b}")
        for length in lengths
        for value in range(2**length)
    )


def labels(x01: torch.Tensor, pattern: str) -> torch.Tensor:
    bits = torch.tensor([int(bit) for bit in pattern], device=x01.device, dtype=x01.dtype)
    return (x01.unfold(-1, len(pattern), 1) == bits).all(-1).any(-1)


def _sample_balanced(task: PatternTask, per_class: int, config: MultiLengthConfig,
                     seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = torch.Generator(device="cpu").manual_seed(seed)
    positive_parts: list[torch.Tensor] = []
    negative_parts: list[torch.Tensor] = []
    positives = negatives = 0
    while positives < per_class or negatives < per_class:
        x01 = torch.randint(0, 2, (4096, config.seq_len), generator=rng, dtype=torch.float32)
        y = labels(x01, task.pattern)
        if positives < per_class:
            selected = x01[y][:per_class - positives]
            positive_parts.append(selected)
            positives += selected.shape[0]
        if negatives < per_class:
            selected = x01[~y][:per_class - negatives]
            negative_parts.append(selected)
            negatives += selected.shape[0]
    positive = torch.cat(positive_parts)[:per_class]
    negative = torch.cat(negative_parts)[:per_class]
    x = torch.cat((positive, negative))
    y = torch.cat((torch.ones(per_class), torch.zeros(per_class)))
    order = torch.randperm(x.shape[0], generator=rng)
    return x[order].mul(2).sub(1), y[order]


def make_data(tasks: Sequence[PatternTask], config: MultiLengthConfig, device: torch.device,
              tag: str) -> MultiLengthData:
    counts = (config.support_per_class, config.validation_per_class, config.query_per_class)
    total = sum(counts)
    fields_x: list[list[torch.Tensor]] = [[], [], []]
    fields_y: list[list[torch.Tensor]] = [[], [], []]
    for task in tasks:
        x, y = _sample_balanced(
            task, total, config, _seed(config.seed, "data", tag, task.length, task.pattern),
        )
        positive, negative = x[y == 1], x[y == 0]
        offset = 0
        for split, count in enumerate(counts):
            part_x = torch.cat((positive[offset:offset + count], negative[offset:offset + count]))
            part_y = torch.cat((torch.ones(count), torch.zeros(count)))
            rng = torch.Generator(device="cpu").manual_seed(
                _seed(config.seed, "shuffle", tag, task.length, task.pattern, split)
            )
            order = torch.randperm(part_x.shape[0], generator=rng)
            fields_x[split].append(part_x[order])
            fields_y[split].append(part_y[order])
            offset += count
    xs = [torch.stack(value).to(device) for value in fields_x]
    ys = [torch.stack(value).to(device) for value in fields_y]
    return MultiLengthData(tuple(tasks), xs[0], ys[0], xs[1], ys[1], xs[2], ys[2])


def cyclic_assignment(config: MultiLengthConfig, device: torch.device) -> torch.Tensor:
    result = torch.zeros(config.seq_len, config.hidden, config.filter_dim, device=device)
    for column in range(config.hidden):
        for offset in range(config.filter_dim):
            result[(column + offset) % config.seq_len, column, offset] = 1
    return result


def _task_groups(tasks: Sequence[PatternTask], config: MultiLengthConfig,
                 variant: str, device: torch.device) -> torch.Tensor:
    if variant == "global":
        return torch.zeros(len(tasks), dtype=torch.long, device=device)
    if variant == "length_latent":
        lookup = {length: index for index, length in enumerate(config.pattern_lengths)}
        return torch.tensor([lookup[task.length] for task in tasks], device=device)
    raise ValueError(f"unknown variant: {variant}")


def _balanced_length_mean(values: torch.Tensor, tasks: Sequence[PatternTask],
                          lengths: Sequence[int]) -> torch.Tensor:
    return torch.stack([
        values[torch.tensor([task.length == length for task in tasks], device=values.device)].mean(0)
        for length in lengths
    ])


def _sample_task_indices(tasks: Sequence[PatternTask], config: MultiLengthConfig,
                         rng: torch.Generator, device: torch.device) -> torch.Tensor:
    selected: list[int] = []
    for length in config.pattern_lengths:
        pool = torch.tensor([index for index, task in enumerate(tasks) if task.length == length])
        if config.tasks_per_length <= len(pool):
            order = torch.randperm(len(pool), generator=rng)[:config.tasks_per_length]
        else:
            order = torch.randint(len(pool), (config.tasks_per_length,), generator=rng)
        selected.extend(pool[order].tolist())
    return torch.tensor(selected, device=device)


def _select_restarts(validation: torch.Tensor, tasks: Sequence[PatternTask],
                     config: MultiLengthConfig, variant: str) -> tuple[torch.Tensor, torch.Tensor]:
    by_length = _balanced_length_mean(validation, tasks, config.pattern_lengths)
    if variant == "global":
        restart = by_length.mean(0).argmin()
        selected = restart.expand(len(tasks))
        score = by_length.mean(0).min()
    else:
        per_length = by_length.argmin(1)
        lookup = {length: index for index, length in enumerate(config.pattern_lengths)}
        selected = torch.tensor(
            [int(per_length[lookup[task.length]]) for task in tasks], device=validation.device,
        )
        score = by_length.min(1).values.mean()
    return selected, score


def _checkpoint_generator(
    generator: SharingGenerator,
    z: torch.Tensor,
    data: MultiLengthData,
    config: MultiLengthConfig,
    variant: str,
    device: torch.device,
) -> tuple[float, torch.Tensor, list[float]]:
    """Evaluate hard structures on a fixed, balanced full-task panel."""
    with torch.no_grad():
        candidates = assignments(
            generator, z, config, config.temperature_end, "hard",
        )  # type: ignore[arg-type]
    validation_parts: list[torch.Tensor] = []
    for start in range(0, len(data.tasks), config.checkpoint_task_chunk):
        stop = min(start + config.checkpoint_task_chunk, len(data.tasks))
        indices = torch.arange(start, stop, device=device)
        batch = data.subset(indices)
        group_index = _task_groups(batch.tasks, config, variant, device)
        structure = candidates[group_index]
        weights = adapt_weights(
            structure, batch.support_x, batch.support_y, config,
            steps=config.inner_steps,
            seed=_seed(config.seed, variant, "checkpoint-adapt", start),
            create_graph=False,
        )
        with torch.no_grad():
            validation_parts.append(_network_losses(
                sharing_forward(batch.validation_x, weights, structure), batch.validation_y,
            ))
    validation = torch.cat(validation_parts)
    by_length = _balanced_length_mean(validation, data.tasks, config.pattern_lengths)
    if variant == "global":
        group_restarts = by_length.mean(0).argmin().reshape(1)
        score = by_length.mean(0)[group_restarts[0]]
    else:
        group_restarts = by_length.argmin(1)
        score = by_length[torch.arange(len(config.pattern_lengths), device=device), group_restarts].mean()
    length_scores = [float(value) for value in by_length[
        torch.arange(len(config.pattern_lengths), device=device), group_restarts
        if variant == "length_latent" else group_restarts.expand(len(config.pattern_lengths))
    ]]
    return float(score), group_restarts, length_scores


def train_generator(config: MultiLengthConfig, variant: str, device: torch.device,
                    ) -> tuple[SharingGenerator, torch.Tensor, list[dict[str, Any]]]:
    tasks = all_tasks(config.pattern_lengths)
    data = make_data(tasks, config, device, "meta")
    generator = SharingGenerator(config).to(device)  # type: ignore[arg-type]
    groups = 1 if variant == "global" else len(config.pattern_lengths)
    rng = torch.Generator(device="cpu").manual_seed(_seed(config.seed, variant, "task-batches"))
    z_rng = torch.Generator(device="cpu").manual_seed(_seed(config.seed, variant, "z"))
    z = nn.Parameter(torch.randn(groups, config.train_restarts, config.latent_dim, generator=z_rng).to(device))
    _project(z, config.z_radius)
    optimizer_z = torch.optim.Adam([z], lr=config.latent_lr)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=config.generator_lr)
    best_score, best_restarts, initial_length_scores = _checkpoint_generator(
        generator, z, data, config, variant, device,
    )
    best_step = 0
    best_generator = {name: value.detach().clone() for name, value in generator.state_dict().items()}
    best_z = z[torch.arange(groups, device=device), best_restarts][:, None].detach().clone()
    history: list[dict[str, Any]] = [{
        "outer_step": 0,
        "checkpoint_validation_bce": best_score,
        "checkpoint_validation_by_length": initial_length_scores,
        "checkpoint_improved": True,
    }]
    stale_checkpoints = 0
    for outer in range(1, config.outer_steps + 1):
        ratio = min(1.0, (outer - 1) / max(1, config.temperature_anneal_steps - 1))
        temperature = config.temperature_start * (
            config.temperature_end / config.temperature_start
        ) ** ratio
        indices = _sample_task_indices(tasks, config, rng, device)
        batch = data.subset(indices)
        group_index = _task_groups(batch.tasks, config, variant, device)
        # Optimize the continuous relaxation.  Hard assignments are used by
        # the fixed checkpoint panel and final evaluation.
        soft_base = assignments(generator, z, config, temperature, "soft")  # type: ignore[arg-type]
        base = soft_base
        structure = base[group_index]
        weights = adapt_weights(
            structure, batch.support_x, batch.support_y, config, steps=config.inner_steps,
            seed=_seed(config.seed, variant, "meta-adapt", outer), create_graph=True,
        )
        validation = _network_losses(
            sharing_forward(batch.validation_x, weights, structure), batch.validation_y,
        )
        selected, score_tensor = _select_restarts(validation.detach(), batch.tasks, config, variant)
        chosen_structure = _gather_restart(structure, selected)
        chosen_weights = tuple(_gather_restart(value, selected) for value in weights)
        query_per_task = _network_losses(
            sharing_forward(batch.query_x, chosen_weights, chosen_structure), batch.query_y,
        ).squeeze(1)
        query = _balanced_length_mean(
            query_per_task[:, None], batch.tasks, config.pattern_lengths,
        ).mean()
        penalty, penalty_parts = _regularization(soft_base, config)  # type: ignore[arg-type]
        optimizer_z.zero_grad(set_to_none=True)
        optimizer_generator.zero_grad(set_to_none=True)
        z_gradient, = torch.autograd.grad(validation.mean(), z, retain_graph=True)
        generator_gradients = torch.autograd.grad(query + penalty, tuple(generator.parameters()))
        z.grad = z_gradient
        for parameter, gradient in zip(generator.parameters(), generator_gradients):
            parameter.grad = gradient
        z_norm = float(torch.nn.utils.clip_grad_norm_([z], 10.0))
        generator_norm = float(torch.nn.utils.clip_grad_norm_(generator.parameters(), 10.0))
        optimizer_z.step()
        optimizer_generator.step()
        _project(z, config.z_radius)
        row = {
            "outer_step": outer, "query_bce": float(query.detach()),
            "validation_bce": float(score_tensor), "temperature": temperature,
            "z_gradient_norm": z_norm, "generator_gradient_norm": generator_norm,
            **penalty_parts,
        }
        should_checkpoint = outer % config.checkpoint_every == 0 or outer == config.outer_steps
        if should_checkpoint:
            checkpoint_score, checkpoint_restarts, length_scores = _checkpoint_generator(
                generator, z, data, config, variant, device,
            )
            improved = checkpoint_score < best_score - config.outer_min_delta
            row.update({
                "checkpoint_validation_bce": checkpoint_score,
                "checkpoint_validation_by_length": length_scores,
                "checkpoint_improved": improved,
            })
            if improved:
                best_score = checkpoint_score
                best_step = outer
                best_generator = {
                    name: value.detach().clone() for name, value in generator.state_dict().items()
                }
                best_z = z[
                    torch.arange(groups, device=device), checkpoint_restarts
                ][:, None].detach().clone()
                stale_checkpoints = 0
            elif outer >= config.temperature_anneal_steps:
                stale_checkpoints += 1
        history.append(row)
        if outer == 1 or outer % 5 == 0 or outer == config.outer_steps:
            print(
                f"MULTILENGTH variant={variant} seed={config.seed} "
                f"outer={outer}/{config.outer_steps} query={row['query_bce']:.6f} "
                f"val={row['validation_bce']:.6f}"
                + (f" checkpoint={row['checkpoint_validation_bce']:.6f}" if should_checkpoint else ""),
                flush=True,
            )
        if (outer >= config.temperature_anneal_steps
                and stale_checkpoints >= config.outer_patience):
            break
    stopped_by_patience = (
        outer >= config.temperature_anneal_steps
        and stale_checkpoints >= config.outer_patience
    )
    generator.load_state_dict(best_generator)
    history.append({
        "selected_outer_step": best_step,
        "selected_validation_bce": best_score,
        "completed_outer_steps": outer,
        "converged": stopped_by_patience,
    })
    return generator, best_z, history


def _fixed_generated(generator: SharingGenerator, z: torch.Tensor, tasks: Sequence[PatternTask],
                     config: MultiLengthConfig, variant: str, device: torch.device) -> torch.Tensor:
    group_index = _task_groups(tasks, config, variant, device)
    with torch.no_grad():
        base = assignments(generator, z, config, config.temperature_end, "hard")  # type: ignore[arg-type]
        task = base[group_index]
    return task.expand(-1, config.eval_restarts, -1, -1, -1).clone()


def _random_fixed(tasks: Sequence[PatternTask], config: MultiLengthConfig, variant: str,
                  device: torch.device, tag: str) -> torch.Tensor:
    groups = 1 if variant == "global" else len(config.pattern_lengths)
    rng = torch.Generator(device="cpu").manual_seed(_seed(config.seed, variant, "random", tag))
    scores = torch.rand(groups, config.hidden, config.seq_len, generator=rng).to(device)
    indices = scores.topk(config.filter_dim, dim=-1).indices
    categories = torch.rand(groups, config.hidden, config.filter_dim, generator=rng).argsort(-1).to(device)
    by_column = torch.zeros(groups, config.hidden, config.seq_len, config.filter_dim, device=device)
    by_column.scatter_(
        -2, indices[..., None].expand(*indices.shape, config.filter_dim),
        F.one_hot(categories, config.filter_dim).to(torch.float32),
    )
    base = by_column.transpose(-3, -2)[:, None]
    group_index = _task_groups(tasks, config, variant, device)
    return base[group_index].expand(-1, config.eval_restarts, -1, -1, -1).clone()


def _oracle_fixed(tasks: Sequence[PatternTask], config: MultiLengthConfig,
                  device: torch.device) -> torch.Tensor:
    oracle = cyclic_assignment(config, device)[None, None]
    return oracle.expand(len(tasks), config.eval_restarts, -1, -1, -1).clone()


def _active_iou(candidate: torch.Tensor, gold: torch.Tensor) -> float:
    aligned = align_columns(gold, candidate)
    intersection = (aligned.bool() & gold.bool()).sum()
    union = (aligned.bool() | gold.bool()).sum()
    return float(intersection / union)


def _summarize(query: list[float], accuracy: list[float], iou: list[float],
               tasks: Sequence[PatternTask], method: str) -> dict[str, Any]:
    result: dict[str, Any] = {"method": method, "per_length": {}}
    for length in sorted(set(task.length for task in tasks)):
        selected = [index for index, task in enumerate(tasks) if task.length == length]
        result["per_length"][str(length)] = {
            "tasks": len(selected),
            "mean_query_bce": sum(query[index] for index in selected) / len(selected),
            "mean_query_accuracy": sum(accuracy[index] for index in selected) / len(selected),
            "mean_active_iou": sum(iou[index] for index in selected) / len(selected),
        }
    # Pattern counts grow exponentially with length.  The primary aggregate
    # therefore weights lengths equally; task-weighted values are retained as
    # diagnostics rather than silently letting the longest length dominate.
    per_length = list(result["per_length"].values())
    result.update({
        "mean_query_bce": sum(row["mean_query_bce"] for row in per_length) / len(per_length),
        "mean_query_accuracy": sum(row["mean_query_accuracy"] for row in per_length) / len(per_length),
        "mean_active_iou": sum(row["mean_active_iou"] for row in per_length) / len(per_length),
        "task_weighted_query_bce": sum(query) / len(query),
        "task_weighted_query_accuracy": sum(accuracy) / len(accuracy),
        "task_weighted_active_iou": sum(iou) / len(iou),
    })
    return result


def _fit_shared(candidate: torch.Tensor, data: MultiLengthData, config: MultiLengthConfig,
                device: torch.device, method: str, tag: str) -> dict[str, Any]:
    query_values: list[float] = []
    accuracy_values: list[float] = []
    iou_values: list[float] = []
    gold = cyclic_assignment(config, device).sum(-1)
    for start in range(0, len(data.tasks), config.eval_task_chunk):
        stop = min(start + config.eval_task_chunk, len(data.tasks))
        chunk = slice(start, stop)
        structure = candidate[chunk]
        tasks = stop - start
        weights = SharedWeights(
            tasks, config.eval_restarts, config, device,
            _seed(config.seed, tag, method, "weights", start),
        )
        optimizer = torch.optim.Adam(weights.parameters(), lr=config.final_refit_lr)
        best_validation = torch.full((tasks, config.eval_restarts), math.inf, device=device)
        best_weights = tuple(value.detach().clone() for value in weights.values())
        for step in range(1, config.final_refit_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            loss = _network_losses(
                sharing_forward(data.support_x[chunk], weights.values(), structure),
                data.support_y[chunk],
            ).mean()
            loss.backward()
            optimizer.step()
            if step % config.refit_validate_every == 0 or step == config.final_refit_steps:
                with torch.no_grad():
                    current = _network_losses(
                        sharing_forward(data.validation_x[chunk], weights.values(), structure),
                        data.validation_y[chunk],
                    )
                    improved = current < best_validation
                    best_validation.copy_(torch.minimum(best_validation, current))
                    for best, value in zip(best_weights, weights.values()):
                        shape = (*improved.shape, *(1 for _ in range(value.ndim - improved.ndim)))
                        best.copy_(torch.where(improved.reshape(shape), value, best))
        with torch.no_grad():
            validation = best_validation
            selected = validation.argmin(1)
            chosen_structure = _gather_restart(structure, selected)
            chosen_weights = tuple(_gather_restart(value, selected) for value in best_weights)
            logits = sharing_forward(data.query_x[chunk], chosen_weights, chosen_structure).squeeze(1)
            query = F.binary_cross_entropy_with_logits(
                logits, data.query_y[chunk], reduction="none",
            ).mean(1)
            accuracy = ((logits > 0) == data.query_y[chunk].bool()).float().mean(1)
            active = chosen_structure.squeeze(1).sum(-1)
        query_values.extend(query.cpu().tolist())
        accuracy_values.extend(accuracy.cpu().tolist())
        iou_values.extend(_active_iou(mask, gold) for mask in active)
    return _summarize(query_values, accuracy_values, iou_values, data.tasks, method)


def _fit_dense(mask: torch.Tensor, data: MultiLengthData, config: MultiLengthConfig,
               device: torch.device, method: str, tag: str) -> dict[str, Any]:
    query_values: list[float] = []
    accuracy_values: list[float] = []
    iou_values: list[float] = []
    gold = cyclic_assignment(config, device).sum(-1)
    for start in range(0, len(data.tasks), config.eval_task_chunk):
        stop = min(start + config.eval_task_chunk, len(data.tasks))
        chunk = slice(start, stop)
        chunk_mask = mask[chunk]
        tasks = stop - start
        weights = DenseWeights(
            tasks, config.eval_restarts, config, device,
            _seed(config.seed, tag, method, "weights", start),
        )
        optimizer = torch.optim.Adam(weights.parameters(), lr=config.final_refit_lr)
        best_validation = torch.full((tasks, config.eval_restarts), math.inf, device=device)
        best_weights = tuple(value.detach().clone() for value in weights.values())
        for step in range(1, config.final_refit_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            loss = _network_losses(
                _dense_forward(data.support_x[chunk], weights, chunk_mask), data.support_y[chunk],
            ).mean()
            loss.backward()
            optimizer.step()
            if step % config.refit_validate_every == 0 or step == config.final_refit_steps:
                with torch.no_grad():
                    current = _network_losses(
                        _dense_forward(data.validation_x[chunk], weights, chunk_mask),
                        data.validation_y[chunk],
                    )
                    improved = current < best_validation
                    best_validation.copy_(torch.minimum(best_validation, current))
                    for best, value in zip(best_weights, weights.values()):
                        shape = (*improved.shape, *(1 for _ in range(value.ndim - improved.ndim)))
                        best.copy_(torch.where(improved.reshape(shape), value, best))
        with torch.no_grad():
            selected = best_validation.argmin(1)
            chosen_mask = _gather_restart(chunk_mask, selected)
            chosen_weights = tuple(_gather_restart(value, selected) for value in best_weights)
            logits = _dense_forward(
                data.query_x[chunk], _DenseView(chosen_weights), chosen_mask,
            ).squeeze(1)
            query = F.binary_cross_entropy_with_logits(
                logits, data.query_y[chunk], reduction="none",
            ).mean(1)
            accuracy = ((logits > 0) == data.query_y[chunk].bool()).float().mean(1)
        query_values.extend(query.cpu().tolist())
        accuracy_values.extend(accuracy.cpu().tolist())
        iou_values.extend(_active_iou(item, gold) for item in chosen_mask.squeeze(1))
    return _summarize(query_values, accuracy_values, iou_values, data.tasks, method)


def evaluate(generator: SharingGenerator, z: torch.Tensor, config: MultiLengthConfig,
             variant: str, device: torch.device, tag: str) -> dict[str, Any]:
    evaluation_config = replace(
        config,
        support_per_class=config.evaluation_support_per_class,
        validation_per_class=config.evaluation_validation_per_class,
        query_per_class=config.evaluation_query_per_class,
    )
    tasks = all_tasks(evaluation_config.pattern_lengths)
    data = make_data(tasks, evaluation_config, device, tag)
    generated = _fixed_generated(generator, z, tasks, evaluation_config, variant, device)
    random = _random_fixed(tasks, evaluation_config, variant, device, tag)
    oracle = _oracle_fixed(tasks, evaluation_config, device)
    generated_mask = generated.sum(-1)
    dense_mask = torch.ones_like(generated_mask)
    return {
        "tasks": len(tasks),
        "strategies": {
            "generated_sharing": _fit_shared(
                generated, data, evaluation_config, device, "generated sharing", tag,
            ),
            "generated_connectivity": _fit_dense(
                generated_mask, data, evaluation_config, device, "generated connectivity only", tag,
            ),
            "random_sharing": _fit_shared(
                random, data, evaluation_config, device, "random sharing", tag,
            ),
            "analytic_sharing": _fit_shared(
                oracle, data, evaluation_config, device, "analytic cyclic sharing", tag,
            ),
            "dense": _fit_dense(dense_mask, data, evaluation_config, device, "dense", tag),
        },
    }


def _results_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Multi-length generated sharing",
        "",
        f"Variant: `{summary['variant']}`; seed: `{summary['config']['seed']}`.",
        "",
        "| Split | Method | BCE | Accuracy | Active IoU |",
        "|---|---|---:|---:|---:|",
    ]
    for split, evaluation in summary["evaluations"].items():
        for result in evaluation["strategies"].values():
            lines.append(
                f"| {split} | {result['method']} | {result['mean_query_bce']:.6f} | "
                f"{result['mean_query_accuracy']:.6f} | {result['mean_active_iou']:.6f} |"
            )
    lines.extend(["", "## Generated sharing by length", "",
                  "| Split | Length | BCE | Accuracy |", "|---|---:|---:|---:|"])
    for split, evaluation in summary["evaluations"].items():
        result = evaluation["strategies"]["generated_sharing"]
        for length, row in result["per_length"].items():
            lines.append(
                f"| {split} | {length} | {row['mean_query_bce']:.6f} | "
                f"{row['mean_query_accuracy']:.6f} |"
            )
    return "\n".join(lines) + "\n"


def run(config: MultiLengthConfig, variant: str, output: str | Path,
        device: str | torch.device = "cuda") -> Path:
    if variant not in ("global", "length_latent"):
        raise ValueError("variant must be global or length_latent")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(device)
    torch.manual_seed(config.seed)
    started = time.monotonic()
    generator, z, history = train_generator(config, variant, device)
    checkpoint = {
        "config": config.to_dict(), "variant": variant,
        "generator_state": {name: value.cpu() for name, value in generator.state_dict().items()},
        "z": z.cpu(), "history": history,
    }
    torch.save(checkpoint, output / "training.pt")
    evaluations = {
        "validation": evaluate(generator, z, config, variant, device, "fresh-validation"),
        "test": evaluate(generator, z, config, variant, device, "fresh-test"),
    }
    summary = {
        "config": config.to_dict(), "variant": variant, "history": history,
        "evaluations": evaluations,
        "generator_parameters": sum(parameter.numel() for parameter in generator.parameters()),
        "task_specific_parameters": config.filter_dim + 2 * config.hidden + 1,
        "seconds": time.monotonic() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "RESULTS.md").write_text(_results_markdown(summary))
    return output
