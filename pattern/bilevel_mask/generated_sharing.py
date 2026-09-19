"""Compact generator of parameter-sharing schemes for pattern tasks.

The generator maps a latent and edge coordinates to a categorical assignment.
Every active first-layer edge selects one of a few task-specific filter values,
so the generated object is a reparameterization W = U v rather than a gate.
Task weights are freshly adapted while U is fixed, and gradients to z/psi pass
through the adaptation steps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from pattern.evaluation.decoder_agreement import align_columns, hard_topk, soft_topk
from .length32_joint import Data, TEST_PATTERNS, TRAIN_PATTERNS, VALIDATION_PATTERNS, make_data


ALL_PATTERNS = tuple(f"{value:04b}" for value in range(16))


def _seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


@dataclass(frozen=True)
class SharingConfig:
    seed: int = 42
    seq_len: int = 32
    pattern_len: int = 4
    hidden: int = 29
    filter_dim: int = 4
    latent_dim: int = 4
    generator_width: int = 16
    k_active: int = 116
    train_restarts: int = 16
    eval_restarts: int = 64
    outer_steps: int = 100
    inner_steps: int = 20
    latent_search_steps: int = 100
    latent_patience: int = 15
    latent_min_delta: float = 1e-5
    final_refit_steps: int = 2000
    refit_validate_every: int = 20
    inner_lr: float = 0.1
    final_refit_lr: float = 0.001
    latent_lr: float = 0.03
    generator_lr: float = 0.01
    z_radius: float = 4.0
    temperature_start: float = 1.0
    temperature_end: float = 0.10
    binary_penalty: float = 0.01
    category_entropy_penalty: float = 0.001
    category_balance_penalty: float = 0.01
    support_per_class: int = 512
    validation_per_class: int = 256
    query_per_class: int = 2048

    def __post_init__(self) -> None:
        if self.filter_dim <= 0:
            raise ValueError("filter_dim must be positive")
        if self.hidden != self.seq_len - self.pattern_len + 1:
            raise ValueError(
                "hidden must equal the number of contiguous pattern windows: "
                "seq_len - pattern_len + 1"
            )
        if self.k_active != self.pattern_len * self.hidden:
            raise ValueError("k_active must equal pattern_len * hidden")
        if self.filter_dim != self.pattern_len:
            raise ValueError("the current oracle comparison requires filter_dim == pattern_len")
        integer_names = (
            "seq_len", "pattern_len", "hidden", "latent_dim", "generator_width",
            "k_active", "train_restarts", "eval_restarts", "outer_steps", "inner_steps",
            "latent_search_steps", "latent_patience", "final_refit_steps", "refit_validate_every", "support_per_class",
            "validation_per_class", "query_per_class",
        )
        if any(getattr(self, name) <= 0 for name in integer_names):
            raise ValueError("integer configuration values must be positive")

    @property
    def edge_count(self) -> int:
        return self.seq_len * self.hidden

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SharingGenerator(nn.Module):
    """Coordinate decoder for active-edge and shared-value assignments."""

    def __init__(self, config: SharingConfig) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.network = nn.Sequential(
                nn.Linear(config.latent_dim + 2, config.generator_width),
                nn.Tanh(),
                nn.Linear(config.generator_width, config.generator_width),
                nn.Tanh(),
                nn.Linear(config.generator_width, 1 + config.filter_dim),
            )
            nn.init.normal_(self.network[-1].weight, std=0.01)
            nn.init.zeros_(self.network[-1].bias)
        row = torch.linspace(-1, 1, config.seq_len)[:, None].expand(-1, config.hidden)
        column = torch.linspace(-1, 1, config.hidden)[None, :].expand(config.seq_len, -1)
        self.register_buffer("coordinates", torch.stack((row, column), dim=-1))
        permutations = torch.tensor(list(itertools.permutations(range(config.filter_dim))))
        self.register_buffer("assignment_permutations", permutations, persistent=False)
        self.register_buffer(
            "assignment_permutation_one_hot",
            F.one_hot(permutations, config.filter_dim).to(torch.float32),
            persistent=False,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        prefix = z.shape[:-1]
        coordinates = self.coordinates.view(*(1 for _ in prefix), *self.coordinates.shape).expand(*prefix, -1, -1, -1)
        latent = z[..., None, None, :].expand(*prefix, self.coordinates.shape[0], self.coordinates.shape[1], -1)
        return self.network(torch.cat((latent, coordinates), dim=-1))


class _HardForwardSoftBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
        return hard

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[None, torch.Tensor]:
        return None, gradient


def assignments(generator: SharingGenerator, z: torch.Tensor, config: SharingConfig,
                temperature: float, mode: str) -> torch.Tensor:
    """Return U with shape (..., input, hidden, filter_dim)."""
    logits = generator(z)
    active_logits, category_logits = logits[..., 0], logits[..., 1:]
    # Every hidden unit receives exactly pattern_len inputs. This preserves the
    # known complexity while leaving their positions entirely learnable.
    scores_by_column = active_logits.transpose(-2, -1)
    active_soft = soft_topk(scores_by_column, config.pattern_len, temperature).transpose(-2, -1)
    active_hard = hard_topk(scores_by_column, config.pattern_len).transpose(-2, -1)
    category_soft = torch.softmax(category_logits / temperature, dim=-1)
    soft = active_soft[..., None] * category_soft
    if mode == "soft":
        return soft

    # Selected inputs of every hidden unit receive a permutation of the shared
    # filter parameters. Exact scoring avoids categorical collapse.
    active_indices = scores_by_column.topk(config.pattern_len, dim=-1).indices
    category_by_column = category_logits.transpose(-3, -2)
    gather_index = active_indices[..., None].expand(*active_indices.shape, config.filter_dim)
    selected_scores = torch.gather(category_by_column, -2, gather_index)
    permutations = generator.assignment_permutations
    permutation_one_hot = generator.assignment_permutation_one_hot.to(category_logits.dtype)
    permutation_scores = torch.einsum("...hkq,pkq->...hp", selected_scores, permutation_one_hot)
    best_permutation = permutations[permutation_scores.argmax(-1)]
    selected_hard = F.one_hot(best_permutation, config.filter_dim).to(category_logits.dtype)
    hard_by_column = torch.zeros_like(category_by_column)
    hard_by_column.scatter_(-2, gather_index, selected_hard)
    hard = hard_by_column.transpose(-3, -2) * active_hard[..., None]
    if mode == "hard":
        return hard
    elif mode == "ste":
        return _HardForwardSoftBackward.apply(hard, soft)
    else:
        raise ValueError(f"unknown assignment mode: {mode}")


def analytic_assignment(config: SharingConfig, device: torch.device) -> torch.Tensor:
    result = torch.zeros(config.seq_len, config.hidden, config.filter_dim, device=device)
    windows = config.seq_len - config.pattern_len + 1
    for column in range(config.hidden):
        start = column % windows
        for offset in range(config.pattern_len):
            result[start + offset, column, offset] = 1
    return result


FunctionalWeights = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _init_functional_weights(tasks: int, restarts: int, config: SharingConfig,
                             device: torch.device, seed: int) -> FunctionalWeights:
    rng = torch.Generator(device="cpu").manual_seed(seed)
    filt = (0.1 * torch.randn(tasks, restarts, config.filter_dim, generator=rng)).to(device).requires_grad_(True)
    b1 = torch.zeros(tasks, restarts, config.hidden, device=device, requires_grad=True)
    w2 = (0.1 * torch.randn(tasks, restarts, config.hidden, generator=rng)).to(device).requires_grad_(True)
    b2 = torch.zeros(tasks, restarts, device=device, requires_grad=True)
    return filt, b1, w2, b2


def sharing_forward(x: torch.Tensor, weights: FunctionalWeights, assignment: torch.Tensor) -> torch.Tensor:
    filt, b1, w2, b2 = weights
    w1 = torch.einsum("trihq,trq->trih", assignment, filt)
    hidden = F.relu(torch.einsum("tbi,trih->trbh", x, w1) + b1[:, :, None, :])
    return torch.einsum("trbh,trh->trb", hidden, w2) + b2[:, :, None]


def _network_losses(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    target = targets[:, None].expand_as(logits)
    return F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(-1)


def adapt_weights(assignment: torch.Tensor, x: torch.Tensor, y: torch.Tensor, config: SharingConfig,
                  *, steps: int, seed: int, create_graph: bool) -> FunctionalWeights:
    """Fresh functional Adam adaptation with an optional exact hypergradient."""
    tasks, restarts = assignment.shape[:2]
    weights = _init_functional_weights(tasks, restarts, config, x.device, seed)
    first = tuple(torch.zeros_like(value) for value in weights)
    second = tuple(torch.zeros_like(value) for value in weights)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    for step in range(1, steps + 1):
        objective = _network_losses(sharing_forward(x, weights, assignment), y).sum()
        gradients = torch.autograd.grad(objective, weights, create_graph=create_graph)
        first = tuple(beta1 * moment + (1 - beta1) * gradient for moment, gradient in zip(first, gradients))
        second = tuple(beta2 * moment + (1 - beta2) * gradient.square() for moment, gradient in zip(second, gradients))
        weights = tuple(
            value - config.inner_lr * (mean / (1 - beta1**step)) /
            (torch.sqrt((variance / (1 - beta2**step)).clamp_min(1e-16)) + epsilon)
            for value, mean, variance in zip(weights, first, second)
        )  # type: ignore[assignment]
        if not create_graph:
            weights = tuple(value.detach().requires_grad_(True) for value in weights)  # type: ignore[assignment]
            first = tuple(value.detach() for value in first)
            second = tuple(value.detach() for value in second)
    return weights


def _project(z: torch.Tensor, radius: float) -> None:
    with torch.no_grad():
        norm = z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        z.mul_(torch.clamp(radius / norm, max=1.0))


def _new_z(tasks: int, restarts: int, config: SharingConfig, device: torch.device, tag: str) -> nn.Parameter:
    rng = torch.Generator(device="cpu").manual_seed(_seed(config.seed, "z", tag))
    z = nn.Parameter(torch.randn(tasks, restarts, config.latent_dim, generator=rng).to(device))
    _project(z, config.z_radius)
    return z


def _temperature(config: SharingConfig, step: int) -> float:
    if config.outer_steps == 1:
        return config.temperature_end
    ratio = (step - 1) / (config.outer_steps - 1)
    return config.temperature_start * (config.temperature_end / config.temperature_start) ** ratio


def _gather_restart(value: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    task = torch.arange(value.shape[0], device=value.device)
    return value[task, selected][:, None]


def _regularization(soft: torch.Tensor, config: SharingConfig) -> tuple[torch.Tensor, dict[str, float]]:
    active = soft.sum(-1)
    binary = (active * (1 - active)).mean()
    category = soft / active[..., None].clamp_min(1e-8)
    entropy = -(soft * category.clamp_min(1e-8).log()).sum((-3, -2, -1)).mean() / config.k_active
    # Each category should be used once within every hidden unit.
    usage = soft.sum(-3)
    balance = (usage - 1).square().mean()
    total = (config.binary_penalty * binary + config.category_entropy_penalty * entropy
             + config.category_balance_penalty * balance)
    return total, {"binary": float(binary.detach()), "category_entropy": float(entropy.detach()),
                   "category_balance": float(balance.detach())}


def train_generator(config: SharingConfig, device: torch.device,
                    training_patterns: Sequence[str] = TRAIN_PATTERNS,
                    ) -> tuple[SharingGenerator, torch.Tensor, list[dict[str, Any]]]:
    training_patterns = tuple(training_patterns)
    data = make_data(training_patterns, config, device)
    generator = SharingGenerator(config).to(device)
    # Pattern identities differ, but their translation structure is shared.
    # A single bank of latents prevents the generator from memorizing a
    # separate structure for every meta-training task.
    z = _new_z(1, config.train_restarts, config, device, "meta-train-shared")
    optimizer_z = torch.optim.Adam([z], lr=config.latent_lr)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=config.generator_lr)
    history: list[dict[str, Any]] = []
    best_score = math.inf
    best_step = 0
    best_generator = {name: value.detach().clone() for name, value in generator.state_dict().items()}
    best_z = z.detach().clone()
    for outer in range(1, config.outer_steps + 1):
        temperature = _temperature(config, outer)
        expanded_z = z.expand(len(training_patterns), -1, -1)
        structure = assignments(generator, expanded_z, config, temperature, "ste")
        weights = adapt_weights(
            structure, data.support_x, data.support_y, config, steps=config.inner_steps,
            seed=_seed(config.seed, "meta-adapt"), create_graph=True,
        )
        validation = _network_losses(sharing_forward(data.validation_x, weights, structure), data.validation_y)
        shared_restart = validation.detach().mean(0).argmin()
        selected = shared_restart.expand(len(training_patterns))
        chosen_structure = _gather_restart(structure, selected)
        chosen_weights = tuple(_gather_restart(value, selected) for value in weights)
        query = _network_losses(sharing_forward(data.query_x, chosen_weights, chosen_structure), data.query_y).mean()
        soft = assignments(generator, expanded_z, config, temperature, "soft")
        penalty, penalty_parts = _regularization(soft, config)
        score = float(validation.mean(0).min().detach())
        if score < best_score:
            best_score = score
            best_step = outer
            best_generator = {name: value.detach().clone() for name, value in generator.state_dict().items()}
            best_restart = validation.detach().mean(0).argmin()
            best_z = z[:, best_restart:best_restart + 1].detach().clone()

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
        row = {"outer_step": outer, "query_bce": float(query.detach()),
               "validation_bce": float(validation.min(1).values.mean().detach()),
               "temperature": temperature, "z_gradient_norm": z_norm,
               "generator_gradient_norm": generator_norm, **penalty_parts}
        history.append(row)
        if outer == 1 or outer % 5 == 0 or outer == config.outer_steps:
            print(f"SHARING seed={config.seed} outer={outer}/{config.outer_steps} "
                  f"query={row['query_bce']:.6f} val={row['validation_bce']:.6f}", flush=True)
    generator.load_state_dict(best_generator)
    history.append({"selected_outer_step": best_step, "selected_validation_bce": best_score})
    return generator, best_z, history


def search_latent(generator: SharingGenerator, data: Data, config: SharingConfig,
                  device: torch.device, split: str) -> tuple[torch.Tensor, dict[str, Any]]:
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    z = _new_z(len(data.patterns), config.eval_restarts, config, device, f"{split}-search")
    optimizer = torch.optim.Adam([z], lr=config.latent_lr)
    best_z = z.detach().clone()
    best_per_restart = torch.full(z.shape[:2], math.inf, device=device)
    best_score = math.inf
    stale = 0
    for step in range(1, config.latent_search_steps + 1):
        structure = assignments(generator, z, config, config.temperature_end, "ste")
        weights = adapt_weights(
            structure, data.support_x, data.support_y, config, steps=config.inner_steps,
            seed=_seed(config.seed, split, "search-adapt"), create_graph=True,
        )
        validation = _network_losses(sharing_forward(data.validation_x, weights, structure), data.validation_y)
        with torch.no_grad():
            improved_restart = validation < best_per_restart
            best_per_restart.copy_(torch.minimum(best_per_restart, validation))
            best_z.copy_(torch.where(improved_restart[..., None], z, best_z))
            score = float(validation.min(1).values.mean())
            improved = score < best_score - config.latent_min_delta
            best_score = min(best_score, score)
            stale = 0 if improved else stale + 1
        optimizer.zero_grad(set_to_none=True)
        gradient, = torch.autograd.grad(validation.mean(), z)
        z.grad = gradient
        optimizer.step()
        _project(z, config.z_radius)
        if stale >= config.latent_patience:
            break
    with torch.no_grad():
        z.copy_(best_z)
        result = assignments(generator, z, config, config.temperature_end, "hard")
    for parameter in generator.parameters():
        parameter.requires_grad_(True)
    return result, {"steps": step, "validation_bce": best_score,
                    "converged": int(step < config.latent_search_steps)}


class SharedWeights(nn.Module):
    def __init__(self, tasks: int, restarts: int, config: SharingConfig,
                 device: torch.device, seed: int) -> None:
        super().__init__()
        initial = _init_functional_weights(tasks, restarts, config, device, seed)
        self.filt = nn.Parameter(initial[0].detach())
        self.b1 = nn.Parameter(initial[1].detach())
        self.w2 = nn.Parameter(initial[2].detach())
        self.b2 = nn.Parameter(initial[3].detach())

    def values(self) -> FunctionalWeights:
        return self.filt, self.b1, self.w2, self.b2


class DenseWeights(nn.Module):
    def __init__(self, tasks: int, restarts: int, config: SharingConfig,
                 device: torch.device, seed: int) -> None:
        super().__init__()
        rng = torch.Generator(device="cpu").manual_seed(seed)
        self.w1 = nn.Parameter((0.1 * torch.randn(
            tasks, restarts, config.seq_len, config.hidden, generator=rng)).to(device))
        self.b1 = nn.Parameter(torch.zeros(tasks, restarts, config.hidden, device=device))
        self.w2 = nn.Parameter((0.1 * torch.randn(tasks, restarts, config.hidden, generator=rng)).to(device))
        self.b2 = nn.Parameter(torch.zeros(tasks, restarts, device=device))

    def values(self) -> tuple[torch.Tensor, ...]:
        return self.w1, self.b1, self.w2, self.b2


def _dense_forward(x: torch.Tensor, weights: DenseWeights, mask: torch.Tensor) -> torch.Tensor:
    w1, b1, w2, b2 = weights.values()
    hidden = F.relu(torch.einsum("tbi,trih->trbh", x, w1 * mask) + b1[:, :, None, :])
    return torch.einsum("trbh,trh->trb", hidden, w2) + b2[:, :, None]


def _active_iou(candidate: torch.Tensor, gold: torch.Tensor) -> float:
    aligned = align_columns(gold, candidate)
    intersection = (aligned.bool() & gold.bool()).sum()
    union = (aligned.bool() | gold.bool()).sum()
    return float(intersection / union)


def _fit_shared(candidate: torch.Tensor, data: Data, config: SharingConfig,
                device: torch.device, method: str) -> dict[str, Any]:
    tasks, restarts = candidate.shape[:2]
    weights = SharedWeights(tasks, restarts, config, device, _seed(config.seed, "fair-shared-weights"))
    optimizer = torch.optim.Adam(weights.parameters(), lr=config.final_refit_lr)
    best_validation = torch.full((tasks, restarts), math.inf, device=device)
    best_weights = tuple(value.detach().clone() for value in weights.values())
    for step in range(1, config.final_refit_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = _network_losses(sharing_forward(data.support_x, weights.values(), candidate), data.support_y).mean()
        loss.backward()
        optimizer.step()
        if step % config.refit_validate_every == 0 or step == config.final_refit_steps:
            with torch.no_grad():
                current = _network_losses(sharing_forward(data.validation_x, weights.values(), candidate), data.validation_y)
                improved = current < best_validation
                best_validation.copy_(torch.minimum(best_validation, current))
                for best, value in zip(best_weights, weights.values()):
                    shape = (*improved.shape, *(1 for _ in range(value.ndim - improved.ndim)))
                    best.copy_(torch.where(improved.reshape(shape), value, best))
    with torch.no_grad():
        for target, best in zip(weights.values(), best_weights):
            target.copy_(best)
    with torch.no_grad():
        validation = _network_losses(sharing_forward(data.validation_x, weights.values(), candidate), data.validation_y)
        selected = validation.argmin(1)
        chosen_assignment = _gather_restart(candidate, selected)
        chosen_weights = tuple(_gather_restart(value, selected) for value in weights.values())
        logits = sharing_forward(data.query_x, chosen_weights, chosen_assignment).squeeze(1)
        query = F.binary_cross_entropy_with_logits(logits, data.query_y, reduction="none").mean(1)
        accuracy = ((logits > 0) == data.query_y.bool()).float().mean(1)
    gold = analytic_assignment(config, device).sum(-1)
    active = chosen_assignment.squeeze(1).sum(-1)
    ious = [_active_iou(mask, gold) for mask in active]
    return {"method": method, "mean_query_bce": float(query.mean()),
            "mean_query_accuracy": float(accuracy.mean()), "mean_active_iou": sum(ious) / len(ious),
            "chosen_active_masks": active.cpu().tolist(),
            "tasks": [{"pattern": pattern, "query_bce": float(query[index]),
                       "accuracy": float(accuracy[index]), "active_iou": ious[index],
                       "selected_restart": int(selected[index])}
                      for index, pattern in enumerate(data.patterns)]}


def _fit_dense(mask: torch.Tensor, data: Data, config: SharingConfig,
               device: torch.device, method: str) -> dict[str, Any]:
    tasks, restarts = mask.shape[:2]
    weights = DenseWeights(tasks, restarts, config, device, _seed(config.seed, "fair-dense-weights"))
    optimizer = torch.optim.Adam(weights.parameters(), lr=config.final_refit_lr)
    best_validation = torch.full((tasks, restarts), math.inf, device=device)
    best_weights = tuple(value.detach().clone() for value in weights.values())
    for step in range(1, config.final_refit_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = _network_losses(_dense_forward(data.support_x, weights, mask), data.support_y).mean()
        loss.backward()
        optimizer.step()
        if step % config.refit_validate_every == 0 or step == config.final_refit_steps:
            with torch.no_grad():
                current = _network_losses(_dense_forward(data.validation_x, weights, mask), data.validation_y)
                improved = current < best_validation
                best_validation.copy_(torch.minimum(best_validation, current))
                for best, value in zip(best_weights, weights.values()):
                    shape = (*improved.shape, *(1 for _ in range(value.ndim - improved.ndim)))
                    best.copy_(torch.where(improved.reshape(shape), value, best))
    with torch.no_grad():
        for target, best in zip(weights.values(), best_weights):
            target.copy_(best)
    with torch.no_grad():
        validation = _network_losses(_dense_forward(data.validation_x, weights, mask), data.validation_y)
        selected = validation.argmin(1)
        chosen_mask = _gather_restart(mask, selected)
        chosen_weights = tuple(_gather_restart(value, selected) for value in weights.values())
        logits = _dense_forward(data.query_x, _DenseView(chosen_weights), chosen_mask).squeeze(1)
        query = F.binary_cross_entropy_with_logits(logits, data.query_y, reduction="none").mean(1)
        accuracy = ((logits > 0) == data.query_y.bool()).float().mean(1)
    gold = analytic_assignment(config, device).sum(-1)
    ious = [_active_iou(item, gold) for item in chosen_mask.squeeze(1)]
    return {"method": method, "mean_query_bce": float(query.mean()),
            "mean_query_accuracy": float(accuracy.mean()), "mean_active_iou": sum(ious) / len(ious)}


class _DenseView:
    def __init__(self, values: tuple[torch.Tensor, ...]) -> None:
        self._values = values

    def values(self) -> tuple[torch.Tensor, ...]:
        return self._values


def _random_assignment(tasks: int, config: SharingConfig, device: torch.device, split: str) -> torch.Tensor:
    rng = torch.Generator(device="cpu").manual_seed(_seed(config.seed, "random", split))
    scores = torch.rand(tasks, config.eval_restarts, config.hidden, config.seq_len, generator=rng).to(device)
    indices = scores.topk(config.pattern_len, dim=-1).indices
    category_order = torch.rand(
        tasks, config.eval_restarts, config.hidden, config.filter_dim, generator=rng,
    ).argsort(-1).to(device)
    result_by_column = torch.zeros(
        tasks, config.eval_restarts, config.hidden, config.seq_len, config.filter_dim, device=device,
    )
    scatter_index = indices[..., None].expand(*indices.shape, config.filter_dim)
    result_by_column.scatter_(-2, scatter_index, F.one_hot(category_order, config.filter_dim).to(torch.float32))
    return result_by_column.transpose(-3, -2)


def _fixed_assignment(generator: SharingGenerator, shared_z: torch.Tensor, tasks: int,
                      restarts: int, config: SharingConfig) -> torch.Tensor:
    with torch.no_grad():
        single = assignments(
            generator, shared_z.expand(tasks, -1, -1), config,
            config.temperature_end, "hard",
        )
    return single.expand(-1, restarts, -1, -1, -1).clone()


def evaluate(generator: SharingGenerator, shared_z: torch.Tensor, patterns: Sequence[str], config: SharingConfig,
             device: torch.device, split: str) -> dict[str, Any]:
    data = make_data(patterns, config, device)
    generated = _fixed_assignment(generator, shared_z, len(patterns), config.eval_restarts, config)
    random = _random_assignment(len(patterns), config, device, split)
    oracle = analytic_assignment(config, device)[None, None].expand(
        len(patterns), config.eval_restarts, -1, -1, -1).clone()
    generated_mask = generated.sum(-1)
    dense_mask = torch.ones_like(generated_mask)
    return {
        "patterns": list(patterns),
        "strategies": {
            "generated_sharing": _fit_shared(generated, data, config, device, "generated sharing"),
            "generated_connectivity": _fit_dense(generated_mask, data, config, device, "generated connectivity only"),
            "random_sharing": _fit_shared(random, data, config, device, "random sharing best-of-64"),
            "analytic_sharing": _fit_shared(oracle, data, config, device, "analytic sharing"),
            "dense": _fit_dense(dense_mask, data, config, device, "dense"),
        },
        "convergence": {"mode": "fixed global structure"},
    }


def run(config: SharingConfig, output: str | Path, device: str | torch.device = "cuda",
        pattern_coverage: str = "all16") -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(device)
    torch.manual_seed(config.seed)
    started = time.monotonic()
    if pattern_coverage == "train10":
        training_patterns = TRAIN_PATTERNS
        validation_patterns, validation_config = VALIDATION_PATTERNS, config
        test_patterns, test_config = TEST_PATTERNS, config
    elif pattern_coverage == "all16":
        training_patterns = ALL_PATTERNS
        validation_patterns = test_patterns = ALL_PATTERNS
        validation_config = SharingConfig(**{**config.to_dict(), "seed": config.seed + 10_000})
        test_config = SharingConfig(**{**config.to_dict(), "seed": config.seed + 20_000})
    else:
        raise ValueError("pattern_coverage must be 'train10' or 'all16'")
    generator, shared_z, history = train_generator(config, device, training_patterns)
    torch.save({"config": config.to_dict(), "generator": generator.state_dict(), "shared_z": shared_z,
                "history": history}, output / "training.pt")
    evaluations = {
        "validation": evaluate(generator, shared_z, validation_patterns, validation_config, device, "validation"),
        "test": evaluate(generator, shared_z, test_patterns, test_config, device, "test"),
    }
    summary = {"config": config.to_dict(), "history": history, "evaluations": evaluations,
               "pattern_coverage": pattern_coverage, "training_patterns": list(training_patterns),
               "generator_parameters": sum(parameter.numel() for parameter in generator.parameters()),
               "task_specific_parameters": config.filter_dim + 2 * config.hidden + 1,
               "seconds": time.monotonic() - started}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_report(summary, output / "RESULTS.md")
    _plot(summary, output / "summary.png")
    return output


def reevaluate(checkpoint: str | Path, output: str | Path, device: str | torch.device = "cuda",
               pattern_coverage: str = "all16") -> Path:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = SharingConfig(**payload["config"])
    device = torch.device(device)
    generator = SharingGenerator(config).to(device)
    generator.load_state_dict(payload["generator"])
    shared_z = payload["shared_z"].to(device)
    if pattern_coverage == "all16":
        validation_patterns = test_patterns = ALL_PATTERNS
        validation_config = SharingConfig(**{**config.to_dict(), "seed": config.seed + 10_000})
        test_config = SharingConfig(**{**config.to_dict(), "seed": config.seed + 20_000})
    elif pattern_coverage == "train10":
        validation_patterns, test_patterns = VALIDATION_PATTERNS, TEST_PATTERNS
        validation_config = test_config = config
    else:
        raise ValueError("pattern_coverage must be 'train10' or 'all16'")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    evaluations = {
        "validation": evaluate(generator, shared_z, validation_patterns, validation_config, device, "validation"),
        "test": evaluate(generator, shared_z, test_patterns, test_config, device, "test"),
    }
    summary = {"config": config.to_dict(), "history": payload["history"], "evaluations": evaluations,
               "pattern_coverage": pattern_coverage,
               "generator_parameters": sum(parameter.numel() for parameter in generator.parameters()),
               "task_specific_parameters": config.filter_dim + 2 * config.hidden + 1,
               "seconds": time.monotonic() - started, "reevaluated_from": str(checkpoint)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_report(summary, output / "RESULTS.md")
    _plot(summary, output / "summary.png")
    return output


def _write_report(summary: dict[str, Any], output: Path) -> None:
    lines = ["# Generated parameter sharing", "",
             f"Generator parameters: `{summary['generator_parameters']}`; task-specific parameters: "
             f"`{summary['task_specific_parameters']}`.", "",
             "| Split | Method | Query BCE | Accuracy | Active IoU |", "|---|---|---:|---:|---:|"]
    for split in ("validation", "test"):
        for result in summary["evaluations"][split]["strategies"].values():
            lines.append(f"| {split} | {result['method']} | {result['mean_query_bce']:.6f} | "
                         f"{result['mean_query_accuracy']:.4f} | {result['mean_active_iou']:.4f} |")
    output.write_text("\n".join(lines) + "\n")


def _plot(summary: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    strategies = summary["evaluations"]["test"]["strategies"]
    names = list(strategies)
    labels = [strategies[name]["method"] for name in names]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fields = (("mean_query_bce", "Test BCE ↓"), ("mean_query_accuracy", "Accuracy ↑"),
              ("mean_active_iou", "Active IoU ↑"))
    for axis, (field, title) in zip(axes, fields):
        axis.bar(range(len(names)), [strategies[name][field] for name in names])
        axis.set_xticks(range(len(names)), labels, rotation=25, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)
