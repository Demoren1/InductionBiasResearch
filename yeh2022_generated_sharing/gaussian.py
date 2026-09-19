"""Batched Gaussian shared-means benchmark from Yeh et al. (AISTATS 2022).

The paper samples ``y ~ N(A_gt psi, sigma^2 I)`` and chooses a sharing matrix
on a train/validation split.  This module retains that protocol, solves the
lower mean-estimation problem analytically, and adds a coordinate-conditioned
generator as an alternative parameterization of the assignment matrix.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Literal

import torch
from torch import nn

from .core import (
    CoordinateAssignmentGenerator,
    DirectAssignment,
    assignment_regularizers,
    partition_distance,
    release_assignment_regularizers,
    solve_shared_mean,
)


OptimizerName = Literal["adam", "rmsprop"]
GaussianLowerSolver = Literal["paper_pinv", "release_normalized"]


@dataclass(frozen=True)
class GaussianConfig:
    """Configuration; defaults match the Gaussian experiment in Yeh et al."""

    seed: int = 0
    runs: int = 200
    dimensions: int = 10
    true_rank: int = 1
    num_samples: int = 100
    num_train: int = 30
    noise_std: float = 1.0
    mean_spacing: float = 3.0
    epochs: int = 1000
    restarts: int = 3
    learning_rate: float = 2e-2
    weight_decay: float = 1e-4
    entropy_weight: float = 1e-2
    nuclear_weight: float = 1e-2
    ridge: float = 0.0
    temperature: float = 1.0
    optimizer: OptimizerName = "rmsprop"
    lower_solver: GaussianLowerSolver = "release_normalized"
    latent_dim: int = 8
    generator_width: int = 64
    log_every: int = 50

    def __post_init__(self) -> None:
        if self.runs <= 0 or self.dimensions <= 0 or self.num_samples <= 1:
            raise ValueError("runs, dimensions, and num_samples must be positive")
        if not 1 <= self.true_rank <= self.dimensions:
            raise ValueError("true_rank must be in [1, dimensions]")
        if not 1 <= self.num_train < self.num_samples:
            raise ValueError("num_train must leave at least one validation sample")
        if self.noise_std <= 0 or self.mean_spacing <= 0:
            raise ValueError("noise_std and mean_spacing must be positive")
        if self.epochs <= 0 or self.restarts <= 0 or self.learning_rate <= 0 or self.temperature <= 0:
            raise ValueError("epochs, restarts, learning_rate, and temperature must be positive")
        if self.weight_decay < 0 or self.entropy_weight < 0 or self.nuclear_weight < 0:
            raise ValueError("regularization weights must be non-negative")
        if self.ridge < 0 or self.latent_dim <= 0 or self.generator_width <= 0:
            raise ValueError("ridge must be non-negative; generator sizes must be positive")
        if self.optimizer not in {"adam", "rmsprop"}:
            raise ValueError("optimizer must be 'adam' or 'rmsprop'")
        if self.lower_solver not in {"paper_pinv", "release_normalized"}:
            raise ValueError("unknown Gaussian lower solver")

    @property
    def num_validation(self) -> int:
        return self.num_samples - self.num_train

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def quick(cls, **overrides: Any) -> "GaussianConfig":
        """A CPU-friendly smoke configuration; paper defaults remain ``cls()``."""
        values: dict[str, Any] = {
            "runs": 12,
            "dimensions": 8,
            "epochs": 80,
            "restarts": 2,
            "log_every": 20,
            "generator_width": 32,
        }
        values.update(overrides)
        return cls(**values)


@dataclass
class GaussianBatch:
    train: torch.Tensor
    validation: torch.Tensor
    full: torch.Tensor
    theta_gt: torch.Tensor
    assignment_gt: torch.Tensor


def _partitions(items: tuple[int, ...]) -> list[tuple[tuple[int, ...], ...]]:
    """Enumerate set partitions; only used for the paper's K <= 6 branch."""
    if len(items) == 1:
        return [((items[0],),)]
    first, rest = items[0], items[1:]
    result: list[tuple[tuple[int, ...], ...]] = []
    for smaller in _partitions(rest):
        for index, cluster in enumerate(smaller):
            result.append(smaller[:index] + ((first,) + cluster,) + smaller[index + 1 :])
        result.append(((first,),) + smaller)
    return result


def make_gaussian_batch(config: GaussianConfig, device: torch.device) -> GaussianBatch:
    """Make all independent trials in one tensor, deterministically.

    The ground-truth construction reproduces the released Yeh et al. code for
    fixed rank and dimensions larger than six: the first ``rank`` coordinates
    seed clusters and all other coordinates sample one of those clusters.
    """
    generator = torch.Generator(device="cpu").manual_seed(int(config.seed))
    batch, k = config.runs, config.dimensions
    labels = torch.empty(batch, k, dtype=torch.long)
    if k <= 6:
        # The released implementation samples uniformly from all set
        # partitions having the requested number of clusters at these small K.
        candidates = [part for part in _partitions(tuple(range(k))) if len(part) == config.true_rank]
        selected = torch.randint(len(candidates), (batch,), generator=generator)
        for run, candidate_index in enumerate(selected.tolist()):
            for cluster_index, cluster in enumerate(candidates[candidate_index]):
                labels[run, list(cluster)] = cluster_index
    else:
        # This is the fixed-rank branch in the authors' released code.
        labels[:, : config.true_rank] = torch.arange(config.true_rank)
        if config.true_rank < k:
            labels[:, config.true_rank :] = torch.randint(
                config.true_rank, (batch, k - config.true_rank), generator=generator
            )
    assignment_gt = torch.nn.functional.one_hot(labels, num_classes=k).to(torch.float32)
    psi = torch.arange(k, dtype=torch.float32) * config.mean_spacing
    theta_gt = assignment_gt @ psi
    data = theta_gt[:, None, :] + config.noise_std * torch.randn(
        batch, config.num_samples, k, generator=generator
    )
    data = data.to(device)
    return GaussianBatch(
        train=data[:, : config.num_train],
        validation=data[:, config.num_train :],
        full=data,
        theta_gt=theta_gt.to(device),
        assignment_gt=assignment_gt.to(device),
    )


def _release_shared_mean(samples: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
    """Reproduce the released lower estimator for relaxed assignments.

    Each category value is the assignment-weighted empirical mean, normalized
    by its soft membership count.  This coincides with ordinary group means
    after hardening and, unlike an OLS projection, matches the authors' code.
    """
    empirical_mean = samples.mean(dim=-2)
    counts = assignment.sum(dim=-2).clamp_min(torch.finfo(assignment.dtype).eps)
    category_mean = (
        assignment.transpose(-2, -1) @ empirical_mean.unsqueeze(-1)
    ).squeeze(-1) / counts
    return (assignment @ category_mean.unsqueeze(-1)).squeeze(-1)


def _fit_gaussian_mean(
    samples: torch.Tensor,
    assignment: torch.Tensor,
    config: GaussianConfig,
) -> torch.Tensor:
    if config.lower_solver == "paper_pinv":
        return solve_shared_mean(samples, assignment, ridge=config.ridge)
    return _release_shared_mean(samples, assignment)


def _validation_loss(
    batch: GaussianBatch,
    assignment: torch.Tensor,
    config: GaussianConfig,
) -> torch.Tensor:
    estimate = _fit_gaussian_mean(batch.train, assignment, config)
    return (batch.validation - estimate[:, None, :]).square().mean(dim=(-1, -2))


def _repeat_for_restarts(batch: GaussianBatch, restarts: int) -> GaussianBatch:
    """View each independent Gaussian trial once for every random restart."""
    if restarts == 1:
        return batch

    def repeat(value: torch.Tensor) -> torch.Tensor:
        return value[:, None].expand(-1, restarts, *value.shape[1:]).reshape(-1, *value.shape[1:])

    return GaussianBatch(
        train=repeat(batch.train),
        validation=repeat(batch.validation),
        full=repeat(batch.full),
        theta_gt=repeat(batch.theta_gt),
        assignment_gt=repeat(batch.assignment_gt),
    )


def _build_optimizer(parameters: list[nn.Parameter], config: GaussianConfig) -> torch.optim.Optimizer:
    if config.optimizer == "adam":
        return torch.optim.Adam(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    return torch.optim.RMSprop(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)


def _seeded_latents(config: GaussianConfig, device: torch.device) -> nn.Parameter:
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 17)
    return nn.Parameter(torch.randn(config.runs * config.restarts, config.latent_dim, generator=generator).to(device))


def _fit_relaxed_assignment(
    method: Literal["direct", "generated"],
    batch: GaussianBatch,
    config: GaussianConfig,
) -> tuple[torch.Tensor, dict[str, list[float]], dict[str, torch.Tensor]]:
    """Optimize a soft assignment using the paper's analytic lower estimator."""
    device = batch.train.device
    candidates = _repeat_for_restarts(batch, config.restarts)
    candidate_count = config.runs * config.restarts
    if method == "direct":
        model: nn.Module = DirectAssignment(
            config.dimensions, batch_size=candidate_count, seed=config.seed + 1
        ).to(device)
        latents: nn.Parameter | None = None
        parameters = list(model.parameters())

        def current_assignment() -> torch.Tensor:
            return model.assignment(temperature=config.temperature, mode="soft")  # type: ignore[union-attr]

    elif method == "generated":
        model = CoordinateAssignmentGenerator(
            config.dimensions,
            latent_dim=config.latent_dim,
            width=config.generator_width,
            seed=config.seed + 2,
        ).to(device)
        latents = _seeded_latents(config, device)
        parameters = list(model.parameters()) + [latents]

        def current_assignment() -> torch.Tensor:
            return model.assignment(latents, temperature=config.temperature, mode="soft")  # type: ignore[union-attr]

    else:  # pragma: no cover - retained as an explicit public error boundary
        raise ValueError(f"unknown method: {method}")

    optimizer = _build_optimizer(parameters, config)
    history: dict[str, list[float]] = {"epoch": [], "validation_mse": [], "entropy": [], "nuclear": []}
    for epoch in range(1, config.epochs + 1):
        assignment = current_assignment()
        validation = _validation_loss(candidates, assignment, config)
        regularizers = (
            release_assignment_regularizers
            if config.lower_solver == "release_normalized"
            else assignment_regularizers
        )
        entropy, nuclear = regularizers(assignment)
        per_candidate_objective = (
            validation
            + config.entropy_weight * entropy
            + config.nuclear_weight * nuclear
        )
        # Direct logits are completely task-local, so summation reproduces
        # independent optimization (including weight-decay scale).  For the
        # generated model G is shared and therefore uses a mean; its latents
        # remain task-local and have their gradient rescaled below.
        objective = (
            per_candidate_objective.sum()
            if method == "direct"
            else per_candidate_objective.mean()
        )
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        if method == "generated":
            assert latents is not None and latents.grad is not None
            latents.grad.mul_(candidate_count)
        optimizer.step()
        if epoch == 1 or epoch % config.log_every == 0 or epoch == config.epochs:
            history["epoch"].append(float(epoch))
            history["validation_mse"].append(float(validation.detach().mean()))
            history["entropy"].append(float(entropy.detach().mean()))
            history["nuclear"].append(float(nuclear.detach().mean()))

    with torch.no_grad():
        soft = current_assignment()
        hard = torch.nn.functional.one_hot(soft.argmax(dim=-1), num_classes=config.dimensions).to(soft.dtype)
        # The released experiment selects the restart using the final relaxed
        # upper objective, then hardens and refits that selected assignment.
        relaxed_validation = _validation_loss(candidates, soft, config)
        regularizers = (
            release_assignment_regularizers
            if config.lower_solver == "release_normalized"
            else assignment_regularizers
        )
        relaxed_entropy, relaxed_group = regularizers(soft)
        relaxed_objective = (
            relaxed_validation
            + config.entropy_weight * relaxed_entropy
            + config.nuclear_weight * relaxed_group
        ).reshape(config.runs, config.restarts)
        hard_validation = _validation_loss(candidates, hard, config).reshape(config.runs, config.restarts)
        selected_restart = relaxed_objective.argmin(dim=1)
        grouped_hard = hard.reshape(config.runs, config.restarts, config.dimensions, config.dimensions)
        selected = grouped_hard[torch.arange(config.runs, device=device), selected_restart]
    state: dict[str, torch.Tensor] = {
        "soft_assignment_candidates": soft.detach().cpu(),
        "hard_assignment_candidates": hard.detach().cpu(),
        "hard_validation_mse_candidates": hard_validation.detach().cpu(),
        "relaxed_objective_candidates": relaxed_objective.detach().cpu(),
        "selected_restart": selected_restart.detach().cpu(),
        "hard_assignment": selected.detach().cpu(),
    }
    if latents is not None:
        state["latents"] = latents.detach().cpu()
    for name, value in model.state_dict().items():
        state[f"model.{name}"] = value.detach().cpu()
    return selected, history, state


def _metrics_for_assignment(
    assignment: torch.Tensor,
    batch: GaussianBatch,
    *,
    config: GaussianConfig,
) -> dict[str, Any]:
    estimate = _fit_gaussian_mean(batch.full, assignment, config)
    squared = (estimate - batch.theta_gt).square()
    per_run_sum = squared.sum(dim=-1)
    per_run_mean = squared.mean(dim=-1)
    pd = [partition_distance(assignment[index], batch.assignment_gt[index]) for index in range(assignment.shape[0])]
    return {
        "mse_sum_per_run": per_run_sum.detach().cpu(),
        "mse_per_dimension_per_run": per_run_mean.detach().cpu(),
        "partition_distance_per_run": torch.tensor(pd, dtype=torch.float32),
        "estimate": estimate.detach().cpu(),
        "assignment": assignment.detach().cpu(),
    }


def _summarize(metrics: dict[str, Any]) -> dict[str, float]:
    def stat(key: str) -> tuple[float, float, float]:
        value = metrics[key].to(torch.float64)
        mean = float(value.mean())
        std = float(value.std(unbiased=True)) if value.numel() > 1 else 0.0
        ci95 = 1.96 * std / math.sqrt(value.numel())
        return mean, std, ci95

    mse_sum, mse_sum_std, mse_sum_ci = stat("mse_sum_per_run")
    mse, mse_std, mse_ci = stat("mse_per_dimension_per_run")
    pd, pd_std, pd_ci = stat("partition_distance_per_run")
    return {
        "mse_sum_mean": mse_sum,
        "mse_sum_std": mse_sum_std,
        "mse_sum_ci95": mse_sum_ci,
        "mse_per_dimension_mean": mse,
        "mse_per_dimension_std": mse_std,
        "mse_per_dimension_ci95": mse_ci,
        "partition_distance_mean": pd,
        "partition_distance_std": pd_std,
        "partition_distance_ci95": pd_ci,
    }


def run_gaussian_benchmark(config: GaussianConfig, device: torch.device | str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run no-sharing, oracle, direct-A, and generated-A on batched trials.

    Returns a small JSON-safe summary and tensor artifacts suitable for a
    ``.pt`` file.  The selected direct/generated partition is hardened and
    refit on all 100 samples, exactly as in Yeh et al.'s final stage.
    """
    device = torch.device(device)
    batch = make_gaussian_batch(config, device)
    identity = torch.eye(config.dimensions, device=device).expand(config.runs, -1, -1)
    methods: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, Any] = {
        "config": config.to_dict(),
        "theta_gt": batch.theta_gt.detach().cpu(),
        "assignment_gt": batch.assignment_gt.detach().cpu(),
        "train": batch.train.detach().cpu(),
        "validation": batch.validation.detach().cpu(),
    }

    for name, assignment in (("no_sharing", identity), ("oracle", batch.assignment_gt)):
        metrics = _metrics_for_assignment(assignment, batch, config=config)
        methods[name] = _summarize(metrics)
        artifacts[name] = metrics

    for name in ("direct", "generated"):
        assignment, history, state = _fit_relaxed_assignment(name, batch, config)
        metrics = _metrics_for_assignment(assignment, batch, config=config)
        selected_indices = state["selected_restart"].to(torch.long)
        hard_candidates = state["hard_validation_mse_candidates"]
        selected_validation = hard_candidates[
            torch.arange(config.runs), selected_indices
        ]
        methods[name] = _summarize(metrics) | {
            "selected_hard_validation_mse_mean": float(selected_validation.mean()),
            "history": history,
        }
        artifacts[name] = metrics | state

    summary: dict[str, Any] = {
        "protocol": {
            "source": "Yeh et al. (2022), Gaussian data with shared means",
            "selection": f"relaxed A selected on validation through {config.lower_solver}",
            "final_evaluation": "for each run select restart by final relaxed upper objective, harden A, and refit on train+validation",
            "generated_parameterization": "one coordinate-conditioned G_psi shared transductively across all evaluated runs with one z per run",
            "comparison_scope": "direct A is task-local; generated A shares G_psi across the evaluated Monte Carlo tasks, so generated results are transductive rather than held-out transfer",
        },
        "config": config.to_dict(),
        "methods": methods,
    }
    return summary, artifacts


def markdown_report(summary: dict[str, Any]) -> str:
    """Render a compact standalone report for the CLI output directory."""
    config = summary["config"]
    lines = [
        "# Yeh et al. (2022): generated assignment matrices — Gaussian benchmark",
        "",
        "Batched synthetic shared-means protocol. `direct` optimizes one relaxed assignment per run; "
        "`generated` optimizes one coordinate-conditioned generator shared across runs and task latents. "
        "`direct` has independent task-local restarts; `generated` has independent latent restarts but one shared generator.",
        "",
        f"- runs: {config['runs']}; dimensions: {config['dimensions']}; true rank: {config['true_rank']}",
        f"- samples: {config['num_train']} train / {config['num_samples'] - config['num_train']} validation; "
        f"outer epochs: {config['epochs']}; restarts: {config['restarts']}",
        "- each learned run selects a restart by the final relaxed upper objective, hardens, then refits on all samples; "
        "`MSE sum` follows the paper's convention.",
        "",
        "| method | MSE sum (mean ± 95% CI) | MSE / dimension | partition distance |",
        "|---|---:|---:|---:|",
    ]
    for name, row in summary["methods"].items():
        lines.append(
            f"| {name} | {row['mse_sum_mean']:.5f} ± {row['mse_sum_ci95']:.5f} | "
            f"{row['mse_per_dimension_mean']:.5f} ± {row['mse_per_dimension_ci95']:.5f} | "
            f"{row['partition_distance_mean']:.3f} ± {row['partition_distance_ci95']:.3f} |"
        )
    return "\n".join(lines) + "\n"
