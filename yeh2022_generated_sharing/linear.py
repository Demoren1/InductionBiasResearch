"""Bilevel generated-sharing experiments for the linear Yeh benchmarks.

This module covers Sec. 5.3 (cross-correlation) and Appendix D.1
(unit-step denoising).  The lower problem is solved analytically.  The
paper's dense-then-project solve and an exact constrained solve are both
available; result files record which one was used.  The outer parameterization is:

* ``direct`` optimizes free assignment logits, following Yeh et al.;
* ``generated`` uses ``A = G_psi(z)``;
* ``no_sharing`` and ``oracle`` are fixed controls.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F

from .core import (
    CoordinateAssignmentGenerator,
    DirectAssignment,
    assignment_regularizers,
    partition_distance,
)
from .tasks import Benchmark, Split


Method = Literal["direct", "generated"]
LowerSolver = Literal["paper_projection", "exact_constrained"]


@dataclass(frozen=True, slots=True)
class LinearExperimentConfig:
    """Optimization settings shared by cross-correlation and denoising."""

    seed: int = 0
    outer_steps: int = 1_000
    outer_lr: float = 0.1
    weight_decay: float = 1e-4
    latent_dim: int = 8
    generator_width: int = 64
    temperature_start: float = 1.0
    temperature_end: float = 1.0
    entropy_weight: float = 0.01
    nuclear_weight: float = 0.01
    ridge: float = 0.0
    lower_solver: LowerSolver = "paper_projection"
    checkpoint_every: int = 10
    patience: int = 1_000
    restarts: int = 3
    min_delta: float = 1e-7

    def __post_init__(self) -> None:
        positive_ints = (
            self.outer_steps,
            self.latent_dim,
            self.generator_width,
            self.checkpoint_every,
            self.patience,
            self.restarts,
        )
        if any(value <= 0 for value in positive_ints):
            raise ValueError("step, width, latent, checkpoint, and patience values must be positive")
        if min(self.outer_lr, self.temperature_start, self.temperature_end) <= 0:
            raise ValueError("learning rate and temperatures must be positive")
        if min(self.weight_decay, self.entropy_weight, self.nuclear_weight, self.ridge) < 0:
            raise ValueError("regularization values must be non-negative")
        if self.lower_solver not in {"paper_projection", "exact_constrained"}:
            raise ValueError("unknown lower solver")


def labels_to_assignment(
    labels: torch.Tensor,
    *,
    n_categories: int | None = None,
) -> torch.Tensor:
    """Convert non-negative partition labels to one-hot assignment rows.

    Label zero is an ordinary learned sharing group, as in Yeh et al.'s
    ``Flatten(G) = A psi`` formulation; it is not a fixed-zero connection.
    """
    if labels.ndim != 1:
        labels = labels.reshape(-1)
    labels = labels.to(torch.long)
    if labels.numel() == 0 or int(labels.min()) < 0:
        raise ValueError("labels must be a non-empty vector of non-negative integers")
    needed = int(labels.max()) + 1
    categories = needed if n_categories is None else n_categories
    if categories < needed:
        raise ValueError("n_categories cannot omit a label")
    return F.one_hot(labels, num_classes=categories).to(torch.float32)


def matrix_coordinates(output_dim: int, input_dim: int, device: torch.device) -> torch.Tensor:
    """Coordinates in the same row-major order as ``weight.reshape(-1)``."""
    output = torch.linspace(-1.0, 1.0, output_dim, device=device)
    inputs = torch.linspace(-1.0, 1.0, input_dim, device=device)
    output_grid, input_grid = torch.meshgrid(output, inputs, indexing="ij")
    return torch.stack((output_grid, input_grid), dim=-1).reshape(-1, 2)


def dense_ols(split: Split, ridge: float) -> torch.Tensor:
    """Return the dense linear map with shape ``[output, input]``."""
    x, y = split.x, split.y
    if y.ndim == 1:
        y = y[:, None]
    gram = x.T @ x
    identity = torch.eye(gram.shape[0], dtype=x.dtype, device=x.device)
    coefficients = torch.linalg.solve(gram + ridge * identity, x.T @ y)
    return coefficients.T


def fit_shared_linear(split: Split, assignment: torch.Tensor, ridge: float) -> torch.Tensor:
    """Solve the exact lower problem under ``vec(W)=A v``.

    For every sample/output pair we build its induced design vector and solve
    the resulting least-squares problem directly.  This differs from the
    paper's Euclidean projection of an unconstrained dense optimum for finite,
    non-whitened inputs.
    """
    x, y = split.x, split.y
    if y.ndim == 1:
        y = y[:, None]
    output_dim, input_dim = y.shape[-1], x.shape[-1]
    if assignment.shape[0] != output_dim * input_dim or assignment.shape[1] < 1:
        raise ValueError("assignment must cover every weight")
    structured = assignment.reshape(output_dim, input_dim, -1)
    design = torch.einsum("ni,oic->noc", x, structured).reshape(-1, structured.shape[-1])
    target = y.reshape(-1, 1)
    if ridge <= 0:
        # Unused sharing groups make columns of the design exactly zero.
        # The default CPU least-squares driver assumes full rank and may
        # return different, non-minimal solutions for that case.
        values = torch.linalg.pinv(design) @ target
    else:
        gram = design.T @ design
        identity = torch.eye(gram.shape[0], dtype=design.dtype, device=design.device)
        values = torch.linalg.solve(gram + ridge * identity, design.T @ target)
    return (assignment @ values).reshape(output_dim, input_dim)


def fit_paper_projection(split: Split, assignment: torch.Tensor, ridge: float) -> torch.Tensor:
    """Appendix F.3: dense OLS followed by ``A A^+ vec(G*)``."""
    dense = dense_ols(split, ridge)
    values = torch.linalg.pinv(assignment) @ dense.reshape(-1, 1)
    return (assignment @ values).reshape_as(dense)


def _fit_lower(
    split: Split,
    assignment: torch.Tensor,
    config: LinearExperimentConfig,
) -> torch.Tensor:
    if config.lower_solver == "paper_projection":
        return fit_paper_projection(split, assignment, config.ridge)
    return fit_shared_linear(split, assignment, config.ridge)


def prediction_mse(split: Split, weight: torch.Tensor) -> torch.Tensor:
    prediction = split.x @ weight.T
    target = split.y[:, None] if split.y.ndim == 1 else split.y
    return (prediction - target).square().mean()


def _combined(first: Split, second: Split) -> Split:
    return Split(torch.cat((first.x, second.x)), torch.cat((first.y, second.y)))


def _temperature(config: LinearExperimentConfig, step: int) -> float:
    if config.outer_steps == 1:
        return config.temperature_end
    fraction = step / (config.outer_steps - 1)
    return config.temperature_start * (config.temperature_end / config.temperature_start) ** fraction


def _evaluate_assignment(
    benchmark: Benchmark,
    assignment: torch.Tensor,
    config: LinearExperimentConfig,
    *,
    refit_on_all: bool,
) -> dict[str, float | int]:
    fit_split = (
        _combined(benchmark.splits.train, benchmark.splits.validation)
        if refit_on_all
        else benchmark.splits.train
    )
    weight = _fit_lower(fit_split, assignment, config)
    evaluation_split = benchmark.splits.test if refit_on_all else benchmark.splits.validation
    result: dict[str, float | int] = {
        "mse": float(prediction_mse(evaluation_split, weight).detach()),
        "partition_distance": partition_distance(
            assignment.detach(), benchmark.oracle_categories.reshape(-1).detach()
        ),
    }
    result["normalized_partition_distance"] = result["partition_distance"] / assignment.shape[0]
    if benchmark.oracle_weight is not None:
        result["weight_mse"] = float((weight - benchmark.oracle_weight).square().mean().detach())
    return result


def _train_assignment(
    benchmark: Benchmark,
    method: Method,
    config: LinearExperimentConfig,
) -> tuple[torch.Tensor, list[dict[str, float | int]], dict[str, Any]]:
    device = benchmark.splits.train.x.device
    output_dim = int(benchmark.splits.train.y.shape[-1]) if benchmark.splits.train.y.ndim > 1 else 1
    input_dim = int(benchmark.splits.train.x.shape[-1])
    n_items = output_dim * input_dim
    if method == "direct":
        model: torch.nn.Module = DirectAssignment(
            n_items, n_categories=n_items, seed=config.seed
        ).to(device)
        latent = None
        parameters = list(model.parameters())
    else:
        coordinates = matrix_coordinates(output_dim, input_dim, device)
        model = CoordinateAssignmentGenerator(
            n_items,
            n_categories=n_items,
            latent_dim=config.latent_dim,
            width=config.generator_width,
            seed=config.seed,
            item_coordinates=coordinates,
        ).to(device)
        latent_generator = torch.Generator(device="cpu").manual_seed(config.seed + 71_003)
        latent = torch.nn.Parameter(
            torch.randn(config.latent_dim, generator=latent_generator, device="cpu").to(device) * 0.1
        )
        parameters = list(model.parameters()) + [latent]
    optimizer = torch.optim.Adam(parameters, lr=config.outer_lr, weight_decay=config.weight_decay)
    history: list[dict[str, float | int]] = []
    best_validation = float("inf")
    best_step = 0
    best_assignment: torch.Tensor | None = None
    stale = 0

    def assignment(mode: str, temperature: float) -> torch.Tensor:
        if method == "direct":
            return model.assignment(temperature=temperature, mode=mode).squeeze(0)  # type: ignore[attr-defined]
        assert latent is not None
        return model.assignment(latent, temperature=temperature, mode=mode)  # type: ignore[attr-defined]

    for step in range(config.outer_steps):
        temperature = _temperature(config, step)
        soft = assignment("soft", temperature)
        fitted = _fit_lower(benchmark.splits.train, soft, config)
        validation = prediction_mse(benchmark.splits.validation, fitted)
        entropy, nuclear = assignment_regularizers(soft)
        objective = (
            validation
            + config.entropy_weight * entropy
            + config.nuclear_weight * nuclear
        )
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()

        if step % config.checkpoint_every == 0 or step + 1 == config.outer_steps:
            hard = assignment("hard", temperature).detach()
            metrics = _evaluate_assignment(
                benchmark, hard, config, refit_on_all=False
            )
            row: dict[str, float | int] = {
                "step": step,
                "temperature": temperature,
                "soft_validation_mse": float(validation.detach()),
                "hard_validation_mse": float(metrics["mse"]),
                "partition_distance": int(metrics["partition_distance"]),
            }
            history.append(row)
            current = float(metrics["mse"])
            if current < best_validation - config.min_delta:
                best_validation = current
                best_step = step
                best_assignment = hard.cpu().clone()
                stale = 0
            else:
                stale += 1
                if stale >= config.patience:
                    break

    if best_assignment is None:
        raise RuntimeError("training did not produce a hard checkpoint")
    state = {
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "latent": None if latent is None else latent.detach().cpu(),
        "best_step": best_step,
    }
    return best_assignment.to(device), history, state


def run_linear_experiment(
    benchmark: Benchmark,
    config: LinearExperimentConfig,
    output: Path,
) -> dict[str, Any]:
    """Run all four controls and persist a self-contained result directory."""
    output.mkdir(parents=True, exist_ok=False)
    n_items = int(benchmark.oracle_categories.numel())
    device = benchmark.splits.train.x.device
    no_sharing = torch.eye(n_items, device=device)
    oracle = labels_to_assignment(
        benchmark.oracle_categories.to(device), n_categories=n_items
    )
    results: dict[str, dict[str, float | int]] = {
        "no_sharing": _evaluate_assignment(benchmark, no_sharing, config, refit_on_all=True),
        "oracle": _evaluate_assignment(benchmark, oracle, config, refit_on_all=True),
    }
    histories: dict[str, list[dict[str, float | int]]] = {}
    states: dict[str, Any] = {}
    learned_assignments: dict[str, torch.Tensor] = {}
    for method in ("direct", "generated"):
        candidates: list[tuple[float, torch.Tensor, list[dict[str, float | int]], dict[str, Any]]] = []
        restart_states: list[dict[str, Any]] = []
        for restart in range(config.restarts):
            restart_config = replace(config, seed=config.seed + 100_003 * restart)
            candidate, history, state = _train_assignment(benchmark, method, restart_config)
            validation = _evaluate_assignment(
                benchmark, candidate, config, refit_on_all=False
            )
            candidates.append((float(validation["mse"]), candidate, history, state))
            restart_states.append(state)
        selected_restart = min(range(config.restarts), key=lambda index: candidates[index][0])
        _, learned, history, selected_state = candidates[selected_restart]
        results[method] = _evaluate_assignment(benchmark, learned, config, refit_on_all=True)
        histories[method] = history
        states[method] = {
            "selected_restart": selected_restart,
            "selected_state": selected_state,
            "restart_states": restart_states,
            "hard_validation_mse": [candidate[0] for candidate in candidates],
        }
        learned_assignments[method] = learned.cpu()

    summary: dict[str, Any] = {
        "benchmark": benchmark.name,
        "benchmark_metadata": benchmark.metadata,
        "config": asdict(config),
        "dimensions": {
            "input": int(benchmark.splits.train.x.shape[-1]),
            "output": int(benchmark.splits.train.y.shape[-1])
            if benchmark.splits.train.y.ndim > 1
            else 1,
            "assignment_items": n_items,
        },
        "methods": results,
        "history": histories,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save(
        {
            "states": states,
            "assignments": learned_assignments,
            "oracle_categories": benchmark.oracle_categories.cpu(),
        },
        output / "training.pt",
    )
    lines = [
        f"# {benchmark.name}",
        "",
        "| Method | Test MSE | PD | normalized PD |",
        "|---|---:|---:|---:|",
    ]
    for method in ("no_sharing", "oracle", "direct", "generated"):
        metrics = results[method]
        lines.append(
            f"| {method} | {metrics['mse']:.6g} | {metrics['partition_distance']} | "
            f"{metrics['normalized_partition_distance']:.4f} |"
        )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return summary


__all__ = [
    "LinearExperimentConfig",
    "dense_ols",
    "fit_paper_projection",
    "fit_shared_linear",
    "labels_to_assignment",
    "matrix_coordinates",
    "prediction_mse",
    "run_linear_experiment",
]
