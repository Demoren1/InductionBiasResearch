"""One generated sharing prior trained jointly across several linear tasks."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Literal, Sequence

import torch
import torch.nn.functional as F

from .core import CoordinateAssignmentGenerator, assignment_regularizers, partition_distance
from .linear import LinearExperimentConfig, dense_ols, labels_to_assignment, matrix_coordinates
from .tasks import Benchmark, Split


def _combine(first: Split, second: Split) -> Split:
    return Split(torch.cat((first.x, second.x)), torch.cat((first.y, second.y)))


def _project(dense: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
    """Batched Appendix F.3 projection, shapes [T,O,I] and [T,P,P]."""
    values = torch.linalg.pinv(assignment) @ dense.flatten(1).unsqueeze(-1)
    return (assignment @ values).squeeze(-1).reshape_as(dense)


def _exact_lower(
    x: torch.Tensor,
    y: torch.Tensor,
    assignment: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    """Batched exact constrained regression under ``vec(W)=A v``."""
    tasks, _, input_dim = x.shape
    output_dim = y.shape[-1]
    structured = assignment.reshape(tasks, output_dim, input_dim, -1)
    design = torch.einsum("tni,toic->tnoc", x, structured).flatten(1, 2)
    target = y.flatten(1).unsqueeze(-1)
    if ridge <= 0:
        values = torch.linalg.pinv(design) @ target
    else:
        gram = design.transpose(-2, -1) @ design
        identity = torch.eye(gram.shape[-1], device=x.device, dtype=x.dtype)
        values = torch.linalg.solve(
            gram + ridge * identity,
            design.transpose(-2, -1) @ target,
        )
    return (assignment @ values).squeeze(-1).reshape(tasks, output_dim, input_dim)


def _fit_lower(
    x: torch.Tensor,
    y: torch.Tensor,
    dense: torch.Tensor,
    assignment: torch.Tensor,
    config: LinearExperimentConfig,
) -> torch.Tensor:
    if config.lower_solver == "paper_projection":
        return _project(dense, assignment)
    return _exact_lower(x, y, assignment, config.ridge)


def _mse(x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    prediction = torch.einsum("tni,toi->tno", x, weight)
    return (prediction - y).square().mean(dim=(-1, -2))


def _ci(values: list[float]) -> tuple[float, float]:
    return mean(values), 0.0 if len(values) == 1 else 1.96 * stdev(values) / math.sqrt(len(values))


def run_multitask_generated_linear(
    benchmarks: Sequence[Benchmark],
    config: LinearExperimentConfig,
    output: Path,
    *,
    shared_latent: bool = False,
    assignment_mode: Literal["soft", "ste"] = "soft",
) -> dict[str, Any]:
    """Train one ``G_psi`` and task-local latents on a collection of tasks."""
    if not benchmarks:
        raise ValueError("at least one task is required")
    device = benchmarks[0].splits.train.x.device
    input_dim = int(benchmarks[0].splits.train.x.shape[-1])
    output_dim = int(benchmarks[0].splits.train.y.shape[-1])
    for benchmark in benchmarks:
        if benchmark.splits.train.x.device != device:
            raise ValueError("all tasks must be on one device")
        if benchmark.splits.train.x.shape[-1] != input_dim or benchmark.splits.train.y.shape[-1] != output_dim:
            raise ValueError("all tasks must have the same linear dimensions")
    n_tasks = len(benchmarks)
    n_items = input_dim * output_dim
    coordinates = matrix_coordinates(output_dim, input_dim, device)
    generator = CoordinateAssignmentGenerator(
        n_items,
        n_categories=n_items,
        latent_dim=config.latent_dim,
        width=config.generator_width,
        seed=config.seed,
        item_coordinates=coordinates,
    ).to(device)
    latent_rng = torch.Generator(device="cpu").manual_seed(config.seed + 70_001)
    latent_count = 1 if shared_latent else n_tasks
    latents = torch.nn.Parameter(
        0.1 * torch.randn(latent_count, config.latent_dim, generator=latent_rng, device="cpu").to(device)
    )
    optimizer = torch.optim.Adam(
        list(generator.parameters()) + [latents],
        lr=config.outer_lr,
        weight_decay=config.weight_decay,
    )
    train_dense = torch.stack([dense_ols(task.splits.train, config.ridge) for task in benchmarks])
    train_x = torch.stack([task.splits.train.x for task in benchmarks])
    train_y = torch.stack([task.splits.train.y for task in benchmarks])
    validation_x = torch.stack([task.splits.validation.x for task in benchmarks])
    validation_y = torch.stack([task.splits.validation.y for task in benchmarks])
    history: list[dict[str, float | int]] = []
    best_score = float("inf")
    best_step = 0
    best_assignment: torch.Tensor | None = None
    for step in range(config.outer_steps):
        fraction = 0.0 if config.outer_steps == 1 else step / (config.outer_steps - 1)
        temperature = config.temperature_start * (
            config.temperature_end / config.temperature_start
        ) ** fraction
        task_latents = latents.expand(n_tasks, -1) if shared_latent else latents
        relaxed = generator.assignment(task_latents, temperature=temperature, mode=assignment_mode)
        weight = _fit_lower(train_x, train_y, train_dense, relaxed, config)
        validation = _mse(validation_x, validation_y, weight)
        if config.entropy_weight or config.nuclear_weight:
            penalty_assignment = (
                generator.assignment(task_latents, temperature=temperature, mode="soft")
                if assignment_mode == "ste"
                else relaxed
            )
            entropy, nuclear = assignment_regularizers(penalty_assignment)
        else:
            entropy = torch.zeros_like(validation)
            nuclear = torch.zeros_like(validation)
        objective = (
            validation
            + config.entropy_weight * entropy
            + config.nuclear_weight * nuclear
        ).mean()
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        # G is shared (mean gradient); each z is task-local (independent scale).
        if not shared_latent and latents.grad is not None:
            latents.grad.mul_(n_tasks)
        optimizer.step()
        if step % config.checkpoint_every == 0 or step + 1 == config.outer_steps:
            with torch.no_grad():
                task_latents = latents.expand(n_tasks, -1) if shared_latent else latents
                hard = generator.assignment(task_latents, temperature=temperature, mode="hard")
                hard_weight = _fit_lower(train_x, train_y, train_dense, hard, config)
                hard_score = float(_mse(validation_x, validation_y, hard_weight).mean())
            history.append(
                {
                    "step": step,
                    "temperature": temperature,
                    "relaxed_validation_mse": float(validation.mean().detach()),
                    "hard_validation_mse": hard_score,
                }
            )
            if hard_score < best_score:
                best_score = hard_score
                best_step = step
                best_assignment = hard.detach().cpu().clone()
    if best_assignment is None:
        raise RuntimeError("training produced no checkpoint")
    learned = best_assignment.to(device)

    method_rows: dict[str, list[dict[str, float | int]]] = {
        "no_sharing": [],
        "oracle": [],
        "generated": [],
    }
    identity = torch.eye(n_items, device=device)
    for index, task in enumerate(benchmarks):
        combined = _combine(task.splits.train, task.splits.validation)
        dense = dense_ols(combined, config.ridge)
        assignments = {
            "no_sharing": identity,
            "oracle": labels_to_assignment(task.oracle_categories.to(device), n_categories=n_items),
            "generated": learned[index],
        }
        for method, assignment in assignments.items():
            weight = _fit_lower(
                combined.x.unsqueeze(0),
                combined.y.unsqueeze(0),
                dense.unsqueeze(0),
                assignment.unsqueeze(0),
                config,
            )[0]
            mse = float(((task.splits.test.x @ weight.T - task.splits.test.y).square().mean()).detach())
            pd = partition_distance(assignment.detach(), task.oracle_categories.reshape(-1).detach())
            method_rows[method].append(
                {"task_index": index, "mse": mse, "partition_distance": pd}
            )

    methods: dict[str, dict[str, Any]] = {}
    for method, rows in method_rows.items():
        mse_mean, mse_ci = _ci([float(row["mse"]) for row in rows])
        pd_mean, pd_ci = _ci([float(row["partition_distance"]) for row in rows])
        methods[method] = {
            "mse_mean": mse_mean,
            "mse_ci95": mse_ci,
            "partition_distance_mean": pd_mean,
            "partition_distance_ci95": pd_ci,
            "per_task": rows,
        }
    summary: dict[str, Any] = {
        "benchmark": benchmarks[0].name,
        "benchmark_metadata": benchmarks[0].metadata,
        "protocol": {
            "generated_parameterization": (
                "one G_psi and one z shared across all tasks"
                if shared_latent
                else "one G_psi shared across tasks, one optimized z per task"
            ),
            "lower_solver": config.lower_solver,
            "selection": "lowest mean hard validation MSE; test used once after train+validation refit",
        },
        "config": asdict(config),
        "task_count": n_tasks,
        "latent_mode": "global" if shared_latent else "per_task",
        "assignment_mode": assignment_mode,
        "best_step": best_step,
        "best_hard_validation_mse": best_score,
        "methods": methods,
        "history": history,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "multitask_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save(
        {
            "generator": generator.state_dict(),
            "latents": latents.detach().cpu(),
            "hard_assignments": best_assignment,
        },
        output / "multitask_training.pt",
    )
    lines = [
        f"# Multi-task generated sharing: {benchmarks[0].name}",
        "",
        "| method | test MSE | PD |",
        "|---|---:|---:|",
    ]
    for method in ("no_sharing", "oracle", "generated"):
        row = methods[method]
        lines.append(
            f"| {method} | {row['mse_mean']:.6g} ± {row['mse_ci95']:.2g} | "
            f"{row['partition_distance_mean']:.4g} ± {row['partition_distance_ci95']:.2g} |"
        )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return summary
