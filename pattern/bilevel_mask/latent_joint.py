"""Vectorized joint learning of a latent structure generator and masked MLPs.

For each task and latent restart, MLP weights and a latent code are persistent.
Weights are fitted on a support split, latents are optimized to convergence on
a disjoint validation split, and the shared generator is updated on a third
query split.  Every task/restart MLP is evaluated in one tensor operation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from pattern import config as pattern_config
from pattern.data.generate import ideal_mask
from pattern.evaluation.decoder_agreement import align_columns, hard_topk, soft_topk

from .core import TEST_PATTERNS, TRAIN_PATTERNS, VALIDATION_PATTERNS, _all_sequences, _seed


@dataclass(frozen=True)
class LatentJointConfig:
    seed: int = 42
    latent_dim: int = 8
    generator_width: int = 32
    restarts: int = 64
    outer_steps: int = 100
    weight_steps: int = 20
    eval_weight_steps: int = 1000
    eval_rounds: int = 4
    z_max_steps: int = 200
    z_patience: int = 15
    z_min_delta: float = 1e-5
    weight_lr: float = 0.03
    z_lr: float = 0.03
    generator_lr: float = 0.003
    z_radius: float = 4.0
    temperature_start: float = 1.0
    temperature_end: float = 0.15
    binary_penalty: float = 0.01
    support_positive: int = 1024
    support_negative: int = 1024
    validation_positive: int = 512
    validation_negative: int = 512
    query_positive: int = 512
    query_negative: int = 512
    k_active: int = pattern_config.K_ACTIVE
    relaxation: str = "ste"
    exact_population_loss: bool = False
    sample_with_replacement: bool = True

    def __post_init__(self) -> None:
        if self.relaxation not in {"soft", "ste"}:
            raise ValueError("relaxation must be soft or ste")
        integer_names = (
            "latent_dim", "generator_width", "restarts", "outer_steps", "weight_steps",
            "eval_weight_steps", "eval_rounds", "z_max_steps", "z_patience",
            "support_positive", "support_negative", "validation_positive",
            "validation_negative", "query_positive", "query_negative", "k_active",
        )
        if any(getattr(self, name) <= 0 for name in integer_names):
            raise ValueError("integer configuration values must be positive")
        if any(getattr(self, name) <= 0 for name in (
            "weight_lr", "z_lr", "generator_lr", "z_radius",
            "temperature_start", "temperature_end",
        )):
            raise ValueError("learning rates, radius, and temperatures must be positive")
        if self.z_min_delta < 0 or self.binary_penalty < 0:
            raise ValueError("z_min_delta and binary_penalty must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LatentMaskGenerator(nn.Module):
    """Small shared map from a task-specific latent to 64 structure logits."""

    def __init__(self, latent_dim: int, width: int, seed: int) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.network = nn.Sequential(
                nn.Linear(latent_dim, width), nn.Tanh(),
                nn.Linear(width, width), nn.Tanh(),
                nn.Linear(width, pattern_config.MASK_DIM),
            )
            nn.init.normal_(self.network[-1].weight, std=0.02)
            nn.init.zeros_(self.network[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


class BatchedWeights(nn.Module):
    """Weights for T x R independent masked MLPs."""

    def __init__(self, tasks: int, restarts: int, seed: int, device: torch.device) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        shape = (tasks, restarts)
        self.w1 = nn.Parameter(
            (0.1 * torch.randn(*shape, pattern_config.SEQ_LEN, pattern_config.H, generator=generator)).to(device)
        )
        self.b1 = nn.Parameter(torch.zeros(*shape, pattern_config.H, device=device))
        self.w2 = nn.Parameter((0.1 * torch.randn(*shape, pattern_config.H, generator=generator)).to(device))
        self.b2 = nn.Parameter(torch.zeros(*shape, device=device))

    def detached(self) -> tuple[torch.Tensor, ...]:
        return self.w1.detach(), self.b1.detach(), self.w2.detach(), self.b2.detach()


@dataclass
class TaskSplits:
    patterns: tuple[str, ...]
    support_x: torch.Tensor
    support_y: torch.Tensor
    validation_x: torch.Tensor
    validation_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor


def _split_counts(config: LatentJointConfig) -> tuple[tuple[int, int], ...]:
    return (
        (config.support_positive, config.support_negative),
        (config.validation_positive, config.validation_negative),
        (config.query_positive, config.query_negative),
    )


def make_task_splits(
    patterns: Sequence[str], config: LatentJointConfig, device: torch.device,
) -> TaskSplits:
    """Build equal-size, disjoint, class-stratified splits for every task."""
    if config.exact_population_loss:
        populations = [_all_sequences(pattern, device) for pattern in patterns]
        x = torch.stack([population[0] for population in populations])
        y = torch.stack([population[1] for population in populations])
        # The finite toy domain can be integrated exactly.  The three tensors
        # have distinct optimization roles; no finite-sample estimate can be
        # overfit because each contains the complete input population.
        return TaskSplits(tuple(patterns), x, y, x.clone(), y.clone(), x.clone(), y.clone())

    split_x: list[list[torch.Tensor]] = [[], [], []]
    split_y: list[list[torch.Tensor]] = [[], [], []]
    for pattern in patterns:
        x, y = _all_sequences(pattern, torch.device("cpu"))
        generator = torch.Generator(device="cpu").manual_seed(_seed(config.seed, "data", pattern))
        positive = torch.nonzero(y == 1, as_tuple=False).flatten()
        negative = torch.nonzero(y == 0, as_tuple=False).flatten()
        positive = positive[torch.randperm(positive.numel(), generator=generator)]
        negative = negative[torch.randperm(negative.numel(), generator=generator)]
        positive_offset = negative_offset = 0
        for split_index, (n_positive, n_negative) in enumerate(_split_counts(config)):
            if config.sample_with_replacement:
                positive_indices = positive[torch.randint(positive.numel(), (n_positive,), generator=generator)]
                negative_indices = negative[torch.randint(negative.numel(), (n_negative,), generator=generator)]
            else:
                if positive_offset + n_positive > positive.numel() or negative_offset + n_negative > negative.numel():
                    raise ValueError(f"not enough examples to split task {pattern}")
                positive_indices = positive[positive_offset:positive_offset + n_positive]
                negative_indices = negative[negative_offset:negative_offset + n_negative]
            indices = torch.cat((positive_indices, negative_indices))
            indices = indices[torch.randperm(indices.numel(), generator=generator)]
            split_x[split_index].append(x[indices])
            split_y[split_index].append(y[indices])
            if not config.sample_with_replacement:
                positive_offset += n_positive
                negative_offset += n_negative
    tensors_x = [torch.stack(values).to(device) for values in split_x]
    tensors_y = [torch.stack(values).to(device) for values in split_y]
    return TaskSplits(tuple(patterns), tensors_x[0], tensors_y[0], tensors_x[1], tensors_y[1], tensors_x[2], tensors_y[2])


def batched_forward(
    x: torch.Tensor, weights: tuple[torch.Tensor, ...] | BatchedWeights, masks: torch.Tensor,
) -> torch.Tensor:
    """Return logits shaped (tasks, restarts, examples)."""
    if isinstance(weights, BatchedWeights):
        w1, b1, w2, b2 = weights.w1, weights.b1, weights.w2, weights.b2
    else:
        w1, b1, w2, b2 = weights
    hidden = F.relu(torch.einsum("tbi,trih->trbh", x, w1 * masks) + b1[:, :, None, :])
    return torch.einsum("trbh,trh->trb", hidden, w2) + b2[:, :, None]


def per_network_balanced_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Class-balanced BCE for each (task, restart), returning (T, R)."""
    losses = F.binary_cross_entropy_with_logits(
        logits, targets[:, None, :].expand_as(logits), reduction="none",
    )
    positive = targets[:, None, :] == 1
    negative = ~positive
    pos_count = positive.sum(-1).clamp_min(1)
    neg_count = negative.sum(-1).clamp_min(1)
    return 0.5 * (losses * positive).sum(-1) / pos_count + 0.5 * (losses * negative).sum(-1) / neg_count


def relaxed_masks(
    generator: LatentMaskGenerator, z: torch.Tensor, config: LatentJointConfig, temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = generator(z)
    soft_flat = soft_topk(logits, config.k_active, temperature)
    hard_flat = hard_topk(logits, config.k_active)
    if config.relaxation == "soft":
        forward_flat = soft_flat
    else:
        forward_flat = _HardForwardSoftBackward.apply(hard_flat, soft_flat)
    shape = (*z.shape[:-1], pattern_config.SEQ_LEN, pattern_config.H)
    return forward_flat.reshape(shape), soft_flat.reshape(shape), logits.reshape(shape)


class _HardForwardSoftBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
        return hard

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[None, torch.Tensor]:
        return None, gradient


def _project_latents(z: torch.Tensor, radius: float) -> None:
    with torch.no_grad():
        norms = z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        z.mul_(torch.clamp(radius / norms, max=1.0))


def optimize_z_to_convergence(
    generator: LatentMaskGenerator,
    z: nn.Parameter,
    weights: BatchedWeights,
    x: torch.Tensor,
    y: torch.Tensor,
    config: LatentJointConfig,
    temperature: float,
) -> dict[str, float | int]:
    """Optimize all T x R latents on the smooth validation surrogate.

    The produced structure is hard in weight/generator updates, but convergence
    cannot be defined on that piecewise-constant objective.  Latent search thus
    uses the exact-K soft relaxation and restores its best validation iterate.
    """
    requires_grad = [parameter.requires_grad for parameter in generator.parameters()]
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam([z], lr=config.z_lr)
    best_loss = math.inf
    best_z = z.detach().clone()
    stale = 0
    for step in range(1, config.z_max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = generator(z)
        masks = soft_topk(logits, config.k_active, temperature).reshape(
            *z.shape[:-1], pattern_config.SEQ_LEN, pattern_config.H,
        )
        loss = per_network_balanced_bce(batched_forward(x, weights.detached(), masks), y).mean()
        loss.backward()
        value = float(loss.detach())
        improved = value < best_loss - config.z_min_delta
        if value < best_loss:
            best_loss = value
            best_z.copy_(z.detach())
        optimizer.step()
        _project_latents(z, config.z_radius)
        stale = 0 if improved else stale + 1
        if stale >= config.z_patience:
            break
    with torch.no_grad():
        z.copy_(best_z)
    for parameter, flag in zip(generator.parameters(), requires_grad):
        parameter.requires_grad_(flag)
    return {"steps": step, "validation_bce": best_loss, "converged": int(step < config.z_max_steps)}


def _temperature(config: LatentJointConfig, step: int) -> float:
    if config.outer_steps == 1:
        return config.temperature_end
    fraction = (step - 1) / (config.outer_steps - 1)
    return config.temperature_start * (config.temperature_end / config.temperature_start) ** fraction


def _train_weights(
    weights: BatchedWeights,
    optimizer: torch.optim.Optimizer,
    masks: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int,
) -> float:
    value = math.nan
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = per_network_balanced_bce(batched_forward(x, weights, masks), y).mean()
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
    return value


def _gather_restarts(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    task_index = torch.arange(tensor.shape[0], device=tensor.device)
    return tensor[task_index, indices][:, None]


def _hard_iou(masks: torch.Tensor) -> torch.Tensor:
    gold = ideal_mask().to(masks.device, masks.dtype)
    values = []
    for mask in masks:
        aligned = align_columns(gold, mask)
        intersection = (aligned.bool() & gold.bool()).sum()
        union = (aligned.bool() | gold.bool()).sum()
        values.append(intersection.float() / union.float())
    return torch.stack(values)


def _fresh_weights(tasks: int, restarts: int, config: LatentJointConfig, device: torch.device, tag: str) -> BatchedWeights:
    return BatchedWeights(tasks, restarts, _seed(config.seed, "weights", tag), device)


def _fit_and_score_hard_masks(
    masks: torch.Tensor, data: TaskSplits, config: LatentJointConfig, tag: str,
) -> dict[str, Any]:
    tasks, restarts = masks.shape[:2]
    weights = _fresh_weights(tasks, restarts, config, masks.device, "common-eval")
    optimizer = torch.optim.Adam(weights.parameters(), lr=config.weight_lr)
    _train_weights(weights, optimizer, masks, data.support_x, data.support_y, config.eval_weight_steps)
    with torch.no_grad():
        validation = per_network_balanced_bce(batched_forward(data.validation_x, weights, masks), data.validation_y)
        best = validation.argmin(dim=1)
        chosen_masks = _gather_restarts(masks, best)
        chosen_weights = tuple(_gather_restarts(value, best) for value in weights.detached())
        query_logits = batched_forward(data.query_x, chosen_weights, chosen_masks)
        query_loss = per_network_balanced_bce(query_logits, data.query_y).squeeze(1)
        predictions = query_logits.squeeze(1) > 0
        targets = data.query_y.bool()
        positive, negative = targets, ~targets
        correct = predictions == targets
        accuracy = 0.5 * (correct & positive).sum(1).float() / positive.sum(1).clamp_min(1)
        accuracy += 0.5 * (correct & negative).sum(1).float() / negative.sum(1).clamp_min(1)
        iou = _hard_iou(chosen_masks.squeeze(1))
    return {
        "method": tag,
        "mean_query_bce": float(query_loss.mean()),
        "mean_query_balanced_accuracy": float(accuracy.mean()),
        "mean_iou_to_analytic": float(iou.mean()),
        "tasks": [
            {"pattern": pattern, "selected_restart": int(best[index]),
             "validation_bce": float(validation[index, best[index]]),
             "query_bce": float(query_loss[index]), "query_balanced_accuracy": float(accuracy[index]),
             "iou_to_analytic": float(iou[index])}
            for index, pattern in enumerate(data.patterns)
        ],
        "chosen_masks": chosen_masks.squeeze(1).detach().cpu().tolist(),
    }


def evaluate_generator(
    generator: LatentMaskGenerator,
    patterns: Sequence[str],
    config: LatentJointConfig,
    device: torch.device,
    tag: str,
) -> dict[str, Any]:
    """Adapt z/w in parallel, then fairly refit and compare hard structures."""
    data = make_task_splits(patterns, config, device)
    # Final metrics integrate over the complete finite input population.  This
    # is evaluation-only: support and latent-validation samples stay separate.
    populations = [_all_sequences(pattern, device) for pattern in patterns]
    data.query_x = torch.stack([population[0] for population in populations])
    data.query_y = torch.stack([population[1] for population in populations])
    tasks = len(patterns)
    latent_rng = torch.Generator(device="cpu").manual_seed(_seed(config.seed, "eval-z", tag))
    z = nn.Parameter(torch.randn(tasks, config.restarts, config.latent_dim, generator=latent_rng).to(device))
    _project_latents(z, config.z_radius)
    weights = _fresh_weights(tasks, config.restarts, config, device, f"latent-{tag}")
    weight_optimizer = torch.optim.Adam(weights.parameters(), lr=config.weight_lr)
    convergence = []
    for _ in range(config.eval_rounds):
        with torch.no_grad():
            masks, _, _ = relaxed_masks(generator, z, config, config.temperature_end)
        _train_weights(weights, weight_optimizer, masks, data.support_x, data.support_y, config.weight_steps)
        convergence.append(optimize_z_to_convergence(
            generator, z, weights, data.validation_x, data.validation_y, config, config.temperature_end,
        ))
    with torch.no_grad():
        generated_logits = generator(z)
        generated_masks = hard_topk(generated_logits, config.k_active).reshape(
            tasks, config.restarts, pattern_config.SEQ_LEN, pattern_config.H,
        )
    generated = _fit_and_score_hard_masks(generated_masks, data, config, "generated")

    random_rng = torch.Generator(device="cpu").manual_seed(_seed(config.seed, "random", tag))
    scores = torch.rand(tasks, config.restarts, pattern_config.MASK_DIM, generator=random_rng).to(device)
    random_masks = hard_topk(scores, config.k_active).reshape_as(generated_masks)
    random_result = _fit_and_score_hard_masks(random_masks, data, config, "random exact-K")

    analytic = ideal_mask().to(device=device, dtype=generated_masks.dtype)
    analytic_masks = analytic[None, None].expand(tasks, config.restarts, -1, -1).clone()
    analytic_result = _fit_and_score_hard_masks(analytic_masks, data, config, "analytic")
    return {
        "patterns": list(patterns), "z_convergence": convergence,
        "generated": generated, "random": random_result, "analytic": analytic_result,
    }


def train_latent_generator(
    config: LatentJointConfig, output: str | Path, device: str | torch.device = "cuda",
) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(device)
    torch.manual_seed(config.seed)
    started = time.monotonic()
    data = make_task_splits(TRAIN_PATTERNS, config, device)
    tasks = len(TRAIN_PATTERNS)
    generator = LatentMaskGenerator(config.latent_dim, config.generator_width, config.seed).to(device)
    initial_state = {name: value.detach().cpu().clone() for name, value in generator.state_dict().items()}
    latent_rng = torch.Generator(device="cpu").manual_seed(_seed(config.seed, "train-z"))
    z = nn.Parameter(torch.randn(tasks, config.restarts, config.latent_dim, generator=latent_rng).to(device))
    _project_latents(z, config.z_radius)
    weights = _fresh_weights(tasks, config.restarts, config, device, "train")
    weight_optimizer = torch.optim.Adam(weights.parameters(), lr=config.weight_lr)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=config.generator_lr)
    history: list[dict[str, Any]] = []

    for outer_step in range(1, config.outer_steps + 1):
        temperature = _temperature(config, outer_step)
        with torch.no_grad():
            masks, _, _ = relaxed_masks(generator, z, config, temperature)
        support_loss = _train_weights(
            weights, weight_optimizer, masks, data.support_x, data.support_y, config.weight_steps,
        )
        z_info = optimize_z_to_convergence(
            generator, z, weights, data.validation_x, data.validation_y, config, temperature,
        )

        generator_optimizer.zero_grad(set_to_none=True)
        all_masks, all_soft, _ = relaxed_masks(generator, z.detach(), config, temperature)
        with torch.no_grad():
            validation_losses = per_network_balanced_bce(
                batched_forward(data.validation_x, weights.detached(), all_masks.detach()), data.validation_y,
            )
            selected = validation_losses.argmin(dim=1)
        selected_masks = _gather_restarts(all_masks, selected)
        selected_soft = _gather_restarts(all_soft, selected)
        selected_weights = tuple(_gather_restarts(value, selected) for value in weights.detached())
        query_loss = per_network_balanced_bce(
            batched_forward(data.query_x, selected_weights, selected_masks), data.query_y,
        ).mean()
        penalty = config.binary_penalty * (selected_soft * (1.0 - selected_soft)).mean()
        (query_loss + penalty).backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), 10.0)
        generator_optimizer.step()

        row = {
            "step": outer_step, "temperature": temperature, "support_bce": support_loss,
            "validation_bce": z_info["validation_bce"], "query_bce": float(query_loss.detach()),
            "binary_penalty": float(penalty.detach()), "generator_gradient_norm": float(gradient_norm),
            "z_steps": z_info["steps"], "z_converged": z_info["converged"],
        }
        history.append(row)
        if outer_step == 1 or outer_step % 10 == 0 or outer_step == config.outer_steps:
            print(
                f"LATENT seed={config.seed} step={outer_step}/{config.outer_steps} "
                f"support={support_loss:.5f} val={float(z_info['validation_bce']):.5f} "
                f"query={float(query_loss.detach()):.5f} z_steps={z_info['steps']}", flush=True,
            )

    torch.save({
        "config": config.to_dict(), "generator": generator.state_dict(), "initial_generator": initial_state,
        "latents": z.detach().cpu(), "history": history,
    }, output / "training_artifacts.pt")

    evaluations = {
        "validation": evaluate_generator(generator, VALIDATION_PATTERNS, config, device, "validation"),
        "test": evaluate_generator(generator, TEST_PATTERNS, config, device, "test"),
    }
    parameter_count = sum(parameter.numel() for parameter in generator.parameters())
    summary = {
        "config": config.to_dict(),
        "protocol": {
            "weights": "persistent batched MLP weights minimize support loss",
            "latent": "batched Adam minimizes dedicated validation loss to tolerance/patience",
            "generator": "shared parameters minimize a disjoint query loss",
            "gold_access": "analytic mask is evaluation-only",
            "final_masks": "hard exact-K; MLP weights are refit from scratch for comparison",
            "input_loss": "exact class-balanced expectation over all 2^8 inputs" if config.exact_population_loss else "separate class-balanced support/validation/query samples",
        },
        "task_splits": {"train": list(TRAIN_PATTERNS), "validation": list(VALIDATION_PATTERNS), "test": list(TEST_PATTERNS)},
        "generator_parameters": parameter_count,
        "parallel_mlps": tasks * config.restarts,
        "history": history,
        "evaluations": evaluations,
        "seconds": time.monotonic() - started,
    }
    with (output / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    _write_report(summary, output / "RESULTS.md")
    _plot(summary, output / "summary.png")
    return output


def _write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Joint latent structure learning", "",
        f"Seed: `{summary['config']['seed']}`; parallel train MLPs: `{summary['parallel_mlps']}`; "
        f"generator parameters: `{summary['generator_parameters']}`.", "",
        "All MLPs and latent restarts are trained in vectorized task × restart tensors. "
        "Gold structure is evaluation-only.", "",
        "| Split | Method | Query BCE | Balanced accuracy | Regret vs analytic | IoU to analytic |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split_name in ("validation", "test"):
        split = summary["evaluations"][split_name]
        analytic = split["analytic"]["mean_query_bce"]
        for key in ("generated", "random", "analytic"):
            row = split[key]
            lines.append(
                f"| {split_name} | {row['method']} | {row['mean_query_bce']:.6f} | "
                f"{row['mean_query_balanced_accuracy']:.4f} | {row['mean_query_bce'] - analytic:+.6f} | "
                f"{row['mean_iou_to_analytic']:.4f} |"
            )
    z_steps = [row["z_steps"] for row in summary["history"]]
    converged = [row["z_converged"] for row in summary["history"]]
    lines += [
        "", "| Optimization statistic | Value |", "|---|---:|",
        f"| Mean z steps per outer update | {sum(z_steps) / len(z_steps):.1f} |",
        f"| Fraction stopped by convergence criterion | {sum(converged) / len(converged):.3f} |",
        f"| Wall time, seconds | {summary['seconds']:.1f} |",
    ]
    path.write_text("\n".join(lines) + "\n")


def _plot(summary: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    history = summary["history"]
    test = summary["evaluations"]["test"]
    masks = torch.tensor(test["generated"]["chosen_masks"])
    gold = ideal_mask()
    figure, axes = plt.subplots(2, 4, figsize=(12, 6))
    axis = axes[0, 0]
    axis.plot([row["step"] for row in history], [row["support_bce"] for row in history], label="support")
    axis.plot([row["step"] for row in history], [row["validation_bce"] for row in history], label="validation")
    axis.plot([row["step"] for row in history], [row["query_bce"] for row in history], label="query")
    axis.set(xlabel="Outer step", ylabel="BCE", title="Joint optimization")
    axis.legend(fontsize=8)
    axis = axes[0, 1]
    names = ("generated", "random", "analytic")
    axis.bar(names, [test[name]["mean_query_bce"] for name in names])
    axis.set(ylabel="Query BCE", title="Held-out tasks")
    axis.tick_params(axis="x", rotation=20)
    for index, pattern in enumerate(test["patterns"]):
        axis = axes.flat[index + 2]
        axis.imshow(masks[index], cmap="Greys", vmin=0, vmax=1)
        axis.set_title(f"Generated: {pattern}")
        axis.set(xticks=[], yticks=[])
    axes.flat[6].imshow(gold, cmap="Greys", vmin=0, vmax=1)
    axes.flat[6].set_title("Analytic mask")
    axes.flat[6].set(xticks=[], yticks=[])
    axes.flat[7].axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)
