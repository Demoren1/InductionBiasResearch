"""Bilevel optimization of a shared exact-K mask through task losses.

The outer variables are mask logits.  For every task, fresh MLP weights are
adapted to the current structure and the task loss is differentiated through
all inner updates.  The analytic Toeplitz mask is never used by the optimizer;
it is an evaluation-only oracle reference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from pattern import config as pattern_config
from pattern.data.generate import contains_pattern, ideal_mask
from pattern.evaluation.decoder_agreement import align_columns, hard_topk, soft_topk
from pattern.models.mlp import generate_fixed_sparsity_masks


TRAIN_PATTERNS = (
    "0000", "1111", "0001", "1000", "1110", "0111",
    "0010", "0100", "1101", "1011",
)
VALIDATION_PATTERNS = ("0110", "1001")
TEST_PATTERNS = ("0011", "1100", "0101", "1010")


@dataclass(frozen=True)
class BilevelMaskConfig:
    seed: int = 42
    relaxation: str = "soft"
    outer_steps: int = 400
    inner_steps: int = 50
    tasks_per_step: int = 5
    inner_lr: float = 0.03
    outer_lr: float = 0.05
    temperature_start: float = 1.0
    temperature_end: float = 0.10
    binary_penalty: float = 0.01
    validate_every: int = 25
    validation_restarts: int = 2
    eval_steps: int = 1000
    eval_restarts: int = 4
    random_masks: int = 16
    k_active: int = pattern_config.K_ACTIVE

    def __post_init__(self) -> None:
        if self.relaxation not in {"soft", "ste"}:
            raise ValueError("relaxation must be 'soft' or 'ste'")
        for name in (
            "outer_steps", "inner_steps", "tasks_per_step", "validate_every",
            "validation_restarts", "eval_steps", "eval_restarts", "random_masks",
            "k_active",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("inner_lr", "outer_lr", "temperature_start", "temperature_end"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.binary_penalty < 0:
            raise ValueError("binary_penalty must be non-negative")
        if not 0 < self.k_active < pattern_config.MASK_DIM:
            raise ValueError("k_active must be strictly between zero and mask size")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MaskStructure(nn.Module):
    """A shared mask represented by unconstrained logits."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        initial = torch.randn(pattern_config.MASK_DIM, generator=generator) * 0.05
        self.logits = nn.Parameter(initial.reshape(pattern_config.SEQ_LEN, pattern_config.H))

    def masks(self, temperature: float, k_active: int, relaxation: str) -> tuple[torch.Tensor, torch.Tensor]:
        flat = self.logits.reshape(1, -1)
        soft = soft_topk(flat, k_active, temperature).reshape_as(self.logits)
        hard = hard_topk(flat, k_active).reshape_as(self.logits)
        if relaxation == "soft":
            forward_mask = soft
        elif relaxation == "ste":
            forward_mask = _HardForwardSoftBackward.apply(hard, soft)
        else:
            raise ValueError("unknown relaxation")
        return forward_mask, soft

    def hard_mask(self, k_active: int) -> torch.Tensor:
        return hard_topk(self.logits.reshape(1, -1), k_active).reshape_as(self.logits)


class _HardForwardSoftBackward(torch.autograd.Function):
    """Return bit-exact hard values and route their gradient to ``soft``."""

    @staticmethod
    def forward(ctx: Any, hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
        return hard

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[None, torch.Tensor]:
        return None, gradient


def _seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _all_sequences(pattern: str, device: torch.device, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    if len(pattern) != pattern_config.PATTERN_LEN or any(bit not in "01" for bit in pattern):
        raise ValueError("invalid pattern")
    count = 2 ** pattern_config.SEQ_LEN
    ids = torch.arange(count, device=device, dtype=torch.long)
    shifts = torch.arange(pattern_config.SEQ_LEN - 1, -1, -1, device=device)
    x01 = ((ids[:, None] >> shifts) & 1).to(dtype)
    bits = torch.tensor([int(bit) for bit in pattern], device=device)
    y = contains_pattern(x01, bits).to(dtype)
    return x01.mul(2).sub(1), y


def balanced_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Exact class-balanced BCE on an enumerated task population."""
    losses = F.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype), reduction="none")
    positive = targets == 1
    negative = ~positive
    if not bool(positive.any()) or not bool(negative.any()):
        raise ValueError("both classes are required")
    return 0.5 * losses[positive].mean() + 0.5 * losses[negative].mean()


def balanced_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits > 0
    positive = targets == 1
    negative = ~positive
    return float(0.5 * (predictions[positive] == targets[positive].bool()).float().mean()
                 + 0.5 * (predictions[negative] == targets[negative].bool()).float().mean())


Weights = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def init_weights(seed: int, device: torch.device, dtype: torch.dtype = torch.float32) -> Weights:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    w1 = (torch.randn(pattern_config.SEQ_LEN, pattern_config.H, generator=generator) * 0.1).to(device, dtype).requires_grad_(True)
    b1 = torch.zeros(pattern_config.H, device=device, dtype=dtype, requires_grad=True)
    w2 = (torch.randn(pattern_config.H, generator=generator) * 0.1).to(device, dtype).requires_grad_(True)
    b2 = torch.zeros((), device=device, dtype=dtype, requires_grad=True)
    return w1, b1, w2, b2


def mlp_forward(x: torch.Tensor, weights: Weights, mask: torch.Tensor) -> torch.Tensor:
    w1, b1, w2, b2 = weights
    hidden = F.relu(x @ (w1 * mask) + b1)
    return hidden @ w2 + b2


def adapt_weights(
    mask: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    steps: int,
    lr: float,
    seed: int,
    create_graph: bool,
) -> Weights:
    """Functional Adam adaptation, optionally retaining the full hypergradient."""
    weights = init_weights(seed, x.device, x.dtype)
    first = tuple(torch.zeros_like(value) for value in weights)
    second = tuple(torch.zeros_like(value) for value in weights)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    for step in range(1, steps + 1):
        loss = balanced_bce(mlp_forward(x, weights, mask), y)
        gradients = torch.autograd.grad(loss, weights, create_graph=create_graph)
        first = tuple(beta1 * moment + (1.0 - beta1) * gradient for moment, gradient in zip(first, gradients))
        second = tuple(beta2 * moment + (1.0 - beta2) * gradient.square() for moment, gradient in zip(second, gradients))
        corrected_first = tuple(moment / (1.0 - beta1**step) for moment in first)
        corrected_second = tuple(moment / (1.0 - beta2**step) for moment in second)
        updated = tuple(
            value - lr * mean / (torch.sqrt(variance.clamp_min(1e-16)) + epsilon)
            for value, mean, variance in zip(weights, corrected_first, corrected_second)
        )
        if create_graph:
            weights = updated  # type: ignore[assignment]
        else:
            weights = tuple(value.detach().requires_grad_(True) for value in updated)  # type: ignore[assignment]
            first = tuple(value.detach() for value in first)
            second = tuple(value.detach() for value in second)
    return weights


def task_loss(
    pattern: str,
    mask: torch.Tensor,
    *,
    steps: int,
    lr: float,
    seed: int,
    create_graph: bool,
) -> tuple[torch.Tensor, Weights, torch.Tensor, torch.Tensor]:
    x, y = _all_sequences(pattern, mask.device, mask.dtype)
    weights = adapt_weights(mask, x, y, steps=steps, lr=lr, seed=seed, create_graph=create_graph)
    logits = mlp_forward(x, weights, mask)
    return balanced_bce(logits, y), weights, logits, y


def _temperature(config: BilevelMaskConfig, step: int) -> float:
    if config.outer_steps == 1:
        return config.temperature_end
    fraction = (step - 1) / (config.outer_steps - 1)
    return config.temperature_start * (config.temperature_end / config.temperature_start) ** fraction


def evaluate_mask(
    mask: torch.Tensor,
    patterns: Iterable[str],
    config: BilevelMaskConfig,
    *,
    seed_tag: object,
) -> dict[str, Any]:
    """Approximate the inner minimum with independent Adam restarts."""
    rows: list[dict[str, Any]] = []
    for pattern in patterns:
        restarts = []
        for restart in range(config.eval_restarts):
            loss, _, logits, targets = task_loss(
                pattern, mask, steps=config.eval_steps, lr=config.inner_lr,
                seed=_seed(config.seed, seed_tag, pattern, restart), create_graph=False,
            )
            restarts.append({"bce": float(loss.detach()), "balanced_accuracy": balanced_accuracy(logits.detach(), targets)})
        best = min(restarts, key=lambda row: row["bce"])
        rows.append({"pattern": pattern, "best": best, "restarts": restarts})
    return {
        "mean_best_bce": sum(row["best"]["bce"] for row in rows) / len(rows),
        "mean_best_balanced_accuracy": sum(row["best"]["balanced_accuracy"] for row in rows) / len(rows),
        "tasks": rows,
    }


def _batched_mlp_forward(x: torch.Tensor, weights: Weights, masks: torch.Tensor) -> torch.Tensor:
    """Evaluate independent MLPs, returning ``(examples, networks)`` logits."""
    w1, b1, w2, b2 = weights
    hidden = F.relu(torch.einsum("bl,nlh->bnh", x, w1 * masks) + b1)
    return torch.einsum("bnh,nh->bn", hidden, w2) + b2


def _batched_balanced_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    expanded = targets[:, None].expand_as(logits)
    losses = F.binary_cross_entropy_with_logits(logits, expanded, reduction="none")
    positive = targets == 1
    negative = ~positive
    return 0.5 * losses[positive].mean(0) + 0.5 * losses[negative].mean(0)


def _evaluate_mask_collection(
    masks: torch.Tensor,
    patterns: Iterable[str],
    config: BilevelMaskConfig,
    *,
    split_name: str,
) -> list[dict[str, Any]]:
    """Evaluate many hard masks in parallel with paired MLP initializations."""
    if masks.ndim != 3 or masks.shape[1:] != (pattern_config.SEQ_LEN, pattern_config.H):
        raise ValueError("masks must have shape (n, 8, 8)")
    n_masks = masks.shape[0]
    per_mask_tasks: list[list[dict[str, Any]]] = [[] for _ in range(n_masks)]
    for pattern in patterns:
        x, y = _all_sequences(pattern, masks.device, masks.dtype)
        base = [init_weights(
            _seed(config.seed, "evaluation", split_name, pattern, restart), masks.device, masks.dtype,
        ) for restart in range(config.eval_restarts)]
        # Every mask receives the same initialization for a given restart.
        w1 = torch.stack([base[r][0] for _ in range(n_masks) for r in range(config.eval_restarts)]).requires_grad_(True)
        b1 = torch.stack([base[r][1] for _ in range(n_masks) for r in range(config.eval_restarts)]).requires_grad_(True)
        w2 = torch.stack([base[r][2] for _ in range(n_masks) for r in range(config.eval_restarts)]).requires_grad_(True)
        b2 = torch.stack([base[r][3] for _ in range(n_masks) for r in range(config.eval_restarts)]).requires_grad_(True)
        weights: Weights = (w1, b1, w2, b2)
        expanded_masks = masks[:, None].expand(-1, config.eval_restarts, -1, -1).reshape(
            n_masks * config.eval_restarts, pattern_config.SEQ_LEN, pattern_config.H
        )
        first = tuple(torch.zeros_like(value) for value in weights)
        second = tuple(torch.zeros_like(value) for value in weights)
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for step in range(1, config.eval_steps + 1):
            losses = _batched_balanced_bce(_batched_mlp_forward(x, weights, expanded_masks), y)
            gradients = torch.autograd.grad(losses.sum(), weights)
            first = tuple(beta1 * moment + (1.0 - beta1) * gradient for moment, gradient in zip(first, gradients))
            second = tuple(beta2 * moment + (1.0 - beta2) * gradient.square() for moment, gradient in zip(second, gradients))
            weights = tuple(
                (value - config.inner_lr * (mean / (1.0 - beta1**step)) /
                 (torch.sqrt((variance / (1.0 - beta2**step)).clamp_min(1e-16)) + epsilon)).detach().requires_grad_(True)
                for value, mean, variance in zip(weights, first, second)
            )  # type: ignore[assignment]
            first = tuple(value.detach() for value in first)
            second = tuple(value.detach() for value in second)
        with torch.no_grad():
            logits = _batched_mlp_forward(x, weights, expanded_masks)
            losses = _batched_balanced_bce(logits, y).reshape(n_masks, config.eval_restarts)
            predictions = logits > 0
            positive, negative = y == 1, y == 0
            accuracies = (0.5 * (predictions[positive] == y[positive, None].bool()).float().mean(0)
                          + 0.5 * (predictions[negative] == y[negative, None].bool()).float().mean(0))
            accuracies = accuracies.reshape(n_masks, config.eval_restarts)
        for mask_index in range(n_masks):
            restarts = [
                {"bce": float(losses[mask_index, restart]),
                 "balanced_accuracy": float(accuracies[mask_index, restart])}
                for restart in range(config.eval_restarts)
            ]
            per_mask_tasks[mask_index].append({
                "pattern": pattern, "best": min(restarts, key=lambda row: row["bce"]), "restarts": restarts,
            })
    return [{
        "mean_best_bce": sum(row["best"]["bce"] for row in tasks) / len(tasks),
        "mean_best_balanced_accuracy": sum(row["best"]["balanced_accuracy"] for row in tasks) / len(tasks),
        "tasks": tasks,
    } for tasks in per_mask_tasks]


def _validation_score(model: MaskStructure, config: BilevelMaskConfig, device: torch.device) -> float:
    hard = model.hard_mask(config.k_active).detach()
    losses = []
    for pattern in VALIDATION_PATTERNS:
        candidates = []
        for restart in range(config.validation_restarts):
            loss, _, _, _ = task_loss(
                pattern, hard, steps=config.inner_steps, lr=config.inner_lr,
                seed=_seed(config.seed, "validation", pattern, restart), create_graph=False,
            )
            candidates.append(float(loss.detach()))
        losses.append(min(candidates))
    return sum(losses) / len(losses)


def _iou_to_ideal(mask: torch.Tensor) -> float:
    gold = ideal_mask().to(mask.device, mask.dtype)
    aligned = align_columns(gold, mask)
    intersection = (aligned.bool() & gold.bool()).sum()
    union = (aligned.bool() | gold.bool()).sum()
    return float(intersection / union)


def train_structure(config: BilevelMaskConfig, output: str | Path, device: str | torch.device = "cuda") -> Path:
    """Train, evaluate, and save one shared mask structure."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(device)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    model = MaskStructure(config.seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.outer_lr)
    initial_mask = model.hard_mask(config.k_active).detach().clone()
    best_validation = math.inf
    best_logits = model.logits.detach().clone()
    history: list[dict[str, Any]] = []
    started = time.monotonic()

    for step in range(1, config.outer_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        temperature = _temperature(config, step)
        episode_rng = random.Random(_seed(config.seed, "outer", step))
        tasks = [episode_rng.choice(TRAIN_PATTERNS) for _ in range(config.tasks_per_step)]
        total_loss = 0.0
        for slot, pattern in enumerate(tasks):
            forward_mask, _ = model.masks(temperature, config.k_active, config.relaxation)
            loss, _, _, _ = task_loss(
                pattern, forward_mask, steps=config.inner_steps, lr=config.inner_lr,
                seed=_seed(config.seed, "train", step, slot, pattern), create_graph=True,
            )
            (loss / config.tasks_per_step).backward()
            total_loss += float(loss.detach()) / config.tasks_per_step
        _, soft = model.masks(temperature, config.k_active, config.relaxation)
        penalty = config.binary_penalty * (soft * (1.0 - soft)).mean()
        penalty.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        row = {
            "step": step,
            "task_bce": total_loss,
            "binary_penalty": float(penalty.detach()),
            "temperature": temperature,
            "gradient_norm": float(gradient_norm),
        }
        if step == 1 or step % config.validate_every == 0 or step == config.outer_steps:
            validation = _validation_score(model, config, device)
            row["validation_hard_bce"] = validation
            if validation < best_validation:
                best_validation = validation
                best_logits = model.logits.detach().clone()
            print(
                f"BILEVEL relaxation={config.relaxation} seed={config.seed} "
                f"step={step}/{config.outer_steps} task_bce={total_loss:.6f} "
                f"hard_val={validation:.6f} temp={temperature:.3f}",
                flush=True,
            )
        history.append(row)

    final_logits = model.logits.detach().clone()
    model.logits.data.copy_(best_logits)
    learned_mask = model.hard_mask(config.k_active).detach().clone()
    gold = ideal_mask().to(device=device, dtype=learned_mask.dtype)
    random_masks = generate_fixed_sparsity_masks(
        config.random_masks, pattern_config.SEQ_LEN, pattern_config.H,
        config.k_active, seed=_seed(config.seed, "random_masks"),
    ).to(device)

    # Preserve the learned structure before the potentially long downstream evaluation.
    torch.save({
        "best_logits": best_logits.cpu(), "final_logits": final_logits.cpu(),
        "initial_mask": initial_mask.cpu(), "learned_mask": learned_mask.cpu(),
        "analytic_mask": gold.cpu(), "random_masks": random_masks.cpu(),
        "config": config.to_dict(), "history": history,
    }, output / "training_artifacts.pt")

    evaluations: dict[str, Any] = {
        "learned": {}, "analytic": {}, "initial": {}, "random": {},
    }
    for split_name, patterns in (
        ("train", TRAIN_PATTERNS), ("validation", VALIDATION_PATTERNS), ("test", TEST_PATTERNS),
    ):
        all_masks = torch.cat((learned_mask[None], gold[None], initial_mask[None], random_masks), dim=0)
        evaluated = _evaluate_mask_collection(all_masks, patterns, config, split_name=split_name)
        evaluations["learned"][split_name] = evaluated[0]
        evaluations["analytic"][split_name] = evaluated[1]
        evaluations["initial"][split_name] = evaluated[2]
        random_rows = evaluated[3:]
        evaluations["random"][split_name] = {
            "mean_mask_bce": sum(row["mean_best_bce"] for row in random_rows) / len(random_rows),
            "mean_mask_balanced_accuracy": sum(row["mean_best_balanced_accuracy"] for row in random_rows) / len(random_rows),
            "best_mask_bce": min(row["mean_best_bce"] for row in random_rows),
            "masks": random_rows,
        }

    summary = {
        "config": config.to_dict(),
        "task_splits": {
            "train": list(TRAIN_PATTERNS), "validation": list(VALIDATION_PATTERNS), "test": list(TEST_PATTERNS),
        },
        "protocol": {
            "outer_objective": "class-balanced population BCE after full differentiation through fresh inner Adam",
            "mask_relaxation": config.relaxation,
            "cardinality": "soft-top-K during training; hard exact-K for validation and final evaluation",
            "gold_access": "analytic mask is used only after outer optimization as an oracle reference",
            "input_distribution": "all 2^8 binary sequences, class-balanced loss",
        },
        "history": history,
        "best_validation_hard_bce": best_validation,
        "mask_metrics": {
            "initial_iou_to_ideal": _iou_to_ideal(initial_mask),
            "learned_iou_to_ideal": _iou_to_ideal(learned_mask),
            "learned_active": int(learned_mask.sum()),
            "initial_active": int(initial_mask.sum()),
        },
        "evaluations": evaluations,
        "seconds": time.monotonic() - started,
    }
    torch.save({
        "best_logits": best_logits.cpu(), "final_logits": final_logits.cpu(),
        "initial_mask": initial_mask.cpu(), "learned_mask": learned_mask.cpu(),
        "analytic_mask": gold.cpu(), "random_masks": random_masks.cpu(),
        "config": config.to_dict(),
    }, output / "artifacts.pt")
    with (output / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    _write_report(summary, output / "RESULTS.md")
    _plot(summary, output / "summary.png")
    return output


def _write_report(summary: dict[str, Any], path: Path) -> None:
    metrics = summary["mask_metrics"]
    lines = [
        "# Direct bilevel binary-mask learning", "",
        f"Relaxation: `{summary['config']['relaxation']}`; seed: `{summary['config']['seed']}`.", "",
        "Analytic structure is an evaluation-only reference and never enters the outer objective.", "",
        "| Split | Method | Best BCE | Balanced accuracy | Regret vs analytic |", "|---|---|---:|---:|---:|",
    ]
    for split in ("train", "validation", "test"):
        analytic = summary["evaluations"]["analytic"][split]["mean_best_bce"]
        for method in ("initial", "learned", "analytic"):
            row = summary["evaluations"][method][split]
            lines.append(
                f"| {split} | {method} | {row['mean_best_bce']:.6f} | "
                f"{row['mean_best_balanced_accuracy']:.4f} | {row['mean_best_bce'] - analytic:+.6f} |"
            )
        random_row = summary["evaluations"]["random"][split]
        lines.append(
            f"| {split} | random exact-K mean | {random_row['mean_mask_bce']:.6f} | "
            f"{random_row['mean_mask_balanced_accuracy']:.4f} | {random_row['mean_mask_bce'] - analytic:+.6f} |"
        )
    lines += [
        "", "| Mask metric | Value |", "|---|---:|",
        f"| Active entries | {metrics['learned_active']} |",
        f"| Initial IoU to analytic | {metrics['initial_iou_to_ideal']:.4f} |",
        f"| Learned IoU to analytic | {metrics['learned_iou_to_ideal']:.4f} |",
    ]
    path.write_text("\n".join(lines) + "\n")


def _plot(summary: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    history = summary["history"]
    artifacts = torch.load(path.parent / "artifacts.pt", weights_only=True)
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    axes[0].plot([row["step"] for row in history], [row["task_bce"] for row in history])
    axes[0].set(xlabel="Outer step", ylabel="Task BCE", title="Outer objective")
    axes[1].imshow(artifacts["learned_mask"], cmap="Greys", vmin=0, vmax=1)
    axes[1].set_title("Learned hard mask")
    axes[2].imshow(artifacts["analytic_mask"], cmap="Greys", vmin=0, vmax=1)
    axes[2].set_title("Analytic mask")
    for axis in axes[1:]:
        axis.set(xticks=[], yticks=[])
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
