"""Calibrate the optimizer for task vectors under the analytic pattern basis.

This module deliberately has a narrow role.  It uses only train/validation
patterns at the four meta-train lengths 3, 4, 6 and 8, and samples only the
support/query input partitions.  In particular it never imports the final
evaluation routine and cannot select hyperparameters on test patterns or on
the interpolated lengths 5 and 7.

The grid is sharded by independent optimizer configurations.  Within a
length, all task/repeat/configuration instances in a shard are adapted in one
batched calculation.  Their losses are still reduced *per model* before
``autograd.grad`` so that this is mathematically the same as independent
adaptation.  Repeated configurations of an episode receive identical initial
Gaussian draws (up to init_scale) and identical minibatch permutations.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from .common import dataset, seed_for, setup, source_hashes, task_splits, write_json
from .config import Config
from .models import UTuple, VTuple, ideal_u, init_v


KNOWN_LENGTHS = (3, 4, 6, 8)
BUDGETS = (20, 100, 500, 2000)
SUPPORT_SIZE = 8192
QUERY_SIZE = 2048
BATCH_SIZE = 128
REPEATS = 2
TASKS_PER_SPLIT = 4
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8


@dataclass(frozen=True)
class CalibrationConfig:
    """One independently evaluated optimizer setting."""

    optimizer: str
    lr: float
    init_scale: float

    @property
    def identifier(self) -> str:
        return f"{self.optimizer}_lr{self.lr:g}_init{self.init_scale:g}"


def grid() -> tuple[CalibrationConfig, ...]:
    """Return the fixed, pre-registered calibration grid."""
    result = []
    for optimizer, lrs in (("sgd", (0.1, 1.0, 3.0)), ("adam", (0.01, 0.03, 0.1))):
        for lr in lrs:
            for init_scale in (0.1, 1.0):
                result.append(CalibrationConfig(optimizer, lr, init_scale))
    return tuple(result)


def _selected_tasks(seed: int, split: str, length: int, limit: int = TASKS_PER_SPLIT):
    """Uniform-hash task selection, rather than a lexical prefix."""
    if split not in {"train", "val"}:
        raise ValueError("calibration may use only train or val tasks")
    candidates = [task for task in task_splits(Config()) [split] if task.length == length]
    # SHA-derived seed_for is uniform enough for a deterministic sample without
    # replacement.  Pattern text is only a tie breaker, never the selection key.
    candidates.sort(key=lambda task: (seed_for("calibration-task", seed, split, length, task.pattern), task.pattern))
    return candidates[:limit]


def selected_task_manifest(seed: int) -> dict[str, dict[str, list[str]]]:
    """Expose the exact admissible task catalogue for protocol review/tests."""
    return {
        str(length): {
            split: [task.pattern for task in _selected_tasks(seed, split, length)]
            for split in ("train", "val")
        }
        for length in KNOWN_LENGTHS
    }


def _batched_logits(x: torch.Tensor, u: UTuple, v: VTuple) -> torch.Tensor:
    """Evaluate independent task vectors in parallel.

    ``x`` is [models, examples, seq_len], and every row of ``v`` belongs to
    the correspondingly indexed model.  ``u`` is deliberately shared.
    """
    v1, v2 = v
    if x.ndim != 3 or x.shape[0] != v1.shape[0] or v1.shape[0] != v2.shape[0]:
        raise ValueError("x and batched v must agree on the model dimension")
    seq_len = x.shape[-1]
    hidden = u[1].shape[0] - 1
    if u[0].shape[0] != (seq_len + 1) * hidden:
        raise ValueError("U shape does not agree with batched input")
    w1 = (u[0] @ v1.T).T.reshape(v1.shape[0], seq_len + 1, hidden)
    w2 = (u[1] @ v2.T).T
    ones = torch.ones((*x.shape[:2], 1), dtype=x.dtype, device=x.device)
    hidden_values = F.relu(torch.bmm(torch.cat((x, ones), dim=-1), w1))
    return torch.bmm(torch.cat((hidden_values, ones), dim=-1), w2.unsqueeze(-1)).squeeze(-1)


def _batch_indices(
    seeds: torch.Tensor,
    n_examples: int,
    steps: int,
    batch_size: int,
) -> torch.Tensor:
    """Generate scalar-adapt_v-compatible permutations, sharing equal seeds."""
    if seeds.ndim != 1:
        raise ValueError("seeds must be one-dimensional")
    if not 1 <= batch_size <= n_examples:
        raise ValueError("batch_size must lie in [1, n_examples]")
    # Generate once per episode seed.  Configurations repeat a seed on purpose
    # so paired settings see exactly the same examples in each update.
    unique: dict[int, torch.Tensor] = {}
    output = []
    for seed in seeds.detach().cpu().tolist():
        seed = int(seed)
        cached = unique.get(seed)
        if cached is None:
            generator = torch.Generator(device="cpu").manual_seed(seed + 17_171)
            cached = torch.stack([
                torch.randperm(n_examples, generator=generator)[:batch_size]
                for _ in range(steps)
            ])
            unique[seed] = cached
        output.append(cached)
    return torch.stack(output)


def batched_adapt_v(
    u: UTuple,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    steps: int,
    lrs: torch.Tensor,
    seeds: torch.Tensor,
    optimizers: Sequence[str],
    init_scales: torch.Tensor,
    batch_size: int,
    checkpoints: Iterable[int] = (),
) -> dict[int, VTuple]:
    """Functional batched SGD/Adam, matching scalar :func:`adapt_v` updates.

    ``x`` and ``y`` are replicated by model.  This keeps the helper useful to
    final evaluation too, where episodes generally differ.  The returned
    checkpoint vectors are detached snapshots, suitable for metrics only.
    """
    if x.ndim != 3 or y.shape != x.shape[:2]:
        raise ValueError("x must be [models, examples, features] and y [models, examples]")
    models, n_examples = y.shape
    if steps < 0 or lrs.shape != (models,) or seeds.shape != (models,):
        raise ValueError("invalid batched optimizer inputs")
    if init_scales.shape != (models,) or len(optimizers) != models:
        raise ValueError("every model needs one optimizer and initialization scale")
    if any(name not in {"sgd", "adam"} for name in optimizers):
        raise ValueError("optimizer must be 'sgd' or 'adam'")
    wanted = set(int(step) for step in checkpoints)
    if any(step < 0 or step > steps for step in wanted):
        raise ValueError("checkpoints must lie in [0, steps]")
    if not 1 <= batch_size <= n_examples:
        raise ValueError("batch_size must lie in [1, n_examples]")

    # Calling init_v preserves the private CPU RNG specification of the scalar
    # implementation exactly.  It is small (20 coefficients/model) and avoids
    # a second subtly different initializer.
    initial = [init_v(u, int(seed), float(scale)) for seed, scale in zip(seeds.cpu(), init_scales.cpu())]
    v1 = torch.stack([value[0] for value in initial]).detach().requires_grad_(True)
    v2 = torch.stack([value[1] for value in initial]).detach().requires_grad_(True)
    first1, first2 = torch.zeros_like(v1), torch.zeros_like(v2)
    second1, second2 = torch.zeros_like(v1), torch.zeros_like(v2)
    indices = _batch_indices(seeds, n_examples, steps, batch_size).to(x.device)
    adam_mask = torch.tensor([name == "adam" for name in optimizers], device=x.device).view(models, 1)
    results: dict[int, VTuple] = {}

    def record(step: int) -> None:
        if step in wanted:
            results[step] = (v1.detach().clone(), v2.detach().clone())

    record(0)
    model_index = torch.arange(models, device=x.device).unsqueeze(1)
    for step in range(steps):
        current_indices = indices[:, step]
        xb = x[model_index, current_indices]
        yb = y[model_index, current_indices]
        logits = _batched_logits(xb, u, (v1, v2))
        losses = F.binary_cross_entropy_with_logits(logits, yb.to(logits.dtype), reduction="none").mean(dim=1)
        # A diverged model must not turn the scalar reduction for healthy
        # independent models into NaN.  Its own state remains non-finite and is
        # reported as such at every requested budget.
        safe_losses = torch.where(torch.isfinite(losses), losses, torch.zeros_like(losses))
        gradient1, gradient2 = torch.autograd.grad(safe_losses.sum(), (v1, v2))
        sgd1 = v1 - lrs[:, None] * gradient1
        sgd2 = v2 - lrs[:, None] * gradient2
        first1 = ADAM_BETA1 * first1 + (1.0 - ADAM_BETA1) * gradient1
        first2 = ADAM_BETA1 * first2 + (1.0 - ADAM_BETA1) * gradient2
        second1 = ADAM_BETA2 * second1 + (1.0 - ADAM_BETA2) * gradient1.square()
        second2 = ADAM_BETA2 * second2 + (1.0 - ADAM_BETA2) * gradient2.square()
        correction1 = 1.0 - ADAM_BETA1 ** (step + 1)
        correction2 = 1.0 - ADAM_BETA2 ** (step + 1)
        denom1 = torch.sqrt((second1 / correction2).clamp_min(1e-16)) + ADAM_EPS
        denom2 = torch.sqrt((second2 / correction2).clamp_min(1e-16)) + ADAM_EPS
        adam1 = v1 - lrs[:, None] * (first1 / correction1) / denom1
        adam2 = v2 - lrs[:, None] * (first2 / correction1) / denom2
        v1 = torch.where(adam_mask, adam1, sgd1).detach().requires_grad_(True)
        v2 = torch.where(adam_mask, adam2, sgd2).detach().requires_grad_(True)
        # Scalar adapt_v only retains/detaches Adam moments for Adam branches;
        # detached tensors here have the identical forward values for both.
        first1, first2 = first1.detach(), first2.detach()
        second1, second2 = second1.detach(), second2.detach()
        record(step + 1)
    return results


def _case_metrics(logits: torch.Tensor, labels: torch.Tensor) -> list[dict[str, float | bool | None]]:
    """Metrics per independent model without serializing non-finite floats."""
    result = []
    for row_logits, row_labels in zip(logits, labels):
        finite = bool(torch.isfinite(row_logits).all())
        if not finite:
            result.append({"bce": None, "accuracy": None, "nan": True})
            continue
        bce = F.binary_cross_entropy_with_logits(row_logits, row_labels).item()
        if not torch.isfinite(torch.tensor(bce)):
            result.append({"bce": None, "accuracy": None, "nan": True})
            continue
        result.append({
            "bce": float(bce),
            "accuracy": float(((row_logits > 0) == row_labels.bool()).float().mean().item()),
            "nan": False,
        })
    return result


def _episodes_for_length(seed: int, length: int, device: str):
    """Create the fixed support/query episodes used for all grid settings."""
    c = Config()
    episodes = []
    for split in ("train", "val"):
        for task in _selected_tasks(seed, split, length):
            for repeat in range(REPEATS):
                base = seed_for("inner-calibration", seed, split, task.pattern, repeat)
                support = dataset(c, task, SUPPORT_SIZE, seed_for(base, "support"), "support", device)
                query = dataset(c, task, QUERY_SIZE, seed_for(base, "query"), "query", device)
                episodes.append({
                    "split": split,
                    "pattern": task.pattern,
                    "repeat": repeat,
                    "v_seed": seed_for(base, "v"),
                    "support_x": support["x"],
                    "support_y": support["y"],
                    "query_x": query["x"],
                    "query_y": query["y"],
                })
    return episodes


def _output_path(path: Path, shard: int) -> Path:
    """Accept an explicit JSON path or a parent-created output directory."""
    if path.suffix == ".json":
        return path
    return path / f"calibration_shard{shard}.json"


def calibrate(out: Path, shard: int, shards: int = 5, device: str = "cuda", seed: int = 42) -> Path:
    """Run one independent shard of the optimizer grid."""
    if shards != 5 or not 0 <= shard < shards:
        raise ValueError("this pre-registered protocol requires --shards 5 and --shard in 0..4")
    all_configs = grid()
    shard_configs = tuple(config for index, config in enumerate(all_configs) if index % shards == shard)
    if not shard_configs:
        raise RuntimeError("empty calibration shard")
    target = _output_path(Path(out), shard)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite calibration result: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    setup(seed, device)
    started = time.monotonic()
    rows: list[dict] = []
    task_manifest = selected_task_manifest(seed)

    for length in KNOWN_LENGTHS:
        episodes = _episodes_for_length(seed, length, device)
        episode_count = len(episodes)
        case_config = torch.arange(len(shard_configs), device=device).repeat_interleave(episode_count)
        case_episode = torch.arange(episode_count, device=device).repeat(len(shard_configs))
        support_x = torch.stack([episode["support_x"] for episode in episodes])[case_episode]
        support_y = torch.stack([episode["support_y"] for episode in episodes])[case_episode]
        query_x = torch.stack([episode["query_x"] for episode in episodes])[case_episode]
        query_y = torch.stack([episode["query_y"] for episode in episodes])[case_episode]
        lrs = torch.tensor([shard_configs[index].lr for index in case_config.tolist()], device=device)
        init_scales = torch.tensor([shard_configs[index].init_scale for index in case_config.tolist()], device=device)
        optimizers = [shard_configs[index].optimizer for index in case_config.tolist()]
        v_seeds = torch.tensor([episodes[index]["v_seed"] for index in case_episode.tolist()], dtype=torch.int64)
        u = ideal_u(length, rank1=Config().rank1, rank2=Config().rank2, device=device)
        snapshots = batched_adapt_v(
            u, support_x, support_y, steps=max(BUDGETS), lrs=lrs, seeds=v_seeds,
            optimizers=optimizers, init_scales=init_scales, batch_size=BATCH_SIZE,
            checkpoints=BUDGETS,
        )
        for budget in BUDGETS:
            with torch.no_grad():
                scores = _case_metrics(_batched_logits(query_x, u, snapshots[budget]), query_y)
            for index, score in enumerate(scores):
                config = shard_configs[int(case_config[index])]
                episode = episodes[int(case_episode[index])]
                rows.append({
                    "config": config.identifier,
                    "optimizer": config.optimizer,
                    "lr": config.lr,
                    "init_scale": config.init_scale,
                    "length": length,
                    "pattern": episode["pattern"],
                    "task_split": episode["split"],
                    "repeat": episode["repeat"],
                    "budget": budget,
                    **score,
                })
        print(
            f"CALIBRATE shard={shard}/{shards} length={length} cases={len(case_config)} "
            f"episodes={episode_count} seconds={time.monotonic() - started:.1f}",
            flush=True,
        )

    result = {
        "protocol": {
            "known_lengths": list(KNOWN_LENGTHS),
            "forbidden_lengths": [5, 7],
            "pattern_splits": ["train", "val"],
            "input_splits": ["support", "query"],
            "support_size": SUPPORT_SIZE,
            "query_size": QUERY_SIZE,
            "batch_size": BATCH_SIZE,
            "budgets": list(BUDGETS),
            "repeats": REPEATS,
            "tasks_per_split_maximum": TASKS_PER_SPLIT,
            "task_selection": "uniform SHA256-derived hash without replacement",
            "adam": {"beta1": ADAM_BETA1, "beta2": ADAM_BETA2, "eps": ADAM_EPS},
        },
        "seed": seed,
        "shard": shard,
        "shards": shards,
        "all_grid": [asdict(config) | {"config": config.identifier} for config in all_configs],
        "shard_grid": [asdict(config) | {"config": config.identifier} for config in shard_configs],
        "selected_tasks": task_manifest,
        "source_sha256": source_hashes(),
        "seconds": time.monotonic() - started,
        "rows": rows,
    }
    write_json(target, result)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = calibrate(**vars(args))
    print(f"WROTE {result}", flush=True)


if __name__ == "__main__":
    main()
