"""Paired condition-swap evaluation for a frozen meta_pattern full-U generator.

Every target task is evaluated with every supplied pattern-length condition.
The task-specific vectors are freshly fit from identical initializations and
minibatches for all six conditions. No checkpoint or hyperparameter is chosen
using the held-out target lengths.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

from meta_pattern.calibrate import _batched_logits, batched_adapt_v
from meta_pattern.common import seed_for, task_splits, write_json
from meta_pattern.config import Config
from meta_pattern.data import sample_dataset
from meta_pattern.evaluate_interpolation import load_interpolation_checkpoint


EVALUATION_SEED = 20260919
CONDITIONS = (3, 4, 5, 6, 7, 8)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _basis_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    """Projector distance; invariant to an invertible basis change."""
    def orthonormal_basis(value: torch.Tensor) -> torch.Tensor:
        vectors, singular, _ = torch.linalg.svd(value.double(), full_matrices=False)
        threshold = singular.max() * max(value.shape) * torch.finfo(torch.float64).eps
        return vectors[:, singular > threshold]

    q_left, q_right = orthonormal_basis(left), orthonormal_basis(right)
    rank_sum = q_left.shape[1] + q_right.shape[1]
    overlap = (q_left.T @ q_right).square().sum()
    return float(((rank_sum - 2 * overlap).clamp_min(0) / rank_sum).sqrt())


def _episodes(config: Config, length: int, repeats: int, support_size: int, test_size: int,
              patterns_per_length: int | None):
    tasks = [task for task in task_splits(config)["test"] if task.length == length]
    tasks.sort(key=lambda task: seed_for("condition-matrix-pattern-selection",
                                         EVALUATION_SEED, length, task.pattern))
    if patterns_per_length is not None:
        tasks = tasks[:patterns_per_length]
    if not tasks:
        raise ValueError(f"no test tasks for length {length}")
    result = []
    for task in tasks:
        for repeat in range(repeats):
            base = seed_for("condition-matrix", EVALUATION_SEED, task.pattern, repeat)
            support = sample_dataset(task, support_size, seed=seed_for(base, "support"),
                                     split="support", split_seed=config.input_split_seed,
                                     balanced=True, seq_len=config.seq_len)
            test = sample_dataset(task, test_size, seed=seed_for(base, "test"),
                                  split="test", split_seed=config.input_split_seed,
                                  balanced=True, seq_len=config.seq_len)
            result.append((task.pattern, repeat, seed_for(base, "v"), support, test))
    return result


def evaluate(checkpoint: Path, output: Path, *, device: str, repeats: int,
             support_size: int, test_size: int, steps: tuple[int, ...], batch_size: int,
             chunk_size: int, patterns_per_length: int | None = None) -> dict:
    if output.exists():
        raise FileExistsError(output)
    state, config, model = load_interpolation_checkpoint(checkpoint, device)
    if not config.condition_length or config.method != "generator":
        raise ValueError("condition matrix requires a length-conditioned generator")
    if (any(step <= 0 for step in steps) or repeats < 1 or
            min(support_size, test_size, chunk_size) < 1 or
            (patterns_per_length is not None and patterns_per_length < 1)):
        raise ValueError("invalid evaluation size")
    if batch_size > support_size:
        raise ValueError("batch size exceeds support size")

    with torch.no_grad():
        structures = {condition: tuple(value.detach() for value in model(condition))
                      for condition in CONDITIONS}
    structural_distance = {
        f"{left},{right}": [
            _basis_distance(a, b) for a, b in zip(structures[left], structures[right])
        ]
        for left in CONDITIONS for right in CONDITIONS
    }
    rows = []
    counts = {}
    for length in CONDITIONS:
        episodes = _episodes(config, length, repeats, support_size, test_size,
                             patterns_per_length)
        counts[str(length)] = len(episodes)
        for condition in CONDITIONS:
            u = structures[condition]
            for offset in range(0, len(episodes), chunk_size):
                chunk = episodes[offset:offset + chunk_size]
                support_x = torch.stack([item[3]["x"] for item in chunk]).to(device)
                support_y = torch.stack([item[3]["y"] for item in chunk]).to(device)
                test_x = torch.stack([item[4]["x"] for item in chunk]).to(device)
                test_y = torch.stack([item[4]["y"] for item in chunk]).to(device)
                n = len(chunk)
                vectors = batched_adapt_v(
                    u, support_x, support_y, steps=max(steps),
                    lrs=torch.full((n,), config.inner_lr, device=device),
                    seeds=torch.tensor([item[2] for item in chunk], dtype=torch.int64),
                    optimizers=[config.inner_optimizer] * n,
                    init_scales=torch.full((n,), config.init_scale, device=device),
                    batch_size=batch_size, checkpoints=steps,
                )
                with torch.no_grad():
                    for budget in steps:
                        scores = _batched_logits(test_x, u, vectors[budget])
                        losses = F.binary_cross_entropy_with_logits(
                            scores, test_y, reduction="none"
                        ).mean(1)
                        accuracies = ((scores > 0) == test_y.bool()).float().mean(1)
                        if not bool(torch.isfinite(losses).all()):
                            raise FloatingPointError(f"non-finite test loss for length={length}")
                        for item, loss, accuracy in zip(chunk, losses, accuracies):
                            rows.append({"target_length": length, "condition_length": condition,
                                         "pattern": item[0], "repeat": item[1], "steps": budget,
                                         "test_bce": float(loss), "test_accuracy": float(accuracy)})
            print(f"MATRIX target={length} condition={condition} episodes={len(episodes)}", flush=True)

    result = {"protocol": {"evaluation_seed": EVALUATION_SEED, "conditions": list(CONDITIONS),
                           "task_set": "test patterns; all patterns at unseen lengths 5 and 7 before optional sampling",
                           "patterns_per_length": patterns_per_length,
                           "pattern_selection": "deterministic hash order, independent of checkpoint",
                           "paired_v_initialization_and_minibatches": True,
                           "support_split": "support", "test_split": "test"},
              "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": _sha256(checkpoint),
              "checkpoint_step": state.get("step"), "config": config.to_dict(),
              "repeats": repeats, "support_size": support_size, "test_size": test_size,
              "steps": list(steps), "batch_size": batch_size, "episode_counts": counts,
              "structural_projector_distance": structural_distance, "rows": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--support-size", type=int, default=4096)
    parser.add_argument("--test-size", type=int, default=2048)
    parser.add_argument("--steps", type=int, nargs="+", default=[50, 500])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--patterns-per-length", type=int)
    args = parser.parse_args()
    args.steps = tuple(sorted(set(args.steps)))
    evaluate(args.checkpoint, args.out, device=args.device, repeats=args.repeats,
             support_size=args.support_size, test_size=args.test_size,
             steps=args.steps, batch_size=args.batch_size, chunk_size=args.chunk_size,
             patterns_per_length=args.patterns_per_length)


if __name__ == "__main__":
    main()
