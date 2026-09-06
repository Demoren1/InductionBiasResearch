"""Evaluate full-U models on the two held-out interpolation lengths.

The protocol is deliberately narrower than :mod:`meta_pattern.evaluate`:
models must have been meta-trained only at lengths 3, 4, 6, and 8.  It then
adapts fresh task vectors for every one of the 32 length-5 and 128 length-7
patterns.  The two neighbouring training-length conditions are evaluated with
the exact same episode, initial ``v``, and inner-loop minibatches as the
correct condition.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from .calibrate import _batched_logits, batched_adapt_v
from .common import make_model, seed_for, setup, source_hashes, write_json
from .config import Config
from .data import PatternTask, sample_dataset


TRAIN_LENGTHS = (3, 4, 6, 8)
HELD_LENGTHS = (5, 7)
DEFAULT_STEPS = (20, 100, 500, 2000)
CASE_CHUNK_SIZE = 32
EVALUATION_SEED = 20_260_906


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(*values: torch.Tensor) -> str:
    """Hash tensors including layout metadata, after canonical CPU copying."""
    digest = hashlib.sha256()
    for value in values:
        cpu = value.detach().contiguous().cpu()
        digest.update(str(tuple(cpu.shape)).encode())
        digest.update(str(cpu.dtype).encode())
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def _validate_protocol(config: Config) -> None:
    if config.method == "table":
        raise ValueError("A learned U table cannot define U for held-out lengths 5 and 7")
    if tuple(sorted(config.lengths)) != (3, 4, 5, 6, 7, 8):
        raise ValueError("Interpolation evaluation requires configured lengths exactly (3, 4, 5, 6, 7, 8)")
    if tuple(sorted(config.train_lengths)) != TRAIN_LENGTHS:
        raise ValueError("Checkpoint must be trained at exactly lengths (3, 4, 6, 8), never 5 or 7")
    if config.input_split_seed != 1729:
        raise ValueError("Interpolation protocol fixes input_split_seed to 1729")


def load_interpolation_checkpoint(checkpoint: Path | str, device: str):
    """Load a checkpoint only after checking the held-length protocol."""
    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "config" not in state or "model" not in state:
        raise ValueError("Checkpoint must contain config and model entries")
    config = Config(**state["config"])
    _validate_protocol(config)
    # Set cuBLAS deterministic workspace settings before ``model.to(cuda)``
    # can initialize a CUDA context.
    setup(config.seed, device)
    model = make_model(config, device)
    model.load_state_dict(state["model"])
    model.eval()
    return state, config, model


def _condition_lengths(length: int) -> tuple[tuple[str, int], ...]:
    if length == 5:
        return (("correct", 5), ("wrong_lower_neighbor", 4), ("wrong_upper_neighbor", 6))
    if length == 7:
        return (("correct", 7), ("wrong_lower_neighbor", 6), ("wrong_upper_neighbor", 8))
    raise ValueError(f"held interpolation length must be one of {HELD_LENGTHS}, got {length}")


def _episodes_for_length(length: int, repeats: int, support_size: int, test_size: int):
    """Generate the fixed CPU episodes before copying one chunk to a GPU."""
    episodes = []
    hashes: dict[str, dict[str, str]] = {}
    for pattern_value in range(1 << length):
        task = PatternTask(format(pattern_value, f"0{length}b"))
        for repeat in range(repeats):
            base = seed_for("evaluation", EVALUATION_SEED, task.pattern, repeat)
            support = sample_dataset(
                task, support_size, seed=seed_for(base, "support"), split="support",
                split_seed=1729, balanced=True, seq_len=32,
            )
            balanced = sample_dataset(
                task, test_size, seed=seed_for(base, "balanced"), split="test",
                split_seed=1729, balanced=True, seq_len=32,
            )
            natural = sample_dataset(
                task, test_size, seed=seed_for(base, "natural"), split="test",
                split_seed=1729, balanced=False, seq_len=32,
            )
            key = f"k{length}:{task.pattern}:repeat{repeat}"
            hashes[key] = {
                "support": _tensor_sha256(support["ids"], support["y"]),
                "balanced": _tensor_sha256(balanced["ids"], balanced["y"]),
                "natural": _tensor_sha256(natural["ids"], natural["y"]),
            }
            episodes.append({
                "task": task,
                "repeat": repeat,
                "v_seed": seed_for(base, "v"),
                "support": support,
                "tests": {"balanced": balanced, "natural": natural},
            })
    return episodes, hashes


def _row_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float | bool | None]:
    """Compute serializable per-model metrics without allowing NaNs in JSON."""
    if not bool(torch.isfinite(logits).all()):
        return {"bce": None, "accuracy": None, "positive_fraction": float(labels.mean()),
                "tpr": None, "tnr": None, "nonfinite": True}
    prediction = logits > 0
    positive = labels.bool()
    negative = ~positive
    bce = F.binary_cross_entropy_with_logits(logits, labels)
    if not bool(torch.isfinite(bce)):
        return {"bce": None, "accuracy": None, "positive_fraction": float(labels.mean()),
                "tpr": None, "tnr": None, "nonfinite": True}
    return {
        "bce": float(bce.item()),
        "accuracy": float((prediction == positive).float().mean().item()),
        "positive_fraction": float(labels.mean().item()),
        "tpr": float((prediction[positive] == positive[positive]).float().mean().item()) if bool(positive.any()) else None,
        "tnr": float((prediction[negative] == positive[negative]).float().mean().item()) if bool(negative.any()) else None,
        "nonfinite": False,
    }


def _u_diagnostics(correct_u, current_u) -> dict:
    """Record basis diagnostics only; they are not claims of learned structure."""
    output = {"singular_values": [torch.linalg.svdvals(value).cpu().tolist() for value in current_u]}
    relative = []
    projection_residual = []
    for reference, value in zip(correct_u, current_u):
        relative.append(float(torch.linalg.vector_norm(value - reference) /
                              torch.linalg.vector_norm(reference).clamp_min(1e-12)))
        basis, singular, _ = torch.linalg.svd(reference, full_matrices=False)
        threshold = torch.finfo(reference.dtype).eps * max(reference.shape) * singular.max()
        q = basis[:, singular > threshold]
        residual = value - q @ (q.T @ value)
        projection_residual.append(float(torch.linalg.vector_norm(residual) /
                                         torch.linalg.vector_norm(value).clamp_min(1e-12)))
    output["raw_relative_frobenius_to_correct"] = relative
    output["projection_residual_to_correct_column_space"] = projection_residual
    output["exactly_equal_to_correct"] = all(torch.equal(a, b) for a, b in zip(correct_u, current_u))
    return output


def _aggregate(rows: Iterable[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["length"], row["steps"], row["condition"], row["condition_length"], row["distribution"])
        groups.setdefault(key, []).append(row)
    output = []
    for key in sorted(groups):
        group = groups[key]
        finite = [row for row in group if not row["nonfinite"]]
        result = {
            "length": key[0], "steps": key[1], "condition": key[2],
            "condition_length": key[3], "distribution": key[4],
            "n": len(group), "n_finite": len(finite),
        }
        for metric in ("bce", "accuracy", "positive_fraction", "tpr", "tnr"):
            values = [row[metric] for row in finite if row[metric] is not None]
            # A failed trajectory disqualifies the primary aggregate rather
            # than silently improving it by dropping the failed episode.
            result[metric] = (sum(values) / len(values)) if values and len(finite) == len(group) else None
        output.append(result)
    return output


def evaluate_interpolation(
    checkpoint: Path | str,
    out: Path | str,
    device: str = "cuda",
    steps: Iterable[int] = DEFAULT_STEPS,
    repeats: int = 2,
    support_size: int = 8192,
    test_size: int = 2048,
    batch_size: int = 128,
) -> dict:
    """Evaluate every held pattern and write one non-overwritable JSON artifact."""
    steps = tuple(sorted(set(int(step) for step in steps)))
    if not steps or steps[0] < 1 or repeats < 1 or support_size < 1 or test_size < 1 or batch_size < 1:
        raise ValueError("steps, repeats, support_size, test_size, and batch_size must be positive")
    if batch_size > support_size:
        raise ValueError("batch_size cannot exceed support_size")
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite interpolation evaluation: {out}")

    checkpoint = Path(checkpoint)
    state, config, model = load_interpolation_checkpoint(checkpoint, device)
    model.eval()

    rows: list[dict] = []
    data_hashes: dict[str, dict[str, str]] = {}
    diagnostics: dict[str, dict[str, dict]] = {}
    for length in HELD_LENGTHS:
        episodes, length_hashes = _episodes_for_length(length, repeats, support_size, test_size)
        data_hashes.update(length_hashes)
        conditions = _condition_lengths(length)
        with torch.no_grad():
            u_by_condition = {
                name: tuple(value.detach() for value in model(condition_length))
                for name, condition_length in conditions
            }
        diagnostics[str(length)] = {
            name: {"condition_length": condition_length,
                   **_u_diagnostics(u_by_condition["correct"], u_by_condition[name])}
            for name, condition_length in conditions
        }

        for condition, condition_length in conditions:
            u = u_by_condition[condition]
            for start in range(0, len(episodes), CASE_CHUNK_SIZE):
                chunk = episodes[start:start + CASE_CHUNK_SIZE]
                support_x = torch.stack([episode["support"]["x"] for episode in chunk]).to(device)
                support_y = torch.stack([episode["support"]["y"] for episode in chunk]).to(device)
                seeds = torch.tensor([episode["v_seed"] for episode in chunk], dtype=torch.int64)
                count = len(chunk)
                snapshots = batched_adapt_v(
                    u, support_x, support_y, steps=max(steps),
                    lrs=torch.full((count,), config.inner_lr, device=device),
                    seeds=seeds, optimizers=[config.inner_optimizer] * count,
                    init_scales=torch.full((count,), config.init_scale, device=device),
                    batch_size=batch_size, checkpoints=steps,
                )
                for budget in steps:
                    v = snapshots[budget]
                    for distribution in ("balanced", "natural"):
                        test_x = torch.stack([episode["tests"][distribution]["x"] for episode in chunk]).to(device)
                        test_y = torch.stack([episode["tests"][distribution]["y"] for episode in chunk]).to(device)
                        with torch.no_grad():
                            scores = [_row_metrics(logits, labels) for logits, labels in zip(
                                _batched_logits(test_x, u, v), test_y
                            )]
                        for episode, score in zip(chunk, scores):
                            rows.append({
                                "pattern": episode["task"].pattern, "length": length,
                                "repeat": episode["repeat"], "steps": budget,
                                "condition": condition, "condition_length": condition_length,
                                "distribution": distribution, **score,
                            })
                del support_x, support_y, snapshots
            print(f"INTERP length={length} condition={condition} episodes={len(episodes)}", flush=True)

    result = {
        "protocol": {
            "train_lengths": list(TRAIN_LENGTHS), "held_interpolation_lengths": list(HELD_LENGTHS),
            "all_patterns": {"5": 32, "7": 128}, "no_extrapolation": True,
            "evaluation_seed": EVALUATION_SEED, "input_split_seed": 1729,
            "support_split": "support", "test_split": "test",
            "same_v_and_minibatches_across_conditions": True,
            "case_chunk_size": CASE_CHUNK_SIZE,
        },
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": _file_sha256(checkpoint),
        "checkpoint_step": state.get("step"), "config": config.to_dict(),
        "steps": list(steps), "repeats": repeats, "support_size": support_size,
        "test_size": test_size, "batch_size": batch_size,
        "training_source_sha256": state.get("source_sha256"),
        "evaluation_source_sha256": source_hashes(), "data_hashes": data_hashes,
        "u_diagnostics": diagnostics, "aggregates": _aggregate(rows), "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", nargs="+", type=int, default=list(DEFAULT_STEPS))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--support-size", type=int, default=8192)
    parser.add_argument("--test-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    result = evaluate_interpolation(**vars(parser.parse_args()))
    print(f"WROTE {Path(result['checkpoint']).parent}", flush=True)


if __name__ == "__main__":
    main()
