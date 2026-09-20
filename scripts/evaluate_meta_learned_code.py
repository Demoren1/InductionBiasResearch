"""Evaluate learned length codes on known and unseen meta_pattern tasks.

An unseen code is fitted with a frozen decoder on calibration pattern
identities. A disjoint set of patterns evaluates transfer after fresh task-v
adaptation. All compared U variants use identical episodes and v seeds.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

from meta_pattern.calibrate import _batched_logits, batched_adapt_v
from meta_pattern.common import seed_for, task_splits, write_json
from meta_pattern.data import PatternTask, _orbit, build_task_splits, sample_dataset
from meta_pattern.evaluate_interpolation import load_interpolation_checkpoint
from meta_pattern.models import adapt_v, forward_with_u
from scripts.meta_learned_code import TRAIN_LENGTHS, load


HELD_LENGTHS = (5, 7)
EVALUATION_SEED = 20260919


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _select_tasks(length: int) -> tuple[list[PatternTask], list[PatternTask], list[PatternTask]]:
    raw = build_task_splits(lengths=(3, 4, 5, 6, 7, 8), seed=42)
    def selected(split: str) -> list[PatternTask]:
        return sorted((task for task in raw[split] if task.length == length),
                      key=lambda task: seed_for("code-calibration", EVALUATION_SEED, task.pattern))
    train, validation = selected("train")[:4], selected("val")[:4]
    excluded = {equivalent for task in train + validation
                for equivalent in _orbit(task.pattern)}
    evaluation = [PatternTask(format(value, f"0{length}b")) for value in range(2**length)
                  if format(value, f"0{length}b") not in excluded]
    if len(train) != 4 or len(validation) != 4 or not evaluation:
        raise ValueError("insufficient calibration or evaluation tasks")
    return train, validation, evaluation


def _calibration_data(tasks: list[PatternTask], config, device: str, tag: str,
                      examples: int) -> list[tuple[PatternTask, dict, dict, int]]:
    result = []
    for task in tasks:
        base = seed_for("learn-new-code", EVALUATION_SEED, tag, task.pattern)
        support = sample_dataset(task, examples, seed=seed_for(base, "support"),
                                 split="support", split_seed=config.input_split_seed,
                                 balanced=True, seq_len=config.seq_len)
        query = sample_dataset(task, examples, seed=seed_for(base, "query"),
                               split="query", split_seed=config.input_split_seed,
                               balanced=True, seq_len=config.seq_len)
        result.append((task, {key: value.to(device) if key in ("x", "y") else value
                              for key, value in support.items()},
                       {key: value.to(device) if key in ("x", "y") else value
                        for key, value in query.items()}, seed_for(base, "v")))
    return result


def _score_input(decode, value: torch.Tensor, cases: list[tuple], config,
                 inner_steps: int) -> float:
    with torch.no_grad():
        u = tuple(part.detach() for part in decode(value))
    scores = []
    for _, support, query, v_seed in cases:
        v = adapt_v(u, support["x"], support["y"], steps=inner_steps,
                    lr=config.inner_lr, seed=v_seed, create_graph=False,
                    batch_size=min(config.batch_size, len(support["y"])),
                    optimizer=config.inner_optimizer, init_scale=config.init_scale)
        with torch.no_grad():
            logits = forward_with_u(query["x"], u, v,
                                    seq_len=config.seq_len, hidden=config.hidden)
            scores.append(float(F.binary_cross_entropy_with_logits(logits, query["y"])))
    return sum(scores) / len(scores)


def learn_unseen_code(model, config, length: int, device: str, *, outer_steps: int,
                      inner_steps: int, examples: int, code_lr: float) -> tuple[torch.Tensor, dict]:
    train_tasks, val_tasks, evaluation_tasks = _select_tasks(length)
    train = _calibration_data(train_tasks, config, device, "train", examples)
    validation = _calibration_data(val_tasks, config, device, "validation", examples)
    lower, upper = (4, 6) if length == 5 else (6, 8)
    with torch.no_grad():
        midpoint = (model.codes[TRAIN_LENGTHS.index(lower)] +
                    model.codes[TRAIN_LENGTHS.index(upper)]) / 2
    code = midpoint.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([code], lr=code_lr)
    best_score = _score_input(model.decode, code, validation, config, inner_steps)
    best_code = code.detach().clone()
    best_step = 0
    history = [{"step": 0, "calibration_validation_bce": best_score}]
    for step in range(1, outer_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        u = model.decode(code)
        objective = 0
        for _, support, query, v_seed in train:
            v = adapt_v(u, support["x"], support["y"], steps=inner_steps,
                        lr=config.inner_lr, seed=v_seed, create_graph=True,
                        batch_size=min(config.batch_size, len(support["y"])),
                        optimizer=config.inner_optimizer, init_scale=config.init_scale)
            logits = forward_with_u(query["x"], u, v,
                                    seq_len=config.seq_len, hidden=config.hidden)
            objective = objective + F.binary_cross_entropy_with_logits(logits, query["y"]) / len(train)
        objective.backward()
        torch.nn.utils.clip_grad_norm_([code], 5.0)
        optimizer.step()
        with torch.no_grad():
            code.mul_((4.0 / code.norm().clamp_min(1e-12)).clamp(max=1.0))
        if step % 10 == 0 or step == outer_steps:
            score = _score_input(model.decode, code, validation, config, inner_steps)
            history.append({"step": step, "calibration_validation_bce": score,
                            "calibration_train_bce": float(objective.detach())})
            if score < best_score:
                best_score, best_code, best_step = score, code.detach().clone(), step
            print(f"ADAPT_CODE length={length} step={step} val_bce={score:.6f} best={best_score:.6f}", flush=True)
    return best_code, {
        "length": length, "train_patterns": [task.pattern for task in train_tasks],
        "validation_patterns": [task.pattern for task in val_tasks],
        "evaluation_pattern_count": len(evaluation_tasks),
        "excluded_calibration_orbit_patterns": 2**length - len(evaluation_tasks),
        "selected_step": best_step,
        "selected_calibration_validation_bce": best_score, "midpoint_code": midpoint.cpu().tolist(),
        "selected_code": best_code.cpu().tolist(), "history": history,
    }


def learn_unseen_scalar(model, config, length: int, device: str, *, outer_steps: int,
                        inner_steps: int, examples: int, code_lr: float) -> tuple[torch.Tensor, dict]:
    """Calibrate the observed one-dimensional condition with the same episodes."""
    train_tasks, val_tasks, _ = _select_tasks(length)
    train = _calibration_data(train_tasks, config, device, "train", examples)
    validation = _calibration_data(val_tasks, config, device, "validation", examples)
    condition = torch.tensor(float(length), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([condition], lr=code_lr)
    best_score = _score_input(model, condition, validation, config, inner_steps)
    best_value = condition.detach().clone()
    best_step = 0
    history = [{"step": 0, "calibration_validation_bce": best_score}]
    for step in range(1, outer_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        u = model(condition)
        objective = 0
        for _, support, query, v_seed in train:
            v = adapt_v(u, support["x"], support["y"], steps=inner_steps,
                        lr=config.inner_lr, seed=v_seed, create_graph=True,
                        batch_size=min(config.batch_size, len(support["y"])),
                        optimizer=config.inner_optimizer, init_scale=config.init_scale)
            logits = forward_with_u(query["x"], u, v,
                                    seq_len=config.seq_len, hidden=config.hidden)
            objective = objective + F.binary_cross_entropy_with_logits(logits, query["y"]) / len(train)
        objective.backward()
        torch.nn.utils.clip_grad_norm_([condition], 5.0)
        optimizer.step()
        with torch.no_grad():
            condition.clamp_(model.length_min, model.length_max)
        if step % 10 == 0 or step == outer_steps:
            score = _score_input(model, condition, validation, config, inner_steps)
            history.append({"step": step, "calibration_validation_bce": score,
                            "calibration_train_bce": float(objective.detach())})
            if score < best_score:
                best_score, best_value, best_step = score, condition.detach().clone(), step
            print(f"ADAPT_SCALAR length={length} step={step} val_bce={score:.6f} best={best_score:.6f}", flush=True)
    return best_value, {"length": length, "selected_step": best_step,
                        "selected_calibration_validation_bce": best_score,
                        "selected_input_length": float(best_value), "history": history}


def _episodes(tasks: list[PatternTask], config, repeats: int, support_size: int, test_size: int):
    result = []
    for task in tasks:
        for repeat in range(repeats):
            base = seed_for("learned-code-evaluation", EVALUATION_SEED, task.pattern, repeat)
            support = sample_dataset(task, support_size, seed=seed_for(base, "support"),
                                     split="support", split_seed=config.input_split_seed,
                                     balanced=True, seq_len=config.seq_len)
            test = sample_dataset(task, test_size, seed=seed_for(base, "test"),
                                  split="test", split_seed=config.input_split_seed,
                                  balanced=True, seq_len=config.seq_len)
            result.append((task.pattern, repeat, seed_for(base, "v"), support, test))
    return result


def _evaluate_variants(variants: dict[str, tuple], episodes: list[tuple], config, device: str,
                       steps: tuple[int, ...], chunk_size: int) -> list[dict]:
    rows = []
    for variant, u in variants.items():
        for offset in range(0, len(episodes), chunk_size):
            chunk = episodes[offset:offset + chunk_size]
            x = torch.stack([item[3]["x"] for item in chunk]).to(device)
            y = torch.stack([item[3]["y"] for item in chunk]).to(device)
            tx = torch.stack([item[4]["x"] for item in chunk]).to(device)
            ty = torch.stack([item[4]["y"] for item in chunk]).to(device)
            n = len(chunk)
            snapshots = batched_adapt_v(
                u, x, y, steps=max(steps),
                lrs=torch.full((n,), config.inner_lr, device=device),
                seeds=torch.tensor([item[2] for item in chunk], dtype=torch.int64),
                optimizers=[config.inner_optimizer] * n,
                init_scales=torch.full((n,), config.init_scale, device=device),
                batch_size=min(config.batch_size, y.shape[1]), checkpoints=steps,
            )
            with torch.no_grad():
                for step in steps:
                    logits = _batched_logits(tx, u, snapshots[step])
                    losses = F.binary_cross_entropy_with_logits(logits, ty, reduction="none").mean(1)
                    accuracies = ((logits > 0) == ty.bool()).float().mean(1)
                    if not bool(torch.isfinite(losses).all()):
                        raise FloatingPointError(f"non-finite evaluation for {variant}")
                    for item, loss, accuracy in zip(chunk, losses, accuracies):
                        rows.append({"pattern": item[0], "repeat": item[1], "variant": variant,
                                     "steps": step, "test_bce": float(loss),
                                     "test_accuracy": float(accuracy)})
        print(f"EVAL variant={variant} episodes={len(episodes)}", flush=True)
    return rows


def evaluate(code_checkpoint: Path, scalar_checkpoint: Path,
             unconditional_checkpoint: Path, output: Path, *, device: str,
             calibration_steps: int, calibration_inner_steps: int,
             calibration_examples: int, code_lr: float, repeats: int,
             support_size: int, test_size: int, eval_steps: tuple[int, ...],
             chunk_size: int, evaluation_patterns_per_length: int | None = None,
             known_patterns_per_length: int | None = None) -> dict:
    if output.exists():
        raise FileExistsError(output)
    code_state, config, model = load(code_checkpoint, device)
    scalar_state, scalar_config, scalar = load_interpolation_checkpoint(scalar_checkpoint, device)
    uncond_state, uncond_config, unconditional = load_interpolation_checkpoint(unconditional_checkpoint, device)
    if not scalar_config.condition_length or uncond_config.condition_length:
        raise ValueError("incorrect scalar or unconditional baseline")
    if (evaluation_patterns_per_length is not None and evaluation_patterns_per_length < 1 or
            known_patterns_per_length is not None and known_patterns_per_length < 1):
        raise ValueError("pattern limits must be positive")
    if any(getattr(config, key) != getattr(scalar_config, key) or
           getattr(config, key) != getattr(uncond_config, key)
           for key in ("rank1", "rank2", "seq_len", "hidden", "inner_lr", "inner_optimizer", "init_scale")):
        raise ValueError("mismatched adaptation protocol or U shape")
    for parameter in model.network.parameters():
        parameter.requires_grad_(False)
    for parameter in scalar.parameters():
        parameter.requires_grad_(False)
    rows = []
    calibration = {}
    scalar_calibration = {}
    task_counts = {}
    with torch.no_grad():
        trained_codes = {length: model.codes[TRAIN_LENGTHS.index(length)].detach().clone()
                         for length in TRAIN_LENGTHS}
    for length in HELD_LENGTHS:
        optimized, detail = learn_unseen_code(
            model, config, length, device, outer_steps=calibration_steps,
            inner_steps=calibration_inner_steps, examples=calibration_examples,
            code_lr=code_lr,
        )
        calibration[str(length)] = detail
        adapted_scalar, scalar_detail = learn_unseen_scalar(
            scalar, scalar_config, length, device, outer_steps=calibration_steps,
            inner_steps=calibration_inner_steps, examples=calibration_examples,
            code_lr=code_lr,
        )
        scalar_calibration[str(length)] = scalar_detail
        lower, upper = (4, 6) if length == 5 else (6, 8)
        train_tasks, val_tasks, evaluation_tasks = _select_tasks(length)
        evaluation_tasks.sort(key=lambda task: seed_for("code-evaluation-pattern-selection",
                                                   EVALUATION_SEED, length, task.pattern))
        if evaluation_patterns_per_length is not None:
            evaluation_tasks = evaluation_tasks[:evaluation_patterns_per_length]
        episodes = _episodes(evaluation_tasks, config, repeats, support_size, test_size)
        task_counts[str(length)] = {"calibration_train": len(train_tasks),
                                    "calibration_validation": len(val_tasks),
                                    "evaluation": len(evaluation_tasks), "episodes": len(episodes)}
        with torch.no_grad():
            variants = {
                "code_midpoint": tuple(value.detach() for value in model.decode(
                    (trained_codes[lower] + trained_codes[upper]) / 2)),
                "code_adapted": tuple(value.detach() for value in model.decode(optimized)),
                "code_lower": tuple(value.detach() for value in model.decode(trained_codes[lower])),
                "code_upper": tuple(value.detach() for value in model.decode(trained_codes[upper])),
                "scalar_correct": tuple(value.detach() for value in scalar(length)),
                "scalar_adapted": tuple(value.detach() for value in scalar(adapted_scalar)),
                "scalar_unconditional": tuple(value.detach() for value in unconditional(length)),
            }
        length_rows = _evaluate_variants(variants, episodes, config, device, eval_steps, chunk_size)
        rows.extend({"target_length": length, **row} for row in length_rows)

    known_rows = []
    for length in TRAIN_LENGTHS:
        test_tasks = [task for task in task_splits(config)["test"] if task.length == length]
        test_tasks.sort(key=lambda task: seed_for("code-known-pattern-selection",
                                              EVALUATION_SEED, length, task.pattern))
        if known_patterns_per_length is not None:
            test_tasks = test_tasks[:known_patterns_per_length]
        episodes = _episodes(test_tasks, config, repeats, support_size, test_size)
        with torch.no_grad():
            variants = {f"code_{source}": tuple(value.detach() for value in model.decode(code))
                        for source, code in trained_codes.items()}
        known_steps = tuple(step for step in (50, 500) if step <= max(eval_steps)) or (max(eval_steps),)
        length_rows = _evaluate_variants(variants, episodes, config, device, known_steps, chunk_size)
        known_rows.extend({"target_length": length, **row} for row in length_rows)

    result = {
        "protocol": {"evaluation_seed": EVALUATION_SEED,
                     "calibration_pattern_orbits_disjoint_from_evaluation": True,
                     "held_lengths_never_used_to_train_decoder": True,
                     "support_query_test_input_hash_partitions": True,
                     "fresh_paired_v_for_every_structure": True,
                     "evaluation_patterns_per_length": evaluation_patterns_per_length,
                     "known_patterns_per_length": known_patterns_per_length,
                     "pattern_selection": "deterministic hash order, independent of checkpoint"},
        "checkpoints": {name: {"path": str(path.resolve()), "sha256": _sha256(path),
                               "step": state.get("step")}
                        for name, path, state in (("code", code_checkpoint, code_state),
                                                  ("scalar", scalar_checkpoint, scalar_state),
                                                  ("unconditional", unconditional_checkpoint, uncond_state))},
        "config": config.to_dict(), "latent_dim": len(next(iter(trained_codes.values()))),
        "trained_codes": {str(length): value.cpu().tolist() for length, value in trained_codes.items()},
        "calibration": calibration, "scalar_calibration": scalar_calibration,
        "task_counts": task_counts,
        "evaluation": {"repeats": repeats, "support_size": support_size,
                       "test_size": test_size, "steps": list(eval_steps),
                       "calibration_steps": calibration_steps,
                       "calibration_inner_steps": calibration_inner_steps,
                       "calibration_examples": calibration_examples, "code_lr": code_lr},
        "held_rows": rows, "known_condition_matrix_rows": known_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-checkpoint", type=Path, required=True)
    parser.add_argument("--scalar-checkpoint", type=Path, required=True)
    parser.add_argument("--unconditional-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calibration-steps", type=int, default=80)
    parser.add_argument("--calibration-inner-steps", type=int, default=20)
    parser.add_argument("--calibration-examples", type=int, default=512)
    parser.add_argument("--code-lr", type=float, default=0.03)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--support-size", type=int, default=8192)
    parser.add_argument("--test-size", type=int, default=2048)
    parser.add_argument("--eval-steps", nargs="+", type=int, default=[50, 500, 2000])
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--evaluation-patterns-per-length", type=int)
    parser.add_argument("--known-patterns-per-length", type=int)
    args = parser.parse_args()
    evaluate(args.code_checkpoint, args.scalar_checkpoint, args.unconditional_checkpoint,
             args.out, device=args.device, calibration_steps=args.calibration_steps,
             calibration_inner_steps=args.calibration_inner_steps,
             calibration_examples=args.calibration_examples, code_lr=args.code_lr,
             repeats=args.repeats, support_size=args.support_size, test_size=args.test_size,
             eval_steps=tuple(sorted(set(args.eval_steps))), chunk_size=args.chunk_size,
             evaluation_patterns_per_length=args.evaluation_patterns_per_length,
             known_patterns_per_length=args.known_patterns_per_length)


if __name__ == "__main__":
    main()
