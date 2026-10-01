"""Recovery-v2 evaluator: exact 6k replay followed by a common LR annealing tail.

The frozen adaptive evaluator remains the authority for the first 6,000
updates.  This independent implementation keeps its numerical recipe and
only changes learning rates after update 6,000.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from . import core
from .adaptive_eval import (BatchedAdaptiveConditions, _loss, _move_splits,
                            _optimizer_snapshot, _plateau, _replicas,
                            _restore_optimizer_rows, _sets)


@dataclass(frozen=True)
class ContinuedConfig:
    min_steps: int = 800
    max_steps: int = 18000
    eval_every: int = 50
    plateau_window: int = 8
    plateau_tolerance: float = .01
    checkpoint_relative_improvement: float = .001
    checkpoint_floor: float = .01
    checkpoint_patience: int = 400
    required_plateau_passes: int = 3
    batch_size: int = 32
    set_size: int = 5
    score_sets: int = 512
    reference_models: int = 20
    batch_conditions: int = 8
    replay_steps: int = 6000
    decay_interval: int = 1000
    minimum_multiplier: float = 1 / 16


def lr_multiplier(step: int, cfg: ContinuedConfig) -> float:
    """Base through 6000; /2 at 6001,...,/16 at 9001 then fixed."""
    if step <= cfg.replay_steps:
        return 1.0
    reductions = min(4, 1 + (step - cfg.replay_steps - 1) // cfg.decay_interval)
    return 0.5 ** reductions


def _cpu_optimizer_state(optimizers: list[torch.optim.Adam]) -> list[dict[str, Any]]:
    result = []
    for optimizer in optimizers:
        state = optimizer.state_dict()
        state["state"] = {key: {name: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
                                for name, value in values.items()}
                          for key, values in state["state"].items()}
        result.append(state)
    return result


def _logical_state(model: BatchedAdaptiveConditions) -> dict[str, torch.Tensor]:
    state = model.snapshot()
    state["masks"] = model.masks[:, model.inverse].detach().clone()
    state["effective_weight"] = state["weight"] * state["masks"]
    return state


def evaluate_continued(
    splits: dict[str, core.Split], costs: torch.Tensor | np.ndarray, masks: dict[str, torch.Tensor],
    manifest: list[dict[str, Any]], *, seed: int, device: str | torch.device, phase: str,
    cfg: ContinuedConfig = ContinuedConfig(), score_key: str, support_sizes: Sequence[int] = (32, 64, 128, 256),
    conditions_override: Sequence[tuple[int, int]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate selected global condition rows while preparing all task RNG streams.

    ``conditions_override`` uses indices in the complete supplied cost table;
    this lets shards retain the original task-index seed offsets.
    """
    device = torch.device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    names = [row["method"] for row in manifest]
    if len(names) != len(masks) or set(names) != set(masks):
        raise ValueError("manifest/masks mismatch")
    support_sizes = tuple(map(int, support_sizes))
    if not support_sizes or min(support_sizes) < 2: raise ValueError("invalid budgets")
    stacked = torch.cat([masks[name].to(device=device, dtype=torch.float32) for name in names])
    logical_methods = [name for name in names for _ in range(4)]
    lrs = [float(row["lr"]) for row in manifest for _ in range(4)]
    replicas = _replicas(logical_methods)
    by_name = {row["method"]: row for row in manifest}
    train, checkpoint, score = _move_splits({"train": splits[f"{phase}_train"],
                                              "checkpoint": splits[f"{phase}_checkpoint"],
                                              "score": splits[score_key]}, device).values()
    cost_values = torch.as_tensor(costs, device=device, dtype=torch.float32)
    if cost_values.ndim != 2 or cost_values.shape[1] != 10: raise ValueError("cost table must be [tasks,10]")
    cost_values = torch.stack([core.centred_costs(row, device) for row in cost_values])
    valid = {budget: max(1, round(.2 * budget)) for budget in support_sizes}
    fitted = {budget: budget - valid[budget] for budget in support_sizes}
    prepared = {}
    offset = 500000 if phase == "selection" else 600000
    for task, cost in enumerate(cost_values):
        generator = torch.Generator(device=device).manual_seed(seed + offset + 10007 * (task + 1))
        sx, sy = _sets(train, cost, max(support_sizes), cfg.set_size, generator)
        vx, vy = _sets(checkpoint, cost, max(valid.values()), cfg.set_size, generator)
        tx, ty = _sets(score, cost, cfg.score_sets, cfg.set_size, generator)
        prepared[task] = (sx, sy, vx, vy, tx, ty)
    conditions = list(conditions_override) if conditions_override is not None else [
        (task, budget) for task in range(len(cost_values)) for budget in support_sizes]
    if not conditions or any(task not in prepared or budget not in fitted for task, budget in conditions):
        raise ValueError("invalid condition override")
    records: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {"curves": {}, "best_states": {}, "last_states": {}, "step800": {}, "step6000": {},
                                  "schedule": {"replay_steps": cfg.replay_steps, "decay_steps": [6001, 7001, 8001, 9001],
                                               "multipliers": [1.0, .5, .25, .125, .0625], "fixed_after": 9001}}
    for first in range(0, len(conditions), cfg.batch_conditions):
        group = conditions[first:first + cfg.batch_conditions]
        print({"continued_progress": "group_start", "phase": phase, "seed": seed, "conditions": group,
               "models": len(logical_methods)}, flush=True)
        model = BatchedAdaptiveConditions(stacked, lrs, [seed + 3001 * task + budget for task, budget in group], replicas,
                                          cfg.reference_models).to(device)
        optimizers = [torch.optim.Adam(model.chunk_parameters(index), lr=base) for index, (base, _, _) in enumerate(model.groups)]
        batch_generators = [torch.Generator(device=device).manual_seed(seed + offset + 70001 * (task + 1) + budget)
                            for task, budget in group]
        c, m = len(group), len(logical_methods)
        best = torch.full((c, m), float("inf"), device=device); best_step = torch.zeros((c, m), dtype=torch.long, device=device)
        best_train = torch.full_like(best, float("nan")); best_state = model.snapshot()
        last_improved = torch.zeros((c, m), dtype=torch.long, device=device)
        consecutive = torch.zeros((c, m), dtype=torch.long, device=device); frozen = torch.zeros((c, m), dtype=torch.bool, device=device)
        stopped = torch.zeros((c, m), dtype=torch.long, device=device)
        frozen_state: dict[str, torch.Tensor] | None = None
        frozen_optim: list[list[dict[str, torch.Tensor]]] | None = None
        train_history: list[list[torch.Tensor]] = [[] for _ in group]; check_history: list[list[torch.Tensor]] = [[] for _ in group]
        curve_steps: list[int] = []; curves_train: list[torch.Tensor] = []; curves_check: list[torch.Tensor] = []
        for step in range(1, cfg.max_steps + 1):
            multiplier = lr_multiplier(step, cfg)
            for optimizer, (base, _, _) in zip(optimizers, model.groups): optimizer.param_groups[0]["lr"] = base * multiplier
            xs, ys = [], []
            for (task, budget), generator in zip(group, batch_generators):
                indices = torch.randint(fitted[budget], (cfg.batch_size,), generator=generator, device=device)
                sx, sy, *_ = prepared[task]; xs.append(sx[indices]); ys.append(sy[indices])
            loss = (model(torch.stack(xs)) - torch.stack(ys)[:, None]).square().mean(-1) / cfg.set_size
            for optimizer in optimizers: optimizer.zero_grad(set_to_none=True)
            (loss.sum() / cfg.reference_models).backward()
            for group_index, (_, lo, hi) in enumerate(model.groups):
                dead = frozen[:, model.order[lo:hi]]
                for parameter in model.chunk_parameters(group_index): parameter.grad[dead] = 0
            for optimizer in optimizers: optimizer.step()
            if frozen_state is not None:
                model.restore(frozen_state, frozen)
                assert frozen_optim is not None
                _restore_optimizer_rows(optimizers, frozen_optim, model, frozen)
            if step % cfg.eval_every and step != cfg.max_steps: continue
            train_rows, checkpoint_rows = [], []
            with torch.no_grad():
                for condition, (task, budget) in enumerate(group):
                    sx, sy, vx, vy, *_ = prepared[task]
                    train_rows.append(_loss(model, condition, sx[:fitted[budget]], sy[:fitted[budget]], cfg.set_size))
                    checkpoint_rows.append(_loss(model, condition, vx[:valid[budget]], vy[:valid[budget]], cfg.set_size))
            train_loss, checkpoint_loss = torch.stack(train_rows), torch.stack(checkpoint_rows)
            significant = checkpoint_loss < best - cfg.checkpoint_relative_improvement * torch.maximum(
                best.abs(), torch.full_like(best, cfg.checkpoint_floor))
            improved = (significant | ~torch.isfinite(best)) & ~frozen
            state = model.snapshot()
            for field in best_state: best_state[field][improved] = state[field][improved]
            best[improved] = checkpoint_loss[improved]; best_step[improved] = step; best_train[improved] = train_loss[improved]
            last_improved[improved] = step
            for condition in range(c): train_history[condition].append(train_loss[condition]); check_history[condition].append(checkpoint_loss[condition])
            passes = torch.stack([_plateau(train_history[i], cfg) & _plateau(check_history[i], cfg) for i in range(c)]) if step >= cfg.min_steps else torch.zeros_like(frozen)
            eligible = (step >= cfg.min_steps) & ((step - last_improved) >= cfg.checkpoint_patience)
            consecutive = torch.where(passes & eligible & ~frozen, consecutive + 1,
                                      torch.where(frozen, consecutive, torch.zeros_like(consecutive)))
            newly = (consecutive >= cfg.required_plateau_passes) & ~frozen
            if newly.any():
                model.restore(best_state, newly)
                frozen_state = model.snapshot() if frozen_state is None else frozen_state
                for field in frozen_state: frozen_state[field][newly] = best_state[field][newly]
                now_optim = _optimizer_snapshot(optimizers)
                if frozen_optim is None: frozen_optim = now_optim
                else:
                    for group_index, (old, current) in enumerate(zip(frozen_optim, now_optim)):
                        _, lo, hi = model.groups[group_index]; rows = newly[:, model.order[lo:hi]]
                        for old_param, now_param in zip(old, current):
                            for name in old_param: old_param[name][rows] = now_param[name][rows]
                frozen |= newly; stopped[newly] = step
            curve_steps.append(step); curves_train.append(train_loss.detach().cpu()); curves_check.append(checkpoint_loss.detach().cpu())
            key = str(group)
            if step == 800:
                artifacts["step800"][key] = {"state": {name: value.detach().cpu() for name, value in _logical_state(model).items()},
                                               "train": train_loss.detach().cpu(), "checkpoint": checkpoint_loss.detach().cpu()}
            if step == cfg.replay_steps:
                artifacts["step6000"][key] = {"last_state": {name: value.detach().cpu() for name, value in _logical_state(model).items()},
                                                "best_state": {name: value.detach().cpu() for name, value in best_state.items()},
                                                "optimizer": _cpu_optimizer_state(optimizers), "frozen": frozen.detach().cpu(),
                                                "best_step": best_step.detach().cpu(), "last_improved": last_improved.detach().cpu()}
            if step % 400 == 0 or step == cfg.max_steps or bool(frozen.all()):
                print({"continued_progress": "checkpoint", "phase": phase, "seed": seed, "conditions": group, "step": step,
                       "multiplier": multiplier, "mean_train": float(train_loss.mean()), "mean_checkpoint": float(checkpoint_loss.mean()),
                       "frozen": int(frozen.sum()), "models": m}, flush=True)
            if bool(frozen.all()): break
        final_step = step
        last_state = _logical_state(model)
        model.restore(best_state)
        for condition, (task, budget) in enumerate(group):
            *_, tx, ty = prepared[task]; score_loss = _loss(model, condition, tx, ty, cfg.set_size)
            for index, (method, replica) in enumerate(zip(logical_methods, replicas)):
                row = by_name[method]; stop = int(stopped[condition, index]) or final_step
                records.append({"phase": phase, "task": task, "support_size": budget, "method": method, "init": replica,
                                "family": row["family"], "rho": row["rho"], "lr": row["lr"], "initial_lr": row["lr"],
                                "final_lr_effective": float(row["lr"]) * lr_multiplier(stop, cfg), "score_mse": float(score_loss[index]),
                                "checkpoint_mse": float(best[condition, index]), "best_train_mse": float(best_train[condition, index]),
                                "best_step": int(best_step[condition, index]), "stop_step": stop, "converged": bool(frozen[condition, index]),
                                "status": "converged" if frozen[condition, index] else "max_cap"})
        key = str(group)
        artifacts["curves"][key] = {"steps": np.asarray(curve_steps, dtype=np.int32), "train": torch.stack(curves_train).numpy(),
                                      "checkpoint": torch.stack(curves_check).numpy(), "stop_steps": stopped.detach().cpu().numpy()}
        artifacts["best_states"][key] = {name: value.detach().cpu() for name, value in _logical_state(model).items()}
        artifacts["last_states"][key] = {name: value.detach().cpu() for name, value in last_state.items()}
    if not all(np.isfinite(row["score_mse"]) for row in records): raise AssertionError("non-finite recovery score")
    return records, artifacts


def prefix_equivalence_test() -> bool:
    """The recovery schedule is exactly base LR through the replay boundary."""
    cfg = ContinuedConfig(max_steps=100, replay_steps=6000)
    return all(lr_multiplier(step, cfg) == 1.0 for step in range(1, 101)) and (
        lr_multiplier(6001, cfg) == .5 and lr_multiplier(9001, cfg) == .0625 and lr_multiplier(18000, cfg) == .0625)


def tiny_prefix_tensor_test() -> bool:
    """CPU tensor-level comparison to the frozen evaluator for a 100-step prefix."""
    from .adaptive_eval import EvalConfig, evaluate_adaptive
    device = "cpu"; generator = torch.Generator().manual_seed(73)
    split = core.Split(torch.rand(40, 784, generator=generator), torch.arange(40) % 10, torch.arange(40))
    splits = {"selection_train": split, "selection_checkpoint": split, "selection_score": split}
    masks = {"tiny": torch.ones(4, 784, 1)}
    manifest = [{"method": "tiny", "family": "tiny", "rho": 1., "lr": .002}]
    costs = torch.tensor([[.3, -.1, .4, -.2, .1, .2, -.3, .5, -.4, .0]])
    old_records, old_artifacts = evaluate_adaptive(splits, costs, masks, manifest, seed=91, device=device,
                                                    phase="selection", cfg=EvalConfig(max_steps=100, batch_size=2, score_sets=2,
                                                    batch_conditions=1), score_key="selection_score", support_sizes=(32,))
    new_records, new_artifacts = evaluate_continued(splits, costs, masks, manifest, seed=91, device=device,
                                                     phase="selection", cfg=ContinuedConfig(max_steps=100, batch_size=2, score_sets=2,
                                                     batch_conditions=1), score_key="selection_score", support_sizes=(32,),
                                                     conditions_override=[(0, 32)])
    old_state = next(iter(old_artifacts["states"].values())); new_state = next(iter(new_artifacts["best_states"].values()))
    tensors = all(torch.equal(old_state[name], new_state[name]) for name in ("masks", "weight", "effective_weight", "bias", "readout", "per_image_offset"))
    rows = all(a["score_mse"] == b["score_mse"] and a["checkpoint_mse"] == b["checkpoint_mse"] and
               a["best_step"] == b["best_step"] for a, b in zip(old_records, new_records))
    return tensors and rows
