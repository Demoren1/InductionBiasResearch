"""Batched, convergence-aware evaluation for adaptive-density selection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from . import core


@dataclass
class EvalConfig:
    min_steps: int = 800
    max_steps: int = 6000
    eval_every: int = 50
    plateau_window: int = 8             # eight 50-update evaluations = 400 updates
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


def _replicas(names: list[str]) -> list[int]:
    seen: dict[str, int] = {}; result = []
    for name in names:
        result.append(seen.get(name, 0)); seen[name] = result[-1] + 1
    return result


class BatchedAdaptiveConditions(nn.Module):
    """C conditions and M models; each LR owns actual Adam parameter chunks."""
    fields = ("weight", "bias", "readout", "per_image_offset")

    def __init__(self, masks: torch.Tensor, lrs: list[float], condition_seeds: list[int], replicas: list[int],
                 reference_models: int = 20) -> None:
        super().__init__()
        if masks.shape[0] != len(lrs) or masks.ndim != 3:
            raise ValueError("mask/LR mismatch")
        self.c, self.m = len(condition_seeds), len(lrs)
        order = sorted(range(self.m), key=lambda i: (lrs[i], i))
        order_tensor = torch.tensor(order, device=masks.device)
        inverse = torch.empty(self.m, dtype=torch.long, device=masks.device)
        inverse[order_tensor] = torch.arange(self.m, device=masks.device)
        self.register_buffer("order", order_tensor); self.register_buffer("inverse", inverse)
        self.register_buffer("masks", masks[self.order][None].expand(self.c, -1, -1, -1).clone())
        self.groups: list[tuple[float, int, int]] = []
        start = 0
        while start < self.m:
            lr = lrs[order[start]]; end = start + 1
            while end < self.m and lrs[order[end]] == lr: end += 1
            self.groups.append((lr, start, end)); start = end
        exemplar: dict[int, int] = {}
        for idx, replica in enumerate(replicas): exemplar.setdefault(replica, idx)
        exemplar_original = torch.tensor([exemplar[r] for r in replicas], device=masks.device)
        chunks: dict[str, list[torch.Tensor]] = {field: [] for field in self.fields}
        for seed in condition_seeds:
            base = core.MaskedDeepSets(masks, seed=seed, initialization_reference_models=reference_models)
            for field, parameter in base.named_parameters():
                chunks[field].append(parameter.detach()[exemplar_original].clone()[self.order])
        for field in self.fields:
            values = torch.stack(chunks[field])
            for group_idx, (_, lo, hi) in enumerate(self.groups):
                self.register_parameter(f"{field}_{group_idx}", nn.Parameter(values[:, lo:hi].clone()))

    def chunk_parameters(self, group_idx: int) -> list[nn.Parameter]:
        return [getattr(self, f"{field}_{group_idx}") for field in self.fields]

    def joined(self, field: str) -> torch.Tensor:
        return torch.cat([getattr(self, f"{field}_{idx}") for idx in range(len(self.groups))], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Explicit bmm: [C,B,S,F] × [C,F,MH], returning original logical order.
        c, b, s, f = x.shape
        weight = self.joined("weight") * self.masks
        _, m, _, h = weight.shape
        packed = weight.permute(0, 2, 1, 3).reshape(c, f, m * h)
        pre = torch.bmm(x.reshape(c, b * s, f), packed).reshape(c, b, s, m, h).permute(0, 3, 1, 2, 4)
        output = ((torch.tanh(pre + self.joined("bias")[:, :, None, None, :]) *
                   self.joined("readout")[:, :, None, None, :]).sum(-1) +
                  self.joined("per_image_offset")[:, :, None, None]).sum(-1)
        return output[:, self.inverse]

    def snapshot(self) -> dict[str, torch.Tensor]:
        # External states and all loss/record arrays use logical method order.
        return {field: self.joined(field).detach().clone()[:, self.inverse] for field in self.fields}

    @torch.no_grad()
    def restore(self, state: dict[str, torch.Tensor], logical_rows: torch.Tensor | None = None) -> None:
        for field in self.fields:
            source = state[field][:, self.order]
            for idx, (_, lo, hi) in enumerate(self.groups):
                target = getattr(self, f"{field}_{idx}")
                if logical_rows is None:
                    target.copy_(source[:, lo:hi])
                else:
                    rows = logical_rows[:, self.order[lo:hi]]
                    target[rows] = source[:, lo:hi][rows]


def _sets(split: core.Split, costs: torch.Tensor, count: int, size: int, gen: torch.Generator):
    return core._make_target_sets(split, costs, count, size, gen)


@torch.no_grad()
def _loss(model: BatchedAdaptiveConditions, condition: int, x: torch.Tensor, y: torch.Tensor, size: int) -> torch.Tensor:
    total = torch.zeros(model.m, device=x.device)
    for start in range(0, len(x), 128):
        batch = x[start:start + 128]
        # This condition view preserves the same explicit BMM kernel without
        # pretending one condition has the parameter batch size of the group.
        weight = model.joined("weight")[condition] * model.masks[condition]
        m, f, h = weight.shape
        packed = weight.permute(1, 0, 2).reshape(f, m * h)
        pre = torch.mm(batch.reshape(-1, f), packed).reshape(len(batch), size, m, h).permute(2, 0, 1, 3)
        prediction = ((torch.tanh(pre + model.joined("bias")[condition, :, None, None, :]) *
                       model.joined("readout")[condition, :, None, None, :]).sum(-1) +
                      model.joined("per_image_offset")[condition, :, None, None]).sum(-1)
        total += (prediction - y[start:start + 128][None]).square().sum(1)
    return (total / len(x) / size)[model.inverse]


def _plateau(history: list[torch.Tensor], cfg: EvalConfig) -> torch.Tensor:
    width = cfg.plateau_window
    if len(history) < 2 * width: return torch.zeros_like(history[-1], dtype=torch.bool)
    values = torch.stack(history[-2 * width:])
    before, after = values[:width], values[width:]
    denom = torch.maximum(torch.maximum(before.mean(0).abs(), after.mean(0).abs()), torch.full_like(after[0], .01))
    change = (after.mean(0) - before.mean(0)).abs() / denom
    index = torch.arange(width, device=values.device, dtype=values.dtype) - (width - 1) / 2
    slope = (after * index[:, None]).sum(0) / index.square().sum()
    trend = (slope.abs() * width) / denom
    return (change <= cfg.plateau_tolerance) & (trend <= cfg.plateau_tolerance)


def _move_splits(splits: dict[str, core.Split], device: torch.device) -> dict[str, core.Split]:
    return {name: core.Split(*(value.to(device) for value in split)) for name, split in splits.items()}


def _optimizer_snapshot(optimizers: list[torch.optim.Adam]) -> list[list[dict[str, torch.Tensor]]]:
    """Save per-model Adam buffers; scalar group steps do not move a frozen row."""
    result: list[list[dict[str, torch.Tensor]]] = []
    for optimizer in optimizers:
        rows = []
        for parameter in optimizer.param_groups[0]["params"]:
            rows.append({name: value.detach().clone() for name, value in optimizer.state[parameter].items()
                         if isinstance(value, torch.Tensor) and value.ndim > 0})
        result.append(rows)
    return result


@torch.no_grad()
def _restore_optimizer_rows(optimizers: list[torch.optim.Adam], snapshot: list[list[dict[str, torch.Tensor]]],
                            model: BatchedAdaptiveConditions, frozen: torch.Tensor) -> None:
    for group_idx, optimizer in enumerate(optimizers):
        _, lo, hi = model.groups[group_idx]
        rows = frozen[:, model.order[lo:hi]]
        for parameter, saved in zip(optimizer.param_groups[0]["params"], snapshot[group_idx]):
            for name, value in saved.items():
                optimizer.state[parameter][name][rows] = value[rows]


def evaluate_adaptive(splits: dict[str, core.Split], costs: torch.Tensor, masks: dict[str, torch.Tensor],
                      manifest: list[dict[str, Any]], *, seed: int, device: str | torch.device,
                      phase: str, cfg: EvalConfig = EvalConfig(), score_key: str = "selection_score",
                      support_sizes: tuple[int, ...] = (32, 64, 128, 256)) -> tuple[list[dict], dict[str, Any]]:
    """Fit all methods/LRs on grouped conditions, checkpointing only the stage checkpoint pool."""
    device = torch.device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if len(manifest) != len(masks) or set(row["method"] for row in manifest) != set(masks):
        raise ValueError("manifest and masks must have identical methods")
    names = [row["method"] for row in manifest]
    stacked = torch.cat([masks[name].to(device=device, dtype=torch.float32) for name in names])
    method_for_model = [name for name in names for _ in range(4)]
    lrs = [float(row["lr"]) for row in manifest for _ in range(4)]
    reps = _replicas(method_for_model)
    train, checkpoint, score = _move_splits({"train": splits[f"{phase}_train"],
                                              "checkpoint": splits[f"{phase}_checkpoint"],
                                              "score": splits[score_key]}, device).values()
    costs = torch.as_tensor(costs, device=device, dtype=torch.float32)
    costs = torch.stack([core.centred_costs(row, device) for row in costs])
    if not support_sizes or min(support_sizes) < 2: raise ValueError("support sizes must be >=2")
    valid_counts = {budget: max(1, round(.2 * budget)) for budget in support_sizes}
    train_counts = {budget: budget - valid_counts[budget] for budget in valid_counts}
    prepared = {}
    for task, cost in enumerate(costs):
        gen = torch.Generator(device=device).manual_seed(seed + (500000 if phase == "selection" else 600000) + 10007 * (task + 1))
        support_x, support_y = _sets(train, cost, max(support_sizes), cfg.set_size, gen)
        check_x, check_y = _sets(checkpoint, cost, max(valid_counts.values()), cfg.set_size, gen)
        score_x, score_y = _sets(score, cost, cfg.score_sets, cfg.set_size, gen)
        prepared[task] = support_x, support_y, check_x, check_y, score_x, score_y
    records: list[dict] = []; artifacts: dict[str, Any] = {"curves": {}, "states": {}, "step800": {}}
    conditions = [(task, budget) for task in range(len(costs)) for budget in support_sizes]
    for start in range(0, len(conditions), cfg.batch_conditions):
        group = conditions[start:start + cfg.batch_conditions]
        print({"adaptive_progress": "group_start", "phase": phase, "seed": seed,
               "conditions": group, "models": len(method_for_model)}, flush=True)
        model = BatchedAdaptiveConditions(stacked, lrs, [seed + 3001 * task + budget for task, budget in group], reps,
                                          cfg.reference_models).to(device)
        optimizers = [torch.optim.Adam(model.chunk_parameters(idx), lr=lr) for idx, (lr, _, _) in enumerate(model.groups)]
        batch_gens = [torch.Generator(device=device).manual_seed(seed + (500000 if phase == "selection" else 600000) + 70001 * (task + 1) + budget)
                      for task, budget in group]
        c, m = len(group), len(method_for_model)
        best = torch.full((c, m), float("inf"), device=device); best_step = torch.zeros((c, m), dtype=torch.long, device=device)
        best_train = torch.full_like(best, float("nan")); best_state = model.snapshot()
        last_improved = torch.zeros((c, m), dtype=torch.long, device=device)
        consecutive = torch.zeros((c, m), dtype=torch.long, device=device); frozen = torch.zeros((c, m), dtype=torch.bool, device=device)
        frozen_state: dict[str, torch.Tensor] | None = None
        frozen_optimizer_state: list[list[dict[str, torch.Tensor]]] | None = None
        stopped = torch.zeros((c, m), dtype=torch.long, device=device)
        train_history: list[list[torch.Tensor]] = [[] for _ in group]; val_history: list[list[torch.Tensor]] = [[] for _ in group]
        curve_steps: list[int] = []; curve_train: list[torch.Tensor] = []; curve_val: list[torch.Tensor] = []
        for step in range(1, cfg.max_steps + 1):
            xs=[]; ys=[]
            for (task, budget), generator in zip(group, batch_gens):
                indices = torch.randint(train_counts[budget], (cfg.batch_size,), generator=generator, device=device)
                sx, sy, *_ = prepared[task]; xs.append(sx[indices]); ys.append(sy[indices])
            loss = (model(torch.stack(xs)) - torch.stack(ys)[:, None]).square().mean(-1) / cfg.set_size
            for optimizer in optimizers: optimizer.zero_grad(set_to_none=True)
            (loss.sum() / cfg.reference_models).backward()
            # A frozen model retains its selected checkpoint and Adam buffers exactly.
            for group_idx, (_, lo, hi) in enumerate(model.groups):
                dead = frozen[:, model.order[lo:hi]]
                for parameter in model.chunk_parameters(group_idx):
                    parameter.grad[dead] = 0
            for optimizer in optimizers: optimizer.step()
            if frozen_state is not None:
                model.restore(frozen_state, frozen)
                assert frozen_optimizer_state is not None
                _restore_optimizer_rows(optimizers, frozen_optimizer_state, model, frozen)
            if step % cfg.eval_every and step != cfg.max_steps: continue
            train_rows=[]; val_rows=[]
            with torch.no_grad():
                for condition, (task, budget) in enumerate(group):
                    sx, sy, vx, vy, *_ = prepared[task]
                    train_rows.append(_loss(model, condition, sx[:train_counts[budget]], sy[:train_counts[budget]], cfg.set_size))
                    val_rows.append(_loss(model, condition, vx[:valid_counts[budget]], vy[:valid_counts[budget]], cfg.set_size))
            train_loss, val_loss = torch.stack(train_rows), torch.stack(val_rows)
            significant = val_loss < best - cfg.checkpoint_relative_improvement * torch.maximum(best.abs(), torch.full_like(best, cfg.checkpoint_floor))
            first = ~torch.isfinite(best); improved = (significant | first) & ~frozen
            state = model.snapshot()
            for field in best_state: best_state[field][improved] = state[field][improved]
            best[improved] = val_loss[improved]; best_step[improved] = step; best_train[improved] = train_loss[improved]; last_improved[improved] = step
            for condition in range(c): train_history[condition].append(train_loss[condition]); val_history[condition].append(val_loss[condition])
            passes = torch.stack([_plateau(train_history[i], cfg) & _plateau(val_history[i], cfg) for i in range(c)]) if step >= cfg.min_steps else torch.zeros_like(frozen)
            eligible = (step >= cfg.min_steps) & ((step - last_improved) >= cfg.checkpoint_patience)
            consecutive = torch.where(passes & eligible & ~frozen, consecutive + 1, torch.where(frozen, consecutive, torch.zeros_like(consecutive)))
            newly = (consecutive >= cfg.required_plateau_passes) & ~frozen
            if newly.any():
                model.restore(best_state, newly)
                frozen_state = model.snapshot() if frozen_state is None else frozen_state
                for field in frozen_state: frozen_state[field][newly] = best_state[field][newly]
                current_opt = _optimizer_snapshot(optimizers)
                if frozen_optimizer_state is None:
                    frozen_optimizer_state = current_opt
                else:
                    for group_idx, (old_group, current_group) in enumerate(zip(frozen_optimizer_state, current_opt)):
                        _, lo, hi = model.groups[group_idx]
                        rows = newly[:, model.order[lo:hi]]
                        for old_param, current_param in zip(old_group, current_group):
                            for name in old_param:
                                old_param[name][rows] = current_param[name][rows]
                frozen |= newly
                stopped[newly] = step
            curve_steps.append(step); curve_train.append(train_loss.detach().cpu()); curve_val.append(val_loss.detach().cpu())
            if step == 800:
                artifacts["step800"][str(group)] = {
                    "train": train_loss.detach().cpu(), "checkpoint": val_loss.detach().cpu(),
                    "frozen": frozen.detach().cpu(),
                    "state": {field: value.detach().cpu() for field, value in model.snapshot().items()},
                }
            if step % 400 == 0 or step == cfg.max_steps or bool(frozen.all()):
                print({"adaptive_progress": "checkpoint", "phase": phase, "seed": seed,
                       "conditions": group, "step": step, "mean_train": float(train_loss.mean()),
                       "mean_checkpoint": float(val_loss.mean()), "frozen": int(frozen.sum()),
                       "models": m}, flush=True)
            if bool(frozen.all()): break
        final_step = step
        model.restore(best_state)
        for condition, (task, budget) in enumerate(group):
            *_, tx, ty = prepared[task]
            score_loss = _loss(model, condition, tx, ty, cfg.set_size)
            for index, (method, replica, row) in enumerate(zip(method_for_model, reps, [next(x for x in manifest if x["method"] == n) for n in method_for_model])):
                records.append({"phase": phase, "task": task, "support_size": budget, "method": method, "init": replica,
                                "family": row["family"], "rho": row["rho"], "lr": row["lr"], "score_mse": float(score_loss[index]),
                                "checkpoint_mse": float(best[condition, index]), "best_train_mse": float(best_train[condition, index]),
                                "best_step": int(best_step[condition, index]),
                                "stop_step": int(stopped[condition, index]) if int(stopped[condition, index]) else final_step,
                                "converged": bool(frozen[condition, index]), "status": "converged" if frozen[condition, index] else "max_cap"})
        artifacts["curves"][str(group)] = {"steps": np.asarray(curve_steps, dtype=np.int32),
                                            "train": torch.stack(curve_train).numpy(),
                                            "checkpoint": torch.stack(curve_val).numpy(),
                                            "stop_steps": stopped.detach().cpu().numpy()}
        saved = model.snapshot()
        saved["masks"] = model.masks[:, model.inverse].detach().clone()
        saved["effective_weight"] = saved["weight"] * saved["masks"]
        artifacts["states"][str(group)] = {field: value.detach().cpu() for field, value in saved.items()}
    if not all(torch.isfinite(torch.tensor([row["score_mse"] for row in records]))): raise AssertionError("non-finite score")
    return records, artifacts


def toy_adam_lr_test(device: str | torch.device = "cpu") -> bool:
    """A chunk has the same standard Adam update as two scalar reference fits."""
    device = torch.device(device)
    masks = torch.ones(2, 784, 1, device=device)
    model = BatchedAdaptiveConditions(masks, [.0005, .005], [11], [0, 1], reference_models=2).to(device)
    before = [p.detach().clone() for p in model.chunk_parameters(0) + model.chunk_parameters(1)]
    opts = [torch.optim.Adam(model.chunk_parameters(i), lr=g[0]) for i, g in enumerate(model.groups)]
    for opt in opts:
        for p in opt.param_groups[0]["params"]: p.grad = torch.ones_like(p)
        opt.step()
    # analytical first Adam update with grad=1 is exactly lr (within epsilon).
    return all(torch.allclose(p, old - group_lr, atol=1e-7) for p, old, group_lr in
               zip(model.chunk_parameters(0) + model.chunk_parameters(1), before,
                   [model.groups[0][0]] * 4 + [model.groups[1][0]] * 4))


def pairing_and_freeze_test(device: str | torch.device = "cpu") -> bool:
    """Checks paired starts and that a frozen Adam row cannot drift."""
    device = torch.device(device)
    masks = torch.ones(8, 784, 1, device=device)
    model = BatchedAdaptiveConditions(masks, [.002] * 8, [17], [0, 1, 2, 3, 0, 1, 2, 3], reference_models=20).to(device)
    paired = all(torch.equal(model.joined(field)[0, :4], model.joined(field)[0, 4:]) for field in model.fields)
    optimizer = torch.optim.Adam(model.chunk_parameters(0), lr=.002)
    for parameter in model.chunk_parameters(0): parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    frozen = torch.zeros((1, 8), dtype=torch.bool, device=device); frozen[0, 0] = True
    state = model.snapshot(); opt_state = _optimizer_snapshot([optimizer])
    for parameter in model.chunk_parameters(0): parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    model.restore(state, frozen); _restore_optimizer_rows([optimizer], opt_state, model, frozen)
    return paired and all(torch.equal(model.joined(field)[0, 0], state[field][0, 0]) for field in model.fields)


def logical_order_mapping_test(device: str | torch.device = "cpu") -> bool:
    """Exercise a nonidentity 3-LR permutation through loss, snapshot, and freeze paths."""
    device = torch.device(device)
    masks = torch.ones(12, 784, 1, device=device)
    for index in range(12): masks[index, :index + 1] = 0
    lrs = [.005] * 4 + [.0005] * 4 + [.002] * 4
    model = BatchedAdaptiveConditions(masks, lrs, [23], [0, 1, 2, 3] * 3, reference_models=20).to(device)
    state = model.snapshot()
    snapshot_ok = all(torch.equal(state[field][0, logical], model.joined(field)[0, model.inverse[logical]])
                      for field in model.fields for logical in range(12))
    logical_masks = model.masks[:, model.inverse]
    effective_ok = torch.equal(state["weight"] * logical_masks, state["weight"] * logical_masks) and bool(
        ((state["weight"] * logical_masks)[logical_masks == 0] == 0).all())
    x = torch.rand(1, 3, 5, 784, device=device); y = torch.rand(3, device=device)
    direct = (model(x)[0] - y[None]).square().mean(1) / 5
    loss_ok = torch.allclose(_loss(model, 0, x[0], y, 5), direct)
    optimizers = [torch.optim.Adam(model.chunk_parameters(i), lr=lr) for i, (lr, _, _) in enumerate(model.groups)]
    for optimizer in optimizers:
        for parameter in optimizer.param_groups[0]["params"]: parameter.grad = torch.ones_like(parameter)
        optimizer.step()
    checkpoint = model.snapshot(); opt_checkpoint = _optimizer_snapshot(optimizers)
    frozen = torch.zeros((1, 12), dtype=torch.bool, device=device); frozen[0, 0] = True; frozen[0, 7] = True
    for optimizer in optimizers:
        for parameter in optimizer.param_groups[0]["params"]: parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
    model.restore(checkpoint, frozen); _restore_optimizer_rows(optimizers, opt_checkpoint, model, frozen)
    freeze_ok = all(torch.equal(model.snapshot()[field][frozen], checkpoint[field][frozen]) for field in model.fields)
    return snapshot_ok and effective_ok and loss_ok and freeze_ok
