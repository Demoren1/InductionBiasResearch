"""Signed fresh-C fitting and teacher-free canonicalization for common I."""

import itertools
import math
import time

import numpy as np
import torch
from torch.nn import functional as F


def canonical_index(index, codes):
    best, result = None, None
    for permutation in itertools.permutations(range(1, codes + 1)):
        for signs in itertools.product((-1, 1), repeat=codes):
            labels = np.r_[0, np.asarray(permutation) * signs]
            renamed = labels[np.abs(index)] * np.sign(index)
            renamed = renamed[:, np.lexsort(renamed[::-1])]
            key = renamed.tobytes()
            if best is None or key < best:
                best, result = key, renamed.copy()
    return result.astype(np.int16)


def features(x, index, codes):
    index = torch.as_tensor(index, device=x.device).long()
    basis = (
        F.one_hot(index.abs(), codes + 1)[..., 1:].to(x.dtype) * index.sign()[..., None]
    )
    return torch.einsum("ni,ihk->nhk", x, basis)


def code_prediction(feature, code, plan):
    return (torch.einsum("nhk,sk->snh", feature, code) + plan["bias"]).relu().sum(
        -1
    ) / math.sqrt(plan["hidden"])


def fresh_codes(plan, family, task, device):
    rng = np.random.default_rng(plan["seed"] + family * 37 + task * 177 + 71)
    signs = np.repeat(
        np.array(list(itertools.product((-1, 1), repeat=plan["codes"]))), 2, axis=0
    )
    return torch.tensor(
        signs * np.exp(rng.normal(-0.3, 0.7, size=signs.shape)),
        device=device,
        dtype=torch.float64,
    ).clamp(-plan["axis_code_max"], plan["axis_code_max"])


@torch.no_grad()
def code_loss(feature, y, code, plan, weights=None):
    total = code.new_zeros(len(code))
    normalizer = len(y) if weights is None else weights.sum()
    for begin in range(0, len(y), plan["numeric_chunk"]):
        end = min(len(y), begin + plan["numeric_chunk"])
        loss = (
            code_prediction(feature[begin:end], code, plan) - y[None, begin:end]
        ).square()
        total += (
            loss.sum(1) if weights is None else (loss * weights[None, begin:end]).sum(1)
        )
    return total / normalizer


@torch.no_grad()
def fit_codes(index, x, y, vx, vy, initial, plan, *, weights=None, max_steps=None):
    started = time.monotonic()
    feature, validation_feature = (
        features(x, index, plan["codes"]),
        features(vx, index, plan["codes"]),
    )
    code = initial.clone().clamp(-plan["axis_code_max"], plan["axis_code_max"])
    if code.ndim == 1:
        code = code[None]
    max_steps = plan["numeric_code_steps"] if max_steps is None else max_steps
    best_score = code.new_full((len(code),), torch.inf)
    best_code = code.clone()
    best_step = torch.zeros(len(code), device=code.device, dtype=torch.long)
    alive = torch.ones(len(code), device=code.device, dtype=torch.bool)
    history = []
    normalizer = len(y) if weights is None else weights.sum()
    identity = torch.eye(plan["codes"], device=code.device, dtype=code.dtype)
    for step in range(max_steps + 1):
        train = code_loss(feature, y, code, plan, weights)
        validation = code_loss(validation_feature, vy, code, plan)
        better = validation < best_score
        best_score[better], best_code[better], best_step[better] = (
            validation[better],
            code[better],
            step,
        )
        history.append(
            {
                "step": step,
                "train_mse": train.tolist(),
                "validation_mse": validation.tolist(),
                "active_starts": int(alive.sum()),
            }
        )
        if step == max_steps or not alive.any():
            break
        gram = code.new_zeros((len(code), plan["codes"], plan["codes"]))
        score = code.new_zeros((len(code), plan["codes"]))
        for begin in range(0, len(y), plan["numeric_chunk"]):
            end = min(len(y), begin + plan["numeric_chunk"])
            local = feature[begin:end]
            activation = torch.einsum("nhk,sk->snh", local, code) + plan["bias"]
            jacobian = torch.einsum(
                "nhk,snh->snk", local, (activation > 0).to(code.dtype)
            ) / math.sqrt(plan["hidden"])
            residual = y[None, begin:end] - activation.relu().sum(-1) / math.sqrt(
                plan["hidden"]
            )
            local_weights = (
                torch.ones(end - begin, device=code.device, dtype=code.dtype)
                if weights is None
                else weights[begin:end]
            )
            gram += torch.einsum("snk,snl,n->skl", jacobian, jacobian, local_weights)
            score += torch.einsum("snk,sn,n->sk", jacobian, residual, local_weights)
        gram /= normalizer
        score /= normalizer
        scale = torch.linalg.eigvalsh(gram)[:, -1].clamp_min(1e-12)
        direction = torch.linalg.solve(
            gram + plan["numeric_ridge"] * scale[:, None, None] * identity,
            score[..., None],
        )[..., 0]
        accepted = ~alive
        proposed_code = code.clone()
        for exponent in range(plan["numeric_line_search"]):
            candidate = code + 2.0 ** (-exponent) * direction
            feasible = candidate.abs().amax(1) <= plan["axis_code_max"]
            value = code_loss(feature, y, candidate, plan, weights)
            take = (~accepted) & feasible & (value < train)
            proposed_code[take] = candidate[take]
            accepted |= take
            if accepted.all():
                break
        alive &= accepted
        code = proposed_code
    winner = int(best_score.argmin())
    if not torch.isfinite(best_code).all() or not torch.isfinite(best_score).all():
        raise FloatingPointError("Nonfinite signed-C fit")
    return best_code[winner], {
        "selected_start": winner,
        "selected_step": int(best_step[winner]),
        "validation_mse": float(best_score[winner]),
        "history": history,
        "best_validation_per_start": best_score.tolist(),
        "best_step_per_start": best_step.tolist(),
        "seconds": time.monotonic() - started,
    }
