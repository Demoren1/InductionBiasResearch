"""CUDA-only batched independent MLP fits and initialization-rewound IMP."""

import math

import torch
from torch.nn import functional as F

from ..problem import HIDDEN, INPUTS, logits


def initialization(seeds, device, options):
    states = []
    for seed in seeds:
        generator = torch.Generator().manual_seed(seed)
        states.append(
            {
                "w": torch.randn(INPUTS, HIDDEN, generator=generator) * 0.1,
                "b": torch.full((HIDDEN,), 0.1),
                "a": torch.randn(HIDDEN, generator=generator) * 0.2,
                "c": torch.zeros(()),
            }
        )
        if options.get("initialization") == "positive_weights_zero_readout":
            states[-1]["w"].abs_()
            states[-1]["a"].zero_()
    return {
        key: torch.stack([state[key] for state in states]).to(device)
        for key in states[0]
    }


def expanded_targets(target, prediction):
    while target.ndim < prediction.ndim:
        target = target.unsqueeze(-2)
    return target.expand_as(prediction)


@torch.no_grad()
def bce(state, mask, data):
    x, y = data
    prediction = logits(state, mask, x, packed=state["w"].ndim == 4)
    return F.binary_cross_entropy_with_logits(
        prediction, expanded_targets(y, prediction), reduction="none"
    ).mean(-1)


@torch.no_grad()
def entropy(data):
    y = data[1]
    return F.binary_cross_entropy(y, y)


def fit(initial, mask, data, options, active_models=None):
    if mask.device.type != "cuda" or any(
        value.device.type != "cuda" for value in initial.values()
    ):
        raise RuntimeError("MLP parameters and masks must be on CUDA")
    state = {
        key: value.detach().clone().requires_grad_()
        for key, value in initial.items()
    }
    optimizer = torch.optim.Adam(state.values(), lr=options["lr"])
    best = {key: value.detach().clone() for key, value in state.items()}
    best_loss = bce(state, mask, data["query"])
    best_steps = torch.zeros_like(best_loss, dtype=torch.long)
    history = []
    for step in range(1, options["steps"] + 1):
        slope = (
            options["negative_slope"]
            if step <= options["warmup_steps"]
            else 0.0
        )
        x, y = data["support"]
        prediction = logits(
            state, mask, x, slope, packed=options.get("layout") == "packed"
        )
        loss = F.binary_cross_entropy_with_logits(
            prediction, expanded_targets(y, prediction), reduction="none"
        ).mean(-1)
        penalty = (state["w"] * mask).square().sum((-2, -1))
        penalty += state["b"].square().sum(-1)
        penalty += state["a"].square().sum(-1)
        penalty += state["c"].square()
        objective = loss + 0.5 * options["l2"] * penalty
        if active_models is not None:
            objective = objective * active_models
        optimizer.zero_grad(set_to_none=True)
        objective.sum().backward()
        optimizer.step()
        with torch.no_grad():
            state["w"].mul_(mask)
            if step % options["select_every"] == 0 or step == options["steps"]:
                query = bce(state, mask, data["query"])
                if not torch.isfinite(query).all():
                    raise RuntimeError("Nonfinite validation BCE")
                improved = query < best_loss
                best_loss = torch.minimum(best_loss, query)
                best_steps = torch.where(improved, step, best_steps)
                for key, value in state.items():
                    selector = improved.reshape(
                        improved.shape + (1,) * (value.ndim - improved.ndim)
                    )
                    best[key] = torch.where(
                        selector, value.detach(), best[key]
                    )
                history.append(
                    {
                        "step": step,
                        "support_bce": bce(state, mask, data["support"])
                        .cpu()
                        .tolist(),
                        "query_bce": query.cpu().tolist(),
                    }
                )
    return best, best_steps, history


def prune(mask, weights, keep):
    scores = (
        weights.detach().abs().masked_fill(mask == 0, -torch.inf).flatten(-2)
    )
    indices = scores.argsort(dim=-1, descending=True, stable=True)[..., :keep]
    return torch.zeros_like(scores).scatter_(-1, indices, 1).reshape_as(mask)


def imp(
    initial,
    dense_state,
    dense_steps,
    dense_history,
    data,
    options,
    protocol,
    log,
    active_models=None,
):
    mask = torch.ones_like(initial["w"])
    state, selected, history = dense_state, dense_steps, dense_history
    rounds = []
    while True:
        active = int(mask.reshape(-1, INPUTS, HIDDEN)[0].sum().item())
        rounds.append(
            {
                "active_edges": active,
                "selected_steps": selected.cpu().tolist(),
                "query_bce": bce(state, mask, data["query"]).cpu().tolist(),
                "history": history,
            }
        )
        log(f"IMP {active} edges")
        if active == protocol["keep"]:
            break
        keep = max(
            protocol["keep"],
            math.ceil(active * (1 - protocol["prune_fraction"])),
        )
        mask = prune(mask, state["w"], keep)
        state, selected, history = fit(
            initial, mask, data, options, active_models=active_models
        )
    return state, mask, rounds
