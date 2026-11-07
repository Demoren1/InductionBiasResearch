"""Fresh numerical fitting for fixed discrete or continuous first-layer bases."""

import itertools
import math

import numpy as np
import torch

from experiments.sharing_structure_research_v1.shared_structure_bilevel_optimization import (
    ScenarioAdam,
)


def feature(x, basis):
    return torch.einsum("tnd,dhk->tnhk", x, basis)


def values(x, basis, parameter, hidden, outputs, precomputed=None, mask=None):
    if basis is None:
        dimensions = x.shape[-1]
        codes = dimensions * hidden
        weight = parameter[..., :codes].reshape(
            *parameter.shape[:2], dimensions, hidden
        )
        if mask is not None:
            weight = weight * mask
        preactivation = torch.einsum("tnd,rtdh->rtnh", x, weight)
    else:
        codes = basis.shape[-1]
        feat = feature(x, basis) if precomputed is None else precomputed
        preactivation = torch.einsum("tnhk,rtk->rtnh", feat, parameter[..., :codes])
    activation = (preactivation + parameter[:, :, None, codes : codes + hidden]).relu()
    head = parameter[..., codes + hidden : -outputs].reshape(
        *parameter.shape[:2], hidden, outputs
    )
    return (
        torch.einsum("rtnh,rtho->rtno", activation, head) / math.sqrt(hidden)
        + parameter[:, :, None, -outputs:]
    )


def losses(value, target, kind):
    if kind == "binary_cross_entropy":
        return torch.nn.functional.binary_cross_entropy_with_logits(
            value, target[None].expand_as(value), reduction="none"
        ).mean((-1, -2))
    if kind == "cross_entropy":
        return -(value.log_softmax(-1) * target[None]).sum(-1).mean(-1)
    return (value - target[None]).square().mean((-1, -2))


def fit_adam(
    x,
    y,
    vx,
    vy,
    basis,
    hidden,
    outputs,
    *,
    steps=2000,
    loss="mse",
    seed=761003,
    checkpoint=25,
    keep=4,
    fixed_step=None,
    initial=None,
    settings=None,
    mask=None,
):
    """Select numeric states on H0, or stop at a preselected support-only budget."""
    rows = (
        settings
        if settings is not None
        else [
            {"lr": lr, "quantile": quantile, "restart": restart}
            for lr, quantile, restart in itertools.product(
                (0.003, 0.01, 0.03), (0.1, 0.5, 0.9), range(2)
            )
        ]
    )
    tasks, dimensions = x.shape[0], x.shape[-1]
    codes = dimensions * hidden if basis is None else basis.shape[-1]
    length = codes + hidden + hidden * outputs + outputs
    generator = torch.Generator(device=x.device).manual_seed(seed)
    parameter = torch.randn(
        (len(rows), tasks, length), device=x.device, dtype=x.dtype, generator=generator
    )
    energy = 1.0 if basis is None else float(basis.square().sum(-1).mean())
    parameter[..., :codes] *= math.sqrt(2 / dimensions / max(energy, 1e-12))
    if mask is not None:
        parameter[..., :codes].reshape(len(rows), tasks, dimensions, hidden).mul_(
            (dimensions / mask.sum(0).clamp_min(1)).sqrt()
        )
    parameter[..., codes : codes + hidden] = 0
    parameter[..., -outputs:] = 0
    fx = None if basis is None else feature(x, basis)
    fv = None if basis is None or vx is None else feature(vx, basis)
    with torch.no_grad():
        if basis is None:
            initial_weight = parameter[..., :codes].reshape(
                len(rows), tasks, dimensions, hidden
            )
            if mask is not None:
                initial_weight = initial_weight * mask
            pre = torch.einsum(
                "tnd,rtdh->rtnh",
                x,
                initial_weight,
            )
        else:
            pre = torch.einsum("tnhk,rtk->rtnh", fx, parameter[..., :codes])
        for row, setting in enumerate(rows):
            parameter[row, :, codes : codes + hidden] = -torch.quantile(
                pre[row], setting["quantile"], dim=1
            )
        if initial is not None:
            count = min(len(initial), len(rows))
            parameter[:count] = initial[:count]
    parameter.requires_grad_()
    optimizer = ScenarioAdam(parameter, [row["lr"] for row in rows])
    best = parameter.new_full((len(rows), tasks), torch.inf)
    selected = parameter.detach().clone()
    selected_step = torch.zeros_like(best, dtype=torch.long)
    history = {"steps": [], "train": [], "validation": []}
    stop = steps if fixed_step is None else fixed_step
    for step in range(stop + 1):
        if step % checkpoint == 0 or step == stop:
            with torch.no_grad():
                train = losses(
                    values(x, basis, parameter, hidden, outputs, fx, mask), y, loss
                )
                score = (
                    train
                    if vx is None
                    else losses(
                        values(vx, basis, parameter, hidden, outputs, fv, mask),
                        vy,
                        loss,
                    )
                )
                improved = (
                    score < best
                    if fixed_step is None
                    else torch.full_like(best, step == stop, dtype=torch.bool)
                )
                best[improved], selected[improved], selected_step[improved] = (
                    score[improved],
                    parameter[improved],
                    step,
                )
                history["steps"].append(step)
                history["train"].append(train.cpu().numpy())
                history["validation"].append(score.cpu().numpy())
        if step == stop:
            break
        if x.shape[1] > 512:
            positions = torch.randint(
                x.shape[1], (512,), generator=generator, device=x.device
            )
            bx, by = x[:, positions], y[:, positions]
            bf = None if fx is None else fx[:, positions]
        else:
            bx, by, bf = x, y, fx
        objective = losses(
            values(bx, basis, parameter, hidden, outputs, bf, mask), by, loss
        ).sum()
        gradient = torch.autograd.grad(objective, parameter)[0]
        optimizer.step(parameter, gradient)
    ranking = best.argsort(0)[:keep]
    task = torch.arange(tasks, device=x.device)[None]
    return selected[ranking, task].detach(), {
        "settings": rows,
        "steps": np.asarray(history["steps"]),
        "train_history": np.stack(history["train"]),
        "validation_history": np.stack(history["validation"]),
        "numerical_selection_loss": best[ranking, task].cpu().numpy(),
        "selected_step": selected_step[ranking, task].cpu().numpy(),
    }


def signed_basis(index, codes, device, dtype=torch.float64):
    index = torch.as_tensor(index, device=device, dtype=torch.long)
    return (
        torch.nn.functional.one_hot(index.abs(), codes + 1)[..., 1:].to(dtype)
        * index.sign()[..., None]
    )
