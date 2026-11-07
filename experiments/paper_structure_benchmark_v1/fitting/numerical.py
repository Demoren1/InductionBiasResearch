"""Common fixed-structure numerical fitting with a free linear readout."""

import math
import time

import numpy as np
import torch

from experiments.sharing_structure_research_v1.blind_numeric import (
    normalize_gauge,
    prediction,
    profile_head,
)


@torch.no_grad()
def fit_features(features, validation_features, initial, y, vy, codes, steps=48):
    count, _, hidden, _ = features.shape
    outputs = y.shape[-1]
    parameter = normalize_gauge(initial.clone(), codes, hidden, outputs)
    eye = torch.eye(parameter.shape[1], device=parameter.device, dtype=parameter.dtype)
    best = parameter.new_full((count,), torch.inf)
    selected = parameter.clone()
    selected_step = torch.zeros(count, device=parameter.device, dtype=torch.long)
    training, validation = [], []
    started = time.perf_counter()
    for step in range(steps + 1):
        parameter = profile_head(features, parameter, y, codes, hidden, outputs)
        value, jac = prediction(
            features, parameter, codes, hidden, outputs, jacobian=True
        )
        residual = (y - value).flatten(1)
        train = residual.square().mean(1)
        score = (
            (prediction(validation_features, parameter, codes, hidden, outputs) - vy)
            .square()
            .mean((1, 2))
        )
        improved = score < best
        best[improved], selected[improved], selected_step[improved] = (
            score[improved],
            parameter[improved],
            step,
        )
        training.append(train.cpu().numpy())
        validation.append(score.cpu().numpy())
        if step == steps:
            break
        gram = jac.transpose(1, 2) @ jac / residual.shape[1]
        cross = jac.transpose(1, 2) @ residual[..., None] / residual.shape[1]
        damping = gram.diagonal(dim1=1, dim2=2).amax(1).clamp_min(1e-12) * 1e-6
        direction = torch.linalg.solve(gram + damping[:, None, None] * eye, cross)[
            ..., 0
        ]
        accepted = train < 1e-24
        updated = parameter.clone()
        for exponent in range(12):
            proposed = normalize_gauge(
                parameter + 2.0 ** (-exponent) * direction, codes, hidden, outputs
            )
            proposed = profile_head(features, proposed, y, codes, hidden, outputs)
            loss = (
                (prediction(features, proposed, codes, hidden, outputs) - y)
                .square()
                .mean((1, 2))
            )
            take = (~accepted) & torch.isfinite(proposed).all(1) & (loss < train)
            updated[take] = proposed[take]
            accepted |= take
            if accepted.all():
                break
        parameter = updated
    return selected, {
        "numerical_selection_mse": best.cpu().numpy(),
        "selected_step": selected_step.cpu().numpy(),
        "train_history": np.stack(training),
        "validation_history": np.stack(validation),
        "seconds": time.perf_counter() - started,
    }


def random_states(count, codes, hidden, outputs, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    length = codes + hidden + hidden * outputs + outputs
    parameter = torch.randn(
        (count, length), generator=generator, device=device, dtype=torch.float64
    )
    parameter[:, codes : codes + hidden] *= 0.5
    parameter[:, codes + hidden : -outputs] *= 1 / math.sqrt(hidden)
    parameter[:, -outputs:] = 0
    return parameter
