"""Observed covariance calibration for signed multi-task fusion penalties."""

import torch


@torch.no_grad()
def calibrated_weights(differences, inverse_gram, variance, rows, outputs, metric):
    tasks, inputs = inverse_gram.shape[:2]
    entries = outputs * inputs
    left, right = torch.triu_indices(entries, entries, 1, device=inverse_gram.device)
    diagonal = inverse_gram.diagonal(dim1=-2, dim2=-1).repeat(1, outputs)
    off = (
        inverse_gram[:, left % inputs, right % inputs]
        * (left // inputs == right // inputs)[None]
    )
    covariance = torch.cat(
        (
            diagonal,
            diagonal[:, left] + diagonal[:, right] - 2 * off,
            diagonal[:, left] + diagonal[:, right] + 2 * off,
        ),
        -1,
    )
    expected_noise = variance * covariance.mean(0)
    if bool((expected_noise <= 0).any()):
        raise ValueError(
            "Calibration requires strictly positive observed noise variance"
        )
    raw_square = differences.square().mean(-1)
    floor = expected_noise.sqrt() * tasks ** (-0.25)
    if metric in ("debiased", "raw_shrinking_floor"):
        square = raw_square - expected_noise if metric == "debiased" else raw_square
        distance = square.clamp_min(0).sqrt()
        values = torch.stack([(distance + floor).pow(-r["gamma"]) for r in rows])
    elif metric == "heat":
        values = []
        scale = expected_noise.median()
        for row in rows:
            logarithm = -raw_square / (row["bandwidth"] * scale)
            logarithm -= logarithm.median()
            values.append(logarithm.clamp(-30, 30).exp())
        values = torch.stack(values)
    else:
        raise ValueError(f"Unknown observed fusion calibration: {metric}")
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("Nonfinite fixed fusion weights")
    return values, floor.median()
