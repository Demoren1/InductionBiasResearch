"""Unrestricted connections, signed categorical sharing and independent fits."""

import math

import numpy as np
import torch
from torch.nn import functional as F

from experiments.darts_indexator_v1.model import canonical_signed_probabilities


class BatchedOptimizer:
    """An independent native-equation optimizer per first-dimension row."""

    def __init__(self, parameter, rows, key="numeric_lr"):
        self.parameter = parameter
        self.rates = parameter.new_tensor([row[key] for row in rows]).reshape(
            (-1,) + (1,) * (parameter.ndim - 1)
        )
        self.momentum = parameter.new_tensor(
            [row.get("momentum", 0.0) for row in rows]
        ).reshape((-1,) + (1,) * (parameter.ndim - 1))
        self.adam = (
            parameter.new_tensor(
                [row.get("optimizer", "adam") == "adam" for row in rows]
            )
            .bool()
            .reshape((-1,) + (1,) * (parameter.ndim - 1))
        )
        self.first = torch.zeros_like(parameter)
        self.second = torch.zeros_like(parameter)
        self.velocity = torch.zeros_like(parameter)
        self.steps = torch.zeros_like(self.rates, dtype=torch.long)
        self.calls = 0

    @torch.no_grad()
    def step(self, gradient, active=None):
        self.calls += 1
        active = (
            torch.ones_like(self.steps, dtype=torch.bool)
            if active is None
            else active.reshape(self.steps.shape)
        )
        self.steps.add_(active.long())
        self.first.copy_(
            torch.where(active, self.first * 0.9 + gradient * 0.1, self.first)
        )
        self.second.copy_(
            torch.where(
                active, self.second * 0.999 + gradient.square() * 0.001, self.second
            )
        )
        elapsed = self.steps.clamp_min(1)
        adam_update = (self.first / (1 - 0.9**elapsed)) / (
            (self.second / (1 - 0.999**elapsed)).sqrt() + 1e-8
        )
        self.velocity.copy_(
            torch.where(active, self.velocity * self.momentum + gradient, self.velocity)
        )
        self.parameter.sub_(
            self.rates * torch.where(self.adam, adam_update, self.velocity) * active
        )


def clip_gradient(gradient, maximum, dimensions):
    norm = gradient.square().sum(dim=dimensions, keepdim=True).sqrt()
    return gradient * (maximum / norm.clamp_min(1e-12)).clamp_max(1)


def signed_values(numeric, codes):
    values = numeric[..., :codes]
    return torch.cat([torch.zeros_like(values[..., :1]), values, -values], -1)


def soft_weights(probabilities, numeric, codes):
    return torch.einsum("sdhk,srk->srdh", probabilities, signed_values(numeric, codes))


def fixed_weights(indices, numeric, codes):
    index = indices[:, None].expand(-1, numeric.shape[1], -1, -1)
    values = F.pad(numeric[..., :codes], (1, 0))
    weights = values.gather(2, index.abs().flatten(2)).reshape(index.shape)
    return weights * index.sign()


def hard_indices(probabilities):
    canonical = canonical_signed_probabilities(probabilities)
    classes = canonical.argmax(-1)
    codes = (probabilities.shape[-1] - 1) // 2
    return torch.where(classes > codes, codes - classes, classes).to(torch.int16)


def predict(inputs, weights, numeric, specification):
    codes = specification["codes"]
    hidden = torch.einsum("nd,srdh->srnh", inputs, weights)
    if specification["kind"] == "sine":
        return hidden[..., 0].sin() * numeric[..., codes + 1, None]
    if specification.get("head") == "full":
        width = specification["hidden"]
        bias = numeric[..., codes : codes + width]
        readout = numeric[..., codes + width : codes + 2 * width]
        return ((hidden + bias[..., None, :]).relu() * readout[..., None, :]).sum(
            -1
        ) / math.sqrt(width) + numeric[..., -1, None]
    hidden = (hidden + numeric[..., codes, None, None]).relu()
    return (
        hidden.sum(-1)
        * numeric[..., codes + 1, None]
        / math.sqrt(specification["hidden"])
        + numeric[..., codes + 2, None]
    )


def initial_numeric(
    rows, restarts, specification, seed, device, probabilities=None, indices=None
):
    codes = specification["codes"]
    parameters = []
    for row in rows:
        row_parameters = []
        rng = np.random.default_rng(seed + 100003 * row["seed"])
        for _ in range(restarts):
            values = rng.normal(size=codes)
            if row.get("aligned_initial_codes", True):
                values = np.sort(np.abs(values)) + 0.15
            row_parameters.append(
                np.r_[
                    values,
                    0.1 if specification["kind"] == "deepsets" else 0.0,
                    1.0,
                    specification["target_mean"]
                    if specification["kind"] == "deepsets"
                    else 0.0,
                ]
            )
        parameters.append(row_parameters)
    numeric = torch.tensor(parameters, dtype=torch.float32, device=device)
    weights = (
        soft_weights(probabilities, numeric, codes)
        if probabilities is not None
        else fixed_weights(indices, numeric, codes)
    )
    norm = weights.square().sum((-1, -2)).sqrt().clamp_min(1e-8)
    requested = numeric.new_tensor([row["radius"] for row in rows])[:, None]
    if specification["kind"] == "deepsets":
        requested = requested * math.sqrt(specification["hidden"])
    numeric[..., :codes] *= (requested / norm)[..., None]
    return numeric


def source_tensors(output, device):
    with np.load(output / "source.npz") as archive:
        return {
            name: torch.as_tensor(archive[name], device=device)
            for name in archive.files
        }


@torch.no_grad()
def score(inputs, targets, weights, numeric, specification, batch=512):
    total = torch.zeros(numeric.shape[:2], device=numeric.device)
    for start in range(0, len(inputs), batch):
        predicted = predict(
            inputs[start : start + batch], weights, numeric, specification
        )
        total += (predicted - targets[start : start + batch]).square().sum(-1)
    return total / len(inputs)
