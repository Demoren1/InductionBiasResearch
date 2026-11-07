"""Shared-head Pattern model and independent batched Adam updates."""

import numpy as np
import torch
from torch.nn import functional as F


def balanced_loss(logits, target):
    target = target.expand_as(logits)
    prevalence = target.mean(-1, keepdim=True).clamp(1e-6, 1 - 1e-6)
    weight = target * (0.5 / prevalence)
    weight = weight + (1 - target) * (0.5 / (1 - prevalence))
    return (
        F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weight
    ).mean(-1)


def metrics(logits, target):
    positive = target.bool()
    decision = logits >= 0
    recall = (decision & positive).sum(-1) / positive.sum(-1).clamp_min(1)
    specificity = (~decision & ~positive).sum(-1) / (~positive).sum(-1).clamp_min(1)
    return torch.stack(
        [
            balanced_loss(logits, target),
            (recall + specificity) / 2,
            recall,
            specificity,
        ],
        -1,
    )


def assignment_values(numeric, classes):
    """Expand numerical parameters into the actual categorical code values."""
    values = numeric[..., :-3]
    codes = values.shape[-1]
    if classes == 2 * codes + 1:
        values = torch.cat([values, -values], -1)
    elif classes != codes + 1:
        raise ValueError("Assignment classes must select C or the paired values C/-C")
    return torch.cat([torch.zeros_like(values[..., :1]), values], -1)


def assignment_variance(probabilities, numeric):
    """Normalized categorical value variance; gradients update only P."""
    values = assignment_values(numeric.detach(), probabilities.shape[-1])
    average = torch.einsum("sihk,strk->strih", probabilities, values)
    second = torch.einsum("sihk,strk->strih", probabilities, values.square())
    energy = numeric[..., :-3].detach().square().mean(-1).clamp_min(1e-8)
    return (second - average.square()).clamp_min(0) / energy[..., None, None]


def canonical_signed_probabilities(probabilities):
    """Move exactly cancelling mass in each C/-C pair to the zero class."""
    classes = probabilities.shape[-1]
    if classes < 3 or classes % 2 != 1:
        raise ValueError("Signed probabilities require zero and matched C/-C slots")
    codes = (classes - 1) // 2
    positive = probabilities[..., 1 : codes + 1]
    negative = probabilities[..., codes + 1 :]
    common = torch.minimum(positive, negative)
    zero = probabilities[..., :1] + 2 * common.sum(-1, keepdim=True)
    return torch.cat([zero, positive - common, negative - common], -1)


def soft_predict(inputs, probabilities, numeric):
    codes = numeric.shape[-1] - 3
    values = assignment_values(numeric, probabilities.shape[-1])[..., 1:]
    weights = torch.einsum("sihk,strk->strih", probabilities[..., 1:], values)
    hidden = (
        torch.einsum("ni,strih->strnh", inputs, weights)
        + numeric[..., codes, None, None]
    ).relu()
    return (
        hidden.sum(-1) * numeric[..., codes + 1, None] + numeric[..., codes + 2, None]
    )


def initial_numeric(
    seeds, tasks, restarts, norm, device, codes=4, ordered_signed=False
):
    if ordered_signed and codes != 2:
        raise ValueError("Ordered signed initialization requires two trainable codes")
    arrays = []
    for seed in seeds:
        per_task = []
        for task in range(tasks):
            replicas = []
            for restart in range(restarts):
                rng = np.random.default_rng(seed + 100003 * task + 1009 * restart)
                codebook = rng.normal(size=codes)
                if ordered_signed:
                    codebook = np.array([-abs(codebook[0]), abs(codebook[1])])
                codebook *= norm / max(np.linalg.norm(codebook), 1e-8)
                replicas.append(np.r_[codebook, 0.0, 1.0, 0.0])
            per_task.append(replicas)
        arrays.append(per_task)
    return torch.tensor(np.array(arrays), dtype=torch.float32, device=device)


def clipped(gradient, maximum, dimensions):
    norm = gradient.square().sum(dim=dimensions, keepdim=True).sqrt().clamp_min(1e-12)
    return gradient * (maximum / norm).clamp_max(1)


class BatchedAdam:
    def __init__(self, parameter, learning_rates):
        self.parameter = parameter
        self.first = torch.zeros_like(parameter)
        self.second = torch.zeros_like(parameter)
        self.steps = 0
        self.rates = torch.as_tensor(
            learning_rates, device=parameter.device, dtype=parameter.dtype
        ).reshape((-1,) + (1,) * (parameter.ndim - 1))

    def virtual(self, parameter, gradient):
        step = self.steps + 1
        first = self.first.detach() * 0.9 + gradient * 0.1
        second = self.second.detach() * 0.999 + gradient.square() * 0.001
        denominator = (second / (1 - 0.999**step) + 1e-12).sqrt() + 1e-8
        return parameter - self.rates * (first / (1 - 0.9**step)) / denominator

    @torch.no_grad()
    def step(self, gradient):
        self.steps += 1
        self.first.mul_(0.9).add_(gradient, alpha=0.1)
        self.second.mul_(0.999).addcmul_(gradient, gradient, value=0.001)
        denominator = (self.second / (1 - 0.999**self.steps) + 1e-12).sqrt() + 1e-8
        self.parameter.add_(
            -self.rates * (self.first / (1 - 0.9**self.steps)) / denominator
        )


def schedule(configurations, step, steps, specification, device):
    progress = step / max(steps, 1)
    cooling = min(1.0, max(0.0, (progress - 0.3) / 0.7))
    temperatures, penalties = [], []
    for config in configurations:
        annealed = config["schedule"] == "annealed"
        temperatures.append(
            specification["minimum_temperature"] ** cooling if annealed else 1.0
        )
        penalties.append(
            specification["late_entropy_penalty"] * max(0.0, 2 * progress - 1)
            if annealed
            else 0.0
        )
    temperature = torch.tensor(temperatures, device=device).reshape(-1, 1, 1, 1)
    return temperature, torch.tensor(penalties, device=device)
