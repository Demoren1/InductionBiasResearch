"""Real minibatch SGD for a small sine network with squared prediction loss."""

import json
import time
from pathlib import Path

import numpy as np
import torch

ROWS, COLUMNS, DIMENSION, KEEP = 8, 2, 16, 2


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def task(a, b, function=0):
    rng = np.random.default_rng(731001 + function * 97 + a * 11 + b)
    return {
        "rows": [a, b + 4],
        "low": rng.uniform(0.65, 0.75, 2).tolist(),
        "high": rng.uniform(3.8, 4.2, 2).tolist(),
        "small": rng.uniform(0.35, 0.45, 2).tolist(),
    }


def teacher_vectors(spec, device, dtype=torch.float32):
    low = torch.zeros(8, 2, device=device, dtype=dtype)
    high = torch.zeros_like(low)
    for j, row in enumerate(spec["rows"]):
        low[row, j] = spec["low"][j]
        high[row, j] = spec["high"][j]
    small = torch.tensor(spec["small"], device=device, dtype=dtype)
    return low, high, small


def prediction(x, weights, amplitude):
    return amplitude[:, None, :] * torch.sin(
        torch.einsum("bi,nij->nbj", x, weights)
    )


def population_loss(weights, amplitude, spec):
    """Exact Gaussian integral of ordinary MSE; used for diagnostics only."""
    low, high, small = teacher_vectors(spec, weights.device, weights.dtype)

    def kernel(left, right):
        return 0.5 * (
            (-(left - right).square().sum(-2) / 2).exp()
            - (-(left + right).square().sum(-2) / 2).exp()
        )

    target = (
        small.square() * kernel(low, low)
        + kernel(high, high)
        + 2 * small * kernel(low, high)
    )
    return (
        amplitude.square() * kernel(weights, weights)
        - 2
        * amplitude
        * (small * kernel(weights, low) + kernel(weights, high))
        + target
    ).mean(-1)


def fit(
    masks,
    spec,
    seed,
    steps=1024,
    lr=0.05,
    scale=1.25,
    batch=64,
    device="cuda:1",
    initial=None,
    checkpoints=(),
):
    """Train all active input weights and both output amplitudes with SGD.

    Each step samples real Gaussian examples and computes autograd of MSE.
    Restarts share the minibatch stream, independently of their initialization.
    A step's loss sums independent model losses to avoid batch-count scaling.
    """
    started = time.perf_counter()
    generator = torch.Generator(device=device).manual_seed(seed)
    data_generator = torch.Generator(device=device).manual_seed(seed + 700001)
    masks = torch.as_tensor(masks, device=device, dtype=torch.float32)
    n = len(masks)
    weights = torch.randn(n, 8, 2, device=device, generator=generator) * scale
    amplitude = torch.ones(n, 2, device=device)
    if initial is not None:
        weights = torch.as_tensor(initial[0], device=device).clone()
        amplitude = torch.as_tensor(initial[1], device=device).clone()
    weights *= masks
    weights.requires_grad_()
    amplitude.requires_grad_()
    low, high, small = teacher_vectors(spec, device)
    history = {}
    for step in range(1, steps + 1):
        x = torch.randn(batch, 8, device=device, generator=data_generator)
        y = small * torch.sin(x @ low) + torch.sin(x @ high)
        errors = prediction(x, weights, amplitude) - y
        losses = errors.square().mean((1, 2))
        losses.sum().backward()
        with torch.no_grad():
            weights -= lr * weights.grad * masks
            amplitude -= lr * amplitude.grad
        weights.grad = None
        amplitude.grad = None
        if step in checkpoints:
            history[str(step)] = (
                population_loss(weights, amplitude, spec)
                .detach()
                .cpu()
                .numpy()
            )
    losses = population_loss(weights, amplitude, spec)
    evaluation_generator = torch.Generator(device=device).manual_seed(
        seed + 800001
    )
    empirical = torch.zeros(n, device=device)
    with torch.no_grad():
        for _ in range(8):
            x = torch.randn(
                512, 8, device=device, generator=evaluation_generator
            )
            y = small * torch.sin(x @ low) + torch.sin(x @ high)
            empirical += (prediction(x, weights, amplitude) - y).square().mean(
                (1, 2)
            ) / 8
    errors = prediction(x, weights, amplitude) - y
    gradients = torch.autograd.grad(
        errors.square().mean((1, 2)).sum(), weights
    )[0]
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    return {
        "weights": weights.detach().cpu().numpy(),
        "amplitude": amplitude.detach().cpu().numpy(),
        "gradients": gradients.detach().cpu().numpy(),
        "losses": losses.detach().cpu().numpy(),
        "empirical": empirical.detach().cpu().numpy(),
        "history": history,
        "seconds": time.perf_counter() - started,
    }


def random_masks(count, rng):
    indices = rng.random((count, 16)).argsort(1)[:, :2]
    masks = np.zeros((count, 16), dtype=np.float32)
    np.put_along_axis(masks, indices, 1, axis=1)
    return masks.reshape(-1, 8, 2)


def correct_mask(spec, count):
    masks = np.zeros((count, 8, 2), dtype=np.float32)
    masks[:, spec["rows"], np.arange(2)] = 1
    return masks
