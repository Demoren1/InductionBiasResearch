"""Archive the pilot's original shared initialization/data RNG convention.

The main experiment separates these RNG streams. Keep the original pilot
reproducible without altering its observations or repeating its computation.
"""

import time

import torch

from core import population_loss, prediction, teacher_vectors


def fit(
    masks,
    spec,
    seed,
    steps=1024,
    lr=0.05,
    scale=1.25,
    batch=64,
    device="cuda:1",
    checkpoints=(),
):
    started = time.perf_counter()
    generator = torch.Generator(device=device).manual_seed(seed)
    masks = torch.as_tensor(masks, device=device, dtype=torch.float32)
    weights = (
        torch.randn(len(masks), 8, 2, device=device, generator=generator)
        * scale
        * masks
    )
    amplitude = torch.ones(len(masks), 2, device=device)
    weights.requires_grad_()
    amplitude.requires_grad_()
    low, high, small = teacher_vectors(spec, device)
    history = {}
    for step in range(1, steps + 1):
        x = torch.randn(batch, 8, device=device, generator=generator)
        y = small * torch.sin(x @ low) + torch.sin(x @ high)
        (prediction(x, weights, amplitude) - y).square().mean(
            (1, 2)
        ).sum().backward()
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
    gradients = torch.autograd.grad(losses.sum(), weights)[0]
    torch.cuda.synchronize(device)
    return {
        "weights": weights.detach().cpu().numpy(),
        "amplitude": amplitude.detach().cpu().numpy(),
        "gradients": gradients.detach().cpu().numpy(),
        "losses": losses.detach().cpu().numpy(),
        "history": history,
        "seconds": time.perf_counter() - started,
    }
