"""Fit untied, WSR, LUT-Q, and fixed-index models with the same forward pass."""

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from experiments.paper_structure_benchmark_v1.methods.real_competitors import (
    quantize,
    update_centers,
)
from experiments.wsr_omniglot_v1.source import proximal_groups


@dataclass(frozen=True)
class Layout:
    sites: int
    fan_in: int
    channels: int
    outputs: int

    @property
    def hidden(self):
        return self.sites * self.channels

    @property
    def connections(self):
        return self.fan_in * self.hidden


def predict(x, weight, tail, layout):
    sites, fan_in, channels = layout.sites, layout.fan_in, layout.channels
    weight = weight.reshape(*weight.shape[:2], sites, fan_in, channels)
    bias = tail[..., : layout.hidden].reshape(*tail.shape[:2], 1, sites, channels)
    hidden = torch.einsum("bnsi,brsih->brnsh", x, weight).add(bias).relu()
    head = tail[..., layout.hidden : -layout.outputs].reshape(
        *tail.shape[:2], layout.hidden, layout.outputs
    )
    return (
        torch.einsum("brnh,brho->brno", hidden.flatten(-2), head)
        / math.sqrt(layout.hidden)
        + tail[:, :, None, -layout.outputs :]
    )


def losses(logits, labels):
    indices = labels[:, None, :, None].expand(-1, logits.shape[1], -1, 1)
    return -logits.log_softmax(-1).gather(-1, indices).squeeze(-1)


def optimization_panel(panel, device):
    sx = torch.tensor(panel.support_x, device=device, dtype=torch.float32)
    qx = torch.tensor(panel.query_x, device=device, dtype=torch.float32)
    sy = torch.tensor(panel.support_y, device=device, dtype=torch.long)
    qy = torch.tensor(panel.query_y, device=device, dtype=torch.long)
    return sx, sy, qx, qy


def initialization(layout, connections, restarts, tasks, device, seed):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(restarts, connections, generator=generator)
    weight *= math.sqrt(2 / layout.fan_in)
    tail = torch.randn(
        restarts,
        layout.hidden * (layout.outputs + 1) + layout.outputs,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    tail[:, : layout.hidden] *= 0.1
    tail[:, -layout.outputs :] = 0
    state = torch.cat((weight, tail), dim=-1).to(device)[None]
    return state.expand(tasks, -1, -1).clone().requires_grad_()


def fit_dense(panel, layout, *, steps, restarts, device, seed):
    sx, sy, qx, qy = panel
    n = layout.connections
    parameter = initialization(layout, n, restarts, len(sx), device, seed)
    optimizer = torch.optim.Adam([parameter], lr=0.003)
    generator = torch.Generator(device=device).manual_seed(seed + 2)
    best = torch.full(parameter.shape[:2], torch.inf, device=device)
    saved = parameter.detach().clone()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(steps + 1):
        weight, tail = parameter[..., :n], parameter[..., n:]
        if step % 25 == 0 or step == steps:
            with torch.no_grad():
                value = losses(predict(qx, weight, tail, layout), qy).mean(-1)
                improved = value < best
                best[improved], saved[improved] = value[improved], parameter[improved]
        if step == steps:
            break
        sample = torch.randint(
            sx.shape[1], (min(256, sx.shape[1]),), device=device, generator=generator
        )
        error = losses(predict(sx[:, sample], weight, tail, layout), sy[:, sample])
        optimizer.zero_grad(set_to_none=True)
        error.mean(-1).sum().backward()
        optimizer.step()
    ordered = best.argsort(-1)
    rows = torch.arange(len(sx), device=device)[:, None]
    model = saved[rows, ordered].cpu().numpy()
    validation = best[rows, ordered].cpu().numpy()
    return model, validation, time.perf_counter() - started


def fit_competitor(method, initial, panel, layout, codes, steps, device, seed):
    sx, sy, qx, qy = panel
    n = layout.connections
    parameter = torch.tensor(initial[:, None], device=device, requires_grad=True)
    optimizer = torch.optim.Adam([parameter], lr=0.003)
    generator = torch.Generator(device=device).manual_seed(seed + 3)
    best = np.full(len(sx), np.inf)
    saved = np.zeros((len(sx), n), dtype=np.int16)
    saved_steps = np.zeros(len(sx), dtype=int)
    strengths = [0.001] * len(sx)
    if method == "lutq":
        quantiles = torch.linspace(0, 1, codes, device=device)
        centers = torch.quantile(parameter[:, 0, :n].detach().abs(), quantiles, dim=1).T
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(steps + 1):
        weight, tail = parameter[..., :n], parameter[..., n:]
        with torch.no_grad():
            if method == "wsr":
                projected, index, strengths, _, _ = zip(
                    *[
                        proximal_groups(row, codes, strength, 0.98, True)
                        for row, strength in zip(weight[:, 0].cpu().numpy(), strengths)
                    ]
                )
                weight[:, 0].copy_(torch.as_tensor(np.stack(projected), device=device))
            else:
                index, projected = quantize(weight[:, 0], centers)
            if step % 25 == 0 or step == steps:
                effective = weight if method == "wsr" else projected[:, None]
                score = losses(predict(qx, effective, tail, layout), qy)[:, 0].mean(-1)
                score = score.cpu().numpy()
                improved = score < best
                best[improved] = score[improved]
                labels = np.stack(index) if method == "wsr" else index.cpu().numpy()
                saved[improved] = labels[improved]
                saved_steps[improved] = step
        if step == steps:
            break
        if method == "lutq":
            weight = weight + (projected[:, None] - weight).detach()
        sample = torch.randint(
            sx.shape[1], (min(256, sx.shape[1]),), device=device, generator=generator
        )
        error = losses(predict(sx[:, sample], weight, tail, layout), sy[:, sample])
        optimizer.zero_grad(set_to_none=True)
        error.mean(-1).sum().backward()
        optimizer.step()
        if method == "lutq":
            centers = update_centers(parameter[:, 0, :n].detach(), centers)
    torch.cuda.synchronize(device)
    return saved, best, saved_steps, time.perf_counter() - started


def fit_target(panel, layout, index, device, *, steps, seed):
    sx, sy, qx, qy = panel
    if index is None:
        size = layout.connections
    else:
        size = int(np.abs(index).max())
        labels = torch.as_tensor(
            index.reshape(layout.fan_in, layout.sites, layout.channels)
            .transpose(1, 0, 2)
            .ravel(),
            device=device,
            dtype=torch.long,
        )
        position = labels.abs().sub(1).clamp_min(0)
        sign = labels.sign().float()
    parameter = initialization(layout, size, 1, len(sx), device, seed)
    optimizer = torch.optim.Adam([parameter], lr=0.01)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(steps):
        tail = parameter[..., size:]
        weight = parameter[..., :size]
        if index is not None:
            weight = weight[..., position] * sign
        error = losses(predict(sx, weight, tail, layout), sy)
        optimizer.zero_grad(set_to_none=True)
        error.mean(-1).sum().backward()
        optimizer.step()
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    memory = torch.cuda.max_memory_allocated(device)
    with torch.no_grad():
        tail = parameter[..., size:]
        weight = parameter[..., :size]
        if index is not None:
            weight = weight[..., position] * sign
        value = predict(qx, weight, tail, layout)
        return (
            value[:, 0].argmax(-1).cpu().numpy(),
            losses(value, qy)[:, 0].mean(-1).cpu().numpy(),
            seconds,
            memory,
        )
