"""Estimate support-query cross-gradient kernels at shared reference fits."""

import math

import torch

from experiments.asrr_benchmark_families_v1.gradient_mixture import (
    clean_source_queries,
)
from experiments.asrr_benchmark_families_v1.gradient_transfer import (
    initial_tail,
    per_input_gradient,
    reference,
)
from experiments.paper_transfer_v2.bilevel import stratified_sources

from .common import GRADIENT_DRAWS, SEED


def source_kernels(dataset, layout, shot, basis, device):
    sx, qx, qy = stratified_sources(dataset, device, layout)
    queries = {"adapted": qx}
    if dataset == "aug_omniglot":
        clean = clean_source_queries()[:, -qx.shape[1] :]
        queries["clean"] = (
            torch.tensor(clean, device=device, dtype=torch.float32) / 255
        )
    rng = torch.Generator(device=device).manual_seed(SEED + 143)
    classes = torch.arange(layout.outputs, device=device)
    tasks = torch.arange(len(sx), device=device)
    sy = classes.repeat_interleave(shot).expand(len(tasks), -1)
    count = min(16, layout.hidden)
    dims = (count, layout.fan_in, layout.fan_in)
    signals = {
        name: torch.zeros(dims, device=device)
        for name in ("initial", *queries)
    }
    basis = torch.as_tensor(basis, device=device, dtype=torch.float32)
    for _ in range(GRADIENT_DRAWS):
        order = torch.rand(
            sx.shape[:3], device=device, generator=rng
        ).argsort()[..., :shot]
        support = sx[
            tasks[:, None, None], classes[None, :, None], order
        ].flatten(1, 2)
        tail = initial_tail(len(support), layout, device, rng)
        common = torch.randn(
            len(support),
            1,
            layout.sites,
            1,
            layout.channels,
            device=device,
            generator=rng,
        ) * math.sqrt(2 / layout.fan_in)
        common = common.expand(-1, -1, -1, layout.fan_in, -1).flatten(2)
        fitted = reference(support, sy, layout, basis, tail, rng)
        for (weight, head), panels in (
            ((common, tail), {"initial": qx}),
            (fitted, queries),
        ):
            grad = per_input_gradient(support, sy, weight, head, layout)
            gs = (
                grad.reshape(len(grad), layout.fan_in, count, -1)
                .permute(2, 1, 0, 3)
                .flatten(2)
            )
            for name, panel in panels.items():
                grad = per_input_gradient(panel, qy, weight, head, layout)
                gq = (
                    grad.reshape(len(grad), layout.fan_in, count, -1)
                    .permute(2, 1, 0, 3)
                    .flatten(2)
                )
                cross = gs @ gq.mT
                signals[name] += 0.5 * (cross + cross.mT)
    return {
        key: (val / GRADIENT_DRAWS).cpu().numpy()
        for key, val in signals.items()
    }
