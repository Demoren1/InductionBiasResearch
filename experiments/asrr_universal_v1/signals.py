"""Estimate the two source signals from independent support and queries."""

import numpy as np
import torch

from experiments.asrr_benchmark_families_v1.gradient_transfer import (
    initial_tail,
    per_input_gradient,
    reference,
)
from experiments.asrr_transfer_refinement_v1.common import GRADIENT_DRAWS, SEED
from experiments.asrr_transfer_refinement_v1.contrasts import means_and_noise
from experiments.asrr_transfer_refinement_v1.reliable import (
    covariance_differences,
    symmetric,
)
from experiments.real_group_transfer_v1.optimization import losses, predict

from .structure import unit_frobenius


def profile_kernels(profiles, blocks):
    matrices = []
    for block in np.split(profiles, blocks, axis=-1):
        rows = block.transpose(1, 0, 2).reshape(block.shape[1], -1)
        matrices.append(unit_frobenius(rows @ rows.T))
    return matrices


def class_signals(sx, sy, queries, qy, layout):
    sy_cpu, qy_cpu = sy.cpu().numpy(), qy.cpu().numpy()
    a, sa = means_and_noise(sx.cpu().numpy(), sy_cpu, layout.outputs)
    left = covariance_differences(sx, sy, layout)
    mean = np.zeros_like(sa)
    cov = np.zeros_like(sa)
    noise = np.zeros_like(sa)
    agreement = np.zeros(2)
    for qx in queries.values():
        b, sb = means_and_noise(qx.cpu().numpy(), qy_cpu, layout.outputs)
        right = covariance_differences(qx, qy, layout)
        cross = np.einsum("tcsi,tcsj->sij", a, b, optimize=True)
        mean += symmetric(cross / (len(a) * layout.outputs))
        cov += symmetric((left @ right).mean((0, 1))).cpu().numpy()
        noise += (sa + sb) / 2
        agreement[0] += np.vdot(a, b) / max(
            np.linalg.norm(a) * np.linalg.norm(b), 1e-12
        )
        denominator = (
            torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
        ).item()
        agreement[1] += torch.dot(
            left.flatten(), right.flatten()
        ).item() / max(denominator, 1e-12)
    return (
        unit_frobenius(mean),
        unit_frobenius(cov),
        noise / len(queries),
        np.clip(agreement / len(queries), 0, 1 - 1e-6),
    )


def gradient_signals(ordered, queries, qy, layout, shot, basis, *, splits=1):
    device = ordered.device
    rng = torch.Generator(device=device).manual_seed(SEED + 143)
    tasks = torch.arange(len(ordered), device=device)
    classes = torch.arange(layout.outputs, device=device)
    sy = classes.repeat_interleave(shot).expand(len(tasks), -1)
    shape = (splits, layout.hidden, layout.fan_in, layout.fan_in)
    kernels = {name: torch.zeros(shape, device=device) for name in queries}
    difficulty = dict.fromkeys(queries, 0.0)
    basis = torch.as_tensor(basis, device=device, dtype=torch.float32)

    for _ in range(GRADIENT_DRAWS):
        order = torch.rand(
            ordered.shape[:3], device=device, generator=rng
        ).argsort()[..., :shot]
        sx = ordered[tasks[:, None, None], classes[None, :, None], order]
        sx = sx.flatten(1, 2)
        tail = initial_tail(len(tasks), layout, device, rng)
        weights, tail = reference(sx, sy, layout, basis, tail, rng)
        gs = per_input_gradient(sx, sy, weights, tail, layout).permute(2, 1, 0)
        for name, qx in queries.items():
            gq = per_input_gradient(qx, qy, weights, tail, layout).permute(
                2, 1, 0
            )
            for split in range(splits):
                a, b = gs[..., split::splits], gq[..., split::splits]
                kernels[name][split] += symmetric(a @ b.mT)
            with torch.no_grad():
                difficulty[name] += (
                    losses(predict(qx, weights, tail, layout), qy)
                    .mean()
                    .item()
                )

    result = {}
    for name, value in kernels.items():
        estimates = value.cpu().numpy() / GRADIENT_DRAWS
        if splits > 1:
            counts = [len(range(i, len(tasks), splits)) for i in range(splits)]
            estimates /= np.asarray(counts)[:, None, None, None]
        result[name] = estimates[0] if splits == 1 else estimates
    return result, {
        name: value / GRADIENT_DRAWS for name, value in difficulty.items()
    }


def block_gradients(kernels, difficulty, blocks):
    panels = sorted(kernels)
    aggregated = {
        name: [
            unit_frobenius(chunk.sum(0)) for chunk in np.split(value, blocks)
        ]
        for name, value in kernels.items()
    }
    if blocks < len(panels):
        return [
            unit_frobenius(sum(values))
            for values in zip(*aggregated.values(), strict=True)
        ]
    hardest = max(panels, key=difficulty.__getitem__)
    assignment = panels + [hardest] * (blocks - len(panels))
    return [aggregated[name][i] for i, name in enumerate(assignment)]
