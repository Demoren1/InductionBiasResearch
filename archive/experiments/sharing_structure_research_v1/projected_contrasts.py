"""Independent codebook directions for exact signed Gaussian contrasts."""

import numpy as np
import torch
from scipy.stats import f

from .contrasts import full_relations, noise_whitened_basis, resolve_graph
from .curvature_control import reduced_fit
from .gaussian_stats import fit_stats
from .gradient_spectrum import decode, spectral


@torch.no_grad()
def learned_directions(stats, n, plan, seed, union=False, mode="codebook"):
    if mode == "random":
        generator = torch.Generator(device=stats["gram"].device).manual_seed(seed)
        values = torch.randn(
            (len(stats["gram"]), plan["codes"]),
            generator=generator,
            device=stats["gram"].device,
            dtype=stats["gram"].dtype,
        )
        return torch.linalg.qr(values)[0]
    fitted, residual, inverse = fit_stats(stats)
    tasks, outputs, inputs = fitted.shape
    variance = residual.sum() / (tasks * outputs * (n - inputs))
    profiles = fitted.flatten(1).T
    raw = profiles @ profiles.T / tasks
    noise = torch.block_diag(*[inverse.mean(0) * variance] * outputs)
    if mode == "pca":
        return spectral(profiles.T @ profiles, plan["codes"])[0]
    if mode == "whitened_pca":
        eig, u = torch.linalg.eigh(noise)
        inverse_noise = (u * eig.reciprocal()[None]) @ u.T
        return spectral(profiles.T @ inverse_noise @ profiles, plan["codes"])[0]
    if mode != "codebook":
        raise ValueError(f"Unknown direction estimator: {mode}")
    values = {"whitened": noise_whitened_basis(profiles, noise, plan["codes"])[0]}
    if union:
        values["profiles"] = profiles
        values["corrected"] = spectral(raw - noise, plan["codes"])[0]
    codebooks = []
    for embedding in values.values():
        index, _, _ = decode(embedding, plan, seed)
        local, _ = reduced_fit(
            index.reshape(outputs, inputs), stats["gram"], stats["cross"], plan["codes"]
        )
        codebooks.append(local)
    matrix = torch.cat(codebooks, -1)
    u, singular, _ = torch.linalg.svd(matrix, full_matrices=False)
    rank = int((singular > singular.max() * 1e-10).sum())
    if rank == 0:
        raise FloatingPointError("No nonzero source-codebook direction")
    return u[:, :rank]


@torch.no_grad()
def projected_relations(fitted, inverse_gram, variance, degrees, directions, alpha):
    _tasks, outputs, inputs = fitted.shape
    entries, rank = outputs * inputs, directions.shape[-1]
    mean = fitted.flatten(1).T @ directions
    local_covariance = torch.einsum(
        "ta,tb,tij->ijab", directions, directions, inverse_gram
    )
    diagonal = torch.stack([local_covariance[i, i] for i in range(inputs)]).repeat(
        outputs, 1, 1
    )
    zero_q = (mean * torch.linalg.solve(diagonal, mean[..., None])[..., 0]).sum(-1)
    cutoff = float(f.isf(alpha / entries**2, rank, degrees))
    accepted_zero = (zero_q / (rank * variance) <= cutoff).cpu().numpy()
    relations = {}
    for sign in (-1, 1):
        difference = mean[:, None] - sign * mean[None]
        covariance = diagonal[:, None] + diagonal[None]
        for output in range(outputs):
            selected = slice(output * inputs, (output + 1) * inputs)
            covariance[selected, selected] -= 2 * sign * local_covariance
        if sign == 1:
            positions = torch.arange(entries, device=fitted.device)
            covariance[positions, positions] = torch.eye(
                rank, device=fitted.device, dtype=fitted.dtype
            )
        solution = torch.linalg.solve(covariance, difference[..., None])[..., 0]
        statistic = (difference * solution).sum(-1) / (rank * variance)
        relations[sign] = (statistic <= cutoff).cpu().numpy()
    return (
        accepted_zero,
        relations,
        {
            "rank": rank,
            "cutoff": cutoff,
            "alpha_this_statistic": alpha,
            "degrees": degrees,
        },
    )


@torch.no_grad()
def crossfit_graph(
    total,
    halves,
    n,
    plan,
    seed,
    union=False,
    fallback=True,
    direction_mode="codebook",
):
    alpha = plan["alpha"] / len(plan["observations"])
    count = 3 if fallback else 2
    zero_results, pair_results, statistics = [], [], []
    for learn, assess in ((halves[0], halves[1]), (halves[1], halves[0])):
        directions = learned_directions(
            learn, n // 2, plan, seed, union=union, mode=direction_mode
        )
        fitted, residual, inverse = fit_stats(assess)
        tasks, outputs, inputs = fitted.shape
        degrees = tasks * outputs * (n // 2 - inputs)
        variance = residual.sum() / degrees
        zero, pairs, metadata = projected_relations(
            fitted, inverse, variance, degrees, directions, alpha / count
        )
        zero_results.append(zero)
        pair_results.append(pairs)
        statistics.append(metadata)
    if fallback:
        fitted, residual, inverse = fit_stats(total)
        degrees = fitted.shape[0] * fitted.shape[1] * (n - fitted.shape[2])
        zero, pairs, metadata = full_relations(
            fitted, inverse, residual.sum() / degrees, degrees, alpha / count
        )
        zero_results.append(zero)
        pair_results.append(pairs)
        statistics.append(metadata)
    zero = np.logical_and.reduce(zero_results)
    pairs = {
        sign: np.logical_and.reduce([local[sign] for local in pair_results])
        for sign in (-1, 1)
    }
    index, metadata, components = resolve_graph(
        zero, pairs, plan["codes"], fitted.shape[1:]
    )
    return (
        index,
        {
            **metadata,
            "statistics": statistics,
            "union": union,
            "fallback": fallback,
            "alpha_this_look": alpha,
            "direction_mode": direction_mode,
        },
        components,
    )
