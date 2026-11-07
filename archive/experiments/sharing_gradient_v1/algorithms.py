"""Common hard assignments, task-specific numbers and observation-only search."""

import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.sharing_benchmark_v1.hard import ProposalSelection
from experiments.sharing_benchmark_v1.linear import Adam, adaptation_errors, index_basis
from experiments.sharing_diagnostics_v1.algorithms import vector_lloyd


def source_statistics(data):
    x = torch.cat((data["train_support_x"], data["train_query_x"]), 1).double()
    y = torch.cat((data["train_support_y"], data["train_query_y"]), 1).double()
    gram = x.transpose(-1, -2) @ x / x.shape[1]
    cross = (x.transpose(-1, -2) @ y / x.shape[1]).transpose(-1, -2)
    identity = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
    dense = torch.linalg.solve(gram + 1e-8 * identity, cross.transpose(-1, -2))
    return x, y, gram, cross, dense.transpose(-1, -2)


def weights_from_codes(index, centers):
    basis = index_basis(index, centers.shape[-1]).to(centers.dtype)
    return torch.einsum("boik,btk->btoi", basis, centers)


@torch.no_grad()
def common_start(reference, count, restarts, codes, generator):
    tasks = len(reference)
    latent = reference[None].expand(count * restarts, -1, -1, -1).clone()
    vectors = latent.flatten(2).transpose(1, 2) / np.sqrt(tasks)
    centers = vectors.new_zeros((len(vectors), codes, tasks))
    distance = vectors.square().sum(-1)
    for code in range(codes):
        position = torch.multinomial(distance + 1e-20, 1, generator=generator)
        centers[:, code] = vectors.gather(1, position[..., None].expand(-1, -1, tasks))[
            :, 0
        ]
        plus = (vectors - centers[:, code, None]).square().sum(-1)
        minus = (vectors + centers[:, code, None]).square().sum(-1)
        distance = torch.minimum(distance, torch.minimum(plus, minus))
    labels, centers, projected = vector_lloyd(
        vectors, centers, vectors.new_zeros(len(vectors))
    )
    index = labels.reshape(len(vectors), *reference.shape[1:])
    quantized = (projected.transpose(1, 2) * np.sqrt(tasks)).reshape_as(latent)
    centers = centers.transpose(1, 2) * np.sqrt(tasks)
    return latent, index, centers, quantized


@torch.no_grad()
def metric_centers(index, metric, cross, old_centers):
    basis = index_basis(index, old_centers.shape[-1]).to(cross.dtype)
    transformed = torch.einsum("btij,bojk->btoik", metric, basis)
    gram = torch.einsum("boik,btoil->btkl", basis, transformed)
    right = torch.einsum("boik,btoi->btk", basis, cross)
    eye = torch.eye(old_centers.shape[-1], device=cross.device, dtype=cross.dtype)
    # Preserve an unused center; populated groups get a negligible proximal term.
    return torch.linalg.solve(
        gram + 1e-9 * eye, (right + 1e-9 * old_centers)[..., None]
    )[..., 0]


@torch.no_grad()
def metric_projection(index, centers, metric, cross, sweeps):
    """Local full-metric assignment/C minimization, without per-candidate C fits."""
    codes = centers.shape[-1]
    for _ in range(sweeps):
        previous = index.clone()
        centers = metric_centers(index, metric, cross, centers)
        weights = weights_from_codes(index, centers)
        residual = torch.einsum("btoi,btij->btoj", weights, metric) - cross
        dictionary = torch.cat((centers[..., :1] * 0, centers, -centers), -1)
        for position in range(index.shape[-1]):
            delta = dictionary[:, :, None] - weights[..., position, None]
            costs = (
                2 * residual[..., position, None] * delta
                + metric[..., position, position, None, None] * delta.square()
            ).mean(1)
            label = costs.argmin(-1)
            selected = dictionary.gather(
                2, label[:, None].expand(-1, centers.shape[1], -1)
            )
            change = selected - weights[..., position]
            weights[..., position] = selected
            residual += change[..., None] * metric[..., position, None, :]
            index[..., position] = torch.where(label > codes, codes - label, label)
        if torch.equal(previous, index):
            break
    centers = metric_centers(index, metric, cross, centers)
    return index, centers, weights_from_codes(index, centers)


def packed_result(selector, spec, started, history, checkpoints):
    return {
        "indices": selector.indices.cpu().numpy(),
        "basis": index_basis(selector.indices, spec["codes"]).cpu().numpy(),
        "selection_mse": selector.best_selection.cpu().numpy(),
        "step": selector.steps.cpu().numpy(),
        "history": selector.history,
        "optimization_history": history,
        "checkpoint_indices": np.stack(checkpoints),
        "seconds": time.perf_counter() - started,
    }


@torch.no_grad()
def projection_search(method, rows, spec, plan, data, seed):
    x, y, gram, cross, reference = source_statistics(data)
    restarts = plan["restarts"]
    generator = torch.Generator(device=x.device).manual_seed(
        plan["seed"] + seed * 100003
    )
    latent, index, centers, quantized = common_start(
        reference, len(rows), restarts, spec["codes"], generator
    )
    dual = torch.zeros_like(latent)
    finals = x.new_tensor([row["rho_final"] for row in rows]).repeat_interleave(
        restarts
    )
    dual_steps = x.new_tensor([row["dual_step"] for row in rows]).repeat_interleave(
        restarts
    )
    eye = torch.eye(spec["inputs"], device=x.device, dtype=x.dtype)
    previous_rho = torch.full_like(finals, 0.01)
    selector = ProposalSelection(rows, restarts, spec, data)
    started = time.perf_counter()
    history, checkpoints = [], []
    initial_index = index.clone()
    for step in range(spec["consensus_rounds"] + 1):
        rho = 0.01 * (finals / 0.01).pow(step / spec["consensus_rounds"])
        alpha = rho / (2 * spec["inputs"])
        dual *= (previous_rho / rho)[:, None, None, None]
        if step:
            right = cross[None] + alpha[:, None, None, None] * (quantized - dual)
            latent = torch.linalg.solve(
                gram[None] + alpha[:, None, None, None] * eye,
                right.transpose(-1, -2),
            ).transpose(-1, -2)
            values = latent + dual
            if method == "z_euclidean":
                vectors = values.flatten(2).transpose(1, 2) / np.sqrt(len(x))
                signed, fitted, projected = vector_lloyd(
                    vectors,
                    centers.transpose(1, 2) / np.sqrt(len(x)),
                    values.new_zeros(len(values)),
                    iterations=plan["projection_sweeps"],
                )
                index = signed.reshape_as(index)
                centers = fitted.transpose(1, 2) * np.sqrt(len(x))
                quantized = (projected.transpose(1, 2) * np.sqrt(len(x))).reshape_as(
                    values
                )
            else:
                metric = gram[None] + alpha[:, None, None, None] * eye
                if method == "z_hessian":
                    target_cross = torch.einsum("btoi,btij->btoj", values, metric)
                elif method == "z_taylor":
                    target_cross = cross[None] + alpha[:, None, None, None] * values
                else:
                    raise ValueError(method)
                index, centers, quantized = metric_projection(
                    index, centers, metric, target_cross, plan["projection_sweeps"]
                )
            dual += dual_steps[:, None, None, None] * (latent - quantized)
        previous_rho = rho
        if (
            step % plan["projection_checkpoint"] == 0
            or step == spec["consensus_rounds"]
        ):
            prediction = torch.einsum("tni,btoi->btno", x, quantized)
            loss = (prediction - y[None]).square().mean((1, 2, 3))
            selector.update(
                step,
                index,
                loss.reshape(len(rows), restarts).mean(-1),
                time.perf_counter() - started,
            )
            history.append(
                {
                    "step": step,
                    "current_source_fit_mse": loss.reshape(len(rows), restarts)
                    .mean(-1)
                    .tolist(),
                    "consensus_mse": float((latent - quantized).square().mean()),
                    "active_mean": (index != 0)
                    .sum((-1, -2))
                    .reshape(len(rows), restarts)
                    .double()
                    .mean(-1)
                    .tolist(),
                }
            )
            checkpoints.append(index.cpu().numpy().astype(np.int16))
    result = packed_result(selector, spec, started, history, checkpoints)
    result.update(
        initial_indices=initial_index.cpu().numpy().astype(np.int16),
        final_latent=latent.cpu().numpy(),
        final_values=(latent + dual).cpu().numpy(),
        final_indices=index.cpu().numpy().astype(np.int16),
        final_codes=centers.cpu().numpy(),
        source_fit_reference=reference.cpu().numpy(),
    )
    return result


def support_fit(basis, x, y, ridge):
    features = torch.einsum("boik,tni->btnok", basis, x).flatten(2, 3)
    target = y.flatten(1, 2)
    gram = features.transpose(-1, -2) @ features / features.shape[-2]
    cross = (features.transpose(-1, -2) @ target[None, ..., None])[..., 0]
    cross /= features.shape[-2]
    eye = torch.eye(basis.shape[-1], device=basis.device, dtype=basis.dtype)
    return torch.linalg.solve(gram + ridge * eye, cross[..., None])[..., 0]


def meta_objective(basis, data, ridge, stop_inner=False):
    centers = support_fit(
        basis, data["train_support_x"].double(), data["train_support_y"].double(), ridge
    )
    if stop_inner:
        centers = centers.detach()
    weights = torch.einsum("boik,btk->btoi", basis, centers)
    prediction = torch.einsum("tni,btoi->btno", data["train_query_x"].double(), weights)
    return (prediction - data["train_query_y"].double()[None]).square().mean((1, 2, 3))


def hard_basis(latent):
    size, which = latent.abs().max(-1)
    selected = latent.gather(-1, which[..., None])[..., 0].sign()
    index = torch.where(size > 0.5, (which + 1) * selected, 0).long()
    return index, index_basis(index, latent.shape[-1]).to(latent.dtype)


def meta_search(method, rows, spec, plan, data, seed, initial):
    device = data["train_support_x"].device
    restarts = plan["restarts"]
    base = index_basis(
        torch.tensor(initial[None], device=device), spec["codes"]
    ).double()
    latent = base.repeat(len(rows) * restarts, 1, 1, 1)
    generator = torch.Generator(device=device).manual_seed(
        plan["seed"] + seed * 100003 + 17
    )
    latent += 0.01 * torch.randn(
        latent.shape, generator=generator, device=device, dtype=latent.dtype
    )
    latent.requires_grad_()
    rates = latent.new_tensor([row["lr"] for row in rows]).repeat_interleave(restarts)
    optimizer = Adam([latent], rates)
    selector = ProposalSelection(rows, restarts, spec, data)
    started = time.perf_counter()
    history, checkpoints = [], []
    for step in range(plan["meta_steps"] + 1):
        index, projected = hard_basis(latent)
        basis = latent + (projected - latent).detach()
        loss = meta_objective(basis, data, plan["inner_ridge"], method == "meta_stop")
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Non-finite hard-forward meta loss")
        if step % plan["meta_checkpoint"] == 0 or step == plan["meta_steps"]:
            selector.update(
                step,
                index,
                loss.detach().reshape(len(rows), restarts).mean(-1),
                time.perf_counter() - started,
            )
            history.append(
                {
                    "step": step,
                    "current_meta_mse": loss.detach()
                    .reshape(len(rows), restarts)
                    .mean(-1)
                    .tolist(),
                    "active_mean": (index != 0)
                    .sum((-1, -2))
                    .reshape(len(rows), restarts)
                    .double()
                    .mean(-1)
                    .tolist(),
                    "latent_max": float(latent.detach().abs().max()),
                }
            )
            checkpoints.append(index.detach().cpu().numpy().astype(np.int16))
        if step == plan["meta_steps"]:
            break
        gradient = torch.autograd.grad(loss.sum(), latent)[0]
        optimizer.step([gradient])
        with torch.no_grad():
            latent.clamp_(-plan["latent_clip"], plan["latent_clip"])
    return packed_result(selector, spec, started, history, checkpoints)


def gradient_audit(initial, data, codes, ridge):
    basis = index_basis(initial[None], codes).double().requires_grad_()
    full = torch.autograd.grad(meta_objective(basis, data, ridge).sum(), basis)[0]
    stopped = torch.autograd.grad(
        meta_objective(basis, data, ridge, stop_inner=True).sum(), basis
    )[0]
    direction = full / full.norm().clamp_min(1e-15)
    epsilon = 1e-5
    with torch.no_grad():
        finite = (
            meta_objective(basis + epsilon * direction, data, ridge)
            - meta_objective(basis - epsilon * direction, data, ridge)
        ) / (2 * epsilon)
    analytical = (full * direction).sum()
    error = float((finite[0] - analytical).abs() / analytical.abs().clamp_min(1e-8))
    if error > 1e-4:
        raise FloatingPointError(
            "Codebook hypergradient disagrees with directional derivative"
        )
    return {
        "full_gradient_norm": float(full.norm()),
        "stop_gradient_norm": float(stopped.norm()),
        "inner_response_norm": float((full - stopped).norm()),
        "cosine_full_stop": float(
            F.cosine_similarity(full.flatten(), stopped.flatten(), dim=0)
        ),
        "directional_relative_error": error,
    }


def projected_meta_search(method, spec, plan, data, initial):
    """Raw meta-gradient, parallel hard proposals and exact outer line search."""
    device = data["train_support_x"].device
    index = torch.tensor(initial[None], device=device).long()
    selector = ProposalSelection([{}], 1, spec, data)
    settings = plan["projected_gradient"]
    scales = torch.tensor(settings["displacements"], device=device, dtype=torch.float64)
    started = time.perf_counter()
    history, checkpoints = [], []
    for step in range(settings["max_steps"] + 1):
        basis = index_basis(index, spec["codes"]).double().requires_grad_()
        objective = meta_objective(
            basis, data, plan["inner_ridge"], method == "projected_stop"
        )
        selector.update(step, index, objective.detach(), time.perf_counter() - started)
        history.append(
            {
                "step": step,
                "current_meta_mse": objective.detach().tolist(),
                "active_mean": (index != 0).sum((-1, -2)).double().tolist(),
            }
        )
        checkpoints.append(index.cpu().numpy().astype(np.int16))
        if step == settings["max_steps"]:
            break
        gradient = torch.autograd.grad(objective.sum(), basis)[0]
        direction = gradient / gradient.abs().max().clamp_min(1e-12)
        with torch.no_grad():
            proposals, _ = hard_basis(
                basis.detach() - scales[:, None, None, None] * direction
            )
            proposals = torch.cat((index, proposals))
            scores = adaptation_errors(
                index_basis(proposals, spec["codes"]),
                data["train_support_x"],
                data["train_support_y"],
                data["train_query_x"],
                data["train_query_y"],
            ).mean(-1)
            winner = int(scores.argmin())
            if float(scores[winner]) >= float(scores[0]) - 1e-9:
                break
            index = proposals[winner : winner + 1].clone()
    return packed_result(selector, spec, started, history, checkpoints)
