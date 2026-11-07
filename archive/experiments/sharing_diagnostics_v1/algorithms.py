"""Hard assignment search using observed source tasks only."""

import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.sharing_benchmark_v1.hard import (
    ProposalSelection,
    exact_projection,
    kmeans_start,
)
from experiments.sharing_benchmark_v1.linear import (
    adaptation_errors,
    index_basis,
)


def late_charge(charge, progress):
    return charge * min(1.0, max(0.0, (progress - 0.3) / 0.4))


@torch.no_grad()
def scalar_bank(
    rows,
    spec,
    plan,
    data,
    seed,
    *,
    late=False,
    trace=False,
    anchors=None,
    rounds=None,
    oracle_weights=None,
):
    device = data["train_support_x"].device
    anchors = min(anchors or plan["proposal_anchors"], spec["train_tasks"])
    rounds = rounds or spec["consensus_rounds"]
    count = len(rows)
    x = torch.cat((data["train_support_x"], data["train_query_x"]), 1)[:anchors]
    y = torch.cat((data["train_support_y"], data["train_query_y"]), 1)[:anchors]
    gram = x.transpose(-1, -2) @ x / x.shape[1]
    cross = x.transpose(-1, -2) @ y / x.shape[1]
    identity = torch.eye(spec["inputs"], device=device)
    reference = (
        torch.linalg.solve(gram.double() + 1e-5 * identity.double(), cross.double())
        .transpose(-1, -2)
        .float()
    )
    latent = reference[None].repeat(count, 1, 1, 1).flatten(0, 1)
    quantized = torch.zeros_like(latent)
    if oracle_weights is not None:
        latent = (
            oracle_weights[:anchors][None].repeat(count, 1, 1, 1).flatten(0, 1).clone()
        )
        quantized = latent.clone()
    generator = torch.Generator(device=device).manual_seed(plan["seed"] + 100003 * seed)
    centers = kmeans_start(latent, spec["codes"], generator)
    dual = torch.zeros_like(latent)
    last_rho = latent.new_full((count * anchors,), 0.01)
    finals = latent.new_tensor([row["rho_final"] for row in rows]).repeat_interleave(
        anchors
    )
    charges = latent.new_tensor([row["charge"] for row in rows]).repeat_interleave(
        anchors
    )
    dual_steps = latent.new_tensor(
        [row["dual_step"] for row in rows]
    ).repeat_interleave(anchors)
    local_gram = gram[None].repeat(count, 1, 1, 1).flatten(0, 1)
    local_cross = cross[None].repeat(count, 1, 1, 1).flatten(0, 1)
    selector = ProposalSelection(rows, anchors, spec, data)
    states = []
    started = time.perf_counter()
    for step in range(rounds + 1):
        progress = step / max(rounds, 1)
        rho = 0.01 * (finals / 0.01).pow(progress)
        charge = late_charge(charges, progress) if late else charges
        dual *= (last_rho / rho)[:, None, None]
        if step:
            alpha = rho / (2 * spec["inputs"])
            right = local_cross + alpha[:, None, None] * (quantized - dual).transpose(
                -1, -2
            )
            latent = torch.linalg.solve(
                local_gram + alpha[:, None, None] * identity, right
            ).transpose(-1, -2)
        values = latent + dual
        index, centers, quantized = exact_projection(values, spec["codes"], charge, rho)
        if trace:
            states.append(
                {
                    "step": step,
                    "indices": index.cpu().numpy(),
                    "values": values.cpu().numpy(),
                    "quantized": quantized.cpu().numpy(),
                    "centers": centers.cpu().numpy(),
                    "rho": rho.cpu().numpy(),
                    "charge": charge.cpu().numpy(),
                }
            )
        dual += dual_steps[:, None, None] * (latent - quantized)
        last_rho = rho
        if step % max(1, rounds // 4) == 0 or step == rounds:
            prediction = torch.einsum(
                "ani,raoi->rano",
                x,
                quantized.reshape(count, anchors, *reference.shape[1:]),
            )
            loss = (prediction - y[None]).square().mean((1, 2, 3))
            selector.update(step, index, loss, time.perf_counter() - started)
    return {
        "indices": selector.indices.cpu().numpy(),
        "basis": index_basis(selector.indices, spec["codes"]).cpu().numpy(),
        "selection_mse": selector.best_selection.cpu().numpy(),
        "step": selector.steps.cpu().numpy(),
        "history": selector.history,
        "seconds": time.perf_counter() - started,
        "trace": states,
    }


@torch.no_grad()
def vector_lloyd(values, centers, penalty, iterations=40):
    """Project all task weights onto one shared signed assignment, locally."""
    labels = None
    for _ in range(iterations):
        products = torch.bmm(values, centers.transpose(-1, -2))
        norms = centers.square().sum(-1)
        costs = torch.cat(
            (
                products[..., :1] * 0,
                norms[:, None] - 2 * products + penalty[:, None, None],
                norms[:, None] + 2 * products + penalty[:, None, None],
            ),
            -1,
        )
        proposed = costs.argmin(-1)
        if labels is not None and torch.equal(proposed, labels):
            break
        labels = proposed
        codes = centers.shape[1]
        signed = torch.where(labels > codes, codes - labels, labels)
        active = F.one_hot(signed.abs(), codes + 1)[..., 1:].to(values.dtype)
        counts = active.sum(1)
        weighted = active * signed.sign()[..., None]
        averages = torch.bmm(weighted.transpose(-1, -2), values) / counts[
            ..., None
        ].clamp_min(1)
        centers = torch.where((counts > 0)[..., None], averages, centers)
    dictionary = torch.cat((centers[:, :1] * 0, centers, -centers), 1)
    projected = dictionary.gather(1, labels[..., None].expand(-1, -1, values.shape[-1]))
    return signed, centers, projected


@torch.no_grad()
def joint_consensus(
    rows,
    spec,
    plan,
    data,
    seed,
    *,
    late,
    restarts=4,
    task_limit=None,
    projection_only=False,
):
    x = torch.cat((data["train_support_x"], data["train_query_x"]), 1)
    y = torch.cat((data["train_support_y"], data["train_query_y"]), 1)
    if task_limit is not None:
        x, y = x[:task_limit], y[:task_limit]
    device = x.device
    tasks, inputs = len(x), spec["inputs"]
    codes = spec["codes"]
    count = len(rows) * restarts
    gram = x.transpose(-1, -2) @ x / x.shape[1]
    cross = x.transpose(-1, -2) @ y / x.shape[1]
    identity = torch.eye(inputs, device=device)
    reference = (
        torch.linalg.solve(gram.double() + 1e-5 * identity.double(), cross.double())
        .transpose(-1, -2)
        .float()
    )
    latent = reference[None].expand(count, -1, -1, -1).clone()
    quantized, dual = torch.zeros_like(latent), torch.zeros_like(latent)
    finals = x.new_tensor([row["rho_final"] for row in rows]).repeat_interleave(
        restarts
    )
    charges = x.new_tensor([row["charge"] for row in rows]).repeat_interleave(restarts)
    dual_steps = x.new_tensor([row["dual_step"] for row in rows]).repeat_interleave(
        restarts
    )
    previous_rho = x.new_full((count,), 0.01)
    generator = torch.Generator(device=device).manual_seed(plan["seed"] + seed * 100003)
    vectors = latent.flatten(2).transpose(1, 2) / np.sqrt(tasks)
    centers = vectors.new_zeros((count, codes, tasks))
    distances = vectors.square().sum(-1)
    for code in range(codes):
        positions = torch.multinomial(distances + 1e-20, 1, generator=generator)
        centers[:, code] = vectors.gather(
            1, positions[..., None].expand(-1, -1, tasks)
        )[:, 0]
        plus = (vectors - centers[:, code, None]).square().sum(-1)
        minus = (vectors + centers[:, code, None]).square().sum(-1)
        distances = torch.minimum(distances, torch.minimum(plus, minus))
    selector = ProposalSelection(rows, restarts, spec, data)
    started = time.perf_counter()
    history = []
    rounds = 0 if projection_only else spec["consensus_rounds"]
    for step in range(rounds + 1):
        progress = step / rounds if rounds else 1.0
        rho = 0.01 * (finals / 0.01).pow(progress)
        charge = late_charge(charges, progress) if late else charges
        dual *= (previous_rho / rho)[:, None, None, None]
        if step:
            alpha = rho / (2 * inputs)
            right = cross[None] + alpha[:, None, None, None] * (
                quantized - dual
            ).transpose(-1, -2)
            latent = torch.linalg.solve(
                gram[None] + alpha[:, None, None, None] * identity, right
            ).transpose(-1, -2)
        vectors = (latent + dual).flatten(2).transpose(1, 2) / np.sqrt(tasks)
        signed, centers, projected = vector_lloyd(vectors, centers, 2 * charge / rho)
        index = signed.reshape(count, spec["outputs"], inputs)
        quantized = (projected.transpose(1, 2) * np.sqrt(tasks)).reshape_as(latent)
        dual += dual_steps[:, None, None, None] * (latent - quantized)
        previous_rho = rho
        if step % max(1, rounds // 4) == 0 or step == rounds:
            loss = (
                (torch.einsum("tni,rtoi->rtno", x, quantized) - y[None])
                .square()
                .mean((1, 2, 3))
            )
            selector.update(
                step,
                index,
                loss.reshape(len(rows), restarts).mean(-1),
                time.perf_counter() - started,
            )
            history.append(
                {
                    "step": step,
                    "consensus_mse": float((latent - quantized).square().mean()),
                    "active": (index != 0).sum((-1, -2)).tolist(),
                }
            )
    return {
        "indices": selector.indices.cpu().numpy(),
        "basis": index_basis(selector.indices, codes).cpu().numpy(),
        "selection_mse": selector.best_selection.cpu().numpy(),
        "step": selector.steps.cpu().numpy(),
        "history": selector.history,
        "algorithm_history": history,
        "seconds": time.perf_counter() - started,
        "trainable_numbers_per_task": codes,
        "projector": "Local vector assignment/center steps inside every consensus round; one common I and task-specific centers. No prelearned U, no fixed mask, no post-hoc clustering of another method.",
    }


def coordinate_statistics(index, x, y, codes):
    basis = index_basis(index[None], codes)[0].double()
    features = torch.einsum("oik,tni->tnok", basis, x)
    flat = features.flatten(1, 2)
    gram = flat.transpose(-1, -2) @ flat
    cross = (flat.transpose(-1, -2) @ y.flatten(1)[..., None])[..., 0]
    edge_cross = torch.einsum("tni,tnok->toik", x, features).flatten(1, 2)
    edge_target = torch.einsum("tni,tno->toi", x, y).flatten(1)
    edge_norm = x.square().sum(1)[:, None].expand(-1, index.shape[0], -1).flatten(1)
    return gram, cross, edge_cross, edge_target, edge_norm, y.square().sum((1, 2))


@torch.no_grad()
def coordinate_candidates(index, data, codes):
    """Exactly refit C for every one-entry edit using rank-two Gram updates."""
    labels = torch.arange(-codes, codes + 1, device=index.device)
    edges = torch.arange(index.numel(), device=index.device).repeat_interleave(
        len(labels)
    )
    proposed = labels.repeat(index.numel())
    current = index.flatten()[edges]
    changed = current != proposed
    edges, current, proposed = edges[changed], current[changed], proposed[changed]
    proposed_basis = (
        F.one_hot(proposed.abs().long(), codes + 1)[..., 1:].double()
        * proposed.sign()[..., None]
    )
    current_basis = (
        F.one_hot(current.abs().long(), codes + 1)[..., 1:].double()
        * current.sign()[..., None]
    )
    delta = proposed_basis - current_basis
    stats = []
    for prefix in ("train_support", "train_query"):
        x, y = data[f"{prefix}_x"].double(), data[f"{prefix}_y"].double()
        gram, cross, h, r, norm, squared_y = coordinate_statistics(index, x, y, codes)
        h, r, norm = h[:, edges], r[:, edges], norm[:, edges]
        update = h[..., :, None] * delta[None, ..., None, :]
        local_gram = gram[:, None] + update + update.transpose(-1, -2)
        local_gram += (
            norm[..., None, None]
            * delta[None, ..., :, None]
            * delta[None, ..., None, :]
        )
        local_cross = cross[:, None] + r[..., None] * delta
        stats.append((local_gram, local_cross, squared_y, y.shape[1] * y.shape[2]))
    coefficient = (
        torch.linalg.pinv(stats[0][0], hermitian=True, rtol=1e-10)
        @ stats[0][1][..., None]
    )[..., 0]
    gram, cross, squared_y, denominator = stats[1]
    errors = (
        (coefficient[..., None, :] @ gram @ coefficient[..., :, None])[..., 0, 0]
        - 2 * (cross * coefficient).sum(-1)
        + squared_y[:, None]
    ) / denominator
    return edges, proposed, errors.mean(0).clamp_min(0)


@torch.no_grad()
def coordinate_refine(initial, spec, data, steps=64):
    index = torch.tensor(initial, device=data["train_support_x"].device)
    codes = spec["codes"]
    best_index, best = index.clone(), np.inf
    history, selected_step = [], 0
    started = time.perf_counter()
    for step in range(steps + 1):
        basis = index_basis(index[None], codes)
        source = float(
            adaptation_errors(
                basis,
                data["train_support_x"],
                data["train_support_y"],
                data["train_query_x"],
                data["train_query_y"],
            ).mean()
        )
        validation = float(
            adaptation_errors(
                basis,
                data["selection_support_x"],
                data["selection_support_y"],
                data["selection_query_x"],
                data["selection_query_y"],
            ).mean()
        )
        if validation < best:
            best, best_index, selected_step = validation, index.clone(), step
        history.append(
            {
                "step": step,
                "source_query_mse": [source],
                "selection_mse": [validation],
                "best_selection_mse": [best],
                "seconds": time.perf_counter() - started,
            }
        )
        if step == steps:
            break
        edges, proposed, errors = coordinate_candidates(index, data, codes)
        winner = int(errors.argmin())
        if float(errors[winner]) >= source - 1e-9:
            break
        candidate = index.clone()
        candidate.flatten()[edges[winner]] = proposed[winner]
        direct = float(
            adaptation_errors(
                index_basis(candidate[None], codes),
                data["train_support_x"],
                data["train_support_y"],
                data["train_query_x"],
                data["train_query_y"],
            ).mean()
        )
        if abs(direct - float(errors[winner])) > 1e-6 * max(1, direct):
            raise FloatingPointError(
                "Incremental candidate score disagrees with explicit numeric refit"
            )
        index = candidate
    return {
        "indices": best_index[None].cpu().numpy(),
        "basis": index_basis(best_index[None], codes).cpu().numpy(),
        "selection_mse": np.array([best]),
        "step": np.array([selected_step]),
        "history": history,
        "seconds": time.perf_counter() - started,
        "algorithm": "At each round enumerate all one-entry signed/zero edits anywhere in I, exactly fit independent source C_t on support and choose the lowest mean source-query loss. Validation selects the checkpoint, never the edit. Every edge and sharing group remains editable.",
    }
