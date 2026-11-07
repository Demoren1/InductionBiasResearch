"""Exact bilevel categorical sharing and native continuous MSR/MTSR models."""

import itertools
import time

import numpy as np
import torch

from .hard import ProposalSelection
from .linear import Adam, adaptation_errors, design


def configurations(method):
    if method == "yeh":
        rows = [
            {"lr": lr, "entropy": entropy, "nuclear": nuclear, "inner_ridge": 1e-4}
            for lr, (entropy, nuclear) in itertools.product(
                (0.003, 0.01, 0.03), ((0.0, 0.0), (0.01, 0.0), (0.1, 0.0), (0.01, 0.01))
            )
        ]
    elif method == "msr":
        rows = [
            {"lr": lr, "inner_lr": inner}
            for lr, inner in itertools.product(
                (0.0005, 0.003, 0.01), (0.02, 0.1, 1.0, 5.0)
            )
        ]
    elif method == "mtsr":
        rows = [
            {"lr": lr, "l2": l2}
            for lr, l2 in itertools.product(
                (0.0005, 0.003, 0.01, 0.03), (0.0, 1e-4, 0.01)
            )
        ]
    else:
        raise ValueError(method)
    return [{"configuration": i, **row} for i, row in enumerate(rows)]


def run_yeh(rows, spec, plan, data, seed):
    device = data["train_support_x"].device
    anchors = min(plan["proposal_anchors"], spec["train_tasks"])
    codes, count = spec["codes"], len(rows)
    generator = torch.Generator(device=device).manual_seed(plan["seed"] + 100003 * seed)
    logits = 0.01 * torch.randn(
        (anchors, spec["outputs"], spec["inputs"], 2 * codes + 1),
        device=device,
        generator=generator,
    )
    logits[..., 0] += np.log(2 * codes)
    logits = logits.repeat(count, 1, 1, 1).requires_grad_()
    x = data["train_support_x"][:anchors][None].repeat(count, 1, 1, 1).flatten(0, 1)
    y = data["train_support_y"][:anchors][None].repeat(count, 1, 1, 1).flatten(0, 1)
    qx = data["train_query_x"][:anchors][None].repeat(count, 1, 1, 1).flatten(0, 1)
    qy = data["train_query_y"][:anchors][None].repeat(count, 1, 1, 1).flatten(0, 1)
    rates = logits.new_tensor([row["lr"] for row in rows]).repeat_interleave(anchors)
    entropy = logits.new_tensor([row["entropy"] for row in rows]).repeat_interleave(
        anchors
    )
    nuclear = logits.new_tensor([row["nuclear"] for row in rows]).repeat_interleave(
        anchors
    )
    ridge = logits.new_tensor([row["inner_ridge"] for row in rows]).repeat_interleave(
        anchors
    )
    optimizer = Adam([logits], rates)
    selector = ProposalSelection(rows, anchors, spec, data)
    soft_source = logits.new_full((count,), torch.inf)
    soft_best = logits.new_full((count,), torch.inf)
    soft_source_basis = None
    soft_basis = None
    soft_step = torch.zeros(count, device=device, dtype=torch.long)
    soft_history = []
    started = time.perf_counter()
    identity = torch.eye(codes, device=device)
    for step in range(spec["gradient_steps"] + 1):
        probability = logits.softmax(-1)
        basis = probability[..., 1 : codes + 1] - probability[..., codes + 1 :]
        features = torch.einsum("roik,rni->rnok", basis, x).flatten(1, 2)
        gram = features.transpose(-1, -2) @ features / features.shape[1]
        cross = (features.transpose(-1, -2) @ y.flatten(1)[..., None])[
            ..., 0
        ] / features.shape[1]
        coefficient = torch.linalg.solve(gram + ridge[:, None, None] * identity, cross)
        predicted = torch.einsum("roik,rk,rni->rno", basis, coefficient, qx)
        loss = (predicted - qy).square().mean((-1, -2))
        if step % spec["checkpoint_every"] == 0 or step == spec["gradient_steps"]:
            labels = probability.detach().argmax(-1)
            index = torch.where(labels > codes, codes - labels, labels)
            train_prediction = torch.einsum("roik,rk,rni->rno", basis, coefficient, x)
            training = (
                (train_prediction - y)
                .square()
                .mean((-1, -2))
                .detach()
                .reshape(count, anchors)
                .mean(-1)
            )
            selector.update(step, index, training, time.perf_counter() - started)
            with torch.no_grad():
                source_scores = (
                    adaptation_errors(
                        basis.detach(),
                        data["train_support_x"],
                        data["train_support_y"],
                        data["train_query_x"],
                        data["train_query_y"],
                    )
                    .mean(-1)
                    .reshape(count, anchors)
                )
                values, winners = source_scores.min(-1)
                proposals = basis.detach().reshape(count, anchors, *basis.shape[1:])[
                    torch.arange(count, device=device), winners
                ]
                changed = values < soft_source
                soft_source = torch.minimum(soft_source, values)
                if soft_source_basis is None:
                    soft_source_basis = proposals.clone()
                    soft_basis = proposals.clone()
                else:
                    soft_source_basis[changed] = proposals[changed]
                selection = adaptation_errors(
                    soft_source_basis,
                    data["selection_support_x"],
                    data["selection_support_y"],
                    data["selection_query_x"],
                    data["selection_query_y"],
                ).mean(-1)
                changed = selection < soft_best
                soft_basis[changed] = soft_source_basis[changed]
                soft_step[changed] = step
                soft_best = torch.minimum(soft_best, selection)
                soft_history.append(
                    {
                        "step": step,
                        "train_mse": training.tolist(),
                        "source_query_mse": soft_source.tolist(),
                        "selection_mse": selection.tolist(),
                        "best_selection_mse": soft_best.tolist(),
                        "seconds": time.perf_counter() - started,
                    }
                )
            print(
                "Yeh checkpoint",
                step,
                "soft",
                float(soft_best.min()),
                "MAP",
                float(selector.best_selection.min()),
                flush=True,
            )
        if step == spec["gradient_steps"]:
            break
        ent = -(probability * probability.clamp_min(1e-12).log()).sum(-1).mean((-1, -2))
        # Nuclear norm on the assignment matrix, as in Yeh et al.; zero/sign
        # constraints and the fixed K budget are explicit benchmark adaptations.
        matrix = probability.flatten(1, 2)
        penalized = nuclear > 0
        nuclear_loss = torch.zeros_like(loss)
        if penalized.any():
            nuclear_loss[penalized] = (
                nuclear[penalized]
                * torch.linalg.svdvals(matrix[penalized]).sum(-1)
                / np.sqrt(spec["inputs"] * spec["outputs"])
            )
        objective = loss + entropy * ent + nuclear_loss
        optimizer.step(torch.autograd.grad(objective.sum(), [logits]))
        with torch.no_grad():
            logits.sub_(logits.mean(-1, keepdim=True))
    from .linear import index_basis

    return {
        "indices": selector.indices.cpu().numpy(),
        "basis": index_basis(selector.indices, codes).cpu().numpy(),
        "selection_mse": selector.best_selection.cpu().numpy(),
        "step": selector.steps.cpu().numpy(),
        "history": selector.history,
        "seconds": time.perf_counter() - started,
        "proposal_anchors": anchors,
        "soft_basis": soft_basis.cpu().numpy(),
        "soft_selection_mse": soft_best.cpu().numpy(),
        "soft_step": soft_step.cpu().numpy(),
        "soft_history": soft_history,
    }


def inner_adapt(basis, initial, rate, x, y, steps):
    features = design(basis, x)
    gram = features.transpose(-1, -2) @ features / features.shape[2]
    cross = (features.transpose(-1, -2) @ y.flatten(1)[None, ..., None])[
        ..., 0
    ] / features.shape[2]
    coefficients = initial[:, None].expand(-1, len(x), -1)
    for _ in range(steps):
        coefficients = coefficients - 2 * rate[:, None] * (
            (gram @ coefficients[..., None])[..., 0] - cross
        )
    return coefficients


def run_continuous(method, rows, spec, plan, data, seed):
    count, codes = len(rows), spec["codes"]
    device = data["train_support_x"].device
    generator = torch.Generator(device=device).manual_seed(plan["seed"] + 100003 * seed)
    # Exactly ShareLinearFull initialization from the author implementation.
    initial_basis = 0.01 * torch.randn(
        (1, spec["outputs"], spec["inputs"], codes), device=device, generator=generator
    )
    basis = initial_basis.repeat(count, 1, 1, 1).requires_grad_()
    rates = basis.new_tensor([row["lr"] for row in rows])
    x = torch.cat((data["train_support_x"], data["train_query_x"]), 1)
    y = torch.cat((data["train_support_y"], data["train_query_y"]), 1)
    if method == "msr":
        initial = (
            (
                torch.randn((1, codes), device=device, generator=generator)
                / spec["outputs"]
            )
            .repeat(count, 1)
            .requires_grad_()
        )
        inner_rate = basis.new_tensor(
            [[row["inner_lr"]] for row in rows]
        ).requires_grad_()
        parameters = [basis, initial, inner_rate]
    else:
        initial = (
            (
                torch.randn((1, len(x), codes), device=device, generator=generator)
                / spec["outputs"]
            )
            .repeat(count, 1, 1)
            .requires_grad_()
        )
        parameters = [basis, initial]
    optimizer = Adam(parameters, rates)
    best = basis.new_full((count,), torch.inf)
    best_basis = basis.detach().clone()
    best_initial = initial.detach().clone()
    best_inner = (
        inner_rate.detach().clone() if method == "msr" else basis.new_zeros((count, 1))
    )
    best_step = torch.zeros(count, device=device, dtype=torch.long)
    history = []
    started = time.perf_counter()
    for step in range(spec["gradient_steps"] + 1):
        if method == "msr":
            tasks = torch.randperm(len(x), device=device, generator=generator)[:32]
            support_x, support_y = (
                data["train_support_x"][tasks],
                data["train_support_y"][tasks],
            )
            query_x, query_y = (
                data["train_query_x"][tasks],
                data["train_query_y"][tasks],
            )
            coefficients = inner_adapt(
                basis, initial, inner_rate, support_x, support_y, 3
            )
            predicted = torch.einsum("soik,stk,tni->stno", basis, coefficients, query_x)
            loss = (predicted - query_y[None]).square().mean((1, 2, 3))
        else:
            predicted = torch.einsum("soik,stk,tni->stno", basis, initial, x)
            loss = (predicted - y[None]).square().mean((1, 2, 3))
            l2 = basis.new_tensor([row["l2"] for row in rows])
            loss = loss + l2 * (
                basis.square().mean((1, 2, 3)) + initial.square().mean((1, 2))
            )
        if step % spec["checkpoint_every"] == 0 or step == spec["gradient_steps"]:
            selection = adaptation_errors(
                basis.detach(),
                data["selection_support_x"],
                data["selection_support_y"],
                data["selection_query_x"],
                data["selection_query_y"],
            ).mean(-1)
            with torch.no_grad():
                improved = selection < best
                best_basis[improved] = basis[improved]
                best_initial[improved] = initial[improved]
                if method == "msr":
                    best_inner[improved] = inner_rate[improved]
                best_step[improved] = step
                best = torch.minimum(best, selection)
            source = adaptation_errors(
                basis.detach(),
                data["train_support_x"],
                data["train_support_y"],
                data["train_query_x"],
                data["train_query_y"],
            ).mean(-1)
            history.append(
                {
                    "step": step,
                    "train_mse": loss.detach().tolist(),
                    "source_query_mse": source.tolist(),
                    "selection_mse": selection.tolist(),
                    "best_selection_mse": best.tolist(),
                    "seconds": time.perf_counter() - started,
                }
            )
        if step == spec["gradient_steps"]:
            break
        optimizer.step(torch.autograd.grad(loss.sum(), parameters))
    return {
        "basis": best_basis.cpu().numpy(),
        "selection_mse": best.cpu().numpy(),
        "initial": best_initial.cpu().numpy(),
        "inner_rate": best_inner.cpu().numpy(),
        "step": best_step.cpu().numpy(),
        "history": history,
        "seconds": time.perf_counter() - started,
        "final_basis": basis.detach().cpu().numpy(),
        "final_initial": initial.detach().cpu().numpy(),
        "final_inner_rate": inner_rate.detach().cpu().numpy()
        if method == "msr"
        else np.zeros((count, 1)),
    }
