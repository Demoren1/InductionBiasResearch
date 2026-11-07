"""Scalar discovery algorithms with a shared, observation-only proposal bank."""

import itertools
import time

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.codebook_projection import (
    exact_signed_projection,
)

from .linear import Adam, adaptation_errors, index_basis


def configurations(method):
    if method in ("consensus_dp", "lc_matched"):
        rows = [
            {"rho_final": rho, "charge": charge, "dual_step": dual}
            for rho, charge, dual in itertools.product(
                (10.0, 100.0), (0.0, 0.03, 0.3), (0.25, 1.0)
            )
        ]
    elif method == "lc":
        rows = [
            {
                "rho_initial": initial,
                "rho_final": final,
                "charge": 0.0,
                "dual_step": 1.0,
            }
            for initial, final in itertools.product(
                (0.001, 0.01, 0.1), (1.0, 10.0, 100.0, 1000.0)
            )
        ]
    elif method == "lutq":
        rows = [
            {"lr": lr, "initial": initial}
            for lr, initial in itertools.product(
                (0.001, 0.003, 0.01, 0.03), ("dense", "random_small", "random")
            )
        ]
    elif method in ("posthoc_dp", "posthoc_lloyd"):
        rows = [{"tau": float(tau)} for tau in np.geomspace(1e-6, 1, 12)]
    else:
        raise ValueError(method)
    return [{"configuration": i, **row} for i, row in enumerate(rows)]


@torch.no_grad()
def kmeans_start(values, codes, generator):
    flat = values.flatten(1)
    centers = values.new_zeros((len(values), codes))
    distance = flat.square()
    for code in range(codes):
        probabilities = distance + 1e-20
        positions = torch.multinomial(probabilities, 1, generator=generator)
        center = flat.gather(1, positions)[:, 0].abs()
        centers[:, code] = center
        distance = torch.minimum(distance, (flat.abs() - center[:, None]).square())
    return centers.sort(-1).values


@torch.no_grad()
def lloyd(values, centers, tau, iterations):
    flat = values.flatten(1)
    codes = centers.shape[-1]
    for _ in range(iterations):
        dictionary = torch.cat((centers[:, :1] * 0, centers, -centers), -1)
        costs = (flat[..., None] - dictionary[:, None]).square()
        costs[..., 1:] += tau[:, None, None]
        labels = costs.argmin(-1)
        sign_index = torch.where(labels > codes, codes - labels, labels)
        for code in range(1, codes + 1):
            active = sign_index.abs() == code
            counts = active.sum(-1)
            means = (flat.abs() * active).sum(-1) / counts.clamp_min(1)
            centers[:, code - 1] = torch.where(counts > 0, means, centers[:, code - 1])
        # Keep labels attached to their updated centers, including for M=1 LUT-Q.
    dictionary = torch.cat((centers[:, :1] * 0, centers, -centers), -1)
    indices = torch.where(labels > codes, codes - labels, labels).reshape(values.shape)
    quantized = dictionary.gather(1, labels).reshape(values.shape)
    return indices, centers, quantized


@torch.no_grad()
def exact_projection(values, codes, charge, rho):
    # Bound temporary N*N interval storage, including the original-size benchmark.
    chunk = max(1, 2_000_000 // values[0].numel() ** 2)
    states = []
    for start in range(0, len(values), chunk):
        stop = start + chunk
        labels, centers, quantized = exact_signed_projection(
            values[start:stop], codes, charge[start:stop], rho[start:stop]
        )
        indices = torch.where(labels > codes, codes - labels, labels)
        states.append((indices, centers, quantized))
    return tuple(torch.cat([state[i] for state in states]) for i in range(3))


class ProposalSelection:
    """Select a single common I using all source tasks, then independent tasks."""

    def __init__(self, rows, anchors, spec, data):
        self.rows = rows
        self.anchors = anchors
        self.spec = spec
        self.data = data
        self.best_source = torch.full(
            (len(rows),), torch.inf, device=data["train_support_x"].device
        )
        self.best_selection = torch.full_like(self.best_source, torch.inf)
        self.source_indices = None
        self.indices = None
        self.steps = torch.zeros(
            len(rows), dtype=torch.long, device=self.best_source.device
        )
        self.history = []

    @torch.no_grad()
    def update(self, step, indices, train_loss, elapsed):
        basis = index_basis(indices, self.spec["codes"])
        source = (
            adaptation_errors(
                basis,
                self.data["train_support_x"],
                self.data["train_support_y"],
                self.data["train_query_x"],
                self.data["train_query_y"],
            )
            .mean(-1)
            .reshape(len(self.rows), self.anchors)
        )
        values, winners = source.min(-1)
        proposed = indices.reshape(len(self.rows), self.anchors, *indices.shape[1:])[
            torch.arange(len(self.rows), device=indices.device), winners
        ]
        improved = values < self.best_source
        self.best_source = torch.minimum(self.best_source, values)
        if self.source_indices is None:
            self.source_indices = proposed.clone()
            self.indices = proposed.clone()
        else:
            self.source_indices[improved] = proposed[improved]
        selection = adaptation_errors(
            index_basis(self.source_indices, self.spec["codes"]),
            self.data["selection_support_x"],
            self.data["selection_support_y"],
            self.data["selection_query_x"],
            self.data["selection_query_y"],
        ).mean(-1)
        improved = selection < self.best_selection
        self.indices[improved] = self.source_indices[improved]
        self.steps[improved] = step
        self.best_selection = torch.minimum(self.best_selection, selection)
        self.history.append(
            {
                "step": step,
                "train_mse": train_loss.tolist(),
                "source_query_mse": self.best_source.tolist(),
                "selection_mse": selection.tolist(),
                "best_selection_mse": self.best_selection.tolist(),
                "seconds": elapsed,
            }
        )


def run(method, rows, spec, plan, data, seed):
    device = data["train_support_x"].device
    anchors = min(plan["proposal_anchors"], spec["train_tasks"])
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
    local_x = x[None].repeat(count, 1, 1, 1).flatten(0, 1)
    local_y = y[None].repeat(count, 1, 1, 1).flatten(0, 1)
    generator = torch.Generator(device=device).manual_seed(plan["seed"] + 100003 * seed)
    selector = ProposalSelection(rows, anchors, spec, data)
    started = time.perf_counter()
    histories = []
    if method in ("consensus_dp", "lc", "lc_matched", "posthoc_dp", "posthoc_lloyd"):
        centers = kmeans_start(latent, spec["codes"], generator)
        dual = torch.zeros_like(latent)
        quantized = torch.zeros_like(latent)
        last_rho = latent.new_full((count * anchors,), 0.01)
        rounds = 0 if method.startswith("posthoc") else spec["consensus_rounds"]
        row_initial = latent.new_tensor(
            [row.get("rho_initial", 0.01) for row in rows]
        ).repeat_interleave(anchors)
        row_final = latent.new_tensor(
            [row.get("rho_final", 1.0) for row in rows]
        ).repeat_interleave(anchors)
        charge = latent.new_tensor(
            [row.get("charge", row.get("tau", 0) / 2) for row in rows]
        ).repeat_interleave(anchors)
        dual_step = latent.new_tensor(
            [row.get("dual_step", 0) for row in rows]
        ).repeat_interleave(anchors)
        local_gram = gram[None].repeat(count, 1, 1, 1).flatten(0, 1)
        local_cross = cross[None].repeat(count, 1, 1, 1).flatten(0, 1)
        for step in range(rounds + 1):
            progress = step / max(rounds, 1)
            rho = row_initial * (row_final / row_initial).pow(progress)
            if method.startswith("posthoc"):
                rho = torch.ones_like(rho)
            dual *= (last_rho / rho)[:, None, None]
            if step and not method.startswith("posthoc"):
                alpha = rho / (2 * spec["inputs"])
                right = local_cross + alpha[:, None, None] * (
                    quantized - dual
                ).transpose(-1, -2)
                latent = torch.linalg.solve(
                    local_gram + alpha[:, None, None] * identity, right
                ).transpose(-1, -2)
            if method in ("consensus_dp", "posthoc_dp"):
                index, centers, quantized = exact_projection(
                    latent + dual, spec["codes"], charge, rho
                )
            else:
                index, centers, quantized = lloyd(
                    latent + dual, centers, 2 * charge / rho, 30
                )
            dual += dual_step[:, None, None] * (latent - quantized)
            last_rho = rho
            if step % max(1, rounds // 4) == 0 or step == rounds:
                loss = (
                    (torch.einsum("rni,roi->rno", local_x, quantized) - local_y)
                    .square()
                    .mean((-1, -2))
                    .reshape(count, anchors)
                    .mean(-1)
                )
                selector.update(step, index, loss, time.perf_counter() - started)
                histories.append(
                    {
                        "step": step,
                        "consensus_mse": (latent - quantized).square().mean().item(),
                    }
                )
    elif method == "lutq":
        for number, row in enumerate(rows):
            if row["initial"] != "dense":
                scale = 0.01 if row["initial"] == "random_small" else 0.1
                latent[number * anchors : (number + 1) * anchors] = (
                    torch.randn(
                        (anchors, spec["outputs"], spec["inputs"]),
                        generator=generator,
                        device=device,
                    )
                    * scale
                )
        latent.requires_grad_()
        rates = latent.new_tensor([row["lr"] for row in rows]).repeat_interleave(
            anchors
        )
        optimizer = Adam([latent], rates)
        centers = kmeans_start(latent.detach(), spec["codes"], generator)
        zeros = rates * 0
        for step in range(spec["gradient_steps"] + 1):
            index, centers, quantized = lloyd(latent.detach(), centers, zeros, 1)
            weights = latent + (quantized - latent).detach()
            loss = (
                (torch.einsum("rni,roi->rno", local_x, weights) - local_y)
                .square()
                .mean((-1, -2))
            )
            if step % spec["checkpoint_every"] == 0 or step == spec["gradient_steps"]:
                selector.update(
                    step,
                    index,
                    loss.detach().reshape(count, anchors).mean(-1),
                    time.perf_counter() - started,
                )
            if step == spec["gradient_steps"]:
                break
            optimizer.step(torch.autograd.grad(loss.sum(), [latent]))
    else:
        raise ValueError(method)
    return {
        "indices": selector.indices.cpu().numpy(),
        "basis": index_basis(selector.indices, spec["codes"]).cpu().numpy(),
        "selection_mse": selector.best_selection.cpu().numpy(),
        "step": selector.steps.cpu().numpy(),
        "history": selector.history,
        "algorithm_history": histories,
        "seconds": time.perf_counter() - started,
        "proposal_anchors": anchors,
    }
