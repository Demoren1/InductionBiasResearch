"""Explicit multi-task competitors using exact regression sufficient statistics.

These are matched adaptations of objectives, not unchanged author scripts.
Categorical MAP, LC, and LUT-Q return their native assignments. Continuous MSR
and its additional signed-row readout are recorded as separate methods.
"""

import itertools
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.sharing_benchmark_v1.linear import Adam
from experiments.sharing_benchmark_v1.yeh_shared import configurations as yeh_rows

from .gradient_spectrum import decode


def configurations(method):
    if method == "yeh":
        return yeh_rows()
    if method == "msr_exact":
        return [{"lr": lr} for lr in (0.001, 0.003, 0.01, 0.03)]
    if method == "lc":
        return [
            {"rho_initial": first, "rho_final": last}
            for first, last in itertools.product((0.001, 0.01, 0.1), (1.0, 10.0, 100.0))
        ]
    if method == "lutq":
        return [
            {"lr": lr, "initial": initial}
            for lr, initial in itertools.product(
                (0.001, 0.003, 0.01, 0.03), ("ols", "small")
            )
        ]
    raise ValueError(method)


def signed_basis(indices, codes):
    return (
        F.one_hot(indices.abs().long(), codes + 1)[..., 1:].double()
        * indices.sign()[..., None]
    )


def adapted_stats(assigned, data, n, ridge=0.0):
    product = torch.einsum("tij,sojk->stoik", data["gram"] / n, assigned)
    hessian = torch.einsum("soik,stoil->stkl", assigned, product)
    cross = torch.einsum("soik,toi->stk", assigned, data["cross"] / n)
    if ridge:
        hessian = hessian + ridge * torch.eye(hessian.shape[-1], device=hessian.device)
        return torch.linalg.solve(hessian, cross[..., None])[..., 0]
    return (torch.linalg.pinv(hessian, hermitian=True, rtol=1e-10) @ cross[..., None])[
        ..., 0
    ]


def stats_loss(weights, data, n):
    value = (
        torch.einsum("stoi,tij,stoj->sto", weights, data["gram"] / n, weights)
        - 2 * (weights * (data["cross"] / n)[None]).sum(-1)
        + (data["square_y"] / n)[None]
    )
    return value.mean((1, 2))


@torch.no_grad()
def validation_loss(assigned, data):
    coefficient = adapted_stats(assigned, data["support"], data["support_n"])
    weights = torch.einsum("soik,stk->stoi", assigned, coefficient)
    return stats_loss(weights, data["query"], data["query_n"])


class NativeSelection:
    """Checkpoint selection uses observed validation tasks, never teacher I."""

    def __init__(self, count, codes, validation, device):
        self.codes, self.validation = codes, validation
        self.best = torch.full((count,), torch.inf, device=device, dtype=torch.float64)
        self.indices = None
        self.steps = torch.zeros(count, device=device, dtype=torch.long)
        self.history = []
        self.trace_indices = []

    @torch.no_grad()
    def update(self, step, indices, loss, elapsed):
        selection = validation_loss(signed_basis(indices, self.codes), self.validation)
        improved = selection < self.best
        if self.indices is None:
            self.indices = indices.clone()
        self.indices[improved] = indices[improved]
        self.steps[improved] = step
        self.best = torch.minimum(self.best, selection)
        self.trace_indices.append(indices.cpu().numpy().astype(np.int16))
        self.history.append(
            {
                "step": step,
                "source_mse": loss.tolist(),
                "validation_mse": selection.tolist(),
                "best_validation_mse": self.best.tolist(),
                "seconds": elapsed,
            }
        )

    def result(self):
        return {
            "indices": self.indices.cpu().numpy(),
            "selection_mse": self.best.cpu().numpy(),
            "selected_step": self.steps.cpu().numpy(),
            "trace_indices": np.stack(self.trace_indices),
            "history": self.history,
        }


def train_relaxed(method, rows, data, validation, plan, seed, steps):
    device, codes = data["gram"].device, plan["codes"]
    count, outputs, inputs = len(rows), plan["outputs"], plan["inputs"]
    generator = torch.Generator(device=device).manual_seed(plan["seed"] + 100003 * seed)
    if method == "yeh":
        state = 0.01 * torch.randn(
            (1, outputs, inputs, 2 * codes + 1),
            generator=generator,
            device=device,
            dtype=torch.float64,
        )
        state = state.repeat(count, 1, 1, 1)
        state[..., 0] += state.new_tensor(
            [np.log(2 * codes) if row["zero_bias"] else 0 for row in rows]
        )[:, None, None]
    else:
        state = 0.01 * torch.randn(
            (1, outputs, inputs, codes),
            generator=generator,
            device=device,
            dtype=torch.float64,
        )
        state = state.repeat(count, 1, 1, 1)
    state.requires_grad_()
    optimizer = Adam([state], state.new_tensor([row["lr"] for row in rows]))
    selector = NativeSelection(count, codes, validation, device)
    best_basis = state.new_full((count,), torch.inf)
    saved_basis = state.new_zeros((count, outputs, inputs, codes))
    basis_steps, raw_history = np.zeros(count, dtype=int), []
    started = time.perf_counter()
    for step in range(steps + 1):
        if method == "yeh":
            cooling = max(0.0, (step / steps - 0.3) / 0.7)
            temperature = state.new_tensor(
                [row["minimum_temperature"] for row in rows]
            ).pow(cooling)
            probability = (state / temperature[:, None, None, None]).softmax(-1)
            assigned = probability[..., 1 : codes + 1] - probability[..., codes + 1 :]
        else:
            assigned = state
        coefficient = adapted_stats(assigned, data, plan["n"], ridge=1e-4)
        weights = torch.einsum("soik,stk->stoi", assigned, coefficient)
        loss = stats_loss(weights, data, plan["n"])
        if step % plan["checkpoint_every"] == 0 or step == steps:
            if method == "yeh":
                labels = probability.detach().argmax(-1)
                indices = torch.where(labels > codes, codes - labels, labels)
            else:
                decoded = [
                    decode(local.flatten(0, 1).detach(), plan, plan["seed"] + seed)[0]
                    for local in assigned
                ]
                indices = torch.stack(decoded).reshape(count, outputs, inputs)
            selector.update(step, indices, loss.detach(), time.perf_counter() - started)
            with torch.no_grad():
                selection = validation_loss(assigned.detach(), validation)
                improved = selection < best_basis
                saved_basis[improved] = assigned[improved]
                basis_steps[improved.cpu().numpy()] = step
                best_basis = torch.minimum(best_basis, selection)
                raw_history.append(
                    {
                        "step": step,
                        "selection_mse": selection.tolist(),
                        "source_mse": loss.detach().tolist(),
                    }
                )
        if step == steps:
            break
        objective = loss
        if method == "yeh":
            entropy = (
                -(probability * probability.clamp_min(1e-12).log())
                .sum(-1)
                .mean((-1, -2))
            )
            entropy_weight = state.new_tensor([row["entropy"] for row in rows])
            nuclear_weight = state.new_tensor([row["nuclear"] for row in rows])
            nuclear = torch.zeros_like(loss)
            selected = nuclear_weight > 0
            if selected.any():
                nuclear[selected] = (
                    nuclear_weight[selected]
                    * torch.linalg.svdvals(probability[selected].flatten(1, 2)).sum(-1)
                    / np.sqrt(outputs * inputs)
                )
            objective = loss + entropy_weight * entropy + nuclear
        optimizer.step(torch.autograd.grad(objective.sum(), [state]))
        if method == "yeh":
            with torch.no_grad():
                state.sub_(state.mean(-1, keepdim=True))
    return {
        "native_or_decoded": selector.result(),
        "continuous": {
            "basis": saved_basis.detach().cpu().numpy(),
            "selection_mse": best_basis.cpu().numpy(),
            "selected_step": basis_steps,
            "history": raw_history,
        },
        "seconds": time.perf_counter() - started,
    }


@torch.no_grad()
def vector_start(values, codes, generator):
    centers = values.new_zeros((len(values), codes, values.shape[-1]))
    distance = values.square().sum(-1)
    for code in range(codes):
        position = torch.multinomial(distance + 1e-30, 1, generator=generator)[:, 0]
        centers[:, code] = values[
            torch.arange(len(values), device=values.device), position
        ]
        positive = (values - centers[:, code, None]).square().sum(-1)
        negative = (values + centers[:, code, None]).square().sum(-1)
        distance = torch.minimum(distance, torch.minimum(positive, negative))
    return centers


@torch.no_grad()
def vector_lloyd(values, centers, iterations):
    codes = centers.shape[1]
    for _ in range(iterations):
        product = values @ centers.transpose(-1, -2)
        norm = centers.square().sum(-1)[:, None]
        costs = torch.cat(
            (product[..., :1] * 0, norm - 2 * product, norm + 2 * product), -1
        )
        labels = costs.argmin(-1)
        indices = torch.where(labels > codes, codes - labels, labels)
        assigned = signed_basis(indices, codes)
        counts = assigned.abs().sum(1)
        means = (assigned.transpose(-1, -2) @ values) / counts.clamp_min(1)[..., None]
        centers = torch.where((counts > 0)[..., None], means, centers)
    return indices, centers, assigned @ centers


def train_quantized(method, rows, data, validation, plan, seed, steps):
    device, codes = data["gram"].device, plan["codes"]
    count, tasks = len(rows), data["gram"].shape[0]
    outputs, inputs = plan["outputs"], plan["inputs"]
    gram, cross = data["gram"] / plan["n"], data["cross"] / plan["n"]
    initial = torch.linalg.solve(gram, cross.transpose(-1, -2)).transpose(-1, -2)
    latent = initial[None].repeat(count, 1, 1, 1)
    generator = torch.Generator(device=device).manual_seed(plan["seed"] + 100003 * seed)
    if method == "lutq":
        for number, row in enumerate(rows):
            if row["initial"] == "small":
                latent[number] = 0.01 * torch.randn(
                    initial.shape,
                    device=device,
                    generator=generator,
                    dtype=latent.dtype,
                )
        latent.requires_grad_()
        optimizer = Adam([latent], latent.new_tensor([row["lr"] for row in rows]))
    centers = vector_start(latent.detach().flatten(2).transpose(1, 2), codes, generator)
    selector = NativeSelection(count, codes, validation, device)
    dual, quantized = torch.zeros_like(latent), torch.zeros_like(latent)
    old_rho = latent.new_full((count,), 0.01)
    started = time.perf_counter()
    for step in range(steps + 1):
        if method == "lc":
            with torch.no_grad():
                first = latent.new_tensor([row["rho_initial"] for row in rows])
                last = latent.new_tensor([row["rho_final"] for row in rows])
                rho = first * (last / first).pow(step / steps)
                dual *= (old_rho / rho)[:, None, None, None]
                if step:
                    right = cross[None] + rho[:, None, None, None] * (quantized - dual)
                    matrix = gram[None] + rho[:, None, None, None] * torch.eye(
                        inputs, device=device
                    )
                    latent = torch.linalg.solve(
                        matrix, right.transpose(-1, -2)
                    ).transpose(-1, -2)
                index, centers, local = vector_lloyd(
                    (latent + dual).flatten(2).transpose(1, 2), centers, 30
                )
                quantized = local.transpose(1, 2).reshape(count, tasks, outputs, inputs)
                dual += latent - quantized
                old_rho = rho
            loss = stats_loss(quantized, data, plan["n"])
        else:
            index, centers, local = vector_lloyd(
                latent.detach().flatten(2).transpose(1, 2), centers, 1
            )
            quantized = local.transpose(1, 2).reshape(count, tasks, outputs, inputs)
            weights = latent + (quantized - latent).detach()
            loss = stats_loss(weights, data, plan["n"])
        if step % plan["checkpoint_every"] == 0 or step == steps:
            selector.update(
                step,
                index.reshape(count, outputs, inputs),
                loss.detach(),
                time.perf_counter() - started,
            )
        if step == steps:
            break
        if method == "lutq":
            optimizer.step(torch.autograd.grad(loss.sum(), [latent]))
    return {
        "native_or_decoded": selector.result(),
        "seconds": time.perf_counter() - started,
    }
