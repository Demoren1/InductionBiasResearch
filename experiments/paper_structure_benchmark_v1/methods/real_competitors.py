"""Native LUT-Q, categorical implicit learning and MSR-FC on real source tasks."""

import argparse
import itertools
import json
import math
import resource
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.flop_counter import FlopCounterMode

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from experiments.sharing_structure_research_v1.shared_structure_bilevel_optimization import (
    ScenarioAdam,
)
from ..fitting.adaptation import losses
from .bilevel import basis_for, predict
from ..protocol.real_boundary import StudyBoundary
from ..fitting.real_dense import predict as dense_predict


@torch.no_grad()
def quantize(weight, centers):
    dictionary = torch.cat((centers[:, :1] * 0, centers, -centers), -1)
    ordered, names = dictionary.sort(dim=-1, stable=True)
    position = torch.searchsorted(ordered, weight.contiguous())
    low = (position - 1).clamp(0, ordered.shape[1] - 1)
    high = position.clamp(0, ordered.shape[1] - 1)
    low_value, high_value = ordered.gather(1, low), ordered.gather(1, high)
    # The first equal entry preserves argmin's original label tie convention.
    low = torch.searchsorted(ordered, low_value.contiguous())
    low_name, high_name = names.gather(1, low), names.gather(1, high)
    left, right = (weight - low_value).abs(), (weight - high_value).abs()
    choose_low = (left < right) | ((left == right) & (low_name < high_name))
    label = torch.where(choose_low, low_name, high_name)
    codes = centers.shape[1]
    return torch.where(label > codes, codes - label, label), dictionary.gather(1, label)


@torch.no_grad()
def update_centers(weight, centers):
    index, _ = quantize(weight, centers)
    counts = weight.new_zeros((len(weight), centers.shape[1] + 1))
    total = torch.zeros_like(counts)
    counts.scatter_add_(1, index.abs(), torch.ones_like(weight))
    total.scatter_add_(1, index.abs(), weight * index.sign())
    return torch.where(
        counts[:, 1:] > 0, total[:, 1:] / counts[:, 1:].clamp_min(1), centers
    )


def objective(x, y, basis, parameter, hidden, outputs, kind):
    return losses(predict(x, basis, parameter, hidden, outputs), y, kind)


def adapt(x, y, basis, initial, rates, h, o, kind, steps, higher):
    tasks = len(x)
    parameter = initial[:, None].expand(-1, tasks, -1).clone().requires_grad_()
    k = basis.shape[-1]
    rate = torch.cat(
        (
            rates[:, :1].expand(-1, k),
            rates[:, 1:2].expand(-1, h),
            rates[:, 2:3].expand(-1, h * o),
            rates[:, 3:4].expand(-1, o),
        ),
        -1,
    )[:, None]
    for _ in range(steps):
        gradient = torch.autograd.grad(
            objective(x, y, basis, parameter, h, o, kind).sum(),
            parameter,
            create_graph=higher,
        )[0]
        parameter = parameter - rate * gradient
        if not higher:
            parameter = parameter.detach().requires_grad_()
    return parameter


def initial_parameters(dense, codes, d, h, o, device, count):
    generator = torch.Generator(device=device).manual_seed(767001 + codes)
    tasks = len(dense)
    parameter = torch.randn(
        (count, tasks, codes + h + h * o + o), device=device, generator=generator
    )
    parameter[..., :codes] *= math.sqrt(2 / d)
    parameter[..., codes : codes + h] *= 0.1
    parameter[..., -o:] = 0
    weight = torch.tensor(dense[:, : d * h], device=device)
    centers = torch.quantile(
        weight[:1].abs(), torch.linspace(0.1, 0.95, codes, device=device), dim=1
    ).T
    for _ in range(10):
        centers = update_centers(weight[:1], centers)
    index, _ = quantize(weight[:1], centers)
    index = index.reshape(d, h)
    signed = (
        F.one_hot(index.abs(), codes + 1)[..., 1:].float() * index.sign()[..., None]
    )
    dictionary = torch.cat((centers[:, :1] * 0, centers, -centers), -1)
    logits = (
        -2
        * (
            (weight[0].reshape(d, h, 1) - dictionary) / weight[0].std().clamp_min(0.01)
        ).square()
    ).clamp_min(-20)
    design = signed.flatten(0, 1)
    counts = design.square().sum(0).clamp_min(1)
    code = weight @ design / counts
    tail = torch.tensor(dense[:, d * h :], device=device)
    warm = torch.cat((code, tail), -1)
    return parameter, signed, logits, warm


def regularizer(state, rows):
    probability = state.softmax(-1)
    entropy = -(probability * probability.clamp_min(1e-20).log()).sum(-1).mean((1, 2))
    matrix = probability.flatten(1, 2)
    gram = matrix.transpose(1, 2) @ matrix / matrix.shape[1]
    nuclear = torch.linalg.eigvalsh(gram).clamp_min(1e-12).sqrt().sum(-1)
    weights = state.new_tensor([[row["entropy"], row["nuclear"]] for row in rows])
    return (weights[:, 0] * entropy + weights[:, 1] * nuclear).sum()


def run(root, method, codes, device, steps):
    torch.set_num_threads(2)
    boundary = StudyBoundary(root)
    boundary.install()
    folder = root / "source" / f"{method}_k{codes}"
    folder.mkdir(exist_ok=False)
    protocol = json.loads((root / "protocol.json").read_text())
    d, h, o = [protocol[key] for key in ("inputs", "hidden", "outputs")]
    kind = protocol["loss"]
    rows = [
        {
            "lr": lr,
            "initialization": initial,
            "entropy": extra[0] if method == "yeh" else 0.0,
            "nuclear": extra[1] if method == "yeh" else 0.0,
            "inner_lr": extra if method == "msr" else 0.01,
        }
        for lr, initial, extra in itertools.product(
            (0.0003, 0.001, 0.003) if method == "msr" else (0.003, 0.01, 0.03),
            ("cold", "dense_adam"),
            ((0.0, 0.0), (0.025, 0.05))
            if method == "yeh"
            else (0.1, 0.4)
            if method == "msr"
            else (0, 1),
        )
    ]
    save_json(
        folder / "registered.json",
        {
            "method": method,
            "codes": codes,
            "settings": rows,
            "steps": steps,
            "code_sha256": digest(Path(__file__)),
            "native_algorithms": {
                "lutq": "Nearest-code forward, latent STE update, one Lloyd center update per minibatch; task-independent source models",
                "yeh": "Categorical A*phi, signed/fixed-zero adaptation; true implicit support/query gradient with Neumann20; 250 initial and20 subsequent inner Adam updates",
                "msr": "Continuous U*phi; full higher-order MAML, inner1/train3/checkpoint; learned initialization and unconstrained per-group SGD rates",
            },
            "minibatch": "Up to32 source tasks and up to512 support examples; source H0 only for outer loss/checkpoints",
            "transferred": "Structure only in primary comparison, no post-clustering of any competitor; native MSR initialization saved for separate auxiliary evaluation",
            "published_comparison": "Adapted same FC architecture and data/loss as ours; not literal published CNN/init-transfer results",
        },
    )
    shutil.copy2(__file__, folder / "executed_code.py")
    support, numerical = [
        dict(np.load(root / "public" / f"{name}.npz"))
        for name in ("support", "numerical")
    ]
    x, y, vx, vy = [
        torch.tensor(panel[key], device=device, dtype=torch.float32)
        for panel, key in (
            (support, "x"),
            (support, "y"),
            (numerical, "x"),
            (numerical, "y"),
        )
    ]
    x, vx = x / 255, vx / 255
    center, scale = y.mean(1), y.std((1, 2), correction=0).clamp_min(1e-8)
    if kind == "cross_entropy":
        center, scale = torch.zeros_like(center), torch.ones_like(scale)
    y, vy = (
        (y - center[:, None]) / scale[:, None, None],
        (vy - center[:, None]) / scale[:, None, None],
    )
    dense = np.load(root / "source" / "dense" / "models.npz")["parameters"].copy()
    dense[:, d * h + h : -o] /= scale.cpu().numpy()[:, None]
    dense[:, -o:] = (dense[:, -o:] - center.cpu().numpy()) / scale.cpu().numpy()[
        :, None
    ]
    numeric, signed, logits, warm = initial_parameters(
        dense, codes, d, h, o, device, len(rows)
    )
    generator = torch.Generator(device=device).manual_seed(767003 + codes)
    if method == "lutq":
        state = torch.randn(
            (len(rows), len(x), d * h + h + h * o + o),
            device=device,
            generator=generator,
        )
        state[..., : d * h] *= math.sqrt(2 / d)
        state[..., d * h : d * h + h] *= 0.1
        state[..., -o:] = 0
        for row, settings in enumerate(rows):
            if settings["initialization"] == "dense_adam":
                state[row] = torch.tensor(dense, device=device)
        centers = torch.quantile(
            state[..., : d * h].flatten(0, 1).abs(),
            torch.linspace(0.1, 0.95, codes, device=device),
            dim=1,
        ).T
    else:
        shape = (len(rows), d, h, codes if method == "msr" else 2 * codes + 1)
        state = torch.randn(shape, generator=generator, device=device) * (
            1 / math.sqrt(codes) if method == "msr" else 0.01
        )
        for row, settings in enumerate(rows):
            if settings["initialization"] == "dense_adam":
                state[row] = signed if method == "msr" else logits
                numeric[row] = warm
    state.requires_grad_()
    numeric.requires_grad_()
    optimizer = ScenarioAdam(state, [row["lr"] for row in rows])
    inner_optimizer = ScenarioAdam(numeric, [0.01] * len(rows))
    initial = numeric[:, 0].detach().clone().requires_grad_()
    rates = (
        state.new_tensor([row["inner_lr"] for row in rows])[:, None]
        .repeat(1, 4)
        .requires_grad_()
    )
    meta_optimizers = [
        ScenarioAdam(value, [row["lr"] for row in rows]) for value in (initial, rates)
    ]
    best = state.new_full(
        (len(rows), len(x)) if method == "lutq" else (len(rows),), torch.inf
    )
    saved_state = state.detach().clone()
    saved_numeric = numeric.detach().clone()
    saved_initial, saved_rates = initial.detach().clone(), rates.detach().clone()
    saved_centers = centers.clone() if method == "lutq" else None
    saved_indices = (
        torch.zeros((len(rows), len(x), d, h), dtype=torch.int16, device=device)
        if method == "lutq"
        else torch.zeros((len(rows), d, h), dtype=torch.int16, device=device)
    )
    history = {"steps": [], "train": [], "validation": []}
    diagnostics = []
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    counter = FlopCounterMode(display=False)
    status = "failed"
    try:
        with counter:
            for step in range(steps + 1):
                if method == "yeh":
                    fixed = basis_for(state, "yeh", codes).detach()
                    for _ in range(250 if step == 0 else 20):
                        tasks = torch.randperm(
                            len(x), device=device, generator=generator
                        )[:32]
                        positions = torch.randint(
                            x.shape[1],
                            (min(512, x.shape[1]),),
                            device=device,
                            generator=generator,
                        )
                        lower = (
                            objective(
                                x[tasks][:, positions],
                                y[tasks][:, positions],
                                fixed,
                                numeric[:, tasks],
                                h,
                                o,
                                kind,
                            ).sum()
                            + 0.5e-4 * numeric[:, tasks].square().sum()
                        )
                        inner_optimizer.step(
                            numeric, torch.autograd.grad(lower, numeric)[0]
                        )
                if step % 50 == 0 or step == steps:
                    if method == "msr":
                        fitted = adapt(
                            x,
                            y,
                            state.detach(),
                            initial.detach(),
                            rates.detach(),
                            h,
                            o,
                            kind,
                            3,
                            False,
                        )
                    else:
                        fitted = numeric
                    with torch.no_grad():
                        if method == "lutq":
                            index, tied = quantize(
                                state[..., : d * h].flatten(0, 1), centers
                            )
                            effective = torch.cat(
                                (
                                    tied.reshape(len(rows), len(x), -1),
                                    state[..., d * h :],
                                ),
                                -1,
                            )
                            train = losses(
                                dense_predict(x, effective, d, h, o), y, kind
                            )
                            validation = losses(
                                dense_predict(vx, effective, d, h, o), vy, kind
                            )
                            improved = validation < best
                            best[improved], saved_state[improved] = (
                                validation[improved],
                                state[improved],
                            )
                            saved_indices[improved] = index.reshape(
                                len(rows), len(x), d, h
                            )[improved].to(torch.int16)
                            saved_centers.reshape(len(rows), len(x), codes)[
                                improved
                            ] = centers.reshape(len(rows), len(x), codes)[improved]
                        else:
                            basis = basis_for(
                                state, "yeh" if method == "yeh" else "msr", codes
                            )
                            train = objective(x, y, basis, fitted, h, o, kind)
                            validation = objective(vx, vy, basis, fitted, h, o, kind)
                            score = validation.mean(1)
                            improved = score < best
                            (
                                best[improved],
                                saved_state[improved],
                                saved_numeric[improved],
                            ) = score[improved], state[improved], fitted[improved]
                            saved_initial[improved], saved_rates[improved] = (
                                initial[improved],
                                rates[improved],
                            )
                            if method == "yeh":
                                label = state.argmax(-1)
                                saved_indices[improved] = torch.where(
                                    label > codes, codes - label, label
                                )[improved].to(torch.int16)
                        history["steps"].append(step)
                        history["train"].append(train.cpu().numpy())
                        history["validation"].append(validation.cpu().numpy())
                        print(
                            json.dumps(
                                {
                                    "method": method,
                                    "codes": codes,
                                    "step": step,
                                    "best_H0_loss": float(best.min()),
                                    "seconds": time.perf_counter() - started,
                                }
                            ),
                            flush=True,
                        )
                if step == steps:
                    break
                tasks = torch.randperm(len(x), device=device, generator=generator)[:32]
                positions = torch.randint(
                    x.shape[1],
                    (min(512, x.shape[1]),),
                    device=device,
                    generator=generator,
                )
                bx, by, qx, qy = (
                    x[tasks][:, positions],
                    y[tasks][:, positions],
                    vx[tasks],
                    vy[tasks],
                )
                if method == "lutq":
                    _, tied = quantize(
                        state[..., : d * h].detach().flatten(0, 1), centers
                    )
                    weight = (
                        state[..., : d * h]
                        + (
                            tied.reshape(len(rows), len(x), -1) - state[..., : d * h]
                        ).detach()
                    )
                    effective = torch.cat(
                        (weight[:, tasks], state[:, tasks, d * h :]), -1
                    )
                    lower = losses(
                        dense_predict(bx, effective, d, h, o), by, kind
                    ).sum()
                    optimizer.step(state, torch.autograd.grad(lower, state)[0])
                    centers = update_centers(
                        state[..., : d * h].detach().flatten(0, 1), centers
                    )
                elif method == "msr":
                    fitted = adapt(bx, by, state, initial, rates, h, o, kind, 1, True)
                    upper = objective(qx, qy, state, fitted, h, o, kind).mean(1).sum()
                    gradients = torch.autograd.grad(upper, (state, initial, rates))
                    for variable, opt, gradient in zip(
                        (state, initial, rates),
                        (optimizer, *meta_optimizers),
                        gradients,
                    ):
                        opt.step(variable, gradient)
                else:
                    basis = basis_for(state, "yeh", codes)
                    lower = (
                        objective(bx, by, basis, numeric[:, tasks], h, o, kind).sum()
                        + 0.5e-4 * numeric[:, tasks].square().sum()
                    )
                    train_gradient = torch.autograd.grad(
                        lower, numeric, create_graph=True
                    )[0]
                    upper = (
                        objective(qx, qy, basis, numeric[:, tasks], h, o, kind)
                        .mean(1)
                        .sum()
                    )
                    direct, vector = torch.autograd.grad(
                        upper, (state, numeric), retain_graph=True
                    )
                    vector = vector.detach()
                    product = torch.autograd.grad(
                        train_gradient, numeric, grad_outputs=vector, retain_graph=True
                    )[0]
                    radius = (
                        product.norm(dim=-1) / vector.norm(dim=-1).clamp_min(1e-12)
                    ).amax(1)
                    eta = (0.5 / radius.clamp_min(1)).clamp_max(0.1)[:, None, None]
                    current, total = vector.clone(), vector.clone()
                    for _ in range(19):
                        product = torch.autograd.grad(
                            train_gradient,
                            numeric,
                            grad_outputs=current,
                            retain_graph=True,
                        )[0]
                        current = (current - eta * product).detach()
                        total += current
                    inverse = (eta * total).detach()
                    if step % 50 == 0:
                        product = torch.autograd.grad(
                            train_gradient,
                            numeric,
                            grad_outputs=inverse,
                            retain_graph=True,
                        )[0]
                        diagnostics.append(
                            {
                                "step": step,
                                "inverse_residual": (
                                    (product - vector).norm(dim=-1)
                                    / vector.norm(dim=-1).clamp_min(1e-12)
                                )
                                .amax(1)
                                .cpu()
                                .tolist(),
                            }
                        )
                    mixed = torch.autograd.grad(
                        train_gradient, state, grad_outputs=inverse
                    )[0]
                    extra = torch.autograd.grad(regularizer(state, rows), state)[0]
                    gradient = direct - mixed + extra
                    magnitude = gradient.flatten(1).norm(dim=1).clamp_min(1e-12)
                    optimizer.step(
                        state,
                        gradient * (25 / magnitude).clamp_max(1)[:, None, None, None],
                    )
                    with torch.no_grad():
                        state -= state.mean(-1, keepdim=True)
        torch.cuda.synchronize(device)
        arrays = {
            "steps": np.asarray(history["steps"]),
            "train_history": np.stack(history["train"]),
            "validation_history": np.stack(history["validation"]),
            "indices": saved_indices.cpu().numpy(),
            "H0_loss": best.cpu().numpy(),
            "normalization_center": center.cpu().numpy(),
            "normalization_scale": scale.cpu().numpy(),
        }
        if method == "lutq":
            arrays.update(
                latent_parameters=saved_state.cpu().numpy(),
                codes=saved_centers.reshape(len(rows), len(x), codes).cpu().numpy(),
            )
        else:
            arrays.update(
                basis=basis_for(saved_state, "yeh" if method == "yeh" else "msr", codes)
                .cpu()
                .numpy(),
                source_parameters=saved_numeric.cpu().numpy(),
                native_initialization=saved_initial.cpu().numpy(),
                native_rates=saved_rates.cpu().numpy(),
            )
        np.savez_compressed(folder / "native.npz", **arrays)
        save_json(folder / "diagnostics.json", {"implicit": diagnostics})
        status = "complete"
    finally:
        torch.cuda.synchronize(device)
        save_json(
            folder / "metrics.json",
            {
                "status": status,
                "wall_seconds": time.perf_counter() - started,
                "torch_operator_flops": counter.get_total_flops(),
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
            },
        )
        boundary.save(folder / "access_receipt.json", status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--method", choices=("lutq", "yeh", "msr"), required=True)
    parser.add_argument("--codes", type=int, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    steps = (
        args.steps
        if args.steps is not None
        else 400
        if args.method == "yeh"
        else 2000
        if args.method == "msr"
        else 2500
    )
    run(args.root.resolve(), args.method, args.codes, torch.device(args.device), steps)


if __name__ == "__main__":
    main()
