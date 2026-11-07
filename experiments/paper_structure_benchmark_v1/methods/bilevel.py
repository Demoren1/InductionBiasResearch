"""Matched A-phi and continuous MSR with genuine support/query hypergradients."""

import argparse
import itertools
import json
import math
import resource
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F
from torch.utils.flop_counter import FlopCounterMode

from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    load_sizes,
    save_json,
)
from experiments.sharing_structure_research_v1.blind_runtime import ObservationBoundary
from experiments.sharing_structure_research_v1.shared_structure_bilevel_optimization import (
    ScenarioAdam,
)

from .lutq import quantize, update_centers


def basis_for(state, method, codes):
    if method == "msr":
        return state
    probability = state.softmax(-1)
    return probability[..., 1 : codes + 1] - probability[..., codes + 1 :]


def predict(x, basis, parameter, hidden, outputs):
    codes = basis.shape[-1]
    weight = torch.einsum("rdhk,rtk->rtdh", basis, parameter[..., :codes])
    bias = parameter[..., codes : codes + hidden]
    head = parameter[..., codes + hidden : -outputs].reshape(
        parameter.shape[0], parameter.shape[1], hidden, outputs
    )
    activation = (torch.einsum("tnd,rtdh->rtnh", x, weight) + bias[:, :, None]).relu()
    return (
        torch.einsum("rtnh,rtho->rtno", activation, head) / math.sqrt(hidden)
        + parameter[:, :, None, -outputs:]
    )


def losses(x, y, basis, parameter, hidden, outputs):
    return (
        (predict(x, basis, parameter, hidden, outputs) - y[None]).square().mean((2, 3))
    )


def initialize(root, case, condition, rows, sizes, center, scale, device):
    d, h, k, o = [sizes[name] for name in ("inputs", "hidden", "codes", "outputs")]
    tasks, count = len(scale), len(rows)
    generator = torch.Generator(device=device).manual_seed(748901)
    channels = k if rows[0]["method"] == "msr" else 2 * k + 1
    state = 0.01 * torch.randn(
        (count, d, h, channels), device=device, generator=generator
    )
    length = k + h + h * o + o
    numeric = torch.randn((count, tasks, length), device=device, generator=generator)
    numeric[..., :k] *= 0.5
    numeric[..., k : k + h] *= 0.2
    numeric[..., -o:] = 0
    dense = np.load(root / "dense" / f"{case}_c{condition}" / "fitted.npz")[
        "parameters"
    ].copy()
    dense[:, d * h + h : -o] /= scale[:, None]
    dense[:, -o:] = (dense[:, -o:] - center) / scale[:, None]
    reference = torch.tensor(
        dense[0, : d * h][None], device=device, dtype=torch.float32
    )
    centers = torch.quantile(
        reference.abs(), torch.linspace(0.1, 0.95, k, device=device), dim=1
    ).T
    for _ in range(10):
        centers = update_centers(reference, centers)
    index, _ = quantize(reference, centers)
    index = index.reshape(d, h)
    signed = F.one_hot(index.abs(), k + 1)[..., 1:].float() * index.sign()[..., None]
    dictionary = torch.cat((centers[:, :1] * 0, centers, -centers), 1)
    width = reference.std().clamp_min(0.05)
    warm_logits = -2 * ((reference.reshape(d, h, 1) - dictionary) / width).square()
    warm_logits = warm_logits.clamp_min(-20)
    prototype = dense[0, : d * h].reshape(d, h).copy()
    prototype /= np.maximum(np.linalg.norm(prototype, axis=0), 1e-12)
    warm_numeric = []
    for task in range(tasks):
        matrix = dense[task, : d * h].reshape(d, h)
        normal = matrix / np.maximum(np.linalg.norm(matrix, axis=0), 1e-12)
        cost = ((prototype[:, :, None] - normal[:, None]) ** 2).sum(0)
        left, right = linear_sum_assignment(cost)
        mapping = right[np.argsort(left)]
        matrix = matrix[:, mapping]
        code = np.linalg.lstsq(
            signed.cpu().numpy().reshape(d * h, k), matrix.ravel(), rcond=1e-10
        )[0]
        bias = dense[task, d * h : d * h + h][mapping]
        head = dense[task, d * h + h : -o].reshape(h, o)[mapping]
        warm_numeric.append(np.r_[code, bias, head.ravel(), dense[task, -o:]])
    warm_numeric = torch.tensor(
        np.stack(warm_numeric), device=device, dtype=torch.float32
    )
    for scenario, row in enumerate(rows):
        if row["initialization"] == "dense_adam":
            state[scenario] = signed if row["method"] == "msr" else warm_logits
            numeric[scenario] = warm_numeric
    return state.requires_grad_(), numeric.requires_grad_()


def regularization_gradient(state, rows, codes):
    if rows[0]["method"] == "msr":
        return torch.zeros_like(state)
    leaf = state.detach().requires_grad_()
    probability = leaf.softmax(-1)
    entropy = -(probability * probability.clamp_min(1e-20).log()).sum(-1).mean((1, 2))
    nuclear = torch.linalg.svdvals(probability.flatten(1, 2)).sum(-1) / math.sqrt(
        state.shape[1] * state.shape[2]
    )
    weights = state.new_tensor([[row["entropy"], row["nuclear"]] for row in rows])
    return torch.autograd.grad(
        (weights[:, 0] * entropy + weights[:, 1] * nuclear).sum(), leaf
    )[0]


def run(
    root,
    case,
    condition,
    method,
    outer_steps,
    device,
    *,
    output_method=None,
    following_inner_steps=20,
    direct_hessian=False,
):
    all_started = time.perf_counter()
    torch.set_num_threads(2)
    folder = root / "competitors" / f"{case}_c{condition}" / (output_method or method)
    folder.mkdir(parents=True, exist_ok=False)
    rows = [
        {
            "method": method,
            "lr": lr,
            "initialization": initial,
            "entropy": regularizer[0],
            "nuclear": regularizer[1],
        }
        for lr, initial, regularizer in itertools.product(
            (0.003, 0.01, 0.03),
            ("cold", "dense_adam"),
            ((0, 0), (0.025, 0.05)) if method == "yeh" else ((0, 0), (0, 0)),
        )
    ]
    registration = {
        "method": "Yeh categorical A-phi"
        if method == "yeh"
        else "MSR continuous U-phi",
        "sources": [
            "https://proceedings.mlr.press/v151/yeh22b/yeh22b.pdf",
            "https://arxiv.org/pdf/2007.02933",
        ],
        "representation": "Native categorical probabilities and native MAP"
        if method == "yeh"
        else "Unconstrained continuous basis; never discretized or post-clustered",
        "adaptation": "Shared structure across source functions, free task-specific hidden biases/readout/offset, fixed-zero signed lookup for Yeh. Common support/query implicit optimization for both representations, not unchanged author scripts or the original MSR learned-initialization benchmark.",
        "hypergradient": "True validation direct gradient minus mixed train derivative times an explicitly constructed, spectrally damped block-Hessian inverse action; damping and residuals recorded."
        if direct_hessian
        else "True validation direct gradient minus mixed train derivative times a scaled 20-term Neumann inverse-Hessian approximation. Local inverse residual and inner stationarity are recorded.",
        "settings": rows,
        "outer_steps": outer_steps,
        "initial_inner_steps": 250,
        "following_inner_steps": following_inner_steps,
        "inner_lr": 0.01,
        "inner_parameter_ridge": 1e-4,
        "neumann_terms": 20,
        "source_batch": 512,
        "query_batch": 256,
        "checkpoint_every": 20,
        "code_sha256": digest(Path(__file__)),
        "registered_utc": datetime.now(UTC).isoformat(),
    }
    save_json(folder / "registered.json", registration)
    shutil.copy2(__file__, folder / "executed_code.py")
    boundary = ObservationBoundary(root, case, condition)
    boundary.phase = method
    boundary.install()
    sizes = load_sizes(root, case)
    d, h, k, o = [sizes[name] for name in ("inputs", "hidden", "codes", "outputs")]
    fixed = dict(np.load(root / "public" / f"{case}_c{condition}" / "fixed_train.npz"))
    selection = dict(
        np.load(root / "public" / f"{case}_c{condition}" / "selection.npz")
    )
    tasks = len(fixed) // 2
    x, y, vx, vy = [
        np.stack([panel[f"{name}_{task}"] for task in range(tasks)])
        for panel, name in (
            (fixed, "x"),
            (fixed, "y"),
            (selection, "x"),
            (selection, "y"),
        )
    ]
    center, scale = y.mean(1), np.maximum(y.std((1, 2)), 1e-8)
    tx, ty, tvx, tvy = [
        torch.tensor(a, device=device, dtype=torch.float32)
        for a in (
            x,
            (y - center[:, None]) / scale[:, None, None],
            vx,
            (vy - center[:, None]) / scale[:, None, None],
        )
    ]
    state, numeric = initialize(
        root, case, condition, rows, sizes, center, scale, device
    )
    outer_optimizer = ScenarioAdam(state, [row["lr"] for row in rows])
    inner_optimizer = ScenarioAdam(numeric, [0.01] * len(rows))
    generator = torch.Generator(device=device).manual_seed(748903)
    best = state.new_full((len(rows),), torch.inf)
    selected_basis = torch.zeros((len(rows), d, h, k), device=device)
    selected_numeric = numeric.detach().clone()
    history = {"steps": [], "train": [], "validation": [], "seconds": []}
    candidates, candidate_scores = [], []
    inverse_diagnostics = []
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    counter = FlopCounterMode(display=False)
    started = time.perf_counter()
    status = "failed"
    try:
        with counter:
            for step in range(outer_steps + 1):
                fixed_basis = basis_for(state, method, k).detach()
                for _ in range(250 if step == 0 else following_inner_steps):
                    positions = torch.randint(
                        tx.shape[1], (512,), device=device, generator=generator
                    )
                    lower = losses(
                        tx[:, positions], ty[:, positions], fixed_basis, numeric, h, o
                    )
                    lower = lower + 0.5e-4 * numeric.square().sum(-1)
                    gradient = torch.autograd.grad(lower.sum(), numeric)[0]
                    inner_optimizer.step(numeric, gradient)
                if step % 20 == 0 or step == outer_steps:
                    with torch.no_grad():
                        basis = basis_for(state, method, k)
                        train = losses(tx, ty, basis, numeric, h, o)
                        validation_loss = losses(tvx, tvy, basis, numeric, h, o)
                        score = validation_loss.mean(1)
                        improved = score < best
                        best[improved] = score[improved]
                        selected_basis[improved] = basis[improved]
                        selected_numeric[improved] = numeric[improved]
                        history["steps"].append(step)
                        history["train"].append(train.cpu().numpy())
                        history["validation"].append(validation_loss.cpu().numpy())
                        history["seconds"].append(time.perf_counter() - started)
                        if method == "yeh":
                            label = state.argmax(-1)
                            index = torch.where(label > k, k - label, label)
                            hard = (
                                F.one_hot(index.abs(), k + 1)[..., 1:].float()
                                * index.sign()[..., None]
                            )
                            hard_score = losses(tvx, tvy, hard, numeric, h, o).mean(1)
                            candidates.append(index.cpu().numpy().astype(np.int16))
                            candidate_scores.append(hard_score.cpu().numpy())
                        print(
                            json.dumps(
                                {
                                    "method": method,
                                    "case": case,
                                    "outer_step": step,
                                    "best_source_architecture_mse": float(best.min()),
                                    "seconds": time.perf_counter() - started,
                                }
                            ),
                            flush=True,
                        )
                if step == outer_steps:
                    break
                positions = torch.randint(
                    tx.shape[1], (512,), device=device, generator=generator
                )
                query_positions = torch.randint(
                    tvx.shape[1], (256,), device=device, generator=generator
                )
                basis = basis_for(state, method, k)
                lower = losses(tx[:, positions], ty[:, positions], basis, numeric, h, o)
                lower = lower + 0.5e-4 * numeric.square().sum(-1)
                train_gradient = torch.autograd.grad(
                    lower.sum(), numeric, create_graph=True
                )[0]
                query = (
                    losses(
                        tvx[:, query_positions],
                        tvy[:, query_positions],
                        basis,
                        numeric,
                        h,
                        o,
                    )
                    .mean(1)
                    .sum()
                )
                direct, vector = torch.autograd.grad(
                    query, (state, numeric), retain_graph=True
                )
                vector = vector.detach()
                damping_record = None
                if direct_hessian:
                    length = numeric.shape[-1]
                    identity = torch.eye(length, device=device)
                    hessian_rows = identity[:, None, None].expand(-1, *numeric.shape)
                    hessian = (
                        torch.autograd.grad(
                            train_gradient,
                            numeric,
                            grad_outputs=hessian_rows,
                            is_grads_batched=True,
                            retain_graph=True,
                        )[0]
                        .movedim(0, -1)
                        .detach()
                        .double()
                    )
                    hessian = (hessian + hessian.transpose(-1, -2)) * 0.5
                    eigen = torch.linalg.eigvalsh(hessian)
                    damping = (
                        1e-3 * eigen.abs().amax(-1).clamp_min(1) - eigen[..., 0]
                    ).clamp_min(0)
                    system = hessian + damping[..., None, None] * identity.double()
                    inverse = (
                        torch.linalg.solve(system, vector.double()[..., None])[..., 0]
                        .to(vector.dtype)
                        .detach()
                    )
                    damping_record = damping.amax(1).cpu().tolist()
                else:
                    h_vector = torch.autograd.grad(
                        train_gradient, numeric, grad_outputs=vector, retain_graph=True
                    )[0]
                    radius = (
                        h_vector.norm(dim=-1) / vector.norm(dim=-1).clamp_min(1e-12)
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
                        total = total + current
                    inverse = (eta * total).detach()
                if step % 20 == 0:
                    product = torch.autograd.grad(
                        train_gradient, numeric, grad_outputs=inverse, retain_graph=True
                    )[0]
                    relative = (product - vector).norm(dim=-1) / vector.norm(
                        dim=-1
                    ).clamp_min(1e-12)
                    inverse_diagnostics.append(
                        {
                            "outer_step": step,
                            "spectral_damping_max_task": damping_record,
                            "inverse_relative_residual_max_task": relative.amax(1)
                            .detach()
                            .cpu()
                            .tolist(),
                            "inner_gradient_norm_max_task": train_gradient.norm(dim=-1)
                            .amax(1)
                            .detach()
                            .cpu()
                            .tolist(),
                        }
                    )
                mixed = torch.autograd.grad(
                    train_gradient, state, grad_outputs=inverse
                )[0]
                gradient = direct - mixed + regularization_gradient(state, rows, k)
                magnitude = gradient.flatten(1).norm(dim=1).clamp_min(1e-12)
                gradient = gradient * (25 / magnitude).clamp_max(1).reshape(
                    -1, *([1] * (gradient.ndim - 1))
                )
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError("Nonfinite implicit hypergradient")
                outer_optimizer.step(state, gradient)
                if method == "yeh":
                    with torch.no_grad():
                        state.sub_(state.mean(-1, keepdim=True))
        torch.cuda.synchronize(device)
        raw_numeric = selected_numeric.cpu().numpy()
        raw_numeric[..., k + h : -o] *= scale[None, :, None]
        raw_numeric[..., -o:] = (
            raw_numeric[..., -o:] * scale[None, :, None] + center[None]
        )
        arrays = {
            "basis": selected_basis.cpu().numpy(),
            "source_parameters": raw_numeric,
            "source_architecture_mse": best.cpu().numpy(),
            "steps": np.array(history["steps"]),
            "train_history": np.stack(history["train"]),
            "validation_history": np.stack(history["validation"]),
            "elapsed_history": np.array(history["seconds"]),
            "normalization_scale": scale,
        }
        np.savez_compressed(folder / "continuous.npz", **arrays)
        if method == "yeh":
            np.savez_compressed(
                folder / "native.npz",
                indices=np.concatenate(candidates),
                numerical_selection_mse=np.concatenate(candidate_scores),
                normalization_scale=scale,
            )
        save_json(
            folder / "hypergradient_diagnostics.json",
            {"checkpoints": inverse_diagnostics},
        )
        status = "complete"
    finally:
        save_json(
            folder / "metrics.json",
            {
                "status": status,
                "wall_seconds": time.perf_counter() - all_started,
                "training_wall_seconds": time.perf_counter() - started,
                "torch_operator_flops": counter.get_total_flops(),
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
                "continuous_sha256": digest(folder / "continuous.npz")
                if status == "complete"
                else None,
            },
        )
        boundary.save(folder / "access_receipt.json", status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--condition", type=int, default=0)
    parser.add_argument("--method", choices=("yeh", "msr"), required=True)
    parser.add_argument("--outer-steps", type=int, default=400)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-method")
    parser.add_argument("--following-inner-steps", type=int, default=20)
    parser.add_argument("--direct-hessian", action="store_true")
    args = parser.parse_args()
    run(
        args.root.resolve(),
        args.case,
        args.condition,
        args.method,
        args.outer_steps,
        torch.device(args.device),
        output_method=args.output_method,
        following_inner_steps=args.following_inner_steps,
        direct_hessian=args.direct_hessian,
    )


if __name__ == "__main__":
    main()
