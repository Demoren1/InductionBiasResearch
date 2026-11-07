"""MSR-FC meta-learning with a shared basis, learned initialization and step sizes."""

import argparse
import itertools
import json
import resource
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
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

from .bilevel import initialize, losses


def rate_vector(rates, codes, hidden, outputs):
    return torch.cat(
        [
            rates[:, group : group + 1].expand(-1, size)
            for group, size in enumerate((codes, hidden, hidden * outputs, outputs))
        ],
        dim=1,
    )


def adapt(x, y, basis, initial, rates, hidden, outputs, steps, higher_order):
    parameter = initial[:, None].expand(-1, x.shape[0], -1)
    vector = rate_vector(rates, basis.shape[-1], hidden, outputs)[:, None]
    for _ in range(steps):
        if not higher_order:
            parameter = parameter.detach().requires_grad_()
        value = losses(x, y, basis, parameter, hidden, outputs).sum()
        gradient = torch.autograd.grad(value, parameter, create_graph=higher_order)[0]
        parameter = parameter - vector * gradient
    return parameter


def run(root, case, condition, steps, device):
    all_started = time.perf_counter()
    torch.set_num_threads(2)
    folder = root / "competitors" / f"{case}_c{condition}" / "msr_maml"
    folder.mkdir(parents=True, exist_ok=False)
    rows = [
        {"method": "msr", "lr": lr, "initialization": initial, "inner_lr": inner}
        for lr, initial, inner in itertools.product(
            (0.0003, 0.001, 0.003), ("cold", "dense_adam"), (0.01, 0.1)
        )
    ]
    save_json(
        folder / "registered.json",
        {
            "method": "MSR-FC, full higher-order MAML",
            "source": "https://github.com/AllanYangZhou/metalearning-symmetries",
            "author_components": "ShareLinearFull U*v representation; inner loop excludes U; learned numerical initialization and unconstrained learned per-parameter-group SGD step sizes; outer Adam.",
            "adaptation": "Shared U only for W1 of the prescribed two-layer ReLU model; hidden biases and free W2/offset adapted per task. No post-hoc clustering.",
            "transfer_policy": "Primary comparison discards learned numerical initialization and transfers U only. Save the native initialization separately for an explicitly labelled auxiliary control.",
            "settings": rows,
            "outer_steps": steps,
            "train_inner_steps": 1,
            "checkpoint_inner_steps": 3,
            "train_batch": 512,
            "query_batch": 256,
            "checkpoint_every": 50,
            "code_sha256": digest(Path(__file__)),
            "helper_sha256": digest(Path(__file__).with_name("bilevel.py")),
            "registered_utc": datetime.now(UTC).isoformat(),
        },
    )
    shutil.copy2(__file__, folder / "executed_code.py")
    boundary = ObservationBoundary(root, case, condition)
    boundary.phase = "msr_maml"
    boundary.install()
    sizes = load_sizes(root, case)
    d, h, k, o = [sizes[name] for name in ("inputs", "hidden", "codes", "outputs")]
    fixed = dict(np.load(root / "public" / f"{case}_c{condition}/fixed_train.npz"))
    validation = dict(np.load(root / "public" / f"{case}_c{condition}/selection.npz"))
    tasks, count = len(fixed) // 2, len(rows)
    x, y, vx, vy = [
        np.stack([panel[f"{name}_{t}"] for t in range(tasks)])
        for panel, name in (
            (fixed, "x"),
            (fixed, "y"),
            (validation, "x"),
            (validation, "y"),
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
    basis, numeric = initialize(
        root, case, condition, rows, sizes, center, scale, device
    )
    initial = numeric[:, 0].detach().clone().requires_grad_()
    rates = (
        basis.new_tensor([row["inner_lr"] for row in rows])[:, None]
        .repeat(1, 4)
        .requires_grad_()
    )
    optimizers = [
        ScenarioAdam(p, [row["lr"] for row in rows]) for p in (basis, initial, rates)
    ]
    generator = torch.Generator(device=device).manual_seed(750101)
    best = basis.new_full((count,), torch.inf)
    selected_basis = basis.detach().clone()
    selected_initial = initial.detach().clone()
    selected_rates = rates.detach().clone()
    selected_parameters = numeric.detach().clone()
    history = {"steps": [], "train": [], "validation": [], "seconds": []}
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    counter = FlopCounterMode(display=False)
    status = "failed"
    try:
        with counter:
            for step in range(steps + 1):
                if step % 50 == 0 or step == steps:
                    adapted = adapt(
                        tx,
                        ty,
                        basis.detach(),
                        initial.detach(),
                        rates.detach(),
                        h,
                        o,
                        3,
                        False,
                    )
                    with torch.no_grad():
                        train = losses(tx, ty, basis, adapted, h, o)
                        validation_loss = losses(tvx, tvy, basis, adapted, h, o)
                        score = validation_loss.mean(1)
                        improved = score < best
                        best[improved] = score[improved]
                        selected_basis[improved] = basis[improved]
                        selected_initial[improved] = initial[improved]
                        selected_rates[improved] = rates[improved]
                        selected_parameters[improved] = adapted[improved]
                        history["steps"].append(step)
                        history["train"].append(train.cpu().numpy())
                        history["validation"].append(validation_loss.cpu().numpy())
                        history["seconds"].append(time.perf_counter() - started)
                        print(
                            json.dumps(
                                {
                                    "method": "msr_maml",
                                    "case": case,
                                    "outer_step": step,
                                    "best_source_architecture_mse": float(best.min()),
                                    "seconds": time.perf_counter() - started,
                                }
                            ),
                            flush=True,
                        )
                if step == steps:
                    break
                positions = torch.randint(
                    tx.shape[1], (512,), device=device, generator=generator
                )
                queries = torch.randint(
                    tvx.shape[1], (256,), device=device, generator=generator
                )
                adapted = adapt(
                    tx[:, positions],
                    ty[:, positions],
                    basis,
                    initial,
                    rates,
                    h,
                    o,
                    1,
                    True,
                )
                objective = (
                    losses(tvx[:, queries], tvy[:, queries], basis, adapted, h, o)
                    .mean(1)
                    .sum()
                )
                gradients = torch.autograd.grad(objective, (basis, initial, rates))
                if any(not torch.isfinite(gradient).all() for gradient in gradients):
                    raise FloatingPointError("Nonfinite MSR meta-gradient")
                for parameter, optimizer, gradient in zip(
                    (basis, initial, rates), optimizers, gradients
                ):
                    optimizer.step(parameter, gradient)
        torch.cuda.synchronize(device)
        parameters = selected_parameters.cpu().numpy()
        parameters[..., k + h : -o] *= scale[None, :, None]
        parameters[..., -o:] = (
            parameters[..., -o:] * scale[None, :, None] + center[None]
        )
        np.savez_compressed(
            folder / "continuous.npz",
            basis=selected_basis.cpu().numpy(),
            source_parameters=parameters,
            source_meta_initialization=selected_initial.cpu().numpy(),
            inner_learning_rates=selected_rates.cpu().numpy(),
            source_architecture_mse=best.cpu().numpy(),
            steps=np.array(history["steps"]),
            train_history=np.stack(history["train"]),
            validation_history=np.stack(history["validation"]),
            elapsed_history=np.array(history["seconds"]),
            normalization_scale=scale,
            normalization_center=center,
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
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(
        args.root.resolve(),
        args.case,
        args.condition,
        args.steps,
        torch.device(args.device),
    )


if __name__ == "__main__":
    main()
