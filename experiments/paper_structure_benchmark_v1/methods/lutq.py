"""Native minibatch LUT-Q with declared fixed-zero signed-code adaptation."""

import argparse
import itertools
import json
import math
import os
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
from experiments.sharing_structure_research_v1.blind_dense import dense_prediction
from experiments.sharing_structure_research_v1.blind_runtime import (
    ObservationBoundary,
)
from experiments.sharing_structure_research_v1.shared_structure_bilevel_optimization import (
    ScenarioAdam,
)


@torch.no_grad()
def quantize(weight, centers):
    dictionary = torch.cat((centers[:, :1] * 0, centers, -centers), 1)
    labels = (weight[..., None] - dictionary[:, None]).square().argmin(-1)
    codes = centers.shape[1]
    index = torch.where(labels > codes, codes - labels, labels)
    selected = dictionary.gather(1, labels)
    return index, selected


@torch.no_grad()
def update_centers(weight, centers):
    index, _ = quantize(weight, centers)
    labels = index.abs()
    counts = torch.zeros((len(weight), centers.shape[1] + 1), device=weight.device)
    total = torch.zeros_like(counts)
    counts.scatter_add_(1, labels, torch.ones_like(weight))
    total.scatter_add_(1, labels, weight * index.sign())
    return torch.where(
        counts[:, 1:] > 0, total[:, 1:] / counts[:, 1:].clamp_min(1), centers
    )


def run(root, case, condition, steps, device):
    all_started = time.perf_counter()
    torch.set_num_threads(2)
    folder = root / "competitors" / f"{case}_c{condition}" / "lutq"
    folder.mkdir(parents=True, exist_ok=False)
    rows = [
        {"lr": lr, "initialization": initial, "restart": restart}
        for lr, initial, restart in itertools.product(
            (0.003, 0.01, 0.03), ("cold", "dense_adam"), range(2)
        )
    ]
    registration = {
        "method": "LUT-Q",
        "source": "https://arxiv.org/pdf/1811.05355",
        "adaptation": "Zero is fixed; positive/negative dictionary entries are tied in pairs. Every source is trained independently with the native nearest-code forward, STE latent update and one Lloyd center update after each minibatch. Native indexators are retained without a second clustering.",
        "settings": rows,
        "steps": steps,
        "checkpoint_every": 50,
        "code_sha256": digest(Path(__file__)),
        "registered_utc": datetime.now(UTC).isoformat(),
    }
    save_json(folder / "registered.json", registration)
    shutil.copy2(__file__, folder / "executed_code.py")
    boundary = ObservationBoundary(root, case, condition)
    boundary.phase = "lutq"
    boundary.install()
    sizes = load_sizes(root, case)
    d, h, k, o = [sizes[name] for name in ("inputs", "hidden", "codes", "outputs")]
    fixed = dict(np.load(root / "public" / f"{case}_c{condition}" / "fixed_train.npz"))
    selection = dict(
        np.load(root / "public" / f"{case}_c{condition}" / "selection.npz")
    )
    tasks, count = len(fixed) // 2, len(rows)
    x = np.stack([fixed[f"x_{t}"] for t in range(tasks)])
    y = np.stack([fixed[f"y_{t}"] for t in range(tasks)])
    vx = np.stack([selection[f"x_{t}"] for t in range(tasks)])
    vy = np.stack([selection[f"y_{t}"] for t in range(tasks)])
    center = y.mean(1)
    scale = np.maximum(y.std((1, 2)), 1e-8)
    target = (y - center[:, None]) / scale[:, None, None]
    validation = (vy - center[:, None]) / scale[:, None, None]
    tx, ty, tvx, tvy = [
        torch.tensor(np.tile(a, (count, 1, 1)), device=device, dtype=torch.float32)
        for a in (x, target, vx, validation)
    ]
    generator = torch.Generator(device=device).manual_seed(745003)
    length = d * h + h + h * o + o
    parameter = torch.randn((count, tasks, length), device=device, generator=generator)
    parameter[..., : d * h] *= math.sqrt(2 / d)
    parameter[..., d * h : d * h + h] *= 0.5
    parameter[..., -o:] = 0
    dense = np.load(root / "dense" / f"{case}_c{condition}" / "fitted.npz")[
        "parameters"
    ]
    dense = dense.copy()
    dense[:, d * h + h : -o] /= scale[:, None]
    dense[:, -o:] = (dense[:, -o:] - center) / scale[:, None]
    warm = torch.tensor(dense, device=device, dtype=torch.float32)
    for scenario, row in enumerate(rows):
        if row["initialization"] == "dense_adam":
            parameter[scenario] = warm
    parameter = parameter.flatten(0, 1).requires_grad_()
    quantiles = torch.linspace(0.15, 0.95, k, device=device)
    centers = torch.quantile(parameter[:, : d * h].detach().abs(), quantiles, dim=1).T
    optimizer = ScenarioAdam(
        parameter, [row["lr"] for row in rows for _ in range(tasks)]
    )
    best = parameter.new_full((count * tasks,), torch.inf)
    selected = parameter.detach().clone()
    selected_index = torch.zeros(
        (count * tasks, d * h), device=device, dtype=torch.long
    )
    selected_centers = centers.clone()
    selected_step = torch.zeros(count * tasks, device=device, dtype=torch.long)
    history = {"steps": [], "train": [], "validation": [], "seconds": []}
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    counter = FlopCounterMode(display=False)
    started = time.perf_counter()
    status = "failed"
    try:
        with counter:
            for step in range(steps + 1):
                index, quantized = quantize(parameter[:, : d * h].detach(), centers)
                if step % 50 == 0 or step == steps:
                    with torch.no_grad():
                        effective = torch.cat((quantized, parameter[:, d * h :]), 1)
                        train = (
                            (dense_prediction(tx, effective, d, h, o) - ty)
                            .square()
                            .mean((1, 2))
                        )
                        score = (
                            (dense_prediction(tvx, effective, d, h, o) - tvy)
                            .square()
                            .mean((1, 2))
                        )
                        better = score < best
                        best[better] = score[better]
                        selected[better] = parameter[better]
                        selected_index[better] = index[better]
                        selected_centers[better] = centers[better]
                        selected_step[better] = step
                        history["steps"].append(step)
                        history["train"].append(train.cpu().numpy())
                        history["validation"].append(score.cpu().numpy())
                        history["seconds"].append(time.perf_counter() - started)
                if step == steps:
                    break
                positions = torch.randint(
                    tx.shape[1], (512,), device=device, generator=generator
                )
                latent = parameter[:, : d * h]
                tied = latent + (quantized - latent).detach()
                effective = torch.cat((tied, parameter[:, d * h :]), 1)
                prediction = dense_prediction(tx[:, positions], effective, d, h, o)
                loss = (prediction - ty[:, positions]).square().mean((1, 2)).sum()
                gradient = torch.autograd.grad(loss, parameter)[0]
                optimizer.step(parameter, gradient)
                centers = update_centers(parameter[:, : d * h].detach(), centers)
        torch.cuda.synchronize(device)
        selected = selected.detach().cpu().numpy().reshape(count, tasks, -1)
        selected[..., d * h + h : -o] *= scale[None, :, None]
        selected[..., -o:] = selected[..., -o:] * scale[None, :, None] + center[None]
        np.savez_compressed(
            folder / "native.npz",
            indices=selected_index.cpu().numpy().reshape(count, tasks, d, h),
            codes=selected_centers.cpu().numpy().reshape(count, tasks, k),
            latent_parameters=selected,
            numerical_selection_mse=best.cpu().numpy().reshape(count, tasks)
            * scale[None] ** 2,
            selected_step=selected_step.cpu().numpy().reshape(count, tasks),
            steps=np.array(history["steps"]),
            train_history=np.stack(history["train"]).reshape(-1, count, tasks),
            validation_history=np.stack(history["validation"]).reshape(
                -1, count, tasks
            ),
            elapsed_history=np.array(history["seconds"]),
            normalization_scale=scale,
        )
        status = "complete"
    finally:
        save_json(
            folder / "metrics.json",
            {
                "status": status,
                "pid": os.getpid(),
                "wall_seconds": time.perf_counter() - all_started,
                "training_wall_seconds": time.perf_counter() - started,
                "torch_operator_flops": counter.get_total_flops(),
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
                "warmup_cost": "Selected dense-Adam starts reuse the measured source dense fit, whose cost must be added when reporting warm LUT-Q.",
                "native_sha256": digest(folder / "native.npz")
                if status == "complete"
                else None,
            },
        )
        boundary.save(folder / "access_receipt.json", status)
    print(json.dumps({"method": "lutq", "case": case, "status": status}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--condition", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2500)
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
