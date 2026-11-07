"""Ordinary Adam control with learned hidden biases and a free first layer."""

import argparse
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from .blind_contract import digest, load_sizes, save_json


def dense_prediction(x, parameters, dimensions, hidden, outputs):
    end = dimensions * hidden
    weight = parameters[:, :end].reshape(-1, dimensions, hidden)
    bias = parameters[:, end : end + hidden]
    head = parameters[:, end + hidden : -outputs].reshape(-1, hidden, outputs)
    offset = parameters[:, -outputs:]
    activation = (torch.einsum("bnd,bdh->bnh", x, weight) + bias[:, None]).relu()
    return (
        torch.einsum("bnh,bho->bno", activation, head) / math.sqrt(hidden)
        + offset[:, None]
    )


def run(root, case, condition, device, steps):
    torch.set_num_threads(2)
    started = time.monotonic()
    sizes = load_sizes(root, case)
    dimensions, hidden, outputs = [sizes[k] for k in ("inputs", "hidden", "outputs")]
    folder = root / "dense" / f"{case}_c{condition}"
    folder.mkdir(parents=True, exist_ok=False)
    fixed_path = root / "public" / f"{case}_c{condition}" / "fixed_train.npz"
    if fixed_path.exists():
        fixed = dict(np.load(fixed_path))
        data = [
            {"fit_x": fixed[f"x_{task}"], "fit_y": fixed[f"y_{task}"]}
            for task in range(len(fixed) // 2)
        ]
    else:
        sources = sorted((root / "source").glob(f"{case}_c{condition}_t*/learned.npz"))
        data = [dict(np.load(p)) for p in sources]
    validation = dict(
        np.load(root / "public" / f"{case}_c{condition}" / "selection.npz")
    )
    starts, tasks = 12, len(data)
    x = np.stack([item["fit_x"] for item in data])
    y = np.stack([item["fit_y"] for item in data])
    vx = np.stack([validation[f"x_{t}"] for t in range(tasks)])
    vy = np.stack([validation[f"y_{t}"] for t in range(tasks)])
    center = y.mean(axis=1)
    scale = np.maximum(y.std(axis=(1, 2)), 1e-8)
    target = (y - center[:, None]) / scale[:, None, None]
    vtarget = (vy - center[:, None]) / scale[:, None, None]
    tx, ty, tvx, tvy = [
        torch.tensor(np.repeat(a, starts, axis=0), device=device, dtype=torch.float32)
        for a in (x, target, vx, vtarget)
    ]
    rng = np.random.default_rng(334001)
    weight = rng.normal(size=(tasks * starts, dimensions * hidden)) * math.sqrt(
        2 / dimensions
    )
    bias = rng.normal(0, 0.5, (tasks * starts, hidden))
    head = rng.normal(size=(tasks * starts, hidden * outputs))
    offset = np.zeros((tasks * starts, outputs))
    parameters = torch.tensor(
        np.c_[weight, bias, head, offset],
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    learning_rate = torch.tensor(
        np.tile(np.repeat([0.003, 0.01, 0.03], 4), tasks), device=device
    )[:, None]
    first, second = torch.zeros_like(parameters), torch.zeros_like(parameters)
    best = parameters.new_full((tasks * starts,), torch.inf)
    selected = parameters.detach().clone()
    history = {"steps": [], "train": [], "validation": []}
    generator = torch.Generator(device=device).manual_seed(334007)
    for step in range(steps + 1):
        if step % 50 == 0 or step == steps:
            with torch.no_grad():
                train = (
                    (dense_prediction(tx, parameters, dimensions, hidden, outputs) - ty)
                    .square()
                    .mean((1, 2))
                )
                score = (
                    (
                        dense_prediction(tvx, parameters, dimensions, hidden, outputs)
                        - tvy
                    )
                    .square()
                    .mean((1, 2))
                )
                better = score < best
                best[better], selected[better] = score[better], parameters[better]
                history["steps"].append(step)
                history["train"].append(train.cpu().numpy())
                history["validation"].append(score.cpu().numpy())
        if step == steps:
            break
        rows = torch.randint(tx.shape[1], (512,), generator=generator, device=device)
        fitted = dense_prediction(tx[:, rows], parameters, dimensions, hidden, outputs)
        loss = (fitted - ty[:, rows]).square().mean()
        loss.backward()
        with torch.no_grad():
            gradient = parameters.grad
            first.mul_(0.9).add_(gradient, alpha=0.1)
            second.mul_(0.999).addcmul_(gradient, gradient, value=0.001)
            update = (first / (1 - 0.9 ** (step + 1))) / (
                (second / (1 - 0.999 ** (step + 1))).sqrt() + 1e-8
            )
            parameters.add_(-learning_rate * update)
            parameters.grad = None
    ids = torch.arange(tasks, device=device) * starts + best.reshape(
        tasks, starts
    ).argmin(1)
    all_selected = selected.cpu().numpy().astype(np.float64).reshape(tasks, starts, -1)
    for task in range(tasks):
        all_selected[task, :, dimensions * hidden + hidden : -outputs] *= scale[task]
        all_selected[task, :, -outputs:] = (
            all_selected[task, :, -outputs:] * scale[task] + center[task]
        )
    chosen = all_selected[np.arange(tasks), (ids % starts).cpu().numpy()]
    arrays = {
        "parameters": chosen,
        "all_parameters": all_selected,
        "all_validation_mse": best.cpu().numpy().reshape(tasks, starts)
        * scale[:, None] ** 2,
        "steps": np.array(history["steps"]),
        "train_history": np.stack(history["train"]),
        "validation_history": np.stack(history["validation"]),
        "scale": scale,
    }
    np.savez_compressed(folder / "fitted.npz", **arrays)
    save_json(
        folder / "frozen.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "sizes": sizes,
            "starts": starts,
            "steps": steps,
            "learning_rates": [0.003, 0.01, 0.03],
            "selected_starts": (ids % starts).cpu().tolist(),
            "seconds": time.monotonic() - started,
            "sha256": digest(folder / "fitted.npz"),
            "teacher_access": False,
        },
    )
    print(
        json.dumps(
            {
                "dense": case,
                "condition": condition,
                "seconds": time.monotonic() - started,
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--condition", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=1500)
    args = parser.parse_args()
    run(
        args.root.resolve(),
        args.case,
        args.condition,
        torch.device(args.device),
        args.steps,
    )


if __name__ == "__main__":
    main()
