"""E2CNN synthetic experiments using exact quadratic sufficient statistics."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .linear import save_json


def prepare(root, problem, device):
    # Dependencies and the author generator stay local to this reproduction.
    sys.path.insert(0, str(root.parent / "dependencies"))
    from e2cnn import gspaces
    from e2cnn import nn as enn
    from e2cnn.nn.init import _generalized_he_init_variances

    torch.set_num_threads(2)
    torch.manual_seed({"rot4": 8411, "rot8": 8412, "flip8": 8413}[problem])
    space = gspaces.FlipRot2dOnR2(N=8) if problem == "flip8" else gspaces.Rot2dOnR2(N=4 if problem == "rot4" else 8)
    layer = enn.R2Conv(enn.FieldType(space, [space.trivial_repr]),
                      enn.FieldType(space, [space.regular_repr] * 3), 3, bias=False)
    coefficients = layer.weights.numel()
    with torch.no_grad():
        identity = torch.eye(coefficients)
        basis = torch.stack([layer.basisexpansion(row).reshape(-1, 9) for row in identity])
    deviation = _generalized_he_init_variances(layer.basisexpansion)
    folder = root / problem
    folder.mkdir(parents=True, exist_ok=True)
    save_json(folder / "registration.json", {
        "problem": problem, "train_tasks": 8000, "validation_tasks": 200,
        "test_tasks": 2000, "examples_per_task": 20, "input_shape": [1, 32, 32],
        "output_channels": basis.shape[1], "support": 1, "query": 19,
        "outer_steps": 1000, "meta_batch": 32, "train_inner_steps": 1,
        "test_inner_steps": 1, "initial_inner_lr": 0.1,
        "e2cnn": "0.2.3", "generator_weights": "Author code: generalized He, not unscaled normal",
        "compression": "Per-example X^T X, X^T Y and Y^T Y; exact MSE and derivatives, no altered images or objective",
    })
    for split, tasks in (("train", 8000), ("val", 200), ("test", 2000)):
        cov = np.lib.format.open_memmap(folder / f"{split}_cov.npy", mode="w+", dtype="float32", shape=(tasks, 20, 9, 9))
        cross = np.lib.format.open_memmap(folder / f"{split}_cross.npy", mode="w+", dtype="float32", shape=(tasks, 20, basis.shape[1], 9))
        square = np.lib.format.open_memmap(folder / f"{split}_square.npy", mode="w+", dtype="float32", shape=(tasks, 20))
        for start in range(0, tasks, 16):
            count = min(16, tasks - start)
            z = torch.randn(count, coefficients) * deviation
            weights = torch.einsum("bp,pol->bol", z, basis).to(device)
            x = torch.randn(count * 20, 1, 32, 32, device=device)
            patches = torch.nn.functional.unfold(x, 3).reshape(count, 20, 9, 900)
            gram = torch.einsum("bslp,bsmp->bslm", patches, patches) / 900
            xy = torch.einsum("bslm,bom->bsol", gram, weights)
            yy = torch.einsum("bsol,bol->bs", xy, weights) / weights.shape[1]
            cov[start:start + count] = gram.cpu().numpy()
            cross[start:start + count] = xy.cpu().numpy()
            square[start:start + count] = yy.cpu().numpy()
        cov.flush()
        cross.flush()
        square.flush()
        print(problem, split, tasks, flush=True)
    np.savez_compressed(folder / "teacher_diagnostic.npz", basis=basis.numpy(), coefficient_std=deviation.numpy())


def kernel(numeric, factors, method, outputs):
    if method == "hard":
        index = factors[0]
        dictionary = torch.cat((numeric.new_zeros((len(numeric), 1)), numeric), -1)
        return (dictionary[:, index.abs()] * index.sign()).reshape(-1, outputs, 9)
    weight = numeric.reshape(-1, outputs, 9)
    if method == "msr":
        a, b, c = factors
        weight = torch.einsum("ij,bjk,lk->bil", a, weight, c) * b
    return weight


def losses(numeric, factors, method, gram, cross, square):
    weight = kernel(numeric, factors, method, cross.shape[-2])
    return (torch.einsum("bol,blm,bom->b", weight, gram, weight)
            - 2 * (weight * cross).sum((1, 2))) / weight.shape[1] + square


def adapt(initial, rate, factors, method, gram, cross, square, higher):
    numeric = initial.expand(len(gram), -1).clone().requires_grad_()
    loss = losses(numeric, factors, method, gram, cross, square).sum()
    gradient = torch.autograd.grad(loss, numeric, create_graph=higher)[0]
    return numeric - rate * gradient


def load_panel(folder, split, device):
    return [torch.as_tensor(np.load(folder / f"{split}_{kind}.npy"), device=device)
            for kind in ("cov", "cross", "square")]


def evaluate(folder, split, initial, rate, factors, method, device):
    panel = load_panel(folder, split, device)
    values = []
    for start in range(0, len(panel[0]), 128):
        batch = [array[start:start + 128] for array in panel]
        numeric = adapt(initial, rate, factors, method, *[a[:, 0] for a in batch], False)
        with torch.no_grad():
            result = losses(numeric, factors, method, *[a[:, 1:].mean(1) for a in batch])
            values.extend(result.clamp_min(0).cpu().tolist())
    return np.asarray(values)


def run(folder, method, outer_lr, seed, device, index_path=None):
    torch.set_num_threads(2)
    torch.manual_seed(712000 + seed)
    registration = json.loads((folder / "registration.json").read_text())
    outputs = registration["output_channels"]
    name = index_path.stem if index_path else method
    output = folder / "fits" / f"{name}_lr{outer_lr:g}_s{seed}"
    if (output / "complete.json").exists():
        return
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "registered.json", {
        "method": method, "outer_lr": outer_lr, "seed": seed,
        "train_inner_steps": 1, "test_inner_steps": 1, "outer_steps": 1000,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    index = None if index_path is None else np.load(index_path)["index"]
    count = outputs * 9 if index is None else int(np.abs(index).max())
    initial = torch.empty(count, device=device).uniform_(-1/3, 1/3).requires_grad_()
    rate = torch.tensor(0.1, device=device, requires_grad=True)
    factors = [tensor.requires_grad_() for tensor in
               (torch.eye(outputs, device=device), torch.ones((), device=device), torch.eye(9, device=device))] if method == "msr" else []
    if method == "hard":
        factors = [torch.as_tensor(index.T.ravel(), device=device, dtype=torch.long)]
    optimizer = torch.optim.Adam([initial, rate, *(factors if method == "msr" else [])], lr=outer_lr)
    panel = load_panel(folder, "train", device)
    history = []
    started = time.perf_counter()
    for step in range(1000):
        optimizer.zero_grad()
        task = torch.randperm(8000, device=device)[:32]
        example = torch.randperm(20, device=device)
        batch = [array[task][:, example] for array in panel]
        numeric = adapt(initial, rate, factors, method, *[a[:, 0] for a in batch], True)
        loss = losses(numeric, factors, method, *[a[:, 1:].mean(1) for a in batch]).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite rotation objective")
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == 999:
            history.append([step, loss.item(), time.perf_counter() - started])
            print(json.dumps({"step": step, "loss": loss.item(), "seconds": time.perf_counter() - started}), flush=True)
    frozen = [a.detach() for a in factors]
    validation = evaluate(folder, "val", initial.detach(), rate.detach(), frozen, method, device)
    np.savez_compressed(output / "model.npz", initial=initial.detach().cpu().numpy(),
                        rate=rate.detach().cpu().numpy(), history=np.asarray(history),
                        validation=validation, **{f"factor_{i}": a.cpu().numpy() for i, a in enumerate(frozen)})
    save_json(output / "complete.json", {"status": "complete", "validation_mse": float(validation.mean()),
                                        "seconds": time.perf_counter() - started, "test_targets_read": False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--problem", required=True)
    parser.add_argument("--method", default="msr")
    parser.add_argument("--outer-lr", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--index-path", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.root.resolve(), args.problem, torch.device(args.device))
    else:
        run(args.root / args.problem, args.method, args.outer_lr, args.seed, torch.device(args.device), args.index_path)


if __name__ == "__main__":
    main()
