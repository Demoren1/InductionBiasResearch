"""Paper-protocol linear experiments, with native numerical adaptation."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

INPUTS, OUTPUTS = 70, 68


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def make_panel(root, rank, size):
    """Keep original finite tasks; add separate validation coefficient-tasks."""
    rng = np.random.default_rng(14112026 + rank)
    tasks = 400 if rank == 1 else 800
    count = 100 if rank == 1 else 200
    mixing = rng.normal(size=(OUTPUTS, rank))
    mixing = np.exp(mixing - mixing.max(1, keepdims=True))
    mixing /= mixing.sum(1, keepdims=True)
    folder = root / f"rank{rank}_{size}"
    folder.mkdir(parents=True, exist_ok=True)
    for split, number in (("train", tasks), ("val", 200), ("test", count)):
        x = rng.normal(size=(number, 20, INPUTS)).astype(np.float32)
        coefficients = rng.normal(size=(number, rank, 3)).astype(np.float32)
        kernels = np.einsum("or,trl->tol", mixing, coefficients)
        windows = np.lib.stride_tricks.sliding_window_view(x, 3, axis=-1)
        y = np.einsum("tsol,tol->tso", windows, kernels).astype(np.float32)
        # Test targets are kept separate and are never read in training workers.
        np.savez(folder / f"{split}.npz", x=x, y=y)
    np.savez(folder / "teacher_diagnostic.npz", mixing=mixing)
    save_json(folder / "registration.json", {
        "rank": rank, "size": size, "train_tasks": tasks,
        "test_tasks": count, "additional_validation_tasks": 200,
        "examples_per_train_task": 20 if size == "large" else (4 if rank == 5 else 2),
        "meta_train_support": 1, "test_support": 1,
        "test_query": 19, "meta_train_steps": 3, "meta_test_steps": 9,
        "outer_steps": 1000, "meta_batch": 32,
        "published_outer_lr": 0.0005, "initial_inner_lr": 0.02,
        "generator": "Author's fixed softmax mixing and standard normal filters/inputs",
        "source_weight_matrix": [OUTPUTS, INPUTS],
        "main_msr_latent": 70, "released_code_latent": {1: 3, 2: 6, 5: 30}[rank],
        "provenance_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })


def predict(x, numeric, basis, method):
    if method == "hard":
        dictionary = torch.cat((numeric.new_zeros((len(numeric), 1)), numeric), -1)
        weight = (dictionary[:, basis.abs()] * basis.sign()).reshape(-1, OUTPUTS, INPUTS)
        return torch.einsum("bsi,boi->bso", x, weight)
    if method in ("msr", "joint", "mtsr"):
        weight = (numeric @ basis.T).reshape(-1, OUTPUTS, INPUTS)
        return torch.einsum("bsi,boi->bso", x, weight)
    if method == "fc":
        return torch.einsum("bsi,boi->bso", x, numeric.reshape(-1, OUTPUTS, INPUTS))
    patches = x.unfold(-1, 3, 1)
    if method == "conv":
        return torch.einsum("bsol,bl->bso", patches, numeric)
    return torch.einsum("bsol,bol->bso", patches, numeric.reshape(-1, OUTPUTS, 3))


def adapt(x, y, initial, basis, rates, method, steps, higher):
    numeric = initial.expand(len(x), -1).clone().requires_grad_()
    if method == "joint":
        local_basis = basis.expand(len(x), -1, -1).clone().requires_grad_()
    else:
        local_basis = basis
    for _ in range(steps):
        if method == "joint":
            weight = torch.einsum("bnk,bk->bn", local_basis, numeric).reshape(
                -1, OUTPUTS, INPUTS
            )
            loss = (torch.einsum("bsi,boi->bso", x, weight) - y).square().mean((1, 2)).sum()
            gn, gb = torch.autograd.grad(loss, (numeric, local_basis), create_graph=higher)
            numeric = numeric - rates[0] * gn
            local_basis = local_basis - rates[1] * gb
        else:
            loss = (predict(x, numeric, basis, method) - y).square().mean((1, 2)).sum()
            gradient = torch.autograd.grad(loss, numeric, create_graph=higher)[0]
            numeric = numeric - rates[0] * gradient
        if not higher:
            numeric = numeric.detach().requires_grad_()
            if method == "joint":
                local_basis = local_basis.detach().requires_grad_()
    return numeric, local_basis


def task_losses(x, y, numeric, basis, method):
    if method == "joint" and basis.ndim == 3:
        weight = torch.einsum("bnk,bk->bn", basis, numeric).reshape(-1, OUTPUTS, INPUTS)
        output = torch.einsum("bsi,boi->bso", x, weight)
    else:
        output = predict(x, numeric, basis, method)
    return (output - y).square().mean((1, 2))


def initialize(method, latent, device, train_tasks, index_path=None):
    if method == "hard":
        basis = torch.as_tensor(np.load(index_path)["index"].T.ravel(), device=device,
                                dtype=torch.long)
        initial = torch.empty(latent, device=device).uniform_(-1 / math.sqrt(INPUTS),
                                                             1 / math.sqrt(INPUTS)).requires_grad_()
    elif method in ("msr", "joint", "mtsr"):
        basis = (torch.randn(INPUTS * OUTPUTS, latent, device=device) * 0.01).requires_grad_()
        shape = (train_tasks, latent) if method == "mtsr" else (latent,)
        initial = (torch.randn(shape, device=device) / OUTPUTS).requires_grad_()
    else:
        basis = None
        count = {"fc": INPUTS * OUTPUTS, "lc": OUTPUTS * 3, "conv": 3}[method]
        bound = 1 / math.sqrt(INPUTS if method == "fc" else 3)
        initial = torch.empty(count, device=device).uniform_(-bound, bound).requires_grad_()
    rates = torch.tensor([0.02] * (2 if method == "joint" else 1), device=device,
                         requires_grad=True)
    return basis, initial, rates


def evaluate(panel, initial, basis, rates, method, device):
    values = []
    for begin in range(0, len(panel["x"]), 32):
        x, y = [torch.as_tensor(panel[key][begin:begin + 32], device=device)
                for key in ("x", "y")]
        if method == "mtsr":
            generator = torch.Generator(device=device).manual_seed(7811 + begin)
            numeric = (torch.randn(len(x), basis.shape[1], device=device,
                                   generator=generator) / OUTPUTS).requires_grad_()
            optimizer = torch.optim.Adam([numeric], lr=0.001)
            for _ in range(500):
                optimizer.zero_grad()
                loss = task_losses(x[:, :1], y[:, :1], numeric, basis, method).sum()
                loss.backward()
                optimizer.step()
            local_basis = basis
        else:
            numeric, local_basis = adapt(x[:, :1], y[:, :1], initial, basis, rates,
                                         method, 9, False)
        with torch.no_grad():
            values.extend(task_losses(x[:, 1:], y[:, 1:], numeric, local_basis,
                                      method).cpu().tolist())
    return np.asarray(values)


def run(folder, method, latent, outer_lr, seed, device, index_path=None):
    torch.set_num_threads(2)
    torch.manual_seed(712000 + seed)
    registration = json.loads((folder / "registration.json").read_text())
    name = index_path.stem if index_path else method
    output = folder / "fits" / f"{name}_m{latent}_lr{outer_lr:g}_s{seed}"
    if (output / "complete.json").exists():
        return
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "registered.json", {
        "method": method, "latent": latent, "outer_lr": outer_lr,
        "seed": seed, "phase": "train_and_validation_only", "steps": 500 if method == "mtsr" else 1000,
        "implementation": "Batched exact higher-order gradients of the author's objective; no first-order approximation",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    source = dict(np.load(folder / "train.npz"))
    count = registration["examples_per_train_task"]
    x, y = [torch.as_tensor(source[key][:, :count], device=device) for key in ("x", "y")]
    basis, initial, rates = initialize(method, latent, device, len(x), index_path)
    parameters = [initial] + ([] if basis is None or not basis.requires_grad else [basis])
    if method != "mtsr":
        parameters.append(rates)
    optimizer = torch.optim.Adam(parameters, lr=outer_lr)
    history = []
    started = time.perf_counter()
    for step in range(500 if method == "mtsr" else 1000):
        optimizer.zero_grad()
        if method == "mtsr":
            loss = task_losses(x, y, initial, basis, method).mean()
        else:
            tasks = torch.randperm(len(x), device=device)[:32]
            positions = torch.randperm(count, device=device)
            bx, by = x[tasks][:, positions], y[tasks][:, positions]
            numeric, local_basis = adapt(bx[:, :1], by[:, :1], initial, basis,
                                         rates, method, 3, True)
            loss = task_losses(bx[:, 1:], by[:, 1:], numeric, local_basis, method).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training objective at {step}")
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == 999:
            history.append([step, loss.item(), time.perf_counter() - started])
            print(json.dumps({"job": output.name, "step": step,
                              "loss": loss.item(), "seconds": time.perf_counter() - started}), flush=True)
    validation = evaluate(dict(np.load(folder / "val.npz")), initial.detach(),
                          None if basis is None else basis.detach(), rates.detach(), method, device)
    np.savez_compressed(output / "model.npz", initial=initial.detach().cpu().numpy(),
                        basis=np.empty(0) if basis is None else basis.detach().cpu().numpy(),
                        rates=rates.detach().cpu().numpy(), validation=validation,
                        history=np.asarray(history))
    save_json(output / "complete.json", {"status": "complete", "validation_mse": float(validation.mean()),
                                        "seconds": time.perf_counter() - started,
                                        "test_targets_read": False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--case")
    parser.add_argument("--method", default="msr")
    parser.add_argument("--latent", type=int, default=70)
    parser.add_argument("--outer-lr", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--index-path", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.prepare:
        for rank in (1, 2, 5):
            for size in ("small", "large"):
                make_panel(args.root, rank, size)
    else:
        run(args.root / args.case, args.method, args.latent, args.outer_lr, args.seed,
            torch.device(args.device), args.index_path)


if __name__ == "__main__":
    main()
