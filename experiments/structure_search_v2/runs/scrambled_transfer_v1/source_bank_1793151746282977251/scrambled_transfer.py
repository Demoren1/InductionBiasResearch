"""Distill quality-ranked, moment-guided source structures into NF+VAE.

The Gaussian moment proposer is an explicit source-data ingredient, not an
achievement attributed to NF. It is evaluated directly as an essential control.
"""

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from nf_prior import NFVAE, normalized_features
from scipy.optimize import linear_sum_assignment
from scrambled_pattern import predict
from torch.nn import functional as F

ROOT = Path("experiments/structure_search_v2")
SOURCE = ROOT / "runs/scrambled_pattern_source_pilot_v1"
OUT = ROOT / "runs/scrambled_transfer_v1"


def permutation_matrix(logits, straight_through=False):
    shape = logits.shape
    flat = logits.reshape(-1, 64, 64)
    costs = -flat.detach().cpu().numpy()
    labels = []
    for cost in costs:
        rows, columns = linear_sum_assignment(cost)
        assignment = np.empty(64, dtype=np.int64)
        assignment[rows] = columns
        labels.append(assignment)
    indices = torch.tensor(np.array(labels), device=logits.device)
    hard = F.one_hot(indices, 64).to(logits.dtype)
    if straight_through:
        soft = flat.softmax(-1)
        hard = hard + soft - soft.detach()
    return hard.reshape(shape)


def bank(device):
    spec = {
        "seed": 281300,
        "tasks": 4,
        "candidates": 128,
        "starts": 2,
        "steps": 1536,
        "train_genomes": 24,
        "validation_genomes": 4,
        "test_genomes": 4,
        "lr": 0.01,
    }
    (OUT / "bank_protocol.json").write_text(json.dumps(spec, indent=2))
    source = np.load(SOURCE / "source_data.npz")
    x = torch.tensor(source["x"], device=device)
    y = torch.tensor(source["y"], device=device)
    # This is inferred from source observations, not read from hidden_teacher.
    base = np.load(SOURCE / "moment_proposal.npz")["permutation"]
    rng = np.random.default_rng(spec["seed"])
    genomes = [base.copy()]
    seen = {tuple(base)}
    while len(genomes) < spec["candidates"]:
        pi = base.copy()
        for _ in range(rng.integers(1, 9)):
            pair = rng.choice(64, 2, replace=False)
            pi[pair] = pi[pair[::-1]]
        if tuple(pi) not in seen:
            genomes.append(pi)
            seen.add(tuple(pi))
    genomes = torch.tensor(np.array(genomes), device=device)
    m, tasks, starts = spec["candidates"], spec["tasks"], spec["starts"]
    torch.manual_seed(spec["seed"] + 1)
    parameters = [
        torch.randn(m, tasks, starts, 4, 8, device=device) * 0.5,
        torch.full((m, tasks, starts, 4), -1.0, device=device),
        torch.ones(m, tasks, starts, 4, device=device),
        torch.zeros(m, tasks, starts, device=device),
    ]
    parameters = [p.requires_grad_() for p in parameters]
    optimizer = torch.optim.Adam(parameters, lr=spec["lr"])
    scale = y[:, :8192].var(1, unbiased=False).clamp_min(1e-8)
    best = [p.detach().clone() for p in parameters]
    best_loss = torch.full((m, tasks, starts), torch.inf, device=device)
    history = []
    for step in range(spec["steps"] + 1):
        if step % 128 == 0:
            with torch.no_grad():
                error = torch.zeros_like(best_loss)
                for ids in torch.arange(8192, len(x), device=device).split(256):
                    residual = (
                        predict(x[ids], genomes, parameters) - y[None, :, None, ids]
                    )
                    error += residual.square().sum(-1) / 2048 / scale[None, :, None]
                better = error < best_loss
                best_loss[better] = error[better]
                for stored, current in zip(best, parameters):
                    stored[better] = current[better]
                record = {
                    "step": step,
                    "best_source_nmse": float(best_loss.min(-1).values.mean(-1).min()),
                    "mean_source_nmse": float(best_loss.mean()),
                }
                history.append(record)
                print("BANK", record, flush=True)
        if step == spec["steps"]:
            break
        ids = torch.randint(8192, (128,), device=device)
        residual = predict(x[ids], genomes, parameters) - y[None, :, None, ids]
        loss = (residual.square().mean(-1) / scale[None, :, None]).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    quality = best_loss.min(-1).values.mean(-1)
    selected = quality.argsort()[:32]
    # Split unique structures, keeping all numerical replicas in the same split.
    generator = torch.Generator(device=device).manual_seed(spec["seed"] + 2)
    selected = selected[
        torch.randperm(len(selected), generator=generator, device=device)
    ]
    selected_genomes = genomes[selected]
    fitted = [p.detach()[selected] for p in best]
    # Expand a fitted codebook to numerical weights, preserving its exact indices.
    n = len(selected) * tasks * starts
    w = torch.zeros(len(selected), tasks, starts, 64, 32, device=device)
    for candidate, permutation in enumerate(selected_genomes):
        for group in range(8):
            for k in range(4):
                w[
                    candidate,
                    :,
                    :,
                    permutation[group * 8 : group * 8 + 8],
                    group * 4 + k,
                ] = fitted[0][candidate, :, :, k]
    bias = fitted[1][:, :, :, None].expand(-1, -1, -1, 8, -1).reshape(n, 32)
    readout = fitted[2][:, :, :, None].expand(-1, -1, -1, 8, -1).reshape(
        n, 32
    ) / math.sqrt(32)
    offset = fitted[3].reshape(n)
    weights = w.reshape(n, 64, 32).detach().requires_grad_()
    task_ids = (
        torch.arange(tasks, device=device)[None, :, None]
        .expand(len(selected), -1, starts)
        .reshape(n)
    )
    probe = x[:128]
    pre = probe @ weights + bias[:, None]
    logits = (pre.relu() * readout[:, None]).sum(-1) + offset[:, None]
    loss = ((logits - y[task_ids, :128]).square().mean(1) / scale[task_ids]).sum()
    gradient = torch.autograd.grad(loss, weights)[0]
    response = (weights * readout[:, None]).abs() * (pre > 0).float().mean(1)[:, None]
    features = normalized_features(weights.detach(), gradient, response)
    targets = (
        selected_genomes[:, None, None].expand(-1, tasks, starts, -1).reshape(n, 64)
    )
    roles = torch.tensor([0] * 24 + [1] * 4 + [2] * 4, device=device)
    roles = roles[:, None, None].expand(-1, tasks, starts).reshape(n)
    np.savez_compressed(
        OUT / "bank.npz",
        features=features.detach().cpu().numpy(),
        targets=targets.cpu().numpy(),
        task=task_ids.cpu().numpy(),
        role=roles.cpu().numpy(),
        quality=best_loss[selected].reshape(n).cpu().numpy(),
        selected_genomes=selected_genomes.cpu().numpy(),
        all_genomes=genomes.cpu().numpy(),
        all_quality=best_loss.cpu().numpy(),
        codebook=fitted[0].cpu().numpy(),
        bias=fitted[1].cpu().numpy(),
        readout=fitted[2].cpu().numpy(),
        offset=fitted[3].cpu().numpy(),
    )
    (OUT / "bank_history.json").write_text(json.dumps(history, indent=2))


def prior(device):
    data = np.load(OUT / "bank.npz")
    features = torch.tensor(data["features"], device=device)
    targets = torch.tensor(data["targets"], device=device)
    task_ids, role = data["task"], data["role"]
    train_pools = [
        torch.tensor(np.flatnonzero((task_ids == task) & (role == 0)), device=device)
        for task in range(4)
    ]
    validation = torch.stack(
        [
            torch.tensor(
                np.flatnonzero((task_ids == task) & (role == 1)), device=device
            )
            for task in range(4)
        ]
    )
    torch.manual_seed(281310)
    model = NFVAE(4, 64, 64 * 64).to(device)
    torch.save(model.state_dict(), OUT / "untrained_prior.pt")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history, best = [], float("inf")
    for epoch in range(301):
        model.eval()
        with torch.no_grad():
            logits, _, _ = model.encode(features[validation], sample=False)
            val = F.cross_entropy(logits.reshape(-1, 64), targets[validation].flatten())
            if float(val) < best:
                best = float(val)
                torch.save(model.state_dict(), OUT / "prior.pt")
        if epoch == 300:
            break
        model.train()
        for _ in range(2):
            selected = torch.stack(
                [
                    pool[torch.randint(len(pool), (24,), device=device)]
                    for pool in train_pools
                ]
            )
            logits, mu, logvar = model.encode(features[selected])
            ce = F.cross_entropy(logits.reshape(-1, 64), targets[selected].flatten())
            kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).mean()
            loss = ce + 0.1 * min(1, (epoch + 1) / 50) * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        history.append(
            {
                "epoch": epoch,
                "validation_ce": float(val),
                "ce": float(ce.detach()),
                "kl": float(kl.detach()),
            }
        )
        if epoch % 25 == 0:
            print("PRIOR", history[-1], flush=True)
    model.load_state_dict(torch.load(OUT / "prior.pt", weights_only=True))
    with torch.no_grad():
        z = torch.randn(512, 16, device=device)
        samples = permutation_matrix(model.decoder(z).reshape(-1, 64, 64)).argmax(-1)
    np.savez_compressed(OUT / "prior_samples.npz", permutations=samples.cpu().numpy())
    (OUT / "prior_history.json").write_text(json.dumps(history))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["bank", "prior"], required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot = OUT / ("source_" + args.stage + "_" + str(time.time_ns()))
    snapshot.mkdir()
    for file in ["scrambled_transfer.py", "nf_prior.py", "scrambled_pattern.py"]:
        shutil.copy2(ROOT / file, snapshot / file)
    torch.set_num_threads(4)
    started = time.time()
    {"bank": bank, "prior": prior}[args.stage](torch.device("cuda"))
    (OUT / (args.stage + "_complete.json")).write_text(
        json.dumps({"seconds": time.time() - started})
    )


if __name__ == "__main__":
    main()
