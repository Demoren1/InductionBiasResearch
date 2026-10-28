"""Collect partial solutions for a larger shared-filter regression problem."""

import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch


def teacher_labels(x, permutation, codebook, bias, readout):
    grouped = x[:, permutation].reshape(len(x), 8, 8)
    hidden = (
        torch.einsum("bgr,tkr->tbgk", grouped, codebook) + bias[:, None, None, :]
    ).relu()
    return (hidden * readout[:, None, None, :]).sum((-1, -2)) / math.sqrt(32)


def predict(x, permutations, parameters):
    codebook, bias, readout, offset = parameters
    grouped = (
        x[:, permutations].permute(1, 0, 2).reshape(len(permutations), len(x), 8, 8)
    )
    hidden = (
        torch.einsum("mbgr,mtskr->mtsbgk", grouped, codebook)
        + bias[:, :, :, None, None, :]
    ).relu()
    return (hidden * readout[:, :, :, None, None, :]).sum((-1, -2)) / math.sqrt(
        32
    ) + offset[:, :, :, None]


def main():
    root = Path("experiments/structure_search_v2")
    spec = json.loads((root / "scrambled_protocol.json").read_text())
    out = root / "runs" / spec["run"]
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / "source.py")
    shutil.copy2(root / "scrambled_protocol.json", out / "protocol.json")
    torch.set_num_threads(4)
    device = torch.device("cuda")
    torch.manual_seed(spec["seed"])
    permutation = torch.randperm(64, device=device)
    tasks, population, restarts = (
        spec["source_tasks"],
        spec["population"],
        spec["restarts"],
    )
    codebook = 0.75 + torch.rand(tasks, 4, 8, device=device)
    codebook *= torch.where(torch.rand_like(codebook) < 0.5, -1, 1)
    bias = -(2 + torch.rand(tasks, 4, device=device))
    readout = 0.7 + torch.rand(tasks, 4, device=device) * 0.6
    x = torch.randn(spec["support"] + spec["query"], 64, device=device)
    y = teacher_labels(x, permutation, codebook, bias, readout)
    support = slice(0, spec["support"])
    query = slice(spec["support"], None)
    variance = y[:, query].var(1, unbiased=False).clamp_min(1e-8)
    np.savez_compressed(
        out / "hidden_teacher.npz",
        permutation=permutation.cpu().numpy(),
        codebook=codebook.cpu().numpy(),
        bias=bias.cpu().numpy(),
        readout=readout.cpu().numpy(),
    )
    np.savez_compressed(out / "source_data.npz", x=x.cpu().numpy(), y=y.cpu().numpy())
    # The search below receives observations, not teacher parameters or indices.
    torch.manual_seed(spec["seed"] + 1)
    permutations = torch.stack(
        [torch.randperm(64, device=device) for _ in range(population)]
    )
    parameters = [
        torch.randn(population, tasks, restarts, 4, 8, device=device) * 0.5,
        torch.full((population, tasks, restarts, 4), -1.0, device=device),
        torch.ones(population, tasks, restarts, 4, device=device),
        torch.zeros(population, tasks, restarts, device=device),
    ]
    records = []
    saved = {
        "permutations": [],
        "codebook": [],
        "bias": [],
        "readout": [],
        "offset": [],
        "query_nmse": [],
        "generation": [],
    }
    started = time.time()
    for generation in range(spec["generations"]):
        parameters = [p.detach().clone().requires_grad_() for p in parameters]
        optimizer = torch.optim.Adam(parameters, lr=spec["learning_rate"])
        for _ in range(spec["inner_steps"]):
            ids = torch.randint(spec["support"], (spec["batch_size"],), device=device)
            residual = predict(x[ids], permutations, parameters) - y[None, :, None, ids]
            loss = (residual.square().mean(-1) / variance[None, :, None]).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            error = torch.zeros(population, tasks, restarts, device=device)
            for ids in torch.arange(spec["support"], len(x), device=device).split(256):
                residual = (
                    predict(x[ids], permutations, parameters) - y[None, :, None, ids]
                )
                error += (
                    residual.square().sum(-1) / spec["query"] / variance[None, :, None]
                )
            best_error, best_start = error.min(-1)
            fitness = best_error.mean(-1)
            elite = fitness.argsort()[: spec["elite"]]
            record = {
                "generation": generation,
                "best_source_nmse": float(fitness.min()),
                "median_source_nmse": float(fitness.median()),
                "best_task_nmse": best_error[elite[0]].cpu().tolist(),
            }
            records.append(record)
            print("SEARCH", record, flush=True)
            saved["permutations"].append(permutations.cpu().numpy())
            for key, value in zip(
                ["codebook", "bias", "readout", "offset"], parameters
            ):
                saved[key].append(value.detach().cpu().numpy())
            saved["query_nmse"].append(error.cpu().numpy())
            saved["generation"].append(np.full(population, generation))
            np.savez_compressed(
                out / "partial_bank.npz",
                **{key: np.concatenate(value) for key, value in saved.items()},
            )
            (out / "history.json").write_text(json.dumps(records, indent=2))
            if generation + 1 == spec["generations"]:
                break
            parents = elite[torch.randint(len(elite), (population,), device=device)]
            parents[: len(elite)] = elite
            permutations = permutations[parents].clone()
            parameters = [p[parents].detach().clone() for p in parameters]
            for candidate in range(spec["elite"], population):
                for _ in range(spec["swaps"]):
                    pair = torch.randperm(64, device=device)[:2]
                    permutations[candidate, pair] = permutations[
                        candidate, pair.flip(0)
                    ].clone()
    (out / "complete.json").write_text(
        json.dumps(
            {
                "seconds": time.time() - started,
                "phase": "partial_source_bank_only",
                "heldout_tasks_opened": False,
                "source_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "distinct_structures": len(
                    np.unique(np.concatenate(saved["permutations"]), axis=0)
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
