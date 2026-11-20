"""Check shared input groups against exact and nonuniform linear teachers."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
import torch
from scipy.cluster.hierarchy import cut_tree, linkage
from sklearn.metrics import adjusted_rand_score

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json

from .gradient_groups import (
    gradient_kernel,
    gradient_refine,
    gradient_tree,
    membership,
    orthogonal_groups,
)
from .groups import ROOT
from .linear import atomic_npz, tolerance_interval

GRADIENT_PLAN = Path(__file__).with_name("gradient_plan.json")
GRADIENT_CONFIG = json.loads(GRADIENT_PLAN.read_text())
OUT = ROOT / "gradient_asrr"
CONFIG = GRADIENT_CONFIG["synthetic"]


def teacher(seed, case):
    rng = np.random.default_rng(seed)
    sizes = CONFIG["group_sizes"]
    labels = rng.permutation(np.repeat(np.arange(len(sizes)), sizes))
    basis = orthogonal_groups(labels)
    if case == "scaled":
        basis *= np.exp(rng.normal(0, 0.85, size=len(labels)))[:, None]
        basis /= np.linalg.norm(basis, axis=0)
    return labels, basis


def tasks(basis, count, points, seed):
    rng = np.random.default_rng(seed)
    feature = rng.normal(size=(count, points, CONFIG["inputs"]))
    codes = rng.normal(size=(count, basis.shape[1], CONFIG["outputs"]))
    clean = (feature @ basis) @ codes / np.sqrt(basis.shape[1])
    observed = clean + CONFIG["noise"] * rng.normal(size=clean.shape)
    return feature, observed, clean


def row_profiles(weights):
    return weights.transpose(1, 0, 2).reshape(weights.shape[1], -1)


def fit_msr(x, y, split, rank, seed):
    """Train a continuous factor through one SGD update on each source task."""
    torch.set_num_threads(1)
    rng = torch.Generator().manual_seed(seed)
    x, y = (torch.from_numpy(array.astype(np.float32)) for array in (x, y))
    basis = torch.linalg.qr(torch.randn(x.shape[-1], rank, generator=rng)).Q
    initial = torch.zeros(rank, y.shape[-1], requires_grad=True)
    rate = torch.tensor(-1.0, requires_grad=True)
    basis.requires_grad_()
    optimizer = torch.optim.Adam([basis, initial, rate], lr=0.004)
    checkpoints = {}
    for step in range(1, max(CONFIG["msr_checkpoints"]) + 1):
        ids = torch.randint(len(x), (32,), generator=rng)
        support = x[ids, :split] @ basis
        query = x[ids, split:] @ basis
        gradient = support.mT @ (support @ initial - y[ids, :split]) / split
        updated = initial - rate.sigmoid() * gradient
        error = (query @ updated - y[ids, split:]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        error.backward()
        optimizer.step()
        if step in CONFIG["msr_checkpoints"]:
            checkpoints[f"msr_{step}"] = np.linalg.qr(
                basis.detach().numpy().astype(np.float64)
            )[0]
    return checkpoints


def ridge_losses(panel, shots, basis):
    x, y, clean = panel
    split = max(CONFIG["target_shots"])
    support = x[:, :shots] @ basis
    query = x[:, split:] @ basis
    gram = support.mT @ support
    rhs = support.mT @ y[:, :shots]
    eyes = np.eye(gram.shape[-1])
    errors = []
    for penalty in CONFIG["ridge"]:
        coefficient = np.linalg.solve(gram + penalty * eyes, rhs)
        errors.append(np.square(query @ coefficient - clean[:, split:]).mean((1, 2)))
    return np.array(errors)


def run_geometry(job):
    case, size, replicate = job
    seed = GRADIENT_CONFIG["seed"] + replicate * 16519
    truth, basis = teacher(seed, case)
    shots, rank = CONFIG["target_shots"], basis.shape[1]
    count = max(shots) + CONFIG["query_size"]
    maximum = max(CONFIG["samples_per_source_half"])
    xs, ys, _ = tasks(basis, CONFIG["source_tasks"], 2 * maximum, seed + 17)
    x, y = (
        np.concatenate((a[:, :size], a[:, maximum : maximum + size]), axis=1)
        for a in (xs, ys)
    )
    dev = tasks(basis, CONFIG["development_tasks"], count, seed + 43)
    dense = np.linalg.pinv(x) @ y
    baseline = cut_tree(linkage(row_profiles(dense), method="ward"), rank).ravel()
    supports = x[:, :size].mT @ y[:, :size] / size
    queries = x[:, size:].mT @ y[:, size:] / size
    kernel = gradient_kernel(row_profiles(supports), row_profiles(queries))
    partitions = {
        "ward": baseline,
        "gradient_tree": gradient_refine(kernel, gradient_tree(kernel, rank)),
        "gradient_refined": gradient_refine(kernel, baseline),
    }
    candidates = {
        "ward": membership(baseline),
        "normalized": orthogonal_groups(baseline),
        **{k: orthogonal_groups(v) for k, v in partitions.items() if k != "ward"},
        **fit_msr(x, y, size, rank, seed + 285),
        "oracle": basis,
    }
    methods = {
        "ward": ("ward",),
        "normalized": ("normalized",),
        "gradient": ("gradient_tree", "gradient_refined"),
        "msr": tuple(k for k in candidates if k.startswith("msr_")),
        "oracle": ("oracle",),
    }
    choices = {}
    for shot in shots:
        scores = {
            key: ridge_losses(dev, shot, value).mean(1)
            for key, value in candidates.items()
        }
        choices[shot] = {
            method: min(
                ((key, j) for key in names for j in range(len(CONFIG["ridge"]))),
                key=lambda pair: scores[pair[0]][pair[1]],
            )
            for method, names in methods.items()
        }
    test = tasks(basis, CONFIG["test_tasks"], count, seed + 119)
    fitted = {}
    for shot, selected in choices.items():
        fitted[shot] = {
            kind: ridge_losses(test, shot, candidates[name])[row]
            for kind, (name, row) in selected.items()
        }
    agreement = {
        key: adjusted_rand_score(truth, value) for key, value in partitions.items()
    }
    return {
        "replicate": replicate,
        "choices": choices,
        "ari": agreement,
        "mse": fitted,
        "geometry": {"truth": truth, **partitions},
    }


def run(workers):
    folder = OUT / "synthetic"
    folder.mkdir(parents=True, exist_ok=True)
    save_json(
        folder / "registration.json",
        {
            "plan_sha256": digest(GRADIENT_PLAN),
            "code_sha256": digest(Path(__file__)),
            "config": CONFIG,
            "protocol": "Nested source budgets, common test curves per geometry, no teacher input in the learnable models.",
        },
    )
    summary, arrays = {}, {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for case, size in product(CONFIG["cases"], CONFIG["samples_per_source_half"]):
            runs = list(
                pool.map(
                    run_geometry, [(case, size, i) for i in range(CONFIG["geometries"])]
                )
            )
            key = f"{case}_{2 * size}"
            summary[key] = {}
            for shot in CONFIG["target_shots"]:
                selected_ari = [
                    r["ari"][r["choices"][shot]["gradient"][0]] for r in runs
                ]
                mse = {
                    k: np.stack([r["mse"][shot][k] for r in runs])
                    for k in runs[0]["mse"][shot]
                }
                summary[key][str(shot)] = {
                    "mse": {k: float(v.mean()) for k, v in mse.items()},
                    "gain_mse_95_ci": {
                        k: tolerance_interval((v - mse["gradient"]).mean(1))
                        for k, v in mse.items()
                        if k != "gradient"
                    },
                    "ari": {
                        "ward": float(np.mean([r["ari"]["ward"] for r in runs])),
                        "gradient": float(np.mean(selected_ari)),
                    },
                    "exact": {
                        "ward": sum(r["ari"]["ward"] == 1 for r in runs),
                        "gradient": sum(a == 1 for a in selected_ari),
                    },
                    "choices": [r["choices"][shot] for r in runs],
                }
                arrays.update({f"{key}_{shot}_{m}": v for m, v in mse.items()})
            arrays.update(
                {
                    f"{key}_groups_{m}": np.stack([row["geometry"][m] for row in runs])
                    for m in runs[0]["geometry"]
                }
            )
            print(key, json.dumps(summary[key]), flush=True)
    save_json(folder / "results.json", summary)
    atomic_npz(folder / "per_task.npz", **arrays)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    run(parser.parse_args().workers)
