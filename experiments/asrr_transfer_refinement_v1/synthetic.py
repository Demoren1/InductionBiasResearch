"""Compare hard groups and ratio-weighted groups on fresh linear tasks."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np
from sklearn.metrics import adjusted_rand_score

from experiments.asrr_benchmark_families_v1.gradient_groups import (
    gradient_kernel,
    gradient_refine,
    gradient_tree,
    orthogonal_groups,
)
from experiments.asrr_benchmark_families_v1.gradient_synthetic import (
    CONFIG,
    fit_msr,
    ridge_losses,
    row_profiles,
    tasks,
    teacher,
)
from experiments.paper_transfer_v2.report import confidence
from experiments.sharing_structure_research_v1.blind_contract import save_json
from experiments.srr_cluster_v1.clustering import centroid_fit

from .bases import local_basis, ward_profiles
from .common import OUT, SEED


def geometry(job):
    case, examples, i = job
    seed = SEED + 33793 * i
    true_groups, true_basis = teacher(seed, case)
    rank = true_basis.shape[1]
    source_x, source_y, _ = tasks(
        true_basis, CONFIG["source_tasks"], examples, seed + 1
    )
    count = max(CONFIG["target_shots"]) + CONFIG["query_size"]
    dev = tasks(true_basis, CONFIG["development_tasks"], count, seed + 2)
    weights = np.linalg.pinv(source_x) @ source_y
    profiles = row_profiles(weights)
    unit = profiles / np.linalg.norm(profiles, axis=1, keepdims=True).clip(
        1e-8
    )
    groups = {
        "ward": ward_profiles(profiles, rank),
        "lloyd": centroid_fit(profiles, rank, "lloyd", seed + 77, "cpu")[0],
        "angular": ward_profiles(unit, rank),
    }
    halves = examples // 2
    s = row_profiles(source_x[:, :halves].mT @ source_y[:, :halves]) / halves
    q = row_profiles(source_x[:, halves:].mT @ source_y[:, halves:]) / halves
    kernel = gradient_kernel(s, q)
    groups["gradient"] = gradient_refine(kernel, gradient_tree(kernel, rank))
    covariance = profiles @ profiles.T
    models = {key: [orthogonal_groups(ids)] for key, ids in groups.items()}
    models["ratio"] = [
        local_basis(covariance, ids, rank)
        for ids in (groups["angular"], groups["ward"])
    ]
    models["local_rank"] = [
        local_basis(signal, ward_profiles(profiles, rank // 2), rank)
        for signal in (covariance, kernel)
    ]
    models["msr"] = list(
        fit_msr(source_x, source_y, halves, rank, seed + 3).values()
    )
    models["oracle"] = [true_basis]
    scores = {}
    for shot in CONFIG["target_shots"]:
        for name, bank in models.items():
            losses = np.stack(
                [ridge_losses(dev, shot, value).mean(1) for value in bank]
            )
            candidate, penalty = np.unravel_index(
                losses.argmin(), losses.shape
            )
            scores[shot, name] = (candidate, penalty)
    test = tasks(true_basis, CONFIG["test_tasks"], count, seed + 4)
    return {
        "ari": {
            name: adjusted_rand_score(true_groups, ids)
            for name, ids in groups.items()
        },
        "mse": {
            f"{shot}_{name}": float(
                ridge_losses(test, shot, models[name][int(i)])[int(j)].mean()
            )
            for (shot, name), (i, j) in scores.items()
        },
    }


def run(workers):
    folder = OUT / "synthetic"
    folder.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for case, size in product(CONFIG["cases"], (12, 64)):
            records = list(
                pool.map(
                    geometry,
                    [(case, size, i) for i in range(CONFIG["geometries"])],
                )
            )
            means = {
                key: confidence([r["mse"][key] for r in records])
                for key in records[0]["mse"]
            }
            ari = {
                key: float(np.mean([r["ari"][key] for r in records]))
                for key in records[0]["ari"]
            }
            save_json(
                folder / f"{case}_{size}.json",
                {"mse": means, "ari": ari, "geometries": records},
            )
            print(
                case,
                size,
                {key: round(v["mean"], 4) for key, v in means.items()},
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    run(parser.parse_args().workers)
