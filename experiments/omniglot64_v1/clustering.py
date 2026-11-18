"""Compare fixed input clusterers on the shared width-64 Omniglot episodes."""

import argparse
import time

import numpy as np
import torch

from experiments.asrr_tuning_v1.study import run as run_fits
from experiments.paper_asrr_v1.extract import source_profiles
from experiments.paper_structure_benchmark_v1.protocol.real_boundary import (
    StudyBoundary,
)
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from experiments.srr_cluster_v1.clustering import VENDOR, build_bank, source_objective

from .source import BANKS, CODES, HIDDEN, INPUTS, OUTPUT, ROOT, read_json, save_groups
from .transfer import CHOICES, PLAN, score, write_target_queues

CLUSTERERS = {
    "asrr_lloyd": "lloyd",
    "asrr_emeans": "emeans",
    "asrr_average": "average",
    "asrr_complete": "complete",
    "asrr_single": "single",
}
CLUSTER_SEED = 3511159150
PREFIX = "clustering_"
REGISTRATION = OUTPUT / "clustering_registration.json"
DENSE = ROOT / "source/dense/models.npz"


def register():
    if REGISTRATION.exists():
        raise FileExistsError(REGISTRATION)
    original = read_json(OUTPUT / "registration.json")
    dense_hash = digest(DENSE)
    if dense_hash != read_json(DENSE.with_name("metrics.json"))["models_sha256"]:
        raise ValueError("Changed width-64 Dense source fits")
    save_json(
        REGISTRATION,
        {
            "methods": CLUSTERERS,
            "reference": read_json(CHOICES)["asrr"],
            "source_tasks": original["source_tasks"],
            "source_archive_sha256": dense_hash,
            "input_profiles": "Aligned average over retained Dense fits",
            "profile_shape": [INPUTS, len(original["source_tasks"]) * HIDDEN],
            "input_clusters": CODES // HIDDEN,
            "hidden": HIDDEN,
            "codes": CODES,
            "centroid_seed": CLUSTER_SEED,
            "centroid_selection": "Archived source-only SSE rule, including empty-group repair",
            "lloyd": {"initializations": 4, "max_updates": 25},
            "emeans_recipes": [["auto", 5], ["quality", 5], ["quality", 25]],
            "emeans_vendor_sha256": {
                name: digest(VENDOR / name)
                for name in ("__init__.py", "e_means_quantized.py", "e_means_torch.py")
            },
            "signed_lloyd": "Aliases Lloyd for the unsigned shared-input scope",
            "numeric_rule": original["numeric_rule"],
            "target_plan_sha256": digest(PLAN),
        },
    )


def construct(name):
    config = read_json(REGISTRATION)
    method = CLUSTERERS[name]
    if (
        method != config["methods"][name]
        or digest(DENSE) != config["source_archive_sha256"]
    ):
        raise ValueError("Changed clustering setup")
    if (BANKS / f"{name}.npz").exists():
        raise FileExistsError(BANKS / f"{name}.npz")
    torch.set_num_threads(2)
    boundary = StudyBoundary(ROOT)
    boundary.install()
    started = time.perf_counter()
    _, profiles, _, _ = source_profiles(DENSE, INPUTS, HIDDEN)
    clustering_started = time.perf_counter()
    banks, records = build_bank(
        profiles, method, (CODES,), ("input",), seed=config["centroid_seed"]
    )
    clustering_seconds = time.perf_counter() - clustering_started
    index = banks[f"{method}_input_m{CODES}"]
    _, labels, sizes = np.unique(index[:, 0], return_inverse=True, return_counts=True)
    flattened = profiles.transpose(1, 0, 2).reshape(INPUTS, -1)
    sse, _ = source_objective(flattened, labels, np.ones(INPUTS), CODES // HIDDEN)
    save_groups(
        name,
        [
            (
                name,
                index,
                {
                    "clusterer": method,
                    "input_clusters": len(sizes),
                    "largest_input_cluster": int(sizes.max()),
                    "singleton_input_clusters": int((sizes == 1).sum()),
                    "profile_sse_per_input": sse,
                    "clustering_seconds": clustering_seconds,
                    "optimizer": records[0]["clustering"][0],
                },
            )
        ],
        started,
        None,
    )
    boundary.save(OUTPUT / f"{name}_source_access.json", "complete")


def freeze():
    if digest(PLAN) != read_json(REGISTRATION)["target_plan_sha256"]:
        raise ValueError("Changed shared held-out panels")
    if (OUTPUT / f"{PREFIX}target_jobs.json").exists():
        raise FileExistsError("Clustering policies have already been frozen")
    policies = {}
    for name in CLUSTERERS:
        row = read_json(OUTPUT / f"{name}_source.json")["candidates"][0]
        choice = {key: row[key] for key in ("name", "model_sha256")}
        if digest(BANKS / f"{name}.npz") != choice["model_sha256"]:
            raise ValueError(f"Changed input grouping: {name}")
        save_json(OUTPUT / f"choices_{name}.json", choice)
        policies[name] = choice
    write_target_queues(policies, read_json(PLAN), prefix=PREFIX)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("register", "construct", "freeze", "benchmark", "score")
    )
    parser.add_argument("--method", choices=tuple(CLUSTERERS))
    parser.add_argument("--gpus", type=int, nargs="+", default=[7])
    args = parser.parse_args()
    if args.action == "register":
        register()
    elif args.action == "construct":
        if args.method is None:
            parser.error("construct requires --method")
        construct(args.method)
    elif args.action == "freeze":
        freeze()
    elif args.action == "benchmark":
        run_fits(
            f"{PREFIX}target",
            args.gpus,
            root=ROOT,
            output=OUTPUT,
            timing_first=(f"{PREFIX}timing", args.gpus[-1]),
        )
    else:
        score(added_methods=tuple(CLUSTERERS), result_file="clustering_results.json")
