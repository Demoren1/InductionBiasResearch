"""G3.1 registered codebook-direction ablations on already opened pilot cases."""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.sharing_benchmark_v1.evaluate import alignment

from .gaussian_stats import nested_observations
from .gradient_spectrum import BASE
from .projected_contrasts import crossfit_graph

VARIANTS = {
    "codebook_crossfit": {"union": False, "fallback": True},
    "codebook_crossfit_only": {"union": False, "fallback": False},
    "codebook_union_crossfit": {"union": True, "fallback": True},
}
DIRECTION_CONTROLS = {
    "random_direction_crossfit": {"direction_mode": "random"},
    "pca_direction_crossfit": {"direction_mode": "pca"},
    "whitened_pca_direction_crossfit": {"direction_mode": "whitened_pca"},
}


def register(root, direction_controls=False):
    variants = DIRECTION_CONTROLS if direction_controls else VARIANTS
    folder = "direction_controls" if direction_controls else "projected"
    destination = root / (
        "direction_controls_registration.json"
        if direction_controls
        else "projection_registration.json"
    )
    if destination.exists():
        raise RuntimeError("Ablation already registered")
    save_json(
        destination,
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "scope": "Exploratory ablations on every registered job of the supplied opened diagnostic cohort; not independent competitive confirmation.",
            "variants": variants,
            "main_variant_fixed_before_run": (
                "fixed_direction_controls_no_selection"
                if direction_controls
                else "codebook_crossfit"
            ),
            "purpose": "Reduce T-dimensional contrast noise using independently adapted C directions. Compare same nested observations, retain explicit abstention and exact-K stopping.",
            "theory_sha256": digest(BASE / "THEORY.md"),
            "executed_sha256": {
                name: digest(BASE / name)
                for name in (
                    "contrasts.py",
                    "projected_contrasts.py",
                    "projected_pilot.py",
                    "gaussian_stats.py",
                )
            },
        },
    )
    (root / folder).mkdir()


@torch.no_grad()
def run_job(root, job_id, plan, device, direction_controls=False):
    variants = DIRECTION_CONTROLS if direction_controls else VARIANTS
    folder = "direction_controls" if direction_controls else "projected"
    destination = root / folder / f"job_{job_id:03d}.json"
    if destination.exists():
        raise RuntimeError("Projection job already complete")
    job = plan["jobs"][job_id]
    family, records = job["family"]["id"], []
    first = {name: None for name in variants}
    started = time.perf_counter()
    observed = nested_observations(
        family,
        job["family"]["kind"],
        job["condition"],
        job["excitation"],
        job["noise"],
        plan,
        device,
    )
    for n, truth, total, halves in observed:
        arrays = {}
        for name, variant in variants.items():
            index, metadata, components = crossfit_graph(
                total, halves, n, plan, plan["seed"] + family + n, **variant
            )
            if index is not None:
                metadata["recovery"], aligned = alignment(
                    index, truth.cpu().numpy(), plan["codes"]
                )
                arrays[f"{name}_indices"] = index
                arrays[f"{name}_aligned_indices"] = aligned
                if first[name] is None:
                    first[name] = {"n": n, "recovery": metadata["recovery"]}
            arrays[f"{name}_components"] = components
            records.append(
                {
                    "n": n,
                    "method": name,
                    "graph": metadata,
                    "recovery": metadata.get("recovery"),
                }
            )
        np.savez_compressed(root / folder / f"job_{job_id:03d}_n{n}.npz", **arrays)
        print(
            job_id,
            n,
            {r["method"]: r["graph"]["components"] for r in records if r["n"] == n},
            flush=True,
        )
    save_json(
        destination,
        {
            "job": job,
            "records": records,
            "first_graph_return": first,
            "seconds": time.perf_counter() - started,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--direction-controls", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "register":
        register(args.root, args.direction_controls)
    else:
        plan = json.loads((args.root / "protocol.json").read_text())
        for job in range(args.shard, len(plan["jobs"]), args.shards):
            run_job(args.root, job, plan, args.device, args.direction_controls)


if __name__ == "__main__":
    main()
