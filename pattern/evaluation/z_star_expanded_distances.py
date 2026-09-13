"""Recompute task-z distances to all exact oracle latents found in the radius sweep.

The primary experiment originally measured distance to the exact solutions found
with radius 12.  Those solutions lie almost entirely on that boundary.  This
post-hoc analysis uses the union of exact solutions found at radii
1, 2, 3, 4, 6, and 12 for each frozen decoder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PATTERN_ROOT = HERE.parent
sys.path.insert(0, str(PATTERN_ROOT))

from evaluation.z_star_noise import ci, load_protocol, sha256_file, write_json  # noqa: E402


RADII = (1.0, 2.0, 3.0, 4.0, 6.0, 12.0)
STAGES = {"initial": "initial_z", "final": "final_z", "best_query": "best_val_z"}


def oracle_path(out: Path, seed: int, radius: float) -> Path:
    if radius == 12:
        return out / f"seed_{seed}/oracle.pt"
    return out / f"seed_{seed}/radius_sweep/radius_{radius:g}.pt"


def exact_references(out: Path, seed: int) -> tuple[torch.Tensor, dict]:
    pieces = []
    counts = {}
    hashes = {}
    for radius in RADII:
        path = oracle_path(out, seed, radius)
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        exact = artifact["best_soft"]["iou"] == 1
        pieces.append(artifact["best_soft"]["z"][exact].float())
        counts[f"{radius:g}"] = int(exact.sum())
        hashes[str(path)] = sha256_file(path)
    references = torch.cat(pieces)
    if not len(references):
        raise RuntimeError(f"seed {seed}: no exact oracle references")
    return references, {"count_by_radius": counts, "total_count": len(references),
                        "artifact_sha256": hashes}


def mean_nearest(codes: torch.Tensor, references: torch.Tensor) -> float:
    return float(torch.cdist(codes.float(), references.float()).min(dim=1).values.mean())


def analyze_seed(out: Path, seed: int, patterns: list[str], groups: list[str]) -> dict:
    references, provenance = exact_references(out, seed)
    setup_path = out / f"seed_{seed}/initials.pt"
    setup = torch.load(setup_path, map_location="cpu", weights_only=True)
    result = {
        "model_seed": seed,
        "reference_set": provenance,
        "initials_sha256": sha256_file(setup_path),
        "groups": {},
    }
    tasks = {
        pattern: torch.load(out / f"seed_{seed}/task_{pattern}.pt",
                            map_location="cpu", weights_only=True)
        for pattern in patterns
    }
    for group_name in groups:
        group = setup["groups"][group_name]
        start, stop = group["start"], group["stop"]
        result["groups"][group_name] = {}
        for stage, key in STAGES.items():
            per_task = {
                pattern: mean_nearest(task[key][start:stop], references)
                for pattern, task in tasks.items()
            }
            result["groups"][group_name][stage] = {
                "mean_nearest_exact_distance": float(np.mean(list(per_task.values()))),
                "per_task": per_task,
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    settings, protocol = load_protocol(out)
    patterns = list(protocol["heldout_patterns"])
    groups = ["prior", *(f"noise_{radius:g}" for radius in settings.noise_radii)]
    per_seed = [analyze_seed(out, seed, patterns, groups) for seed in settings.model_seeds]

    aggregate = {}
    for group_name in groups:
        aggregate[group_name] = {}
        for stage in STAGES:
            values = [row["groups"][group_name][stage]["mean_nearest_exact_distance"]
                      for row in per_seed]
            aggregate[group_name][stage] = ci(values)

    payload = {
        "description": "Distances to the finite union of exact oracle latents found per decoder",
        "reference_radii": list(RADII),
        "distance_interpretation": (
            "Nearest distance to this finite reference set is an upper bound on distance to "
            "the decoder's full ideal-mask preimage."
        ),
        "independence_unit": "frozen VAE seed; tasks and starts are nested within seed",
        "patterns": patterns,
        "aggregate": aggregate,
        "per_seed": per_seed,
        "source_sha256": sha256_file(__file__),
    }
    destination = out / "expanded_reference_distances.json"
    write_json(destination, payload)
    print(json.dumps({group: {stage: round(row[stage]["mean"], 6)
                              for stage in STAGES}
                      for group, row in aggregate.items()}, indent=2))
    print(f"saved: {destination}")


if __name__ == "__main__":
    main()
