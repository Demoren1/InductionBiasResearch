"""Check nearest-code search from a fully unrestricted matrix."""

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from .data import BASE, digest, save_json
from .sharing_ablation import fresh
from .sharing_geometry import run_phase


def prepare(parent, output):
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((parent / "protocol.json").read_text())
    spec = json.loads((parent / "specification.json").read_text())
    spec.update(name="deepsets_sharing_full", full_connectivity_search=True)
    plan["search"].update(
        pilot_seeds=[161, 162],
        confirmation_seeds=[163, 164, 165],
        pilot_steps=12000,
        confirmation_steps=12000,
        restarts=1,
        numeric_warmup=0,
        top_configurations=4,
    )
    plan["sharing_heldout_offset"] = 7300000
    plan["support_file"] = "search_support.npz"
    np.savez_compressed(
        output / "search_support.npz",
        mask=np.ones((spec["inputs"], spec["hidden"]), dtype=bool),
    )
    for name in ("source.npz", "teacher_audit_only.npz"):
        shutil.copy2(parent / name, output / name)
    save_json(output / "protocol.json", plan)
    save_json(output / "specification.json", spec)
    candidates = []
    for strength in (0.0, 0.0001, 0.001, 0.01):
        for rate in (0.003, 0.01):
            candidates.append(
                {
                    "configuration": len(candidates),
                    "family": f"full_ste_l1_{strength:g}",
                    "method": "nearest_ste",
                    "warmup": 0,
                    "numeric_lr": rate,
                    "architecture_lr": 0.03,
                    "latent_lr": rate,
                    "radius": 0.5,
                    "commitment": 0.01,
                    "latent_l1": strength,
                    "mode": "joint",
                    "exact_forward_assignment": True,
                }
            )
    snapshot = output / "code_snapshot"
    snapshot.mkdir()
    for name in (
        "sharing_full.py",
        "sharing_geometry.py",
        "sharing_ablation.py",
        "train.py",
        "model.py",
        "run.py",
        "data.py",
    ):
        shutil.copy2(BASE / name, snapshot / name)
    save_json(
        output / "registration.json",
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "status": "Full-matrix extension after conditional source validation",
            "candidates": candidates,
            "rules": [
                "All256 positions are allowed. No learned or teacher mask enters this run.",
                "Random latent weights, generic empirical quantile centers, no numerical pretraining, no k-means.",
                "Nearest signed code in forward, identity straight-through gradient for latent weights and grouped gradient for C.",
                "Compare normalized latent L1 weights0/1e-4/1e-3/1e-2; numerical rates0.003/0.01.",
                "Stored indexator is exactly the forward nearest-code assignment; canonical mass transfer is an identity for this one-hot distribution.",
                "Same12000 search steps and3000 hard refit steps per candidate. One inherited and three fresh C in source ranking.",
                "Choose rate per L1 family on pilot161/162, confirm on163/164/165, freeze before teacher and new heldout.",
                "Teacher is used only after source selection. No software tests.",
            ],
            "source_sha256": digest(output / "source.npz"),
            "snapshot_sha256": {p.name: digest(p) for p in snapshot.glob("*.py")},
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage", required=True, choices=("prepare", "pilot", "confirmation", "fresh")
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == "prepare":
        prepare(args.parent, args.output)
    elif args.stage == "fresh":
        fresh(args.output, torch.device(args.device))
    else:
        run_phase(args.output, args.stage, torch.device(args.device))


if __name__ == "__main__":
    main()
