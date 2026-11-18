"""Prepare the shared sources for the compact manuscript comparison."""

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.paper_structure_benchmark_v1.methods.real_competitors import (
    run as fit_competitor,
)
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)
from experiments.wsr_omniglot_v1.source import (
    PANELS as OMNIGLOT_PANELS,
    REGISTRATION as OMNIGLOT_REGISTRATION,
    read_json,
)


OUTPUT = Path("artifacts/compact_experiments_2026-11-18")
ROOT = OUTPUT.parent / "local" / OUTPUT.name
OMNIGLOT = ROOT / "omniglot_source"
HAR = ROOT.parent / "paper_asrr_2026-11-17/har"
SYNTHETIC = ROOT.parent / "paper_asrr_2026-11-17/synthetic_4"
SYNTHETIC_ORIGINAL = OUTPUT.parent / "single_task_recovery_2026-11-14"
TUNING = ROOT.parent / "asrr_tuning_2026-11-17"
NATIVE_STEPS = {"lutq": 2500, "yeh": 400, "msr": 2000}
BUDGETS = {"synthetic": 4, "omniglot10": 1024, "har3": 128}


def prepare():
    config = read_json(OMNIGLOT_REGISTRATION)
    original = ROOT.parent / "omniglot_single_task_2026-11-14/study"
    panels = {}
    for name, expected in config["source_panel_sha256"].items():
        path = OMNIGLOT_PANELS / f"{name}.npz"
        if digest(path) != expected:
            raise ValueError(f"Changed Omniglot source data: {path}")
        with np.load(path) as data:
            panels[name] = dict(data)
    tasks = panels["anchors"]["task_ids"].tolist()
    if tasks != config["source_tasks"]:
        raise ValueError("Omniglot source tasks differ from WSR and aSRR")
    OUTPUT.mkdir(parents=True)
    (OMNIGLOT / "public").mkdir(parents=True)
    (OMNIGLOT / "source/dense").mkdir(parents=True)
    for name in ("support", "numerical"):
        np.savez_compressed(
            OMNIGLOT / "public" / f"{name}.npz", **panels[name]
        )
    with np.load(original / "public/structure.npz") as data:
        np.savez_compressed(
            OMNIGLOT / "public/structure.npz",
            x=data["x"][tasks],
            y=data["y"][tasks],
        )
    np.savez_compressed(
        OMNIGLOT / "source/dense/models.npz",
        parameters=panels["anchors"]["parameters"],
    )
    protocol = read_json(original / "protocol.json")
    save_json(
        OMNIGLOT / "protocol.json",
        {
            key: protocol[key]
            for key in ("inputs", "hidden", "outputs", "loss")
        },
    )
    save_json(
        OUTPUT / "registration.json",
        {
            "datasets": ["synthetic case 04", "Omniglot", "UCI HAR"],
            "methods": [
                "dense",
                "g34",
                "asrr",
                "lutq",
                "yeh_map",
                "yeh_soft",
                "msr",
                "wsr",
            ],
            "budgets": BUDGETS,
            "omniglot_source_tasks": tasks,
            "omniglot_source_sha256": {
                name: digest(OMNIGLOT / "public" / f"{name}.npz")
                for name in ("support", "numerical", "structure")
            },
            "omniglot_native_steps": NATIVE_STEPS,
            "omniglot_native_selection": "At most eight H-ranked candidates, selected on development",
            "omniglot_target_episodes": "Same 1024 paired episodes as initial WSR comparison",
            "har_target_episodes": "Two fresh cohorts of 90, paired by subject",
            "synthetic_cases": [4],
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", *NATIVE_STEPS))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        if args.action == "yeh":
            torch.backends.cuda.preferred_linalg_library("magma")
        fit_competitor(
            OMNIGLOT,
            args.action,
            BUDGETS["omniglot10"],
            torch.device(args.device),
            NATIVE_STEPS[args.action],
        )
