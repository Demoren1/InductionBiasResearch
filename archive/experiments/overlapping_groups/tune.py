"""Tune inner training on eight validation families, then seal one config."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .evaluate import fresh_observations
from .io import (
    DEFAULT_BANK,
    checked_load,
    load_manifest,
    save_json,
    save_torch,
    sha256,
    snapshot,
    source_hashes,
    validate_config,
)
from .io import PACKAGE as HERE
from .masks import structural_metrics
from .problem import reference
from .runtime import configure_cuda, cuda_devices, run_jobs
from .search import Fitness, templates

FAMILIES = [0, 3, 6, 9, 12, 15, 18, 21]


def grid():
    variants = [
        {"optimizer": "sgd", "lr": lr, "warmup": warmup}
        for lr, warmup in [(0.2, 8), (0.2, 32), (1.0, 32)]
    ]
    variants += [
        {"optimizer": "adam", "lr": lr, "warmup": warmup}
        for lr in [0.01, 0.03, 0.1]
        for warmup in [32, 200]
    ]
    for index, variant in enumerate(variants):
        variant["id"] = index
        variant["checkpoints"] = [
            step
            for step in [64, 128, 256, 384, 512, 1000]
            if step > variant["warmup"]
        ]
    return variants


def worker(args):
    torch.set_num_threads(1)
    config = json.loads(args.config.read_text())
    variant = grid()[args.variant]
    destination = (
        args.root / f"variant_{args.variant:02d}" / f"family_{args.family:02d}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.bank)
    controls = checked_load(args.cache / "training_controls.pt")
    if controls["bank_sha256"] != sha256(args.bank / "manifest.json"):
        raise ValueError("Calibration pool belongs to another bank")
    protocol = {
        "config": config,
        "variant": variant,
        "family": args.family,
        "bank_sha256": sha256(args.bank / "manifest.json"),
        "controls_sha256": sha256(args.cache / "training_controls.pt"),
        "source_hashes": source_hashes(),
    }
    saved = destination / "protocol.json"
    if saved.exists() and json.loads(saved.read_text()) != protocol:
        raise ValueError("Calibration worker has a different protocol")
    if not saved.exists() and (destination / "COMPLETE.json").exists():
        raise ValueError("Completed calibration has no protocol")
    save_json(saved, protocol)
    if (destination / "COMPLETE.json").exists():
        return
    device = configure_cuda(args.device, args.gpu_memory_gib)
    options = {
        **config["search"],
        "inner_optimizer": variant["optimizer"],
        "inner_lr": variant["lr"],
        "inner_warmup_steps": variant["warmup"],
        "inner_steps": 1000,
        "inner_adam_eps": 1e-6,
        "gradient_checkpoint_steps": 32,
    }
    data = fresh_observations(
        args.cache, manifest, args.family, [4, 5], device
    )
    template, seeds = templates(
        args.family,
        [4, 5],
        options["replicas"],
        config["seed"] + 1100000000,
        device,
    )
    masks = controls["bank_masks"].to(device)
    started = time.monotonic()
    fitness = Fitness(data, template, options)
    scores = fitness.trajectory(masks, variant["checkpoints"])
    target = reference(manifest["family_definitions"][args.family], "cpu")
    metrics, _ = structural_metrics(controls["bank_masks"], target)
    entropy = float(F.binary_cross_entropy(data["query"][1], data["query"][1]))
    exact = torch.tensor([metric["exact_mask"] for metric in metrics])
    if not exact.any():
        raise ValueError(
            "Calibration requires exact-mask candidates in the training pool"
        )
    results = []
    for steps, values in scores.items():
        values = values.cpu()
        selected = int(values.argmin())
        best_exact = float(values[exact].min())
        results.append(
            {
                "family": args.family,
                "variant": args.variant,
                "optimizer": variant["optimizer"],
                "lr": variant["lr"],
                "warmup": variant["warmup"],
                "steps": steps,
                "selected": selected,
                "query_excess_bce": float(values[selected]) - entropy,
                "best_exact_rank": int((values < best_exact).sum()) + 1,
                "oracle_excess_bce": best_exact - entropy,
                **metrics[selected],
            }
        )
    save_torch(
        destination / "scores.pt",
        {
            "scores": {key: value.cpu() for key, value in scores.items()},
            "seeds": seeds,
        },
    )
    save_json(destination / "results.json", results)
    save_json(
        destination / "COMPLETE.json",
        {
            "elapsed_seconds": time.monotonic() - started,
            "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(device)
            / 1024**3,
        },
    )


def report(root):
    rows = [
        row
        for path in sorted(root.glob("variant_*/family_*/results.json"))
        for row in json.loads(path.read_text())
    ]
    if len(list(root.glob("variant_*/family_*/COMPLETE.json"))) != len(
        FAMILIES
    ) * len(grid()):
        raise ValueError("Calibration is incomplete")
    summaries = []
    for variant in grid():
        for steps in variant["checkpoints"]:
            values = [
                row
                for row in rows
                if row["variant"] == variant["id"] and row["steps"] == steps
            ]
            summaries.append(
                {
                    **variant,
                    "steps": steps,
                    "families": len(values),
                    **{
                        key: float(np.mean([row[key] for row in values]))
                        for key in [
                            "exact_mask",
                            "iou",
                            "variable_iou",
                            "query_excess_bce",
                            "oracle_excess_bce",
                            "best_exact_rank",
                        ]
                    },
                }
            )
    ranked = sorted(
        summaries,
        key=lambda row: (
            -row["exact_mask"],
            -row["iou"],
            row["steps"],
            row["oracle_excess_bce"],
            row["id"],
        ),
    )
    winner = ranked[0]
    config = json.loads((root / "protocol.json").read_text())["base_config"]
    config["search"].update(
        {
            "inner_optimizer": winner["optimizer"],
            "inner_lr": winner["lr"],
            "inner_steps": winner["steps"],
            "inner_warmup_steps": winner["warmup"],
            "inner_adam_eps": 1e-6,
            "gradient_checkpoint_steps": 32,
        }
    )
    save_json(root / "selected_config.json", config)
    save_json(
        root / "summary.json",
        {
            "calibration_families": FAMILIES,
            "functions": [4, 5],
            "rows": rows,
            "ranked_configs": ranked,
            "selected": winner,
            "selection_rule": "Maximize exact-mask fraction, then IoU, then minimize steps; oracle loss breaks remaining ties. No functions 6+ used.",
        },
    )
    save_json(
        root / "selection_seal.json",
        {
            "config_sha256": sha256(root / "selected_config.json"),
            "functions_6_plus_accessed": False,
        },
    )
    print(
        json.dumps(
            {"selected": winner, "config": str(root / "selected_config.json")}
        ),
        flush=True,
    )


def launch(args):
    devices = cuda_devices(args.devices)
    args.root.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    validate_config(config)
    protocol = {
        "grid": grid(),
        "families": FAMILIES,
        "functions": [4, 5],
        "pool_sha256": sha256(args.cache / "training_controls.pt"),
        "bank_sha256": sha256(args.bank / "manifest.json"),
        "source_hashes": source_hashes(),
        "base_config": config,
    }
    saved = args.root / "protocol.json"
    if saved.exists() and json.loads(saved.read_text()) != protocol:
        raise ValueError("Existing calibration has a different protocol")
    save_json(saved, protocol)
    snapshot(args.root / "source", args.config)
    manifest = load_manifest(args.bank)
    available = {
        family["id"]
        for family in manifest["family_definitions"]
        if family["split"] == "train"
    }
    if not set(FAMILIES) <= available:
        raise ValueError(
            "The fixed calibration families are absent from this bank"
        )
    jobs = []
    for variant in grid():
        for family in FAMILIES:
            name = f"variant_{variant['id']:02d}_family_{family:02d}"
            destination = (
                args.root
                / f"variant_{variant['id']:02d}"
                / f"family_{family:02d}"
            )
            if (destination / "COMPLETE.json").exists():
                continue
            jobs.append(
                (
                    name,
                    "overlapping_groups.tune",
                    [
                        "--phase",
                        "worker",
                        "--root",
                        args.root,
                        "--variant",
                        variant["id"],
                        "--family",
                        family,
                        "--bank",
                        args.bank,
                        "--cache",
                        args.cache,
                        "--config",
                        args.config,
                        "--gpu-memory-gib",
                        args.gpu_memory_gib,
                    ],
                )
            )
    run_jobs(jobs, args.root, devices, args.workers_per_device)
    report(args.root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=["launch", "worker", "report"], default="launch"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=HERE / "configs/transfer.json"
    )
    parser.add_argument(
        "--devices", default=",".join(f"cuda:{i}" for i in range(8))
    )
    parser.add_argument("--workers-per-device", type=int, default=4)
    parser.add_argument("--gpu-memory-gib", type=float, default=8)
    parser.add_argument("--variant", type=int)
    parser.add_argument("--family", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    for name in ("root", "bank", "cache", "config"):
        setattr(args, name, getattr(args, name).resolve())
    if args.phase == "launch":
        launch(args)
    elif args.phase == "worker":
        if (
            args.variant not in range(len(grid()))
            or args.family not in FAMILIES
            or args.device is None
        ):
            parser.error(
                "Worker requires a valid variant, calibration family and CUDA device"
            )
        worker(args)
    else:
        report(args.root)
