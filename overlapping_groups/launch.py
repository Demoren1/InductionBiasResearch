"""Launch W/G decoder training or known-mask transfer on CUDA devices."""

import argparse
import json
import shutil
from pathlib import Path

import torch

from .io import (
    DEFAULT_BANK,
    PACKAGE,
    checked_load,
    load_decoder,
    load_manifest,
    save_torch,
    sha256,
    validate_config,
)
from .masks import canonical_mask
from .runtime import cuda_devices, run_jobs
from .search import budget


def training_controls(bank, manifest, destination, candidate_count):
    """A balanced pool from training-role IMP masks; no analytical masks."""
    bank_hash = sha256(bank / "manifest.json")
    if destination.exists():
        saved = checked_load(destination)
        if (
            saved["bank_sha256"] != bank_hash
            or saved["role"] != "train"
            or len(saved["bank_masks"]) != candidate_count
        ):
            raise ValueError(
                "Existing training controls have a different source"
            )
        return
    by_family = {}
    frequency = torch.zeros(48, 8)
    count = 0
    for name, task in manifest["completed_tasks"].items():
        if task["split"] != "train":
            continue
        card = checked_load(
            bank / task["path"] / "cards.pt", task["hashes"]["cards.pt"]
        )
        indices = [
            i
            for i, role in enumerate(card["network_roles"])
            if role == "train"
        ]
        masks = card["targets"][indices].float()
        frequency += masks.sum(0)
        count += len(indices)
        family = int(name.split("_")[1])
        by_family.setdefault(family, []).append(canonical_mask(masks))
    if not by_family:
        raise ValueError("No training-role IMP masks in bank")
    generator = torch.Generator().manual_seed(
        manifest["config"]["seed"] + 1500000000
    )
    pool, origins = [], []
    per_family, remainder = divmod(candidate_count, len(by_family))
    for index, (family, parts) in enumerate(sorted(by_family.items())):
        candidates = torch.cat(parts)
        needed = per_family + (index < remainder)
        permutations = [
            torch.randperm(len(candidates), generator=generator)
            for _ in range((needed + len(candidates) - 1) // len(candidates))
        ]
        chosen = (
            torch.cat(permutations)[:needed]
            if permutations
            else torch.empty(0, dtype=torch.long)
        )
        pool.append(candidates[chosen])
        origins.extend([family] * len(chosen))
    indices = frequency.flatten().argsort(descending=True, stable=True)[:32]
    fixed = torch.zeros(384).scatter_(0, indices, 1).reshape(48, 8)
    save_torch(
        destination,
        {
            "bank_sha256": bank_hash,
            "role": "train",
            "cards": count,
            "bank_masks": torch.cat(pool),
            "origin_families": origins,
            "frequency_mask": canonical_mask(fixed[None])[0],
            "scope": "Balanced training-role IMP pool, reshuffled with replacement when needed; fixed across target functions and decoders.",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("train", "evaluate"), required=True
    )
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument(
        "--vae-runs", type=Path, default=PACKAGE / "runs/vae_v2"
    )
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--devices", default=",".join(f"cuda:{i}" for i in range(8))
    )
    parser.add_argument("--families", type=int, nargs="+")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[4213, 4214, 4215]
    )
    parser.add_argument("--gpu-memory-gib", type=float, default=12)
    parser.add_argument("--workers-per-device", type=int, default=1)
    parser.add_argument("--cache-from", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    devices = cuda_devices(args.devices)
    if (
        len(set(args.seeds)) != len(args.seeds)
        or args.gpu_memory_gib <= 2
        or args.workers_per_device < 1
    ):
        parser.error(
            "Use distinct seeds, positive workers and a memory budget greater than 2 GiB"
        )
    args.bank, args.vae_runs, args.runs = (
        path.resolve() for path in (args.bank, args.vae_runs, args.runs)
    )
    config_path = (
        args.config
        or PACKAGE
        / (
            "configs/vae.json"
            if args.phase == "train"
            else "configs/transfer.json"
        )
    ).resolve()
    manifest = load_manifest(args.bank)
    jobs = []
    if args.phase == "train":
        if args.cache_from or args.families:
            parser.error("cache-from and families apply to evaluation only")
        for seed in args.seeds:
            name = f"wg_{seed}"
            jobs.append(
                (
                    name,
                    "overlapping_groups.train",
                    [
                        "--bank",
                        args.bank,
                        "--config",
                        config_path,
                        "--seed",
                        seed,
                        "--run",
                        args.runs / name,
                        "--gpu-memory-gib",
                        args.gpu_memory_gib,
                    ],
                )
            )
    else:
        config = json.loads(config_path.read_text())
        validate_config(config)
        available = {
            family["id"]
            for family in manifest["family_definitions"]
            if family["split"]
            == (
                "validation"
                if config.get("transfer_mode") == "composition"
                else "train"
            )
        }
        families = (
            args.families if args.families is not None else sorted(available)
        )
        if (
            len(set(families)) != len(families)
            or not set(families) <= available
        ):
            parser.error(
                "Use distinct family IDs allowed by the transfer mode"
            )
        bank_hash = sha256(args.bank / "manifest.json")
        for seed in args.seeds:
            load_decoder(
                args.vae_runs / f"wg_{seed}",
                bank_hash,
                seed,
                families,
                mode=config.get("transfer_mode", "known_mask"),
                manifest=manifest,
            )
        args.runs.mkdir(parents=True, exist_ok=True)
        if args.cache_from:
            cached = args.cache_from.resolve()
            if cached == args.runs:
                parser.error("cache-from must differ from the output root")
            # Old IMP caches fingerprint their original MLP implementation.
            observations = args.runs / "observations"
            if (
                not observations.exists()
                and (cached / "observations").is_dir()
            ):
                observations.symlink_to(
                    cached / "observations", target_is_directory=True
                )
            controls = args.runs / "training_controls.pt"
            if not controls.exists():
                shutil.copyfile(cached / "training_controls.pt", controls)
        training_controls(
            args.bank,
            manifest,
            args.runs / "training_controls.pt",
            budget(config["search"]),
        )
        for seed in args.seeds:
            for family in families:
                name = f"wg_{seed}_family_{family:02d}"
                jobs.append(
                    (
                        name,
                        "overlapping_groups.evaluate",
                        [
                            "--bank",
                            args.bank,
                            "--decoder-run",
                            args.vae_runs / f"wg_{seed}",
                            "--root",
                            args.runs,
                            "--run",
                            args.runs / f"wg_{seed}" / f"family_{family:02d}",
                            "--config",
                            config_path,
                            "--family",
                            family,
                            "--seed",
                            seed,
                            "--gpu-memory-gib",
                            args.gpu_memory_gib,
                        ],
                    )
                )
    run_jobs(jobs, args.runs, devices, args.workers_per_device)
    if args.phase == "evaluate":
        from .reporting.summary import report

        report(args.runs, args.bank)


if __name__ == "__main__":
    main()
