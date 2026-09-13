from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from pattern.length_interp.bank import train_one_pattern

from .common import (atomic_save, bank_path, load_protocol, pair_dir, seed_for,
                     sha256_file, sha256_tensor, write_json)
from .model import FixedVAE, vae_loss


def configure(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_bank_shard(out: Path, shard: int, shards: int, device: str) -> None:
    config, protocol = load_protocol(out)
    patterns = protocol["task_split"]["train_patterns"]
    for pattern in patterns[shard::shards]:
        result = train_one_pattern(config, out, pattern, device)
        print(f"[k5-bank] {pattern} -> {result}", flush=True)


def _load_maps(out: Path, patterns: list[str]) -> tuple[torch.Tensor, list[dict]]:
    parts, provenance = [], []
    for pattern in patterns:
        path = bank_path(out, pattern)
        record = torch.load(path, map_location="cpu", weights_only=True)
        maps = record["selected"]["importance"].float().reshape(-1, 1024)
        parts.append(maps)
        provenance.append({"pattern": pattern, "count": len(maps), "sha256": sha256_file(path)})
    return torch.cat(parts), provenance


def _stratified_partition(out: Path, patterns: list[str], provenance: list[dict], fraction: float,
                          seed: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
    train, val, cursor = [], [], 0
    per_pattern = {}
    for item in provenance:
        count = item["count"]
        generator = torch.Generator().manual_seed(seed_for("map-split", seed, item["pattern"]))
        order = torch.randperm(count, generator=generator)
        n_val = max(1, round(count * fraction))
        val.append(order[:n_val] + cursor)
        train.append(order[n_val:] + cursor)
        per_pattern[item["pattern"]] = {"train": count - n_val, "val": n_val}
        cursor += count
    train_idx, val_idx = torch.cat(train), torch.cat(val)
    details = {
        "seed": seed, "fraction": fraction, "per_pattern": per_pattern,
        "train_indices_sha256": sha256_tensor(train_idx),
        "val_indices_sha256": sha256_tensor(val_idx),
    }
    return train_idx, val_idx, details


@torch.no_grad()
def _validation(model: FixedVAE, values: torch.Tensor, beta: float, batch: int,
                device: torch.device) -> tuple[float, float, float]:
    totals = torch.zeros(3, dtype=torch.float64)
    seen = 0
    model.eval()
    for start in range(0, len(values), batch):
        target = values[start:start + batch].to(device)
        mu, logvar = model.encode(target)
        logits = model.decode(mu)
        metrics = vae_loss(logits, target, mu, logvar, beta)
        totals += torch.tensor([float(value) * len(target) for value in metrics])
        seen += len(target)
    return tuple(float(value / seen) for value in totals)


@torch.no_grad()
def _noncollapse(model: FixedVAE, device: torch.device, seed: int) -> dict:
    generator = torch.Generator(device=device).manual_seed(seed_for("noncollapse", seed))
    latent = torch.randn(256, model.latent_dim, generator=generator, device=device)
    probabilities = torch.sigmoid(model.decode(latent))
    return {
        "prior_samples": 256,
        "probability_mean": float(probabilities.mean()),
        "probability_std": float(probabilities.std()),
        "mean_feature_std": float(probabilities.std(dim=0).mean()),
        "max_feature_std": float(probabilities.std(dim=0).max()),
    }


def train_vae(out: Path, seed: int, device_name: str) -> Path:
    config, protocol = load_protocol(out)
    destination = pair_dir(out, next(pair for pair in config.pairs if seed in pair)) / f"vae_{seed}"
    checkpoint_path, metadata_path = destination / "best.pt", destination / "metadata.json"
    if checkpoint_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("checkpoint_sha256") != sha256_file(checkpoint_path):
            raise ValueError(f"checkpoint provenance mismatch: {checkpoint_path}")
        print(f"[k5-vae] seed={seed}: verified existing checkpoint", flush=True)
        return checkpoint_path
    if checkpoint_path.exists() or metadata_path.exists() or (destination.exists() and any(destination.iterdir())):
        raise FileExistsError(f"partial VAE output; refusing to overwrite {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    patterns = protocol["task_split"]["train_patterns"]
    maps, provenance = _load_maps(out, patterns)
    train_idx, val_idx, partition = _stratified_partition(
        out, patterns, provenance, config.vae_val_fraction, config.map_split_seed)
    train_values, val_values = maps[train_idx], maps[val_idx]
    generator = torch.Generator().manual_seed(config.map_split_seed)
    loader = DataLoader(TensorDataset(train_values), batch_size=config.vae_batch, shuffle=True,
                        generator=generator)
    device = torch.device(device_name)
    configure(seed)
    model = FixedVAE(config.mask_dim, config.latent_dim, config.vae_hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.vae_lr)
    best, best_epoch, history = float("inf"), 0, []
    started = time.monotonic()
    for epoch in range(1, config.vae_epochs + 1):
        model.train()
        sums = torch.zeros(3, dtype=torch.float64)
        seen = 0
        for (target_cpu,) in loader:
            target = target_cpu.to(device)
            logits, mu, logvar = model(target)
            values = vae_loss(logits, target, mu, logvar, config.vae_beta)
            optimizer.zero_grad(set_to_none=True)
            values[0].backward()
            optimizer.step()
            sums += torch.tensor([float(value.detach()) * len(target) for value in values])
            seen += len(target)
        validation = _validation(model, val_values, config.vae_beta, config.vae_batch, device)
        row = {"epoch": epoch, "train_total": float(sums[0] / seen),
               "train_reconstruction": float(sums[1] / seen), "train_kl": float(sums[2] / seen),
               "val_total": validation[0], "val_reconstruction": validation[1], "val_kl": validation[2]}
        history.append(row)
        if validation[0] < best:
            best, best_epoch = validation[0], epoch
            atomic_save(checkpoint_path, {
                "model_config": {"mask_dim": config.mask_dim, "latent_dim": config.latent_dim,
                                 "hidden": config.vae_hidden},
                "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "seed": seed, "epoch": epoch, "validation": validation,
            })
        if epoch == 1 or epoch % 10 == 0 or epoch == config.vae_epochs:
            print(f"[k5-vae] seed={seed} epoch={epoch}/{config.vae_epochs} "
                  f"train={row['train_total']:.4f} val={validation[0]:.4f} best={best:.4f}", flush=True)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True)["model_state"])
    metadata = {
        "seed": seed, "best_epoch": best_epoch, "best_val_loss": best,
        "checkpoint_sha256": sha256_file(checkpoint_path), "elapsed_seconds": time.monotonic() - started,
        "model_config": {"mask_dim": config.mask_dim, "latent_dim": config.latent_dim,
                         "hidden": config.vae_hidden},
        "training": {"epochs": config.vae_epochs, "batch": config.vae_batch, "lr": config.vae_lr,
                     "beta": config.vae_beta, "loss": "BCE sum per map + beta*KL",
                     "checkpoint_selection": "deterministic posterior-mean map validation"},
        "maps": {"count": len(maps), "tensor_sha256": sha256_tensor(maps),
                 "bank_files": provenance, "partition": partition,
                 "train_count": len(train_values), "val_count": len(val_values)},
        "noncollapse": _noncollapse(model, device, seed), "history": history,
    }
    write_json(metadata_path, metadata)
    print(f"[k5-vae] seed={seed}: best epoch={best_epoch}, val={best:.4f}", flush=True)
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    sub = parser.add_subparsers(dest="command", required=True)
    bank = sub.add_parser("bank")
    bank.add_argument("--shard", type=int, required=True)
    bank.add_argument("--shards", type=int, required=True)
    vae = sub.add_parser("vae")
    vae.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.command == "bank":
        train_bank_shard(args.out, args.shard, args.shards, args.device)
    else:
        train_vae(args.out, args.seed, args.device)


if __name__ == "__main__":
    main()
