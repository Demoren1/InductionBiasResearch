"""Train the scalar-length CVAE on top-10% unsigned MLP importance maps.

The protocol deliberately restricts CVAE data to patterns at lengths 3, 4, 6,
and 8.  Lengths 5 and 7 are interpolation-only test lengths and this module
fails closed if a bank at either held-out length is supplied as train or
validation data.

Run with ``python -m pattern.length_interp.train_cvae --out OUTPUT --seed 42
--device cuda`` after the MLP bank stage has completed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .cvae import LengthCVAE, cvae_loss


TRAIN_LENGTHS = frozenset((3, 4, 6, 8))
HELD_INTERPOLATION_LENGTHS = frozenset((5, 7))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _length(pattern: str) -> int:
    if not isinstance(pattern, str) or not pattern or set(pattern) - {"0", "1"}:
        raise ValueError(f"expected a nonempty binary pattern, got {pattern!r}")
    return len(pattern)


def validate_cvae_patterns(pattern_splits: dict[str, list[str] | tuple[str, ...]]) -> tuple[list[str], list[str]]:
    """Return train/val patterns, rejecting any length leakage into the CVAE."""
    try:
        train, val = list(pattern_splits["train"]), list(pattern_splits["val"])
    except KeyError as error:
        raise ValueError("patterns_by_split must contain train and val") from error
    if not train or not val:
        raise ValueError("CVAE needs nonempty train and validation pattern sets")
    if set(train) & set(val):
        raise ValueError("train and validation pattern sets must be disjoint")
    supplied = set(map(_length, train + val))
    if supplied & HELD_INTERPOLATION_LENGTHS:
        raise ValueError("lengths 5 and 7 are interpolation-only and cannot enter CVAE data")
    if not supplied <= TRAIN_LENGTHS:
        raise ValueError(f"unexpected CVAE length(s): {sorted(supplied - TRAIN_LENGTHS)}")
    if set(pattern_splits.get("test", ())) & (set(train) | set(val)):
        raise ValueError("test patterns must be separate from CVAE train/validation patterns")
    return train, val


def _load_bank(path: Path, pattern: str) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"missing selected MLP importance bank for {pattern}: {path}")
    record = torch.load(path, map_location="cpu", weights_only=True)
    # ``bank.py`` stores only the selected maps under ``selected``.  Supporting
    # a flat key keeps this loader usable for compact, independently audited
    # banks, while never falling back to an all-model importance tensor.
    selected = record.get("selected", record)
    if "importance" not in selected:
        raise KeyError(f"{path} lacks the selected unsigned 'importance' tensor")
    if "selected" in record and record.get("top_fraction") != 0.1:
        raise ValueError(f"{path} was not selected with the required top 10% protocol")
    importance = selected["importance"].float()
    if importance.ndim != 3 or importance.shape[1:] != (32, 32):
        raise ValueError(f"{path} importance must have shape (N, 32, 32), got {tuple(importance.shape)}")
    if importance.shape[0] < 1 or not torch.isfinite(importance).all():
        raise ValueError(f"{path} has no finite importance maps")
    if importance.min().item() < 0:
        raise ValueError(f"{path} is signed; the CVAE protocol requires unsigned importance")
    return importance.reshape(importance.shape[0], -1)


def load_split_banks(out: Path, patterns: list[str], bank_path) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """Load banks without selecting another top fraction; bank stage already did so."""
    maps, lengths, provenance = [], [], []
    for pattern in patterns:
        path = Path(bank_path(out, pattern))
        selected = _load_bank(path, pattern)
        k = _length(pattern)
        maps.append(selected)
        lengths.append(torch.full((selected.shape[0],), float(k)))
        provenance.append({"pattern": pattern, "length": k, "path": str(path),
                           "sha256": _sha256(path), "n_selected": int(selected.shape[0])})
    return torch.cat(maps), torch.cat(lengths), provenance


def balanced_loader(maps: torch.Tensor, lengths: torch.Tensor, patterns: list[str],
                    pattern_counts: list[int], batch_size: int, seed: int) -> DataLoader:
    """Sample equally across lengths, then patterns, then maps within pattern."""
    if len(patterns) != len(pattern_counts):
        raise ValueError("pattern counts do not match patterns")
    weights = torch.empty(maps.shape[0], dtype=torch.double)
    cursor = 0
    by_length: dict[int, int] = defaultdict(int)
    for pattern in patterns:
        by_length[_length(pattern)] += 1
    for pattern, count in zip(patterns, pattern_counts):
        k = _length(pattern)
        # P(length)=uniform, P(pattern|length)=uniform, P(map|pattern)=uniform.
        weights[cursor:cursor + count] = 1.0 / (len(by_length) * by_length[k] * count)
        cursor += count
    if cursor != maps.shape[0]:
        raise ValueError("pattern counts do not cover the MLP bank")
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(weights, num_samples=maps.shape[0], replacement=True,
                                    generator=generator)
    return DataLoader(TensorDataset(maps, lengths), batch_size=batch_size, sampler=sampler)


@torch.no_grad()
def deterministic_validation(model: LengthCVAE, maps: torch.Tensor, lengths: torch.Tensor,
                             patterns: list[str], counts: list[int], beta: float,
                             batch_size: int, device: torch.device) -> tuple[float, dict[int, float]]:
    """Posterior-mean reconstruction plus analytic KL, equal-weighted by length.

    We select checkpoints with this deterministic score rather than a sampled
    ELBO, so sampling noise cannot choose a checkpoint.  It is intentionally
    reported as a deterministic reconstruction proxy, not as an exact ELBO.
    """
    totals, offsets = [], 0
    model.eval()
    for pattern, count in zip(patterns, counts):
        xs = maps[offsets:offsets + count]
        ls = lengths[offsets:offsets + count]
        offsets += count
        pieces = []
        for start in range(0, count, batch_size):
            x = xs[start:start + batch_size].to(device)
            length = ls[start:start + batch_size].to(device)
            mu, logvar = model.encode(x, length)
            logits = model.decode_lengths(mu, length)
            total, _, _ = cvae_loss(logits, x, mu, logvar, beta)
            pieces.append((float(total), x.shape[0]))
        totals.append((_length(pattern), sum(v * n for v, n in pieces) / sum(n for _, n in pieces)))
    per_length: dict[int, list[float]] = defaultdict(list)
    for k, score in totals:
        per_length[k].append(score)
    aggregate = {k: sum(values) / len(values) for k, values in per_length.items()}
    return sum(aggregate.values()) / len(aggregate), aggregate


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def train(out: Path | str, seed: int, device: str = "cuda") -> Path:
    """Fit one CVAE seed and write ``cvae/seedN/best.pt`` below ``out``."""
    # Imported late: config/common are produced by the sibling pipeline owner.
    from .common import (atomic_save, bank_path, load_protocol, patterns_by_split, seed_for, source_hashes)

    out = Path(out)
    config = load_protocol(out)
    run_dir = out / "cvae" / f"seed{seed}"
    if (run_dir / "best.pt").exists() or (run_dir / "history.json").exists():
        raise FileExistsError(f"refusing to overwrite completed CVAE run: {run_dir}")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; pass --device cpu or run outside the sandbox")
    run_dir.mkdir(parents=True, exist_ok=False)
    dev = torch.device(device)
    torch.manual_seed(seed)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    train_patterns, val_patterns = validate_cvae_patterns(patterns_by_split(config))
    train_maps, train_lengths, train_provenance = load_split_banks(out, train_patterns, bank_path)
    val_maps, val_lengths, val_provenance = load_split_banks(out, val_patterns, bank_path)
    train_counts = [p["n_selected"] for p in train_provenance]
    val_counts = [p["n_selected"] for p in val_provenance]
    loader = balanced_loader(train_maps, train_lengths, train_patterns, train_counts,
                             config.cvae_batch, seed_for("cvae_loader", seed))
    model = LengthCVAE(mask_dim=1024, latent_dim=config.latent_dim,
                       hidden=config.cvae_hidden).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.cvae_lr)
    beta = float(config.cvae_beta)
    best, history, began = math.inf, [], time.monotonic()
    protocol = {
        "config": config.to_dict(),
        "source_sha256": source_hashes(),
        "seed": seed,
        "model_config": model.model_config(),
        "condition": "scalar c=2*(length-3)/5-1; no one-hot, no integer cast, no pattern bits",
        "train_lengths": sorted(TRAIN_LENGTHS),
        "held_interpolation_lengths": sorted(HELD_INTERPOLATION_LENGTHS),
        "selection": "deterministic posterior-mean reconstruction plus analytic KL; equal mean across lengths",
        "loss": "BCE summed over 1024 map entries then averaged over batch + beta*analytic KL",
        "beta": beta,
        "train_banks": train_provenance,
        "validation_banks": val_provenance,
    }
    _atomic_json(run_dir / "protocol.json", protocol)

    # Train-only mean maps provide a non-generative interpolation baseline.
    means: dict[int, torch.Tensor] = {}
    offset = 0
    grouped: dict[int, list[torch.Tensor]] = defaultdict(list)
    for pattern, count in zip(train_patterns, train_counts):
        grouped[_length(pattern)].append(train_maps[offset:offset + count])
        offset += count
    for k, rows in grouped.items():
        means[k] = torch.cat(rows).mean(dim=0).reshape(32, 32)
    atomic_save(run_dir / "train_length_mean_importance.pt", {
        "means": means, "patterns": train_patterns, "provenance": train_provenance,
        "note": "computed from CVAE-train patterns only; no held interpolation length data",
    })

    for epoch in range(1, int(config.cvae_epochs) + 1):
        model.train()
        sums = defaultdict(float)
        seen = 0
        for x, length in loader:
            x, length = x.to(dev), length.to(dev)
            optimizer.zero_grad(set_to_none=True)
            logits, mu, logvar = model(x, length)
            total, recon, kl = cvae_loss(logits, x, mu, logvar, beta)
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite CVAE loss at epoch {epoch}")
            total.backward()
            optimizer.step()
            n = x.shape[0]
            seen += n
            for name, value in (("total", total), ("recon", recon), ("kl", kl)):
                sums[name] += float(value.detach()) * n
        val_score, val_by_length = deterministic_validation(
            model, val_maps, val_lengths, val_patterns, val_counts, beta, config.cvae_batch, dev)
        row = {"epoch": epoch, "train_total": sums["total"] / seen,
               "train_recon": sums["recon"] / seen, "train_kl": sums["kl"] / seen,
               "validation_deterministic_proxy": val_score,
               "validation_by_length": val_by_length, "seconds": time.monotonic() - began}
        history.append(row)
        if val_score < best:
            best = val_score
            atomic_save(run_dir / "best.pt", {
                "model_config": model.model_config(), "model_state": {
                    name: value.detach().cpu() for name, value in model.state_dict().items()},
                "seed": seed, "epoch": epoch, "best_validation_deterministic_proxy": best,
                "history": history, "protocol": protocol,
            })
        if epoch == 1 or epoch % 10 == 0 or epoch == int(config.cvae_epochs):
            print(f"CVAE seed={seed} epoch={epoch} train={row['train_total']:.5f} "
                  f"val_det={val_score:.5f} best={best:.5f}", flush=True)
    _atomic_json(run_dir / "history.json", {"history": history, "best_validation_deterministic_proxy": best})
    _atomic_json(run_dir / "done.json", {"seed": seed, "best_validation_deterministic_proxy": best,
                                            "seconds": time.monotonic() - began})
    return run_dir / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    train(args.out, args.seed, args.device)


if __name__ == "__main__":
    main()
