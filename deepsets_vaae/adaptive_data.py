"""Immutable inputs, density masks, and post-selection data splits.

This module deliberately only reads the previous experiments.  The selection
and confirmation pools are formed with the original :mod:`core` split routine
so raw-row and exact-pixel exclusions have the same meaning as the bank.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import core

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets/mnist8m"
FUNCTIONAL_ROOT = ROOT / "outputs/deepsets_vaae/20261001_converged_functional_vae"
GNN_ROOT = ROOT / "outputs/deepsets_vaae/20261001_other_generators_corrected"
HIDDEN, FEATURES, FLAT = 32, 784, 784 * 32
SPARSE_RHOS = (.01, .02, .05, .10, .20, .30, .50, .70, .90)
RHOS = SPARSE_RHOS + (1.0,)
LRS = (.0005, .002, .005)
BUDGETS = (32, 64, 128, 256)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pixel_hash(image: np.ndarray) -> bytes:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).digest()


def _original_pools(seed: int, device: torch.device) -> tuple[dict, set[int], set[bytes]]:
    """Rebuild the five original pools and verify the canonical provenance."""
    data = core.load_data(DATA_DIR, seed, device)
    stored = json.loads((FUNCTIONAL_ROOT / f"seed_{seed}" / "data_provenance.json").read_text())
    if data["split_hashes"] != stored["split_hashes"]:
        raise ValueError(f"canonical split hashes differ for seed {seed}")
    ids: set[int] = set()
    pixels: set[bytes] = set()
    for name in ("source_train", "target_train", "source_validation", "target_validation", "target_test"):
        split = data[name]
        ids.update(map(int, split.source_ids.cpu().tolist()))
        pixels.update(_pixel_hash(row) for row in (split.features.cpu().numpy() * 255).round().astype(np.uint8))
    return data, ids, pixels


def _assert_disjoint(splits: dict[str, core.Split]) -> None:
    names = list(splits)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if torch.isin(splits[left].source_ids, splits[right].source_ids).any():
                raise AssertionError(f"row overlap: {left}/{right}")
            left_hashes = {_pixel_hash(x) for x in (splits[left].features.cpu().numpy() * 255).round().astype(np.uint8)}
            right_hashes = {_pixel_hash(x) for x in (splits[right].features.cpu().numpy() * 255).round().astype(np.uint8)}
            if left_hashes & right_hashes:
                raise AssertionError(f"pixel overlap: {left}/{right}")


def _provenance(splits: dict[str, core.Split], *, stage: str, identity_limit: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "split_hashes": {name: core._hash_ids(split.source_ids) for name, split in splits.items()},
        "ids": {name: split.source_ids.cpu().tolist() for name, split in splits.items()},
        "counts": {name: len(split.features) for name, split in splits.items()},
        "row_ids_pairwise_disjoint": True,
        "exact_pixel_pairs_disjoint": True,
        "identity_limit": identity_limit,
    }


def _read_new(images: np.ndarray, labels: np.ndarray, used_rows: set[int], used_hashes: set[bytes],
              *, seed: int, device: torch.device, name: str, block: int, per_digit: int, part: int,
              destination: dict[str, core.Split], duplicates: dict[str, int]) -> None:
    split, skipped = core._read_split(images, labels, block_id=block, per_digit=per_digit, seed=seed,
                                      part=part, device=device, used_rows=used_rows,
                                      used_pixel_hashes=used_hashes)
    destination[name], duplicates[name] = split, skipped


def selection_data(seed: int, device: str | torch.device = "cpu") -> tuple[dict[str, core.Split], dict[str, Any]]:
    """New selection train/checkpoint/score pools; never reads confirmation blocks."""
    device = torch.device(device)
    _, rows, hashes = _original_pools(seed, device)
    images = np.load(DATA_DIR / "images.npy", mmap_mode="r")
    labels = np.load(DATA_DIR / "labels.npy", mmap_mode="r")
    splits: dict[str, core.Split] = {}; duplicates: dict[str, int] = {}
    _read_new(images, labels, rows, hashes, seed=seed, device=device, name="selection_train", block=3,
              per_digit=1000, part=0, destination=splits, duplicates=duplicates)
    _read_new(images, labels, rows, hashes, seed=seed, device=device, name="selection_checkpoint", block=4,
              per_digit=300, part=0, destination=splits, duplicates=duplicates)
    _read_new(images, labels, rows, hashes, seed=seed, device=device, name="selection_score", block=5,
              per_digit=300, part=0, destination=splits, duplicates=duplicates)
    _assert_disjoint(splits)
    meta = _provenance(splits, stage="selection",
                       identity_limit="Distinct raw rows and exact pixels; augmentation-group identity unavailable")
    meta["blocks"] = {"selection_train": [3, 0], "selection_checkpoint": [4, 0], "selection_score": [5, 0]}
    meta["exact_pixel_duplicates_excluded"] = duplicates
    return splits, meta


def confirmation_data(seed: int, device: str | torch.device = "cpu") -> tuple[dict[str, core.Split], dict[str, Any]]:
    """Build final pools after rebuilding the selection exclusions first."""
    device = torch.device(device)
    _, rows, hashes = _original_pools(seed, device)
    images = np.load(DATA_DIR / "images.npy", mmap_mode="r")
    labels = np.load(DATA_DIR / "labels.npy", mmap_mode="r")
    ignored: dict[str, core.Split] = {}; dup: dict[str, int] = {}
    for name, block, count, part in (("selection_train", 3, 1000, 0), ("selection_checkpoint", 4, 300, 0),
                                     ("selection_score", 5, 300, 0)):
        _read_new(images, labels, rows, hashes, seed=seed, device=device, name=name, block=block,
                  per_digit=count, part=part, destination=ignored, duplicates=dup)
    splits: dict[str, core.Split] = {}
    _read_new(images, labels, rows, hashes, seed=seed, device=device, name="confirmation_train", block=6,
              per_digit=1000, part=0, destination=splits, duplicates=dup)
    _read_new(images, labels, rows, hashes, seed=seed, device=device, name="confirmation_checkpoint", block=7,
              per_digit=300, part=0, destination=splits, duplicates=dup)
    _read_new(images, labels, rows, hashes, seed=seed, device=device, name="confirmation_test", block=7,
              per_digit=300, part=1, destination=splits, duplicates=dup)
    _assert_disjoint({**ignored, **splits})
    meta = _provenance(splits, stage="confirmation",
                       identity_limit="Distinct raw rows and exact pixels; augmentation-group identity unavailable")
    meta["blocks"] = {"confirmation_train": [6, 0], "confirmation_checkpoint": [7, 0],
                      "confirmation_test": [7, 1]}
    meta["excluded_selection_split_hashes"] = _provenance(ignored, stage="selection", identity_limit="")["split_hashes"]
    meta["exact_pixel_duplicates_excluded"] = dup
    return splits, meta


def centred_tasks(seed: int, count: int) -> np.ndarray:
    values = np.random.default_rng(seed).normal(size=(count, 10)).astype(np.float32)
    values -= values.mean(axis=1, keepdims=True)
    values /= values.std(axis=1, keepdims=True)
    return values


def _stable_order(score: torch.Tensor) -> torch.Tensor:
    # numpy stable sort is explicit across CPU/CUDA versions and tie-safe.
    flat = score.detach().cpu().numpy().reshape(-1)
    return torch.from_numpy(np.argsort(-flat, kind="stable").astype(np.int64))


def _prefix(order: torch.Tensor, k: int, device: torch.device, replicas: int = 4) -> torch.Tensor:
    if not 0 <= k <= FLAT:
        raise ValueError("invalid edge count")
    mask = torch.zeros(FLAT, dtype=torch.float32)
    mask[order[:k]] = 1
    return mask.reshape(FEATURES, HIDDEN).to(device)[None].expand(replicas, -1, -1).clone()


def k_for_rho(rho: float) -> int:
    return int(round(float(rho) * FLAT))


def source_orders(seed: int) -> dict[str, torch.Tensor]:
    """Rankings from immutable source-only functional/GNN artifacts."""
    array_path = FUNCTIONAL_ROOT / f"seed_{seed}/functional/functional_vae_arrays.npz"
    with np.load(array_path) as arrays:
        value = torch.from_numpy(np.asarray(arrays["function_train_aligned"], dtype=np.float32)).mean((0, 1))
    gnn = torch.load(GNN_ROOT / f"seed_{seed}/gnn_flow/samples.pt", map_location="cpu", weights_only=True)["score"].float()
    if value.shape != (FEATURES, HIDDEN) or gnn.shape != value.shape:
        raise ValueError("source score shape is not [784,32]")
    pixel = value.mean(1, keepdim=True).expand(-1, HIDDEN)
    return {"functional": _stable_order(value), "gnn": _stable_order(gnn), "pixelprior": _stable_order(pixel)}


def selection_masks(seed: int, device: str | torch.device) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    """37 logical source methods × 3 LR; dense is only materialized once/LR."""
    device = torch.device(device); orders = source_orders(seed)
    masks: dict[str, torch.Tensor] = {}; manifest: list[dict[str, Any]] = []
    for family, order in orders.items():
        for rho in SPARSE_RHOS:
            for lr in LRS:
                name = f"{family}_rho{rho:g}_lr{lr:g}"
                masks[name] = _prefix(order, k_for_rho(rho), device)
                manifest.append({"method": name, "family": family, "rho": rho, "lr": lr, "kind": "source"})
    generator = np.random.default_rng(seed + 950000)
    random_orders = [torch.from_numpy(generator.permutation(FLAT).astype(np.int64)) for _ in range(4)]
    for rho in SPARSE_RHOS:
        for lr in LRS:
            name = f"random_rho{rho:g}_lr{lr:g}"; k = k_for_rho(rho)
            masks[name] = torch.stack([_prefix(order, k, device, replicas=1)[0] for order in random_orders])
            manifest.append({"method": name, "family": "random", "rho": rho, "lr": lr, "kind": "random"})
    dense = torch.ones((4, FEATURES, HIDDEN), device=device)
    for lr in LRS:
        name = f"dense_lr{lr:g}"; masks[name] = dense
        manifest.append({"method": name, "family": "dense", "rho": 1.0, "lr": lr, "kind": "dense"})
    return masks, manifest


def assert_mask_protocol(masks: dict[str, torch.Tensor]) -> None:
    for name, mask in masks.items():
        if mask.shape != (4, FEATURES, HIDDEN) or not torch.isfinite(mask).all():
            raise AssertionError(f"invalid mask {name}")
        if not torch.equal(mask, mask[0:1].expand_as(mask)) and not name.startswith("random_"):
            raise AssertionError(f"non-random mask replicas differ: {name}")
        if name.startswith("dense_") and not bool(mask.all()):
            raise AssertionError("dense endpoint is not all-on")
