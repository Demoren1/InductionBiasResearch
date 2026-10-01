"""Prune selected MNIST8m importance maps to a chosen connection density."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

from pattern.mnist8m_importance_bce import save
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        "pattern/outputs/mnist8m_raw_mlp_bce/pair38"))
    parser.add_argument("--out", type=Path, default=Path(
        "pattern/outputs/mnist8m_raw_mlp_bce/pair38_5pct"))
    parser.add_argument("--density", type=float, default=.05)
    args = parser.parse_args()
    if not 0 < args.density <= 1:
        parser.error("--density must be in (0, 1]")
    args.out.mkdir(parents=True, exist_ok=True)
    source_features = args.source / "features.pt"
    target_features = args.out / "features.pt"
    if not target_features.exists():
        temporary = target_features.with_suffix(".tmp")
        shutil.copy2(source_features, temporary)
        temporary.replace(target_features)
    for task in (0, 1):
        source_path = args.source / f"bank_task{task}.pt"
        target_path = args.out / f"bank_task{task}.pt"
        bank = torch.load(source_path, map_location="cpu", weights_only=True)
        shape = bank["importance"].shape
        if len(shape) != 3 or shape[1] != 784:
            raise ValueError(f"unexpected bank shape: {shape}")
        connections = round(args.density * shape[1] * shape[2])
        if connections < 1:
            raise ValueError("density yields zero connections")
        if target_path.exists():
            existing = torch.load(target_path, map_location="cpu", weights_only=True)
            if (existing["settings"].get("derived_connections") != connections or
                    existing["settings"].get("source_bank") != str(source_path)):
                raise ValueError(f"unexpected existing bank: {target_path}")
            continue
        importance = bank["importance"].reshape(len(bank["importance"]), -1)
        mask = torch.zeros_like(importance)
        mask.scatter_(1, importance.topk(connections, dim=1).indices, 1.)
        pruned = importance * mask
        assert torch.all(mask.sum(1) == connections)
        assert torch.all(pruned[mask == 0] == 0)
        prepared = dict(bank)
        prepared["importance"] = pruned.reshape_as(bank["importance"])
        prepared["masks"] = mask.reshape_as(bank["masks"])
        prepared["settings"] = {
            **bank["settings"],
            "derived_connections": connections,
            "derived_density": connections / (shape[1] * shape[2]),
            "source_bank": str(source_path),
            "derivation": "top-K of normalized |W*M| among the selected 10% of 20% bank MLPs",
        }
        save(prepared, target_path)
        print(f"digit {bank['digit']}: {len(pruned)} maps, "
              f"{connections} links each -> {target_path}", flush=True)


if __name__ == "__main__":
    main()
