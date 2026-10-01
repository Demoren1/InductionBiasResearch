"""Align all saved length-11 importance maps without using the gold mask.

The gold Toeplitz mask is used only for reporting the final IoU. This builds a
candidate support prior for later U experiments; it does not train a VAE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from pattern.length11.evaluate import gold_iou, gold_mask
from pattern.length11.settings import ROOT


def align_to(reference: np.ndarray, item: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    old, new = linear_sum_assignment(-(item.T @ reference))
    permutation = np.empty(8, dtype=np.int8)
    permutation[new] = old
    return item[:, permutation], permutation


def fit(maps: np.ndarray, start: int, iterations: int) -> tuple[np.ndarray, np.ndarray, float]:
    reference = maps[start].copy()
    for _ in range(iterations):
        aligned = np.stack([align_to(reference, item)[0] for item in maps])
        reference = aligned.mean(0)
    permutations = np.stack([align_to(reference, item)[1] for item in maps])
    aligned = np.stack([item[:, permutation] for item, permutation in zip(maps, permutations)])
    consensus = aligned.mean(0)
    score = float(np.einsum("nih,ih->", aligned, consensus) / len(maps))
    return consensus, permutations, score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=ROOT / "bank")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "u_importance_consensus")
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    paths = sorted(args.bank.glob("pattern_*.pt"))
    if len(paths) != 16:
        raise ValueError(f"expected 16 importance banks, found {len(paths)}")
    blocks = [torch.load(path, map_location="cpu", weights_only=True)["importance"]
              for path in paths]
    maps = torch.cat(blocks).numpy()
    starts = [0, len(blocks[0]) * 5, len(blocks[0]) * 10, len(blocks[0]) * 15]
    candidates = [fit(maps, start, args.iterations) for start in starts]
    best_index = int(np.argmax([item[2] for item in candidates]))
    consensus, permutations, score = candidates[best_index]
    values = torch.tensor(consensus)
    global_mask = torch.zeros(88)
    global_mask[values.flatten().topk(32).indices] = 1
    global_mask = global_mask.reshape(11, 8)
    column_mask = torch.zeros_like(values)
    column_mask.scatter_(0, values.topk(4, dim=0).indices, 1)
    centers = (column_mask * torch.arange(11)[:, None]).sum(0) / column_mask.sum(0)
    canonical_permutation = torch.argsort(centers, stable=True)
    canonical_mask = column_mask[:, canonical_permutation]
    canonical_overlap = float((canonical_mask * gold_mask()).sum())
    summary = {
        "n_maps": int(len(maps)),
        "n_patterns": len(paths),
        "maps_per_pattern": [int(len(block)) for block in blocks],
        "initializations": starts,
        "chosen_initialization": starts[best_index],
        "selection_metric": "mean aligned importance dot product; no gold used",
        "selection_score": score,
        "iterations": args.iterations,
        "global_top32_toeplitz_iou": gold_iou(global_mask),
        "column_top4_toeplitz_iou": gold_iou(column_mask),
        "canonical_column_permutation": canonical_permutation.tolist(),
        "canonical_mask_toeplitz_iou": canonical_overlap / (64 - canonical_overlap),
        "VAE_trained": False,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"consensus": values, "global_mask": global_mask,
                "column_mask": column_mask, "canonical_mask": canonical_mask,
                "canonical_permutation": canonical_permutation,
                "permutations": torch.tensor(permutations.astype(np.int64)),
                "bank_paths": [str(path) for path in paths],
                "summary": summary}, args.out / "consensus.pt")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
