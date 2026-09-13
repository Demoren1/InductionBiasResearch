"""Pair-level delta analysis shared by the pattern-8 and seq32/k5 runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pattern.evaluation.decoder_agreement import align_columns
from pattern.evaluation.z_star_noise import ci, write_json
from pattern.data.generate import ideal_mask as ideal_mask_k4
from pattern.fixed_k5_agreement.common import ideal_mask as ideal_mask_k5
from pattern.fixed_k5_agreement.config import Config as K5Config


def structure(masks: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    targets = target[None].expand(len(masks), -1, -1).to(masks)
    aligned = align_columns(targets.float(), masks.float())
    intersection = (targets * aligned).sum((1, 2))
    k = float(target.sum())
    iou = intersection / (2 * k - intersection)
    exact = (aligned == targets).all(2).all(1)
    return iou.cpu(), exact.cpu()


def main() -> None:
    # Avoid OpenMP oversubscription across thousands of tiny 32x32 costs.
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kind", choices=("pattern8", "pattern32"), required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    protocol = json.loads((out / "protocol.json").read_text())
    target = ideal_mask_k4().float() if args.kind == "pattern8" else ideal_mask_k5(K5Config()).float()
    groups = protocol.get("groups", ["prior", "oracle_z_star"])
    n = protocol["settings"]["starts_per_group"]
    pairs = protocol["pairs"]
    per_pair = []
    pooled = {group: {"initial_exact": 0, "final_exact": 0, "count": 0} for group in groups}
    for pair in pairs:
        pair_row = {"pair": pair, "groups": {}}
        for group in groups:
            decoder_rows = []
            for seed in pair:
                oracle = torch.load(out / f"seed_{seed}/oracle_same_start.pt",
                                    map_location="cpu", weights_only=True)
                if group == "prior":
                    sl = slice(0, n)
                    keep = torch.ones(n, dtype=torch.bool)
                else:
                    sl = slice(n, 2 * n)
                    keep = (oracle["best_soft"]["iou"][:n] == 1
                            if group == "oracle_z_star" else torch.ones(n, dtype=torch.bool))
                nested = {key: [] for key in (
                    "accuracy_delta", "bce_delta", "iou_delta",
                    "fraction_accuracy_better", "fraction_bce_better")}
                for pattern in protocol["patterns"]:
                    task = torch.load(out / f"seed_{seed}/task_{pattern}.pt",
                                      map_location="cpu", weights_only=True)
                    initial_acc = task["final_eval_initial_accuracy"][sl][keep]
                    final_acc = task["final_eval_final_accuracy"][sl][keep]
                    initial_bce = task["final_eval_initial_bce"][sl][keep]
                    final_bce = task["final_eval_final_bce"][sl][keep]
                    initial_iou, initial_exact = structure(task["initial_masks"][sl][keep], target)
                    final_iou, final_exact = structure(task["final_masks"][sl][keep], target)
                    nested["accuracy_delta"].append(float((final_acc - initial_acc).mean()))
                    nested["bce_delta"].append(float((final_bce - initial_bce).mean()))
                    nested["iou_delta"].append(float((final_iou - initial_iou).mean()))
                    nested["fraction_accuracy_better"].append(float((final_acc > initial_acc).float().mean()))
                    nested["fraction_bce_better"].append(float((final_bce < initial_bce).float().mean()))
                    pooled[group]["initial_exact"] += int(initial_exact.sum())
                    pooled[group]["final_exact"] += int(final_exact.sum())
                    pooled[group]["count"] += len(final_exact)
                decoder_rows.append({key: float(np.mean(values)) for key, values in nested.items()})
            pair_row["groups"][group] = {
                key: float(np.mean([row[key] for row in decoder_rows])) for key in decoder_rows[0]
            }
        per_pair.append(pair_row)
    aggregate = {group: {
        key: ci([row["groups"][group][key] for row in per_pair])
        for key in per_pair[0]["groups"][group]
    } for group in groups}
    for group in pooled:
        pooled[group]["initial_fraction"] = pooled[group]["initial_exact"] / pooled[group]["count"]
        pooled[group]["final_fraction"] = pooled[group]["final_exact"] / pooled[group]["count"]
    payload = {"paired_deltas": aggregate, "pooled_exact": pooled,
               "per_pair": per_pair, "independence_unit": "VAE pair"}
    write_json(out / "paired_analysis.json", payload)
    print(json.dumps({"paired_deltas": aggregate, "pooled_exact": pooled}, indent=2))


if __name__ == "__main__":
    main()
