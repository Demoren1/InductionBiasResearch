"""Extract useful structural parts from partial source networks by ablation.

Each selected feature is replaced by its source-query mean, keeping all network
weights fixed. The increase in query BCE measures its transferable utility.
Only source solutions and source labels enter this computation.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path("experiments/structure_search_v2/runs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["MNIST", "FashionMNIST"], required=True)
    args = parser.parse_args()
    parent = ROOT / ("distractor_" + args.dataset + "_v1")
    out = ROOT / ("distractor_" + args.dataset + "_v2")
    out.mkdir(exist_ok=False)
    shutil.copy2(__file__, out / "ablation_source.py")
    for file in [
        "source_splits.npz",
        "source_observations.npz",
        "hidden_layout.npz",
        "distractor_protocol.json",
        "bank_complete.json",
        "pretrained.pt",
        "pretrain_complete.json",
        "pretrain_history.json",
    ]:
        shutil.copy2(parent / file, out / file)
    bank = dict(np.load(parent / "good_bank.npz"))
    source = np.load(parent / "source_observations.npz")
    torch.set_num_threads(4)
    device = torch.device("cuda")
    qx, qy = (
        torch.tensor(source["query_x"], device=device),
        torch.tensor(source["query_y"], device=device),
    )
    utilities = []
    for start in range(0, len(bank["indices"]), 32):
        ids = slice(start, start + 32)
        tasks = torch.tensor(bank["task"][ids], device=device)
        selected = torch.tensor(bank["indices"][ids], device=device)
        chosen = qx[tasks].gather(2, selected[:, None].expand(-1, qx.shape[1], -1))
        c = torch.tensor(bank["codebook"][ids], device=device)
        b = torch.tensor(bank["bias"][ids], device=device)
        a = torch.tensor(bank["readout"][ids], device=device)
        offset = torch.tensor(bank["offset"][ids], device=device)
        pre = torch.bmm(chosen, c) + b[:, None]
        logits = (pre.relu() * a[:, None]).sum(-1) + offset[:, None]
        original = F.binary_cross_entropy_with_logits(
            logits, qy[tasks], reduction="none"
        ).mean(-1)
        removed = (chosen - chosen.mean(1, keepdim=True)).transpose(1, 2)
        ablated_pre = pre[:, None] - removed[:, :, :, None] * c[:, :, None, :]
        ablated = (ablated_pre.relu() * a[:, None, None]).sum(-1) + offset[
            :, None, None
        ]
        labels = qy[tasks][:, None].expand_as(ablated)
        ablated_ce = F.binary_cross_entropy_with_logits(
            ablated, labels, reduction="none"
        ).mean(-1)
        utilities.append((ablated_ce - original[:, None]).cpu().numpy())
    utility = np.concatenate(utilities)
    positive = utility.clip(0)
    positive /= positive.max(1, keepdims=True).clip(1e-8)
    bank["tokens"][:, :, 2] = 0
    np.put_along_axis(bank["tokens"][:, :, 2], bank["indices"], positive, axis=1)
    bank["ablation_utility"] = utility
    np.savez_compressed(out / "good_bank.npz", **bank)
    (out / "ablation_complete.json").write_text(
        json.dumps(
            {
                "parent": str(parent),
                "source_query_only": True,
                "criterion": "BCE after replacing one input by its query mean minus original BCE",
                "networks": len(utility),
                "positive_slots_mean": float((utility > 0).sum(1).mean()),
                "target_classes_used": False,
                "ground_truth_layout_used": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
