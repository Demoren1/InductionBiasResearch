"""Observe frozen proposal responsiveness on source train/validation tasks.

This diagnostic never selects models or reads held-out test-task labels.
Bank replacement is an out-of-distribution intervention, not a utility test.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .core import build_experiment_data, sample_balanced
from .generator import Generator, permute_hidden_columns


def run(root: Path) -> dict:
    torch.set_num_threads(1)
    summaries = []
    arrays = {}
    for seed in range(8100, 8104):
        data = build_experiment_data(probe_seed=seed)
        bank = torch.load(root / f"seed_{seed}/bank/bank.pt",
                          map_location="cpu", weights_only=False)["feature"]
        task_ids = list(data["pools"])
        rng = torch.Generator().manual_seed(seed + 313_000)
        episodes = [sample_balanced(data["pools"][task]["support"], 128, rng)
                    for task in task_ids]
        x = torch.stack([episode["x"] for episode in episodes])
        y = torch.stack([episode["y"] for episode in episodes])
        checkpoint = torch.load(root / f"seed_{seed}/transformer_mask/meta/best.pt",
                                map_location="cpu", weights_only=False)
        model = Generator().eval()
        model.load_state_dict(checkpoint["model_state"])
        hidden_order = torch.stack([torch.randperm(8, generator=rng)
                                   for _ in range(bank.size(0))])
        permuted = permute_hidden_columns(bank, hidden_order)
        permuted = permuted[torch.randperm(bank.size(0), generator=rng)]
        with torch.no_grad():
            masks, logits = model(bank, x, y)
            perm_masks, perm_logits = model(permuted, x, y)
            zero_masks, zero_logits = model(torch.zeros_like(bank), x, y)
            reversed_masks, reversed_logits = model(bank, x, 1 - y)
        differences = logits.flatten(1) - logits[0].flatten()[None]
        summaries.append({
            "seed": seed, "task_ids": task_ids, "support_size": 128,
            "unique_masks_across_12_source_contexts": int(torch.unique(masks.flatten(1), dim=0).size(0)),
            "context_logit_max_difference_from_first": float(differences.abs().max()),
            "permutation_logits_max_absolute_delta": float((perm_logits - logits).abs().max()),
            "permutation_mask_changed_edges": int((perm_masks != masks).sum()),
            "zero_bank_logits_max_absolute_delta": float((zero_logits - logits).abs().max()),
            "zero_bank_mask_changed_edges": int((zero_masks != masks).sum()),
            "flipped_support_labels_logits_max_absolute_delta": float((reversed_logits - logits).abs().max()),
            "flipped_support_labels_mask_changed_edges": int((reversed_masks != masks).sum()),
            "mean_sigmoid_surrogate_derivative": float((logits.sigmoid() * (1 - logits.sigmoid())).mean()),
            "logits_min": float(logits.min()), "logits_max": float(logits.max()),
            "row_counts_first_context": masks[0].sum(1).tolist(),
            "column_counts_first_context": masks[0].sum(0).tolist(),
        })
        arrays[f"seed_{seed}_masks"] = masks.numpy()
        arrays[f"seed_{seed}_logits"] = logits.numpy()
        arrays[f"seed_{seed}_support_ids"] = torch.stack([e["ids"] for e in episodes]).numpy()
    destination = root / "diagnostics" / "proposal_responsiveness"
    destination.mkdir(parents=True, exist_ok=True)
    payload = {"source_only": True, "used_for_selection": False,
               "bank_replacement_is_ood": True, "summaries": summaries}
    (destination / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(destination / "arrays.npz", **arrays)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().root), indent=2))
