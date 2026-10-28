"""Fit a prior using only source indexators and their task/lineage IDs."""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "structure_search_v2"))
from distractor_prior import MaskVAE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--inputs", type=int, required=True)
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / "prior_source.py")
    # These are the only source-bank fields the extractor reads. Source filtering
    # for relatively good fits happened during generic structure search earlier.
    with np.load(args.bank) as archive:
        indices = archive["indices"].copy()
        task = archive["task"].copy()
        role = archive["role"].copy()
    if args.shuffle:
        rng = np.random.default_rng(281900)
        indices = np.stack([rng.permutation(args.inputs)[row] for row in indices])
    n, slots = indices.shape
    masks = np.zeros((n, args.inputs), dtype=np.float32)
    np.put_along_axis(masks, indices, 1, axis=1)
    # The reused network accepts three channels; only the mask channel is used.
    tokens = np.zeros((n, args.inputs, 3), dtype=np.float32)
    tokens[:, :, 0] = masks
    np.savez_compressed(
        out / "good_bank.npz",
        indices=indices,
        task=task,
        role=role,
        tokens=tokens,
        quality=np.full(n, 0.2, dtype=np.float32),
    )
    frequencies = masks[role == 0].mean(0)
    np.savez_compressed(
        out / "mask_statistics.npz",
        frequencies=frequencies,
        selected=np.argsort(-frequencies, kind="stable")[:slots],
    )
    torch.set_num_threads(4)
    device = torch.device("cuda")
    features = torch.tensor(tokens, device=device)
    targets = torch.tensor(masks, device=device)
    train_pools = [
        torch.tensor(np.flatnonzero((task == t) & (role == 0)), device=device)
        for t in np.unique(task)
    ]
    validation = torch.tensor(np.flatnonzero(role == 1), device=device)
    torch.manual_seed(281910)
    model = MaskVAE(args.inputs).to(device)
    torch.save(model.state_dict(), out / "untrained_prior.pt")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best, history = float("inf"), []

    def reconstruction(scores, ids):
        return (
            targets[ids] * (args.inputs - slots) / slots * F.softplus(-scores)
            + (1 - targets[ids]) * F.softplus(scores)
        ).mean()

    for epoch in range(301):
        model.eval()
        with torch.no_grad():
            value = sum(
                float(reconstruction(model(features[ids], sample=False)[0], ids))
                * len(ids)
                for ids in validation.split(128)
            ) / len(validation)
            if value < best:
                best = value
                torch.save(model.state_dict(), out / "prior.pt")
        if epoch == 300:
            break
        model.train()
        for _ in range(4):
            ids = torch.cat(
                [
                    pool[torch.randint(len(pool), (8,), device=device)]
                    for pool in train_pools
                ]
            )
            scores, mu, logvar = model(features[ids])
            ce = reconstruction(scores, ids)
            kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).mean()
            loss = ce + 0.5 * min(1, (epoch + 1) / 50) * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        history.append(
            {
                "epoch": epoch,
                "validation_bce": value,
                "training_bce": float(ce.detach()),
                "kl": float(kl.detach()),
            }
        )
        if epoch % 50 == 0:
            print(out.name, history[-1], flush=True)
    (out / "prior_history.json").write_text(json.dumps(history))
    (out / "prior_complete.json").write_text(
        json.dumps(
            {
                "inputs": args.inputs,
                "slots": slots,
                "latent": 16,
                "source_bank_fields_read": ["indices", "task", "role"],
                "source_bank_sha256": hashlib.sha256(
                    args.bank.read_bytes()
                ).hexdigest(),
                "mask_shuffle_control": args.shuffle,
                "validation_reconstruction_bce": best,
                "quality_weighting": "equal; bank prefiltered by source fit quality",
                "uses_observations": False,
                "uses_numeric_weights": False,
                "uses_feature_correlations": False,
                "uses_number_of_groups": False,
                "uses_functional_ablations": False,
                "train_masks": int((role == 0).sum()),
                "validation_masks": int((role == 1).sum()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
