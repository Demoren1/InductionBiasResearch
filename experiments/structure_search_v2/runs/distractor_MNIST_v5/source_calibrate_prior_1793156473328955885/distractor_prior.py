"""A shared NF encoder and a binary VAE for quality-ranked source masks."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path("experiments/structure_search_v2")


class MaskVAE(nn.Module):
    def __init__(self, inputs, latent=16):
        super().__init__()
        # Tokens are invariant hidden-column moments of weights, gradients and
        # functional responses. The mask has identical support in every column.
        self.encoder = nn.Sequential(
            nn.Flatten(), nn.Linear(inputs * 3, 128), nn.GELU()
        )
        self.mu = nn.Linear(128, latent)
        self.logvar = nn.Linear(128, latent)
        self.decoder = nn.Sequential(
            nn.Linear(latent, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, inputs),
        )

    def forward(self, tokens, sample=True):
        hidden = self.encoder(tokens)
        mu, logvar = self.mu(hidden), self.logvar(hidden).clamp(-12, 8)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if sample else mu
        return self.decoder(z), mu, logvar


def compact_assignment(scores, slots, straight_through=False):
    indices = scores.topk(slots, dim=-1, sorted=True).indices
    hard = F.one_hot(indices, scores.shape[-1]).to(scores.dtype)
    if straight_through:
        # Sequential soft choices without replacement, with hard forward rows.
        # This is an explicit biased STE; C always has exactly slots*hidden values.
        used_before = hard.cumsum(1) - hard
        soft = (
            scores[:, None]
            .expand_as(hard)
            .masked_fill(used_before.bool(), -1e9)
            .softmax(-1)
        )
        hard = hard + soft - soft.detach()
    return hard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["MNIST", "FashionMNIST"], required=True)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--plain", action="store_true")
    args = parser.parse_args()
    out = ROOT / "runs" / ("distractor_" + args.dataset + "_" + args.version)
    shutil.copy2(__file__, out / "prior_source.py")
    data = np.load(out / "good_bank.npz")
    torch.set_num_threads(4)
    device = torch.device("cuda")
    tokens = torch.tensor(data["tokens"], device=device)
    n, d, _ = tokens.shape
    slots = data["indices"].shape[1]
    masks = torch.zeros(n, d, device=device).scatter_(
        1, torch.tensor(data["indices"], device=device), 1.0
    )
    relevance = tokens[..., 2]
    importance = relevance / (relevance.sum(1, keepdim=True) / slots).clamp_min(1e-8)
    if args.plain:
        importance = torch.ones_like(importance)
    confidence = torch.tensor(np.maximum(0, np.log(2) - data["quality"]), device=device)
    for task in np.unique(data["task"]):
        for role in range(3):
            ids = torch.tensor(
                (data["task"] == task) & (data["role"] == role), device=device
            )
            confidence[ids] /= confidence[ids].mean().clamp_min(1e-8)
    train_pools = [
        torch.tensor(
            np.flatnonzero((data["task"] == task) & (data["role"] == 0)), device=device
        )
        for task in np.unique(data["task"])
    ]
    val = torch.tensor(np.flatnonzero(data["role"] == 1), device=device)
    torch.manual_seed(281420)
    model = MaskVAE(d).to(device)
    torch.save(model.state_dict(), out / "untrained_prior.pt")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best, history = float("inf"), []

    def reconstruction(scores, ids):
        positive = masks[ids] * importance[ids] * (d - slots) / slots
        terms = positive * F.softplus(-scores) + (1 - masks[ids]) * F.softplus(scores)
        return (terms.mean(1) * confidence[ids]).mean()

    for epoch in range(301):
        model.eval()
        with torch.no_grad():
            validation = sum(
                float(reconstruction(model(tokens[ids], sample=False)[0], ids))
                * len(ids)
                for ids in val.split(128)
            ) / len(val)
            if validation < best:
                best = validation
                torch.save(model.state_dict(), out / "prior.pt")
        if epoch == 300:
            break
        model.train()
        for _ in range(4):
            selected = torch.cat(
                [
                    pool[torch.randint(len(pool), (8,), device=device)]
                    for pool in train_pools
                ]
            )
            scores, mu, logvar = model(tokens[selected])
            ce = reconstruction(scores, selected)
            kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).mean()
            loss = ce + 0.5 * min(1, (epoch + 1) / 50) * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        row = {
            "epoch": epoch,
            "validation_weighted_bce": validation,
            "training_weighted_bce": float(ce.detach()),
            "kl": float(kl.detach()),
        }
        history.append(row)
        if epoch % 25 == 0:
            print(args.dataset, "PRIOR", row, flush=True)
    model.load_state_dict(torch.load(out / "prior.pt", weights_only=True))
    model.eval()
    with torch.no_grad():
        samples = (
            model.decoder(torch.randn(512, 16, device=device)).topk(slots, -1).indices
        )
    np.savez_compressed(out / "prior_samples.npz", indices=samples.cpu().numpy())
    (out / "prior_history.json").write_text(json.dumps(history))
    (out / "prior_complete.json").write_text(
        json.dumps(
            {
                "inputs": d,
                "slots": slots,
                "latent": 16,
                "encoder": "shared neural-functional row tokens invariant to hidden-column permutations",
                "loss": (
                    "BCE restoring good source masks; source quality weighting"
                    if args.plain
                    else "BCE restoring good source masks, positives weighted by functional contribution; source quality weighting"
                ),
                "checkpoint": "source held-lineage validation weighted reconstruction BCE",
                "best_validation_bce": best,
                "target_classes_used": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
