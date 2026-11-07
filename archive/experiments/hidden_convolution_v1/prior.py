"""Learn global and column-wise VAE priors from source indexators only."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent


class StructureVAE(nn.Module):
    def __init__(self, kind, inputs=64, hidden=36, codes=36):
        super().__init__()
        self.kind, self.inputs, self.hidden, self.codes = kind, inputs, hidden, codes
        self.embedding = nn.Embedding(codes + 1, 4)
        if kind == "column":
            self.position = nn.Embedding(hidden, 8)
            self.latent = 8
            self.body = nn.Sequential(nn.Linear(inputs * 4 + 8, 128), nn.GELU())
            self.decoder = nn.Sequential(
                nn.Linear(16, 128), nn.GELU(), nn.Linear(128, inputs * (codes + 1))
            )
        else:
            self.latent = 64
            self.body = nn.Sequential(nn.Linear(inputs * hidden * 4, 128), nn.GELU())
            self.decoder = nn.Sequential(
                nn.Linear(64, 128),
                nn.GELU(),
                nn.Linear(128, inputs * hidden * (codes + 1)),
            )
        self.mu = nn.Linear(128, self.latent)
        self.logvar = nn.Linear(128, self.latent)

    def decode(self, z):
        if self.kind == "column":
            position = self.position(torch.arange(self.hidden, device=z.device))
            position = position[None].expand(len(z), -1, -1)
            decoded = self.decoder(torch.cat([z, position], -1))
            return decoded.reshape(
                -1, self.hidden, self.inputs, self.codes + 1
            ).transpose(1, 2)
        return self.decoder(z).reshape(-1, self.inputs, self.hidden, self.codes + 1)

    def forward(self, indices, sample=True):
        embedded = self.embedding(indices)
        if self.kind == "column":
            position = self.position(torch.arange(self.hidden, device=indices.device))
            position = position[None].expand(len(indices), -1, -1)
            embedded = torch.cat([embedded.transpose(1, 2).flatten(2), position], -1)
        else:
            embedded = embedded.flatten(1)
        body = self.body(embedded)
        mu, logvar = self.mu(body), self.logvar(body).clamp(-12, 8)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if sample else mu
        return self.decode(z), mu, logvar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/pilot_v1")
    parser.add_argument("--kind", choices=["column", "global"], required=True)
    args = parser.parse_args()
    if (args.out / f"{args.kind}_prior_complete.json").exists():
        parser.error("Prior already trained")
    torch.set_num_threads(4)
    device = torch.device("cuda")
    bank = np.load(args.out / "structure_bank.npz")
    selected = bank["selected"]
    train = selected[bank["task"][selected] % 4 != 0]
    validation = selected[bank["task"][selected] % 4 == 0]
    np.savez_compressed(
        args.out / "prior_split.npz", train=train, validation=validation
    )
    indices = torch.tensor(bank["indices"], device=device)
    frequency = torch.bincount(indices[train].flatten(), minlength=37).float()
    weights = (frequency.sum() / frequency.clamp_min(1)).sqrt()
    weights /= weights.mean()
    # A generic source-only common support, shared by all structured controls.
    mask = (bank["indices"][train] != 0).mean(0) >= 0.5
    np.savez_compressed(
        args.out / "learned_support.npz",
        mask=mask,
        frequency=(bank["indices"][train] != 0).mean(0),
    )
    torch.manual_seed(283020)
    model = StructureVAE(args.kind).to(device)
    torch.save(model.state_dict(), args.out / f"{args.kind}_untrained.pt")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best, history, best_epoch = float("inf"), [], 0
    for epoch in range(401):
        model.eval()
        with torch.no_grad():
            logits, _, _ = model(indices[validation], sample=False)
            score = F.cross_entropy(
                logits.reshape(-1, 37), indices[validation].flatten(), weight=weights
            )
            if float(score) < best:
                best, best_epoch = float(score), epoch
                torch.save(model.state_dict(), args.out / f"{args.kind}_prior.pt")
        if epoch == 400:
            break
        model.train()
        for _ in range(4):
            ids = torch.tensor(train, device=device)[
                torch.randint(len(train), (16,), device=device)
            ]
            logits, mu, logvar = model(indices[ids])
            reconstruction = F.cross_entropy(
                logits.reshape(-1, 37), indices[ids].flatten(), weight=weights
            )
            kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).mean()
            loss = reconstruction + 0.03 * min(1, (epoch + 1) / 50) * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        history.append(
            {
                "epoch": epoch,
                "validation_ce": float(score),
                "training_ce": float(reconstruction.detach()),
                "kl": float(kl.detach()),
            }
        )
        if epoch % 100 == 0:
            print("PRIOR", args.kind, epoch, best, flush=True)
    (args.out / f"{args.kind}_prior_history.json").write_text(json.dumps(history))
    (args.out / f"{args.kind}_prior_complete.json").write_text(
        json.dumps(
            {
                "kind": args.kind,
                "training_indexators": train.tolist(),
                "validation_indexators": validation.tolist(),
                "source_functions_disjoint": True,
                "validation_ce": best,
                "best_epoch": best_epoch,
                "training_reads": [
                    "source indexators",
                    "source task IDs",
                    "source quality selection",
                ],
                "latent_count": 288 if args.kind == "column" else 64,
                "source_common_support_edges": int(mask.sum()),
                "teacher_or_observations_read": False,
                "inductive_bias": "separate latent per matrix column"
                if args.kind == "column"
                else "global latent",
            },
            indent=2,
        )
    )
    shutil.copy2(__file__, args.out / f"{args.kind}_prior_snapshot.py")


if __name__ == "__main__":
    main()
