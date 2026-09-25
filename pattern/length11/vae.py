"""Train independent pattern-specific VAEs on length-11 importance maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from pattern.length11.settings import MASK_DIM, PATTERNS, ROOT
from pattern.models.cvae import CVAE


LATENT = 32
WIDTH = 256
BETA = 0.1
MAX_EPOCHS = 300
PATIENCE = 30
BATCH = 128


def loss_parts(model: CVAE, maps: torch.Tensor, *, deterministic: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    c = maps.new_zeros(len(maps), 0)
    mu, logvar = model.encode(maps, c)
    z = mu if deterministic else model.reparameterize(mu, logvar)
    logits = model.decode(z, c)
    recon = F.binary_cross_entropy_with_logits(logits, maps, reduction="none").sum(-1).mean()
    kl = (-.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1)).mean()
    return recon + BETA * kl, recon, kl


def run(pattern: str, replicate: int, out: Path, device: torch.device) -> Path:
    bank = torch.load(out / "bank" / f"pattern_{pattern}.pt", map_location="cpu", weights_only=True)
    maps = bank["importance"].reshape(-1, MASK_DIM).float()
    destination = out / "vae" / f"pattern_{pattern}_rep{replicate}.pt"
    if destination.exists():
        return destination
    seed = 20260924 + 100 * int(pattern, 2) + replicate
    torch.manual_seed(seed)
    order = torch.randperm(len(maps), generator=torch.Generator().manual_seed(seed + 1))
    n_val = max(32, round(.15 * len(maps)))
    train = maps[order[n_val:]].to(device)
    val = maps[order[:n_val]].to(device)
    model = CVAE(MASK_DIM, LATENT, WIDTH, cond_dim=0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, stale, best_epoch = float("inf"), 0, 0
    best_state = None
    history = []
    epochs = tqdm(range(1, MAX_EPOCHS + 1), desc=f"vae {pattern} rep{replicate}",
                  unit="epoch", mininterval=2)
    for epoch in epochs:
        model.train()
        shuffle = torch.randperm(len(train), device=device)
        for batch_ids in shuffle.split(BATCH):
            loss, _, _ = loss_parts(model, train[batch_ids], deterministic=False)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            total, recon, kl = loss_parts(model, val, deterministic=True)
            score = float(total)
        history.append((epoch, score, float(recon), float(kl)))
        if score < best - 1e-4:
            best, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch % 10 == 0:
            epochs.set_postfix(val=f"{score:.2f}", best=f"{best:.2f}")
        if stale >= PATIENCE:
            break
    assert best_state is not None
    payload = {"pattern": pattern, "replicate": replicate, "seed": seed,
               "model_config": {"mask_dim": MASK_DIM, "latent_dim": LATENT,
                                "hidden": WIDTH, "cond_dim": 0},
               "model_state": best_state, "beta": BETA,
               "best_epoch": best_epoch, "best_val_loss": best,
               "train_size": len(train), "val_size": len(val),
               "history": torch.tensor(history)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    print(f"vae {pattern} rep{replicate}: best epoch={best_epoch}, loss={best:.3f}", flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT)
    parser.add_argument("--pattern", choices=PATTERNS)
    parser.add_argument("--replicate", type=int, choices=range(4))
    parser.add_argument("--shard", type=int)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if (args.pattern is None) == (args.shard is None):
        raise ValueError("specify exactly one of --pattern or --shard")
    if args.pattern is not None and args.replicate is None:
        raise ValueError("--replicate is required with --pattern")
    torch.set_num_threads(2)
    patterns = (args.pattern,) if args.pattern else PATTERNS[args.shard::args.shards]
    replicates = (args.replicate,) if args.pattern else range(4)
    for pattern in patterns:
        for replicate in replicates:
            run(pattern, replicate, args.out, torch.device(args.device))


if __name__ == "__main__":
    main()
