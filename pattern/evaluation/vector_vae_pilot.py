"""Compare a vector-output VAE with a matrix-output VAE on held-out patterns.

Both VAEs train on the same top-10% importance maps from 12 meta-train tasks,
after hidden-column alignment to a train-only reference. The vector decoder
outputs 8+8 continuous scores; their outer product gives an 8x8 score map.
Each sampled map becomes an exact-32 binary mask for a fresh task MLP.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from data.generate import ideal_mask, make_dataset  # noqa: E402
from evaluation.align_importance import align_map  # noqa: E402
from evaluation.ood_split import make_split  # noqa: E402
from models.cvae import CVAE, kl_divergence  # noqa: E402
from models.mlp import BatchedMaskedMLP, generate_fixed_sparsity_masks  # noqa: E402

METHODS = ("vector_vae", "matrix_vae", "random", "ideal")


class VectorVAE(CVAE):
    def __init__(self, latent_dim: int, hidden: int):
        super().__init__(64, latent_dim, hidden, cond_dim=0)
        self.dec_out = nn.Linear(hidden, 16)

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        factors = torch.sigmoid(logits)
        return (factors[:, :8, None] * factors[:, None, 8:]).reshape(-1, 64)


def load_aligned_train_maps(patterns: list[str]) -> tuple[torch.Tensor, dict]:
    selected = []
    references = []
    for pattern in patterns:
        path = config.pattern_dir(pattern) / "importance.pt"
        payload = torch.load(path, weights_only=True, map_location="cpu")
        count = max(1, round(len(payload["val_loss"]) * 0.1))
        indices = torch.argsort(payload["val_loss"])[:count]
        selected.append(payload["importance"][indices].float())
        best = int(torch.argmin(payload["val_loss"]))
        references.append((float(payload["val_loss"][best]),
                           pattern, payload["importance"][best].float()))
    reference = min(references, key=lambda item: item[0])[2]
    maps = torch.cat(selected)
    # Two reference-refinement passes, both based only on meta-train maps.
    for _ in range(2):
        maps = torch.stack([align_map(item, reference) for item in maps])
        reference = maps.mean(0)
    return maps.flatten(1), {
        "train_map_count": len(maps),
        "reference_from": min(references, key=lambda item: item[0])[1],
        "alignment": "two Hungarian hidden-column passes; train maps only",
    }


def train_vae(model: CVAE, x: torch.Tensor, args: argparse.Namespace,
              seed: int, device: torch.device) -> tuple[CVAE, dict]:
    n = len(x)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n, generator=generator)
    n_val = max(1, round(n * 0.15))
    train_idx, val_idx = permutation[n_val:], permutation[:n_val]
    x = x.to(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.vae_lr)
    best_loss = float("inf")
    best_state = None
    best_epoch = 0

    def loss_for(batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dummy_condition = torch.zeros(len(batch), 4, device=device)
        logits, mu, logvar = model(batch, dummy_condition)
        probabilities = (model.probabilities(logits) if isinstance(model, VectorVAE)
                         else torch.sigmoid(logits))
        reconstruction = F.binary_cross_entropy(
            probabilities.clamp(1e-6, 1 - 1e-6), batch, reduction="none").sum(1).mean()
        kl = kl_divergence(mu, logvar)
        return reconstruction + args.beta * kl, reconstruction, kl

    for epoch in range(1, args.vae_epochs + 1):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), generator=generator)]
        for indices in order.split(args.vae_batch_size):
            loss, _, _ = loss_for(x[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss, val_recon, val_kl = loss_for(x[val_idx])
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
            best_recon, best_kl = val_recon.item(), val_kl.item()
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "best_epoch": best_epoch,
        "val_loss": best_loss,
        "val_reconstruction_bce_sum": best_recon,
        "val_kl": best_kl,
    }


@torch.no_grad()
def sample_masks(model: CVAE, n: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn(n, model.latent_dim, generator=generator, device=device)
    c = torch.empty(n, 0, device=device)
    logits = model.decode(z, c)
    probabilities = (model.probabilities(logits) if isinstance(model, VectorVAE)
                     else torch.sigmoid(logits))
    indices = probabilities.topk(32, dim=1).indices
    masks = torch.zeros_like(probabilities).scatter_(1, indices, 1.0)
    return masks.reshape(n, 8, 8)


@torch.no_grad()
def evaluate(model: BatchedMaskedMLP, data: dict, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = data["x"].to(model.mask.device)
    y = data["y"].to(model.mask.device)
    bce = torch.zeros(len(model.w1), device=x.device)
    correct = torch.zeros_like(bce)
    for start in range(0, len(x), batch_size):
        xb, yb = x[start:start + batch_size], y[start:start + batch_size]
        logits = model(xb)
        target = yb[:, None].expand_as(logits)
        bce += F.binary_cross_entropy_with_logits(logits, target, reduction="none").sum(0)
        correct += ((logits > 0) == (target > 0.5)).sum(0)
    return bce / len(x), correct / len(x)


def evaluate_downstream(pattern: str, models: dict[str, CVAE], args: argparse.Namespace,
                        device: torch.device) -> dict:
    n = args.masks_per_method
    masks_by_method = {
        "vector_vae": sample_masks(models["vector_vae"], n,
                                   args.seed * 1000 + int(pattern, 2), device),
        "matrix_vae": sample_masks(models["matrix_vae"], n,
                                   args.seed * 1000 + int(pattern, 2), device),
        "random": generate_fixed_sparsity_masks(
            n, 8, 8, 32, seed=args.seed * 1000 + int(pattern, 2)).to(device),
        "ideal": ideal_mask().float().to(device).expand(n, -1, -1),
    }
    all_masks = torch.cat([masks_by_method[name] for name in METHODS])
    torch.manual_seed(args.seed * 10000 + int(pattern, 2))
    model = BatchedMaskedMLP(len(all_masks), 8, 8).to(device)
    model.load_masks(all_masks)
    # Match initial weights across methods for each mask index.
    with torch.no_grad():
        model.w1.copy_(model.w1[:n].repeat(len(METHODS), 1, 1))
        model.w2.copy_(model.w2[:n].repeat(len(METHODS), 1, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.mlp_lr)
    validation = make_dataset(pattern, args.eval_samples,
                              seed=400000 + int(pattern, 2),
                              pos_fraction=config.POS_FRACTION)
    test = make_dataset(pattern, args.eval_samples,
                        seed=500000 + int(pattern, 2),
                        pos_fraction=config.POS_FRACTION)
    best_val = torch.full((len(all_masks),), float("inf"), device=device)
    best_w1 = torch.empty_like(model.w1)
    best_b1 = torch.empty_like(model.b1)
    best_w2 = torch.empty_like(model.w2)
    best_b2 = torch.empty_like(model.b2)
    best_step = torch.zeros(len(all_masks), dtype=torch.int64, device=device)
    for step in range(1, args.mlp_steps + 1):
        batch = make_dataset(pattern, args.mlp_batch_size,
                             seed=600000 + int(pattern, 2) * 10000 + step,
                             pos_fraction=config.POS_FRACTION)
        x, y = batch["x"].to(device), batch["y"].to(device)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y[:, None].expand_as(logits))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % args.eval_every == 0 or step == args.mlp_steps:
            model.eval()
            val_bce, _ = evaluate(model, validation, args.eval_batch_size)
            improved = val_bce < best_val
            best_val = torch.where(improved, val_bce, best_val)
            best_w1[improved] = model.w1.detach()[improved]
            best_b1[improved] = model.b1.detach()[improved]
            best_w2[improved] = model.w2.detach()[improved]
            best_b2[improved] = model.b2.detach()[improved]
            best_step[improved] = step
            model.train()
    with torch.no_grad():
        model.w1.copy_(best_w1)
        model.b1.copy_(best_b1)
        model.w2.copy_(best_w2)
        model.b2.copy_(best_b2)
    test_bce, test_acc = evaluate(model, test, args.eval_batch_size)
    gold = ideal_mask().float().to(device)
    result = {}
    for index, name in enumerate(METHODS):
        sl = slice(index * n, (index + 1) * n)
        masks = masks_by_method[name]
        intersection = (masks * gold).sum((1, 2))
        union = masks.sum((1, 2)) + gold.sum() - intersection
        result[name] = {
            "test_acc": float(test_acc[sl].mean()),
            "test_bce": float(test_bce[sl].mean()),
            "test_acc_masks": test_acc[sl].cpu().tolist(),
            "best_steps": best_step[sl].cpu().tolist(),
            "gold_iou": float((intersection / union).mean()),
            "unique_masks": len(torch.unique(masks.flatten(1), dim=0)),
            "active_connections": masks.sum((1, 2)).cpu().tolist(),
            "first_mask": masks[0].cpu().int().tolist(),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vae-epochs", type=int, default=80)
    parser.add_argument("--vae-batch-size", type=int, default=128)
    parser.add_argument("--vae-lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--masks-per-method", type=int, default=64)
    parser.add_argument("--mlp-steps", type=int, default=2000)
    parser.add_argument("--mlp-batch-size", type=int, default=128)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-samples", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--out-dir", type=Path,
                        default=config.OUTPUTS / "vector_vae_pilot" / "split42")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    split = make_split(args.split_seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    x, alignment = load_aligned_train_maps(split["train_patterns"])
    models = {}
    training = {}
    for name, architecture in (
        ("vector_vae", VectorVAE(args.latent_dim, args.hidden)),
        ("matrix_vae", CVAE(64, args.latent_dim, args.hidden, cond_dim=0)),
    ):
        torch.manual_seed(args.seed)
        models[name], training[name] = train_vae(
            architecture, x, args, args.seed, device)
        torch.save(models[name].cpu().state_dict(), args.out_dir / f"{name}.pt")
        models[name].to(device).eval()
        print(f"[vector-vae] {name} train: {training[name]}", flush=True)
    summary = {
        "protocol": {
            "split": split,
            "alignment": alignment,
            "seed": args.seed,
            "vae_epochs": args.vae_epochs,
            "beta": args.beta,
            "latent_dim": args.latent_dim,
            "hidden": args.hidden,
            "masks_per_method": args.masks_per_method,
            "mlp_steps": args.mlp_steps,
            "sparse_connections": 32,
            "mask_selection": "top-32 of decoder score map",
            "vector_decoder": "sigmoid(first 8) outer sigmoid(last 8)",
            "evaluation": "validation-selected MLP checkpoint, fresh test data",
            "device": str(device),
        },
        "vae_training": training,
        "patterns": {},
    }
    out = args.out_dir / "summary.json"
    for pattern in split["test_patterns"]:
        row = evaluate_downstream(pattern, models, args, device)
        summary["patterns"][pattern] = row
        out.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[vector-vae] {pattern} " + " ".join(
            f"{name}={row[name]['test_acc']:.4f}" for name in METHODS), flush=True)
    summary["macro_test_acc"] = {
        name: sum(row[name]["test_acc"] for row in summary["patterns"].values())
        / len(summary["patterns"]) for name in METHODS
    }
    summary["macro_test_bce"] = {
        name: sum(row[name]["test_bce"] for row in summary["patterns"].values())
        / len(summary["patterns"]) for name in METHODS
    }
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print("[vector-vae] macro", summary["macro_test_acc"], flush=True)


if __name__ == "__main__":
    main()
