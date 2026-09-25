"""Optimize frozen length-11 VAE decoder logits against one another."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from pattern.length11.settings import EDGES, HIDDEN, ROOT, SEQ_LEN
from pattern.models.cvae import CVAE


PAIRS = (
    ("0000", "0001"), ("0010", "0011"), ("1100", "1101"),
    ("1110", "1111"), ("0100", "1011"), ("0101", "1010"),
    ("0110", "1001"), ("0111", "1000"),
)
STARTS = 32
STEPS = 2000
LR = .03
RADIUS = 12.


def load_models(patterns: tuple[str, ...], replicate: int, out: Path,
                device: torch.device) -> list[CVAE]:
    models = []
    for pattern in patterns:
        data = torch.load(out / "vae" / f"pattern_{pattern}_rep{replicate}.pt",
                          map_location="cpu", weights_only=True)
        model = CVAE(**data["model_config"]).to(device)
        model.load_state_dict(data["model_state"])
        model.eval().requires_grad_(False)
        models.append(model)
    return models


def decode(models: list[CVAE], z: torch.Tensor) -> torch.Tensor:
    n, starts, _ = z.shape
    return torch.stack([model.decode(z[i], z.new_zeros(starts, 0)).reshape(
        starts, SEQ_LEN, HIDDEN) for i, model in enumerate(models)])


def column_orders(logits: torch.Tensor) -> torch.Tensor:
    """Match each model to the first decoder at initialization; then freeze."""
    n, starts = logits.shape[:2]
    orders = torch.arange(HIDDEN, device=logits.device).repeat(n, starts, 1)
    for i in range(1, n):
        for j in range(starts):
            a = logits[0, j].detach().T
            b = logits[i, j].detach().T
            costs = torch.cdist(a, b).square().cpu().numpy()
            rows, cols = linear_sum_assignment(costs)
            orders[i, j, torch.as_tensor(rows, device=logits.device)] = \
                torch.as_tensor(cols, device=logits.device)
    return orders


def align(logits: torch.Tensor, orders: torch.Tensor) -> torch.Tensor:
    return torch.gather(logits, -1,
                        orders[:, :, None, :].expand(-1, -1, SEQ_LEN, -1))


def topk(value: torch.Tensor) -> torch.Tensor:
    original = value.shape
    flat = value.reshape(-1, SEQ_LEN * HIDDEN)
    result = torch.zeros_like(flat)
    result.scatter_(1, flat.topk(EDGES, dim=1).indices, 1.)
    return result.reshape(original)


def run(patterns: tuple[str, ...], replicate: int, out: Path,
        device: torch.device, *, starts: int = STARTS, steps: int = STEPS) -> Path:
    if len(patterns) < 1 or len(set(patterns)) != len(patterns):
        raise ValueError("at least one pattern required")
    label = "_".join(patterns)
    destination = out / "agreement" / label / f"rep{replicate}.pt"
    if destination.exists():
        return destination
    models = load_models(patterns, replicate, out, device)
    generator = torch.Generator(device=device).manual_seed(
        20260925 + int(patterns[0], 2) * 1000 + replicate + len(patterns) * 100)
    z = torch.nn.Parameter(torch.randn(len(models), starts, 32,
                                       generator=generator, device=device))
    initial_z = z.detach().clone()
    with torch.no_grad():
        orders = column_orders(decode(models, z))
    if len(models) == 1:
        with torch.no_grad():
            logits = align(decode(models, z), orders).cpu()
        payload = {
            "patterns": list(patterns), "replicate": replicate,
            "settings": {"starts": starts, "max_steps": 0, "lr": LR,
                         "radius": RADIUS, "alignment": "identity",
                         "loss": "undefined for one VAE; unoptimized latent baseline"},
            "initial_z": initial_z.cpu(), "best_z": initial_z.cpu(),
            "orders": orders.cpu(), "best_loss": torch.zeros(starts),
            "initial_logits": logits, "final_logits": logits,
            "initial_masks": topk(logits), "final_masks": topk(logits),
            "chosen_start": 0, "history": torch.empty((0, 3)),
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_suffix(".tmp")
        torch.save(payload, temp)
        temp.replace(destination)
        print(f"single VAE {label} rep{replicate}: unoptimized latent baseline",
              flush=True)
        return destination
    optimizer = torch.optim.Adam([z], lr=LR)
    best_loss = torch.full((starts,), float("inf"), device=device)
    best_z = z.detach().clone()
    history = []
    stale = 0
    progress = tqdm(range(steps), desc=f"agreement {label} rep{replicate}",
                    unit="step", mininterval=2)
    for step in progress:
        aligned = align(decode(models, z), orders)
        loss = (aligned - aligned.mean(0, keepdim=True)).square().mean((0, 2, 3))
        optimizer.zero_grad(set_to_none=True)
        loss.sum().backward()
        optimizer.step()
        with torch.no_grad():
            z *= (RADIUS / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=1)
            current = align(decode(models, z), orders)
            current_loss = (current - current.mean(0, keepdim=True)).square().mean((0, 2, 3))
            improved = current_loss < best_loss - 1e-8
            best_loss = torch.where(improved, current_loss, best_loss)
            best_z = torch.where(improved[None, :, None], z.detach(), best_z)
            stale = 0 if improved.any() else stale + 1
        if (step + 1) % 100 == 0:
            history.append((step + 1, float(best_loss.mean()), float(best_loss.min())))
            progress.set_postfix(mse=f"{float(best_loss.min()):.5f}")
        if stale >= 250:
            break
    with torch.no_grad():
        final_logits = align(decode(models, best_z), orders)
        initial_logits = align(decode(models, initial_z), orders)
        chosen = int(best_loss.argmin())
        payload = {
            "patterns": list(patterns), "replicate": replicate,
            "settings": {"starts": starts, "max_steps": steps, "lr": LR,
                         "radius": RADIUS, "alignment": "fixed from initial raw decoder logits",
                         "loss": "mean squared raw logits around per-cell ensemble mean"},
            "initial_z": initial_z.cpu(), "best_z": best_z.cpu(),
            "orders": orders.cpu(), "best_loss": best_loss.cpu(),
            "initial_logits": initial_logits.cpu(),
            "final_logits": final_logits.cpu(),
            "initial_masks": topk(initial_logits).cpu(),
            "final_masks": topk(final_logits).cpu(),
            "chosen_start": chosen, "history": torch.tensor(history),
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(destination)
    print(f"agreement {label} rep{replicate}: MSE={float(best_loss.min()):.6f} "
          f"steps={step+1}", flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patterns", nargs="+", choices=tuple(sorted({p for pair in PAIRS for p in pair})))
    parser.add_argument("--replicate", type=int, choices=range(4))
    parser.add_argument("--shard", type=int)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--out", type=Path, default=ROOT)
    args = parser.parse_args()
    if (args.patterns is None) == (args.shard is None):
        raise ValueError("specify exactly one of --patterns or --shard")
    if args.patterns is not None and args.replicate is None:
        raise ValueError("--replicate is required with --patterns")
    torch.set_num_threads(2)
    if args.patterns is not None:
        run(tuple(args.patterns), args.replicate, args.out, torch.device(args.device))
        return
    for pair in PAIRS[args.shard::args.shards]:
        for replicate in range(4):
            run(pair, replicate, args.out, torch.device(args.device))


if __name__ == "__main__":
    main()
