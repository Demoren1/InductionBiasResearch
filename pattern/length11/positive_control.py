"""Check whether each frozen length-11 decoder can express the analytic mask."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from pattern.length11.agreement import column_orders, load_models, topk
from pattern.length11.evaluate import gold_iou, gold_mask
from pattern.length11.settings import HIDDEN, PATTERNS, ROOT, SEQ_LEN


def run(pattern: str, replicate: int, out: Path, device: torch.device) -> Path:
    destination = out / "positive_control" / f"pattern_{pattern}_rep{replicate}.pt"
    if destination.exists():
        return destination
    model = load_models((pattern,), replicate, out, device)[0]
    generator = torch.Generator(device=device).manual_seed(
        20260928 + int(pattern, 2) * 100 + replicate)
    z = torch.nn.Parameter(torch.randn(64, 32, generator=generator, device=device))
    gold = gold_mask().to(device).expand(64, -1, -1)
    with torch.no_grad():
        logits = model.decode(z, z.new_zeros(64, 0)).reshape(64, SEQ_LEN, HIDDEN)
        orders = column_orders(torch.stack((gold, logits)))[1]
    optimizer = torch.optim.Adam([z], lr=.05)
    best = torch.full((64,), float("inf"), device=device)
    best_z = z.detach().clone()
    progress = tqdm(range(3000), desc=f"gold control {pattern} rep{replicate}",
                    unit="step", mininterval=2)
    for step in progress:
        logits = model.decode(z, z.new_zeros(64, 0)).reshape(64, SEQ_LEN, HIDDEN)
        aligned = logits.gather(-1, orders[:, None, :].expand(-1, SEQ_LEN, -1))
        loss = F.binary_cross_entropy_with_logits(
            aligned, gold, reduction="none").mean((1, 2))
        optimizer.zero_grad(set_to_none=True)
        loss.sum().backward()
        optimizer.step()
        with torch.no_grad():
            z *= (12. / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=1)
            improved = loss < best
            best = torch.where(improved, loss, best)
            best_z = torch.where(improved[:, None], z.detach(), best_z)
        if (step + 1) % 500 == 0:
            progress.set_postfix(bce=f"{float(best.min()):.3f}")
    with torch.no_grad():
        logits = model.decode(best_z, best_z.new_zeros(64, 0)).reshape(64, SEQ_LEN, HIDDEN)
        aligned = logits.gather(-1, orders[:, None, :].expand(-1, SEQ_LEN, -1))
        losses = F.binary_cross_entropy_with_logits(
            aligned, gold, reduction="none").mean((1, 2))
        j = int(losses.argmin())
        mask = topk(aligned[j]).cpu()
    payload = {"pattern": pattern, "replicate": replicate, "best_bce": float(losses[j]),
               "best_iou": gold_iou(mask), "best_mask": mask, "best_z": best_z[j].cpu(),
               "settings": {"starts": 64, "steps": 3000, "lr": .05,
                            "radius": 12., "target": "analytic mask, positive control only"}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(destination)
    print(f"gold control {pattern} rep{replicate}: IoU={payload['best_iou']:.3f}",
          flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", choices=PATTERNS)
    parser.add_argument("--replicate", type=int, choices=range(4))
    parser.add_argument("--shard", type=int)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--out", type=Path, default=ROOT)
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
