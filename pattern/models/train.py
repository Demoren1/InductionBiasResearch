"""Train N_MLPS_PER_PATTERN masked MLPs per pattern, spread over NUM_GPUS.

One process is launched per GPU (see scripts/02_train.sh).
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from models.mlp import (  # noqa: E402
    BatchedMaskedMLP, get_train_batch, generate_masks,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train masked MLPs on pattern task.")
    p.add_argument("--pattern", type=str, required=True)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--num_gpus", type=int, default=config.NUM_GPUS)
    p.add_argument("--mlps_per_gpu", type=int, default=config.MLPS_PER_GPU)
    p.add_argument("--train_steps", type=int, default=config.TRAIN_STEPS)
    p.add_argument("--batch_size", type=int, default=config.TRAIN_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LR)
    return p


def own_mlp_slice(n_total: int, gpu_id: int, num_gpus: int) -> tuple:
    base = n_total // num_gpus
    rem = n_total % num_gpus
    start = gpu_id * base + min(gpu_id, rem)
    count = base + (1 if gpu_id < rem else 0)
    return start, count


def train_round(model, pat, steps, batch_size, lr, seed, x_val, y_val):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(steps):
        xb, yb = get_train_batch(pat, batch_size, seed + step)
        xb, yb = xb.to(model.w1.device), yb.to(model.w1.device)
        pred = model(xb)
        loss = F.binary_cross_entropy_with_logits(
            pred, yb.unsqueeze(1).expand_as(pred))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step and step % config.EVAL_EVERY == 0:
            v = model.val_loss(x_val, y_val, config.VAL_BATCH_SIZE).mean().item()
            a = model.val_acc(x_val, y_val, config.VAL_BATCH_SIZE).mean().item()
            print(f"    [pattern={pat}] step {step}/{steps} "
                  f"train_bce={loss.item():.5f} val_bce={v:.5f} val_acc={a:.4f}",
                  flush=True)
    v = model.val_loss(x_val, y_val, config.VAL_BATCH_SIZE).mean().item()
    a = model.val_acc(x_val, y_val, config.VAL_BATCH_SIZE).mean().item()
    return v, a


def main() -> None:
    args = build_parser().parse_args()
    if args.pattern not in config.PATTERNS:
        raise ValueError(f"unknown pattern {args.pattern}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    dev = torch.device("cuda:0")
    print(f"[gpu {args.gpu_id}] pattern={args.pattern}", flush=True)

    val_p = config.val_path(args.pattern)
    if not val_p.exists():
        raise FileNotFoundError(f"Run scripts/01_generate_data.sh first. Missing {val_p}")
    val = torch.load(val_p, weights_only=False)
    x_val, y_val = val["x"].to(dev), val["y"].to(dev)

    start, count = own_mlp_slice(config.N_MLPS_PER_PATTERN,
                                 args.gpu_id, args.num_gpus)
    print(f"[gpu {args.gpu_id}] pattern={args.pattern} "
          f"owns MLPs [{start}, {start + count})", flush=True)

    out_dir = config.pattern_dir(args.pattern)
    out_dir.mkdir(parents=True, exist_ok=True)

    distinct_masks = generate_masks(
        config.N_MASKS_PER_PATTERN, config.SEQ_LEN, config.H, config.P,
        seed=int(args.pattern, 2) * 1_000_003)
    mask_of_global = torch.arange(config.N_MLPS_PER_PATTERN) // config.MLPS_PER_MASK

    t0 = time.time()
    for rnd, offset in enumerate(range(0, count, args.mlps_per_gpu)):
        n_here = min(args.mlps_per_gpu, count - offset)
        global_start = start + offset
        print(f"[gpu {args.gpu_id}] round {rnd}: training {n_here} MLPs "
              f"(global {global_start}..{global_start + n_here}) "
              f"pattern={args.pattern}", flush=True)

        base_seed = int(args.pattern, 2) * 1_000_003 + global_start * 7
        masks = distinct_masks[mask_of_global[global_start:global_start + n_here]]
        model = BatchedMaskedMLP(n_here, config.SEQ_LEN, config.H).to(dev)
        model.load_masks(masks)

        t_round = time.time()
        mean_val, mean_acc = train_round(
            model, args.pattern, args.train_steps, args.batch_size,
            args.lr, seed=base_seed, x_val=x_val, y_val=y_val)

        ckpt = {
            "pattern": args.pattern,
            "global_idx": torch.arange(global_start, global_start + n_here),
            "params": BatchedMaskedMLP.state_as_dict(
                model.w1.data, model.b1.data, model.w2.data, model.b2.data),
            "masks": masks.cpu(),
            "val_loss": model.val_loss(x_val, y_val, config.VAL_BATCH_SIZE).cpu(),
            "val_acc": model.val_acc(x_val, y_val, config.VAL_BATCH_SIZE).cpu(),
        }
        path = out_dir / f"gpu{args.gpu_id}_round{rnd:03d}.pt"
        torch.save(ckpt, path)
        print(f"[gpu {args.gpu_id}] round {rnd} done in "
              f"{time.time() - t_round:.1f}s, val_bce={mean_val:.5f} "
              f"val_acc={mean_acc:.4f} -> {path}", flush=True)

    print(f"[gpu {args.gpu_id}] pattern={args.pattern} "
          f"finished {count} MLPs in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
