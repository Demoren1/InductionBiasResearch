"""Train N_MLPS_PER_KERNEL masked MLPs per kernel, spread over NUM_GPUS.

One process is launched per GPU (see scripts/02_train.sh). Each process
owns ``base = N_MLPS_PER_KERNEL // NUM_GPUS`` MLPs plus one extra if its
gpu_id < N_MLPS_PER_KERNEL % NUM_GPUS.

Inside a process, MLPs are processed in rounds of ``--mlps_per_gpu``:
the round instantiates a BatchedMaskedMLP, trains it for TRAIN_STEPS
steps with a single batched forward/backward, evaluates per-MLP
validation MSE against the fixed validation set, and dumps the
trained parameters + masks to a checkpoint file.

Run:  CUDA_VISIBLE_DEVICES=$GPU python -m models.train \\
          --kernel 3 --offset 2 --gpu_id 0
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
    BatchedMaskedMLP, get_train_batch,
    generate_masks,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train masked MLPs on MA task.")
    p.add_argument("--kernel", type=int, choices=config.KERNELS, required=True)
    p.add_argument("--offset", type=int, choices=config.OFFSETS, required=True,
                   help="offset of the shifted-MA target")
    p.add_argument("--gpu_id", type=int, default=0,
                   help="GPU index within CUDA_VISIBLE_DEVICES")
    p.add_argument("--num_gpus", type=int, default=config.NUM_GPUS)
    p.add_argument("--mlps_per_gpu", type=int, default=config.MLPS_PER_GPU)
    p.add_argument("--train_steps", type=int, default=config.TRAIN_STEPS)
    p.add_argument("--batch_size", type=int, default=config.TRAIN_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LR)
    return p


def own_mlp_slice(n_total: int, gpu_id: int, num_gpus: int) -> tuple:
    """(start_idx, count) of the global MLP indices assigned to this GPU."""
    base = n_total // num_gpus
    rem = n_total % num_gpus
    start = gpu_id * base + min(gpu_id, rem)
    count = base + (1 if gpu_id < rem else 0)
    return start, count


def train_round(model: BatchedMaskedMLP, kernel: int, offset: int, steps: int,
                batch_size: int, lr: float, seed: int,
                x_val: torch.Tensor, y_val: torch.Tensor) -> float:
    """Train one round of MLPs; returns the mean validation MSE."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(steps):
        xb, yb = get_train_batch(kernel, offset, batch_size,
                                 config.L, seed + step)
        xb, yb = xb.to(model.w1.device), yb.to(model.w1.device)
        pred = model(xb)                                   # (B, M)
        loss = F.mse_loss(pred, yb.unsqueeze(1).expand_as(pred))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step and step % config.EVAL_EVERY == 0:
            v = model.val_loss(x_val, y_val, config.VAL_BATCH_SIZE).mean().item()
            print(f"    [kernel={kernel} offset={offset}] step {step}/{steps} "
                  f"train_mse={loss.item():.5f} val_mse={v:.5f}", flush=True)
    v = model.val_loss(x_val, y_val, config.VAL_BATCH_SIZE)
    return v.mean().item()


def main() -> None:
    args = build_parser().parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required (4 x A100 expected).")
    # Each process is launched with CUDA_VISIBLE_DEVICES=<one physical card>
    # (see scripts/02_train.sh), so our visible device is always index 0.
    # args.gpu_id is used only to select which 1/num_gpus slice of MLPs to
    # train and which filename prefix to write.
    dev = torch.device("cuda:0")
    name = torch.cuda.get_device_name(dev)
    smem = torch.cuda.get_device_properties(dev).total_memory / 1e9
    print(f"[gpu {args.gpu_id}] {name} ({smem:.0f} GB)", flush=True)

    # Validation set: identical for every MLP of this (kernel, offset).
    val_path = config.DATA_DIR / f"val_kernel_{args.kernel}_offset_{args.offset}.pt"
    if not val_path.exists():
        raise FileNotFoundError(
            f"Run scripts/01_generate_data.sh first. Missing {val_path}")
    val = torch.load(val_path)
    x_val, y_val = val["x"].to(dev), val["y"].to(dev)

    start, count = own_mlp_slice(config.N_MLPS_PER_KERNEL,
                                 args.gpu_id, args.num_gpus)
    print(f"[gpu {args.gpu_id}] kernel={args.kernel} offset={args.offset} "
          f"owns global MLPs [{start}, {start + count})", flush=True)

    out_dir = config.kernel_dir(args.kernel, args.offset)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Distinct binary masks for this kernel (drawn once, seeded by kernel).
    # global MLP index g uses mask g // MLPS_PER_MASK, i.e. every mask is
    # reused for MLPS_PER_MASK=10 differently-initialised MLPs.
    distinct_masks = generate_masks(config.N_MASKS_PER_KERNEL,
                                    config.L, config.H, config.P,
                                    seed=args.kernel * 1_000_003)
    mask_of_global = torch.arange(config.N_MLPS_PER_KERNEL) // config.MLPS_PER_MASK

    t0 = time.time()
    for rnd, offset in enumerate(range(0, count, args.mlps_per_gpu)):
        n_here = min(args.mlps_per_gpu, count - offset)
        global_start = start + offset
        print(f"[gpu {args.gpu_id}] round {rnd}: training {n_here} MLPs "
              f"(global {global_start}..{global_start + n_here}) "
              f"kernel={args.kernel} offset={args.offset}", flush=True)

        base_seed = args.kernel * 1_000_003 + global_start * 7
        masks = distinct_masks[mask_of_global[global_start:global_start + n_here]]
        model = BatchedMaskedMLP(n_here, config.L, config.H).to(dev)
        model.load_masks(masks)

        t_round = time.time()
        mean_val = train_round(
            model, args.kernel, args.offset, args.train_steps, args.batch_size,
            args.lr, seed=base_seed, x_val=x_val, y_val=y_val)

        ckpt = {
            "kernel": args.kernel,
            "offset": args.offset,
            "global_idx": torch.arange(global_start, global_start + n_here),
            "params": BatchedMaskedMLP.state_as_dict(
                model.w1.data, model.b1.data, model.w2.data, model.b2.data),
            "masks": masks.cpu(),
            "val_loss": model.val_loss(x_val, y_val,
                                       config.VAL_BATCH_SIZE).cpu(),
        }
        path = out_dir / f"gpu{args.gpu_id}_round{rnd:03d}.pt"
        torch.save(ckpt, path)
        print(f"[gpu {args.gpu_id}] round {rnd} done in "
              f"{time.time() - t_round:.1f}s, val_mse={mean_val:.5f} "
              f"-> {path}", flush=True)

    print(f"[gpu {args.gpu_id}] kernel={args.kernel} offset={args.offset} "
          f"finished {count} MLPs in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()