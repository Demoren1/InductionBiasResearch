"""Select the best TOP_FRACTION (10%) MLPs per (kernel, offset) and persist
them together with their trained weights and masks.

Input : all checkpoint files written by models/train.py
        (outputs/checkpoints/kernel_{k}/offset_{s}/gpu*_round*.pt)
Output: a single file outputs/checkpoints/kernel_{k}/offset_{s}/best10pct.pt
        containing a dict of tensors for the selected MLPs:
            {kernel, offset, global_idx, val_loss, params:{w1,b1,w2,b2}, masks}
        plus a small text summary.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def load_all_checkpoints(kernel: int, offset: int = None) -> list:
    """Concatenate every trained checkpoint for the (kernel, offset) into one dict."""
    ckpt_dir = config.kernel_dir(kernel, offset)
    files = sorted(ckpt_dir.glob("gpu*_round*.pt"))
    if not files:
        raise FileNotFoundError(
            f"No checkpoints in {ckpt_dir}. Run scripts/02_train.sh first.")

    heads = [torch.load(f) for f in files]
    params = {k: torch.cat([h["params"][k] for h in heads], dim=0)
              for k in heads[0]["params"]}
    return {
        "kernel": kernel,
        "offset": offset,
        "global_idx": torch.cat([h["global_idx"] for h in heads]),
        "val_loss": torch.cat([h["val_loss"] for h in heads]),
        "masks": torch.cat([h["masks"] for h in heads]),
        "params": params,
    }


def select_best(all_ckpt: dict, kernel: int, top_fraction: float) -> dict:
    n = all_ckpt["val_loss"].numel()
    n_keep = max(1, int(round(n * top_fraction)))
    idx = torch.argsort(all_ckpt["val_loss"])[:n_keep]

    def sel(key):
        v = all_ckpt[key]
        return v[idx] if torch.is_tensor(v) else v

    best = {k: sel(k) for k in ("global_idx", "val_loss", "masks")}
    best["params"] = {p: all_ckpt["params"][p][idx]
                      for p in all_ckpt["params"]}
    best["kernel"] = kernel
    best["offset"] = all_ckpt["offset"]
    best["top_fraction"] = top_fraction
    best["n_selected"] = n_keep
    return best


def select(kernel: int, offset: int, top_fraction: float) -> Path:
    """Select best 10% for one (kernel, offset) and persist weights + masks."""
    all_ckpt = load_all_checkpoints(kernel, offset)
    best = select_best(all_ckpt, kernel, top_fraction)

    out = config.kernel_dir(kernel, offset) / "best10pct.pt"
    torch.save(best, out)

    top = best["val_loss"]
    summary = config.kernel_dir(kernel, offset) / "best10pct_summary.txt"
    summary.write_text(
        f"kernel={kernel}\n"
        f"offset={offset}\n"
        f"total_mlps={all_ckpt['val_loss'].numel()}\n"
        f"n_selected={best['n_selected']} "
        f"(top {top_fraction:.0%})\n"
        f"val_mse  min={top.min():.6f} "
        f"p50={top.median():.6f} max={top.max():.6f}\n"
    )
    print(f"[select] kernel={kernel} offset={offset}: "
          f"kept {best['n_selected']} MLPs, "
          f"val_mse in [{top.min():.6f}, {top.max():.6f}]")
    print(f"[select] saved -> {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Select best 10% per kernel.")
    parser.add_argument("--kernel", type=int, choices=config.KERNELS,
                        default=None)
    parser.add_argument("--offset", type=int, choices=config.OFFSETS,
                        default=None,
                        help="target offset of the shifted-MA task; if omitted, "
                             "all KERNELS x OFFSETS are processed")
    parser.add_argument("--top_fraction", type=float, default=config.TOP_FRACTION)
    args = parser.parse_args()

    if args.offset is not None:
        if args.kernel is not None:
            jobs = [(args.kernel, args.offset)]
        else:
            jobs = [(k, args.offset) for k in config.KERNELS]
    else:
        jobs = [(k, s) for k in config.KERNELS for s in config.OFFSETS]

    for kernel, offset in jobs:
        select(kernel, offset, args.top_fraction)


if __name__ == "__main__":
    main()