"""Select the best TOP_FRACTION (10%) MLPs per pattern and persist them
together with their trained weights and masks.

Input : all round checkpoint files written by models/train.py
        (outputs/checkpoints/pattern_{pat}/gpu*_round*.pt)
Output: a single file outputs/checkpoints/pattern_{pat}/best10pct.pt
        containing a dict of tensors for the selected MLPs:
            {pattern, global_idx, val_loss, val_acc,
             params:{w1,b1,w2,b2}, masks, top_fraction, n_selected}
        plus a small text summary (best10pct_summary.txt).

Selection criterion: keep the TOP_FRACTION MLPs with the *lowest* validation
BCE (val_loss). The saved val_acc is copied alongside for reporting.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def load_all_checkpoints(pat: str) -> dict:
    """Concatenate every trained round checkpoint for a pattern into one dict.

    Checkpoint keys (written by train.py):
        pattern, global_idx, params{w1,b1,w2,b2}, masks, val_loss, val_acc
    """
    ckpt_dir = config.pattern_dir(pat)
    files = sorted(ckpt_dir.glob("gpu*_round*.pt"))
    if not files:
        raise FileNotFoundError(
            f"No checkpoints in {ckpt_dir}. Run training first.")

    heads = [torch.load(f, weights_only=True) for f in files]
    params = {k: torch.cat([h["params"][k] for h in heads], dim=0)
              for k in heads[0]["params"]}
    return {
        "pattern": heads[0]["pattern"],
        "global_idx": torch.cat([h["global_idx"] for h in heads]),
        "val_loss": torch.cat([h["val_loss"] for h in heads]),
        "val_acc": torch.cat([h["val_acc"] for h in heads]),
        "masks": torch.cat([h["masks"] for h in heads]),
        "params": params,
    }


def select_best(all_ckpt: dict, top_fraction: float) -> dict:
    """Keep the top `top_fraction` MLPs by lowest val_loss (BCE)."""
    n = all_ckpt["val_loss"].numel()
    n_keep = max(1, int(round(n * top_fraction)))
    idx = torch.argsort(all_ckpt["val_loss"])[:n_keep]

    def sel(key):
        v = all_ckpt[key]
        return v[idx] if torch.is_tensor(v) else v

    best = {k: sel(k) for k in ("global_idx", "val_loss", "val_acc", "masks")}
    best["params"] = {p: all_ckpt["params"][p][idx]
                      for p in all_ckpt["params"]}
    best["pattern"] = all_ckpt["pattern"]
    best["top_fraction"] = top_fraction
    best["n_selected"] = n_keep
    return best


def select(pat: str, top_fraction: float) -> Path:
    """Select best fraction for one pattern and persist weights + masks."""
    all_ckpt = load_all_checkpoints(pat)
    best = select_best(all_ckpt, top_fraction)

    out = config.pattern_dir(pat) / "best10pct.pt"
    torch.save(best, out)

    vl, va = best["val_loss"], best["val_acc"]
    summary = config.pattern_dir(pat) / "best10pct_summary.txt"
    summary.write_text(
        f"pattern={pat}\n"
        f"total_mlps={all_ckpt['val_loss'].numel()}\n"
        f"n_selected={best['n_selected']} (top {top_fraction:.0%})\n"
        f"val_loss (BCE)  min={vl.min():.6f} p50={vl.median():.6f} "
        f"max={vl.max():.6f}\n"
        f"val_acc         min={va.min():.6f} p50={va.median():.6f} "
        f"max={va.max():.6f}\n"
    )
    print(f"[select] pattern={pat}: kept {best['n_selected']} MLPs, "
          f"val_bce in [{vl.min():.6f}, {vl.max():.6f}], "
          f"val_acc in [{va.min():.6f}, {va.max():.6f}]")
    print(f"[select] saved -> {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Select best 10% per pattern.")
    parser.add_argument("--pattern", type=str, default=None,
                        help="which pattern to process; if omitted, all "
                             "config.PATTERNS are processed")
    parser.add_argument("--top_fraction", type=float, default=config.TOP_FRACTION)
    args = parser.parse_args()

    jobs = [args.pattern] if args.pattern is not None else config.PATTERNS
    for pat in jobs:
        select(pat, args.top_fraction)


if __name__ == "__main__":
    main()