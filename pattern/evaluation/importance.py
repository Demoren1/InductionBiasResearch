"""Extract importance maps from the trained (not only selected) MLP weights.

For every trained MLP the ``importance`` is defined as the absolute value of
the first-layer weight at each position where the binary mask is active
(mask=0 positions are set to 0).  The per-MLP map is then normalized to [0, 1]
by dividing by its maximum.

Output: outputs/checkpoints/pattern_{pat}/importance.pt
        a dict containing the key "importance" -> (n, SEQ_LEN, H) tensor,
        the per-MLP val_loss/val_acc (copied from the round checkpoints) and
        "pattern", "n_mlps".

This script assumes training has already produced all round checkpoint files.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def load_rounds(pat: str) -> tuple:
    """Load everything written by train.py for one pattern.

    Returns (w1_all, masks_all, vl_all, va_all) concatenated across rounds.
    """
    ckpt_dir = config.pattern_dir(pat)
    files = sorted(ckpt_dir.glob("gpu*_round*.pt"))
    if not files:
        raise FileNotFoundError(f"No round checkpoints in {ckpt_dir}")
    w1_parts, mask_parts, vl_parts, va_parts = [], [], [], []
    for f in files:
        d = torch.load(f, weights_only=True)
        w1_parts.append(d["params"]["w1"])       # (m, SEQ_LEN, H)  round m
        mask_parts.append(d["masks"])            # (m, SEQ_LEN, H)
        vl_parts.append(d["val_loss"])           # (m,)
        va_parts.append(d["val_acc"])            # (m,)
    return (torch.cat(w1_parts, dim=0), torch.cat(mask_parts, dim=0),
            torch.cat(vl_parts, dim=0), torch.cat(va_parts, dim=0))


def importance_from(w1: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(n, SEQ_LEN, H) weights + masks -> (n, SEQ_LEN, H) importance.

    If config.SIGNED_IMPORTANCE: signed map = W1*mask / max(|W1*mask|) in [-1, 1].
    Otherwise: |W1|*mask / max(|W1|*mask|) in [0, 1].
    """
    raw = w1 * mask.float()
    m = raw.abs().view(raw.size(0), -1).max(dim=1, keepdim=True).values
    m = m.view(-1, 1, 1).clamp_min(1e-9)
    if config.SIGNED_IMPORTANCE:
        return raw / m            # [-1, 1], preserves signs
    return raw.abs() / m          # [0, 1], drops signs


def save_importance(pat: str) -> Path:
    w1, masks, vl, va = load_rounds(pat)
    imp = importance_from(w1, masks)
    out = config.pattern_dir(pat) / "importance.pt"
    torch.save({"pattern": pat, "n_mlps": w1.size(0),
                "val_loss": vl, "val_acc": va, "importance": imp}, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-MLP importance maps per pattern.")
    parser.add_argument("--pattern", type=str, default=None,
                        help="single pattern to process; default: all")
    args = parser.parse_args()
    jobs = [args.pattern] if args.pattern is not None else config.PATTERNS
    for pat in jobs:
        out = save_importance(pat)
        print(f"[importance] pattern={pat}  "
              f"({out.stat().st_size // 1_000} KB saved -> {out})")


if __name__ == "__main__":
    main()