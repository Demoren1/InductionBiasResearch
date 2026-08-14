"""Extract importance maps from the trained (not only selected) MLP weights.

For every trained MLP the ``importance`` is defined as the absolute value of
the first-layer weight at each position where the binary mask is active (mask=0
positions are set to 0).  The per-MLP map is then normalised to [0, 1] by
dividing by its maximum.

Output: outputs/checkpoints/kernel_{k}/offset_{s}/importance.pt
        a dict containing the key "importance" -> (n, L, H) tensor and
        "kernel", "offset", "n_mlps", "val_loss" (copied from the round
        checkpoints).

This script assumes that scripts/02_train.sh has already produced all round
checkpoint files.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def load_rounds(kernel: int, offset: int) -> tuple:
    """Load everything written by train.py for one (kernel, offset).

    Returns (w1_all, masks_all, vl_all) concatenated across rounds.
    """
    ckpt_dir = config.kernel_dir(kernel, offset)
    files = sorted(ckpt_dir.glob("gpu*_round*.pt"))
    if not files:
        raise FileNotFoundError(f"No round checkpoints in {ckpt_dir}")
    w1_parts, mask_parts, vl_parts = [], [], []
    for f in files:
        d = torch.load(f, weights_only=True)
        w1_parts.append(d["params"]["w1"])       # (m, L, H)    round m
        mask_parts.append(d["masks"])             # (m, L, H)
        vl_parts.append(d["val_loss"])            # (m,)
    return (torch.cat(w1_parts, dim=0), torch.cat(mask_parts, dim=0),
            torch.cat(vl_parts, dim=0))


def importance_from(w1: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(n, L, H) weights + masks -> (n, L, H) importance in [0, 1].

    importance[i, j, h] = |w1[i, j, h]| where mask == 1, else 0,
    normalised by the per-MLP maximum so every MLP's map lives in [0, 1].
    """
    imp = w1.abs() * mask.float()
    m = imp.view(imp.size(0), -1).max(dim=1, keepdim=True).values
    m = m.view(-1, 1, 1).clamp_min(1e-9)
    return imp / m


def save_importance(kernel: int, offset: int) -> Path:
    w1, masks, vl = load_rounds(kernel, offset)
    imp = importance_from(w1, masks)
    out = config.kernel_dir(kernel, offset) / "importance.pt"
    torch.save({"kernel": kernel, "offset": offset, "n_mlps": w1.size(0),
                "val_loss": vl, "importance": imp}, out)
    return out


def main() -> None:
    for k in config.KERNELS:
        for s in config.OFFSETS:
            out = save_importance(k, s)
            print(f"[importance] kernel={k} offset={s}  "
                  f"({out.stat().st_size // 1_000_000} MB saved -> {out})")


if __name__ == "__main__":
    main()