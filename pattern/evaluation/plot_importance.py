"""Row-profile and mean-map plots of the learned importance maps.

For every pattern in config.PATTERNS, load
outputs/checkpoints/pattern_{pat}/importance.pt and visualize how the learned
first-layer importance is distributed across input positions.

Two figures are written to outputs/plots/importance/:

  1) importance_row_profile.png  -- 4x4 small multiples, one panel per pattern.
     x = input position 0..SEQ_LEN-1, y = mean importance per position
     (averaged over the hidden dimension and over all trained MLPs). Faint
     vertical ticks mark
     the N_WINDOWS sliding-window start positions, so one can see whether the
     mass concentrates on the local 4-tap windows (Toeplitz band) or spreads
     over the whole sequence.

  2) importance_maps_mean.png  -- 4x4 grid of the mean importance map
     (SEQ_LEN x H) per pattern, viridis 0..1 with a shared colorbar; the title
     is the pattern string. Toeplitz bands become visible when the
     learned masks align with the sliding-window support.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

NCOLS = 4
PATTERN_COLOR = "#1a7f37"


def row_profile(pat: str) -> torch.Tensor:
    """Mean importance per input row: (SEQ_LEN,) tensor for one pattern."""
    d = torch.load(config.pattern_dir(pat) / "importance.pt",
                   weights_only=True)
    imp = d["importance"]                      # (n, SEQ_LEN, H)
    return imp.mean(dim=(0, 2))                # average over hidden + MLPs


def mean_map(pat: str) -> np.ndarray:
    """(SEQ_LEN, H) mean importance over MLPs for one pattern."""
    d = torch.load(config.pattern_dir(pat) / "importance.pt",
                   weights_only=True)
    return d["importance"].mean(dim=0).numpy()


def plot_row_profiles() -> Path:
    config.ensure_plot_dirs()
    n = len(config.PATTERNS)
    nrows = -(-n // NCOLS)
    fig, axes = plt.subplots(nrows, NCOLS, figsize=(NCOLS * 3.4, nrows * 3.0),
                             sharex=True, sharey=True)
    pos = np.arange(config.SEQ_LEN)

    win_starts = np.arange(config.N_WINDOWS)

    for i, pat in enumerate(config.PATTERNS):
        ax = axes.flat[i]
        prof = row_profile(pat).numpy()
        ax.plot(pos, prof, "o-", color=PATTERN_COLOR, linewidth=1.6, markersize=3)
        for ws in win_starts:
            ax.axvline(ws, color="0.6", linestyle="--", linewidth=0.6,
                       alpha=0.5)
        ax.set_title(pat, fontsize=9, color=PATTERN_COLOR)
        ax.set_xticks(range(config.SEQ_LEN))
        ax.grid(alpha=0.3)

    for ax in axes.flat[n:]:
        ax.axis("off")

    for ax in axes.flat[:n]:
        ax.set_xlim(-0.5, config.SEQ_LEN - 0.5)
    for ax in axes[-1, :]:
        ax.set_xlabel("input position (0 = oldest)")
    for ax in axes[:, 0]:
        ax.set_ylabel("mean importance")
    fig.suptitle("Mean importance vs input position per pattern "
                 "(dashed = sliding-window starts)")
    fig.tight_layout()

    out = config.PLOT_IMPORTANCE_DIR / "importance_row_profile.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_importance_maps_mean() -> Path:
    config.ensure_plot_dirs()
    n = len(config.PATTERNS)
    nrows = -(-n // NCOLS)
    fig, axes = plt.subplots(nrows, NCOLS,
                             figsize=(NCOLS * 3.2, nrows * 3.0),
                             sharex=True, sharey=True)
    if config.SIGNED_IMPORTANCE:
        vmin, vmax = -1.0, 1.0
        cmap = "RdBu_r"
        cb_label = "mean signed importance (-1..1)"
    else:
        vmin, vmax = 0.0, 1.0
        cmap = "viridis"
        cb_label = "mean importance (0..1)"
    im = None
    for i, pat in enumerate(config.PATTERNS):
        ax = axes.flat[i]
        m = mean_map(pat)
        im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(pat, fontsize=9, color=PATTERN_COLOR)
        ax.set_xticks(range(config.H))
        ax.set_yticks(range(config.SEQ_LEN))

    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("hidden unit")
    for ax in axes[:, 0]:
        ax.set_ylabel("input position")
    fig.colorbar(im, ax=axes.flat[:n], shrink=0.9, label=cb_label)
    fig.suptitle("Mean importance map (SEQ_LEN x H) per pattern "
                 "— Toeplitz bands indicate sliding-window support")
    fig.tight_layout()

    out = config.PLOT_IMPORTANCE_DIR / "importance_maps_mean.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot importance figures.")
    parser.add_argument("--maps", action="store_true",
                        help="only write importance_maps_mean.png")
    parser.add_argument("--profile", action="store_true",
                        help="only write importance_row_profile.png")
    args = parser.parse_args()

    do_profile = not args.maps
    do_maps = not args.profile
    if do_profile:
        print(f"[plot] {plot_row_profiles()}")
    if do_maps:
        print(f"[plot] {plot_importance_maps_mean()}")


if __name__ == "__main__":
    main()
