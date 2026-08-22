"""Plot the selected (best-10%) masks together with their trained weights.

For each pattern a figure is written to
outputs/plots/selected/selected_masks_pattern_{pat}.png:

  * row 1:  first-layer weight heatmaps (SEQ_LEN x H) of the 4 best MLPs
            (lowest val_loss); entries with mask == 0 are blanked to gray,
            entries with mask == 1 keep their signed (coolwarm) weight;
  * row 2:  the binary mask of those same 4 best MLPs;
  * row 3:  aggregate mean active map of the top-10% (col 0), the ideal
            Toeplitz / sliding-window mask (col 1), the gold first-layer
            (N_WINDOWS x SEQ_LEN) as a reference (col 2), and a text cell
            describing the pattern and the ideal
            support (col 3).

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
from data.generate import gold_first_layer, ideal_mask  # noqa: E402

N_BEST = 4  # 4 best MLPs shown in rows 1-2


def overlay_weight_axis(ax, w1: torch.Tensor, mask: torch.Tensor,
                        vmax: float, title: str) -> "plt.Axes":
    """Heatmap of w1 (SEQ_LEN x H) with mask==0 cells blanked to gray."""
    w = w1.numpy()
    masked_w = np.ma.masked_where(~mask.numpy().astype(bool), w)
    ax.set_facecolor("0.85")                    # gray == frozen (mask=0)
    im = ax.imshow(masked_w, cmap="coolwarm", vmin=-vmax, vmax=vmax,
                   aspect="auto")
    ax.set_title(title, fontsize=8)
    return im


def plot_pattern(pat: str) -> Path:
    d = torch.load(config.pattern_dir(pat) / "best10pct.pt",
                   weights_only=True)
    w1, masks = d["params"]["w1"], d["masks"]
    vl, va = d["val_loss"], d["val_acc"]
    n_sel = d["n_selected"]
    order = torch.argsort(vl)[:N_BEST]

    wmax = w1.abs().max().item()
    mean_mask = masks.float().mean(dim=0)          # (SEQ_LEN, H)
    ideal = ideal_mask().float()
    gold = gold_first_layer(pat)

    fig, axes = plt.subplots(3, 4,
                             figsize=(15, 9),
                             gridspec_kw={"height_ratios": [3, 1, 2]})

    # --- rows 1-2: the N_BEST MLPs ---
    for j, gi in enumerate(order):
        im = overlay_weight_axis(
            axes[0, j], w1[gi], masks[gi], wmax,
            f"rank {j + 1}\nval_bce={vl[gi].item():.4f} "
            f"acc={va[gi].item():.3f}")
        axes[1, j].imshow(masks[gi].numpy(), cmap="Greys", vmin=0, vmax=1,
                          aspect="auto")
        axes[1, j].set_title(f"mask  sparsity="
                             f"{masks[gi].mean().item():.2f}", fontsize=8)

    # --- row 3: aggregates + references ---
    axes[2, 0].imshow(mean_mask.numpy(), cmap="viridis", vmin=0, vmax=1,
                      aspect="auto")
    axes[2, 0].set_title("mean active map (top-10%)", fontsize=9)
    axes[2, 1].imshow(ideal.numpy(), cmap="viridis", vmin=0, vmax=1,
                      aspect="auto")
    axes[2, 1].set_title(f"ideal Toeplitz mask "
                         f"(N_WINDOWS={config.N_WINDOWS})", fontsize=9)
    axes[2, 2].imshow(gold.numpy(), cmap="coolwarm", vmin=-1, vmax=1,
                      aspect="auto")
    axes[2, 2].set_title(f"gold first layer ({config.N_WINDOWS}x"
                         f"{config.SEQ_LEN})", fontsize=9)
    axes[2, 3].axis("off")
    axes[2, 3].text(
        0.5, 0.5,
        f"pattern bits: {pat}\n"
        "all patterns train the unconditional VAE\n\n"
        f"ideal support: hidden h -> window w = h % {config.N_WINDOWS};\n"
        f"ones on rows [w : w+{config.PATTERN_LEN}] of column h\n"
        f"(sliding-window Toeplitz band, pattern-independent)",
        ha="center", va="center", fontsize=9, color="#222222")
    axes[2, 3].set_facecolor("0.92")

    # --- axes conventions: xlabel = hidden unit, ylabel = input position ---
    for ax in axes[0, :]:
        ax.set_xlabel("hidden unit", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("input position", fontsize=8)
    for ax in axes.ravel():
        ax.tick_params(labelsize=6)

    fig.colorbar(im, ax=axes[0, :], fraction=0.03, pad=0.01,
                 label="first-layer weight (signed)")
    fig.suptitle(f"pattern {pat}  ·  n_selected={n_sel}  ·  "
                 f"val_bce {vl.min().item():.4f}–{vl.max().item():.4f}  ·  "
                 f"val_acc {va.min().item():.3f}–{va.max().item():.3f}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    config.ensure_plot_dirs()
    out = config.PLOT_SELECTED_DIR / f"selected_masks_pattern_{pat}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot selected best-10% masks per pattern.")
    parser.add_argument("--pattern", type=str, default=None,
                        help="single pattern to plot; default: all")
    args = parser.parse_args()
    jobs = [args.pattern] if args.pattern is not None else config.PATTERNS
    for pat in jobs:
        out = plot_pattern(pat)
        print(f"[plot] {out}")


if __name__ == "__main__":
    main()
