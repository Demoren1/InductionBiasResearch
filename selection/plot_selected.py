"""Plot the selected (best-10%) masks together with their trained weights.

For each (kernel, offset) a figure is written to
outputs/plots/selected_masks_kernel_{k}_offset_{s}.png:
  * row 1:  first-layer weight heatmaps (32 x 16) of the 4 best MLPs; the
            binary mask is overlaid -- entries with mask == 0 are blanked to
            gray, entries with mask == 1 keep their (color-coded) weight;
  * row 2:  the binary mask of those same 4 best MLPs;
  * row 3:  aggregate mean active map, the *ideal* mask (support = rows
            [L-s-k .. L-s)) and the *ideal* first-layer matrix (1/k there).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def overlay_weight_axis(ax, w1: torch.Tensor, mask: torch.Tensor,
                        vmax: float, title: str) -> None:
    """Heatmap of w1 (L x H) with mask==0 cells blanked to gray."""
    w = w1.numpy()
    masked_w = np.ma.masked_where(~mask.numpy().astype(bool), w)
    ax.set_facecolor("0.85")                       # gray == frozen (mask=0)
    im = ax.imshow(masked_w, cmap="coolwarm", vmin=-vmax, vmax=vmax,
                   aspect="auto")
    ax.set_title(title, fontsize=9)
    return im


def ideal_structure(kernel: int, offset: int = None) -> tuple:
    """Ideal first-layer mask/weights for the shifted MA(kernel, offset) task.

    Task: y = mean of inputs at positions L-s-k .. L-s-1 (the ``kernel`` inputs
    ending ``offset`` rows before the current one), so the *only* entries that
    should carry information live on rows [L-s-k .. L-s) of W1. Ideal weights
    are 1/k there, 0 elsewhere.
    """
    s = offset if offset is not None else 0
    lo = config.L - s - kernel
    hi = config.L - s
    mask = torch.zeros(config.L, config.H)
    mask[lo:hi] = 1.0
    w = torch.zeros(config.L, config.H)
    w[lo:hi] = 1.0 / kernel
    return mask, w


def plot_kernel(kernel: int, offset: int, n_examples: int = 4) -> Path:
    d = torch.load(config.kernel_dir(kernel, offset) / "best10pct.pt",
                   weights_only=True)
    w1, masks, vl = d["params"]["w1"], d["masks"], d["val_loss"]
    order = torch.argsort(vl)[:n_examples]

    wmax = w1[:200].abs().max().item()
    mean_mask = masks.float().mean(dim=0)
    ideal_mask, ideal_w = ideal_structure(kernel, offset)

    fig, axes = plt.subplots(3, n_examples,
                             figsize=(2.7 * n_examples, 8.0),
                             gridspec_kw={"height_ratios": [3, 1, 1]})
    for j, gi in enumerate(order):
        im = overlay_weight_axis(axes[0, j], w1[gi], masks[gi], wmax,
                                 f"rank {j + 1}  val_mse={vl[gi].item():.2e}")
        axes[1, j].imshow(masks[gi].numpy(), cmap="viridis", vmin=0, vmax=1,
                          aspect="auto")
        axes[1, j].set_title(f"mask (sparsity "
                             f"{masks[gi].mean().item():.2f})", fontsize=9)

    # --- bottom row: aggregate + ideal reference for the kernel ---
    axes[2, 0].imshow(mean_mask.numpy(), cmap="viridis", vmin=0, vmax=1,
                      aspect="auto")
    axes[2, 0].set_title("mean active map (top-10%)", fontsize=9)
    axes[2, 1].imshow(ideal_mask.numpy(), cmap="viridis", vmin=0, vmax=1,
                      aspect="auto")
    axes[2, 1].set_title("ideal mask (rows L-s-k .. L-s)", fontsize=9)
    axes[2, 2].imshow(ideal_w.numpy(), cmap="coolwarm",
                      vmin=-wmax, vmax=wmax, aspect="auto")
    axes[2, 2].set_title("ideal W1 (1/k on rows L-s-k .. L-s)", fontsize=9)
    axes[2, 3].text(0.5, 0.5,
                    f"ideal support:\nrows {config.L - offset - kernel}"
                    f" .. {config.L - offset - 1}\n"
                    f"i.e. last {kernel} inputs ending {offset} rows ago",
                    ha="center", va="center", fontsize=9, wrap=True)
    axes[2, 3].axis("off")

    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes[0], fraction=0.03, label="first-layer weight")
    fig.suptitle(f"kernel {kernel}, offset {offset}: "
                 f"best-{config.TOP_FRACTION:.0%} masks "
                 f"(rows are input positions, bottom = most recent inputs; "
                 f"frozen mask=0 entries blanked)", fontsize=12)
    fig.subplots_adjust(left=0.05, right=0.93, top=0.9, bottom=0.06,
                        wspace=0.12, hspace=0.5)
    config.PLOT_SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    out = config.PLOT_SELECTED_DIR / \
        f"selected_masks_kernel_{kernel}_offset_{offset}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    for kernel in config.KERNELS:
        for offset in config.OFFSETS:
            out = plot_kernel(kernel, offset)
            print(f"[plot] {out}")


if __name__ == "__main__":
    main()