"""Row-profile plot of the learned importance maps.

For every (kernel, offset) in config.KERNELS x config.OFFSETS, load
outputs/checkpoints/kernel_{k}/offset_{s}/importance.pt and compute the mean
importance per input row, averaging over the hidden dimension (H=16) and over
all trained MLPs.  The figure is a 2x2 grid (one subplot per offset s), each
subplot drawing all kernels (x = input position 0..31, y = mean importance)
with a vertical dashed line at position L - s - k for every kernel (the
theoretically optimal support start for the shifted MA(k, s) task).

Output: outputs/plots/importance/importance_row_profile.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def row_profile(kernel: int, offset: int) -> torch.Tensor:
    """Mean importance per input row: (L,) tensor for one (kernel, offset)."""
    d = torch.load(config.kernel_dir(kernel, offset) / "importance.pt",
                   weights_only=True)
    imp = d["importance"]                       # (n, L, H)
    return imp.mean(dim=(0, 2))                 # average over hidden + MLPs


def plot_row_profiles() -> Path:
    config.ensure_plot_dirs()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    pos = torch.arange(config.L).numpy()
    colors = plt.get_cmap("tab10")

    for oi, offset in enumerate(config.OFFSETS):
        ax = axes.flat[oi]
        for ci, k in enumerate(config.KERNELS):
            color = colors(ci)
            prof = row_profile(k, offset).numpy()
            ax.plot(pos, prof, label=f"kernel {k}", color=color)
            ax.axvline(config.L - offset - k, color=color, linestyle="--",
                       alpha=0.6, linewidth=1)
        ax.set_title(f"offset s={offset}")
        ax.set_xticks(range(0, config.L, 4))
        ax.legend(title="MA kernel", fontsize=9)
        ax.grid(alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("input position (0 = oldest, 31 = most recent)")
        ax.set_ylabel("mean importance")
    fig.suptitle("Importance concentration by input row per (kernel, offset)")
    fig.tight_layout()

    out = config.PLOT_IMPORTANCE_DIR / "importance_row_profile.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    out = plot_row_profiles()
    print(f"[plot] {out}")


if __name__ == "__main__":
    main()