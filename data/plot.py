"""Plot example windows of each shifted-MA dataset (one figure per (k,s)).

Each figure shows 4 example input windows (32 raw points) with the
corresponding shifted MA-target overlaid and the active region
[L - s - k, L - s) shaded, so the reader can see what the network is
asked to predict for every kernel/offset combination.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def plot_kernel(kernel: int, offset: int, n_examples: int = 4) -> Path:
    val = torch.load(
        config.DATA_DIR / f"val_kernel_{kernel}_offset_{offset}.pt")
    x, y = val["x"], val["y"]

    lo = config.L - offset - kernel          # 0-based start of active region
    hi = config.L - offset                   # 0-based exclusive end

    fig, axes = plt.subplots(1, n_examples, figsize=(4.5 * n_examples, 3.2))
    for i, ax in enumerate(axes):
        window = x[i]
        ax.plot(range(1, config.L + 1), window.numpy(), color="tab:blue",
                marker=".", markersize=3, lw=1.2, label="raw window")
        ax.axvspan(lo + 1, hi, color="gray", alpha=0.25,
                   label=f"active [{lo}, {hi})")
        ax.axhline(y[i].item(), color="tab:red", ls="--", lw=1.4,
                   label=f"MA({kernel}) = {y[i].item():.2f}")
        ax.set_title(f"example #{i + 1}")
        ax.set_xlabel("t")
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(f"Shifted MovingAverage dataset, kernel = {kernel}, "
                 f"offset = {offset}, L = {config.L} "
                 f"(val: {config.N_VAL_SAMPLES} windows)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = config.PLOT_DATA_DIR / f"ma_kernel_{kernel}_offset_{offset}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> None:
    config.ensure_plot_dirs()
    for kernel in config.KERNELS:
        for offset in config.OFFSETS:
            out = plot_kernel(kernel, offset)
            print(f"[plot] {out}")


if __name__ == "__main__":
    main()