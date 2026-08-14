"""Plots that visualise the pattern datasets and the gold first-layer filters.

These plots live under outputs/plots/data/ and are meant to be inspected by
humans to build intuition about:
  * what each 4-bit pattern looks like,
  * the CVAE train/test pattern split,
  * the sliding-window (Toeplitz) structure of the gold filters and masks,
  * where in a sequence the pattern can appear (5 possible window offsets).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from generate import gold_first_layer  # noqa: E402

TRAIN = set(config.CVAE_TRAIN_PATTERNS)
TEST = set(config.CVAE_TEST_PATTERNS)

TRAIN_COLOR = "tab:green"
TEST_COLOR = "tab:orange"


def _group_color(pat: str) -> str:
    return TRAIN_COLOR if pat in TRAIN else TEST_COLOR


def _pattern_bits_row(pat: str) -> list:
    return [int(c) for c in pat]


def plot_patterns_overview() -> Path:
    """4x4 grid of all 16 patterns as 1x4 heatmaps (0 black / 1 white)."""
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for ax, pat in zip(axes.ravel(), config.PATTERNS):
        bits = [_pattern_bits_row(pat)]
        ax.imshow(bits, cmap="Greys", vmin=0, vmax=1,
                  aspect="auto", interpolation="nearest")
        ax.set_title(pat, color=_group_color(pat), fontsize=13, fontweight="bold")
        ax.spines[:].set_color(_group_color(pat))
        ax.spines[:].set_linewidth(2.5)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "16 length-4 bit patterns  (green = CVAE TRAIN, orange = CVAE TEST)\n"
        "train: 0000 0011 0101 0110 1001 1010 1100 1111\n"
        "test : 0001 0010 0100 0111 1000 1011 1101 1110",
        fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = config.PLOT_DATA_DIR / "patterns_overview.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_examples(pat: str, n_pos: int = 8, n_neg: int = 8) -> Path:
    """Heatmap of 8 positives (top) + 8 negatives (bottom), shape (16, 8)."""
    val = torch.load(config.val_path(pat))
    x, y = val["x"], val["y"]
    pos = x[y == 1][:n_pos]
    neg = x[y == 0][:n_neg]
    block = torch.cat([pos, neg], dim=0)          # (16, 8) +/-1
    pat_bits = config.pattern_to_bits(pat)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(block.numpy(), cmap="Greys", vmin=-1, vmax=1,
              aspect="auto", interpolation="nearest")
    ax.set_title(
        f"pattern {pat}   (n={config.N_VAL_SAMPLES}, "
        f"pos_fraction={y.mean().item():.3f})",
        color=_group_color(pat), fontweight="bold")
    ax.set_xticks(range(config.SEQ_LEN))
    ax.set_xticklabels(range(config.SEQ_LEN))
    ax.set_xlabel("input position")
    ax.set_ylabel("example row")
    ax.axhline(n_pos - 0.5, color="tab:red", lw=1.2, ls="--")
    ax.text(0.5, -0.12, "top 8 = positives, bottom 8 = negatives",
            transform=ax.transAxes, ha="center", fontsize=9, color="gray")
    # Annotate the pattern bits (as +1/-1) near the top of the figure.
    pm1 = config.pattern_to_pm1(pat)
    pm1_str = "  ".join(f"{int(v):+d}" for v in pm1.tolist())
    ax.text(0.5, 1.03, f"pattern (as +/-1): {pm1_str}",
            transform=ax.transAxes, ha="center", fontsize=10)
    # Highlight the first matched window in each positive row.
    windows = pos.unfold(1, config.PATTERN_LEN, 1)
    match = (windows == pat_bits.view(1, 1, -1)).all(dim=2)   # (n_pos, 5)
    for r in range(pos.shape[0]):
        starts = match[r].nonzero(as_tuple=False).flatten().tolist()
        if starts:
            s = starts[0]
            ax.add_patch(matplotlib.patches.Rectangle(
                (s - 0.5, r - 0.5), config.PATTERN_LEN, 1,
                fill=False, edgecolor="tab:red", lw=1.4))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = config.PLOT_DATA_DIR / f"examples_pattern_{pat}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_gold_filters() -> Path:
    """4x4 of gold_first_layer(pat), RdBu_r, shared colorbar."""
    fig, axes = plt.subplots(4, 4, figsize=(12, 12), sharex=True, sharey=True)
    for ax, pat in zip(axes.ravel(), config.PATTERNS):
        W = gold_first_layer(pat)
        im = ax.imshow(W.numpy(), cmap="RdBu_r", vmin=-1, vmax=1,
                       aspect="auto", interpolation="nearest")
        ax.set_title(pat, color=_group_color(pat), fontsize=12, fontweight="bold")
        ax.set_xticks(range(config.SEQ_LEN))
        ax.set_xticklabels(range(config.SEQ_LEN))
        ax.set_yticks(range(config.N_WINDOWS))
        ax.set_yticklabels(range(config.N_WINDOWS))
    for ax in axes[:, 0]:
        ax.set_ylabel("window start")
    for ax in axes[-1, :]:
        ax.set_xlabel("input position")
    fig.suptitle("Gold first-layer filters  (green = CVAE TRAIN, orange = TEST)\n"
                 "row w holds +pattern on columns [w, w+PATTERN_LEN)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01, label="weight")
    out = config.PLOT_DATA_DIR / "gold_filters.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def match_start_positions(pat: str) -> torch.Tensor:
    """(N,) long -- all match start positions 0..N_WINDOWS-1 among positives."""
    val = torch.load(config.val_path(pat))
    x = val["x"]
    y = val["y"] > 0
    pos = x[y]
    pat_bits = config.pattern_to_bits(pat)
    windows = pos.unfold(1, config.PATTERN_LEN, 1)
    match = (windows == pat_bits.view(1, 1, -1)).all(dim=2)   # (n_pos, 5)
    idx = match.nonzero(as_tuple=False)                        # (n_match, 2)
    return idx[:, 1]


def plot_match_positions(patterns=("0000", "0101", "1111", "1011")) -> Path:
    """Histograms of match start positions on val positives (2x2 subplots)."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, pat in zip(axes.ravel(), patterns):
        starts = match_start_positions(pat)
        bins = list(range(0, config.N_WINDOWS + 1))
        ax.hist(starts.numpy(), bins=bins, align="left", rwidth=0.85,
                color=_group_color(pat), edgecolor="black")
        ax.set_title(f"pattern {pat}", color=_group_color(pat), fontweight="bold")
        ax.set_xticks(range(config.N_WINDOWS))
        ax.set_xticklabels(range(config.N_WINDOWS))
        ax.set_xlabel("match start position")
        ax.set_ylabel("count")
        ax.set_xlim(-0.5, config.N_WINDOWS - 0.5)
    fig.suptitle(
        "Match start positions on validation positives\n"
        "a length-4 pattern can sit at any of the 5 window offsets 0..4",
        fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = config.PLOT_DATA_DIR / "match_positions.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> None:
    config.ensure_plot_dirs()

    out = plot_patterns_overview()
    print(f"[plot] {out}")

    for pat in config.PATTERNS:
        out = plot_examples(pat)
        print(f"[plot] {out}")

    out = plot_gold_filters()
    print(f"[plot] {out}")

    out = plot_match_positions()
    print(f"[plot] {out}")


if __name__ == "__main__":
    main()