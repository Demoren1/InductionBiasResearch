"""Analyze decoded importance maps (|W1| proxy) as a function of latent z.

Produces 3 figures under outputs/plots/z_analysis/:
  A. z_examples_heatmaps.png   - 16 decoded p-maps for 16 random z (4x4 grid)
  B. z_value_histograms.png    - distribution of decoded values for first 4 z
  C. toeplitz_analysis.png     - mean p-map vs gold, window histogram, and
                                 best-permutation IoU histogram over n_z samples
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
import config
from data.generate import ideal_mask
from evaluation.align_importance import align_map
from models.cvae import CVAE

CKPT_DEFAULT = str(
    Path(__file__).resolve().parent.parent
    / "outputs" / "cvae_sweep" / "bce_sum_b0.1" / "cvae_best.pt"
)
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "plots" / "z_analysis"
N_ROWS = config.SEQ_LEN  # 8
N_COLS = config.H  # 8


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot decoded z->map variation and Toeplitz-likeness."
    )
    parser.add_argument("--ckpt", type=str, default=CKPT_DEFAULT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_z", type=int, default=256)
    return parser.parse_args()


def load_cvae(ckpt, device):
    model = CVAE(config.MASK_DIM, config.LATENT_DIM, config.CVAE_HIDDEN)
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.to(device).eval()
    return model


def decode_pmaps(model, z, device):
    """Decode z (N, LATENT_DIM) into sigmoid importance maps (N, 8, 8)."""
    N = z.shape[0]
    c = model.condition(torch.zeros(N, 4, device=device))  # (N, 0); cond_dim = 0
    logits = model.decode(z.to(device), c)
    return torch.sigmoid(logits).reshape(N, N_ROWS, N_COLS)


def binarize_topk(pmaps, k=config.K_ACTIVE):
    """Keep the top-32 of 64 cells per map -> 0/1 mask."""
    N = pmaps.shape[0]
    flat = pmaps.reshape(N, -1)
    _, idx = torch.topk(flat, k, dim=1)
    bins = torch.zeros_like(flat)
    bins.scatter_(1, idx, 1.0)
    return bins.reshape(pmaps.shape)


def build_windows():
    W = torch.zeros(5, N_ROWS)
    for w in range(5):
        W[w, w:w + 4] = 1.0
    return W


def fig_a(maps, out):
    fig, axes = plt.subplots(4, 4, figsize=(10, 10), dpi=130)
    for i in range(16):
        ax = axes[i // 4, i % 4]
        ax.imshow(maps[i].numpy(), cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(f"z[{i}]")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Decoded importance maps |W1| for different z (ckpt: bce_sum_b0.1)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out / "z_examples_heatmaps.png")
    plt.close(fig)


def fig_b(pmaps, out):
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    for i in range(4):
        values = pmaps[i].reshape(-1)
        ax.hist(values.numpy(), bins=20, histtype="step", alpha=0.7, label=f"z[{i}]")
        thr = torch.sort(values, descending=True).values[31].item()
        ax.axvline(thr, linestyle=":", alpha=0.6)
    ax.set_xlabel("decoded importance value")
    ax.set_ylabel("count")
    ax.set_title(
        "Distribution of first-layer importance values at different z "
        "(dotted = top-32 threshold)"
    )
    ax.legend()
    fig.savefig(out / "z_value_histograms.png")
    plt.close(fig)


def fig_c(pmaps, bins, gold, out):
    # Panel 1: mean decoded p-map; Panel 2: gold Toeplitz mask.
    mean_map = pmaps.mean(dim=0).numpy()
    fig, axes = plt.subplots(1, 4, figsize=(22, 5), dpi=140)

    im1 = axes[0].imshow(mean_map, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    axes[0].set_title("Mean decoded p-map")
    fig.colorbar(im1, ax=axes[0], fraction=0.046)

    im2 = axes[1].imshow(gold.numpy(), cmap="gray", vmin=0.0, vmax=1.0, aspect="auto")
    axes[1].set_title("Gold Toeplitz mask")
    fig.colorbar(im2, ax=axes[1], fraction=0.046)

    # Panel 3: window assignment histogram (n_z * 8 columns = 2048 assignments).
    W = build_windows()
    counts = np.zeros(5, dtype=np.float64)
    for col in range(N_COLS):
        overlaps = bins[:, :, col] @ W.t()  # (N_z, 5)
        win = torch.argmax(overlaps, dim=1)  # (N_z,)
        for w in range(5):
            counts[w] += (win == w).sum().item()

    fracs = np.array([0.25, 0.25, 0.25, 0.125, 0.125])
    expected = 2048.0 * fracs
    x = np.arange(5)
    axes[2].bar(x, counts, color="steelblue", alpha=0.8, label="model")
    axes[2].plot(x, expected, "o--", color="crimson", label="gold expectation")
    axes[2].set_xticks(x)
    axes[2].set_xlabel("window id")
    axes[2].set_ylabel("column count")
    axes[2].set_title("Window assignment histogram (over 256 z x 8 cols)")
    axes[2].legend()

    # Panel 4: best-permutation IoU vs gold.
    ious = []
    for m in bins:
        aligned = align_map(m.float(), gold)
        inter = (aligned * gold).sum().float()
        union = aligned.sum().float() + gold.sum().float() - inter
        ious.append((inter / union).item())
    ious = np.array(ious)
    mean_ious = ious.mean()
    median_ious = float(np.median(ious))

    axes[3].hist(ious, bins=20, color="steelblue", alpha=0.8)
    axes[3].axvline(0.561, color="crimson", linestyle="--",
                    label="collapsed ref 0.561")
    axes[3].set_xlabel("best-perm IoU vs gold")
    axes[3].set_ylabel("count")
    axes[3].set_title(f"Best-perm IoU vs gold (mean {mean_ious:.3f})")
    axes[3].legend()

    fig.tight_layout()
    fig.savefig(out / "toeplitz_analysis.png")
    plt.close(fig)
    return counts, expected, ious


def main():
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[zplot] ckpt: {args.ckpt}")
    print(f"[zplot] device: {device}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = load_cvae(args.ckpt, device)
    gold = ideal_mask().float()  # (8, 8); column h covers rows [h % 5, h % 5 + 4)

    # Figure A: 16 examples.
    g = torch.Generator().manual_seed(args.seed)
    z16 = torch.randn(16, config.LATENT_DIM, generator=g)
    pmaps16 = decode_pmaps(model, z16, device).detach().cpu()
    fig_a(pmaps16, OUT_DIR)
    print("[zplot] saved z_examples_heatmaps.png")

    # Figure B: value distributions for the first 4 z.
    fig_b(pmaps16, OUT_DIR)
    print("[zplot] saved z_value_histograms.png")

    # Figure C: n_z samples -> top-32 binarized masks -> window + IoU stats.
    g = torch.Generator().manual_seed(args.seed)
    z = torch.randn(args.n_z, config.LATENT_DIM, generator=g)
    pmap = decode_pmaps(model, z, device).detach().cpu()
    binaries = binarize_topk(pmap)
    counts, expected, ious = fig_c(pmap, binaries, gold, OUT_DIR)
    print("[zplot] saved toeplitz_analysis.png")

    mean_iou = ious.mean()
    median_iou = float(np.median(ious))
    print(f"[zplot] mean best-perm IoU : {mean_iou:.4f}")
    print(f"[zplot] median best-perm IoU : {median_iou:.4f}")
    print("[zplot] window counts   : " + ", ".join(f"{c:.0f}" for c in counts))
    print("[zplot] expected counts : " + ", ".join(f"{e:.0f}" for e in expected))


if __name__ == "__main__":
    main()