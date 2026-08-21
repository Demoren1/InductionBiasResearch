"""Diagnostics: what does the unconditional CVAE latent z encode?

For a trained unconditional CVAE (cond_dim=0, mask_dim=64, latent 32):
  z ~ N(0, I) is sampled N times and decoded to p-maps (N, SEQ_LEN, H).

Metrics computed:
  sharpness      mean of the top-K_ACTIVE sigmoid values per sample
                 (blurry barycenter ~0.3-0.4, sharp map ~0.7-1.0).
  diversity      mean pairwise IoU over binarized (top-32) sampled masks;
                 IoU ~1.0 => the decoder ignores z, low IoU => z changes map.
  best-perm IoU  per sample, Hungarian-align the binarized map's columns to
                 the gold Toeplitz mask and report IoU (mean/median/min).
                 High + diverse => every z yields a valid column
                 permutation of the gold structure.
  window hist    per-column argmax overlap with the 5 Toeplitz window
                 templates; histogram over (samples x 8) columns plus the
                 fraction of unambiguous assignments (top overlap - second
                 best > 0.5).

Two plots are written to config.PLOT_CVAE_DIR:
  latent_samples{tag}.png  - first 16 sampled p-maps (4x4 grid)
  latent_iou_hist{tag}.png - histogram of best-perm IoU values

Usage:
  python evaluation/latent_diagnostics.py --ckpt PATH --tag _name
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from data.generate import ideal_mask  # noqa: E402
from evaluation.align_importance import align_map  # noqa: E402
from models.cvae import CVAE  # noqa: E402


def build_window_templates() -> torch.Tensor:
    """(N_WINDOWS, SEQ_LEN) binary: template w has ones on rows [w, w+4)."""
    W = torch.zeros(config.N_WINDOWS, config.SEQ_LEN)
    for w in range(config.N_WINDOWS):
        W[w, w:w + config.PATTERN_LEN] = 1.0
    return W


def binarize_topk(p: torch.Tensor, k: int) -> torch.Tensor:
    """(N, D) probs -> (N, D) 0/1 with exactly the top-k entries set."""
    _, idx = p.topk(k, dim=-1)
    out = torch.zeros_like(p)
    out.scatter_(-1, idx, 1.0)
    return out


def metric_sharpness(p: torch.Tensor, k: int) -> dict:
    """Top-k prob mass per sample; report mean/min/max over samples."""
    top = p.topk(k, dim=-1).values.mean(dim=-1)
    return {"mean": top.mean().item(),
            "min": top.min().item(),
            "max": top.max().item()}


def metric_diversity(B: torch.Tensor) -> float:
    """Mean pairwise IoU over all pairs of binarized maps (each has k ones)."""
    n = B.size(0)
    inter = (B.float() @ B.float().t())          # (n, n) intersection counts
    k = int(B.sum(dim=-1).max().item())
    union = 2 * k - inter                         # both masks have k ones
    iou = inter / union.clamp(min=1e-9)
    tri = iou.triu(diagonal=1)
    return tri[tri > 0].mean().item()


def metric_best_perm_iou(B: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    """Per-sample IoU of a binarized map vs gold after Hungarian column
    alignment (best column permutation). Runs on CPU (scipy Hungarian)."""
    n = B.size(0)
    m = B.reshape(n, config.SEQ_LEN, config.H).float().cpu()
    g = gold.float().cpu()
    ious = torch.empty(n)
    for i in range(n):
        aligned = align_map(m[i], g)
        inter = (aligned * g).sum()
        union = aligned.sum() + g.sum() - inter
        ious[i] = inter / union
    return ious


def metric_window_hist(B: torch.Tensor, W: torch.Tensor) -> dict:
    """Per-column window-template assignment.

    For each (sample, column) pair, overlap = m[:, c] @ W^T (5,), argmax ->
    window id. Returns the 5-bin histogram over (samples x H) columns and the
    fraction of columns whose best overlap beats the second best by > 0.5.
    """
    m = B.reshape(-1, config.SEQ_LEN, config.H).float()     # (n, 8, 8)
    cols = m.permute(0, 2, 1).reshape(-1, config.SEQ_LEN)  # (n*H, 8)
    ov = cols @ W.t()                                      # (n*H, 5)
    win = ov.argmax(dim=-1)
    top2 = ov.topk(2, dim=-1).values
    unambig = ((top2[:, 0] - top2[:, 1]) > 0.5).float().mean().item()
    counts = torch.bincount(win, minlength=config.N_WINDOWS).tolist()
    return {"counts": counts, "total": int(win.numel()),
            "frac_unambiguous": unambig}


def plot_samples(p: torch.Tensor, tag: str) -> Path:
    """First 16 sampled p-maps as a 4x4 grid; p is (N, SEQ_LEN, H)."""
    n = min(16, p.size(0))
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(16):
        ax = axes[i // 4, i % 4]
        if i < n:
            ax.imshow(p[i].detach().cpu().numpy(),
                      cmap="viridis", vmin=0.0, vmax=1.0)
            ax.set_title(f"sample {i}")
        else:
            ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"First {n} sampled p-maps ({config.SEQ_LEN}x{config.H})")
    fig.tight_layout()
    out = config.PLOT_CVAE_DIR / f"latent_samples{tag}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_iou_hist(ious: torch.Tensor, tag: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ious.cpu().numpy(), bins=40, color="steelblue", edgecolor="white")
    ax.axvline(ious.mean().item(), color="red", ls="--",
               label=f"mean={ious.mean().item():.3f}")
    ax.set_xlabel("best-perm IoU vs gold")
    ax.set_ylabel("count")
    ax.set_title("Latent samples: IoU after best column permutation")
    ax.legend()
    fig.tight_layout()
    out = config.PLOT_CVAE_DIR / f"latent_iou_hist{tag}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose what the latent z of an unconditional CVAE "
                    "encodes (sample -> sharpness/diversity/structure).")
    parser.add_argument("--ckpt", type=Path, required=True,
                        help="path to the CVAE state dict")
    parser.add_argument("--n_samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default="",
                        help="suffix for output plot filenames")
    args = parser.parse_args()

    config.ensure_plot_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    print(f"[diag] ckpt={args.ckpt}", flush=True)
    print(f"[diag] device={device} n_samples={args.n_samples} "
          f"seed={args.seed} tag={args.tag!r}", flush=True)
    print(f"[diag] config: SEQ_LEN={config.SEQ_LEN} H={config.H} "
          f"MASK_DIM={config.MASK_DIM} LATENT={config.LATENT_DIM} "
          f"K_ACTIVE={config.K_ACTIVE}", flush=True)

    # Gold structure.
    gold = ideal_mask().float().to(device)         # (8, 8), 32 ones
    W = build_window_templates().to(device)        # (5, 8) window templates
    print(f"[diag] gold: shape={tuple(gold.shape)} ones={int(gold.sum())} "
          f"windows={config.N_WINDOWS} pattern_len={config.PATTERN_LEN}",
          flush=True)

    # ---- model.
    model = CVAE(config.MASK_DIM, config.LATENT_DIM, config.CVAE_HIDDEN)
    model.load_state_dict(torch.load(args.ckpt, weights_only=True))
    model.to(device).eval()

    # ---- sample z ~ N(0, I), decode to p-maps (cond_dim == 0).
    gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    gen.manual_seed(args.seed)
    c = model.condition(torch.zeros(args.n_samples, 4, device=device))
    z = torch.randn(args.n_samples, model.latent_dim, device=device,
                    generator=gen)
    logits = model.decode(z, c)
    p = torch.sigmoid(logits).reshape(args.n_samples,
                                      config.SEQ_LEN, config.H)
    print(f"[diag] decoded z -> p maps: {tuple(p.shape)}", flush=True)

    # ---- metrics.
    sharp = metric_sharpness(p.reshape(args.n_samples, -1), config.K_ACTIVE)
    print(f"[diag] sharpness (mean top-{config.K_ACTIVE} p): "
          f"mean={sharp['mean']:.3f} min={sharp['min']:.3f} "
          f"max={sharp['max']:.3f}", flush=True)

    B = binarize_topk(p.reshape(args.n_samples, -1), config.K_ACTIVE)
    div = metric_diversity(B)
    print(f"[diag] diversity: mean pairwise IoU (top-{config.K_ACTIVE} "
          f"binarized) = {div:.4f}", flush=True)

    ious = metric_best_perm_iou(B, gold)
    print(f"[diag] best-perm IoU vs gold: mean={ious.mean():.4f} "
          f"median={ious.median():.4f} min={ious.min():.4f} "
          f"max={ious.max():.4f}", flush=True)

    wh = metric_window_hist(B, W)
    cnts = wh["counts"]
    print(f"[diag] window assignment of {wh['total']} columns "
          f"(5 windows {cnts}): unambiguous frac="
          f"{wh['frac_unambiguous']:.3f}", flush=True)

    # ---- plots.
    path_samples = plot_samples(p, args.tag)
    path_hist = plot_iou_hist(ious, args.tag)
    print(f"[diag] plot -> {path_samples}", flush=True)
    print(f"[diag] plot -> {path_hist}", flush=True)

    # ---- summary block.
    print("\n[diag] ===== SUMMARY =====", flush=True)
    print(f"[diag] sharpness     mean={sharp['mean']:.3f} "
          f"min={sharp['min']:.3f} max={sharp['max']:.3f}", flush=True)
    print(f"[diag] pairwise IoU  mean={div:.4f}", flush=True)
    print(f"[diag] best-perm IoU mean={ious.mean():.4f} "
          f"median={ious.median():.4f} min={ious.min():.4f}", flush=True)
    print(f"[diag] window hist  {cnts} "
          f"unambig frac={wh['frac_unambiguous']:.3f}", flush=True)
    print("[diag] ================", flush=True)


if __name__ == "__main__":
    main()