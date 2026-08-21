"""Plot decoded importance maps and metrics along linear z-space traversals.

For each random pair (z0, z1) ~ N(0, I) we linearly interpolate
z_t = (1-t) z0 + t z1 with t in [0, 1], decode all points at once through the
CVAE decoder, and analyze the resulting p-maps:

* Figure 1: raw sigmoid p-maps (n_pairs x n_steps grid of 8x8 maps).
* Figure 2a: IoU of the top-32 binarized mask at t vs masks at t=0 (solid).
  and at t=1 (dashed, same colour per pair).
* 2b: per-column window assignment (argmax overlap with 5 Toeplitz window
  templates) along the traversal -> discrete switch vs continuous morph.

Run: conda activate ras; nvidia-smi; CUDA_VISIBLE_DEVICES=<g> python evaluation/plot_z_traversal.py
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
from models.cvae import CVAE
from data.generate import ideal_mask

# window templates: window w covers rows [w, w+PATTERN_LEN) -> (5, 8)
W = torch.zeros(config.N_WINDOWS, config.SEQ_LEN)
for w in range(config.N_WINDOWS):
    W[w, w: w + config.PATTERN_LEN] = 1.0

GOLD = ideal_mask().float()


def bin_top_k(p_maps, k):
    """(T, 8, 8) p-maps -> (T, 64) top-k binarized masks (float 0/1)."""
    p_flat = p_maps.reshape(p_maps.size(0), -1)
    _, top = p_flat.topk(k, dim=-1)
    out = torch.zeros_like(p_flat)
    out.scatter_(-1, top, 1.0)
    return out


def ious(masks, ref_idx):
    """IoU of each (T, 64) binary mask vs mask at ref_idx -> (T,)."""
    a = masks.bool()
    ref = a[ref_idx].unsqueeze(0)
    inter = (a & ref).sum(dim=-1).float()
    union = (a | ref).sum(dim=-1).float()
    return inter / union


def window_assignments(bin_maps):
    """(T, 8, 8) -> (T, 8) argmax-window-id per column of each map."""
    T = bin_maps.size(0)
    assign = torch.zeros(T, config.H, dtype=torch.long)
    for tt in range(T):
        for cc in range(config.H):
            overlaps = bin_maps[tt, :, cc] @ W.T   # (5,)
            assign[tt, cc] = torch.argmax(overlaps)
    return assign


def analyze_pair(p_maps, n_steps):
    """Derive binary masks, IoUs and window assignments for one traversal."""
    bin_flat = bin_top_k(p_maps, config.K_ACTIVE)             # (T, 64)
    bin_maps = bin_flat.reshape(-1, config.SEQ_LEN, config.H)  # (T, 8, 8)
    return {
        "bin_flat": bin_flat,
        "bin_maps": bin_maps,
        "iou0": ious(bin_flat, 0),
        "iou1": ious(bin_flat, n_steps - 1),
        "assign": window_assignments(bin_maps),
    }


def figure1(p_m_all, t, path, args):
    """n_pairs x n_steps grid of decoded p-maps; titles only on first row."""
    fig, axes = plt.subplots(args.n_pairs, args.n_steps,
                             figsize=(2.2 * args.n_steps, 2.2 * args.n_pairs),
                             squeeze=False)
    for k in range(args.n_pairs):
        for j in range(args.n_steps):
            ax = axes[k, j]
            ax.imshow(p_m_all[k][j].cpu(), cmap="viridis", vmin=0, vmax=1,
                      aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            if k == 0:
                ax.set_title("t=%.2f" % t[j])
        axes[k, 0].set_ylabel("pair %d: z0 -> z1" % k, rotation=0,
                              ha="right", va="center", labelpad=14)
    fig.suptitle("Latent traversal: decoded importance map vs interpolation "
                 "in z-space (ckpt: bce_sum_b0.1)", y=0.99)
    fig.tight_layout(rect=(0.03, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("[ztrav] wrote %s (%d bytes)" % (path, path.stat().st_size))


def figure2(res, t, path, args):
    """Two-panel metric figure."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 0.9, args.n_pairs))
    markers = [".", "o", "v", "^", "<", ">", "s", "D"]

    # panel (a): IoU vs t toward both endpoints
    for k in range(args.n_pairs):
        r = res[k]
        axa.plot(t.numpy(), r["iou0"].cpu().numpy(), color=colors[k], lw=2,
                 label="pair %d: vs start (t=0)" % (k + 1))
        axa.plot(t.numpy(), r["iou1"].cpu().numpy(), color=colors[k], lw=2,
                 linestyle="--",
                 label="pair %d: vs end (t=1)" % (k + 1))
    axa.set_xlabel("t")
    axa.set_ylabel("IoU with endpoint mask")
    axa.set_title("(a) IoU of binarized mask vs endpoints during traversal")
    axa.legend(fontsize=7, loc="best")
    axa.grid(alpha=0.3)

    # panel (b): x = t index(+ tiny per-column jitter), y = window id,
    # colour = pair, marker style = column index
    jitter = np.linspace(-0.015, 0.015, config.H)
    for k in range(args.n_pairs):
        assign = res[k]["assign"].numpy()   # (T, 8)
        for cc in range(config.H):
            axb.scatter(np.arange(args.n_steps) + jitter[cc], assign[:, cc],
                        color=colors[k], marker=markers[cc], s=22,
                        edgecolors="none", alpha=0.9)
    axb.set_xlabel("t (index)")
    axb.set_ylabel("assigned window id (0..4)")
    axb.set_ylim(-0.5, 4.5)
    axb.set_title("(b) per-column window assignment along traversal "
                  "(discrete switches?)")
    axb.grid(alpha=0.3)
    pair_handles = [plt.Line2D([0], [0], color=colors[k], lw=2,
                               label="pair %d" % k)
                    for k in range(args.n_pairs)]
    col_handles = [plt.Line2D([0], [0], color="gray", marker=markers[cc],
                              linestyle="None", markersize=6,
                              label="column %d" % cc)
                   for cc in range(config.H)]
    leg = axb.legend(handles=pair_handles + col_handles, fontsize=6,
                     loc="upper right",
                     title="color=pair; marker=column", title_fontsize=6)
    axb.add_artist(leg)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("[ztrav] wrote %s (%d bytes)" % (path, path.stat().st_size))


def print_metrics(res, t):
    """Per pair: endpoint IoU and per-column window-switch counts."""
    for k, r in enumerate(res):
        iou_end = r["iou1"][0].item()  # IoU of the mask at t=0 vs mask at t=1
        assign = r["assign"]           # (T, 8)
        n_switches = (assign[1:] != assign[:-1]).sum(dim=0)  # (8,)
        print("[ztrav] pair %d: IoU(start, end) = %.4f"
              % (k, iou_end))
        print("[ztrav] pair %d: per-column window switches over %d steps = %s"
              " (total %d)"
              % (k, len(t), n_switches.tolist(), int(n_switches.sum())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default="outputs/cvae_sweep/bce_sum_b0.1/cvae_best.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_steps", type=int, default=12)
    ap.add_argument("--n_pairs", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[ztrav] device=%s seed=%d n_pairs=%d n_steps=%d ckpt=%s"
          % (device, args.seed, args.n_pairs, args.n_steps, args.ckpt))

    model = CVAE(config.MASK_DIM, config.LATENT_DIM, config.CVAE_HIDDEN)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    print("[ztrav] model loaded from %s" % args.ckpt)

    global W
    W = W.to(device)

    rng = torch.Generator().manual_seed(args.seed)
    z_pairs = torch.randn(args.n_pairs, 2, config.LATENT_DIM,
                          generator=rng).to(device)
    t_vals = torch.linspace(0.0, 1.0, args.n_steps, device=device)
    t_cpu = t_vals.cpu()

    p_m_all = []
    res = []
    with torch.no_grad():
        for k in range(args.n_pairs):
            z0, z1 = z_pairs[k, 0], z_pairs[k, 1]
            w0, w1 = 1 - t_vals, t_vals
            z_t = w0[:, None] * z0[None, :] + w1[:, None] * z1[None, :]
            c = model.condition(torch.zeros(z_t.size(0), 4, device=device))
            logits = model.decode(z_t, c)
            pm = torch.sigmoid(logits).reshape(-1, config.SEQ_LEN, config.H)
            p_m_all.append(pm)
            res.append(analyze_pair(pm, args.n_steps))
            print("[ztrav] decoded pair %d: %d maps" % (k, args.n_steps))

    print_metrics(res, t_cpu)

    out_dir = config.ROOT / "outputs" / "plots" / "z_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    figure1(p_m_all, t_cpu, out_dir / "z_traversal.png", args)
    figure2(res, t_cpu, out_dir / "z_traversal_metrics.png", args)
    print("[ztrav] done")


if __name__ == "__main__":
    main()
