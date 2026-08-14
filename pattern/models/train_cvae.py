"""Train an unconditional VAE on |W| importance maps (shared across patterns).

Usage:
    python models/train_cvae.py --epochs 80 --latent_dim 32

The model is unconditional (config.CVAE_COND_DIM == 0): there is no pattern
condition, so the learned mask is shared across all patterns. Best model (by
validation loss), loss curves, mask reconstructions and the shared-map figure
are written to outputs/plots/cvae/.

For sampling synthetic masks with a trained model use the --mode sample flag.
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

print("[cvae] MODULE LOADED", flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from models.cvae import (  # noqa: E402
    CVAE, cvae_loss, cvae_loss_importance,
    load_selected_masks, make_loaders,
    generate_ideal_masks,
    load_importance_maps,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train CVAE on pattern masks.")
    p.add_argument("--mode", choices=["train", "sample"], default="train")
    p.add_argument("--epochs", type=int, default=config.CVAE_EPOCHS)
    p.add_argument("--batch_size", type=int, default=config.CVAE_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.CVAE_LR)
    p.add_argument("--beta", type=float, default=config.CVAE_BETA)
    p.add_argument("--latent_dim", type=int, default=config.LATENT_DIM)
    p.add_argument("--hidden", type=int, default=config.CVAE_HIDDEN)
    p.add_argument("--seed", type=int, default=config.CVAE_SEED)
    p.add_argument("--patterns", nargs="+", default=config.PATTERNS)
    p.add_argument("--ideal_masks", action="store_true",
                   help="train on synthetic ideal Toeplitz masks")
    p.add_argument("--n_ideal_per_pattern", type=int, default=1000,
                   help="synthetic masks per pattern when --ideal_masks is set")
    p.add_argument("--ideal_noise", type=float, default=0.05,
                   help="bit-flip noise fraction for ideal masks")
    p.add_argument("--importance_maps", action="store_true",
                   help="train on continuous importance maps (MSE loss)")
    p.add_argument("--importance_name", type=str, default="importance.pt",
                   help="importance map filename in each pattern dir "
                        "(e.g. importance_aligned.pt for column-aligned maps)")
    p.add_argument("--ckpt_root", type=Path, default=config.CKPT_DIR)
    p.add_argument("--out_dir", type=Path, default=config.CVAE_DIR)
    p.add_argument("--k_active", type=int, default=0,
                   help="use sample_det with this many active entries per mask "
                        "(0 = default Bernoulli sampling)")
    return p


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def show_masks(ax, masks: torch.Tensor, title: str, vmax: float = 1.0) -> None:
    """Draw one mask (or mean) as a 2D heatmap in ax."""
    m = masks.reshape(config.SEQ_LEN, config.H).numpy()
    if config.SIGNED_IMPORTANCE:
        ax.imshow(m, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    else:
        ax.imshow(m, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=8)


def plot_reconstructions(model: CVAE, x, y, patterns, out_dir: Path,
                         vmax: float = 1.0) -> Path:
    """Per pattern: original importance | p(mask=1) map | stochastic recon.

    One example per training pattern (or a subset of 8). vmax controls the
    colormap ceiling (1.0 for binary masks, x.max() for importance maps).
    The condition is the pattern, so the ylabel shows the pattern bit string.
    """
    model.eval()
    device = next(model.parameters()).device
    xd, yd = x.to(device), y.to(device)
    p_map = model.prob(xd, yd).cpu()
    gen = torch.Generator(device=device).manual_seed(0)
    r_sample = model.reconstruct_sample(xd, yd, generator=gen).cpu()

    pats = list(patterns)[:8]
    n_pats = len(pats)
    panels = 3
    fig, axes = plt.subplots(n_pats, panels,
                             figsize=(3.1 * panels, 1.9 * n_pats),
                             squeeze=False)
    if config.SIGNED_IMPORTANCE:
        cmap = "RdBu_r"
        vlim = 1.0
    else:
        cmap = "viridis"
        vlim = vmax
    for r, pat in enumerate(pats):
        pm1 = config.pattern_to_pm1(pat).to(y.device)
        idxs = (y == pm1.unsqueeze(0)).all(dim=-1).nonzero(as_tuple=True)[0]
        gi = idxs[0].item()
        o = axes[r, 0]
        o.imshow(x[gi].reshape(config.SEQ_LEN, config.H), cmap=cmap,
                 vmin=-vlim if config.SIGNED_IMPORTANCE else 0, vmax=vlim,
                 aspect="auto")
        o.set_title("orig", fontsize=8)
        p = axes[r, 1]
        p.imshow(p_map[gi].reshape(config.SEQ_LEN, config.H), cmap=cmap,
                 vmin=-vlim if config.SIGNED_IMPORTANCE else 0, vmax=vlim,
                 aspect="auto")
        p.set_title("p(mask=1)", fontsize=8)
        st = axes[r, 2]
        st.imshow(r_sample[gi].reshape(config.SEQ_LEN, config.H),
                  cmap=cmap, vmin=-vlim if config.SIGNED_IMPORTANCE else 0,
                  vmax=vlim, aspect="auto")
        st.set_title("stoch. recon", fontsize=8)
        axes[r, 0].set_ylabel(pat, fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Unconditional VAE on |W| importance: original | p-map | "
                 "Bernoulli recon")
    fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.03,
                        wspace=0.05, hspace=0.35)
    out = config.PLOT_CVAE_DIR / "cvae_reconstructions.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_losses(train_loss: list, val_loss: list, out_dir: Path) -> Path:
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    epochs = range(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, label="train")
    ax.plot(epochs, val_loss, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("ELBO loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = config.PLOT_CVAE_DIR / "cvae_loss.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_condition_effect(model: CVAE, patterns, out_dir: Path) -> Path:
    """For z = 0 decode all 16 patterns and show the 16 p-maps side by side.

    This is the key figure: does the condition change the map, or is the
    support pattern-independent? One p-map per cell in a 4x4 grid keyed by
    the pattern bit string.
    """
    model.eval()
    device = next(model.parameters()).device
    all_pat = list(patterns)
    if len(all_pat) == 0:
        return None
    n = len(all_pat)
    pm = torch.stack([config.pattern_to_pm1(p) for p in all_pat]).to(device)
    z = torch.zeros(n, model.latent_dim, device=device)
    c = model.condition(pm)
    with torch.no_grad():
        logits = model.decode(z, c)
    if config.SIGNED_IMPORTANCE:
        p_map = torch.tanh(logits).detach().cpu()
    else:
        p_map = torch.sigmoid(logits).detach().cpu()

    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows),
                             squeeze=False)
    cmap = "RdBu_r" if config.SIGNED_IMPORTANCE else "viridis"
    vmin = -1.0 if config.SIGNED_IMPORTANCE else 0.0
    for i, pat in enumerate(all_pat):
        r, c = divmod(i, cols)
        axes[r, c].imshow(p_map[i].reshape(config.SEQ_LEN, config.H),
                          cmap=cmap, vmin=vmin, vmax=1.0, aspect="auto")
        axes[r, c].set_title(pat, fontsize=9)
        axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    for r in range(rows):
        for c in range(cols):
            if r * cols + c >= n:
                axes[r, c].axis("off")
    fig.suptitle("Unconditional VAE z=0 prior-mode p-maps (maps must be "
                 "identical across patterns)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = config.PLOT_CVAE_DIR / "cvae_condition_effect.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_samples(samples: dict, out_dir: Path) -> Path:
    """Plot masks generated by the CVAE (a few per pattern).

    Each row is one pattern; the row label marks TRAIN vs TEST so we can see
    whether held-out patterns get a similar Toeplitz skeleton.
    """
    n_rows = len(samples)
    n_per = next(iter(samples.values())).size(0)
    fig, axes = plt.subplots(n_rows, n_per, figsize=(1.6 * n_per,
                                                     1.7 * n_rows),
                             squeeze=False)
    for r, (label, ms) in enumerate(samples.items()):
        for c in range(n_per):
            axes[r, c].imshow(ms[c].numpy(), cmap="viridis", vmin=0, vmax=1,
                              aspect="auto")
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if c == 0:
                axes[r, c].set_ylabel(label, fontsize=8)
    fig.suptitle("Unconditional VAE-generated masks (shared prior; "
                 "TRAIN vs TEST)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = config.PLOT_CVAE_DIR / "cvae_generated.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def do_train(args) -> None:
    print(f"[cvae] do_train started, mode={args.mode}, "
          f"importance_maps={args.importance_maps}, "
          f"ideal_masks={args.ideal_masks}", flush=True)
    try:
        if args.ideal_masks:
            x, y, info = generate_ideal_masks(
                args.patterns, args.n_ideal_per_pattern,
                noise=args.ideal_noise, seed=args.seed)
        elif args.importance_maps:
            x, y, info = load_importance_maps(
                args.patterns, args.ckpt_root,
                importance_name=args.importance_name)
        else:
            x, y, info = load_selected_masks(args.patterns, args.ckpt_root)
        importance_mode = args.importance_maps
        train_loader, val_loader, _, _ = make_loaders(
            x, y, config.CVAE_VAL_FRACTION, args.batch_size, args.seed)

        loss_fn = cvae_loss_importance if importance_mode else cvae_loss

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[cvae] using {device}", flush=True)
        torch.cuda.empty_cache() if device.type == "cuda" else None
        model = CVAE(config.MASK_DIM, args.latent_dim, args.hidden).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        out_dir = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        best_val = float("inf")
        train_log, val_log = [], []
        t0 = time.time()
        for epoch in range(1, args.epochs + 1):
            model.train()
            tot_tr = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits, mu, logvar = model(xb, yb)   # yb is (B, 4) +-1
                loss, recon, kl = loss_fn(logits, xb, mu, logvar, args.beta)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot_tr += loss.item() * xb.size(0)
            train_log.append(tot_tr / len(train_loader.dataset))

            model.eval()
            tot_va, tot_recon, tot_kl = 0.0, 0.0, 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits, mu, logvar = model(xb, yb)
                    loss, recon, kl = loss_fn(logits, xb, mu, logvar, args.beta)
                    tot_va += loss.item() * xb.size(0)
                    tot_recon += recon.item() * xb.size(0)
                    tot_kl += kl.item() * xb.size(0)
            val_log.append(tot_va / len(val_loader.dataset))

            if val_log[-1] < best_val:
                best_val = val_log[-1]
                torch.save(model.state_dict(), out_dir / "cvae_best.pt")

            if epoch % 10 == 0 or epoch == args.epochs:
                n_va = len(val_loader.dataset)
                print(f"[cvae] epoch {epoch:3d}/{args.epochs} "
                      f"train={train_log[-1]:.4f} val={val_log[-1]:.4f} "
                      f"recon={tot_recon / n_va:.4f} "
                      f"kl={tot_kl / n_va:.4f}",
                      flush=True)

        plot_losses(train_log, val_log, config.PLOT_CVAE_DIR)
        plot_reconstructions(model, x, y, args.patterns, config.PLOT_CVAE_DIR,
                             vmax=x.max().item() if importance_mode else 1.0)
        plot_condition_effect(model, config.PATTERNS, config.PLOT_CVAE_DIR)

        meta = {
            "patterns": args.patterns,
            "latent_dim": args.latent_dim,
            "hidden": args.hidden,
            "beta": args.beta,
            "epochs": args.epochs,
            "best_val_loss": best_val,
            "n_train": len(train_loader.dataset),
            "n_val": len(val_loader.dataset),
            "per_pattern": info,
            "final_train_loss": train_log[-1],
            "final_val_loss": val_log[-1],
        }
        torch.save(meta, out_dir / "cvae_meta.pt")
        print(f"[cvae] best val loss {best_val:.4f}; saved "
              f"{out_dir / 'cvae_best.pt'}", flush=True)
        print(f"[cvae] plots -> {config.PLOT_CVAE_DIR / 'cvae_loss.png'}, "
              f"{config.PLOT_CVAE_DIR / 'cvae_reconstructions.png'}, "
              f"{config.PLOT_CVAE_DIR / 'cvae_condition_effect.png'}",
              flush=True)
    except Exception as e:
        print(f"[cvae] FATAL: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise


def do_sample(args) -> None:
    out_dir = args.out_dir
    model = CVAE(config.MASK_DIM, args.latent_dim, args.hidden)
    model.load_state_dict(torch.load(out_dir / "cvae_best.pt",
                                     weights_only=True))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[cvae] using {device}", flush=True)
    model.to(device).eval()

    gen = torch.Generator(device=device).manual_seed(args.seed)
    n_per = 4
    out = {}
    # Loop over patterns for the labelled plot, but they all share the same
    # unconditional prior (cond_dim == 0), so the masks are pattern-agnostic.
    for tag, pats in (("TRAIN", list(args.patterns)),
                      ("TEST", list(config.CVAE_TEST_PATTERNS))):
        for pat in pats:
            lab = config.pattern_to_pm1(pat).to(device).float().unsqueeze(0)
            if args.k_active:
                m = model.sample_topk(lab, n_per, args.k_active,
                                      generator=gen).cpu()
            else:
                m = model.sample(lab, n_per, generator=gen).cpu()
            key = f"{tag} {pat}"
            out[key] = m.reshape(n_per, config.SEQ_LEN, config.H)
            print(f"[cvae] sampled {tag} {pat}: "
                  f"sparsity={(m == 1).float().mean().item():.3f}", flush=True)
    torch.save(out, out_dir / "cvae_samples.pt")
    plot_samples(out, config.PLOT_CVAE_DIR)
    print(f"[cvae] samples -> {out_dir / 'cvae_samples.pt'}", flush=True)
    print(f"[cvae] plot     -> {config.PLOT_CVAE_DIR / 'cvae_generated.png'}",
          flush=True)


def main() -> None:
    args = build_parser().parse_args()
    print(f"[cvae] main() called, mode={args.mode}", flush=True)
    set_seed(args.seed)
    if args.mode == "train":
        do_train(args)
    else:
        do_sample(args)


if __name__ == "__main__":
    main()