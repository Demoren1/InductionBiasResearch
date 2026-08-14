"""Train the CVAE on the selected masks.

Usage:
    python models/train_cvae.py --epochs 80 --latent_dim 32

The model is conditioned on the MA kernel size and the target offset.
Best model (by validation loss), loss curves and mask reconstructions are
written to outputs/cvae/.

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
    p = argparse.ArgumentParser(description="Train CVAE on selected masks.")
    p.add_argument("--mode", choices=["train", "sample"], default="train")
    p.add_argument("--epochs", type=int, default=config.CVAE_EPOCHS)
    p.add_argument("--batch_size", type=int, default=config.CVAE_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.CVAE_LR)
    p.add_argument("--beta", type=float, default=config.CVAE_BETA)
    p.add_argument("--latent_dim", type=int, default=config.LATENT_DIM)
    p.add_argument("--hidden", type=int, default=config.CVAE_HIDDEN)
    p.add_argument("--seed", type=int, default=config.CVAE_SEED)
    p.add_argument("--kernels", type=int, nargs="+", default=config.KERNELS)
    p.add_argument("--offsets", type=int, nargs="+", default=config.OFFSETS)
    p.add_argument("--ideal_masks", action="store_true",
                   help="train on synthetic ideal-like masks (bottom-k-row support)")
    p.add_argument("--n_ideal_per_kernel", type=int, default=1000,
                   help="synthetic masks per kernel when --ideal_masks is set")
    p.add_argument("--ideal_noise", type=float, default=0.05,
                   help="bit-flip noise fraction for ideal masks")
    p.add_argument("--importance_maps", action="store_true",
                   help="train on continuous importance maps (MSE loss)")
    p.add_argument("--ckpt_root", type=Path, default=config.CKPT_DIR)
    p.add_argument("--out_dir", type=Path, default=config.CVAE_DIR)
    p.add_argument("--k_active", type=int, default=0,
                   help="use sample_topk with this many active entries per mask "
                        "(0 = default Bernoulli sampling)")
    return p


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def show_masks(ax, masks: torch.Tensor, title: str) -> None:
    """Draw one mask (or mean) as a 2D heatmap in ax."""
    m = masks.reshape(config.L, config.H).numpy()
    ax.imshow(m, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_title(title, fontsize=8)


def plot_reconstructions(model: CVAE, x, y, kernels, offsets,
                         out_dir: Path, vmax: float = 1.0) -> Path:
    """Per (k, s) pair: original mask | p(mask=1) map | stochastic recon.

    One example per (k, s) pair (zip pairing of kernels and offsets).
    vmax controls the colormap ceiling (1.0 for binary masks,
    x.max() for continuous importance maps).
    """
    model.eval()
    device = next(model.parameters()).device
    xd, yd = x.to(device), y.to(device)
    p_map = model.prob(xd, yd[:, 0], yd[:, 1]).cpu()
    gen = torch.Generator(device=device).manual_seed(0)
    r_sample = model.reconstruct_sample(xd, yd[:, 0], yd[:, 1],
                                        generator=gen).cpu()

    n_pairs = len(kernels)
    panels = 3
    n_cols = panels
    fig, axes = plt.subplots(n_pairs, n_cols,
                             figsize=(3.1 * n_cols, 1.9 * n_pairs),
                             squeeze=False)
    for r, (k, s) in enumerate(zip(kernels, offsets)):
        idxs = ((y[:, 0] == float(k)) & (y[:, 1] == float(s))
                ).nonzero(as_tuple=True)[0]
        gi = idxs[0]
        o = axes[r, 0]
        o.imshow(x[gi].reshape(config.L, config.H), cmap="viridis",
                 vmin=0, vmax=vmax, aspect="auto")
        o.set_title(f"orig k={k} s={s}", fontsize=8)
        p = axes[r, 1]
        p.imshow(p_map[gi].reshape(config.L, config.H), cmap="viridis",
                 vmin=0, vmax=vmax, aspect="auto")
        p.set_title("p(mask=1)", fontsize=8)
        st = axes[r, 2]
        st.imshow(r_sample[gi].reshape(config.L, config.H),
                  cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
        st.set_title("stoch. recon", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("CVAE on top-10% masks: original | p(mask=1) | Bernoulli recon")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.03,
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


def plot_samples(samples: dict, out_dir: Path) -> Path:
    """Plot masks generated by the CVAE (a few per (k, s) pair)."""
    n_pairs = len(samples)
    n_per = next(iter(samples.values())).size(0)
    fig, axes = plt.subplots(n_pairs, n_per, figsize=(1.6 * n_per,
                                                      1.7 * n_pairs),
                             squeeze=False)
    for r, (key, ms) in enumerate(samples.items()):
        label = (f"k={key[0]} s={key[1]}" if isinstance(key, tuple)
                 else f"kernel {key}")
        for c in range(n_per):
            axes[r, c].imshow(ms[c].numpy(), cmap="viridis", vmin=0, vmax=1,
                              aspect="auto")
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if c == 0:
                axes[r, c].set_ylabel(label, fontsize=9)
    fig.suptitle("CVAE-generated masks (coupled sampling per (k, s))")
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
                args.kernels, args.n_ideal_per_kernel, args.offsets,
                config.L, config.H, noise=args.ideal_noise, seed=args.seed)
        elif args.importance_maps:
            x, y, info = load_importance_maps(args.kernels, args.offsets,
                                              args.ckpt_root)
        else:
            x, y, info = load_selected_masks(args.kernels, args.offsets,
                                             args.ckpt_root)
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
                logits, mu, logvar = model(xb, yb[:, 0], yb[:, 1])
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
                    logits, mu, logvar = model(xb, yb[:, 0], yb[:, 1])
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
        plot_reconstructions(model, x, y, args.kernels, args.offsets,
                             config.PLOT_CVAE_DIR,
                             vmax=x.max().item() if importance_mode else 1.0)

        meta = {
            "kernels": args.kernels,
            "offsets": args.offsets,
            "latent_dim": args.latent_dim,
            "hidden": args.hidden,
            "beta": args.beta,
            "epochs": args.epochs,
            "best_val_loss": best_val,
            "n_train": len(train_loader.dataset),
            "n_val": len(val_loader.dataset),
            "per_kernel": info,
            "final_train_loss": train_log[-1],
            "final_val_loss": val_log[-1],
        }
        torch.save(meta, out_dir / "cvae_meta.pt")
        print(f"[cvae] best val loss {best_val:.4f}; saved "
              f"{out_dir / 'cvae_best.pt'}", flush=True)
        print(f"[cvae] plots -> {config.PLOT_CVAE_DIR / 'cvae_loss.png'} "
              f"and {config.PLOT_CVAE_DIR / 'cvae_reconstructions.png'}",
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
    for k, s in zip(args.kernels, args.offsets):
        lab_k = torch.tensor([float(k)], device=device)
        lab_s = torch.tensor([float(s)], device=device)
        if args.k_active:
            m = model.sample_det(lab_k, lab_s, args.k_active).cpu()
            m = m.expand(n_per, -1)   # (1, mask_dim) -> (n_per, mask_dim)
        else:
            m = model.sample(lab_k, lab_s, n_per, generator=gen).cpu()
        out[(k, s)] = m.reshape(n_per, config.L, config.H)
        print(f"[cvae] sampled k={k} s={s}: "
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