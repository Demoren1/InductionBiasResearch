"""Evaluate statistical plausibility of masks under a trained unconditional CVAE.

For each category we compute a per-sample importance-weighted auto-encoder
(IWAE) estimate of log p(x) with K=64 posterior samples, matching the
BCE-sum training objective of the target CVAE models. The encoder maps the
mask x (a 64-dim vector; importance *or* binary) to a Gaussian posterior
q(z|x); we draw z_k ~ q(z|x) via the reparameterization trick and compute

    w_k = log p(x | z_k) - kl(x)            # kl is shared across k
    log p(x) ~ logsumexp_k(w_k) - log(K)   # in nats per sample

The "recon term" reported is E_q[log p(x|z)] = mean_k log p(x|z_k), and
"KL" is the per-sample posterior KL from the standard-normal prior.

Categories evaluated:
  real_top10_val  held-out (seed=CVAE_SEED) split of top-10% importance maps
  real_bottom90   lowest-priority (non-top-10%) maps, 100 per pattern
  gen_continuous  256 p-maps decoded from z ~ N(0, I)
  gen_binary      same 256 p-maps binarized to their top-32 support
  random_binary   256 i.i.d. Bernoulli(0.5) masks
  gold_perms      ideal Toeplitz mask + 128 random column permutations

Usage:
  python evaluation/mask_plausibility.py --ckpt PATH --tag _name
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from data.generate import ideal_mask  # noqa: E402
from models.cvae import CVAE, load_importance_maps, make_loaders  # noqa: E402


def build_model(ckpt, device) -> CVAE:
    model = CVAE(config.MASK_DIM, config.LATENT_DIM, config.CVAE_HIDDEN)
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.to(device).eval()
    return model


@torch.no_grad()
def per_sample_iwae(model, x, device, K, chunk=512, generator=None):
    """(n, D) masks -> per-sample (iwae, recon, kl) tensors (n,)."""
    x = x.to(device)
    n = x.size(0)
    c = model.condition(torch.zeros(n, 4, device=device))
    mu, logvar = model.encode(x, c)
    std = torch.exp(0.5 * logvar)
    # KL is shared across the K posterior draws (depends only on mu/logvar).
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

    iwave = torch.empty(n, device=device)
    recon = torch.empty(n, device=device)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        m = end - start
        if generator is not None:
            eps = torch.randn(m, K, model.latent_dim, device=device,
                              generator=generator)
        else:
            eps = torch.randn(m, K, model.latent_dim, device=device)
        z = mu[start:end].unsqueeze(1) + std[start:end].unsqueeze(1) * eps
        # cond_dim==0: expand c to (m, K, 0) so it broadcasts with z.
        cb = c[start:end].unsqueeze(1).expand(-1, K, -1)
        logits = model.decode(z, cb)                       # (m, K, D)
        xb = x[start:end].unsqueeze(1)                        # (m, 1, D)
        logp = -F.binary_cross_entropy_with_logits(
            logits, xb.expand_as(logits), reduction="none").sum(dim=-1)
        recon_step = logp.mean(dim=-1)                        # (m,)
        w = logp - kl[start:end].unsqueeze(1)                 # (m, K)
        iwave_step = torch.logsumexp(w, dim=-1) - torch.log(
            torch.tensor(float(K), device=device))
        iwave[start:end] = iwave_step
        recon[start:end] = recon_step
    return iwave, recon, kl


def binarize_topk(p, k):
    """(N, D) probs -> (N, D) 0/1 with exactly the top-k entries set."""
    _, idx = p.topk(k, dim=-1)
    out = torch.zeros_like(p)
    out.scatter_(-1, idx, 1.0)
    return out


def real_top10_val():
    """Held-out (CVAE_SEED) split of the top-10% importance maps (~480)."""
    x, y, info = load_importance_maps(
        config.PATTERNS, config.CKPT_DIR, importance_name="importance.pt",
        top_frac=0.1)
    _, _, _, val_idx = make_loaders(
        x, y, config.CVAE_VAL_FRACTION, config.CVAE_BATCH_SIZE,
        config.CVAE_SEED)
    xv = x[val_idx]
    print(f"[plaus] real_top10_val: top10maps={int(x.size(0))} "
          f"val split={int(xv.size(0))}", flush=True)
    return xv.cpu()


def real_bottom90(seed):
    """100 per pattern lowest-priority (non-top-10%) maps = 1600."""
    g = torch.Generator().manual_seed(seed)
    parts = []
    for pat in config.PATTERNS:
        d = torch.load(config.pattern_dir(pat) / "importance.pt",
                       weights_only=True)
        val_loss = d["val_loss"]
        n = val_loss.size(0)
        idx = torch.argsort(val_loss)
        bottom = idx[int(round(0.1 * n)):]
        pick = bottom[torch.randperm(bottom.size(0), generator=g)[:100]]
        parts.append(d["importance"][pick])
        print(f"[plaus] real_bottom90: pattern={pat} "
              f"picked {len(pick)}/{bottom.size(0)}", flush=True)
    return torch.cat(parts).reshape(-1, config.MASK_DIM).float()


def gen_continuous(model, device, n=256, seed=0):
    """n p-maps decoded from z ~ N(0, I)."""
    g = torch.Generator(device=device).manual_seed(seed)
    c = model.condition(torch.zeros(n, 4, device=device))
    z = torch.randn(n, model.latent_dim, device=device, generator=g)
    logits = model.decode(z, c)
    p = torch.sigmoid(logits)
    print(f"[plaus] gen_continuous: {n} p-maps", flush=True)
    return p


def gold_perms(seed):
    """ideal Toeplitz mask + 128 random column permutations (129 masks)."""
    g = torch.Generator().manual_seed(seed)
    masks = [ideal_mask().float().flatten()]
    for _ in range(128):
        perm = torch.randperm(config.H, generator=g)
        masks.append(ideal_mask()[:, perm].float().flatten())
    out = torch.stack(masks)
    print(f"[plaus] gold_perms: {out.size(0)} masks", flush=True)
    return out


def summarize(cat, tensors):
    iw, rc, kl = tensors
    return {"n": int(iw.size(0)),
            "iwae_mean": float(iw.mean()),
            "iwae_median": float(iw.median()),
            "recon": float(rc.mean()),
            "kl": float(kl.mean()),
            "iwae_all": iw.cpu().numpy()}


def print_table(rows, order):
    print("\n[plaus] ============ TABLE: IWAE log-likelihood (K) ============",
          flush=True)
    hdr = (f"{'category':<16}{'n':>6}{'iwae_mean':>12}{'iwae_med':>12}"
           f"{'recon':>12}{'kl':>10}")
    print("[plaus] " + hdr, flush=True)
    for name in order:
        r = rows[name]
        print(f"[plaus] {name:<16}{r['n']:>6}{r['iwae_mean']:>12.2f}"
              f"{r['iwae_median']:>12.2f}{r['recon']:>12.2f}{r['kl']:>10.2f}",
              flush=True)
    rk = sorted(rows, key=lambda k: rows[k]["iwae_mean"], reverse=True)
    print("[plaus] rank by mean IWAE: "
          + " > ".join(f"{k} ({rows[k]['iwae_mean']:.2f})" for k in rk),
          flush=True)
    print("[plaus] ==========================================================",
          flush=True)


def plot_figure(order, rows, tag, out_dir, K):
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [rows[name]["iwae_all"] for name in order]
    bp = ax.boxplot(data, tick_labels=order, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightsteelblue")
        patch.set_alpha(0.8)
    med = rows["real_top10_val"]["iwae_median"]
    ax.axhline(med, color="crimson", ls="--", lw=1.5,
               label=f"median real_top10_val = {med:.2f}")
    ax.set_ylabel(f"IWAE log p(x), nats (K={K})")
    ax.set_title(f"IWAE log-likelihood by mask category (ckpt{tag})")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / f"mask_plausibility{tag}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute IWAE log-likelihood (K) of mask categories.")
    parser.add_argument("--ckpt", type=Path, required=True,
                        help="path to CVAE state dict")
    parser.add_argument("--tag", type=str, default="",
                        help="suffix for output filenames")
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    out_dir = config.PLOT_DIR / "z_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[plaus] ckpt={args.ckpt} K={args.K} seed={args.seed} "
          f"device={device}", flush=True)

    model = build_model(args.ckpt, device)
    ig = torch.Generator(device=device).manual_seed(args.seed)

    cat_t = {}
    cat_t["real_top10_val"] = per_sample_iwae(
        model, real_top10_val().to(device), device, args.K, generator=ig)
    cat_t["real_bottom90"] = per_sample_iwae(
        model, real_bottom90(args.seed).to(device), device, args.K,
        generator=ig)
    p_cont = gen_continuous(model, device, 256, args.seed)
    cat_t["gen_continuous"] = per_sample_iwae(
        model, p_cont, device, args.K, generator=ig)
    cat_t["gen_binary"] = per_sample_iwae(
        model, binarize_topk(p_cont, config.K_ACTIVE), device, args.K,
        generator=ig)
    rg = torch.Generator(device=device).manual_seed(args.seed)
    rand = torch.rand(256, config.MASK_DIM, device=device, generator=rg)
    cat_t["random_binary"] = per_sample_iwae(
        model, rand, device, args.K, generator=ig)
    cat_t["gold_perms"] = per_sample_iwae(
        model, gold_perms(args.seed).to(device), device, args.K, generator=ig)

    rows = {name: summarize(name, tens)
            for name, tens in cat_t.items()}
    order = ["real_top10_val", "real_bottom90", "gen_continuous",
             "gen_binary", "random_binary", "gold_perms"]
    print_table(rows, order)

    out_path = out_dir / f"mask_plausibility{args.tag}.png"
    plot_figure(order, rows, args.tag, out_dir, args.K)
    print(f"[plaus] figure -> {out_path}", flush=True)


if __name__ == "__main__":
    main()