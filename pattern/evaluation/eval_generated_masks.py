"""Evaluate how well CVAE-generated masks work on the pattern task.

For every pattern, train masked MLPs (BCE) with masks from several sources:
  random, ideal (Toeplitz), cvae, mean_imp, det_reg, top10%.

Report per-method val BCE and val accuracy.
"""

import argparse
import json
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
from data.generate import make_dataset, ideal_mask  # noqa: E402
from evaluation.baselines import DetRegressor, MeanImportance  # noqa: E402
from models.cvae import CVAE  # noqa: E402
from models.mlp import (  # noqa: E402
    BatchedMaskedMLP, generate_masks, get_train_batch,
)

OUT_DIR = config.EVAL_DIR
BASE_METHODS = ["random", "ideal", "cvae", "mean_imp", "det_reg"]


def build_parser():
    p = argparse.ArgumentParser(description="Evaluate generated masks.")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--n_masks", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=config.TRAIN_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--gpu_id", type=int, default=None)
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--patterns", type=str, nargs="+", default=None)
    return p


def masks_for_method(pat, method, n, cvae, device, mean_imp=None,
                     det_reg=None):
    k_active = config.K_ACTIVE
    if method == "random":
        return generate_masks(n, config.SEQ_LEN, config.H, config.P,
                               seed=int(pat, 2) * 1000 + 1).to(device)
    if method == "ideal":
        m = ideal_mask().unsqueeze(0).expand(n, -1, -1)
        return m.to(device)
    if method == "cvae":
        pm1 = config.pattern_to_pm1(pat).view(1, 4).to(device)
        m = cvae.sample_topk(pm1, n, k_active)
        return m.reshape(n, config.SEQ_LEN, config.H).to(device)
    if method == "mean_imp":
        base = mean_imp(pat, k_active)
        return base.unsqueeze(0).expand(n, -1, -1).to(device)
    if method == "det_reg":
        base = det_reg.mask(pat, k_active)
        return base.unsqueeze(0).expand(n, -1, -1).to(device)
    raise ValueError(method)


def train_and_eval(masks, pat, steps, batch_size, lr, x_val, y_val):
    n = masks.size(0)
    model = BatchedMaskedMLP(n, config.SEQ_LEN, config.H).to(x_val.device)
    model.load_masks(masks)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    seed0 = int(pat, 2) * 10_003 + n
    for step in range(steps):
        xb, yb = get_train_batch(pat, batch_size, seed0 + step)
        xb, yb = xb.to(x_val.device), yb.to(x_val.device)
        pred = model(xb)
        loss = F.binary_cross_entropy_with_logits(
            pred, yb.unsqueeze(1).expand_as(pred))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    vl = model.val_loss(x_val, y_val, config.VAL_BATCH_SIZE)
    va = model.val_acc(x_val, y_val, config.VAL_BATCH_SIZE)
    return vl, va


def main():
    args = build_parser().parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    patterns = args.patterns if args.patterns else list(config.PATTERNS)

    suffix = ""
    if args.gpu_id is not None:
        patterns = [p for i, p in enumerate(patterns)
                    if i % args.num_gpus == args.gpu_id]
        suffix = f"_gpu{args.gpu_id}"
        print(f"[eval] gpu {args.gpu_id}/{args.num_gpus} -> {patterns}", flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cvae = CVAE(config.MASK_DIM, config.LATENT_DIM, config.CVAE_HIDDEN)
    cvae.load_state_dict(torch.load(config.CVAE_DIR / "cvae_best.pt",
                                    weights_only=True))
    cvae.to(device).eval()

    mean_imp = MeanImportance()
    det_reg = DetRegressor()
    det_reg.load_state_dict(torch.load(OUT_DIR / "det_reg.pt", weights_only=True))
    det_reg.to(device).eval()

    results = {}
    for pat in patterns:
        val = make_dataset(pat, config.N_VAL_SAMPLES,
                            seed=1000 + int(pat, 2), pos_fraction=config.POS_FRACTION)
        x_val, y_val = val["x"].to(device), val["y"].to(device)

        methods = list(BASE_METHODS)
        methods.append("top10%")

        methods_and_masks = []
        for method in methods:
            if method == "top10%":
                path = config.pattern_dir(pat) / "best10pct.pt"
                if not path.exists():
                    continue
                d = torch.load(path, weights_only=True)
                masks = d["masks"][:args.n_masks].to(device)
            else:
                masks = masks_for_method(pat, method, args.n_masks, cvae,
                                          device, mean_imp, det_reg)
            methods_and_masks.append((method, masks))

        all_masks = torch.cat([m for _, m in methods_and_masks], dim=0)
        all_vl, all_va = train_and_eval(all_masks, pat, args.steps,
                                        args.batch_size, args.lr, x_val, y_val)
        idx = 0
        for method, masks in methods_and_masks:
            n = masks.size(0)
            vl = all_vl[idx:idx + n]
            va = all_va[idx:idx + n]
            idx += n
            stats = {"mean_bce": vl.mean().item(),
                     "median_bce": vl.median().item(),
                     "min_bce": vl.min().item(),
                     "mean_acc": va.mean().item(),
                     "min_acc": va.min().item(),
                     "sparsity": masks.float().mean().item()}
            results[f"{pat}:{method}"] = stats
            print(f"[eval] {pat} {method:7s}: "
                  f"bce={stats['mean_bce']:.3e} "
                  f"acc={stats['mean_acc']:.4f} "
                  f"sp={stats['sparsity']:.3f}", flush=True)

    out_pt = OUT_DIR / f"eval_results{suffix}.pt"
    out_json = OUT_DIR / f"eval_results{suffix}.json"
    torch.save(results, out_pt)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] saved -> {out_json}")

    if args.gpu_id is None:
        plot_results(results)


def plot_results(results):
    config.ensure_plot_dirs()
    order = ["random", "mean_imp", "det_reg", "cvae", "top10%", "ideal"]
    patterns = config.PATTERNS
    train_set = set(config.CVAE_TRAIN_PATTERNS)
    n_methods = len(order)
    width = 0.8 / n_methods

    for metric, ylabel, fname, logscale in [
        ("mean_bce", "mean val BCE (log)", "eval_bce.png", True),
        ("mean_acc", "mean val accuracy", "eval_acc.png", False),
    ]:
        fig, ax = plt.subplots(figsize=(14, 5))
        x = np.arange(len(patterns))
        for i, m in enumerate(order):
            vals = [results.get(f"{pat}:{m}", {}).get(metric, np.nan)
                    for pat in patterns]
            hatch = "" if True else "//"
            ax.bar(x + (i - (n_methods - 1) / 2) * width, vals, width,
                   label=m, hatch=hatch if m == "ideal" else "")
        for xi, pat in enumerate(patterns):
            ax.annotate("T" if pat in train_set else "E",
                        (xi, 0.02), ha="center", fontsize=7,
                        color="green" if pat in train_set else "orange")
        ax.set_xticks(x)
        ax.set_xticklabels(patterns, rotation=45)
        if logscale:
            ax.set_yscale("log")
        ax.set_xlabel("pattern (T=train, E=eval/test)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Generated vs ideal vs random masks ({metric})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        out = config.PLOT_EVAL_DIR / fname
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"[eval] plot -> {out}")


if __name__ == "__main__":
    main()
