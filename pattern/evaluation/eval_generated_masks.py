"""Evaluate how well CVAE-generated masks work on the pattern task.

For every pattern, train masked MLPs (BCE) with masks from several sources:
  random, random_exact32, ideal (Toeplitz), cvae, mean_imp, det_reg, top10%.

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
    BatchedMaskedMLP, generate_fixed_sparsity_masks, generate_masks,
    get_train_batch,
)

OUT_DIR = config.EVAL_DIR
BASE_METHODS = ["random", "ideal", "cvae", "mean_imp", "det_reg"]
ALL_METHODS = ["random", "random_exact32", "ideal", "cvae", "mean_imp",
               "det_reg", "top10%"]


def build_parser():
    p = argparse.ArgumentParser(description="Evaluate generated masks.")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--n_masks", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=config.TRAIN_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--gpu_id", type=int, default=None)
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--patterns", type=str, nargs="+", default=None)
    p.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=None,
                   help="methods to evaluate. Supplying this list does not add "
                        "the held-out top10%% oracle implicitly")
    p.add_argument("--cvae_ckpt", type=Path, default=config.CVAE_DIR / "cvae_best.pt",
                   help="path to the CVAE state dict to evaluate")
    p.add_argument("--out_suffix", type=str, default="",
                   help="suffix appended to eval_results output filenames")
    p.add_argument("--seed", type=int, default=config.CVAE_SEED,
                   help="random seed for model initialization and VAE samples")
    p.add_argument("--include_exact32", action="store_true",
                   help="add a uniform fixed-cardinality random baseline")
    p.add_argument("--prior_patterns", nargs="+", default=None,
                   help="meta-train patterns pooled by mean_imp (default: all)")
    p.add_argument("--train_patterns", nargs="+", default=None,
                   help="meta-train patterns used for det_reg; checked against "
                        "new-format checkpoint metadata when available")
    p.add_argument("--importance_name", type=str, default="importance.pt",
                   help="importance-map filename used by mean_imp")
    p.add_argument("--top_frac", type=float, default=0.0,
                   help="lowest-val-loss fraction per prior task for mean_imp "
                        "(0 keeps all maps)")
    p.add_argument("--det_reg_checkpoint", "--det_reg_ckpt",
                   dest="det_reg_checkpoint", type=Path, default=None,
                   help="det_reg checkpoint (default: <out_dir>/det_reg.pt)")
    p.add_argument("--out_dir", type=Path, default=OUT_DIR,
                   help="directory for evaluation results (and plots for an "
                        "explicit non-default directory)")
    return p


def masks_for_method(pat, method, n, cvae, device, mean_imp=None,
                     det_reg=None, generator=None, exact_seed=0):
    k_active = config.K_ACTIVE
    if method == "random":
        return generate_masks(n, config.SEQ_LEN, config.H, config.P,
                               seed=int(pat, 2) * 1000 + 1).to(device)
    if method == "random_exact32":
        return generate_fixed_sparsity_masks(
            n, config.SEQ_LEN, config.H, k_active,
            seed=exact_seed + int(pat, 2) * 1000 + 2).to(device)
    if method == "ideal":
        m = ideal_mask().unsqueeze(0).expand(n, -1, -1)
        return m.to(device)
    if method == "cvae":
        pm1 = config.pattern_to_pm1(pat).view(1, 4).to(device)
        m = cvae.sample_topk(pm1, n, k_active, generator=generator)
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


def _load_det_reg(checkpoint: Path, device, expected_patterns=None):
    """Load both legacy state_dict-only and provenance-aware checkpoints."""
    payload = torch.load(checkpoint, weights_only=True, map_location=device)
    state_dict = payload.get("state_dict", payload)
    stored_patterns = payload.get("patterns") if isinstance(payload, dict) else None
    if expected_patterns is not None and stored_patterns is not None:
        if list(expected_patterns) != list(stored_patterns):
            raise ValueError(
                "det_reg checkpoint provenance does not match --train_patterns: "
                f"checkpoint={stored_patterns}, requested={list(expected_patterns)}")
    elif expected_patterns is not None and stored_patterns is None:
        print("[eval] warning: legacy det_reg checkpoint has no train-pattern "
              "metadata; --train_patterns cannot be verified", flush=True)
    model = DetRegressor()
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def main():
    args = build_parser().parse_args()
    if args.top_frac and not 0.0 < args.top_frac <= 1.0:
        raise ValueError("--top_frac must be in (0, 1]")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    patterns = args.patterns if args.patterns else list(config.PATTERNS)

    suffix = ""
    if args.gpu_id is not None:
        patterns = [p for i, p in enumerate(patterns)
                    if i % args.num_gpus == args.gpu_id]
        suffix = f"_gpu{args.gpu_id}"
        print(f"[eval] gpu {args.gpu_id}/{args.num_gpus} -> {patterns}", flush=True)
    suffix = f"{suffix}{args.out_suffix}"

    if args.methods is None:
        methods = list(BASE_METHODS)
        # Historic default: include the per-task selected-mask reference.
        methods.append("top10%")
        if args.include_exact32:
            methods.append("random_exact32")
    else:
        methods = list(args.methods)
        if args.include_exact32 and "random_exact32" not in methods:
            methods.append("random_exact32")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cvae = None
    if "cvae" in methods:
        cvae = CVAE(config.MASK_DIM, config.LATENT_DIM, config.CVAE_HIDDEN)
        cvae.load_state_dict(torch.load(args.cvae_ckpt, weights_only=True,
                                        map_location=device))
        cvae.to(device).eval()

    mean_imp = None
    if "mean_imp" in methods:
        mean_imp = MeanImportance(args.prior_patterns,
                                  importance_name=args.importance_name,
                                  top_frac=args.top_frac or None)

    det_reg = None
    if "det_reg" in methods:
        det_checkpoint = (args.out_dir / "det_reg.pt"
                          if args.det_reg_checkpoint is None
                          else args.det_reg_checkpoint)
        det_reg = _load_det_reg(det_checkpoint, device, args.train_patterns)

    results = {}
    for pat in patterns:
        pattern_seed = args.seed + int(pat, 2)
        torch.manual_seed(pattern_seed)
        sample_generator = torch.Generator(device=device).manual_seed(pattern_seed)
        val = make_dataset(pat, config.N_VAL_SAMPLES,
                            seed=1000 + int(pat, 2), pos_fraction=config.POS_FRACTION)
        x_val, y_val = val["x"].to(device), val["y"].to(device)

        methods_and_masks = []
        for method in methods:
            if method == "top10%":
                path = config.pattern_dir(pat) / "best10pct.pt"
                if not path.exists():
                    if args.methods is not None:
                        raise FileNotFoundError(
                            f"explicit top10% oracle requested but missing: {path}")
                    continue
                d = torch.load(path, weights_only=True)
                masks = d["masks"][:args.n_masks].to(device)
            else:
                masks = masks_for_method(pat, method, args.n_masks, cvae,
                                          device, mean_imp, det_reg,
                                          sample_generator,
                                          exact_seed=args.seed * 1_000_000)
            methods_and_masks.append((method, masks))

        if not methods_and_masks:
            raise ValueError(f"no masks available for pattern {pat}")

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
                     "sparsity": masks.float().mean().item(),
                     # Retain the paired per-mask observations for CIs and
                     # uncertainty diagnostics; scalar fields above preserve
                     # the legacy summary schema.
                     "bce": vl.detach().cpu().tolist(),
                     "acc": va.detach().cpu().tolist()}
            results[f"{pat}:{method}"] = stats
            print(f"[eval] {pat} {method:7s}: "
                  f"bce={stats['mean_bce']:.3e} "
                  f"acc={stats['mean_acc']:.4f} "
                  f"sp={stats['sparsity']:.3f}", flush=True)

    out_pt = args.out_dir / f"eval_results{suffix}.pt"
    out_json = args.out_dir / f"eval_results{suffix}.json"
    torch.save(results, out_pt)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] saved -> {out_json}")

    if args.gpu_id is None:
        plot_dir = args.out_dir if args.out_dir != OUT_DIR else None
        plot_results(results, suffix=args.out_suffix, patterns=patterns,
                     out_dir=plot_dir)


def plot_results(results, suffix="", patterns=None, out_dir=None):
    """Plot an evaluation result set, including a held-out subset if supplied."""
    if out_dir is None:
        config.ensure_plot_dirs()
        out_dir = config.PLOT_EVAL_DIR
    else:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    order = ["random", "random_exact32", "mean_imp", "det_reg", "cvae",
             "top10%", "ideal"]
    patterns = list(config.PATTERNS if patterns is None else patterns)
    order = [method for method in order
             if any(f"{pat}:{method}" in results for pat in patterns)]
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
        ax.set_xticks(x)
        ax.set_xticklabels(patterns, rotation=45)
        if logscale:
            ax.set_yscale("log")
        ax.set_xlabel("pattern")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Generated vs ideal vs random masks ({metric})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        out = out_dir / f"{Path(fname).stem}{suffix}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"[eval] plot -> {out}")


if __name__ == "__main__":
    main()
