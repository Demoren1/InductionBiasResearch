"""Bilevel optimization of a latent vector z through a FROZEN VAE decoder.

For each pattern, we maximize the validation performance of a downstream
masked MLP by optimizing the mask through a frozen, unconditional CVAE
decoder.  The optimizable quantity is the latent code z; the mask is the
top-K_ACTIVE support of sigmoid(decoder(z)).  This tests whether the
decoder's learned manifold contains masks that beat random sampling from
the prior (z ~ N(0, I)), i.e. the "z0" prior-mode baseline, and whether a
free 64-bit control ("free" logits) can do better (does the decoder
manifold help or constrain?).

Modes
-----
z    : z = zeros(1, LATENT_DIM) with grad;
       mask_prob = sigmoid(decoder(z)).reshape(SEQ_LEN, H)
free : logits64 = zeros(MASK_DIM) with grad;
       mask_prob = sigmoid(logits64).reshape(SEQ_LEN, H)
z0   : no optimization; mask_prob from the prior mode z = 0 (baseline)

Bilevel loop (modes z/free, per pattern, outer_steps iterations):
  1. fresh MLP (w1 (1,8,8)*0.1, b1 (1,8), w2 (1,8,1)*0.1, b2 (1,1)) and an
     inner Adam (lr=1e-3) created BEFORE warm-up so its state carries over.
  2. warm-up: warmup_steps of standard training with the mask *detached*
     (mask_det = mask_prob.detach()) so no graph reaches z during warm-up.
  3. grad phase: grad_steps more steps with the *live* mask (do NOT detach).
     The mask is recomputed from z each step (fresh graph) so every step's
     backward accumulates d(inner_loss)/d(z) into z.grad; the inner Adam
     keeps updating the MLP in place.
  4. val loss on the first-1024 split, computed with a fresh live mask;
     val_loss.backward() adds d(val)/d(z); grads are clipped to norm 10.0
     and the outer Adam (lr=z_lr) takes a step.

The outer gradient is therefore the sum of the direct-path gradients of
the grad-phase inner losses plus the final validation loss w.r.t. z (the
inner MLP trajectory is unrolled per-step; torch.optim applies its updates
in-place under no_grad, so only direct loss-path gradients reach z).

Final evaluation (all modes incl. z0): top-K_ACTIVE binary mask from the
final mask_prob -> n_repeats fresh batched MLPs trained for final_steps
steps with the standard protocol (Adam 1e-3, get_train_batch stream) ->
mean/std test accuracy and mean test BCE on the last-1024 split.  Two
fixed-reference masks are scored with the identical protocol on the same
split: "random" (seeded Bernoulli(0.5)) and "ideal" (Toeplitz support from
data.generate.ideal_mask).

Machine-readable summary -> outputs/eval/zopt_results.json.  Every mode also
saves its continuous and top-32 binary masks, optimization parameters, test
metrics, and Toeplitz diagnostics -> outputs/eval/zopt_masks.pt.  Heatmaps and
aggregate structural diagnostics go to outputs/plots/z_analysis/.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from data.generate import ideal_mask  # noqa: E402
from evaluation.align_importance import align_map  # noqa: E402
from models.cvae import CVAE  # noqa: E402
from models.mlp import get_train_batch  # noqa: E402

CKPT_DEFAULT = config.CVAE_DIR / "cvae_best.pt"
OUT_JSON = config.EVAL_DIR / "zopt_results.json"
MASK_OUT = config.EVAL_DIR / "zopt_masks.pt"
PLOT_DIR = config.PLOT_DIR / "z_analysis"
LOG_TAG = "[zopt]"


def build_parser():
    p = argparse.ArgumentParser(
        description="Bilevel optimize z through a frozen VAE decoder.")
    p.add_argument("--ckpt", type=Path, default=CKPT_DEFAULT,
                   help="frozen CVAE state dict")
    p.add_argument("--out_json", type=Path, default=OUT_JSON)
    p.add_argument("--mask_out", type=Path, default=MASK_OUT)
    p.add_argument("--plot_dir", type=Path, default=PLOT_DIR)
    p.add_argument("--render_saved", action="store_true",
                   help="render masks and structural metrics without training")
    p.add_argument("--mode", type=str, choices=["z", "free", "z0"],
                   default="z")
    p.add_argument("--outer_steps", type=int, default=30)
    p.add_argument("--warmup_steps", type=int, default=300)
    p.add_argument("--grad_steps", type=int, default=100)
    p.add_argument("--z_lr", type=float, default=0.05,
                   help="outer learning rate (z and free modes)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patterns", type=str, nargs="+", default=None,
                   help="default: all config.PATTERNS")
    p.add_argument("--n_repeats", type=int, default=8,
                   help="MLPs trained per final mask for eval stability")
    p.add_argument("--final_steps", type=int, default=config.TRAIN_STEPS,
                   help="steps for the final fixed-mask protocol (2000)")
    return p


def load_cvae(ckpt, device) -> CVAE:
    model = CVAE(config.MASK_DIM, config.LATENT_DIM, config.CVAE_HIDDEN)
    model.load_state_dict(torch.load(ckpt, weights_only=True,
                                     map_location=device))
    model.to(device).eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    return model


def decode_mask(mode: str, model: CVAE, c: torch.Tensor,
                param: torch.Tensor | None) -> torch.Tensor:
    """(SEQ_LEN, H) mask probs from the optimizable param (z or logits64).

    For mode z0 param is ignored: the prior-mode latent z = 0 is used.
    """
    if mode == "z":
        p = torch.sigmoid(model.decode(param, c))
    elif mode == "free":
        p = torch.sigmoid(param)
    else:  # z0
        z = torch.zeros(1, config.LATENT_DIM, device=c.device)
        p = torch.sigmoid(model.decode(z, c))
    return p.reshape(config.SEQ_LEN, config.H)


def mlp_forward(x: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor,
                w2: torch.Tensor, b2: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
    """Batched masked MLP forward.

    x (B, L); w1 (R, L, H), b1 (R, H), w2 (R, H, 1), b2 (R, 1),
    mask (L, H) -> logits (B, R).
    """
    w1m = w1 * mask                       # mask broadcasts over repeats
    h = F.relu(torch.einsum("bl,rlh->brh", x, w1m) + b1)
    return torch.einsum("brh,rh->br", h, w2.squeeze(-1)) + b2.squeeze(-1)


def fresh_mlp(seed: int, device) -> list:
    """Single fresh inner MLP for the bilevel loop (leading batch dim 1)."""
    g = torch.Generator(device=device).manual_seed(seed)
    w1 = nn.Parameter(torch.randn(1, config.SEQ_LEN, config.H,
                                  generator=g, device=device) * 0.1)
    b1 = nn.Parameter(torch.zeros(1, config.H, device=device))
    w2 = nn.Parameter(torch.randn(1, config.H, 1,
                                  generator=g, device=device) * 0.1)
    b2 = nn.Parameter(torch.zeros(1, 1, device=device))
    return [w1, b1, w2, b2]


def bilevel_optimize(mode: str, pat: str, model: CVAE, c: torch.Tensor,
                     x_val: torch.Tensor, y_val: torch.Tensor,
                     args) -> tuple:
    """Optimize one mask and return its binary, continuous, and parameter forms."""
    pat_int = int(pat, 2)
    if mode == "z":
        param = nn.Parameter(torch.zeros(1, config.LATENT_DIM,
                                         device=x_val.device))
    elif mode == "free":
        param = nn.Parameter(torch.zeros(config.MASK_DIM,
                                         device=x_val.device))
    else:
        param = None
    outer_opt = (torch.optim.Adam([param], lr=args.z_lr)
                 if mode != "z0" else None)

    mask_det = None
    final_val_loss = float("nan")
    z_norm = float("nan")
    n_outer_steps = 1 if mode == "z0" else args.outer_steps
    for outer_idx in range(n_outer_steps):
        if mode != "z0":
            outer_opt.zero_grad(set_to_none=True)
            mask_prob = decode_mask(mode, model, c, param)
            mask_det = mask_prob.detach()
        else:
            mask_prob = None
            mask_det = decode_mask(mode, model, c, None).detach()

        # Fresh MLP + inner Adam; optimizer state carries into grad phase.
        params = fresh_mlp(args.seed * 1_000_000 + pat_int * 1000
                           + outer_idx, x_val.device)
        inner_opt = torch.optim.Adam(params, lr=config.LR)

        # Warm-up: standard training with the mask detached (no graph to z).
        for step in range(args.warmup_steps):
            xb, yb = get_train_batch(pat, config.TRAIN_BATCH_SIZE,
                                     10_000 * pat_int + outer_idx * 1000
                                     + step)
            xb, yb = xb.to(x_val.device), yb.to(x_val.device)
            pred = mlp_forward(xb, *params, mask_det)
            loss = F.binary_cross_entropy_with_logits(
                pred, yb.unsqueeze(1).expand_as(pred))
            inner_opt.zero_grad(set_to_none=True)
            loss.backward()
            inner_opt.step()

        # Grad phase: live mask, graph flows through each step into z.
        if mode != "z0":
            for step in range(args.grad_steps):
                mp = decode_mask(mode, model, c, param)  # fresh graph
                xb, yb = get_train_batch(pat, config.TRAIN_BATCH_SIZE,
                                         10_000 * pat_int + outer_idx * 1000
                                         + args.warmup_steps + step)
                xb, yb = xb.to(x_val.device), yb.to(x_val.device)
                pred = mlp_forward(xb, *params, mp)
                loss = F.binary_cross_entropy_with_logits(
                    pred, yb.unsqueeze(1).expand_as(pred))
                inner_opt.zero_grad(set_to_none=True)
                loss.backward()
                inner_opt.step()

        # Validation loss (first-1024 split), differentiable wrt mask.
        mp = (decode_mask(mode, model, c, param) if mode != "z0"
              else mask_det)
        pred_val = mlp_forward(x_val, *params, mp)
        val_loss = F.binary_cross_entropy_with_logits(
            pred_val, y_val.unsqueeze(1).expand_as(pred_val))

        if mode != "z0":
            val_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_([param], 10.0)
            outer_opt.step()
        else:
            grad_norm = torch.tensor(0.0)

        final_val_loss = val_loss.item()
        z_norm = (param.detach().norm().item() if mode == "z"
                  else float("nan"))
        if (outer_idx % 5 == 0 or outer_idx == n_outer_steps - 1
                or not torch.isfinite(val_loss)):
            print(f"{LOG_TAG} pat={pat} mode={mode} "
                  f"outer={outer_idx + 1}/{n_outer_steps} "
                  f"val_loss={final_val_loss:.4f} "
                  f"z_norm={z_norm:.3f} grad_norm={grad_norm.item():.2f}",
                  flush=True)
        if not torch.isfinite(val_loss):
            print(f"{LOG_TAG} WARNING pat={pat} mode={mode}: non-finite "
                  f"val_loss at outer step {outer_idx}; stopping early",
                  flush=True)
            break

    # Final binary mask: top-K_ACTIVE of the 64 mask probs.
    with torch.no_grad():
        mp = (decode_mask(mode, model, c, param) if mode != "z0"
              else mask_det)
        flat = mp.reshape(-1)
        _, top_idx = flat.topk(config.K_ACTIVE)
        binary = torch.zeros_like(flat)
        binary.scatter_(0, top_idx, 1.0)
        final_mask = binary.reshape(config.SEQ_LEN, config.H)
        p_map = mp.detach().cpu()
        parameter = (param.detach().cpu().reshape(-1)
                     if param is not None else None)
    return final_mask, final_val_loss, z_norm, parameter, p_map


def train_fixed_mask(mask: torch.Tensor, pat: str, n_repeats: int,
                     final_steps: int, x_test: torch.Tensor,
                     y_test: torch.Tensor, seed_base: int) -> dict:
    """Standard protocol on a FIXED binary mask; stats on the test split."""
    device = x_test.device
    g = torch.Generator(device=device).manual_seed(seed_base)
    w1 = nn.Parameter(torch.randn(n_repeats, config.SEQ_LEN, config.H,
                                  generator=g, device=device) * 0.1)
    b1 = nn.Parameter(torch.zeros(n_repeats, config.H, device=device))
    w2 = nn.Parameter(torch.randn(n_repeats, config.H, 1,
                                  generator=g, device=device) * 0.1)
    b2 = nn.Parameter(torch.zeros(n_repeats, 1, device=device))
    opt = torch.optim.Adam([w1, b1, w2, b2], lr=config.LR)
    mask = mask.to(device)
    for step in range(final_steps):
        xb, yb = get_train_batch(pat, config.TRAIN_BATCH_SIZE,
                                 seed_base + step)
        xb, yb = xb.to(device), yb.to(device)
        pred = mlp_forward(xb, w1, b1, w2, b2, mask)
        loss = F.binary_cross_entropy_with_logits(
            pred, yb.unsqueeze(1).expand_as(pred))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred_t = mlp_forward(x_test, w1, b1, w2, b2, mask)   # (n_test, R)
        bce = F.binary_cross_entropy_with_logits(
            pred_t, y_test.unsqueeze(1).expand_as(pred_t),
            reduction="none").mean(dim=0)                    # (R,)
        acc = ((pred_t > 0).float() == y_test.unsqueeze(1)).float().mean(dim=0)
    return {"test_acc_mean": acc.mean().item(),
            "test_acc_std": acc.std(correction=0).item(),
            "test_bce": bce.mean().item(),
            "test_bce_std": bce.std(correction=0).item()}


def random_mask(seed: int, device) -> torch.Tensor:
    """One seeded Bernoulli(0.5) mask, shared across patterns."""
    g = torch.Generator().manual_seed(seed)
    m = (torch.rand(config.SEQ_LEN, config.H, generator=g) < 0.5).float()
    return m.to(device)


def window_matrix() -> torch.Tensor:
    """Return the five length-four contiguous-window templates."""
    windows = torch.zeros(config.N_WINDOWS, config.SEQ_LEN)
    for window in range(config.N_WINDOWS):
        windows[window, window:window + config.PATTERN_LEN] = 1.0
    return windows


def structure_metrics(mask: torch.Tensor) -> dict:
    """Measure permutation-invariant agreement with the target Toeplitz mask."""
    binary = mask.detach().cpu().float()
    gold = ideal_mask().float()
    aligned = align_map(binary, gold)
    intersection = (aligned * gold).sum()
    union = aligned.sum() + gold.sum() - intersection
    assignments = (binary.T @ window_matrix().T).argmax(dim=1)
    counts = torch.bincount(assignments, minlength=config.N_WINDOWS)
    return {
        "best_permutation_iou": float(intersection / union),
        "window_assignments": assignments.tolist(),
        "window_counts": counts.tolist(),
        "row_coverage": binary.sum(dim=1).to(torch.int64).tolist(),
    }


def load_mask_artifacts(path: Path) -> dict:
    """Load mode-indexed masks, migrating the former flat z-only format."""
    if not path.exists():
        return {}
    try:
        saved = torch.load(path, weights_only=True)
    except (OSError, RuntimeError):
        return {}
    if set(saved).issubset(config.PATTERNS):
        return {"z": saved}
    return saved


def save_mask_grid(records: dict, mode: str, output: Path) -> None:
    """Save a labelled 4x4 grid of final binary masks for one mode."""
    fig, axes = plt.subplots(4, 4, figsize=(10, 10), dpi=150)
    for pat, axis in zip(config.PATTERNS, axes.flat):
        record = records.get(pat)
        axis.set_xticks([])
        axis.set_yticks([])
        if record is None:
            axis.axis("off")
            continue
        matrix = record["binary_mask"].float()
        axis.imshow(matrix, cmap="gray_r", vmin=0, vmax=1, aspect="auto")
        iou = record["structure"]["best_permutation_iou"]
        acc = record["test_acc_mean"]
        axis.set_title(f"{pat}: acc {acc:.3f}, IoU {iou:.3f}", fontsize=8)
    fig.suptitle(f"Optimized top-{config.K_ACTIVE} masks: {mode}")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output)
    plt.close(fig)


def save_structure_plot(records: dict, mode: str, output: Path) -> None:
    """Save aggregate window, row-coverage, and IoU diagnostics for one mode."""
    if not records:
        return
    metrics = [record["structure"] for record in records.values()]
    window_counts = torch.tensor([m["window_counts"] for m in metrics]).sum(0)
    row_coverage = torch.tensor([m["row_coverage"] for m in metrics]).float().mean(0)
    ious = torch.tensor([m["best_permutation_iou"] for m in metrics])
    expected_windows = (len(metrics) * torch.tensor([2, 2, 2, 1, 1]))
    expected_rows = ideal_mask().sum(dim=1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), dpi=150)
    x = torch.arange(config.N_WINDOWS)
    axes[0].bar(x - 0.18, window_counts, width=0.36, label="optimized")
    axes[0].bar(x + 0.18, expected_windows, width=0.36, label="Toeplitz target")
    axes[0].set_xticks(x)
    axes[0].set_xlabel("window id")
    axes[0].set_ylabel("assigned hidden columns")
    axes[0].legend()
    axes[0].set_title("Window assignment distribution")

    rows = torch.arange(config.SEQ_LEN)
    axes[1].plot(rows, row_coverage, "o-", label="optimized mean")
    axes[1].plot(rows, expected_rows, "o--", label="Toeplitz target")
    axes[1].set_xticks(rows)
    axes[1].set_xlabel("input position")
    axes[1].set_ylabel("active hidden connections")
    axes[1].legend()
    axes[1].set_title("Row coverage")

    axes[2].hist(ious, bins=min(10, len(ious)), color="steelblue")
    axes[2].axvline(ious.mean(), color="crimson", linestyle="--",
                    label=f"mean {ious.mean():.3f}")
    axes[2].set_xlabel("best-permutation IoU vs Toeplitz target")
    axes[2].set_ylabel("patterns")
    axes[2].legend()
    axes[2].set_title("Per-pattern structural agreement")
    fig.suptitle(f"Toeplitz diagnostics: {mode}")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output)
    plt.close(fig)


def render_saved_artifacts(args) -> None:
    """Add structural metrics and figures to masks saved by an earlier run."""
    all_masks = load_mask_artifacts(args.mask_out)
    if not all_masks:
        raise FileNotFoundError(f"No saved masks found at {args.mask_out}")
    for mode, records in all_masks.items():
        for record in records.values():
            record["structure"] = structure_metrics(record["binary_mask"])
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        save_mask_grid(records, mode, args.plot_dir / f"zopt_masks_{mode}.png")
        save_structure_plot(records, mode,
                            args.plot_dir / f"zopt_toeplitz_{mode}.png")
        print(f"{LOG_TAG} rendered {mode}: "
              f"{args.plot_dir / f'zopt_masks_{mode}.png'}, "
              f"{args.plot_dir / f'zopt_toeplitz_{mode}.png'}", flush=True)

    if args.out_json.exists():
        out = json.loads(args.out_json.read_text())
        for mode, records in all_masks.items():
            for pat, record in records.items():
                if mode in out and pat in out[mode]:
                    out[mode][pat]["structure"] = record["structure"]
        args.out_json.write_text(json.dumps(out, indent=2))
        print(f"{LOG_TAG} updated metrics -> {args.out_json}", flush=True)
    torch.save(all_masks, args.mask_out)
    print(f"{LOG_TAG} updated masks -> {args.mask_out}", flush=True)


def main() -> None:
    args = build_parser().parse_args()
    if args.render_saved:
        render_saved_artifacts(args)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{LOG_TAG} mode={args.mode} ckpt={args.ckpt} "
          f"outer_steps={args.outer_steps} warmup={args.warmup_steps} "
          f"grad={args.grad_steps} z_lr={args.z_lr} seed={args.seed} "
          f"n_repeats={args.n_repeats} device={device}", flush=True)

    model = load_cvae(args.ckpt, device)
    c = model.condition(torch.zeros(1, 4, device=device))
    patterns = args.patterns if args.patterns else list(config.PATTERNS)

    results = {}
    masks_out = {}
    for pat in patterns:
        d = torch.load(config.val_path(pat), weights_only=False)
        x_all = d["x"].to(device)          # (2048, 8)
        y_all = d["y"].to(device)          # (2048,)
        x_val, y_val = x_all[:1024], y_all[:1024]
        x_test, y_test = x_all[1024:], y_all[1024:]

        final_mask, final_val_loss, z_norm, parameter, p_map = bilevel_optimize(
            args.mode, pat, model, c, x_val, y_val, args)

        stats = train_fixed_mask(final_mask, pat, args.n_repeats,
                                 args.final_steps, x_test, y_test,
                                 10_000 * int(pat, 2))
        stats["final_val_loss"] = final_val_loss
        stats["z_norm"] = z_norm
        stats["structure"] = structure_metrics(final_mask)
        results[pat] = stats
        masks_out[pat] = {
            "parameter": parameter,
            "parameter_name": {"z": "z", "free": "logits", "z0": None}[args.mode],
            "z": parameter if args.mode == "z" else None,
            "logits": parameter if args.mode == "free" else None,
            "p_map": p_map,
            "binary_mask": final_mask.cpu(),
            "test_acc_mean": stats["test_acc_mean"],
            "test_acc_std": stats["test_acc_std"],
            "test_bce": stats["test_bce"],
            "structure": stats["structure"],
        }
        print(f"{LOG_TAG} pat={pat} | mode={args.mode} | "
              f"final_val={final_val_loss:.4f} || "
              f"test_acc={stats['test_acc_mean']:.4f}"
              f"±{stats['test_acc_std']:.4f} | "
              f"test_bce={stats['test_bce']:.4f}", flush=True)

    # Fixed reference masks, identical protocol, same test split.
    refs = {}
    for name, mk in [("random", random_mask(12345 + args.seed, device)),
                     ("ideal", ideal_mask().float().to(device))]:
        accs, bces = [], []
        for pat in patterns:
            d = torch.load(config.val_path(pat), weights_only=False)
            x_test = d["x"][1024:].to(device)
            y_test = d["y"][1024:].to(device)
            st = train_fixed_mask(mk, pat, args.n_repeats, args.final_steps,
                                  x_test, y_test, 10_000 * int(pat, 2))
            accs.append(st["test_acc_mean"])
            bces.append(st["test_bce"])
        refs[name] = {"test_acc_mean": float(torch.tensor(accs).mean()),
                      "test_acc_std": float(torch.tensor(accs).std(
                          correction=0)),
                      "test_bce": float(torch.tensor(bces).mean()),
                      "structure": structure_metrics(mk)}
        print(f"{LOG_TAG} ref={name:6s} test_acc="
              f"{refs[name]['test_acc_mean']:.4f}"
              f"±{refs[name]['test_acc_std']:.4f} "
              f"test_bce={refs[name]['test_bce']:.4f}", flush=True)

    # Merge into the machine-readable summary (safe for sequential runs).
    out = {}
    if args.out_json.exists():
        try:
            out = json.loads(args.out_json.read_text())
        except (json.JSONDecodeError, OSError):
            out = {}
    out[args.mode] = results
    out["random"] = refs["random"]
    out["ideal"] = refs["ideal"]
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))
    print(f"{LOG_TAG} saved -> {args.out_json}", flush=True)

    all_masks = load_mask_artifacts(args.mask_out)
    all_masks[args.mode] = masks_out
    args.mask_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_masks, args.mask_out)
    print(f"{LOG_TAG} saved masks -> {args.mask_out}", flush=True)

    args.plot_dir.mkdir(parents=True, exist_ok=True)
    mask_plot = args.plot_dir / f"zopt_masks_{args.mode}.png"
    structure_plot = args.plot_dir / f"zopt_toeplitz_{args.mode}.png"
    save_mask_grid(masks_out, args.mode, mask_plot)
    save_structure_plot(masks_out, args.mode, structure_plot)
    print(f"{LOG_TAG} saved plots -> {mask_plot}, {structure_plot}", flush=True)

    # Summary table over patterns for all modes present in the file.
    print(f"{LOG_TAG} === SUMMARY: mean test acc over {len(patterns)} "
          f"patterns ===", flush=True)
    hdr = f"{LOG_TAG} {'pat':<6} |"
    for mk in ["z", "free", "z0", "random", "ideal"]:
        hdr += f" {mk:>7s} |"
    print(hdr, flush=True)
    for pat in patterns:
        row = f"{LOG_TAG} {pat:<6} |"
        for mk in ["z", "free", "z0"]:
            if mk in out and pat in out[mk]:
                row += f" {out[mk][pat]['test_acc_mean']:7.4f} |"
            else:
                row += f" {'-':>7s} |"
        for mk in ["random", "ideal"]:
            row += f" {'-':>7s} |"  # references are pattern-independent
        print(row, flush=True)
    for mk in ["z", "free", "z0", "random", "ideal"]:
        if mk not in out:
            continue
        if mk in ("random", "ideal"):
            v = out[mk]
            print(f"{LOG_TAG} {mk:6s}: test_acc = {v['test_acc_mean']:.4f} "
                  f"± {v['test_acc_std']:.4f} (test_bce={v['test_bce']:.4f})",
                  flush=True)
        else:
            accs = torch.tensor([out[mk][p]["test_acc_mean"]
                                 for p in patterns])
            ious = torch.tensor([out[mk][p].get("structure", {}).get(
                "best_permutation_iou", float("nan")) for p in patterns])
            print(f"{LOG_TAG} {mk:6s}: test_acc = {accs.mean():.4f} "
                  f"± {accs.std(correction=0):.4f}; toeplitz_iou = "
                  f"{ious.nanmean():.4f}", flush=True)
    if "z" in out:
        accs = {p: out["z"][p]["test_acc_mean"] for p in patterns}
        best = sorted(accs, key=accs.get, reverse=True)[:2]
        worst = sorted(accs, key=accs.get)[:2]
        print(f"{LOG_TAG} mode z best patterns: "
              + ", ".join(f"{p} ({accs[p]:.4f})" for p in best), flush=True)
        print(f"{LOG_TAG} mode z worst patterns: "
              + ", ".join(f"{p} ({accs[p]:.4f})" for p in worst), flush=True)


if __name__ == "__main__":
    main()
