"""Measure the first-layer mask after weighting each unit by its output weight."""

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
from evaluation.optimize_z import structure_metrics  # noqa: E402
from models.mlp import get_train_batch  # noqa: E402


MASKS_DEFAULT = config.EVAL_DIR / "zopt_masks.pt"
OUT_JSON_DEFAULT = config.EVAL_DIR / "effective_mask_analysis.json"
OUT_TENSOR_DEFAULT = config.EVAL_DIR / "effective_mask_analysis.pt"
PLOT_DIR_DEFAULT = config.PLOT_DIR / "z_analysis"
MODES = ("z", "z0", "free")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze |W1 * mask| * |W2| after fixed-mask training.")
    parser.add_argument("--masks", type=Path, default=MASKS_DEFAULT)
    parser.add_argument("--out_json", type=Path, default=OUT_JSON_DEFAULT)
    parser.add_argument("--out_tensor", type=Path, default=OUT_TENSOR_DEFAULT)
    parser.add_argument("--plot_dir", type=Path, default=PLOT_DIR_DEFAULT)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--n_repeats", type=int, default=8)
    parser.add_argument("--final_steps", type=int, default=config.TRAIN_STEPS)
    return parser.parse_args()


def masked_logits(x: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor,
                  w2: torch.Tensor, b2: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
    """Compute batched MLP logits for a fixed first-layer mask."""
    hidden = F.relu(torch.einsum("bl,rlh->brh", x, w1 * mask) + b1)
    return torch.einsum("brh,rh->br", hidden, w2.squeeze(-1)) + b2.squeeze(-1)


def fit_models(mask: torch.Tensor, pat: str, n_repeats: int,
               final_steps: int, device: torch.device) -> tuple:
    """Train the same fixed-mask MLP ensemble used by z optimization."""
    seed_base = 10_000 * int(pat, 2)
    generator = torch.Generator(device=device).manual_seed(seed_base)
    w1 = nn.Parameter(torch.randn(n_repeats, config.SEQ_LEN, config.H,
                                  generator=generator, device=device) * 0.1)
    b1 = nn.Parameter(torch.zeros(n_repeats, config.H, device=device))
    w2 = nn.Parameter(torch.randn(n_repeats, config.H, 1,
                                  generator=generator, device=device) * 0.1)
    b2 = nn.Parameter(torch.zeros(n_repeats, 1, device=device))
    optimizer = torch.optim.Adam([w1, b1, w2, b2], lr=config.LR)
    mask = mask.to(device)
    for step in range(final_steps):
        x, y = get_train_batch(pat, config.TRAIN_BATCH_SIZE, seed_base + step)
        logits = masked_logits(x.to(device), w1, b1, w2, b2, mask)
        loss = F.binary_cross_entropy_with_logits(
            logits, y.to(device).unsqueeze(1).expand_as(logits))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return w1.detach(), b1.detach(), w2.detach(), b2.detach()


def topk_mask(values: torch.Tensor) -> torch.Tensor:
    """Binarize one importance map to the experiment's top-K support."""
    flat = values.reshape(-1)
    indices = flat.topk(config.K_ACTIVE).indices
    binary = torch.zeros_like(flat)
    binary.scatter_(0, indices, 1.0)
    return binary.reshape_as(values)


def weighted_toeplitz_metrics(values: torch.Tensor) -> dict:
    """Measure aligned Toeplitz agreement of a nonnegative importance map."""
    importance = values.detach().cpu().float()
    gold = ideal_mask().float()
    normalized = importance * (gold.sum() / importance.sum().clamp_min(1e-12))
    aligned = align_map(normalized, gold)
    overlap = (aligned * gold).sum()
    total = aligned.sum()
    window_mass = aligned.T @ torch.stack([
        torch.roll(torch.tensor([1., 1., 1., 1., 0., 0., 0., 0.]), shift)
        for shift in range(config.N_WINDOWS)
    ]).T
    return {
        "toeplitz_mass": float(overlap / total),
        "weighted_iou": float(overlap / (total + gold.sum() - overlap)),
        "best_window_mass": float(window_mass.max(dim=1).values.sum() / total),
    }


def evaluate_models(mask: torch.Tensor, params: tuple, pat: str,
                    device: torch.device) -> dict:
    """Evaluate a trained ensemble on the held-out half of its validation set."""
    w1, b1, w2, b2 = params
    data = torch.load(config.val_path(pat), weights_only=False)
    x = data["x"][1024:].to(device)
    y = data["y"][1024:].to(device)
    with torch.no_grad():
        logits = masked_logits(x, w1, b1, w2, b2, mask.to(device))
        accuracy = ((logits > 0) == y.unsqueeze(1)).float().mean(dim=0)
    return {"test_acc_mean": accuracy.mean().item(),
            "test_acc_std": accuracy.std(correction=0).item()}


def analyze_record(record: dict, pat: str, args: argparse.Namespace,
                   device: torch.device) -> tuple[dict, dict]:
    """Train one ensemble and summarize its support and effective importance."""
    mask = record["binary_mask"].float()
    params = fit_models(mask, pat, args.n_repeats, args.final_steps, device)
    w1, _, w2, _ = params
    output_weights = w2.squeeze(-1).abs()
    effective = (w1 * mask.to(device)).abs() * output_weights.unsqueeze(1)
    mean_effective = effective.mean(dim=0).cpu()
    effective_mask = topk_mask(mean_effective)
    metrics = {
        "reproduced_eval": evaluate_models(mask, params, pat, device),
        "support_structure": structure_metrics(mask),
        "support_weighted_structure": weighted_toeplitz_metrics(mask),
        "effective_weighted_structure": weighted_toeplitz_metrics(mean_effective),
        "mean_output_weight": output_weights.mean(dim=0).cpu().tolist(),
    }
    tensors = {"mean_effective": mean_effective,
               "effective_binary_mask": effective_mask,
               "mean_output_weight": output_weights.mean(dim=0).cpu()}
    return metrics, tensors


def save_mode_grid(records: dict, mode: str, output: Path) -> None:
    """Plot effective importance maps and their Toeplitz mass."""
    figure, axes = plt.subplots(4, 4, figsize=(10, 10), dpi=150)
    for pat, axis in zip(config.PATTERNS, axes.flat):
        entry = records[pat]
        support_iou = entry["metrics"]["support_structure"]["best_permutation_iou"]
        effective_mass = entry["metrics"]["effective_weighted_structure"]["toeplitz_mass"]
        axis.imshow(entry["tensors"]["mean_effective"], cmap="viridis", aspect="auto")
        axis.set_title(f"{pat}: support {support_iou:.3f}, mass {effective_mass:.3f}",
                       fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(f"Effective importance maps: {mode}")
    figure.tight_layout(rect=[0, 0, 1, 0.97])
    figure.savefig(output)
    plt.close(figure)


def save_summary_plot(records_by_mode: dict, output: Path) -> None:
    """Plot Toeplitz mass before and after output-layer weighting."""
    modes = list(records_by_mode)
    support = []
    effective = []
    deltas = []
    for mode in modes:
        records = records_by_mode[mode].values()
        before = torch.tensor([r["metrics"]["support_weighted_structure"]
                               ["toeplitz_mass"] for r in records])
        after = torch.tensor([r["metrics"]["effective_weighted_structure"]
                              ["toeplitz_mass"] for r in records])
        support.append(before.mean().item())
        effective.append(after.mean().item())
        deltas.append((after - before).tolist())

    figure, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150)
    x = torch.arange(len(modes))
    axes[0].bar(x - 0.18, support, width=0.36, label="binary support")
    axes[0].bar(x + 0.18, effective, width=0.36, label="|W1·mask|·|W2|")
    axes[0].set_xticks(x, modes)
    axes[0].set_ylabel("mean Toeplitz mass after column alignment")
    axes[0].set_title("Does the output layer remove off-structure support?")
    axes[0].legend()

    axes[1].axhline(0, color="black", linewidth=0.8)
    for index, (mode, values) in enumerate(zip(modes, deltas)):
        axes[1].scatter(torch.full((len(values),), index), values, label=mode)
    axes[1].set_xticks(x, modes)
    axes[1].set_ylabel("effective Toeplitz mass − support mass")
    axes[1].set_title("Per-pattern structural change")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def json_records(records_by_mode: dict) -> dict:
    """Discard tensors from analysis records before JSON serialization."""
    return {mode: {pat: entry["metrics"] for pat, entry in records.items()}
            for mode, records in records_by_mode.items()}


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(args.masks, weights_only=True)
    if set(saved).issubset(config.PATTERNS):
        saved = {"z": saved}
    missing = [mode for mode in args.modes if mode not in saved]
    if missing:
        raise ValueError(f"Missing modes in {args.masks}: {', '.join(missing)}")

    records_by_mode = {}
    for mode in args.modes:
        records = {}
        for pat in config.PATTERNS:
            metrics, tensors = analyze_record(saved[mode][pat], pat, args, device)
            records[pat] = {"metrics": metrics, "tensors": tensors}
            before = metrics["support_structure"]["best_permutation_iou"]
            after = metrics["effective_weighted_structure"]["toeplitz_mass"]
            accuracy = metrics["reproduced_eval"]["test_acc_mean"]
            print(f"[effective] {mode} {pat}: acc={accuracy:.4f} "
                  f"support_IoU={before:.3f} effective_mass={after:.3f}",
                  flush=True)
        records_by_mode[mode] = records

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_tensor.parent.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(json_records(records_by_mode), indent=2))
    torch.save(records_by_mode, args.out_tensor)
    for mode, records in records_by_mode.items():
        save_mode_grid(records, mode, args.plot_dir / f"effective_masks_{mode}.png")
    save_summary_plot(records_by_mode, args.plot_dir / "effective_mask_summary.png")
    print(f"[effective] saved -> {args.out_json}", flush=True)
    print(f"[effective] saved tensors -> {args.out_tensor}", flush=True)


if __name__ == "__main__":
    main()
