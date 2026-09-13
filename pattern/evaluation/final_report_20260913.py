"""Build compact figures and controls for the 2026-09-13 session report."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pattern.data.generate import ideal_mask as ideal_mask_k4
from pattern.evaluation.decoder_agreement import align_columns
from pattern.fixed_k5_agreement.common import ideal_mask as ideal_mask_k5
from pattern.fixed_k5_agreement.config import Config as K5Config
from pattern.length_interp.bank import random_exact_k_masks


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "mds/assets/2026-09-13"
DATA = ROOT / "mds/data/2026-09-13"


REPORT_FILES = {
    "pattern/outputs/decoder_agreement/hparam_tuning_20260912/vae_e240/"
    "agreement_sweep/agreement_hparam_sweep.png":
        ASSETS / "agreement_hparam_sweep.png",
    "pattern/outputs/decoder_agreement/multiseed32_tuned_20260912/"
    "pair_186_187/mask_examples.png":
        ASSETS / "agreement_mask_examples.png",
    "pattern/outputs/z_star_noise/multiseed8_20260912/radius_reachability.png":
        ASSETS / "zstar_radius_reachability.png",
    "pattern/outputs/z_star_noise/multiseed8_20260912/noise_recovery.png":
        ASSETS / "zstar_noise_recovery.png",
    "pattern/outputs/z_star_reachable/r4_multiseed8_20260912/"
    "matched_reachability.png":
        ASSETS / "zstar_matched_reachability.png",
    "pattern/outputs/z_star_reachable/r4_multiseed8_20260912/"
    "matched_mask_example.png":
        ASSETS / "zstar_matched_mask_example.png",
    "pattern/outputs/decoder_agreement/multiseed64_20260912/summary.json":
        DATA / "agreement_baseline_64pairs.json",
    "pattern/outputs/decoder_agreement/multiseed32_tuned_20260912/summary.json":
        DATA / "agreement_confirmatory_32pairs.json",
    "pattern/outputs/z_star_noise/multiseed8_20260912/summary.json":
        DATA / "zstar_noise_summary.json",
    "pattern/outputs/z_star_noise/multiseed8_20260912/radius_sweep_summary.json":
        DATA / "zstar_radius_sweep_summary.json",
    "pattern/outputs/z_star_reachable/r4_multiseed8_20260912/refined_summary.json":
        DATA / "zstar_matched_r4_summary.json",
    "pattern/outputs/z_star_reachable/r4_hard_sampling_robust_pairs32_20260913/"
    "summary.json":
        DATA / "sampling_pattern8_32pairs_summary.json",
    "pattern/outputs/z_star_reachable/r4_hard_sampling_robust_pairs32_20260913/"
    "paired_analysis.json":
        DATA / "sampling_pattern8_32pairs_paired.json",
    "pattern/outputs/fixed_k5_agreement/sampling_32pairs_20260913/"
    "robust_sampling_r4/summary.json":
        DATA / "sampling_pattern32_k5_32pairs_summary.json",
    "pattern/outputs/fixed_k5_agreement/sampling_32pairs_20260913/"
    "robust_sampling_r4/paired_analysis.json":
        DATA / "sampling_pattern32_k5_32pairs_paired.json",
}


def _load(path: str | Path) -> dict:
    return json.loads((ROOT / path).read_text())


def _mean_ci(row: dict) -> tuple[float, float, float]:
    low, high = row.get("ci95_t", row.get("ci95"))
    return float(row["mean"]), float(low), float(high)


def copy_report_files() -> None:
    """Keep the tracked report self-contained without committing raw checkpoints."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    for source_name, destination in REPORT_FILES.items():
        source = ROOT / source_name
        if source.exists():
            shutil.copy2(source, destination)
        elif not destination.exists():
            raise FileNotFoundError(f"missing report source and snapshot: {source}")


def agreement_summary() -> None:
    studies = [
        ("64 pairs\nbaseline", _load(
            "mds/data/2026-09-13/agreement_baseline_64pairs.json")),
        ("32 pairs\nconfirmatory", _load(
            "mds/data/2026-09-13/agreement_confirmatory_32pairs.json")),
    ]
    panels = [
        ("hard_pair_iou", "Hard decoder IoU", False),
        ("hard_exact_fraction", "Exact hard agreement", False),
        ("gold_iou", "Post-hoc Gold IoU", False),
        ("downstream_accuracy", "Fresh-MLP accuracy", False),
        ("soft_mse", "Soft disagreement MSE", True),
    ]
    methods = [("initial", "Prior", "#7f8c8d"),
               ("random_search", "Random search", "#d18f32"),
               ("optimized", "Adam agreement", "#258b78")]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2), layout="constrained")
    for ax, (metric, title, log_scale) in zip(axes.flat, panels):
        width = .23
        for mi, (key, label, color) in enumerate(methods):
            means, lows, highs = [], [], []
            for _, study in studies:
                row = study["across_pair_means"][metric].get(key)
                if row is None:
                    means.append(np.nan); lows.append(np.nan); highs.append(np.nan)
                else:
                    mean, low, high = _mean_ci(row)
                    means.append(mean); lows.append(mean - low); highs.append(high - mean)
            x = np.arange(len(studies)) + (mi - 1) * width
            ax.bar(x, means, width, color=color, label=label)
            ax.errorbar(x, means, yerr=np.array([lows, highs]), fmt="none",
                        ecolor="black", capsize=3, lw=1)
        ax.set_xticks(np.arange(len(studies)), [name for name, _ in studies])
        ax.set_title(title)
        ax.grid(axis="y", alpha=.25)
        if log_scale:
            ax.set_yscale("log")
    axes[1, 2].axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[1, 2].legend(handles, labels, loc="center", frameon=False)
    fig.suptitle("Decoder agreement across independent VAE initializations")
    fig.savefig(ASSETS / "final_agreement_summary.png", dpi=180)
    fig.savefig(ASSETS / "final_agreement_summary.pdf")
    plt.close(fig)


def optimizer_comparison() -> None:
    labels = ["Soft Adam\n8 VAE", "Hard STE\n2 VAE", "Sampling\n8 VAE",
              "Sampling\n32 pairs"]
    prior_iou = [.718, .675, .675, .6716]
    z_exact = [0., .105, .106, .1168]
    z_iou = [.798, .915, .871, .8701]
    z_l2 = [2.307, 1.701, 1.700, 1.6821]
    values = [prior_iou, z_exact, z_iou, z_l2]
    titles = ["Final Gold IoU from prior", "Final exact fraction from z*",
              "Final Gold IoU from z*", "Final L2 distance from z*"]
    colors = ["#4f81bd", "#9b59b6", "#d17b32", "#258b78"]
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.3), layout="constrained")
    for ax, vals, title in zip(axes, values, titles):
        x = np.arange(len(vals))
        ax.bar(x, vals, color=colors)
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=.25)
        for xi, value in zip(x, vals):
            ax.text(xi, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    axes[1].set_ylim(0, .14)
    axes[0].set_ylim(.60, .76)
    axes[2].set_ylim(.75, .96)
    axes[3].set_ylim(0, 2.6)
    fig.suptitle("Task-aware latent optimization, length 8 / pattern 4")
    fig.savefig(ASSETS / "final_optimizer_comparison.png", dpi=180)
    fig.savefig(ASSETS / "final_optimizer_comparison.pdf")
    plt.close(fig)


def sampling_scale_summary() -> None:
    labels = ["L=8, k=4\nprior", "L=8, k=4\nz*",
              "L=32, k=5\nprior", "L=32, k=5\noracle-best"]
    initial_iou = [.6353, 1., .4535, .5189]
    final_iou = [.6716, .8701, .4818, .5212]
    initial_acc = [.91372, .92995, .67827, .68716]
    final_acc = [.92110, .92926, .68459, .68902]
    initial_bce = [.23918, .20451, .59703, .58757]
    final_bce = [.22365, .20610, .59031, .58552]
    series = [(initial_iou, final_iou, "Gold IoU"),
              (initial_acc, final_acc, "Independent-test accuracy"),
              (initial_bce, final_bce, "Independent-test BCE")]
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.6), layout="constrained")
    x = np.arange(len(labels)); width = .34
    for ax, (before, after, title) in zip(axes, series):
        ax.bar(x - width / 2, before, width, label="Initial", color="#7f8c8d")
        ax.bar(x + width / 2, after, width, label="Sampling final", color="#258b78")
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Robust hard-mask sampling on 32 independent VAE pairs")
    fig.savefig(ASSETS / "final_sampling_scale_summary.png", dpi=180)
    fig.savefig(ASSETS / "final_sampling_scale_summary.pdf")
    plt.close(fig)


def _collect_masks(root: Path) -> tuple[torch.Tensor, torch.Tensor, int]:
    protocol = json.loads((root / "protocol.json").read_text())
    n = int(protocol["settings"]["starts_per_group"])
    initial, final = [], []
    max_initial_task_difference = 0
    for seed in protocol["model_seeds"]:
        reference = None
        for pattern in protocol["patterns"]:
            task = torch.load(root / f"seed_{seed}/task_{pattern}.pt",
                              map_location="cpu", weights_only=True)
            current = task["initial_masks"][:n].float()
            if reference is None:
                reference = current
                initial.append(current)
            else:
                max_initial_task_difference = max(
                    max_initial_task_difference,
                    int((current != reference).sum().item()),
                )
            final.append(task["final_masks"][:n].float())
    return torch.cat(initial), torch.cat(final), max_initial_task_difference


def _contrast(masks: torch.Tensor, target: torch.Tensor, *, aligned: bool) -> float:
    if aligned:
        masks = align_columns(target[None].expand(len(masks), -1, -1), masks)
    mean = masks.float().mean(0)
    gold = target.bool()
    return float(mean[gold].mean() - mean[~gold].mean())


def _locality(masks: torch.Tensor, k: int) -> dict[str, float]:
    masks = masks.float()
    counts = masks.sum(1)
    best = masks.unfold(1, k, 1).sum(-1).max(1).values
    adjacent = ((masks[:, 1:] * masks[:, :-1]).sum((1, 2))
                / masks.sum((1, 2)).clamp_min(1))
    return {
        "best_k_window_coverage": float((best / k).mean()),
        "exact_contiguous_k_column_fraction": float(
            ((counts == k) & (best == k)).float().mean()),
        "adjacent_pairs_per_active": float(adjacent.mean()),
    }


def heatmap_control() -> None:
    cases = [
        ("8×8", ROOT / "pattern/outputs/z_star_reachable/"
         "r4_hard_sampling_robust_pairs32_20260913", ideal_mask_k4().float(), 4, 32, 12345),
        ("32×32", ROOT / "pattern/outputs/fixed_k5_agreement/"
         "sampling_32pairs_20260913/robust_sampling_r4",
         ideal_mask_k5(K5Config()).float(), 5, 160, 54321),
    ]
    payload_path = ASSETS / "final_heatmap_alignment_control.json"
    if all((root / "protocol.json").exists() for _, root, *_ in cases):
        payload = {}
        for label, root, target, k, active, seed in cases:
            initial, final, difference = _collect_masks(root)
            random = random_exact_k_masks(
                len(initial), seq_len=target.size(0), hidden=target.size(1),
                k_active=active, seed=seed)
            payload[label] = {
                "unique_initial_masks": len(initial),
                "task_specific_final_masks": len(final),
                "max_initial_difference_across_tasks": difference,
                "contrast": {
                    "random_raw": _contrast(random, target, aligned=False),
                    "random_hungarian": _contrast(random, target, aligned=True),
                    "vae_initial_raw": _contrast(initial, target, aligned=False),
                    "vae_initial_hungarian": _contrast(initial, target, aligned=True),
                },
                "alignment_free_locality": {
                    "random": _locality(random, k),
                    "vae_initial": _locality(initial, k),
                    "vae_final": _locality(final, k),
                },
            }
        payload_path.write_text(json.dumps(payload, indent=2) + "\n")
    else:
        payload = json.loads(payload_path.read_text())

    contrast_rows, locality_rows = [], []
    for label, *_ in cases:
        row = payload[label]
        contrast_rows.append([row["contrast"][name] for name in
                              ("random_raw", "random_hungarian",
                               "vae_initial_raw", "vae_initial_hungarian")])
        locality_rows.append([row["alignment_free_locality"][name]["best_k_window_coverage"]
                              for name in ("random", "vae_initial", "vae_final")])

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), layout="constrained")
    x = np.arange(2)
    names = ["Random raw", "Random +\nHungarian", "VAE raw", "VAE +\nHungarian"]
    colors = ["#9aa0a6", "#d18f32", "#6aaed6", "#258b78"]
    for i, name in enumerate(names):
        axes[0].bar(x + (i - 1.5) * .19, [row[i] for row in contrast_rows],
                    width=.19, label=name, color=colors[i])
    axes[0].set_xticks(x, [case[0] for case in cases])
    axes[0].set_title("Gold contrast: raw vs aligned")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=.25)
    local_names = ["Random exact-K", "VAE initial", "VAE final"]
    for i, name in enumerate(local_names):
        axes[1].bar(x + (i - 1) * .25, [row[i] for row in locality_rows],
                    width=.25, label=name, color=colors[i + 1])
    axes[1].set_xticks(x, [case[0] for case in cases])
    axes[1].set_title("Best k-window coverage (no alignment)")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=.25)
    fig.suptitle("Heatmap alignment control")
    fig.savefig(ASSETS / "final_heatmap_alignment_control.png", dpi=180)
    fig.savefig(ASSETS / "final_heatmap_alignment_control.pdf")
    plt.close(fig)


def main() -> None:
    torch.set_num_threads(1)
    ASSETS.mkdir(parents=True, exist_ok=True)
    copy_report_files()
    agreement_summary()
    optimizer_comparison()
    sampling_scale_summary()
    heatmap_control()
    print(ASSETS)


if __name__ == "__main__":
    main()
