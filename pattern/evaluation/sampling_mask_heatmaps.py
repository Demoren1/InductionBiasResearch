"""Aggregate aligned binary masks into heatmaps for both robust-sampling studies."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from pattern.evaluation.decoder_agreement import align_columns
from pattern.evaluation.z_star_noise import write_json
from pattern.fixed_k5_agreement.common import ideal_mask as ideal_mask_k5
from pattern.fixed_k5_agreement.config import Config as K5Config
from pattern.data.generate import ideal_mask as ideal_mask_k4


def _collect(root: Path, target: torch.Tensor) -> tuple[dict, dict]:
    protocol = json.loads((root / "protocol.json").read_text())
    n = protocol["settings"]["starts_per_group"]
    seeds = protocol.get("model_seeds", protocol["settings"].get("model_seeds"))
    patterns = protocol["patterns"]
    oracle_group = "oracle_z_star" if "oracle_z_star" in protocol.get("groups", ["oracle_z_star"]) else "oracle_best_r4"
    sums = {group: {stage: torch.zeros_like(target, dtype=torch.float64)
                    for stage in ("initial", "final")}
            for group in ("prior", oracle_group)}
    counts = {group: {stage: 0 for stage in ("initial", "final")}
              for group in sums}
    inside = {group: {stage: 0.0 for stage in ("initial", "final")}
              for group in sums}
    k = float(target.sum())
    target_f = target.float()
    for seed in seeds:
        oracle = torch.load(root / f"seed_{seed}/oracle_same_start.pt",
                            map_location="cpu", weights_only=True)
        eligible = oracle["best_soft"]["iou"][:n] == 1
        for pattern in patterns:
            task = torch.load(root / f"seed_{seed}/task_{pattern}.pt",
                              map_location="cpu", weights_only=True)
            oracle_keep = eligible if oracle_group == "oracle_z_star" else torch.ones(n, dtype=torch.bool)
            for group, group_slice, keep in (
                ("prior", slice(0, n), torch.ones(n, dtype=torch.bool)),
                (oracle_group, slice(n, 2 * n), oracle_keep),
            ):
                for stage in ("initial", "final"):
                    masks = task[f"{stage}_masks"][group_slice][keep].float()
                    targets = target_f[None].expand(len(masks), -1, -1)
                    aligned = align_columns(targets, masks)
                    sums[group][stage] += aligned.double().sum(0)
                    counts[group][stage] += len(aligned)
                    inside[group][stage] += float((aligned * targets).sum())
    heatmaps, stats = {}, {}
    gold = target_f.bool()
    outside = ~gold
    for group in sums:
        heatmaps[group], stats[group] = {}, {}
        for stage in sums[group]:
            mean = (sums[group][stage] / counts[group][stage]).float()
            probability = mean.clamp(1e-7, 1 - 1e-7)
            entropy = -(probability * torch.log2(probability)
                        + (1 - probability) * torch.log2(1 - probability)).mean()
            heatmaps[group][stage] = mean
            stats[group][stage] = {
                "mask_count": counts[group][stage],
                "mean_gold_support_recall": inside[group][stage] / (counts[group][stage] * k),
                "consensus_mae_from_gold": float((mean - target_f).abs().mean()),
                "mean_binary_entropy_bits": float(entropy),
                "mean_occupancy_on_gold": float(mean[gold].mean()),
                "mean_occupancy_off_gold": float(mean[outside].mean()),
                "gold_off_contrast": float(mean[gold].mean() - mean[outside].mean()),
                "row_occupancy": mean.mean(1).tolist(),
            }
        stats[group]["change"] = {
            "consensus_l1": float((heatmaps[group]["final"] - heatmaps[group]["initial"]).abs().mean()),
            "gold_recall_delta": (stats[group]["final"]["mean_gold_support_recall"]
                                  - stats[group]["initial"]["mean_gold_support_recall"]),
            "contrast_delta": (stats[group]["final"]["gold_off_contrast"]
                               - stats[group]["initial"]["gold_off_contrast"]),
        }
    return heatmaps, stats


def _plot(heatmaps: dict, target: torch.Tensor, title: str, destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.2), layout="constrained")
    oracle_group = next(group for group in heatmaps if group != "prior")
    oracle_label = "Oracle z*" if oracle_group == "oracle_z_star" else "Oracle-best R=4"
    for row, (group, label) in enumerate((("prior", "Prior"), (oracle_group, oracle_label))):
        initial, final = heatmaps[group]["initial"], heatmaps[group]["final"]
        panels = (("Initial mean", initial, "viridis", 0., 1.),
                  ("Sampling final mean", final, "viridis", 0., 1.),
                  ("Final − initial", final - initial, "RdBu_r", -.35, .35),
                  ("Ideal", target.float(), "viridis", 0., 1.))
        for col, (name, value, cmap, low, high) in enumerate(panels):
            image = axes[row, col].imshow(value, cmap=cmap, vmin=low, vmax=high,
                                          interpolation="nearest", aspect="auto")
            axes[row, col].set_title(f"{label}: {name}", fontsize=9)
            axes[row, col].set_xlabel("Aligned hidden column")
            axes[row, col].set_ylabel("Input position")
            fig.colorbar(image, ax=axes[row, col], fraction=.046, pad=.025)
    fig.suptitle(title)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix(".png"), dpi=180)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)


def _table(label: str, stats: dict) -> list[str]:
    lines = [f"### {label}", "",
             "| Старт | Стадия | Масок | Recall Gold support | MAE consensus→Gold | Entropy, bit | On-Gold | Off-Gold | Contrast |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    oracle_group = next(group for group in stats if group != "prior")
    oracle_label = "Oracle z*" if oracle_group == "oracle_z_star" else "Oracle-best R=4"
    for group, group_label in (("prior", "Prior"), (oracle_group, oracle_label)):
        for stage in ("initial", "final"):
            row = stats[group][stage]
            lines.append(f"| {group_label} | {stage} | {row['mask_count']} | "
                         f"{row['mean_gold_support_recall']:.4f} | {row['consensus_mae_from_gold']:.4f} | "
                         f"{row['mean_binary_entropy_bits']:.4f} | {row['mean_occupancy_on_gold']:.4f} | "
                         f"{row['mean_occupancy_off_gold']:.4f} | {row['gold_off_contrast']:.4f} |")
    lines += ["", "Изменение средней маски после sampling:", ""]
    for group, group_label in (("prior", "Prior"), (oracle_group, oracle_label)):
        row = stats[group]["change"]
        lines.append(f"- {group_label}: L1={row['consensus_l1']:.4f}, "
                     f"Δ recall Gold={row['gold_recall_delta']:+.4f}, "
                     f"Δ contrast={row['contrast_delta']:+.4f}.")
    return lines


def main() -> None:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern8", type=Path, required=True)
    parser.add_argument("--pattern32", type=Path, required=True)
    parser.add_argument("--md", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    target8, target32 = ideal_mask_k4().float(), ideal_mask_k5(K5Config()).float()
    heat8, stats8 = _collect(args.pattern8.resolve(), target8)
    heat32, stats32 = _collect(args.pattern32.resolve(), target32)
    asset_root = args.assets.resolve()
    _plot(heat8, target8, "Aligned hard-mask heatmaps: length 8, pattern 4",
          asset_root / "pattern8_robust_mask_heatmaps")
    _plot(heat32, target32, "Aligned hard-mask heatmaps: length 32, pattern 5",
          asset_root / "pattern32_k5_robust_mask_heatmaps")
    payload = {"pattern8": stats8, "pattern32_k5": stats32,
               "alignment": "Hungarian hidden-column matching to analytic ideal before averaging"}
    write_json(asset_root / "mask_heatmap_stats.json", payload)

    md = args.md.resolve()
    relative = Path("assets") / asset_root.name
    lines = ["# Хитмапы масок robust latent sampling", "", "Дата: 2026-09-13.", "",
             "Все маски бинарные и имеют точную заданную мощность (32 для `8×8`, 160 для `32×32`). "
             "До усреднения скрытые колонки каждой маски сопоставлены с ideal алгоритмом Hungarian; "
             "без этого эквивалентные перестановки колонок дали бы ложное размытие.", "",
             "## Length 8, pattern length 4", "",
             f"![Heatmaps length 8]({relative / 'pattern8_robust_mask_heatmaps.png'})", ""]
    lines += _table("Численные характеристики 8×8", stats8)
    lines += ["", "## Length 32, pattern length 5", "",
              f"![Heatmaps length 32]({relative / 'pattern32_k5_robust_mask_heatmaps.png'})", ""]
    lines += _table("Численные характеристики 32×32", stats32)
    lines += ["", "## Интерпретация", "",
              "Автоматическая часть отчёта сохраняет измерения без субъективной фильтрации. "
              "Содержательная интерпретация дополняется после просмотра итоговых карт и статистики.", "",
              f"Сырые численные данные: `{relative / 'mask_heatmap_stats.json'}`.", ""]
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("\n".join(lines))
    print(md)


if __name__ == "__main__":
    main()
