"""Aggregate latent-adapter experiments over independent frozen VAE pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import t as student_t

from .decoder_agreement import align_columns


METHODS = ("independent", "identity", "constant", "linear", "mlp")
LABELS = {"independent": "Independent prior", "identity": "Identity",
          "constant": "Constant", "linear": "Linear", "mlp": "Residual MLP",
          "adam_independent": "z₂-only Adam", "adam_mlp": "MLP + Adam"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    if len(array) > 1:
        half = float(student_t.ppf(.975, len(array) - 1) * std / np.sqrt(len(array)))
        ci = [float(array.mean() - half), float(array.mean() + half)]
    else:
        ci = None
    return {"n": len(array), "mean": float(array.mean()), "sample_std": std,
            "ci95_t": ci, "min": float(array.min()), "max": float(array.max()),
            "values": array.tolist()}


def gold_iou(masks: torch.Tensor) -> float:
    from ..data.generate import ideal_mask
    gold = ideal_mask().float().unsqueeze(0).expand(len(masks), -1, -1)
    aligned = align_columns(gold, masks.float())
    intersection = (gold.bool() & aligned.bool()).sum((1, 2)).float()
    union = (gold.bool() | aligned.bool()).sum((1, 2)).float()
    return float((intersection / union.clamp_min(1)).mean())


def validate(pair_dir: Path) -> None:
    provenance = json.loads((pair_dir / "provenance.json").read_text())
    for key, path in (("protocol_sha256", pair_dir / "protocol.json"),
                      ("result_sha256", pair_dir / "result.pt")):
        if provenance[key] != sha256_file(path):
            raise ValueError(f"{pair_dir}: {key} mismatch")
    adapter_source = Path(__file__).with_name("latent_adapter.py")
    if provenance.get("source_sha256") != sha256_file(adapter_source):
        raise ValueError(f"{pair_dir}: latent_adapter.py provenance mismatch")
    for model in provenance["models"]:
        checkpoint = Path(model["checkpoint"])
        if not checkpoint.exists() or sha256_file(checkpoint) != model["sha256"]:
            raise ValueError(f"{pair_dir}: checkpoint mismatch: {checkpoint}")


def read_pair(pair_dir: Path) -> dict:
    validate(pair_dir)
    result = torch.load(pair_dir / "result.pt", map_location="cpu", weights_only=True)
    directions = []
    for name, direction in result["directions"].items():
        record = {"direction": name, "methods": {}, "adam": {}}
        for method in METHODS:
            metrics = dict(direction["methods"][method]["metrics"])
            if "target_at_radius_fraction" not in metrics:
                norms = direction["methods"][method]["tensors"]["z_target"].norm(dim=1)
                radius = result["settings"]["radius"]
                metrics["target_at_radius_fraction"] = float(
                    (norms >= radius - 1e-5).float().mean())
            metrics["target_gold_iou"] = gold_iou(direction["methods"][method]["tensors"]["target_hard"])
            metrics["source_gold_iou"] = gold_iou(direction["methods"][method]["tensors"]["source_hard"])
            metrics["target_unique_fraction"] = metrics["target_unique"] / result["settings"]["n_test"]
            record["methods"][method] = metrics
        for source_name, output_name in (("independent_init", "adam_independent"),
                                         ("mlp_init", "adam_mlp")):
            metrics = dict(direction["adam_reference"][source_name]["metrics"])
            if "target_at_radius_fraction" not in metrics:
                norms = direction["adam_reference"][source_name]["tensors"]["z_target"].norm(dim=1)
                radius = result["settings"]["radius"]
                metrics["target_at_radius_fraction"] = float(
                    (norms >= radius - 1e-5).float().mean())
            metrics["target_gold_iou"] = gold_iou(
                direction["adam_reference"][source_name]["tensors"]["target_hard"])
            record["adam"][output_name] = metrics
        record["adam_subset_methods"] = direction["adam_subset_methods"]
        directions.append(record)
    pair_record = {"pair": result["model_seeds"], "settings": result["settings"],
                   "directions": directions, "methods": {}, "adam": {}}
    for method in METHODS:
        keys = directions[0]["methods"][method]
        pair_record["methods"][method] = {
            key: float(np.mean([direction["methods"][method][key] for direction in directions]))
            for key in keys if isinstance(directions[0]["methods"][method][key], (int, float))
        }
    for method in ("adam_independent", "adam_mlp"):
        keys = directions[0]["adam"][method]
        pair_record["adam"][method] = {
            key: float(np.mean([direction["adam"][method][key] for direction in directions]))
            for key in keys if isinstance(directions[0]["adam"][method][key], (int, float))
        }
    # Full methods rescored on the same small held-out subset as Adam.
    pair_record["adam_subset_methods"] = {}
    for method in METHODS:
        keys = directions[0]["adam_subset_methods"][method]
        pair_record["adam_subset_methods"][method] = {
            key: float(np.mean([d["adam_subset_methods"][method][key] for d in directions]))
            for key in keys if isinstance(directions[0]["adam_subset_methods"][method][key], (int, float))
        }
    return pair_record


def aggregate(records: list[dict]) -> dict:
    summary = {"n_pairs": len(records), "unit": "independent VAE pair",
               "pair_metrics": records, "methods": {}, "adam": {}, "paired_differences": {}}
    metric_names = ("soft_mse", "hard_iou", "hard_exact", "hard_hamming",
                    "fixed_soft_mse", "fixed_hard_iou", "fixed_hard_exact",
                    "target_norm_mean", "target_norm_max", "target_at_radius_fraction",
                    "target_unique_fraction",
                    "target_gold_iou", "source_gold_iou")
    for method in METHODS:
        summary["methods"][method] = {
            metric: describe([record["methods"][method][metric] for record in records])
            for metric in metric_names
        }
    adam_metrics = ("soft_mse", "hard_iou", "hard_exact", "hard_hamming",
                    "target_norm_mean", "target_norm_max", "target_at_radius_fraction",
                    "target_gold_iou")
    for method in ("adam_independent", "adam_mlp"):
        summary["adam"][method] = {
            metric: describe([record["adam"][method][metric] for record in records])
            for metric in adam_metrics
        }
    for comparison, left, right in (
            ("mlp_minus_linear", "mlp", "linear"),
            ("linear_minus_constant", "linear", "constant"),
            ("mlp_minus_identity", "mlp", "identity"),
            ("mlp_minus_independent", "mlp", "independent"),
            ("linear_minus_identity", "linear", "identity")):
        summary["paired_differences"][comparison] = {
            metric: describe([record["methods"][left][metric] - record["methods"][right][metric]
                              for record in records])
            for metric in ("soft_mse", "hard_iou", "hard_exact", "target_gold_iou")
        }
    summary["adam_paired_differences"] = {
        metric: describe([record["adam"]["adam_mlp"][metric]
                          - record["adam"]["adam_independent"][metric]
                          for record in records])
        for metric in ("soft_mse", "hard_iou", "hard_exact", "target_gold_iou")
    }
    # Pair-matched subset is the only fair comparison with the Adam references.
    summary["adam_subset"] = {}
    for method in METHODS:
        summary["adam_subset"][method] = {
            metric: describe([record["adam_subset_methods"][method][metric] for record in records])
            for metric in ("soft_mse", "hard_iou", "hard_exact", "hard_hamming")
        }
    return summary


def fmt(record: dict, digits: int = 4) -> str:
    ci = record["ci95_t"]
    if ci is None:
        return f"{record['mean']:.{digits}f}"
    return f"{record['mean']:.{digits}f} [{ci[0]:.{digits}f}; {ci[1]:.{digits}f}]"


def write_markdown(out: Path, summary: dict) -> None:
    lines = [
        "# Latent-adapter между независимо обученными pattern-VAE",
        "",
        f"Независимых VAE-пар: **{summary['n_pairs']}**. Для каждой пары обучены оба направления. "
        "Единица статистического повторения — VAE-пара; два направления сначала усреднены внутри пары.",
        "",
        "Decoder заморожены. Adapter обучался только по agreement мягких exact-32 масок с "
        "per-example Hungarian; Gold и task labels не участвовали в обучении, validation или выборе checkpoint.",
        "",
        "## Held-out latent-коды",
        "",
        "| Метод | Soft MSE ↓ | Hard IoU ↑ | Exact hard ↑ | Fixed-permutation IoU ↑ | Unique target masks | Post-hoc Gold IoU |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        record = summary["methods"][method]
        lines.append(f"| {LABELS[method]} | {fmt(record['soft_mse'], 6)} | "
                     f"{fmt(record['hard_iou'])} | {fmt(record['hard_exact'])} | "
                     f"{fmt(record['fixed_hard_iou'])} | "
                     f"{fmt(record['target_unique_fraction'])} | {fmt(record['target_gold_iou'])} |")
    lines += [
        "",
        "`Fixed-permutation` использует одну перестановку, выбранную на validation и затем замороженную "
        "для test. `Unique target masks` — доля уникальных hard-масок; значение около нуля выявляет collapse.",
        "",
        "## Односторонний Adam на фиксированном test subset",
        "",
        "| Метод | Soft MSE ↓ | Hard IoU ↑ | Exact hard ↑ |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        record = summary["adam_subset"][method]
        lines.append(f"| {LABELS[method]} | {fmt(record['soft_mse'], 6)} | "
                     f"{fmt(record['hard_iou'])} | {fmt(record['hard_exact'])} |")
    for method in ("adam_independent", "adam_mlp"):
        record = summary["adam"][method]
        lines.append(f"| {LABELS[method]} | {fmt(record['soft_mse'], 6)} | "
                     f"{fmt(record['hard_iou'])} | {fmt(record['hard_exact'])} |")
    adam_delta = summary["adam_paired_differences"]
    lines.append(f"| MLP+Adam − Adam | {fmt(adam_delta['soft_mse'], 6)} | "
                 f"{fmt(adam_delta['hard_iou'])} | {fmt(adam_delta['hard_exact'])} |")
    diff = summary["paired_differences"]
    lines += [
        "",
        "## Парные эффекты",
        "",
        "| Сравнение | Δ Soft MSE | Δ Hard IoU | Δ Exact hard | Δ Gold IoU |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, label in (("mlp_minus_linear", "MLP − linear"),
                        ("linear_minus_constant", "Linear − constant"),
                        ("mlp_minus_identity", "MLP − identity"),
                        ("mlp_minus_independent", "MLP − independent"),
                        ("linear_minus_identity", "Linear − identity")):
        row = diff[name]
        lines.append(f"| {label} | {fmt(row['soft_mse'], 6)} | {fmt(row['hard_iou'])} | "
                     f"{fmt(row['hard_exact'])} | {fmt(row['target_gold_iou'])} |")
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")


def render(out: Path, summary: dict) -> None:
    methods = list(METHODS)
    colors = ["#999999", "#7189a8", "#b39162", "#4c956c", "#2f6f59"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), layout="constrained")
    specs = (("soft_mse", "Soft disagreement", "MSE", True),
             ("hard_iou", "Hard agreement", "IoU", False),
             ("fixed_hard_iou", "One validation permutation", "Hard IoU", False))
    for axis, (metric, title, ylabel, log) in zip(axes, specs):
        means = [summary["methods"][method][metric]["mean"] for method in methods]
        cis = [summary["methods"][method][metric]["ci95_t"] for method in methods]
        errors = [[mean - ci[0] if ci is not None else 0. for mean, ci in zip(means, cis)],
                  [ci[1] - mean if ci is not None else 0. for mean, ci in zip(means, cis)]]
        axis.bar(range(len(methods)), means, yerr=errors, color=colors, capsize=3)
        axis.set_xticks(range(len(methods)), [LABELS[m] for m in methods], rotation=25, ha="right")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        if log:
            axis.set_yscale("log")
        else:
            axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=.2)
    for extension in ("png", "pdf"):
        fig.savefig(out / f"latent_adapter_summary.{extension}", dpi=180)
    plt.close(fig)


def render_mask_heatmaps(out: Path, pair_dir: Path) -> None:
    """Visual diagnostic in fixed order; no example is quality-selected."""
    from ..data.generate import ideal_mask

    result = torch.load(pair_dir / "result.pt", map_location="cpu", weights_only=True)
    directions = list(result["directions"].values())
    names = ("independent", "constant", "linear", "mlp")
    titles = ("Source prior", "Independent", "Constant", "Linear", "Residual MLP", "Ideal")
    fig, axes = plt.subplots(len(directions), len(titles), figsize=(13, 4.7),
                             layout="constrained", squeeze=False)
    gold = ideal_mask().float()
    for row, direction in enumerate(directions):
        source = direction["methods"]["identity"]["tensors"]["source_hard"].float()
        heatmaps = [source.mean(0)]
        for name in names:
            target = direction["methods"][name]["tensors"]["target_hard"].float()
            heatmaps.append(align_columns(source, target).mean(0))
        heatmaps.append(gold)
        for axis, matrix, title in zip(axes[row], heatmaps, titles):
            image = axis.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(title)
        axes[row, 0].set_ylabel(direction["direction"].replace("_to_", " → "))
    fig.colorbar(image, ax=axes, shrink=.72, label="Activation frequency")
    for extension in ("png", "pdf"):
        fig.savefig(out / f"latent_adapter_mask_heatmaps.{extension}", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    pair_dirs = sorted(path for path in args.root.glob("pair_*") if (path / "result.pt").exists())
    if not pair_dirs:
        raise FileNotFoundError(f"no completed pair results under {args.root}")
    records = [read_pair(path) for path in pair_dirs]
    settings = [record["settings"] for record in records]
    if any(item != settings[0] for item in settings[1:]):
        raise ValueError("pair settings differ")
    summary = aggregate(records)
    summary["settings"] = settings[0]
    (args.root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    write_markdown(args.root, summary)
    render(args.root, summary)
    render_mask_heatmaps(args.root, pair_dirs[0])
    print(f"[report] {len(records)} pairs -> {args.root / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
