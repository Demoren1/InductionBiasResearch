"""Aggregate pattern decoder-agreement results across independent VAE pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.generate import ideal_mask  # noqa: E402
from evaluation.decoder_agreement import align_columns  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_provenance(pair_root: Path) -> None:
    provenance = json.loads((pair_root / "search_provenance.json").read_text())
    expected = {
        "protocol_sha256": sha256_file(pair_root / "protocol.json"),
        "mask_sha256": sha256_file(pair_root / "masks.pt"),
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"{pair_root}: {key} mismatch")
    for model in provenance.get("models", []):
        checkpoint = Path(model["checkpoint"])
        if not checkpoint.exists() or sha256_file(checkpoint) != model.get("sha256"):
            raise ValueError(f"{pair_root}: checkpoint provenance mismatch for {checkpoint}")


def describe(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    sample_std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    if array.size > 1:
        half_width = float(student_t.ppf(.975, array.size - 1) * sample_std / np.sqrt(array.size))
        ci95 = [float(array.mean() - half_width), float(array.mean() + half_width)]
    else:
        ci95 = None
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "sample_std": sample_std,
        "ci95_t": ci95,
        "min": float(array.min()),
        "max": float(array.max()),
        "values": array.tolist(),
    }


def iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aligned = align_columns(a.float(), b.float())
    a_bool, b_bool = a.bool(), aligned.bool()
    intersection = (a_bool & b_bool).sum((1, 2)).float()
    union = (a_bool | b_bool).sum((1, 2)).float()
    return intersection / union.clamp_min(1)


def exact_fraction(a: torch.Tensor, b: torch.Tensor) -> float:
    aligned = align_columns(a.float(), b.float())
    return float((a == aligned).all(2).all(1).float().mean())


def gold_iou(masks: torch.Tensor) -> torch.Tensor:
    gold = ideal_mask().float().unsqueeze(0).expand(len(masks), -1, -1)
    return iou(gold, masks)


def read_pair(pair_root: Path) -> dict:
    validate_provenance(pair_root)
    protocol = json.loads((pair_root / "protocol.json").read_text())
    optimization = torch.load(pair_root / "optimization.pt", map_location="cpu", weights_only=True)
    random = torch.load(pair_root / "random_search.pt", map_location="cpu", weights_only=True)
    training = json.loads((pair_root / "training.json").read_text())
    stages = {
        "initial": (optimization["initial_masks1"], optimization["initial_masks2"]),
        "optimized": (optimization["final_masks1"], optimization["final_masks2"]),
        "random_search": (random["masks1"], random["masks2"]),
    }
    pair_record = {
        "pair": protocol["model_seeds"],
        "n_starts": protocol["search"]["n_starts"],
        "soft_mse": {
            "initial": float(optimization["initial_loss"].mean()),
            "optimized": float(optimization["final_loss"].mean()),
            "random_search": float(random["loss"].mean()),
        },
        "hard_pair_iou": {name: float(iou(*masks).mean()) for name, masks in stages.items()},
        "hard_exact_fraction": {name: exact_fraction(*masks) for name, masks in stages.items()},
        "gold_iou": {},
        "vae_validation_loss": [float(result["best_val"]) for result in training["results"]],
        "vae_decoder_probability_std": [
            float(result["noncollapse"]["decoder_probability_feature_std_mean"])
            for result in training["results"]
        ],
    }
    for name, masks in stages.items():
        per_decoder = [float(gold_iou(mask).mean()) for mask in masks]
        pair_record["gold_iou"][name] = {
            "decoder_means": per_decoder,
            "pair_mean": float(np.mean(per_decoder)),
        }
    pair_record["gold_iou"]["optimized_minus_initial"] = (
        pair_record["gold_iou"]["optimized"]["pair_mean"]
        - pair_record["gold_iou"]["initial"]["pair_mean"]
    )
    norms = torch.cat([optimization["final_z1"], optimization["final_z2"]]).norm(dim=1)
    soft = torch.cat([optimization["final_soft1"], optimization["final_soft2"]])
    radius = float(protocol["search"]["latent_radius"])
    pair_record["final_latents"] = {
        "norm_mean": float(norms.mean()),
        "at_radius_fraction": float((norms >= radius - 1e-5).float().mean()),
        "softness_mean": float((soft * (1 - soft)).mean()),
    }
    if (pair_root / "summary.json").exists():
        task_summary = json.loads((pair_root / "summary.json").read_text())
        decoder_accuracy = {
            name: task_summary["downstream"][name]["accuracy"]["mean"]
            for name in ("initial_vae1", "initial_vae2", "optimized_vae1", "optimized_vae2",
                         "random_search_vae1", "random_search_vae2", "random_exact32", "ideal")
        }
        initial = float(np.mean([decoder_accuracy["initial_vae1"], decoder_accuracy["initial_vae2"]]))
        optimized = float(np.mean([decoder_accuracy["optimized_vae1"], decoder_accuracy["optimized_vae2"]]))
        random_search = float(np.mean([
            decoder_accuracy["random_search_vae1"], decoder_accuracy["random_search_vae2"]]))
        pair_record["downstream_decoder_accuracy"] = decoder_accuracy
        pair_record["downstream_accuracy"] = {
            "initial": initial,
            "optimized": optimized,
            "random_search": random_search,
            "optimized_minus_initial": optimized - initial,
            "optimized_minus_random_search": optimized - random_search,
            "random_exact32": decoder_accuracy["random_exact32"],
            "ideal": decoder_accuracy["ideal"],
        }
    return pair_record


def aggregate(records: list[dict]) -> dict:
    result = {"n_pairs": len(records), "unit": "independent VAE pair", "pair_metrics": records}
    result["across_pair_means"] = {}
    for family in ("soft_mse", "hard_pair_iou", "hard_exact_fraction"):
        result["across_pair_means"][family] = {
            stage: describe([record[family][stage] for record in records])
            for stage in ("initial", "optimized", "random_search")
        }
    result["across_pair_means"]["gold_iou"] = {
        stage: describe([record["gold_iou"][stage]["pair_mean"] for record in records])
        for stage in ("initial", "optimized", "random_search")
    }
    result["across_pair_means"]["gold_iou"]["optimized_minus_initial"] = describe(
        [record["gold_iou"]["optimized_minus_initial"] for record in records]
    )
    result["across_pair_means"]["gold_iou"]["optimized_minus_random_search"] = describe(
        [record["gold_iou"]["optimized"]["pair_mean"]
         - record["gold_iou"]["random_search"]["pair_mean"] for record in records]
    )
    result["totals"] = {
        "latent_starts": sum(record["n_starts"] for record in records),
        "optimized_exact_matches": sum(
            round(record["hard_exact_fraction"]["optimized"] * record["n_starts"])
            for record in records
        ),
    }
    if all("downstream_accuracy" in record for record in records):
        names = records[0]["downstream_accuracy"]
        result["across_pair_means"]["downstream_accuracy"] = {
            name: describe([record["downstream_accuracy"][name] for record in records])
            for name in names
        }
    return result


def metric_text(record: dict, digits: int = 4) -> str:
    interval = record["ci95_t"]
    if interval is None:
        return f"{record['mean']:.{digits}f} (ДИ не определён для одной пары)"
    return (f"{record['mean']:.{digits}f} "
            f"[{interval[0]:.{digits}f}; {interval[1]:.{digits}f}]")


def render_markdown(root: Path, summary: dict) -> None:
    aggregate_metrics = summary["across_pair_means"]
    lines = [
        "# Agreement pattern-VAE при разных инициализациях",
        "",
        f"Полных независимых пар VAE: **{summary['n_pairs']}**. "
        f"Латентных стартов: **{summary['totals']['latent_starts']}**. "
        "Единица повторения для разброса — пара независимо обученных VAE, а не отдельный latent-старт.",
        "",
        "Данные, task split, map split/order, начальные latent-векторы и гиперпараметры "
        "одинаковы между парами; меняются только seeds обучения VAE. Decoder после обучения заморожены. "
        "Поиск минимизирует только MSE мягких top-32 масок после Hungarian-сопоставления колонок.",
        "",
        "| VAE seeds | Soft MSE: initial → optimized | Hard pair IoU: initial → optimized | Exact hard matches | Gold IoU: initial → optimized | Δ Gold IoU |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for record in summary["pair_metrics"]:
        pair = "/".join(map(str, record["pair"]))
        n_exact = round(record["hard_exact_fraction"]["optimized"] * record["n_starts"])
        lines.append(
            f"| {pair} | {record['soft_mse']['initial']:.6f} → {record['soft_mse']['optimized']:.6f} | "
            f"{record['hard_pair_iou']['initial']:.4f} → {record['hard_pair_iou']['optimized']:.4f} | "
            f"{n_exact}/{record['n_starts']} | "
            f"{record['gold_iou']['initial']['pair_mean']:.4f} → "
            f"{record['gold_iou']['optimized']['pair_mean']:.4f} | "
            f"{record['gold_iou']['optimized_minus_initial']:+.4f} |"
        )
    lines += [
        "",
        "## Сводка по независимым парам",
        "",
        "Среднее и 95% t-ДИ по независимым VAE-парам:",
        "",
        f"- soft MSE после agreement: **{metric_text(aggregate_metrics['soft_mse']['optimized'], 6)}**;",
        f"- hard IoU между двумя decoder: **{metric_text(aggregate_metrics['hard_pair_iou']['optimized'])}**;",
        f"- доля точных совпадений hard-масок: **{metric_text(aggregate_metrics['hard_exact_fraction']['optimized'])}**;",
        f"- Gold IoU общей пары после agreement: **{metric_text(aggregate_metrics['gold_iou']['optimized'])}**;",
        f"- изменение Gold IoU относительно prior: **{metric_text(aggregate_metrics['gold_iou']['optimized_minus_initial'])}**.",
        "",
        "Gold support используется только для постфактум диагностики после сохранения `masks.pt`; "
        "он не входит в loss, выбор итерации или обучение VAE. Почти нулевой disagreement сам по себе "
        "не означает восстановление идеальной структуры: два decoder могут согласовать общий неверный минимум.",
        f"95% t-ДИ условны на одном task split и фиксированном наборе данных; они отражают разброс по "
        f"{summary['n_pairs']} инициализациям пар VAE, но не неопределённость по новым task splits.",
    ]
    if "downstream_accuracy" in aggregate_metrics:
        acc = aggregate_metrics["downstream_accuracy"]
        lines += [
            "",
            "## Held-out fresh-MLP evaluation",
            "",
            f"Средняя accuracy двух decoder: prior **{metric_text(acc['initial'])}**, "
            f"после agreement **{metric_text(acc['optimized'])}**, random search "
            f"**{metric_text(acc['random_search'])}**.",
            "",
            f"Парное изменение agreement − prior: **{metric_text(acc['optimized_minus_initial'])}**; "
            f"agreement − random search: **{metric_text(acc['optimized_minus_random_search'])}**. "
            f"ДИ считаются по {summary['n_pairs']} VAE-парам; внутри каждой пары усреднены оба decoder "
            "и все held-out задачи.",
        ]
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")


def render_plot(root: Path, records: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout="constrained")
    for axis, family, title, ylabel in (
        (axes[0], "hard_pair_iou", "Decoder agreement", "Hard pair IoU"),
        (axes[1], "gold_iou", "Post-hoc structure", "Gold IoU"),
    ):
        if family == "gold_iou":
            before = [record[family]["initial"]["pair_mean"] for record in records]
            after = [record[family]["optimized"]["pair_mean"] for record in records]
        else:
            before = [record[family]["initial"] for record in records]
            after = [record[family]["optimized"] for record in records]
        for idx in range(len(records)):
            axis.plot([0, 1], [before[idx], after[idx]], color="#777777", alpha=.18, linewidth=.8)
        for position, values, color in ((0, before, "#7295b6"), (1, after, "#328369")):
            stats = describe(values)
            ci = stats["ci95_t"]
            error = [[stats["mean"] - ci[0]], [ci[1] - stats["mean"]]] if ci else None
            axis.errorbar(position, stats["mean"], yerr=error, fmt="o", color=color,
                          markersize=7, capsize=5, linewidth=2, zorder=4)
        axis.set(xticks=[0, 1], xticklabels=["prior", "agreement"], ylabel=ylabel, title=title)
        axis.grid(alpha=.2)
    exact = [record["hard_exact_fraction"]["optimized"] for record in records]
    jitter = np.linspace(-.08, .08, len(exact))
    axes[2].scatter(jitter, exact, color="#777777", alpha=.5, s=18)
    exact_stats = describe(exact)
    exact_ci = exact_stats["ci95_t"]
    exact_error = [[exact_stats["mean"] - exact_ci[0]],
                   [exact_ci[1] - exact_stats["mean"]]] if exact_ci else None
    axes[2].errorbar(0, exact_stats["mean"], yerr=exact_error, fmt="o", color="#328369",
                     markersize=8, capsize=6, linewidth=2, zorder=4)
    axes[2].set(xticks=[0], xticklabels=[f"{len(records)} VAE pairs"], ylim=(0, 1.01),
                ylabel="Fraction", title="Exact hard-mask matches")
    axes[2].grid(axis="y", alpha=.2)
    fig.suptitle("Thin lines/dots: VAE pairs; colored point and bars: mean and 95% t-CI", fontsize=11)
    for extension in ("png", "pdf"):
        fig.savefig(root / f"agreement_replicates.{extension}", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    suite = json.loads((root / "suite_protocol.json").read_text())
    pair_roots = [root / name for name in suite["pair_directories"]]
    missing = [str(path) for path in pair_roots if not (path / "masks.pt").exists()]
    if missing:
        raise FileNotFoundError("agreement search is incomplete for: " + ", ".join(missing))
    records = [read_pair(path) for path in pair_roots]
    summary = {"suite_protocol": suite, **aggregate(records)}
    (root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    render_markdown(root, summary)
    render_plot(root, records)
    print(json.dumps(summary["across_pair_means"], indent=2), flush=True)


if __name__ == "__main__":
    main()
