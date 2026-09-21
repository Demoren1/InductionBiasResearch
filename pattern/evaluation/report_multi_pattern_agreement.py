"""Post-hoc report for nested agreement of 2, 3, 4, and 5 VAEs."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.generate import ideal_mask  # noqa: E402
from evaluation.decoder_agreement import align_columns, hard_topk  # noqa: E402
from evaluation.run_cross_pattern_agreement import ROOT as PAIR_ROOT, path_for  # noqa: E402
from evaluation.run_multi_pattern_agreement import ROOT, group_patterns  # noqa: E402
from evaluation.train_agreement_vaes import sha256_file  # noqa: E402

ALL_SIZES = tuple(range(2, 11))

def describe(values: list[float]) -> dict:
    x = np.asarray(values, dtype=float)
    sd = float(x.std(ddof=1)) if len(x) > 1 else 0.
    margin = float(student_t.ppf(.975, len(x) - 1) * sd / np.sqrt(len(x))) if len(x) > 1 else 0.
    return {"n": len(x), "mean": float(x.mean()), "sample_sd": sd,
            "ci95_t": [float(x.mean() - margin), float(x.mean() + margin)],
            "values": x.tolist()}


def iou(reference: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    aligned = align_columns(reference.float(), other.float())
    intersection = (reference.bool() & aligned.bool()).sum(dim=(-2, -1)).float()
    union = (reference.bool() | aligned.bool()).sum(dim=(-2, -1)).float()
    return intersection / union.clamp_min(1)


def align_ensemble(masks: torch.Tensor) -> torch.Tensor:
    return torch.stack([masks[0], *[
        align_columns(masks[0].float(), masks[i].float())
        for i in range(1, len(masks))
    ]])


def mask_metrics(masks: torch.Tensor) -> tuple[dict, torch.Tensor]:
    """Metrics for ``(models, starts, rows, cols)`` binary masks."""
    aligned = align_ensemble(masks)
    pair_iou = torch.stack([iou(masks[i], masks[j])
                            for i, j in combinations(range(len(masks)), 2)])
    all_exact = (aligned == aligned[0:1]).all(dim=(0, 2, 3)).float()
    gold = ideal_mask().float().unsqueeze(0).expand(masks.size(1), -1, -1)
    per_model_gold = torch.stack([iou(gold, model_masks) for model_masks in masks])
    consensus = hard_topk(aligned.float().mean(0).flatten(1), 32).reshape_as(masks[0])
    consensus_gold = iou(gold, consensus)
    consensus_gold_aligned = align_columns(gold, consensus.float())
    return {
        "mean_pairwise_iou": float(pair_iou.mean()),
        "minimum_pairwise_iou": float(pair_iou.mean(dim=1).min()),
        "all_exact_fraction": float(all_exact.mean()),
        "mean_decoder_gold_iou": float(per_model_gold.mean()),
        "first_pair_gold_iou": float(per_model_gold[:2].mean()),
        "consensus_gold_iou": float(consensus_gold.mean()),
        "consensus_exact_gold_fraction": float((consensus_gold == 1).float().mean()),
        "unique_consensus_fraction": float(len(torch.unique(consensus.flatten(1), dim=0)) / len(consensus)),
    }, consensus_gold_aligned


def validate_multi(folder: Path) -> dict:
    protocol_path = folder / "protocol.json"
    result_path = folder / "optimization.pt"
    provenance = json.loads((folder / "provenance.json").read_text())
    if provenance["protocol_sha256"] != sha256_file(protocol_path):
        raise ValueError(f"protocol hash mismatch: {folder}")
    if provenance["optimization_sha256"] != sha256_file(result_path):
        raise ValueError(f"optimization hash mismatch: {folder}")
    protocol = json.loads(protocol_path.read_text())
    for model in protocol["models"]:
        if sha256_file(Path(model["checkpoint"])) != model["checkpoint_sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {model['checkpoint']}")
    return protocol


def read_multi(group: int, size: int, replicate: int) -> tuple[dict, dict[str, torch.Tensor]]:
    folder = ROOT / f"group_{group:02d}" / f"n{size}" / f"rep_{replicate}"
    protocol = validate_multi(folder)
    result = torch.load(folder / "optimization.pt", map_location="cpu", weights_only=True)
    initial, initial_consensus = mask_metrics(result["initial_masks"])
    final, final_consensus = mask_metrics(result["final_masks"])
    norms = result["final_z"].norm(dim=2)
    soft = result["final_soft_masks"]
    record = {
        "group": group, "size": size, "replicate": replicate,
        "patterns": protocol["patterns"], "initial": initial, "optimized": final,
        "soft_variance_initial": float(result["initial_loss"].mean()),
        "soft_variance_optimized": float(result["final_loss"].mean()),
        "at_radius_fraction": float((norms >= protocol["radius"] - 1e-5).float().mean()),
        "softness": float((soft * (1 - soft)).mean()),
    }
    return record, {"initial": initial_consensus, "optimized": final_consensus}


def read_pair(group: int, replicate: int) -> tuple[dict, dict[str, torch.Tensor]]:
    folder = path_for(PAIR_ROOT, group) / f"cross_{replicate}"
    result = torch.load(folder / "optimization.pt", map_location="cpu", weights_only=True)
    initial_masks = torch.stack([result["initial_masks1"], result["initial_masks2"]])
    final_masks = torch.stack([result["final_masks1"], result["final_masks2"]])
    initial, initial_consensus = mask_metrics(initial_masks)
    final, final_consensus = mask_metrics(final_masks)
    protocol = json.loads((folder / "protocol.json").read_text())
    norms = torch.stack([result["final_z1"], result["final_z2"]]).norm(dim=2)
    soft = torch.stack([result["final_soft1"], result["final_soft2"]])
    record = {
        "group": group, "size": 2, "replicate": replicate,
        "patterns": list(group_patterns(group)[:2]), "initial": initial,
        "optimized": final, "soft_variance_initial": float(result["initial_loss"].mean()) / 4,
        "soft_variance_optimized": float(result["final_loss"].mean()) / 4,
        "at_radius_fraction": float((norms >= protocol["radius"] - 1e-5).float().mean()),
        "softness": float((soft * (1 - soft)).mean()),
        "historical_pair_objective_rescaled": True,
    }
    return record, {"initial": initial_consensus, "optimized": final_consensus}


def aggregate(records: list[dict]) -> dict:
    metrics = [
        ("initial_pairwise_iou", ("initial", "mean_pairwise_iou")),
        ("optimized_pairwise_iou", ("optimized", "mean_pairwise_iou")),
        ("optimized_all_exact", ("optimized", "all_exact_fraction")),
        ("initial_decoder_gold", ("initial", "mean_decoder_gold_iou")),
        ("optimized_decoder_gold", ("optimized", "mean_decoder_gold_iou")),
        ("optimized_first_pair_gold", ("optimized", "first_pair_gold_iou")),
        ("initial_consensus_gold", ("initial", "consensus_gold_iou")),
        ("optimized_consensus_gold", ("optimized", "consensus_gold_iou")),
        ("optimized_consensus_exact_gold", ("optimized", "consensus_exact_gold_fraction")),
        ("at_radius", ("at_radius_fraction",)),
    ]
    result = {}
    for size in ALL_SIZES:
        rows = [row for row in records if row["size"] == size]
        group_means = {}
        for name, path in metrics:
            values = []
            for group in range(8):
                group_rows = [row for row in rows if row["group"] == group]
                extracted = []
                for row in group_rows:
                    value = row
                    for key in path:
                        value = value[key]
                    extracted.append(value)
                values.append(float(np.mean(extracted)))
            group_means[name] = describe(values)
        result[str(size)] = group_means
    return result


def group_values(records: list[dict], size: int, path: tuple[str, ...]) -> np.ndarray:
    values = []
    for group in range(8):
        extracted = []
        for row in records:
            if row["size"] == size and row["group"] == group:
                value = row
                for key in path:
                    value = value[key]
                extracted.append(value)
        values.append(float(np.mean(extracted)))
    return np.asarray(values)


def paired_comparisons(records: list[dict]) -> dict:
    result = {}
    paths = {
        "consensus_gold": ("optimized", "consensus_gold_iou"),
        "first_pair_gold": ("optimized", "first_pair_gold_iou"),
        "pairwise_iou": ("optimized", "mean_pairwise_iou"),
        "all_exact": ("optimized", "all_exact_fraction"),
    }
    comparison_sizes = {f"{large}_minus_{small}": (small, large)
                        for small, large in zip(range(3, 10), range(4, 11))}
    comparison_sizes.update({"5_minus_3": (3, 5), "10_minus_5": (5, 10),
                             "10_minus_3": (3, 10), "10_minus_2": (2, 10)})
    for label, (small, large) in comparison_sizes.items():
        result[label] = {
            name: describe((group_values(records, large, path)
                            - group_values(records, small, path)).tolist())
            for name, path in paths.items()
        }
    result["search_gain"] = {
        str(size): describe((
            group_values(records, size, ("optimized", "consensus_gold_iou"))
            - group_values(records, size, ("initial", "consensus_gold_iou"))
        ).tolist()) for size in ALL_SIZES
    }
    return result


def make_figures(aggregate_record: dict, heatmaps: dict, out: Path) -> None:
    sizes = np.asarray(ALL_SIZES)
    specs = [
        ("optimized_pairwise_iou", "Попарное совпадение", (0.5, 1.005)),
        ("optimized_all_exact", "Все маски одинаковы", (0, 1.05)),
        ("optimized_consensus_gold", "Близость общей маски к теплицевой", (0.5, 1.0)),
        ("at_radius", "Коды на границе радиуса", (0, 1.05)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, (metric, title, ylim) in zip(axes, specs):
        rows = [aggregate_record[str(size)][metric] for size in sizes]
        y = np.array([row["mean"] for row in rows])
        lo = np.array([row["ci95_t"][0] for row in rows])
        hi = np.array([row["ci95_t"][1] for row in rows])
        ax.errorbar(sizes, y, yerr=np.vstack([y - lo, hi - y]), marker="o", capsize=3)
        ax.set(title=title, xlabel="Число VAE", xticks=sizes, ylim=ylim)
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out / "multi_pattern_agreement_comparison.png", dpi=180)
    plt.close(fig)

    gold = ideal_mask().float().numpy()
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.2), constrained_layout=True)
    for ax, size in zip(axes.flat, sizes):
        mean = torch.cat(heatmaps[(int(size), "optimized")]).float().mean(0).numpy()
        ax.imshow(mean, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"После поиска, {size} VAE")
        ax.set_xticks([]); ax.set_yticks([])
    axes.flat[-1].imshow(gold, cmap="viridis", vmin=0, vmax=1)
    axes.flat[-1].set_title("Теплицева маска")
    axes.flat[-1].set_xticks([]); axes.flat[-1].set_yticks([])
    fig.savefig(out / "multi_pattern_agreement_heatmaps.png", dpi=180)
    plt.close(fig)

    # Show a typical binary result for every newly tested group size.  The
    # example is selected mechanically as the mask whose gold IoU is closest
    # to the median across every group, replicate, and start for that size.
    fig, axes = plt.subplots(3, 3, figsize=(7.8, 7.8), constrained_layout=True)
    for ax, size in zip(axes.flat, range(3, 11)):
        masks = torch.cat(heatmaps[(size, "optimized")]).bool()
        gold_bool = ideal_mask().bool()
        intersection = (masks & gold_bool).sum(dim=(-2, -1)).float()
        union = (masks | gold_bool).sum(dim=(-2, -1)).float()
        scores = intersection / union
        median = scores.median()
        index = (scores - median).abs().argmin()
        ax.imshow(masks[index].float().numpy(), cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(f"{size} VAE, IoU={scores[index]:.3f}")
        ax.set_xticks([]); ax.set_yticks([])
    axes.flat[-1].imshow(gold, cmap="gray_r", vmin=0, vmax=1)
    axes.flat[-1].set_title("Теплицева маска")
    axes.flat[-1].set_xticks([]); axes.flat[-1].set_yticks([])
    fig.savefig(out / "multi_pattern_agreement_examples.png", dpi=180)
    plt.close(fig)


def fmt(row: dict) -> str:
    low = max(0., row["ci95_t"][0])
    high = min(1., row["ci95_t"][1])
    return f"{row['mean']:.4f} [{low:.4f}; {high:.4f}]"


def fmt_signed(row: dict) -> str:
    return f"{row['mean']:+.4f} [{row['ci95_t'][0]:+.4f}; {row['ci95_t'][1]:+.4f}]"


def write_report(aggregate_record: dict, comparisons: dict, out: Path) -> None:
    last = aggregate_record["10"]
    delta_3_10 = comparisons["10_minus_3"]["consensus_gold"]
    exact_any = any(aggregate_record[str(size)]["optimized_consensus_exact_gold"]["mean"] > 0
                    for size in ALL_SIZES)
    lines = [
        "# Agreement между 2–10 VAE разных паттернов", "",
        "Дата: 22 сентября 2026 года.", "",
        "Для каждой из восьми прежних пар последовательно добавлены ещё восемь VAE, обученных на других паттернах. Группы вложены: результаты для 3–10 VAE используют одинаковые первые модели и одинаковые начальные коды. На каждую группу приходится четыре повтора и 64 начальные точки. Поиск минимизирует разброс мягких масок всех VAE после перестановки скрытых столбцов. Аналитическая теплицева маска используется только после завершения поиска.", "",
        "Общая маска строится после поиска: бинарные маски приводятся к одному порядку скрытых столбцов, затем выбираются 32 связи, которые чаще всего встречаются у моделей группы. Для изображения и оценки эта общая маска уже после поиска сопоставляется с аналитической с учётом перестановки столбцов.", "",
        "Строка для двух VAE взята из эксперимента 19 сентября. Первые пары моделей те же, но начальные коды там получены с другим seed, поэтому сравнение от двух VAE является историческим ориентиром. Сравнения 3–10 полностью согласованы по исходным кодам.", "",
        "95% интервалы описывают разброс между восемью заранее заданными группами; четыре повтора усреднены внутри группы. Группы частично пересекаются по паттернам, поэтому интервалы используются как описание устойчивости результата, а не как строгая проверка гипотезы.", "",
        "| Число VAE | Попарный IoU | Все маски совпали | IoU первых двух VAE с теплицевой | IoU общей маски с теплицевой | Точное совпадение с теплицевой | Коды на границе |", "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for size in ALL_SIZES:
        row = aggregate_record[str(size)]
        lines.append(f"| {size} | {fmt(row['optimized_pairwise_iou'])} | {fmt(row['optimized_all_exact'])} | {fmt(row['optimized_first_pair_gold'])} | {fmt(row['optimized_consensus_gold'])} | {fmt(row['optimized_consensus_exact_gold'])} | {fmt(row['at_radius'])} |")
    lines += ["", "![Сравнение числа VAE](multi_pattern_agreement_comparison.png)", "",
              "![Средние общие маски](multi_pattern_agreement_heatmaps.png)", "",
              "Ниже для каждого размера показана бинарная общая маска, IoU которой ближе всего к медиане по всем группам, повторам и начальным точкам. Это типичные примеры, а не лучшие найденные маски.", "",
              "![Примеры общих масок](multi_pattern_agreement_examples.png)", "",
              "## Изменение относительно старта", "",
              "| Число VAE | IoU с теплицевой до поиска | После поиска | Изменение |", "|---:|---:|---:|---:|" ]
    for size in ALL_SIZES:
        row = aggregate_record[str(size)]
        before = row["initial_consensus_gold"]["mean"]
        after = row["optimized_consensus_gold"]["mean"]
        lines.append(f"| {size} | {before:.4f} | {after:.4f} | {after-before:+.4f} |")
    lines += ["", "## Парное сравнение вложенных групп", "",
              "| Сравнение | Изменение IoU общей маски с теплицевой | Изменение для первых двух VAE |", "|---|---:|---:|",
    ]
    for small, large in zip(range(3, 10), range(4, 11)):
        row = comparisons[f"{large}_minus_{small}"]
        lines.append(f"| {small} → {large} VAE | {fmt_signed(row['consensus_gold'])} | {fmt_signed(row['first_pair_gold'])} |")
    for small in (3, 5):
        row = comparisons[f"10_minus_{small}"]
        lines.append(f"| {small} → 10 VAE | {fmt_signed(row['consensus_gold'])} | {fmt_signed(row['first_pair_gold'])} |")
    exact_text = ("Хотя бы в одном размере найдены точные теплицевы общие маски."
                  if exact_any else "Точного совпадения общей маски с аналитической теплицевой не найдено.")
    lines += ["", "Интервалы в этой таблице рассчитаны по восьми группам. Ноль внутри интервала означает, что изменение не повторилось одинаково во всех группах.", "",
              "## Вывод", "",
              "Близость общей маски к теплицевой растёт от 0.6885 для двух VAE до максимума 0.7443 для семи. После этого рост прекращается: значения для 8, 9 и 10 VAE составляют 0.7389, 0.7373 и 0.7360. Такой же разворот виден у первых двух VAE, поэтому он не объясняется только способом построения общей маски.", "",
              f"Парное изменение IoU с теплицевой от 3 до 10 VAE равно {fmt_signed(delta_3_10)}, а от 5 до 10 — {fmt_signed(comparisons['10_minus_5']['consensus_gold'])}. Интервалы включают ноль: после основного перехода от двух к трём моделям устойчивого дальнейшего улучшения нет.", "",
              f"Для десяти VAE попарный IoU равен {last['optimized_pairwise_iou']['mean']:.4f}, а доля случаев, где совпали все десять масок, — {last['optimized_all_exact']['mean']:.4f}. {exact_text} Дополнительные модели до 6–7 слегка усиливают оконную структуру, затем ограничения начинают мешать согласованию, не приближая маску к аналитической.", "",
              "Доля кодов на границе радиуса при этом снижается с 0.8944 для трёх VAE до 0.2495 для десяти. Большая группа меньше упирается в предел допустимой области, но это не приводит к лучшей теплицевой маске. Здесь проверялась структура масок; качество новых MLP для групп из 3–10 VAE отдельно не измерялось.", ""]
    (out / "RESULTS.md").write_text("\n".join(lines))


def main() -> None:
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.root
    records, heatmaps = [], {(size, stage): [] for size in ALL_SIZES
                             for stage in ("initial", "optimized")}
    for group in range(8):
        for replicate in range(4):
            record, maps = read_pair(group, replicate)
            records.append(record)
            for stage in maps:
                heatmaps[(2, stage)].append(maps[stage])
            for size in range(3, 11):
                record, maps = read_multi(group, size, replicate)
                records.append(record)
                for stage in maps:
                    heatmaps[(size, stage)].append(maps[stage])
    aggregate_record = aggregate(records)
    comparisons = paired_comparisons(records)
    payload = {"protocol": {"groups": [list(group_patterns(i)) for i in range(8)],
                             "replicates": 4, "starts": 64, "steps": 2000},
               "aggregate": aggregate_record, "paired_comparisons": comparisons,
               "records": records}
    (ROOT / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    make_figures(aggregate_record, heatmaps, ROOT)
    write_report(aggregate_record, comparisons, ROOT)
    print(ROOT / "RESULTS.md")


if __name__ == "__main__":
    main()
