from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from pattern.evaluation.decoder_agreement import align_columns

from .common import ideal_mask, load_protocol, pair_dir, write_json


METHODS = (
    "prior", "agreement", "random_latent_pair", "individual_final",
    "individual_best_query", "random_exact_k", "ideal",
)
LABELS = {
    "prior": "VAE prior",
    "agreement": "Decoder agreement",
    "random_latent_pair": "Random latent-pair search",
    "individual_final": "Individual task-z (final)",
    "individual_best_query": "Individual task-z (best query)",
    "random_exact_k": "Random exact-160",
    "ideal": "Ideal support",
}


def describe(values) -> dict:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("report values must be a finite nonempty vector")
    mean = float(array.mean())
    if len(array) == 1:
        low = high = mean
        standard_error = 0.0
    else:
        standard_error = float(array.std(ddof=1) / math.sqrt(len(array)))
        half = float(stats.t.ppf(0.975, len(array) - 1) * standard_error)
        low, high = mean - half, mean + half
    return {"mean": mean, "ci95": [low, high], "standard_error": standard_error,
            "n": len(array), "values": array.tolist()}


def _mean_task(record: dict, method: str, key: str) -> float:
    if key == "gold_iou":
        return float(np.mean([task["structure"][method]["gold_iou_mean"]
                              for task in record["tasks"].values()]))
    return float(np.mean([task["methods"][method][f"{key}_mean"]
                          for task in record["tasks"].values()]))


def aggregate(out: Path) -> dict:
    config, protocol = load_protocol(out)
    records = []
    for pair in config.pairs:
        path = pair_dir(out, pair) / "summary.json"
        if not path.exists():
            raise FileNotFoundError(f"missing pair summary: {path}")
        records.append(json.loads(path.read_text()))
    result = {
        "protocol": protocol,
        "replicates": len(records),
        "independence_unit": "VAE pair",
        "soft_loss": {name: describe([record["soft_loss"][name] for record in records])
                      for name in ("prior", "agreement", "random_latent_pair")},
        "pair_agreement": {},
        "methods": {},
        "comparisons": {},
        "per_task": {},
    }
    for stage in ("prior", "agreement", "random_latent_pair"):
        result["pair_agreement"][stage] = {
            key: describe([record["pair_agreement"][stage][key] for record in records])
            for key in ("iou_mean", "hamming_mean", "hamming_normalized_mean")
        }
        result["pair_agreement"][stage]["exact_fraction"] = describe([
            record["pair_agreement"][stage]["exact_count"] /
            record["pair_agreement"][stage]["count"] for record in records])
    for method in METHODS:
        result["methods"][method] = {
            metric: describe([_mean_task(record, method, metric) for record in records])
            for metric in ("gold_iou", "accuracy", "bce")
        }
    for reference in ("prior", "random_latent_pair", "individual_final", "individual_best_query"):
        result["comparisons"][f"agreement_minus_{reference}"] = {
            metric: describe([_mean_task(record, "agreement", metric) - _mean_task(record, reference, metric)
                              for record in records]) for metric in ("gold_iou", "accuracy", "bce")
        }
    for pattern in protocol["task_split"]["test_patterns"]:
        result["per_task"][pattern] = {}
        for method in METHODS:
            result["per_task"][pattern][method] = {
                metric: describe([
                    record["tasks"][pattern]["structure"][method]["gold_iou_mean"]
                    if metric == "gold_iou" else
                    record["tasks"][pattern]["methods"][method][f"{metric}_mean"]
                    for record in records]) for metric in ("gold_iou", "accuracy")
            }
    best_epochs = []
    val_losses = []
    for pair in config.pairs:
        for seed in pair:
            metadata = json.loads((pair_dir(out, pair) / f"vae_{seed}" / "metadata.json").read_text())
            best_epochs.append(metadata["best_epoch"])
            val_losses.append(metadata["best_val_loss"])
    result["vae_training"] = {"best_epoch": describe(best_epochs), "best_val_loss": describe(val_losses),
                              "models": len(best_epochs)}
    return result


def _metric(value: dict, scale: float = 1.0, digits: int = 4) -> str:
    low, high = value["ci95"]
    return f"{scale * value['mean']:.{digits}f} [{scale * low:.{digits}f}; {scale * high:.{digits}f}]"


def _plots(out: Path, summary: dict) -> None:
    methods = list(METHODS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for axis, metric, title in ((axes[0], "gold_iou", "Gold IoU (post-hoc)"),
                                (axes[1], "accuracy", "Fresh-MLP test accuracy")):
        means = [summary["methods"][method][metric]["mean"] for method in methods]
        errors = [[means[index] - summary["methods"][method][metric]["ci95"][0] for index, method in enumerate(methods)],
                  [summary["methods"][method][metric]["ci95"][1] - means[index] for index, method in enumerate(methods)]]
        axis.barh(range(len(methods)), means, xerr=errors, color="#477b9d", alpha=0.9)
        axis.set_yticks(range(len(methods)), [LABELS[method] for method in methods])
        axis.invert_yaxis()
        axis.set_xlim(0, 1)
        axis.grid(axis="x", alpha=0.2)
        axis.set_title(title)
    fig.suptitle("Pattern-32, k=5: means and pair-level 95% t intervals")
    fig.savefig(out / "comparison.png", dpi=170)
    fig.savefig(out / "comparison.pdf")
    plt.close(fig)

    config, protocol = load_protocol(out)
    pair = config.pairs[0]
    root = pair_dir(out, pair)
    agreement = torch.load(root / "agreement.pt", map_location="cpu", weights_only=True)
    pattern = protocol["task_split"]["test_patterns"][0]
    individual = torch.load(root / "task_z" / f"pattern_{pattern}.pt", map_location="cpu", weights_only=True)
    second = align_columns(agreement["final_masks1"].float(), agreement["final_masks2"].float())
    target = ideal_mask(config)
    shown = min(4, config.n_starts)
    fig, axes = plt.subplots(shown, 5, figsize=(12, 2.2 * shown + 0.5),
                             layout="constrained", squeeze=False)
    for row in range(shown):
        entries = (agreement["initial_masks1"][row], agreement["final_masks1"][row], second[row],
                   individual["best_query"]["masks"][row], target)
        for column, mask in enumerate(entries):
            axes[row, column].imshow(mask, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        axes[row, 0].set_ylabel(f"Start {row}")
    titles = ("Prior VAE 1", "Agreement VAE 1", "Agreement VAE 2\nmatched",
              f"Individual best-query\ntask {pattern}", "Ideal reference")
    for axis, title in zip(axes[0], titles):
        axis.set_title(title, fontsize=10)
    fig.suptitle(f"First four starts of pair {pair[0]}/{pair[1]} (fixed order, no quality selection)")
    fig.savefig(out / "mask_examples.png", dpi=170)
    plt.close(fig)


def _markdown(out: Path, summary: dict) -> None:
    config = summary["protocol"]["config"]
    lines = [
        "# Pattern-32, длина паттерна 5: agreement по независимым VAE",
        "",
        "Автоматическая численная сводка полного запуска. Научную интерпретацию следует добавлять "
        "после проверки логов и provenance-файлов.",
        "",
        "## Протокол",
        "",
        f"Обучено **{config['vae_pairs']} независимых пар VAE** ({2 * config['vae_pairs']} моделей). "
        f"Для каждой пары использовано {config['n_starts']} общих latent-стартов. Вход имеет длину 32, "
        "скрытый слой — 32, паттерн — 5; каждая hard-маска имеет размер 32×32 и ровно 160 активных рёбер.",
        "",
        f"VAE обучаются на top-10% карт из банков по {config['bank_mlps']} MLP для каждой из "
        f"{len(summary['protocol']['task_split']['train_patterns'])} train-задач. "
        f"{len(summary['protocol']['task_split']['test_patterns'])} OOD-паттернов не входят в банки или VAE. "
        f"VAE: {config['vae_epochs']} эпох, latent {config['latent_dim']}, hidden {config['vae_hidden']}, "
        f"beta={config['vae_beta']}; checkpoint выбирается по внутренней map-validation.",
        "",
        f"Agreement не использует задачи или Gold: Adam {config['agreement_steps']} шагов, lr={config['agreement_lr']}, "
        f"temperature={config['temperature']}, radius={config['latent_radius']}. Random latent-pair проверяет "
        f"{config['random_proposals']} состояний на старт. Individual task-z имеет доступ к support/query меткам "
        "конкретной OOD-задачи, но не к финальному test или Gold, поэтому это supervised upper-comparison, "
        "а не равноправный label-free baseline.",
        "",
        "Все ДИ ниже рассчитаны по VAE-парам; latent-старты, OOD-задачи и повторы MLP являются вложенными наблюдениями.",
        "Для честного сравнения с individual task-z структурные и downstream-метрики VAE-зависимых методов "
        "относятся к первому decoder каждой пары; второй decoder входит в метрики pairwise agreement.",
        "",
        "## Основные результаты",
        "",
        "| Метод | Gold IoU, 95% ДИ | Fresh-MLP accuracy, 95% ДИ | BCE, 95% ДИ |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        values = summary["methods"][method]
        lines.append(f"| {LABELS[method]} | {_metric(values['gold_iou'])} | "
                     f"{_metric(values['accuracy'], 100, 2)}% | {_metric(values['bce'])} |")
    lines += [
        "",
        "## Сходимость двух decoder",
        "",
        "| Стадия | Soft MSE | Hard pair IoU | Exact hard agreement | Normalized Hamming |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage, label in (("prior", "Prior"), ("agreement", "Adam agreement"),
                         ("random_latent_pair", "Random latent-pair")):
        pair = summary["pair_agreement"][stage]
        lines.append(f"| {label} | {_metric(summary['soft_loss'][stage], digits=7)} | "
                     f"{_metric(pair['iou_mean'])} | {_metric(pair['exact_fraction'])} | "
                     f"{_metric(pair['hamming_normalized_mean'], digits=6)} |")
    lines += ["", "## Парные разности agreement", "",
              "| Сравнение | Δ Gold IoU | Δ accuracy, п.п. |", "|---|---:|---:|"]
    for name, values in summary["comparisons"].items():
        lines.append(f"| {name.replace('_', ' ')} | {_metric(values['gold_iou'])} | "
                     f"{_metric(values['accuracy'], 100, 3)} |")
    lines += [
        "",
        "![Сравнение методов](comparison.png)",
        "",
        "## Примеры масок",
        "",
        "Показаны первые четыре старта первой пары в заранее фиксированном порядке. Ideal используется только "
        "для визуального post-hoc сравнения. Individual-маска зависит от указанной held-out задачи; agreement-маска — нет.",
        "",
        "![Примеры масок и ideal](mask_examples.png)",
        "",
        "## Ограничения интерпретации",
        "",
        "- Individual task-z использует метки и существенно больше вычислений, чем agreement.",
        "- Random latent-pair выбирается по agreement loss; это matched control для pairwise поиска, но не для task-z.",
        "- Gold IoU и final test accuracy не участвуют ни в выборе agreement, ни в выборе его итерации.",
        "- Интервалы условны на одном task split и одном общем банке importance-карт.",
        "",
        "Полные числа находятся в `summary.json`; pair-level результаты, checkpoints, маски, истории и hashes — в `pairs/`.",
        "",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = aggregate(args.out)
    write_json(args.out / "summary.json", summary)
    _plots(args.out, summary)
    _markdown(args.out, summary)
    print(f"[k5-report] {args.out / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
