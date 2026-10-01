"""Report the fixed-5% MNIST8m importance-map VAE experiment."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import pattern.mnist8m_raw_mlp_bce as exp
from pattern.models.cvae import CVAE


ROOT = Path("pattern/outputs/mnist8m_raw_mlp_bce/pair38_5pct")
ASSETS = Path("mds/assets/2026-09-26")
REPORT = Path("mds/MNIST8M_RAW_MLP_5PCT_2026-09-26.md")
METHODS = (
    ("random", "Средняя случайная маска"),
    ("random_best16", "Лучшая из 16 случайных"),
    ("best_bank_per_task", "Карта лучшего MLP"),
    ("mean_bank_per_task", "Средняя карта банка"),
    ("vae_zero_per_task", "VAE, z=0"),
    ("single_optimized_per_task", "Каждый VAE, BCE своей задачи"),
    ("consensus_lambda0", "Общая маска, своя BCE, λ=0"),
    ("consensus_lambda1", "Общая маска, своя BCE, λ=1"),
    ("consensus_lambda10", "Общая маска, своя BCE, λ=10"),
    ("shared_consensus_lambda0", "Общая маска, обе BCE, λ=0"),
    ("shared_consensus_lambda1", "Общая маска, обе BCE, λ=1"),
    ("shared_consensus_lambda10", "Общая маска, обе BCE, λ=10"),
    ("dense", "Плотный слой"),
)


def paired_iou(left: torch.Tensor, right: torch.Tensor) -> float:
    intersection = (left.bool() & right.bool()).sum().item()
    return intersection / (2 * exp.K - intersection)


def reconstruction_rows() -> list[tuple[int, float, float, float, float, int]]:
    result = []
    for task in (0, 1):
        bank = torch.load(ROOT / f"bank_task{task}.pt", map_location="cpu",
                          weights_only=True)
        payload = torch.load(ROOT / f"vae_task{task}.pt", map_location="cpu",
                             weights_only=True)
        maps = bank["importance"].flatten(1)
        generator = torch.Generator().manual_seed(3130 + task)
        order = torch.randperm(len(maps), generator=generator)
        n_val = max(32, round(.15 * len(maps)))
        train, validation = maps[order[n_val:]], maps[order[:n_val]]
        model = CVAE(**payload["config"])
        model.load_state_dict(payload["model"])
        model.eval()
        with torch.no_grad():
            mu, _ = model.encode(validation,
                                 validation.new_zeros(len(validation), 0))
            logits = model.decode(mu,
                                  validation.new_zeros(len(validation), 0))
            target = exp.topk(validation.reshape(-1, exp.FEATURES, exp.HIDDEN))
            predicted = exp.topk(logits.reshape(-1, exp.FEATURES, exp.HIDDEN))
            intersection = (target * predicted).flatten(1).sum(1)
            vae_iou = float((intersection / (2 * exp.K - intersection)).mean())
            template = exp.topk(train.mean(0).reshape(exp.FEATURES, exp.HIDDEN))
            intersection = (target * template).flatten(1).sum(1)
            mean_iou = float((intersection / (2 * exp.K - intersection)).mean())
            vae_bce = float(F.binary_cross_entropy_with_logits(logits, validation))
            template_values = train.mean(0).clamp(1e-6, 1 - 1e-6)
            mean_bce = float(F.binary_cross_entropy(
                template_values.expand_as(validation), validation))
        result.append((bank["digit"], vae_iou, mean_iou, vae_bce,
                       mean_bce, len(payload["history"])))
    return result


def main() -> None:
    exp.K = round(.05 * exp.FEATURES * exp.HIDDEN)
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name in ("quality.png", "pixel_maps.png"):
        shutil.copy2(ROOT / name, ASSETS / f"mnist8m_5pct_{name}")
    evaluation = torch.load(ROOT / "evaluation.pt", map_location="cpu",
                            weights_only=True)
    names = evaluation["names"]
    random = [i for i, name in enumerate(names) if name.startswith("random_")]
    best_random = random[int(evaluation["validation_bce"][:, random]
                             .mean((0, 2)).argmin())]
    coverage_rows = []
    fig, axes = plt.subplots(2, 4, figsize=(11.5, 5.4))
    for task, digit in enumerate(evaluation["pair"]):
        mask_keys = (f"bank_task{task}_best", f"bank_task{task}_mean",
                     f"vae{task}_lambda0", names[best_random])
        covered = []
        for column, name in enumerate(mask_keys):
            mask = evaluation["masks"][names.index(name)]
            covered.append(int(mask.bool().any(-1).sum()))
            projected = mask.float().mean(-1).reshape(28, 28)
            image = axes[task, column].imshow(projected, cmap="viridis",
                                              vmin=0, vmax=.5)
            axes[task, column].set_xticks([])
            axes[task, column].set_yticks([])
            if column == 0:
                axes[task, column].set_ylabel(f"Цифра {digit}")
        coverage_rows.append((digit, *covered))
    for ax, title in zip(axes[0], ("Карта MLP", "Средняя карта",
                                    "VAE с найденным z", "Случайная")):
        ax.set_title(title)
    fig.subplots_adjust(left=.06, right=.86, top=.88, bottom=.04,
                        wspace=.1, hspace=.12)
    colorbar_axis = fig.add_axes((.89, .18, .015, .64))
    fig.colorbar(image, cax=colorbar_axis,
                 label="Доля связей на пиксель (цвет обрезан на 0.5)")
    fig.savefig(ASSETS / "mnist8m_5pct_mask_coverage.png", dpi=170)
    plt.close(fig)

    def metric(key: str, method: str) -> float:
        values = evaluation["metrics"][key]
        if method == "random":
            return float(values[:, random].mean())
        if method == "random_best16":
            return float(values[:, best_random].mean())
        special = {
            "best_bank_per_task": "bank_task{task}_best",
            "mean_bank_per_task": "bank_task{task}_mean",
            "vae_zero_per_task": "vae{task}_z0",
            "single_optimized_per_task": "vae{task}_lambda0",
        }
        if method in special:
            return float(torch.stack([
                values[task, names.index(special[method].format(task=task))].mean()
                for task in (0, 1)]).mean())
        return float(values[:, names.index(method)].mean())

    agreement_rows = []
    for objective, prefix in (("Своя BCE", "search"),
                              ("Обе BCE", "search_shared")):
        for coefficient in (0, 1, 10):
            payload = torch.load(ROOT / f"{prefix}_lambda{coefficient}.pt",
                                 map_location="cpu", weights_only=True)
            raw = payload["logits"][:, payload["chosen_start"]]
            left, right = raw.flatten(1)
            corr = float(torch.corrcoef(torch.stack((left, right)))[0, 1])
            agreement_rows.append((objective, coefficient,
                                   float((left - right).square().mean()), corr,
                                   paired_iou(exp.topk(raw[0]), exp.topk(raw[1])),
                                   payload["steps"], payload["plateau"]))

    rows = reconstruction_rows()
    steps = evaluation["settings"]["actual_steps"]
    repeats = evaluation["settings"]["repeats"]
    late_best = float((evaluation["best_step"] > .9 * steps).float().mean())
    lines = ["# MNIST8m: VAE и agreement на масках 5%", "",
             "Те же задачи 3 и 8 против остальных и та же разметка MNIST8m. "
             "В исходном банке для каждой цифры обучены 4096 MLP с 20% "
             "случайных связей. Выбраны лучшие 10% по BCE на валидации; "
             "из их importance maps оставлены top-5% сильнейших связей "
             f"({exp.K} из {exp.FEATURES * exp.HIDDEN}). На этих картах "
             "заново обучены два VAE с ранней остановкой. При поиске VAE "
             "заморожены; оптимизируются z и MLP по BCE своей либо обеих "
             "задач и λ·MSE выровненных логитов декодеров. Маски получены "
             "по top-K и проверены после нового обучения MLP на отдельном "
             "тестовом блоке.", "",
             "| Маска | Сбалансированная точность ↑ | BCE ↓ |",
             "|---|---:|---:|"]
    for key, label in METHODS:
        lines.append(f"| {label} | {metric('balanced_accuracy', key):.4f} | "
                     f"{metric('bce', key):.4f} |")
    lines.extend(["", f"Средние по двум задачам и {repeats} новым обучениям "
                  f"MLP; лучший случайный вариант выбран по валидации. "
                  f"Оценка дошла до лимита {steps} шагов, но только "
                  f"{100 * late_best:.1f}% моделей нашли лучший валидационный "
                  "чекпоинт в последние 10% шагов.", "",
                  "![Сравнение качества]"
                  "(assets/2026-09-26/mnist8m_5pct_quality.png)", "",
                  "![Карты по пикселям]"
                  "(assets/2026-09-26/mnist8m_5pct_pixel_maps.png)", "",
                  "Карты на рисунке усреднены по 64 скрытым нейронам для "
                  "каждого пикселя; полные матрицы 784×64 сохранены в "
                  "`evaluation.pt`.", "",
                  "| Цифра | Пикселей со связью: карта MLP | Средняя карта | "
                  "VAE с найденным z | Случайная |",
                  "|---:|---:|---:|---:|---:|"])
    for digit, direct, mean, vae, random_coverage in coverage_rows:
        lines.append(f"| {digit} | {direct} | {mean} | {vae} | "
                     f"{random_coverage} |")
    lines.extend(["", "![Покрытие пикселей масками]"
                  "(assets/2026-09-26/mnist8m_5pct_mask_coverage.png)", "",
                  "| Цифра | IoU VAE на отложенных картах | IoU средней карты | "
                  "BCE VAE | BCE средней карты | Эпох VAE |",
                  "|---:|---:|---:|---:|---:|---:|"])
    for digit, vae_iou, mean_iou, vae_bce, mean_bce, epochs in rows:
        lines.append(f"| {digit} | {vae_iou:.3f} | {mean_iou:.3f} | "
                     f"{vae_bce:.4f} | {mean_bce:.4f} | {epochs} |")
    lines.extend(["", "Случайный IoU при той же плотности — 0.026.", "",
                  "| BCE каждого z | λ | MSE логитов ↓ | Корреляция логитов ↑ | "
                  "IoU масок VAE ↑ | Шагов | Остановка |",
                  "|---|---:|---:|---:|---:|---:|---|"])
    for objective, coefficient, mse, corr, iou, count, plateau in agreement_rows:
        lines.append(f"| {objective} | {coefficient} | {mse:.5f} | "
                     f"{corr:.3f} | {iou:.3f} | {count} | "
                     f"{'по плато' if plateau else 'лимит'} |")
    direct_accuracy = metric("balanced_accuracy", "best_bank_per_task")
    random_accuracy = metric("balanced_accuracy", "random")
    vae_accuracy = metric("balanced_accuracy", "single_optimized_per_task")
    lines.extend(["", "**Вывод.** Прямые top-5% маски сохранили преимущество "
                  f"над случайными ({direct_accuracy:.4f} против "
                  f"{random_accuracy:.4f}), но найденные через VAE маски "
                  f"дали {vae_accuracy:.4f} и ни один вариант agreement "
                  "не превзошёл случайную маску. VAE воспроизводит отложенные "
                  "карты примерно на уровне средней карты банка и сильно "
                  "сужает покрытие входных пикселей. Последующая проверка "
                  "показала, что ограничение числа связей на пиксель при "
                  "неизменных VAE и z заметно повышает качество. Это указывает "
                  "на глобальный top-K как на существенную часть проблемы. "
                  "[Таблицы и графики проверки покрытия]"
                  "(MNIST8M_COVERAGE_DIAGNOSTIC_2026-09-26.md).", "",
                  "[Код эксперимента](../pattern/mnist8m_raw_mlp_bce.py) · "
                  "[подготовка банка](../pattern/prepare_mnist8m_5pct.py) · "
                  "[сырые результаты](../pattern/outputs/mnist8m_raw_mlp_bce/pair38_5pct/) "
                  "· [дальнейшая проверка 16/32 нейронов и 2%]"
                  "(MNIST8M_WIDTH_DENSITY_2026-09-26.md)", ""])
    REPORT.write_text("\n".join(lines))
    print(REPORT)


if __name__ == "__main__":
    main()
