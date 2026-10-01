"""Aggregate MNIST8m importance-map agreement runs into a short report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from pattern.mnist8m_importance_bce import exact_masks, topk
from pattern.models.cvae import CVAE


ROOT = Path("pattern/outputs/mnist8m_importance_bce")
ASSETS = Path("mds/assets/2026-09-26")
REPORT = Path("mds/MNIST8M_IMPORTANCE_BCE_2026-09-26.md")
PAIRS = ("pair01", "pair38", "pair49", "pair56")
KEYS = (
    ("random", "Случайная, средняя"),
    ("random_best16", "Лучшая из 16 случайных"),
    ("best_bank_per_task", "Лучшая маска банка"),
    ("single_optimized_per_task", "Один VAE на задачу"),
    ("consensus_lambda0", "Общая без agreement"),
    ("consensus_lambda1", "Agreement λ=1"),
    ("consensus_lambda10", "Agreement λ=10"),
    ("dense", "Плотный слой"),
)


def pair_iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.logical_and(a, b).sum()
    return float(intersection / (1200 - intersection))


def support_overlap(maps: np.ndarray) -> tuple[float, float]:
    reference = maps[0].astype(bool)
    raw, aligned = [], []
    for subject in maps[1:]:
        subject = subject.astype(bool)
        raw.append(pair_iou(reference, subject))
        rows, columns = linear_sum_assignment(
            -(reference.T.astype(int) @ subject.astype(int)))
        aligned.append(pair_iou(reference, subject[:, columns[np.argsort(rows)]]))
    return float(np.mean(raw)), float(np.mean(aligned))


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    summaries = [json.loads((ROOT / pair / "summary.json").read_text())
                 for pair in PAIRS]

    def values(metric: str, key: str) -> np.ndarray:
        return np.asarray([next(row[key] for row in summary
                                if row["metric"] == metric)
                           for summary in summaries])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric, title in (
        (axes[0], "balanced_accuracy", "Сбалансированная точность ↑"),
        (axes[1], "bce", "BCE ↓"),
    ):
        positions = np.arange(len(KEYS))
        means = np.asarray([values(metric, key).mean() for key, _ in KEYS])
        spread = np.asarray([values(metric, key).std(ddof=1) for key, _ in KEYS])
        ax.errorbar(positions, means, yerr=spread, fmt="o", capsize=4)
        ax.set_xticks(positions, [label for _, label in KEYS], rotation=50, ha="right")
        ax.set_ylim((means - spread).min() - .005,
                    (means + spread).max() + .005)
        ax.grid(axis="y", alpha=.2)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(ASSETS / "mnist8m_importance_bce_quality.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for pair in PAIRS:
        i = PAIRS.index(pair)
        ax.plot([0, 1, 10], [values("balanced_accuracy", f"consensus_lambda{c}")[i]
                             for c in (0, 1, 10)], marker="o", label=pair[4:])
    ax.axhline(values("balanced_accuracy", "random_best16").mean(),
               color="gray", linestyle="--", label="лучшая из 16 случайных")
    ax.axhline(values("balanced_accuracy", "dense").mean(),
               color="black", linestyle=":", label="плотный слой")
    ax.set_xticks((0, 1, 10))
    ax.set_xlabel("Коэффициент agreement λ")
    ax.set_ylabel("Сбалансированная точность")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "mnist8m_importance_bce_coefficients.png", dpi=170)
    plt.close(fig)

    selected_overlaps, random_overlaps, recon_ious = [], [], []
    hard_ious = {coefficient: [] for coefficient in (0, 1, 10)}
    for pair in PAIRS:
        directory = ROOT / pair
        for task in (0, 1):
            bank = torch.load(directory / f"bank_task{task}.pt", map_location="cpu",
                              weights_only=True)
            selected = bank["masks"].numpy()
            random = exact_masks(len(selected), 12345 + task,
                                 torch.device("cpu")).numpy()
            selected_overlaps.append(support_overlap(selected))
            random_overlaps.append(support_overlap(random))
            payload = torch.load(directory / f"vae_task{task}.pt", map_location="cpu",
                                 weights_only=True)
            model = CVAE(**payload["config"])
            model.load_state_dict(payload["model"])
            model.eval()
            with torch.no_grad():
                x = bank["importance"].flatten(1)
                mu, _ = model.encode(x, x.new_zeros(len(x), 0))
                decoded = model.decode(mu, x.new_zeros(len(x), 0))
                source = topk(x.reshape(-1, 100, 30)).flatten(1)
                reproduced = topk(decoded.reshape(-1, 100, 30)).flatten(1)
                overlap = (source * reproduced).sum(1)
                recon_ious.append(float((overlap / (1200 - overlap)).mean()))
        for coefficient in (0, 1, 10):
            search = torch.load(directory / f"search_lambda{coefficient:g}.pt",
                                map_location="cpu", weights_only=True)
            logits = search["logits"][:, search["chosen_start"]]
            masks = topk(logits).bool().numpy()
            hard_ious[coefficient].append(pair_iou(masks[0], masks[1]))

    directory = ROOT / "pair38"
    banks = [torch.load(directory / f"bank_task{task}.pt", map_location="cpu",
                        weights_only=True) for task in (0, 1)]
    search = torch.load(directory / "search_lambda10.pt", map_location="cpu",
                        weights_only=True)
    raw = search["logits"][:, search["chosen_start"]]
    logit_low, logit_high = float(raw.min()), float(raw.max())
    fig, axes = plt.subplots(2, 3, figsize=(9, 8))
    for task in (0, 1):
        axes[task, 0].imshow(banks[task]["importance"][0], aspect="auto",
                             vmin=0, vmax=1, cmap="viridis")
        axes[task, 1].imshow(raw[task], aspect="auto",
                             vmin=logit_low, vmax=logit_high, cmap="viridis")
        axes[task, 2].imshow(topk(raw[task]), aspect="auto",
                             vmin=0, vmax=1, cmap="gray_r")
        axes[task, 0].set_ylabel(f"Цифра {banks[task]['digit']}")
    for ax, title in zip(axes[0], ("Карта из банка", "Логиты после поиска",
                                   "Маска top-600")):
        ax.set_title(title)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(ASSETS / "mnist8m_importance_bce_maps.png", dpi=170)
    plt.close(fig)

    lines = ["# MNIST8m: agreement VAE на importance maps", "",
             "Четыре пары цифр: 0/1, 3/8, 4/9, 5/6. Для каждой цифры "
             "обучены 4096 MLP со случайными масками (600 из 3000 связей) "
             "на задаче «эта цифра против остальных». MLP обучались по BCE "
             "с проверкой сходимости. У лучших 10% по validation BCE сохранены "
             "нормированные карты |W·M|/max(|W·M|). Для каждой цифры на её "
             "картах обучен отдельный VAE с ранней остановкой.", "",
             "При поиске VAE заморожены. Оптимизируются два z и две новые "
             "головы: BCE каждой задачи плюс MSE выровненных логитов декодеров "
             "с коэффициентом λ. Маски — top-600; выравнивание столбцов "
             "фиксируется в начале поиска. Каждая итоговая маска проверена "
             "двумя новыми обучениями головы на изображениях из отдельного "
             "тестового блока MNIST8m.", "",
             "| Маска | Сбалансированная точность ↑ | BCE ↓ |",
             "|---|---:|---:|"]
    for key, label in KEYS:
        accuracy = values("balanced_accuracy", key)
        bce = values("bce", key)
        lines.append(f"| {label} | {accuracy.mean():.4f} ± {accuracy.std(ddof=1):.4f} "
                     f"| {bce.mean():.4f} ± {bce.std(ddof=1):.4f} |")
    lines.extend(["", "Разброс — между четырьмя парами цифр. Сбалансированная "
                  "точность придаёт одинаковый вес положительному и "
                  "отрицательному классу. Лучшая из 16 случайных масок выбрана "
                  "только по валидации. Плотный слой содержит 3000 связей "
                  "против 600 в масках.", "",
                  "![Качество](assets/2026-09-26/mnist8m_importance_bce_quality.png)", "",
                  "![Коэффициент agreement](assets/2026-09-26/mnist8m_importance_bce_coefficients.png)", "",
                  "![Примеры importance maps и масок](assets/2026-09-26/mnist8m_importance_bce_maps.png)", "",
                  "| Пара цифр | Лучшая случайная | Без agreement | λ=1 | λ=10 | Плотный |",
                  "|---|---:|---:|---:|---:|---:|"])
    for i, pair in enumerate(PAIRS):
        lines.append(f"| {pair[4:]} | " + " | ".join(
            f"{values('balanced_accuracy', key)[i]:.4f}"
            for key in ("random_best16", "consensus_lambda0",
                        "consensus_lambda1", "consensus_lambda10", "dense")) + " |")
    lines.extend(["", "BCE на тех же четырёх парах:", "",
                  "| Пара цифр | Лучшая случайная | Без agreement | λ=1 | λ=10 | Плотный |",
                  "|---|---:|---:|---:|---:|---:|"])
    for i, pair in enumerate(PAIRS):
        lines.append(f"| {pair[4:]} | " + " | ".join(
            f"{values('bce', key)[i]:.4f}"
            for key in ("random_best16", "consensus_lambda0",
                        "consensus_lambda1", "consensus_lambda10", "dense")) + " |")
    selected = np.mean(selected_overlaps, axis=0)
    random = np.mean(random_overlaps, axis=0)
    lines.extend(["", "**Проверка карт.** Средний IoU поддержек отобранных "
                  f"карт с первой картой: {selected[0]:.3f} против "
                  f"{random[0]:.3f} у случайных; "
                  f"после оптимальной перестановки столбцов {selected[1]:.3f} "
                  f"против {random[1]:.3f}. IoU top-600 при реконструкции "
                  f"VAE: {np.mean(recon_ious):.3f}. IoU масок двух VAE после "
                  "поиска: " + ", ".join(
                      f"λ={coefficient}: {np.mean(hard_ious[coefficient]):.3f}"
                      for coefficient in (0, 1, 10)) + ".", ""])
    random_acc = values("balanced_accuracy", "random_best16").mean()
    dense_acc = values("balanced_accuracy", "dense").mean()
    agreement_acc = values("balanced_accuracy", "consensus_lambda1").mean()
    random_bce = values("bce", "random_best16").mean()
    dense_bce = values("bce", "dense").mean()
    agreement_bce = values("bce", "consensus_lambda1").mean()
    lines.extend(["**Вывод.** При λ=1 средняя точность отличается от лучшей "
                  f"случайной маски на {agreement_acc-random_acc:+.4f}, от "
                  f"плотного слоя на {agreement_acc-dense_acc:+.4f}. Средняя "
                  f"BCE ниже на {random_bce-agreement_bce:.4f} и "
                  f"{dense_bce-agreement_bce:.4f} соответственно, но выигрыш "
                  "меняется между парами цифр. Устойчивого улучшения точности "
                  "не обнаружено. VAE почти не восстанавливают top-600 "
                  "поддержки исходных карт; это ограничивает вывод о пользе "
                  "agreement. Энкодер изображений был заморожен.", "",
                  "[Код эксперимента](../pattern/mnist8m_importance_bce.py) · "
                  "[сырые результаты](../pattern/outputs/mnist8m_importance_bce/)", ""])
    REPORT.write_text("\n".join(lines))
    print(REPORT)


if __name__ == "__main__":
    main()
