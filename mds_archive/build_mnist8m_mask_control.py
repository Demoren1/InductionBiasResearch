"""Summarize the paired selected-bank versus random-mask retraining test."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path("pattern/outputs/mnist8m_raw_mlp_bce/mask_control")
ASSET = Path("mds/assets/2026-09-26/mnist8m_mask_control.png")
REPORT = Path("mds/MNIST8M_MASK_CONTROL_2026-09-26.md")
METRICS = (("balanced_accuracy", "Сбалансированная точность", 100, "п. п."),
           ("balanced_bce", "BCE", 1, ""))


def interval(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    """Pair bootstrap, conditional on this image split and these restarts."""
    rng = np.random.default_rng(seed)
    sample = rng.integers(len(values), size=(5000, len(values)))
    means = values[sample].mean(1)
    return (float(values.mean()), *np.quantile(means, (.025, .975)).tolist())


def main() -> None:
    results = [torch.load(ROOT / f"task{task}.pt", map_location="cpu",
                          weights_only=True) for task in (0, 1)]
    for result in results:
        assert result["method_order"] == ["selected_bank", "fresh_random"]
    summary = {}
    for metric, _, scale, _ in METRICS:
        rows = []
        for result in results:
            values = result[metric].numpy()
            selected = float(values[0].mean() * scale)
            random = float(values[1].mean() * scale)
            differences = (values[0] - values[1]).mean(1) * scale
            estimate = interval(differences, 35000 + int(result["digit"]))
            rows.append((selected, random, estimate))
        combined_diff = np.concatenate([
            (result[metric][0] - result[metric][1]).numpy().mean(1) * scale
            for result in results])
        combined = interval(combined_diff, 36000)
        summary[metric] = (rows, combined)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
    for ax, (metric, title, _, unit) in zip(axes, METRICS):
        rows, combined = summary[metric]
        estimates = [row[2] for row in rows] + [combined]
        plot_scale = 1000 if metric == "balanced_bce" else 1
        center = np.array([x[0] for x in estimates]) * plot_scale
        lower = center - np.array([x[1] for x in estimates]) * plot_scale
        upper = np.array([x[2] for x in estimates]) * plot_scale - center
        ax.errorbar(center, range(3), xerr=np.stack((lower, upper)), fmt="o",
                    capsize=3, color="#246399")
        ax.axvline(0, color="black", linewidth=.8)
        ax.set_yticks(range(3), ["Цифра 3", "Цифра 8", "Среднее"])
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Лучшие − случайные, BCE × 10⁻³" if plot_scale == 1000
                      else f"Лучшие − случайные, {unit}")
        ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    ASSET.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSET, dpi=170)
    plt.close(fig)

    n = results[0]["settings"]["pairs"]
    repeats = results[0]["settings"]["repeats"]
    lines = ["# MNIST8m: несут ли лучшие маски банка полезный сигнал?", "",
             "Для цифр 3 и 8 сравнили по 410 масок из лучших 10% банка с "
             "410 новыми случайными масками той же плотности (20%). В каждой "
             "паре MLP стартовали с одинаковых весов, получали одинаковые "
             "обучающие пакеты и обучались до ранней остановки. Маски были "
             f"фиксированы. Для каждого сравнения — {repeats} новых старта. "
             "Тестовые изображения не использовались при отборе масок.", "",
             "| Задача | Лучшая маска: точность | Случайная: точность | "
             "Разница, п. п. [95% ДИ] |",
             "|---|---:|---:|---:|"]
    for result, row in zip(results, summary["balanced_accuracy"][0]):
        selected, random, (mean, low, high) = row
        lines.append(f"| {result['digit']} против остальных | "
                     f"{selected / 100:.4f} | {random / 100:.4f} | "
                     f"{mean:+.3f} [{low:+.3f}; {high:+.3f}] |")
    mean, low, high = summary["balanced_accuracy"][1]
    lines.append(f"| Среднее по двум задачам | — | — | "
                 f"{mean:+.3f} [{low:+.3f}; {high:+.3f}] |")
    lines.extend(["", "| Задача | Лучшая маска: BCE | Случайная: BCE | "
                  "Разница [95% ДИ] |", "|---|---:|---:|---:|"])
    for result, row in zip(results, summary["balanced_bce"][0]):
        selected, random, (mean, low, high) = row
        lines.append(f"| {result['digit']} против остальных | "
                     f"{selected:.5f} | {random:.5f} | "
                     f"{mean:+.5f} [{low:+.5f}; {high:+.5f}] |")
    mean, low, high = summary["balanced_bce"][1]
    lines.append(f"| Среднее по двум задачам | — | — | "
                 f"{mean:+.5f} [{low:+.5f}; {high:+.5f}] |")
    lines.extend(["", "![Парные различия](assets/2026-09-26/mnist8m_mask_control.png)",
                  "", "Доверительные интервалы получены парным бутстрэпом "
                  "по маскам; они относятся к этому фиксированному набору "
                  "изображений и двум задачам, а не ко всем возможным цифрам.",
                  "", "**Вывод.** В этом контроле маски от лучших MLP не дали "
                  "заметного преимущества перед случайными масками после "
                  "нового обучения весов. Значит, прежний отбор по качеству "
                  "MLP не выявил воспроизводимо лучших поддержек масок; "
                  "хорошее качество исходных MLP могло зависеть от их весов.",
                  "", f"Пар масок на задачу: {n}; повторов обучения: {repeats}. "
                  f"Шагов: {results[0]['settings']['actual_steps']} для 3, "
                  f"{results[1]['settings']['actual_steps']} для 8.", "",
                  "[Код проверки](../pattern/mnist8m_bank_mask_control.py) · "
                  "[сырые результаты](../pattern/outputs/mnist8m_raw_mlp_bce/mask_control/)", ""])
    REPORT.write_text("\n".join(lines))
    print(REPORT)


if __name__ == "__main__":
    main()
