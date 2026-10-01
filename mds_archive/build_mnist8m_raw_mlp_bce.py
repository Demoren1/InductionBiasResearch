"""Summarize the direct pixel-to-MLP MNIST8m agreement pilot."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pattern.mnist8m_raw_mlp_bce import FEATURES, HIDDEN, K, topk
from pattern.models.cvae import CVAE


ROOT = Path("pattern/outputs/mnist8m_raw_mlp_bce/pair38")
ASSETS = Path("mds/assets/2026-09-26")
REPORT = Path("mds/MNIST8M_RAW_MLP_BCE_2026-09-26.md")
METHODS = (("random_best16", "Лучшая из 16 случайных"),
           ("best_bank_per_task", "Лучшая маска банка"),
           ("single_optimized_per_task", "Один VAE на задачу"),
           ("consensus_lambda0", "Своя задача, λ=0"),
           ("consensus_lambda1", "Своя задача, λ=1"),
           ("consensus_lambda10", "Своя задача, λ=10"),
           ("shared_consensus_lambda0", "Обе задачи, λ=0"),
           ("shared_consensus_lambda1", "Обе задачи, λ=1"),
           ("shared_consensus_lambda10", "Обе задачи, λ=10"),
           ("shared_individual_mean", "Отдельные маски обеих задач"),
           ("dense", "Плотный слой"))


def pair_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    intersection = (a.bool() & b.bool()).sum().item()
    return intersection / (2 * K - intersection)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name in ("quality.png", "pixel_maps.png"):
        shutil.copy2(ROOT / name, ASSETS / f"mnist8m_raw_mlp_{name}")
    summary = {row["metric"]: row for row in json.loads((ROOT / "summary.json").read_text())}
    evaluation = torch.load(ROOT / "evaluation.pt", map_location="cpu", weights_only=True)
    repeats = evaluation["settings"]["repeats"]
    names = evaluation["names"]
    random_indices = [i for i, name in enumerate(names) if name.startswith("random_")]
    best_random = random_indices[int(evaluation["validation_bce"][:, random_indices]
                                     .mean((0, 2)).argmin())]

    def task_metric(metric: str, task: int, key: str) -> float:
        values = evaluation["metrics"][metric]
        if key == "random_best16":
            return float(values[task, best_random].mean())
        if key == "best_bank_per_task":
            return float(values[task, names.index(f"bank_task{task}_best")].mean())
        if key == "single_optimized_per_task":
            return float(values[task, names.index(f"vae{task}_lambda0")].mean())
        if key == "shared_individual_mean":
            return float(torch.stack([
                values[task, names.index(f"shared_vae{source}_lambda1")].mean()
                for source in (0, 1)]).mean())
        return float(values[task, names.index(key)].mean())

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
    for ax, metric, ylabel in ((axes[0], "balanced_accuracy", "Точность ↑"),
                               (axes[1], "bce", "BCE ↓")):
        for task, digit in enumerate(evaluation["pair"]):
            values = [task_metric(metric, task, key) for key, _ in METHODS]
            ax.plot(range(len(METHODS)), values, marker="o", label=f"цифра {digit}")
        ax.set_xticks(range(len(METHODS)), [label for _, label in METHODS],
                      rotation=50, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(ASSETS / "mnist8m_raw_mlp_per_digit.png", dpi=170)
    plt.close(fig)

    bank_rows, recon_ious = [], []
    for task in (0, 1):
        bank = torch.load(ROOT / f"bank_task{task}.pt", map_location="cpu",
                          weights_only=True)
        loss = bank["all_validation_bce"]
        bank_rows.append((bank["digit"], float(loss.mean()),
                          float(loss[bank["selected_indices"]].mean()),
                          len(bank["selected_indices"])))
        payload = torch.load(ROOT / f"vae_task{task}.pt", map_location="cpu",
                             weights_only=True)
        model = CVAE(**payload["config"])
        model.load_state_dict(payload["model"])
        model.eval()
        with torch.no_grad():
            x = bank["importance"].flatten(1)
            mu, _ = model.encode(x, x.new_zeros(len(x), 0))
            decoded = model.decode(mu, x.new_zeros(len(x), 0))
            source = topk(x.reshape(-1, FEATURES, HIDDEN)).flatten(1)
            reproduced = topk(decoded.reshape(-1, FEATURES, HIDDEN)).flatten(1)
            overlap = (source * reproduced).sum(1)
            recon_ious.append(float((overlap / (2 * K - overlap)).mean()))
    agreement_rows = []
    for objective, prefix in (("Своя задача", "search"),
                              ("Обе задачи", "search_shared")):
        for coefficient in (0, 1, 10):
            payload = torch.load(ROOT / f"{prefix}_lambda{coefficient:g}.pt",
                                 map_location="cpu", weights_only=True)
            raw = payload["logits"][:, payload["chosen_start"]]
            hard = topk(raw).bool()
            agreement_rows.append((objective, coefficient,
                                   float((raw[0] - raw[1]).square().mean()),
                                   pair_iou(hard[0], hard[1]),
                                   payload["steps"]))

    lines = ["# MNIST8m: простой MLP на пикселях", "",
             "Пара цифр 3/8, две задачи «цифра против остальных». Каждый MLP "
             "получает 784 пикселя напрямую: один скрытый слой из 64 нейронов "
             "и бинарная голова. В банке каждой задачи обучены 4096 MLP со "
             "случайными масками (" + str(K) + " из " + str(FEATURES * HIDDEN) +
             " связей). Ранжирование идёт по BCE на 512 сбалансированных "
             "валидационных изображениях для каждой цифры. У лучших 10% "
             "сохранены нормированные "
             "importance maps |W·M|/max(|W·M|); на картах каждой цифры обучен "
             "свой VAE с ранней остановкой.", "",
             "При поиске оба VAE заморожены. Сравниваются два варианта: "
             "каждый z обучается по BCE своей задачи либо каждый z обучается "
             "по BCE обеих задач с отдельным MLP для каждой. В обоих "
             "вариантах добавляется λ·MSE выровненных логитов декодеров. "
             "Маски образуются по top-K. "
             f"Итоговое качество измерено после {repeats} новых обучений MLP "
             "с фиксированной маской на "
             "отдельных тестовых изображениях: по 500 изображений каждой "
             "цифры, классы в метриках имеют равный вес.", "",
             "| Маска | Сбалансированная точность ↑ | BCE ↓ |",
             "|---|---:|---:|"]
    for key, label in METHODS:
        lines.append(f"| {label} | {summary['balanced_accuracy'][key]:.4f} | "
                     f"{summary['bce'][key]:.4f} |")
    lines.extend(["", "![Сравнение качества](assets/2026-09-26/mnist8m_raw_mlp_quality.png)",
                  "", "![По каждой цифре](assets/2026-09-26/mnist8m_raw_mlp_per_digit.png)",
                  "", "Карты ниже усреднены по скрытым нейронам для каждого "
                  "пикселя; полные матрицы 784×64 сохранены в результатах.", "",
                  "![Карты по пикселям](assets/2026-09-26/mnist8m_raw_mlp_pixel_maps.png)",
                  "", "| Цифра | BCE всех MLP банка | BCE лучших 10% | Карт для VAE |",
                  "|---:|---:|---:|---:|"])
    for digit, mean, selected, count in bank_rows:
        lines.append(f"| {digit} | {mean:.4f} | {selected:.4f} | {count} |")
    lines.extend(["", "| BCE для каждого z | λ | MSE логитов между VAE ↓ | IoU масок VAE ↑ | Шагов поиска |",
                  "|---|---:|---:|---:|---:|"])
    for objective, coefficient, mse, iou, steps in agreement_rows:
        lines.append(f"| {objective} | {coefficient} | {mse:.4f} | {iou:.3f} | {steps} |")
    lines.extend(["", f"IoU реконструкции top-K importance maps VAE: "
                  f"{recon_ious[0]:.3f} для цифры 3 и {recon_ious[1]:.3f} "
                  "для цифры 8; ориентир для независимых случайных масок "
                  "той же плотности — 0.111.", ""])
    accuracy_gain = (summary["balanced_accuracy"]["shared_consensus_lambda1"] -
                     summary["balanced_accuracy"]["consensus_lambda1"])
    bce_gain = (summary["bce"]["consensus_lambda1"] -
                summary["bce"]["shared_consensus_lambda1"])
    dense_gap = (summary["balanced_accuracy"]["shared_consensus_lambda1"] -
                 summary["balanced_accuracy"]["dense"])
    random_gap = (summary["balanced_accuracy"]["shared_consensus_lambda1"] -
                  summary["balanced_accuracy"]["random_best16"])
    lines.extend(["**Вывод.** При λ=1 обучение каждого z на обеих задачах "
                  f"изменило точность относительно собственной задачи на "
                  f"{accuracy_gain:+.4f}, BCE на {-bce_gain:+.4f}. "
                  f"Разность точности со случайной маской {random_gap:+.4f}, "
                  f"с плотным слоем {dense_gap:+.4f}. "
                  "Это одна пара цифр и один банк; "
                  "результат нельзя переносить на все цифры без повторов.", ""])
    control_root = Path("pattern/outputs/mnist8m_raw_mlp_bce/mask_control")
    if all((control_root / f"task{task}.pt").exists() for task in (0, 1)):
        controls = [torch.load(control_root / f"task{task}.pt", map_location="cpu",
                               weights_only=True) for task in (0, 1)]
        control_accuracy = np.mean([
            (item["balanced_accuracy"][0] - item["balanced_accuracy"][1])
            .mean().item() for item in controls])
        lines.extend([f"**Контроль масок банка.** После одинакового нового "
                      f"обучения MLP лучшие 10% масок дали среднюю разницу "
                      f"точности со случайными {control_accuracy:+.5f}. "
                      "Подробности и график — в "
                      "[парном сравнении](MNIST8M_MASK_CONTROL_2026-09-26.md).",
                      ""])
    sparsity_report = Path("mds/MNIST8M_SPARSITY_CONTROL_2026-09-26.md")
    if sparsity_report.exists():
        lines.extend(["**Проверка меньшей плотности.** Отбор сильнейших "
                      "связей из importance maps дал преимущество над "
                      "случайными масками при 5% и 2%; таблицы и график — "
                      "в [сравнении плотностей]"
                      "(MNIST8M_SPARSITY_CONTROL_2026-09-26.md).", ""])
    five_report = Path("mds/MNIST8M_RAW_MLP_5PCT_2026-09-26.md")
    five_summary_path = Path("pattern/outputs/mnist8m_raw_mlp_bce/pair38_5pct/summary.json")
    if five_report.exists() and five_summary_path.exists():
        five_summary = {row["metric"]: row for row in json.loads(
            five_summary_path.read_text())}
        five_accuracy = five_summary["balanced_accuracy"]
        lines.extend(["**Повтор VAE на 5% без выравнивания карт.** Прямая маска из карты лучшего "
                      f"MLP дала {five_accuracy['best_bank_per_task']:.4f}, "
                      f"случайная {five_accuracy['random']:.4f}, а лучший "
                      "вариант с VAE и agreement остался ниже случайной. "
                      "Таблицы, графики и карты — в "
                      "[отчёте по 5%](MNIST8M_RAW_MLP_5PCT_2026-09-26.md).",
                      ""])
    width_report = Path("mds/MNIST8M_WIDTH_DENSITY_2026-09-26.md")
    if width_report.exists():
        lines.extend(["**Ширина MLP и плотность маски.** Проверены 64, 32 и 16 "
                      "скрытых нейронов при 5% и 2% связей. Таблицы, графики "
                      "и карты покрытия — в [сравнении ширины и плотности]"
                      "(MNIST8M_WIDTH_DENSITY_2026-09-26.md).", ""])
    if Path("mds/MNIST8M_COLUMN_ALIGNMENT_2026-09-26.md").exists():
        lines.extend(["**Выравнивание столбцов карт.** Карты приведены к одной "
                      "опорной карте до обучения VAE; восстановление и качество "
                      "масок приведены в [отдельном сравнении]"
                      "(MNIST8M_COLUMN_ALIGNMENT_2026-09-26.md).", ""])
    lines.extend(["[Код эксперимента](../pattern/mnist8m_raw_mlp_bce.py) · "
                  "[сырые результаты](../pattern/outputs/mnist8m_raw_mlp_bce/pair38/)", ""])
    REPORT.write_text("\n".join(lines))
    print(REPORT)


if __name__ == "__main__":
    main()
