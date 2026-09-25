"""Summarize the nested 1–10 VAE run with figures and random controls."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from pattern.length11.agreement import topk
from pattern.length11.evaluate import gold_mask
from pattern.length11.night import PATTERNS, REPLICATES, SIZES


def mean_metrics(data: dict, indices: list[int]) -> tuple[float, float, float]:
    iou = float(data["iou"][indices].mean())
    accuracy = float(np.mean([
        float(task["accuracy"][indices].mean()) for task in data["tasks"].values()]))
    bce = float(np.mean([
        float(task["bce"][indices].mean()) for task in data["tasks"].values()]))
    return iou, accuracy, bce


def run(out: Path) -> Path:
    out = out.resolve()
    figure_dir = out / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for size in SIZES:
        for rep in REPLICATES:
            path = out / "evaluation_multi" / f"n{size}" / f"rep{rep}.pt"
            if not path.exists():
                raise FileNotFoundError(f"incomplete experiment, missing {path}")
            data = torch.load(path, map_location="cpu", weights_only=True)
            names = data["names"]
            if data["patterns"] != list(PATTERNS[:size]):
                raise ValueError(f"pattern order mismatch: {path}")
            random = [i for i, name in enumerate(names) if name.startswith("random_")]
            individual = [i for i, name in enumerate(names) if name.startswith("joint_vae")]
            selectors = {
                "random": random, "individual": individual,
                "initial": [names.index("initial_consensus")],
                "plain": [names.index("plain_consensus")],
                "joint": [names.index("joint_consensus")],
                "analytic": [names.index("analytic")],
            }
            record = {"size": size, "replicate": rep}
            for key, indices in selectors.items():
                record[key] = mean_metrics(data, indices)
            search = torch.load(out / "joint_multi" / f"n{size}" /
                                f"rep{rep}.pt", map_location="cpu", weights_only=True)
            j = search["chosen_start"]
            logits = search["logits"][:, j]
            record["mse"] = float((logits - logits.mean(0)).square().mean())
            masks = torch.stack([topk(value) for value in logits]).bool()
            agreements = []
            for left in range(size):
                for right in range(left + 1, size):
                    intersection = (masks[left] & masks[right]).sum().item()
                    union = (masks[left] | masks[right]).sum().item()
                    agreements.append(intersection / union)
            record["hard_agreement"] = float(np.mean(agreements)) if agreements else float("nan")
            record["steps"] = search["settings"]["actual_steps"]
            record["plateau"] = search["settings"]["stopped_on_plateau"]
            rows.append(record)

    control_paths = [out / "positive_control" /
                     f"pattern_{pattern}_rep{rep}.pt"
                     for pattern in PATTERNS for rep in REPLICATES]
    control_ious = None
    if all(path.exists() for path in control_paths):
        control_ious = np.asarray([
            torch.load(path, map_location="cpu", weights_only=True)["best_iou"]
            for path in control_paths], dtype=float)

    def values(size: int, key: str, metric: int) -> np.ndarray:
        return np.asarray([row[key][metric] for row in rows if row["size"] == size])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"random": "#777777", "initial": "#9165ad",
              "plain": "#e6862a", "joint": "#258b49", "analytic": "#111111"}
    labels = {"random": "Случайная", "initial": "До поиска",
              "plain": "Только MSE", "joint": "BCE (+ MSE при ≥2)",
              "analytic": "Аналитическая"}
    for metric, ax, title in ((0, axes[0], "IoU"), (1, axes[1], "Точность MLP")):
        for key in ("random", "initial", "plain", "joint", "analytic"):
            sizes = SIZES[1:] if key == "plain" else SIZES
            xs = np.asarray(sizes)
            means = np.asarray([values(size, key, metric).mean() for size in sizes])
            stds = np.asarray([values(size, key, metric).std() for size in sizes])
            ax.plot(xs, means, marker="o", color=colors[key], label=labels[key])
            if key not in ("analytic", "random"):
                ax.fill_between(xs, means - stds, means + stds,
                                color=colors[key], alpha=.12)
        ax.set_xticks(SIZES)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Число VAE")
        ax.set_title(title)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(figure_dir / "iou_accuracy_by_n.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for key in ("random", "initial", "plain", "joint", "analytic"):
        sizes = SIZES[1:] if key == "plain" else SIZES
        xs = np.asarray(sizes)
        means = np.asarray([values(size, key, 2).mean() for size in sizes])
        stds = np.asarray([values(size, key, 2).std() for size in sizes])
        ax.plot(xs, means, marker="o", color=colors[key], label=labels[key])
        if key not in ("analytic", "random"):
            ax.fill_between(xs, means - stds, means + stds,
                            color=colors[key], alpha=.12)
    ax.set_xticks(SIZES)
    ax.set_xlabel("Число VAE")
    ax.set_ylabel("BCE обученного MLP")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "bce_by_n.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for key, ax, label in (("hard_agreement", axes[0], "Попарный IoU масок VAE"),
                           ("mse", axes[1], "MSE выровненных логитов")):
        xs = np.asarray(SIZES[1:])
        series = [np.asarray([row[key] for row in rows if row["size"] == size])
                  for size in SIZES[1:]]
        means = np.asarray([values.mean() for values in series])
        stds = np.asarray([values.std() for values in series])
        ax.plot(xs, means, marker="o", color="#258b49")
        ax.fill_between(xs, means - stds, means + stds,
                        color="#258b49", alpha=.15)
        ax.set_xticks(SIZES)
        ax.set_xlabel("Число VAE")
        ax.set_ylabel(label)
    axes[0].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(figure_dir / "hard_agreement_by_n.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharey=True)
    bank_quality = []
    for index, pattern in enumerate(PATTERNS):
        ax = axes.flat[index]
        for rep in REPLICATES:
            vae = torch.load(out / "vae" / f"pattern_{pattern}_rep{rep}.pt",
                             map_location="cpu", weights_only=True)
            history = vae["history"].numpy()
            ax.plot(history[:, 0], history[:, 1], alpha=.7, linewidth=1)
        ax.set_title(pattern)
        if index >= 5:
            ax.set_xlabel("Эпоха")
        if index % 5 == 0:
            ax.set_ylabel("Проверочный loss VAE")
        bank = torch.load(out / "bank" / f"pattern_{pattern}.pt",
                          map_location="cpu", weights_only=True)
        bank_quality.append(float(bank["selected_query_accuracy"].mean()))
    fig.tight_layout()
    fig.savefig(figure_dir / "vae_training.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].bar(np.arange(len(PATTERNS)), bank_quality)
    axes[0].set_xticks(np.arange(len(PATTERNS)), PATTERNS, rotation=45)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Точность карт в банке")
    axes[1].set_title("Проверочный критерий поиска, сид 0")
    for size in SIZES:
        search = torch.load(out / "joint_multi" / f"n{size}" / "rep0.pt",
                            map_location="cpu", weights_only=True)
        history = search["history"].numpy()
        axes[1].plot(history[:, 0], history[:, 1], label=str(size))
    axes[1].set_xlabel("Шаг")
    axes[1].legend(title="VAE", ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_dir / "bank_and_search.png", dpi=160)
    plt.close(fig)

    chosen_sizes = (1, 2, 3, 6, 8, 10)
    fig, axes = plt.subplots(len(chosen_sizes), 4, figsize=(10, 12))
    for row, size in enumerate(chosen_sizes):
        search = torch.load(out / "joint_multi" / f"n{size}" / "rep0.pt",
                            map_location="cpu", weights_only=True)
        evaluation = torch.load(out / "evaluation_multi" / f"n{size}" / "rep0.pt",
                                map_location="cpu", weights_only=True)
        j = search["chosen_start"]
        mean_logits = search["logits"][:, j].mean(0)
        consensus = topk(mean_logits)
        gold = gold_mask()
        rows_gold, cols = linear_sum_assignment(-torch.matmul(gold.T, consensus).numpy())
        order = cols[np.argsort(rows_gold)]
        aligned_logits = mean_logits[:, order]
        aligned_consensus = consensus[:, order]
        individual_logits = search["logits"][:, j, :, order]
        low = float(individual_logits.min())
        high = float(individual_logits.max())
        individual_fig, individual_axes = plt.subplots(
            size, 2, figsize=(5.5, max(2.4 * size, 5)), squeeze=False)
        for model_index, model_logits in enumerate(individual_logits):
            model_mask = topk(model_logits)
            for model_col, value in enumerate((model_logits, model_mask)):
                ax = individual_axes[model_index, model_col]
                ax.imshow(value, cmap="viridis" if model_col == 0 else "gray_r",
                          vmin=low if model_col == 0 else 0,
                          vmax=high if model_col == 0 else 1, aspect="auto")
                ax.set_xticks([])
                ax.set_yticks([])
                if model_index == 0:
                    ax.set_title("Выровненные логиты" if model_col == 0 else "Top-32")
                if model_col == 0:
                    ax.set_ylabel(f"VAE {model_index + 1}: {PATTERNS[model_index]}")
        individual_fig.tight_layout()
        individual_fig.savefig(figure_dir / f"individual_vaes_n{size}.png", dpi=170)
        plt.close(individual_fig)
        random_index = evaluation["names"].index("random_00")
        random_mask = evaluation["masks"][random_index]
        items = (aligned_logits, aligned_consensus, random_mask, gold)
        titles = ("Средние логиты", "Общая top-32", "Случайная", "Аналитическая")
        for col, (value, title) in enumerate(zip(items, titles)):
            ax = axes[row, col]
            ax.imshow(value, cmap="viridis" if col == 0 else "gray_r",
                      vmin=None if col == 0 else 0,
                      vmax=None if col == 0 else 1, aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"{size} VAE")
    fig.tight_layout()
    fig.savefig(figure_dir / "examples_n1_n2_n3_n6_n8_n10.png", dpi=170)
    plt.close(fig)

    lines = [
        "# Поиск маски для 1–10 VAE на паттернах длины 4", "",
        "Вход длины 11, восемь окон и маски 11×8 с 32 связями. "
        "Одна вложенная группа: " + ", ".join(PATTERNS) + ". "
        "На каждый паттерн — 16 384 исходных MLP, лучшие 10% карт и четыре VAE.", "",
        "При поиске BCE считается на top-32 маске **каждого VAE для каждого паттерна**; "
        "для 2–10 VAE к ней добавлена MSE выровненных логитов с коэффициентом 0,5. "
        "Для одного VAE MSE и попарное agreement не определены: обучается `z` по BCE. "
        "В поиске нескольких VAE половина стартов наследует `z` из предыдущего размера "
        "(для трёх — из отдельного опыта с двумя), остальные случайны. "
        "Аналитическая маска в поиске не участвует.", "",
        "Проверка: новые MLP (4 инициализации, до 20 000 шагов), 16 случайных масок "
        "для каждого запуска, 4 сида VAE. Оценки точности сравнительные: те же входы "
        "участвовали в отборе банка карт.", "",
        "| VAE | IoU случайная | IoU до | IoU только MSE | IoU BCE (+ MSE) | "
        "Точность случайная | Точность только MSE | Точность BCE (+ MSE) | Точность аналитическая |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for size in SIZES:
        def m(key: str, metric: int) -> str:
            return "—" if size == 1 and key == "plain" else f"{values(size, key, metric).mean():.3f}"
        lines.append(f"| {size} | {m('random',0)} | {m('initial',0)} | "
                     f"{m('plain',0)} | {m('joint',0)} | {m('random',1)} | "
                     f"{m('plain',1)} | {m('joint',1)} | {m('analytic',1)} |")
    lines.extend(["", "| VAE | BCE случайная | BCE только MSE | BCE (+ MSE) | "
                  "BCE аналитическая | Попарный IoU масок VAE | MSE логитов |",
                  "|---:|---:|---:|---:|---:|---:|---:|"])
    for size in SIZES:
        subset = [row for row in rows if row["size"] == size]
        pairwise = "—" if size == 1 else f"{np.mean([row['hard_agreement'] for row in subset]):.3f}"
        raw_mse = "—" if size == 1 else f"{np.mean([row['mse'] for row in subset]):.4f}"
        lines.append(f"| {size} | {values(size,'random',2).mean():.3f} | "
                     f"{m('plain',2)} | "
                     f"{values(size,'joint',2).mean():.3f} | "
                     f"{values(size,'analytic',2).mean():.3f} | "
                     f"{pairwise} | {raw_mse} |")
    lines.extend(["", "![IoU и точность по числу VAE](figures/iou_accuracy_by_n.png)", "",
                  "![BCE по числу VAE](figures/bce_by_n.png)", "",
                  "![Совпадение масок и логитов: определено для 2–10 VAE](figures/hard_agreement_by_n.png)", "",
                  "![Обучение VAE](figures/vae_training.png)", "",
                  "![Качество банка и ход поиска](figures/bank_and_search.png)", "",
                  "Примеры ниже: сид 0; столбцы общей маски после поиска переставлены "
                  "по аналитической только для изображения и расчёта IoU.", "",
                  "![Логиты и маски для 1, 2, 3, 6, 8 и 10 VAE](figures/examples_n1_n2_n3_n6_n8_n10.png)", "",
                  "![Один VAE](figures/individual_vaes_n1.png)", "",
                  "![Отдельные VAE, 2 модели](figures/individual_vaes_n2.png)", "",
                  "![Отдельные VAE, 3 модели](figures/individual_vaes_n3.png)", "",
                  "![Отдельные VAE, 6 моделей](figures/individual_vaes_n6.png)", "",
                  "![Отдельные VAE, 8 моделей](figures/individual_vaes_n8.png)", "",
                  "![Отдельные VAE, 10 моделей](figures/individual_vaes_n10.png)", "",
                  "Полные логиты и ход поиска для всех размеров и сидов сохранены "
                  "в `joint_multi/`; оценки и маски — в `evaluation_multi/`.", "",
                  "## Вывод", ""])
    best = max(rows, key=lambda row: row["joint"][0])
    lines.insert(lines.index("## Вывод"),
                 f"Лучший отдельный запуск: {best['size']} VAE, сид {best['replicate']}, "
                 f"IoU {best['joint'][0]:.3f}; это не среднее по четырём сидам.")
    lines.insert(lines.index("## Вывод"), "")
    if control_ious is not None:
        lines.insert(lines.index("## Вывод"),
                     f"Положительный контроль: прямая подгонка `z` каждого VAE к "
                     f"аналитической маске дала средний IoU {control_ious.mean():.3f} "
                     f"(максимум {control_ious.max():.3f}); в agreement эта маска "
                     "не использовалась.")
        lines.insert(lines.index("## Вывод"), "")
    for size in SIZES:
        subset = [row for row in rows if row["size"] == size]
        if not all(row["plateau"] for row in subset):
            lines.append(f"Для {size} VAE часть поисков дошла до лимита шагов без плато.")
    lines.append("Сравнивайте IoU и точность с соответствующими случайными масками; "
                 "малое расхождение логитов само по себе не доказывает теплицевость.")
    report = out / "report.md"
    temp = report.with_suffix(".tmp")
    temp.write_text("\n".join(lines) + "\n")
    temp.replace(report)
    print(report, flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.out)


if __name__ == "__main__":
    main()
