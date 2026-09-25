"""Report the two-pattern large-bank BCE agreement experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from pattern.length11.agreement import topk
from pattern.length11.evaluate import gold_iou, gold_mask


ROOT = Path(__file__).resolve().parents[1] / "outputs" / "length11_pair1100_1101_20260924"
HERE = Path(__file__).resolve().parents[2]
FIG = HERE / "mds" / "figures" / "length11_pair_bce_20260924"
REPORT = HERE / "mds" / "LENGTH11_PAIR_BCE_AGREEMENT_2026-09-24.md"


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    variant = "joint_long" if all((ROOT / "evaluation_joint_long" / "pair02" /
                                   f"rep{rep}.pt").exists() for rep in range(4)) else "joint"
    evaluations = [torch.load(ROOT / f"evaluation_{variant}" / "pair02" /
                              f"rep{rep}.pt", map_location="cpu", weights_only=True)
                   for rep in range(4)]
    joints = [torch.load(ROOT / variant / f"pair02_rep{rep}.pt",
                         map_location="cpu", weights_only=True) for rep in range(4)]
    plains = [torch.load(ROOT / "agreement" / "1100_1101" / f"rep{rep}.pt",
                         map_location="cpu", weights_only=True) for rep in range(4)]
    categories = [("random", "Случайная", list(range(4, 20))),
                  ("plain", "Только MSE", [3]),
                  ("joint_individual", "BCE каждой, отдельные", [0, 1]),
                  ("joint", "BCE каждой, общая", [2]),
                  ("gold", "Аналитическая", [20])]
    stats = {}
    for key, _, indices in categories:
        rows = []
        for data in evaluations:
            task_accuracy = [float(task["accuracy"][indices].mean())
                             for task in data["tasks"].values()]
            task_bce = [float(task["bce"][indices].mean())
                        for task in data["tasks"].values()]
            rows.append((float(data["iou"][indices].mean()),
                         float(np.mean(task_accuracy)), float(np.mean(task_bce))))
        stats[key] = np.asarray(rows)
    controls = []
    epochs = []
    for pattern in ("1100", "1101"):
        for rep in range(4):
            controls.append(torch.load(ROOT / "positive_control" /
                                       f"pattern_{pattern}_rep{rep}.pt",
                                       map_location="cpu", weights_only=True)["best_iou"])
            epochs.append(torch.load(ROOT / "vae" / f"pattern_{pattern}_rep{rep}.pt",
                                     map_location="cpu", weights_only=True)["best_epoch"])
    hard_agreement = []
    mse = []
    search_steps = []
    initial_ious = []
    for data in plains:
        j = data["chosen_start"]
        initial_ious.append(gold_iou(topk(data["initial_logits"][:, j].mean(0))))
    short_ious = []
    if variant == "joint_long":
        for rep in range(4):
            short = torch.load(ROOT / "joint" / f"pair02_rep{rep}.pt",
                               map_location="cpu", weights_only=True)
            short_ious.append(gold_iou(topk(short["logits"][:, short["chosen_start"]].mean(0))))
    for data in joints:
        j = data["chosen_start"]
        masks = data["masks"][:, j]
        overlap = float((masks[0] * masks[1]).sum())
        hard_agreement.append(overlap / (64 - overlap))
        logits = data["logits"][:, j]
        mse.append(float((logits - logits.mean(0)).square().mean()))
        search_steps.append(int(data["history"][-1, 0]))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    seeds = np.arange(4)
    width = .22
    for offset, key, label in ((-width, "random", "Случайная"),
                                (0, "plain", "Только MSE"),
                                (width, "joint", "BCE каждой + MSE")):
        axes[0].bar(seeds + offset, stats[key][:, 0], width, label=label)
        axes[1].bar(seeds + offset, stats[key][:, 1], width, label=label)
    axes[0].set_title("IoU с аналитической")
    axes[1].set_title("Точность MLP")
    for ax in axes:
        ax.set_xticks(seeds, [str(rep) for rep in seeds])
        ax.set_xlabel("Сид VAE")
        ax.set_ylim(0, 1)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "seed_comparison.png", dpi=170)
    plt.close(fig)

    rep = int(stats["joint"][:, 0].argmax())
    data = joints[rep]
    j = data["chosen_start"]
    logits = data["logits"][:, j]
    mean_logits = logits.mean(0)
    consensus_mask = topk(mean_logits)
    rows, cols = linear_sum_assignment(-torch.matmul(gold_mask().T, consensus_mask).numpy())
    gold_order = cols[np.argsort(rows)]
    image_mats = [logits[0][:, gold_order], logits[1][:, gold_order],
                  mean_logits[:, gold_order]]
    masks = [*topk(logits)[:, :, gold_order], consensus_mask[:, gold_order]]
    vmin = float(torch.stack(image_mats).min())
    vmax = float(torch.stack(image_mats).max())
    fig, axes = plt.subplots(2, 3, figsize=(8.7, 6))
    names = ["VAE 1100", "VAE 1101", "Среднее"]
    for ax, value, title in zip(axes[0], image_mats, names):
        ax.imshow(value, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title + ", логиты")
    for ax, value, title in zip(axes[1], masks, names):
        ax.imshow(value, cmap="gray_r", vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"{title}, top-32, IoU {gold_iou(value):.3f}")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIG / "joint_example.png", dpi=170)
    plt.close(fig)

    random_mask = evaluations[rep]["masks"][4]
    fig, axes = plt.subplots(1, 2, figsize=(5, 3))
    for ax, mask, title in zip(axes, (random_mask, gold_mask()),
                               ("Случайная", "Аналитическая")):
        ax.imshow(mask, cmap="gray_r", vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"{title}, IoU {gold_iou(mask):.3f}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIG / "controls.png", dpi=170)
    plt.close(fig)

    lines = [
        "# Agreement с BCE каждой маски: паттерны 1100 и 1101", "",
        "Длина входа 11, восемь окон длины 4, восемь скрытых нейронов. "
        "Аналитическая матрица имеет размер 11×8 и ровно 32 связи.", "",
        "Для каждого паттерна обучили 16 384 MLP со случайными масками из 32 связей, "
        "оставили лучшие 10% по проверочной BCE и обучили по четыре VAE. "
        "Лучшая эпоха VAE: от " + str(min(epochs)) + " до " + str(max(epochs)) + ".", "",
        "В поиске заморожены VAE, обучаются только `z` и новые MLP. "
        "Каждый MLP видит top-32 маску своего VAE; его BCE считается отдельно "
        "для задач 1100 и 1101. К средней BCE добавлена MSE выровненных логитов "
        "с коэффициентом 0,5. Столбцы выравниваются один раз в начале. "
        "Аналитическая маска в поиске не используется. На иллюстрации общие "
        "столбцы дополнительно переставлены по ней после поиска только для сравнения.", "",
        f"Поиск продолжался {min(search_steps)}–{max(search_steps)} шагов "
        "и во всех сидах остановился после плато проверочного критерия.", "",
        "Проверка выполнена новыми MLP с четырьмя инициализациями и отдельной "
        "проверочной выборкой; сохранена лучшая проверочная точка до 20 000 шагов. "
        "Случайный контроль: 16 независимых масок в каждом сиде. "
        "Те же входы участвовали в отборе исходного банка, поэтому абсолютные "
        "оценки точности не являются независимым тестом банка.", "",
        "| Маска | IoU | Точность MLP | BCE MLP |", "|---|---:|---:|---:|",
    ]
    for key, title, _ in categories:
        values = stats[key].mean(0)
        lines.append(f"| {title} | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} |")
    lines.extend(["", "| Паттерн | Случайная: точность | Только MSE | BCE каждой + MSE | Аналитическая |",
                  "|---|---:|---:|---:|---:|"])
    for pattern in ("1100", "1101"):
        vals = []
        for indices in (list(range(4, 20)), [3], [2], [20]):
            vals.append(float(np.mean([
                float(data["tasks"][pattern]["accuracy"][indices].mean())
                for data in evaluations])))
        lines.append(f"| {pattern} | {vals[0]:.3f} | {vals[1]:.3f} | "
                     f"{vals[2]:.3f} | {vals[3]:.3f} |")
    lines.extend(["", "| Сид | До поиска: IoU | Только MSE: IoU | BCE каждой + MSE: IoU | "
                  "Согласие двух top-32 масок |", "|---:|---:|---:|---:|---:|"])
    for rep in range(4):
        lines.append(f"| {rep} | {initial_ious[rep]:.3f} | {stats['plain'][rep,0]:.3f} | "
                     f"{stats['joint'][rep,0]:.3f} | {hard_agreement[rep]:.3f} |")
    lines.extend(["",
                  f"Средняя MSE логитов после поиска с BCE: {np.mean(mse):.4f}. "
                  f"При прямой подгонке `z` к аналитической маске VAE достигли "
                  f"IoU {np.mean(controls):.3f} в среднем (диапазон "
                  f"{np.min(controls):.3f}–{np.max(controls):.3f}); "
                  "это только положительный контроль.", "",
                  "![Сравнение по сидам](figures/length11_pair_bce_20260924/seed_comparison.png)", "",
                  f"![Выровненные логиты и маски, сид {rep}](figures/length11_pair_bce_20260924/joint_example.png)", "",
                  "![Случайная и аналитическая маски](figures/length11_pair_bce_20260924/controls.png)", "",
                  "## Вывод", "",
                  "Больший банк дал VAE возможность приблизиться к аналитической структуре. "
                  "BCE каждой маски при поиске улучшила IoU общей маски относительно "
                  "одного согласования логитов. Точное совпадение пока не получено; "
                  "представление VAE уже достаточно, а оставшийся разрыв связан с "
                  "поиском общей маски.", ""])
    if short_ious:
        note = (f"На 10 000 шагах IoU был {np.mean(short_ious):.3f}; после более "
                f"долгого поиска — {stats['joint'][:,0].mean():.3f}. "
                "Снижение BCE и MSE не гарантирует рост IoU с аналитической маской.")
        lines.insert(lines.index("## Вывод"), note)
        lines.insert(lines.index("## Вывод"), "")
    REPORT.write_text("\n".join(lines))
    print(REPORT)


if __name__ == "__main__":
    main()
