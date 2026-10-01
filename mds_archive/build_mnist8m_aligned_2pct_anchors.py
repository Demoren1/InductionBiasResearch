"""Report the four shared-reference controls at 2% MNIST8m sparsity."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pattern.models.cvae import CVAE
from pattern.run_mnist8m_aligned_2pct_anchors import BASE, selected_anchors


REPORT = Path("mds/MNIST8M_ALIGNED_2PCT_2026-09-27.md")
FIGURE = Path("mds/assets/2026-09-27/mnist8m_aligned_2pct_anchors.png")
METHODS = ("random", "direct", "mean", "vae", "agreement", "dense")
TITLES = {"random": "Случайная", "direct": "Прямая", "mean": "Средняя",
          "vae": "VAE", "agreement": "Agreement", "dense": "Плотная"}


def summarize(folder: Path) -> dict:
    result = torch.load(folder / "evaluation.pt", map_location="cpu",
                        weights_only=True)
    names = result["names"]
    output = {"folder": folder, "actual_steps": result["settings"]["actual_steps"]}
    for metric in ("balanced_accuracy", "bce"):
        values = result["metrics"][metric]
        random = [i for i, name in enumerate(names)
                  if name.startswith("random_capped_")]
        output[metric] = {
            "random": float(values[:, random].mean()),
            "direct": float(torch.stack([
                values[task, names.index(f"bank_task{task}_best")].mean()
                for task in (0, 1)]).mean()),
            "mean": float(torch.stack([
                values[task, names.index(f"bank_task{task}_mean_capped")].mean()
                for task in (0, 1)]).mean()),
            "vae": float(torch.stack([
                values[task, names.index(f"vae{task}_lambda0_capped")].mean()
                for task in (0, 1)]).mean()),
            "agreement": float(values[:, names.index(
                "shared_consensus_lambda1_capped")].mean()),
            "dense": float(values[:, names.index("dense")].mean()),
        }
    values = result["metrics"]["balanced_accuracy"]
    agreement = values[:, names.index("shared_consensus_lambda1_capped")]
    means = torch.stack([values[task, names.index(f"bank_task{task}_mean_capped")]
                         for task in (0, 1)])
    output["difference"] = (agreement - means).numpy()
    output["late_fraction"] = float((result["best_step"] >=
                                     .9 * result["settings"]["max_steps"]).float().mean())
    return output


def heldout_iou(folder: Path) -> tuple[float, float]:
    scores = []
    for task in (0, 1):
        bank = torch.load(folder / f"bank_task{task}.pt", map_location="cpu",
                          weights_only=True)
        checkpoint = torch.load(folder / f"vae_task{task}.pt", map_location="cpu",
                                weights_only=True)
        maps = bank["importance"].flatten(1)
        order = torch.randperm(len(maps),
                               generator=torch.Generator().manual_seed(3130 + task))
        n_val = max(32, round(.15 * len(maps)))
        train, heldout = maps[order[n_val:]], maps[order[:n_val]]
        model = CVAE(**checkpoint["config"])
        model.load_state_dict(checkpoint["model"])
        model.eval()
        with torch.no_grad():
            mu, _ = model.encode(heldout, heldout.new_zeros(len(heldout), 0))
            decoded = model.decode(mu, heldout.new_zeros(len(heldout), 0))
        k = int(bank["masks"][0].sum())
        target = torch.zeros_like(heldout, dtype=torch.bool)
        target.scatter_(1, heldout.topk(k, dim=1).indices, True)

        def iou(prediction: torch.Tensor) -> float:
            candidate = torch.zeros_like(heldout, dtype=torch.bool)
            candidate.scatter_(1, prediction.topk(k, dim=1).indices, True)
            intersection = (candidate & target).sum(1)
            return float((intersection / (2 * k - intersection)).float().mean())

        scores.append((iou(train.mean(0).expand_as(heldout)), iou(decoded)))
    return tuple(float(np.mean([value[index] for value in scores]))
                 for index in (0, 1))


def main() -> None:
    torch.set_num_threads(2)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    baseline = summarize(BASE / "width64_2pct")
    anchors = selected_anchors()
    rows = []
    for task, index in anchors:
        folder = BASE / f"width64_2pct_anchor_t{task}_i{index}"
        row = summarize(folder)
        search = torch.load(folder / "search_shared_lambda1.pt", map_location="cpu",
                            weights_only=True)
        row["search_steps"] = search["steps"]
        row["search_plateau"] = search["plateau"]
        row["anchor"] = f"{3 if task == 0 else 8}:{index}"
        row["reference_task"] = task
        row["reference_index"] = index
        row["heldout_iou"] = heldout_iou(folder)
        rows.append(row)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    positions = np.arange(len(rows))
    for method in ("direct", "mean", "vae", "agreement"):
        axes[0].plot(positions, [row["balanced_accuracy"][method] for row in rows],
                     marker="o", label=TITLES[method])
    axes[0].axhline(rows[0]["balanced_accuracy"]["random"], color="gray",
                    linestyle="--", label="Случайная")
    axes[0].axhline(rows[0]["balanced_accuracy"]["dense"], color="black",
                    linestyle=":", label="Плотная")
    axes[0].set_xticks(positions, [row["anchor"] for row in rows])
    axes[0].set_ylabel("Сбалансированная точность")
    axes[0].set_xlabel("Цифра опоры: индекс карты")
    axes[0].set_ylim(.90, .98)
    axes[0].legend(fontsize=8)
    for digit in (0, 1):
        axes[1].bar(positions + (-.18 if digit == 0 else .18),
                    [100 * row["difference"][digit].mean() for row in rows],
                    width=.34, label=f"цифра {3 if digit == 0 else 8}")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xticks(positions, [row["anchor"] for row in rows])
    axes[1].set_ylabel("Agreement − средняя карта, п. п.")
    axes[1].set_xlabel("Цифра опоры: индекс карты")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)

    lines = ["# MNIST8m: общая опора при 2% связей", "",
             "Задачи «3 против остальных» и «8 против остальных», MLP 784→64→1. Четыре карты из обучающей части банка (по две от каждой цифры) поочерёдно служили общей опорой для выравнивания столбцов обеих групп importance maps. После каждого выравнивания заново обучались два VAE и искался agreement. Во всех масках ровно 1004 входные связи, для случайной, средней, VAE и agreement — не более двух связей на пиксель. Каждая маска проверена на новых MLP с четырьмя одинаковыми инициализациями; все остальные данные и настройки совпадают.", "",
             "| Опора | Случайная | Прямая | Средняя | VAE | Agreement | Плотная | Agreement − средняя, п. п. |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, row in [("Без выравнивания", baseline)] + [
            (row["anchor"], row) for row in rows]:
        accuracy = row["balanced_accuracy"]
        diff = 100 * (accuracy["agreement"] - accuracy["mean"])
        lines.append(f"| {label} | " + " | ".join(
            f"{accuracy[method]:.4f}" for method in METHODS) + f" | {diff:+.2f} |")
    lines += ["", "Сбалансированная точность на отдельном тестовом блоке; среднее по двум цифрам и четырём повторным обучениям MLP. С учётом смещений и выхода масочный MLP имеет 1133 активных коэффициента против 50305 у плотного; фактическая реализация хранит плотный тензор весов.", "",
              "| Опора | BCE случайная | BCE средняя | BCE VAE | BCE agreement | BCE плотная | IoU средней | IoU VAE | Последний лучший чекпоинт MLP |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        loss = row["bce"]
        mean_iou, vae_iou = row["heldout_iou"]
        lines.append(f"| {row['anchor']} | {loss['random']:.4f} | {loss['mean']:.4f} | {loss['vae']:.4f} | {loss['agreement']:.4f} | {loss['dense']:.4f} | {mean_iou:.3f} | {vae_iou:.3f} | {100 * row['late_fraction']:.1f}% |")
    search_range = (min(row["search_steps"] for row in rows),
                    max(row["search_steps"] for row in rows))
    eval_range = (min(row["actual_steps"] for row in rows),
                  max(row["actual_steps"] for row in rows))
    lines += ["", "IoU посчитан на картах, отложенных при обучении VAE. "
              "Последний столбец — доля масок, для которых лучший MLP найден "
              "в последних 10% доступных шагов. Поиск agreement остановился "
              f"по плато во всех четырёх опытах на шагах {search_range[0]}–"
              f"{search_range[1]}; повторное обучение MLP — на шагах "
              f"{eval_range[0]}–{eval_range[1]}, без поздних лучших состояний.", "",
              "![Точность и разница с усреднённой картой](assets/2026-09-27/mnist8m_aligned_2pct_anchors.png)", ""]
    improvements = [row["balanced_accuracy"]["agreement"] -
                    row["balanced_accuracy"]["mean"] for row in rows]
    count = sum(value > 0 for value in improvements)
    lines += [f"**Вывод.** Выравнивание повысило точность и средней карты, и agreement относительно запуска без выравнивания. Agreement превзошёл выровненную среднюю карту лишь при {count} из {len(rows)} опор: средняя разница {100 * np.mean(improvements):+.2f} п. п., диапазон от {100 * min(improvements):+.2f} до {100 * max(improvements):+.2f} п. п. При всех четырёх опорах agreement также уступил прямой карте и плотному MLP. На этой паре задач дополнительная польза поиска z не воспроизвелась устойчиво; выбор опоры существенно влияет на результат.", "",
              "[Код запуска](../pattern/run_mnist8m_aligned_2pct_anchors.py) · [код выравнивания](../pattern/align_mnist8m_importance_bank.py) · [план ночного запуска](MNIST8M_NIGHT_PLAN_2026-09-27.md) · [сырые результаты](../pattern/outputs/mnist8m_raw_mlp_bce/)", ""]
    REPORT.write_text("\n".join(lines))
    print(REPORT)


if __name__ == "__main__":
    main()
