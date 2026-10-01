"""Summarize the fixed-reference hidden-column alignment experiment."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pattern.models.cvae import CVAE


BASE = Path("pattern/outputs/mnist8m_raw_mlp_bce")
RAW = BASE / "width64_5pct"
LOCAL = BASE / "width64_5pct_aligned"
SHARED = BASE / "width64_5pct_global_anchor"
ASSET = Path("mds/assets/2026-09-26")
REPORT = Path("mds/MNIST8M_COLUMN_ALIGNMENT_2026-09-26.md")


def iou(target: torch.Tensor, prediction: torch.Tensor, k: int) -> float:
    truth = torch.zeros_like(target, dtype=torch.bool)
    truth.scatter_(1, target.topk(k, dim=1).indices, True)
    guess = torch.zeros_like(target, dtype=torch.bool)
    guess.scatter_(1, prediction.topk(k, dim=1).indices, True)
    intersection = (truth & guess).sum(1)
    return float((intersection / (2 * k - intersection)).float().mean())


def heldout(folder: Path, task: int) -> tuple[float, float]:
    bank = torch.load(folder / f"bank_task{task}.pt", map_location="cpu",
                      weights_only=True)
    checkpoint = torch.load(folder / f"vae_task{task}.pt", map_location="cpu",
                            weights_only=True)
    maps = bank["importance"].flatten(1)
    indices = torch.randperm(len(maps), generator=torch.Generator().manual_seed(3130 + task))
    n_validation = max(32, round(.15 * len(maps)))
    train, validation = maps[indices[n_validation:]], maps[indices[:n_validation]]
    model = CVAE(**checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(validation, validation.new_zeros(len(validation), 0))
        decoded = model.decode(mu, validation.new_zeros(len(validation), 0))
    k = int(bank["masks"][0].sum())
    return iou(validation, train.mean(0).expand_as(validation), k), iou(
        validation, decoded, k)


def metrics(folder: Path) -> dict[str, dict[str, float]]:
    result = torch.load(folder / "evaluation.pt", map_location="cpu", weights_only=True)
    names = result["names"]
    output = {}
    for metric in ("balanced_accuracy", "bce"):
        values = result["metrics"][metric]
        random_ids = [i for i, name in enumerate(names) if name.startswith("random_capped_")]
        output[metric] = {
            "dense": float(values[:, names.index("dense")].mean()),
            "random": float(values[:, random_ids].mean()),
            "direct": float(np.mean([
                values[task, names.index(f"bank_task{task}_best")].mean().item()
                for task in (0, 1)])),
            "mean": float(np.mean([
                values[task, names.index(f"bank_task{task}_mean_capped")].mean().item()
                for task in (0, 1)])),
            "mean_plain": float(np.mean([
                values[task, names.index(f"bank_task{task}_mean")].mean().item()
                for task in (0, 1)])),
            "vae": float(np.mean([
                values[task, names.index(f"vae{task}_lambda0_capped")].mean().item()
                for task in (0, 1)])),
            "agreement": float(values[:, names.index(
                "shared_consensus_lambda1_capped")].mean()),
            "agreement_plain": float(values[:, names.index(
                "shared_consensus_lambda1")].mean()),
        }
    return output


def per_digit_agreement(folder: Path) -> torch.Tensor:
    result = torch.load(folder / "evaluation.pt", map_location="cpu",
                        weights_only=True)
    index = result["names"].index("shared_consensus_lambda1_capped")
    return result["metrics"]["balanced_accuracy"][:, index]


def main() -> None:
    torch.set_num_threads(2)
    ASSET.mkdir(parents=True, exist_ok=True)
    variants = (("Без", RAW), ("По задаче", LOCAL), ("Общая опора", SHARED))
    recon = {tag: [heldout(folder, task) for task in (0, 1)]
             for tag, folder in variants}
    values = {tag: metrics(folder) for tag, folder in variants}
    agreement_by_digit = {tag: per_digit_agreement(folder)
                          for tag, folder in variants}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    positions = np.arange(4)
    labels = ["3: средняя", "3: VAE", "8: средняя", "8: VAE"]
    for delta, tag in ((-.24, "Без"), (0, "По задаче"), (.24, "Общая опора")):
        axes[0].bar(positions + delta, np.array(recon[tag]).flatten(), .23,
                    label=tag)
    axes[0].axhline(.05 / 1.95, color="black", linestyle="--",
                    linewidth=1, label="Случайное IoU")
    axes[0].set_xticks(positions, labels, rotation=20, ha="right")
    axes[0].set_ylabel("IoU с отложенной картой")
    axes[0].legend(fontsize=8)
    names = ["random", "direct", "mean", "vae", "agreement", "dense"]
    titles = ["Случайная", "Прямая", "Средняя", "VAE", "Agreement", "Плотная"]
    for delta, tag in ((-.24, "Без"), (0, "По задаче"), (.24, "Общая опора")):
        axes[1].bar(np.arange(len(names)) + delta,
                    [values[tag]["balanced_accuracy"][name] for name in names],
                    .23, label=tag)
    axes[1].set_xticks(np.arange(len(names)), titles, rotation=25, ha="right")
    axes[1].set_ylabel("Сбалансированная точность")
    axes[1].set_ylim(.90, .98)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSET / "mnist8m_column_alignment.png", dpi=180)
    plt.close(fig)

    lines = [
        "# MNIST8m: выравнивание столбцов importance maps",
        "",
        "Для задач «3 против остальных» и «8 против остальных» столбцы importance maps сопоставлены по сходству с помощью венгерского алгоритма. Сравнены исходные карты, отдельная опора внутри каждой задачи и **одна общая карта из обучающего банка цифры 3** для обеих задач. Затем заново обучены VAE и найдены маски agreement; прежнее фиксированное выравнивание выходов двух декодеров при поиске z сохранено. MLP: 784→64→1, плотность 5%; количество связей в масках одинаково. Восстановление проверено на отложенных картах, классификация — на отдельном тестовом блоке после нового обучения весов (4 запуска на маску).",
        "",
        "| Цифра | Опора | IoU средней карты | IoU VAE | Случайный IoU |",
        "|---:|---|---:|---:|---:|",
    ]
    for task, digit in enumerate((3, 8)):
        for tag, _ in variants:
            lines.append(f"| {digit} | {tag} | {recon[tag][task][0]:.3f} | {recon[tag][task][1]:.3f} | 0.026 |")
    lines += ["", "IoU: top-K реконструкции против top-K карты, не использованной при обучении VAE.", "",
              "| Маска | Точность без | По задаче | Общая опора | BCE без | BCE по задаче | BCE общая опора |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, title in zip(names, titles):
        lines.append(f"| {title} | " + " | ".join(
            f"{values[tag][metric][name]:.4f}"
            for metric in ("balanced_accuracy", "bce")
            for tag, _ in variants) + " |")
    lines += ["", "| Agreement по цифре | Без | Общая опора | Прирост, п. п. |",
              "|---:|---:|---:|---:|"]
    for task, digit in enumerate((3, 8)):
        before = float(agreement_by_digit["Без"][task].mean())
        after = float(agreement_by_digit["Общая опора"][task].mean())
        lines.append(f"| {digit} | {before:.4f} | {after:.4f} | {100 * (after - before):+.2f} |")
    paired = agreement_by_digit["Общая опора"] - agreement_by_digit["Без"]
    lines += ["", f"С общей опорой точность выросла во всех {paired.numel()} парных повторных обучениях MLP."]
    lines += ["", "Для случайной, средней, VAE и agreement масок ограничено число связей на входной пиксель: максимум четыре. Прямая маска — карта лучшего MLP из банка. Плотная сеть — контроль без маски.", "",
              "**Параметры итогового MLP.** Плотный 784→64→1: 50 176 весов входного слоя, 64 смещения, 64 выходных веса и одно смещение — всего **50 305**. Маска 5% оставляет 2 509 входных связей: **2 638 активных коэффициентов**, в 19,1 раза меньше. Фиксированная маска не входит в число обучаемых параметров. Пока код хранит плотный весовой тензор: фактически выделено те же 50 305 параметров, экономия памяти и ускорение не измерены. Два VAE для поиска маски имеют по 25 897 024 параметра; после фиксации маски они не нужны для работы классификатора.", "",
              "| Точность без лимита на пиксель | Без | По задаче | Общая опора |",
              "|---|---:|---:|---:|"]
    for key, title in (("mean_plain", "Средняя"),
                       ("agreement_plain", "Agreement")):
        lines.append(f"| {title} | " + " | ".join(
            f"{values[tag]['balanced_accuracy'][key]:.4f}"
            for tag, _ in variants) + " |")
    lines += ["",
              "![Восстановление и качество масок](assets/2026-09-26/mnist8m_column_alignment.png)", ""]
    old = values["Без"]["balanced_accuracy"]
    new = values["Общая опора"]["balanced_accuracy"]
    delta = (new["agreement"] - old["agreement"]) * 100
    lines += [f"**Вывод.** Выравнивание улучшило восстановление отложенных importance maps, но VAE всё ещё почти повторяет среднюю карту. С общей опорой точность agreement изменилась на {delta:+.2f} п. п. относительно исходного запуска; она {'выше' if new['agreement'] > new['random'] else 'ниже'} случайной маски и {'выше' if new['agreement'] > new['direct'] else 'ниже'} прямой маски банка. Средняя карта дала {new['mean']:.4f} и осталась лучше agreement ({new['agreement']:.4f}) по точности. По BCE agreement лучше средней карты ({values['Общая опора']['bce']['agreement']:.4f} против {values['Общая опора']['bce']['mean']:.4f}). Значит, главный прирост точности объясняется выравниванием; оптимизация z пока не дала добавочного выигрыша именно в точности.", "",
              "[Код выравнивания](../pattern/align_mnist8m_importance_bank.py) · [эксперимент](../pattern/mnist8m_raw_mlp_bce.py) · [сырые результаты](../pattern/outputs/mnist8m_raw_mlp_bce/width64_5pct_global_anchor/) · [ширина и плотность](MNIST8M_WIDTH_DENSITY_2026-09-26.md)", ""]
    REPORT.write_text("\n".join(lines))
    print(REPORT)


if __name__ == "__main__":
    main()
