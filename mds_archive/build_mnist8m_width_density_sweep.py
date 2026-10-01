"""Compare narrow raw-pixel MNIST8m MLPs at 5% and 2% mask density."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pattern.models.cvae import CVAE


ROOT = Path("pattern/outputs/mnist8m_raw_mlp_bce")
ASSETS = Path("mds/assets/2026-09-26")
REPORT = Path("mds/MNIST8M_WIDTH_DENSITY_2026-09-26.md")
SETTINGS = ((64, 5), (32, 5), (16, 5),
            (64, 2), (32, 2), (16, 2))
METHODS = ("dense", "random", "direct", "mean", "vae", "agreement")
LABELS = ("Плотный", "Случайная", "Прямая карта", "Средняя карта",
          "VAE", "Agreement")


def reconstruction(width: int, percent: int) -> tuple[float, float]:
    folder = ROOT / f"width{width}_{percent}pct"
    results = []
    for task in (0, 1):
        bank = torch.load(folder / f"bank_task{task}.pt", map_location="cpu",
                          weights_only=True)
        checkpoint = torch.load(folder / f"vae_task{task}.pt", map_location="cpu",
                                weights_only=True)
        maps = bank["importance"].flatten(1)
        generator = torch.Generator().manual_seed(3130 + task)
        order = torch.randperm(len(maps), generator=generator)
        n_val = max(32, round(.15 * len(maps)))
        train, held_out = maps[order[n_val:]], maps[order[:n_val]]
        model = CVAE(**checkpoint["config"])
        model.load_state_dict(checkpoint["model"])
        model.eval()
        with torch.no_grad():
            mu, _ = model.encode(held_out, held_out.new_zeros(len(held_out), 0))
            logits = model.decode(mu, held_out.new_zeros(len(held_out), 0))
        k = round(percent / 100 * maps.shape[1])
        target = torch.zeros_like(held_out, dtype=torch.bool)
        target.scatter_(1, held_out.topk(k, dim=1).indices, True)
        predicted = torch.zeros_like(target)
        predicted.scatter_(1, logits.topk(k, dim=1).indices, True)
        template = torch.zeros_like(train[0], dtype=torch.bool)
        template.scatter_(0, train.mean(0).topk(k).indices, True)
        intersection_vae = (target & predicted).sum(1)
        intersection_mean = (target & template).sum(1)
        results.append((float((intersection_vae / (2 * k - intersection_vae)).float().mean()),
                        float((intersection_mean / (2 * k - intersection_mean)).float().mean())))
    return tuple(float(np.mean([row[i] for row in results])) for i in (0, 1))


def summarize(width: int, percent: int) -> dict:
    result = torch.load(ROOT / f"width{width}_{percent}pct" / "evaluation.pt",
                        map_location="cpu", weights_only=True)
    names, masks = result["names"], result["masks"]
    random = [i for i, name in enumerate(names)
              if name.startswith("random_capped_")]
    templates = {"direct": "bank_task{task}_best",
                 "mean": "bank_task{task}_mean_capped",
                 "vae": "vae{task}_lambda0_capped"}
    values = {}
    global_accuracy = {}
    for key in ("balanced_accuracy", "bce"):
        metric = result["metrics"][key]
        row = {"random": float(metric[:, random].mean())}
        for method, template in templates.items():
            row[method] = float(torch.stack([
                metric[task, names.index(template.format(task=task))].mean()
                for task in (0, 1)]).mean())
        row["agreement"] = float(metric[:, names.index(
            "shared_consensus_lambda1_capped")].mean())
        row["dense"] = float(metric[:, names.index("dense")].mean())
        values[key] = row
        if key == "balanced_accuracy":
            global_accuracy["vae"] = float(torch.stack([
                metric[task, names.index(f"vae{task}_lambda0")].mean()
                for task in (0, 1)]).mean())
            global_accuracy["agreement"] = float(metric[:, names.index(
                "shared_consensus_lambda1")].mean())
    coverage = {"random": float(np.mean([
        masks[index].bool().any(-1).sum().item() for index in random]))}
    for method, template in templates.items():
        coverage[method] = float(np.mean([
            masks[names.index(template.format(task=task))].bool().any(-1).sum().item()
            for task in (0, 1)]))
    coverage["agreement"] = float(masks[names.index(
        "shared_consensus_lambda1_capped")].bool().any(-1).sum())
    settings = result["settings"]
    search = torch.load(ROOT / f"width{width}_{percent}pct" /
                        "search_shared_lambda1.pt", map_location="cpu",
                        weights_only=True)
    vae_iou, mean_iou = reconstruction(width, percent)
    accuracy = result["metrics"]["balanced_accuracy"]
    per_task = {}
    for task in (0, 1):
        per_task[task] = {"random": float(accuracy[task, random].mean())}
        for method, template in templates.items():
            per_task[task][method] = float(accuracy[
                task, names.index(template.format(task=task))].mean())
        for method, name in (("agreement", "shared_consensus_lambda1_capped"),
                             ("dense", "dense")):
            per_task[task][method] = float(accuracy[task, names.index(name)].mean())
    return {"width": width, "density": percent, "edges": int(masks[0].sum()),
            "cap": settings["cap"], "accuracy": values["balanced_accuracy"],
            "bce": values["bce"], "coverage": coverage, "per_task": per_task,
            "global_accuracy": global_accuracy,
            "vae_iou": vae_iou, "mean_iou": mean_iou,
            "steps": settings["actual_steps"],
            "search_steps": search["steps"], "search_plateau": search["plateau"],
            "search_best_step": int(search["best_step"][search["chosen_start"]]),
            "late_best": float((result["best_step"] >
                                .9 * settings["actual_steps"]).float().mean())}


def graph(rows: list[dict], key: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    for ax, density in zip(axes, (5, 2)):
        selected = sorted((row for row in rows if row["density"] == density),
                          key=lambda row: row["width"])
        widths = [row["width"] for row in selected]
        for method, label in zip(METHODS, LABELS):
            ax.plot(widths, [row[key][method] for row in selected], "o-",
                    label=label)
        ax.set_xticks(widths)
        ax.set_xlabel("Скрытых нейронов")
        ax.set_title(f"Плотность {density}%")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Сбалансированная точность" if key == "accuracy" else "BCE")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def coverage_graph(path: Path) -> None:
    selectors = (("random_capped_00", "Случайная"),
                 ("bank_task0_best", "Прямая"),
                 ("bank_task0_mean_capped", "Средняя"),
                 ("vae0_lambda0_capped", "VAE"),
                 ("shared_consensus_lambda1_capped", "Agreement"))
    fig, axes = plt.subplots(len(SETTINGS), len(selectors), figsize=(10, 12))
    for row, (width, density) in enumerate(SETTINGS):
        result = torch.load(ROOT / f"width{width}_{density}pct" / "evaluation.pt",
                            map_location="cpu", weights_only=True)
        for column, (name, label) in enumerate(selectors):
            mask = result["masks"][result["names"].index(name)]
            covered = mask.bool().any(-1).reshape(28, 28)
            ax = axes[row, column]
            ax.imshow(covered, cmap="Greys", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(label)
            if column == 0:
                ax.set_ylabel(f"{width} нейр., {density}%")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    torch.set_num_threads(2)
    rows = [summarize(*setting) for setting in SETTINGS]
    ASSETS.mkdir(parents=True, exist_ok=True)
    graph(rows, "accuracy", ASSETS / "mnist8m_width_density_accuracy.png")
    graph(rows, "bce", ASSETS / "mnist8m_width_density_bce.png")
    coverage_graph(ASSETS / "mnist8m_width_density_coverage.png")
    lines = ["# MNIST8m: ширина MLP и плотность масок", "",
             "Две задачи «цифра против остальных» для 3 и 8. MLP получает "
             "784 сырых пикселя, имеет один скрытый слой и один выход. "
             "На каждой ширине заново обучены 4096 MLP со случайными масками "
             "20% на задачу; лучшие 10% выбраны по validation BCE. Из их "
             "importance maps взяты 5% и 2% связей. Для каждой ширины и "
             "плотности заново обучены два VAE и найдено agreement. Маски "
             "проверены на новых MLP; плотный, случайный и прямой варианты "
             "служат контролем. Столбцы importance maps перед обучением VAE "
             "здесь не выравнивались.", "",
             "| Ширина | Плотность | Связей | Лимит/пиксель | "
             "Плотный | Случайная | Прямая | Средняя | VAE | Agreement |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        m = row["accuracy"]
        lines.append(f"| {row['width']} | {row['density']}% | {row['edges']} | "
                     f"{row['cap']} | " + " | ".join(f"{m[k]:.4f}" for k in METHODS) + " |")
    lines.extend(["", "Сбалансированная точность на отдельном тестовом блоке; "
                  "среднее по двум задачам и четырём повторным обучениям. "
                  "Для случайной, средней, VAE и agreement применяется предел "
                  "числа связей на пиксель. Прямая маска получена из карты "
                  "лучшего MLP банка. В каждой маске одинаковое число связей. "
                  "Плотные MLP имеют 50305, 25153 и 12577 параметров при "
                  "ширинах 64, 32 и 16 соответственно. Масочные MLP "
                  "сейчас вычисляются как плотные тензоры: число активных "
                  "связей не является измерением ускорения или памяти.", "",
                  "| Ширина | Плотность | Плотный MLP | Активно с маской | "
                  "Сокращение активных коэффициентов |",
                  "|---:|---:|---:|---:|---:|"])
    for row in rows:
        hidden = row["width"]
        dense_parameters = 786 * hidden + 1
        active_parameters = row["edges"] + 2 * hidden + 1
        lines.append(f"| {hidden} | {row['density']}% | "
                     f"{dense_parameters} | {active_parameters} | "
                     f"{dense_parameters / active_parameters:.1f}× |")
    lines.extend(["", "Счёт включает веса обоих слоёв и смещения. Маска "
                  "фиксирована и не считается обучаемым параметром. "
                  "В текущей реализации PyTorch хранит плотный тензор весов "
                  "и состояния оптимизатора: число выделенных параметров "
                  "такое же, как у плотного MLP. Для экономии памяти и "
                  "ускорения нужна разреженная реализация.", "",
                  "| Ширина | Плотность | Agreement − случайная, п. п. | "
                  "Agreement − средняя, п. п. | Agreement − плотный, п. п. |",
                  "|---:|---:|---:|---:|---:|"])
    for row in rows:
        m = row["accuracy"]
        lines.append(f"| {row['width']} | {row['density']}% | "
                     f"{100 * (m['agreement'] - m['random']):+.2f} | "
                     f"{100 * (m['agreement'] - m['mean']):+.2f} | "
                     f"{100 * (m['agreement'] - m['dense']):+.2f} |")
    lines.extend(["",
                  "![Точность](assets/2026-09-26/mnist8m_width_density_accuracy.png)", "",
                  "| Ширина | Плотность | VAE: top-K | VAE: лимит | "
                  "Agreement: top-K | Agreement: лимит |",
                  "|---:|---:|---:|---:|---:|---:|"])
    for row in rows:
        g, c = row["global_accuracy"], row["accuracy"]
        lines.append(f"| {row['width']} | {row['density']}% | "
                     f"{g['vae']:.4f} | {c['vae']:.4f} | "
                     f"{g['agreement']:.4f} | {c['agreement']:.4f} |")
    lines.extend(["",
                  "| Ширина | Плотность | Цифра | Плотный | Случайная | "
                  "Прямая | Средняя | VAE | Agreement |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in rows:
        for task, digit in ((0, 3), (1, 8)):
            m = row["per_task"][task]
            lines.append(f"| {row['width']} | {row['density']}% | {digit} | " +
                         " | ".join(f"{m[k]:.4f}" for k in METHODS) + " |")
    lines.extend(["",
                  "| Ширина | Плотность | Плотный | Случайная | Прямая | "
                  "Средняя | VAE | Agreement |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in rows:
        m = row["bce"]
        lines.append(f"| {row['width']} | {row['density']}% | " +
                     " | ".join(f"{m[k]:.4f}" for k in METHODS) + " |")
    lines.extend(["", "![BCE](assets/2026-09-26/mnist8m_width_density_bce.png)", "",
                  "| Ширина | Плотность | Покрыто пикселей: случайная | "
                  "Прямая | Средняя | VAE | Agreement |",
                  "|---:|---:|---:|---:|---:|---:|---:|"])
    for row in rows:
        c = row["coverage"]
        lines.append(f"| {row['width']} | {row['density']}% | " +
                     " | ".join(f"{c[k]:.0f}" for k in METHODS[1:]) + " |")
    lines.extend(["", "Покрытие пикселей для задачи 3; чёрный пиксель "
                  "имеет хотя бы одну связь с первым скрытым слоем.", "",
                  "![Покрытие пикселей]"
                  "(assets/2026-09-26/mnist8m_width_density_coverage.png)"])
    lines.extend(["", "| Ширина | Плотность | IoU реконструкции VAE | "
                  "IoU средней карты | Случайный IoU |",
                  "|---:|---:|---:|---:|---:|"])
    for row in rows:
        d = row["density"] / 100
        lines.append(f"| {row['width']} | {row['density']}% | "
                     f"{row['vae_iou']:.3f} | {row['mean_iou']:.3f} | "
                     f"{d / (2 - d):.3f} |")
    lines.extend(["", "IoU рассчитан на картах, отложенных при обучении VAE."])
    lines.extend(["", "| Ширина | Плотность | Шагов поиска z | "
                  "Лучший шаг z | Остановка z | Шагов нового обучения | "
                  "Поздний лучший чекпоинт MLP |",
                  "|---:|---:|---:|---:|---|---:|---:|"])
    for row in rows:
        lines.append(f"| {row['width']} | {row['density']}% | "
                     f"{row['search_steps']} | {row['search_best_step']} | "
                     f"{'плато' if row['search_plateau'] else 'лимит'} | "
                     f"{row['steps']} | {100 * row['late_best']:.1f}% |")
    best_gain = max(rows, key=lambda row: row["accuracy"]["agreement"] -
                    row["accuracy"]["random"])
    gain_pp = 100 * (best_gain["accuracy"]["agreement"] -
                     best_gain["accuracy"]["random"])
    lines.extend(["", "**Вывод.** Уменьшение ширины и плотности увеличило "
                  "разницу между прямыми и случайными масками: структура "
                  "связей стала важнее. Agreement тоже иногда превосходит "
                  f"случайную маску, максимум на {gain_pp:.2f} п. п. "
                  f"при {best_gain['width']} нейронах и {best_gain['density']}%. "
                  "При этом он во всех проверенных вариантах уступает "
                  "прямой карте банка и плотному MLP. IoU реконструкции VAE "
                  "остаётся близким к IoU простой средней карты. "
                  "Сокращение MLP показало ценность структуры маски, но "
                  "не устранило потерю информации при обучении VAE.", ""])
    lines.extend(["", "[Проверка выравнивания столбцов]"
                  "(MNIST8M_COLUMN_ALIGNMENT_2026-09-26.md) · "
                  "[Код](../pattern/mnist8m_raw_mlp_bce.py) · "
                  "[запуск](../pattern/run_mnist8m_width_sweep.py) · "
                  "[сырые результаты](../pattern/outputs/mnist8m_raw_mlp_bce/)", ""])
    REPORT.write_text("\n".join(lines))
    print(REPORT)


if __name__ == "__main__":
    main()
