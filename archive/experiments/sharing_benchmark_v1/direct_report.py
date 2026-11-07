"""Compare frozen methods without any clustering of their learned bases."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.darts_structured_tasks_v1.data import digest, save_json

from .report import LABELS, number, save_figure

METHODS = (
    "consensus_dp",
    "lc",
    "lutq",
    "msr",
    "mtsr",
    "yeh_shared_soft",
    "yeh_shared_map",
    "oracle",
)
COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#d62728",
    "#8c564b",
    "#e377c2",
    "#2ca02c",
    "#9467bd",
    "#17becf",
)


def transfer_figure(summary, output):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for axis, profile in zip(axes, ("hidden", "native")):
        for method, color in zip(METHODS, COLORS):
            values = summary[profile][method]["transfer"]
            axis.plot(
                [1, 4, 16],
                [
                    max(values[str(size)]["population_mse"], 1e-15)
                    for size in (1, 4, 16)
                ],
                "o-",
                color=color,
                label=LABELS[method],
            )
        axis.set(
            xscale="log",
            yscale="log",
            title=profile,
            xlabel="Число support-примеров новой задачи",
            ylabel="Population MSE",
        )
        axis.set_xticks([1, 4, 16], [1, 4, 16])
        axis.grid(alpha=0.25)
    axes[1].text(
        0.04,
        0.06,
        "MSE < 1e−15 показана на уровне 1e−15",
        transform=axes[1].transAxes,
        fontsize=8,
    )
    axes[1].legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle(
        "Без кластеризации: сохранённое представление каждого метода + новые числа"
    )
    save_figure(fig, output, "transfer")


def heatmaps(root, output, profile, family):
    folder = root / profile / f"family_{family}"
    with np.load(folder / "target_after_freeze.npz") as archive:
        teacher = archive["teacher_weights_audit_only"][0]
    limit = np.abs(teacher).max()
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    for axis, method in zip(axes.flat, METHODS):
        seed = 0 if method == "oracle" else 11
        with np.load(folder / "audit" / f"{method}_seed_{seed}.npz") as archive:
            weights = archive["weights_support_4"]
        error = np.square(weights - teacher).sum() / teacher.shape[0]
        image = axis.imshow(
            weights,
            vmin=-limit,
            vmax=limit,
            cmap="RdBu_r",
            aspect="auto",
            interpolation="nearest",
        )
        axis.set_title(f"{LABELS[method]}\nMSE матрицы: {error:.3g}", fontsize=10)
        axis.set_xlabel("Вход")
        axis.set_ylabel("Выход")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, label="Вес")
    fig.suptitle(
        f"{profile}, family {family}: без кластеризации, первая новая задача, 4 support"
    )
    save_figure(fig, output, f"{profile}_weights")

    representations = ("oracle", "consensus_dp", "yeh_shared_soft", "msr", "mtsr")
    fig, axes = plt.subplots(
        3, len(representations), figsize=(16, 9), constrained_layout=True
    )
    for column, method in enumerate(representations):
        seed = 0 if method == "oracle" else 11
        with np.load(folder / "audit" / f"{method}_seed_{seed}.npz") as archive:
            basis = archive["basis"]
        limit = max(float(np.abs(basis).max()), 1e-12)
        for code in range(3):
            axis = axes[code, column]
            image = axis.imshow(
                basis[..., code],
                vmin=-limit,
                vmax=limit,
                cmap="RdBu_r",
                aspect="auto",
                interpolation="nearest",
            )
            axis.set_title(f"{LABELS[method]}\nU[..., {code + 1}]", fontsize=9)
            axis.set_xlabel("Вход")
            axis.set_ylabel("Выход")
        fig.colorbar(
            image, ax=axes[:, column].tolist(), shrink=0.5, label="Исходное значение U"
        )
    fig.suptitle(
        f"{profile}: исходные столбцы U без кластеризации и выравнивания латентных координат"
    )
    save_figure(fig, output, f"{profile}_bases")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / "report/summary.json"
    original = json.loads(source.read_text())
    summary = {
        profile: {method: rows[method] for method in METHODS}
        for profile, rows in original.items()
    }
    output = root / "report/direct"
    output.mkdir(exist_ok=True)
    gradient = {}
    for profile in ("hidden", "native"):
        rows = []
        for path in (root / profile).glob("family_*/audit/summary.json"):
            rows.extend(
                row
                for row in json.loads(path.read_text())["records"]
                if row["method"] == "msr"
            )
        gradient[profile] = {
            size: float(np.mean([row["msr_native_nine_steps"][size] for row in rows]))
            for size in ("1", "4", "16")
        }
    save_json(
        output / "summary.json",
        {
            "methods": summary,
            "msr_nine_sgd_steps_diagnostic": gradient,
            "source_summary_sha256": digest(source),
            "code_sha256": digest(Path(__file__)),
            "scope": "Existing frozen methods before optional row clustering; no training or hyperparameter changes. MSR/MTSR remain continuous U. Main numeric adaptation is common fresh exact LS, not a claim of verbatim author-code reproduction.",
        },
    )
    lines = [
        "# Сравнение без кластеризации",
        "",
        "Это срез уже завершённого запуска до любой кластеризации U. Обучение и гиперпараметры не менялись. MSR/MTSR остаются непрерывными методами; их U не превращается в I и не получает выдуманной метрики точного дискретного восстановления.",
        "",
        "**Статус воспроизведения:** это наши реализации алгоритмов в общей постановке, не запуск неизменённых скриптов авторов. Фиксированный ноль/знаковые пары и общий source-bank для LC/LUT-Q — адаптации. Yeh с общим P, ограничением словаря и температурным подбором — наша многозадачная адаптация. Её результат нельзя называть чистым воспроизведением оригинального Yeh. Добавленная кластеризация U отсутствует во всех строках этого отчёта.",
        "",
        "В основной таблице структура/U фиксируется, все численные параметры сбрасываются и заново fit-ятся точным МНК на четырёх support-примерах новой задачи. Это общий численный контроль представления. Он отличается от конечного числа SGD-шагов в исходной процедуре MSR.",
        "",
        "| Метод | Hidden: MSE4 | Без шума: MSE4 | Hidden: точный I |",
        "|---|---:|---:|---:|",
    ]
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["method", "hidden_mse4", "native_mse4", "hidden_exact_families"]
        )
        for method in METHODS:
            left, right = summary["hidden"][method], summary["native"][method]
            exact = (
                f"{left['exact_all_seeds_families']}/6"
                if "exact_cells" in left
                else "не дискретный I"
            )
            errors = [
                left["transfer"]["4"]["population_mse"],
                right["transfer"]["4"]["population_mse"],
            ]
            writer.writerow([method, *errors, exact])
            lines.append(
                f"| {LABELS[method]} | {number(errors[0])} | {number(errors[1])} | {exact} |"
            )
    lines.extend(
        [
            "",
            "## Численная адаптация MSR",
            "",
            f"Дополнительно в исходном аудите сохранено девять SGD-шагов из обученной MSR-инициализации: MSE4 hidden={gradient['hidden']['4']:.6g}, native={gradient['native']['4']:.6g}. Это диагностика на уже выбранных моделях: checkpoint и гиперпараметры предыдущего запуска выбирались по exact-LS validation, а source-support был 4/10, не 1 как в авторском rank1 few-shot протоколе. Поэтому даже эту строку нельзя выдавать за полную репродукцию авторского результата. Все support-размеры сохранены в summary.json.",
            "",
            "Непрерывный U описывает семейство матриц W=Uφ. Его столбцы могут смешиваться обратимым преобразованием без изменения семейства, поэтому прямое визуальное совпадение отдельных столбцов с one-hot teacher не требуется. Показаны исходные значения; кластеризация, polar-whitening, перестановка или поворот латентных координат к ним не применялись.",
            "",
            "[Перенос](transfer.png), [hidden-веса](hidden_weights.png), [native-веса](native_weights.png), [исходные hidden-U](hidden_bases.png), [исходные native-U](native_bases.png). Source/validation-кривые без кластеризации находятся в [основном отчёте](../selection_curves.png).",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    transfer_figure(summary, output)
    for profile, family in (("hidden", 201), ("native", 301)):
        heatmaps(root, output, profile, family)
    print(output / "REPORT.md")


if __name__ == "__main__":
    main()
