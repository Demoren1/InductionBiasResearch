"""Collect completed projection experiments without repeating numerical fits."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .codebook_projection import exact_signed_projection
from .data import BASE, digest, save_json
from .report import categorical, matched_columns, save_figure

SERIES = {
    "deepsets_projection_full": {
        "nearest": "Nearest + STE + centering",
        "task_centers": "Nearest + STE; C по задаче",
        "proximal": "Nearest с ценой ненулевого кода",
        "proximal_warm": "То же + dense warmup",
    },
    "deepsets_consensus_full": {
        "quadratic": "Lloyd + квадратичное согласование",
        "damped_admm": "Lloyd + ADMM eta0.25",
        "admm": "Lloyd + ADMM eta1",
    },
    "deepsets_consensus_exact": {
        "quadratic": "Точная проекция + квадратичный штраф",
        "damped_admm": "Точная проекция + ADMM eta0.25",
        "admm": "Точная проекция + ADMM eta1",
    },
}


def paired_projection(results, output):
    records, rankings = [], {}
    for name in ("deepsets_consensus_full", "deepsets_consensus_exact"):
        folder = results / name
        spec = json.loads((folder / "specification.json").read_text())
        rankings[name] = {
            row["configuration"]: row
            for row in json.loads((folder / "pilot/ranking.json").read_text())
        }
        with np.load(folder / "pilot/hard_fits.npz") as archive:
            for configuration in (6, 7, 14, 15, 22, 23):
                row = rankings[name][configuration]
                records.append(
                    {
                        "series": name,
                        "configuration": configuration,
                        "family": row["per_seed"][0]["family"],
                        "lambda": row["per_seed"][0]["nonzero_penalty"],
                        "selection_nmse": row["selection_mse"]
                        / spec["target_variance"],
                        "active_edges": [
                            int(np.count_nonzero(archive["indices"][item["row"]]))
                            for item in row["per_seed"]
                        ],
                        "groups_used": [
                            int(
                                np.count_nonzero(
                                    np.bincount(
                                        np.abs(archive["indices"][item["row"]]).ravel(),
                                        minlength=5,
                                    )[1:]
                                )
                            )
                            for item in row["per_seed"]
                        ],
                    }
                )
    folder = results / "deepsets_consensus_full"
    history = json.loads((folder / "pilot/history.json").read_text())
    by_step = {row["step"]: row for row in history}
    choices = json.loads((folder / "pilot_selection.json").read_text())["families"]
    probes = []
    with np.load(folder / "pilot/snapshots.npz") as snapshots:
        for entry in choices.values():
            for row in entry["per_seed"]:
                snap, root = row["snapshot"], row["root"]
                latent = torch.tensor(snapshots["latent"][snap, root : root + 1])
                dual = torch.tensor(snapshots["dual"][snap, root : root + 1])
                centers = torch.tensor(
                    snapshots["numeric"][snap, root : root + 1, 0, :4]
                )
                labels = torch.tensor(
                    snapshots["probabilities"][snap, root : root + 1].argmax(-1)
                )
                dictionary = torch.cat((centers[:, :1] * 0, centers, -centers), 1)
                quantized = dictionary.gather(1, labels.flatten(1)).reshape(
                    latent.shape
                )
                values = latent + dual - row["dual_step"] * (latent - quantized)
                rho = torch.tensor([by_step[row["search_step"]]["rho"][root]])
                charge = torch.tensor([row["nonzero_penalty"]])
                new_labels, new_centers, projected = exact_signed_projection(
                    values, 4, charge, rho
                )
                before = (values - quantized).square().mean() + 2 * charge[0] / rho[
                    0
                ] * (labels != 0).float().mean()
                after = (values - projected).square().mean() + 2 * charge[0] / rho[
                    0
                ] * (new_labels != 0).float().mean()
                probes.append(
                    {
                        "family": row["family"],
                        "seed": row["seed"],
                        "search_step": row["search_step"],
                        "lloyd_projection_objective": float(before),
                        "exact_projection_objective": float(after),
                        "old_counts": torch.bincount(
                            torch.where(labels > 4, 4 - labels, labels).abs().flatten(),
                            minlength=5,
                        ).tolist(),
                        "new_counts": torch.bincount(
                            torch.where(new_labels > 4, 4 - new_labels, new_labels)
                            .abs()
                            .flatten(),
                            minlength=5,
                        ).tolist(),
                        "old_centers": centers[0].tolist(),
                        "new_centers": new_centers[0].tolist(),
                    }
                )
    save_json(
        output / "paired_ablation.json",
        {
            "registered_setting_comparisons": records,
            "same_latent_projection_probes": probes,
            "probe_is_not_a_task_refit": True,
        },
    )
    figure, axis = plt.subplots(figsize=(8.5, 4.4), constrained_layout=True)
    for family, label in (
        ("quadratic", "eta0"),
        ("damped_admm", "eta0.25"),
        ("admm", "eta1"),
    ):
        pair = [
            next(
                row
                for row in records
                if row["series"] == name
                and row["family"] == family
                and row["lambda"] == 0.03
            )
            for name in ("deepsets_consensus_full", "deepsets_consensus_exact")
        ]
        axis.plot(
            [0, 1], [row["selection_nmse"] for row in pair], marker="o", label=label
        )
    axis.set_xticks([0, 1], ["Local Lloyd", "Exact scalar projection"])
    axis.set_ylabel("Pilot mean selection NMSE")
    axis.set_yscale("log")
    axis.set_title("Paired pilot: same seeds, rho100, lambda0.03, equal fit budget")
    axis.grid(alpha=0.2)
    axis.legend()
    save_figure(figure, output / "figures/paired_ablation")


def collect(results):
    output = results / "projection_research"
    output.mkdir(exist_ok=True)
    summaries = {
        name: json.loads((results / name / "summary.json").read_text())
        for name in SERIES
    }
    paired_projection(results, output)
    exact = results / "deepsets_consensus_exact"
    with np.load(exact / "aligned_states.npz") as archive:
        truth, order = archive["true_indices"], archive["row_order"]
    with np.load(exact / "teacher_audit_only.npz") as archive:
        values = np.r_[0.0, archive["numeric"][:4]]
    with np.load(results / "deepsets_refinement/selected_states.npz") as archive:
        dense = matched_columns(
            archive["dense_shared_adam_weights"], values[np.abs(truth)] * np.sign(truth)
        )
    figure, axes = plt.subplots(1, 5, figsize=(20, 5.7), constrained_layout=True)
    image = categorical(axes[0], truth[order], 4, "Истинный I")
    for axis, name, family, title in zip(
        axes[1:4],
        SERIES,
        ("proximal", "admm", "damped_admm"),
        (
            "Nearest + цена связи",
            "Согласование + Lloyd",
            "Согласование + точная проекция",
        ),
    ):
        primary = summaries[name]["families"][family]["primary"]
        with np.load(results / name / "aligned_states.npz") as archive:
            index = archive[f"row_{primary['row']}_indices"]
        image = categorical(
            axis,
            index[order],
            4,
            f"{title}\nошибок {primary['recovery']['categorical_errors']}/256",
        )
    figure.colorbar(
        image,
        ax=list(axes[:4]),
        location="bottom",
        ticks=range(-4, 5),
        shrink=0.55,
        fraction=0.045,
        pad=0.04,
    )
    maximum = np.abs(dense).max()
    image = axes[4].imshow(
        dense[order], cmap="RdBu_r", vmin=-maximum, vmax=maximum, aspect="auto"
    )
    axes[4].set_title("Плотный Adam: численные веса")
    axes[4].set_xlabel("Скрытый нейрон")
    figure.colorbar(image, ax=axes[4], shrink=0.65)
    save_figure(figure, output / "figures/indexators")
    lines = [
        "# Полный индексатор DeepSets: nearest, согласование и точная проекция",
        "",
        "На синтетическом DeepSets точная скалярная проекция внутри ADMM-подобного согласования восстановила полный истинный индексатор в 3/3 запусках. Поиск получил размер матрицы 16×16 и четыре свободных кода со знаковыми парами. Маска, группы и число 32 активных связей использовались только для аудита после выбора.",
        "",
        "## Исходная функция",
        "",
        "| Метод | Mean selection NMSE | Mean test NMSE,3seed | Точных масок | Точных I | Связей выбранного I | Mean test32 fresh C |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, labels in SERIES.items():
        for family, label in labels.items():
            row = summaries[name]["families"][family]
            lines.append(
                f"| {label} | {row['selection_nmse_mean']:.6g} | {np.mean([item['test_nmse'] for item in row['per_seed']]):.6g} | {row['exact_masks']}/3 | {row['exact_indexators']}/3 | {row['primary']['recovery']['active_edges']} | {row['fresh']['mean_test_nmse']:.6g} |"
            )
    lines.extend(
        [
            "",
            "![Выбранные индексаторы и Adam](figures/indexators.png)",
            "",
            "![Train/query успешного варианта](../deepsets_consensus_exact/figures/curves.png)",
            "",
            "У обычного nearest+STE ошибка уменьшилась, но структура осталась почти плотной. Согласование с локальным Lloyd выделило маску, однако могло оставлять пустой код и сливать группы. Точная одномерная проекция совместно выбирает нулевой префикс, центры и назначения без зависимости от инициализации центров. Это исправляет конкретное ограничение локальной проекции, а не доказывает глобальную сходимость нейросетевого поиска.",
            "",
            "![Парный контроль](figures/paired_ablation.png)",
            "",
            "Парный пилот сравнивает одинаковые настройки инициализаций/данных/rho/lambda/бюджета. Числа находятся в paired_ablation.json; там же однократные проекции одних и тех же сохранённых V показывают стоимость до и после замены Lloyd. Эти локальные пробы не являются обучением или task-loss после адаптации C.",
            "",
            "## Перенос фиксированного I",
            "",
            "Сначала выполнено парное сравнение на шести коэффициентных функциях, ранее использованных для отрицательного nearest-контроля. Затем, после фиксации метода и исходных индексаторов, зарегистрированы ещё шесть ранее не оценённых функций (seed-offset 11100000). На каждой функции заново инициализируются все семь численных параметров. Query выбирает численный checkpoint и старт из восьми; selection выбирает скорость и радиус инициализации из шести настроек. Итоговый test создаётся после фиксации численных решений.",
            "",
            "| Набор | Метод | Mean test NMSE, query-selected C | Mean test NMSE, all8 starts |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    transfer = {}
    for cohort, label in (
        ("transfer", "Парные 6 функций"),
        ("transfer_confirmation", "Независимые 6 функций"),
    ):
        transfer[cohort] = json.loads((exact / cohort / "summary.json").read_text())
        for method in (
            "damped_admm/learned",
            "damped_admm/random_sharing",
            "damped_admm/random_all",
            "oracle",
            "dense_adam",
        ):
            row = transfer[cohort]["aggregate"][method]
            lines.append(
                f"| {label} | {method} | {row['mean_query_selected_test_nmse']:.6g} | {row['mean_all_restarts_test_nmse']:.6g} |"
            )
    lines.extend(
        [
            "",
            "![Независимый перенос](../deepsets_consensus_exact/transfer_confirmation/figures/transfer_errors.png)",
            "",
            "![Train/query при переносе](../deepsets_consensus_exact/transfer_confirmation/figures/numerical_curves.png)",
            "",
            "Среднее по всем восьми стартам отдельно показывает численные срывы: наличие правильного I не гарантирует успех каждого запуска Adam. Истинный I является численным контролем. Различия между эквивалентными индексаторами возникают из-за конечного числа случайных инициализаций и выбора численных настроек; они не означают превосходства над истинной структурой.",
            "",
            "## Протокол, кривые и границы",
            "",
            "Сохранены train/query кривые по реальным шагам: [nearest](../deepsets_projection_full/figures/curves.png), [Lloyd](../deepsets_consensus_full/figures/curves.png), [точная проекция](../deepsets_consensus_exact/figures/curves.png), [новый перенос](../deepsets_consensus_exact/transfer_confirmation/figures/numerical_curves.png). На этой задаче регрессии Recall неприменим. Полные I/C/W находятся в selected_indexators.json каждой серии; исходные и согласованные I сохранены отдельно. Выравнивание с учителем применяется только после фиксации решений для аудита и рисунков.",
            "",
            "Результат относится к одному синтетическому учителю и смене его численных коэффициентов. На Pattern, синусах, изображениях и новых топологиях успешный вариант этой серии не проверялся. Скалярный DP имеет сложность O(K*N²), что ограничивает размер слоя. Поиск исследовательский на ранее изученной исходной функции; три seed измеряют вариативность поиска.",
            "",
            "Успешный вариант использует проекционное согласование вместо исходного softmax-DARTS. [LUT-Q](https://arxiv.org/pdf/1911.04951) уже содержит непрерывные веса, nearest-code, дискретный forward и STE. [ADMM-кластеризация](https://arxiv.org/abs/1811.01907) и [точный одномерный k-means](https://journal.r-project.org/articles/RJ-2011-015/) тоже известны. Новизна общей механики не заявляется. Формулы, условная гарантия nearest при разделённых кодах и DP-рекурсия: [PROJECTION_METHOD.md](../../../PROJECTION_METHOD.md).",
            "",
        ]
    )
    seconds, reused = 0.0, 0.0
    for name in SERIES:
        folder = results / name
        for phase in ("pilot", "confirmation"):
            seconds += json.loads((folder / phase / "complete.json").read_text())[
                "seconds"
            ]
        for name_fit in (
            "pilot/hard_fits.npz",
            "confirmation/hard_fits.npz",
            "fresh_calibration.npz",
            "fresh_evaluation.npz",
        ):
            with np.load(folder / name_fit) as archive:
                seconds += float(archive["seconds"])
        for cohort in ("transfer", "transfer_confirmation"):
            if not (folder / cohort / "summary.json").exists():
                continue
            for task in range(6):
                task_folder = folder / cohort / f"task_{task}"
                for kind in ("fixed", "dense"):
                    with np.load(task_folder / f"{kind}_fits.npz") as archive:
                        duration = float(archive["seconds"])
                    if (
                        kind == "dense"
                        and (task_folder / "dense_control_reuse.json").exists()
                    ):
                        reused += duration
                    else:
                        seconds += duration
    lines.extend(
        [
            f"Измеренное новое обучение всех зарегистрированных кандидатов, стартов и контролей переноса: {seconds / 60:.2f} мин. Идентичный dense-контроль стоимостью {reused / 60:.2f} мин повторно не обучался. Стоимость исходного Adam и прежних серий сюда не включена. Программные тесты не создавались и не запускались; Ruff изменённых Python-файлов прошёл.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines))
    save_json(
        output / "summary.json",
        {
            "series": summaries,
            "transfer": transfer,
            "new_training_seconds": seconds,
            "reused_control_seconds": reused,
        },
    )
    (output / "reporting_code_snapshot.py").write_bytes(Path(__file__).read_bytes())
    for folder in [output, *(results / name for name in SERIES)]:
        save_json(
            folder / "complete.json",
            {
                "sha256": {
                    str(path.relative_to(folder)): digest(path)
                    for path in folder.rglob("*")
                    if path.is_file() and path.name != "complete.json"
                }
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=BASE / "results/20261104")
    args = parser.parse_args()
    torch.set_num_threads(2)
    collect(args.results)


if __name__ == "__main__":
    main()
