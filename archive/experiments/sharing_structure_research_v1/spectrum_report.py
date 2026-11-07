"""Scientific figures and interpretation of the registered G1 diagnostics."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import NullFormatter

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.sharing_benchmark_v1.report import save_figure

LABELS = {
    "gradient_profiles": "Сырой градиент: профили",
    "gradient_pca": "Сырой градиент: PCA",
    "population_preconditioned_profiles": "Известная covariance: профили",
    "ols_profiles": "МНК: полные профили",
    "ols_pca": "МНК: PCA-базис",
    "ols_pca_scores": "МНК: PCA с амплитудами",
    "gradient_cross_spectrum": "Градиент: cross-split спектр",
    "ols_cross_spectrum": "МНК: cross-split спектр",
    "ols_noise_debiased_spectrum": "МНК: шумовая коррекция спектра",
}
COLORS = {
    "gradient_profiles": "#777777",
    "gradient_pca": "#555555",
    "population_preconditioned_profiles": "#cc79a7",
    "ols_profiles": "#0072b2",
    "ols_pca": "#e69f00",
    "ols_pca_scores": "#8c564b",
    "gradient_cross_spectrum": "#aaaaaa",
    "ols_cross_spectrum": "#009e73",
    "ols_noise_debiased_spectrum": "#d55e00",
}
SELECTED = (
    "gradient_pca",
    "ols_profiles",
    "ols_pca",
    "ols_cross_spectrum",
    "ols_noise_debiased_spectrum",
)


def select(rows, **conditions):
    return [
        r for r in rows if all(r[name] == value for name, value in conditions.items())
    ]


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    return {
        method: {
            "cases": len(values),
            "exact": sum(v["recovery"]["exact_I"] for v in values),
            "mask_f1": float(np.mean([v["recovery"]["mask_f1"] for v in values])),
            "I_errors": float(
                np.mean([v["recovery"]["categorical_errors"] for v in values])
            ),
            "subspace_error": float(
                np.mean([v["spectral"]["subspace_error"] for v in values])
            ),
            "minimum_subspace_cosine": float(
                np.mean([v["spectral"]["minimum_subspace_cosine"] for v in values])
            ),
        }
        for method, values in grouped.items()
    }


def figures(root, output, rows):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    positions = np.arange(len(SELECTED))
    for axis, condition in zip(axes, (1.0, 100.0)):
        summary = summarize(
            select(
                rows, condition=condition, excitation="full", noise=0.2, n=128, tasks=32
            )
        )
        axis.barh(
            positions,
            [summary[m]["exact"] / summary[m]["cases"] for m in SELECTED],
            color=[COLORS[m] for m in SELECTED],
        )
        axis.set_yticks(positions, [LABELS[m] for m in SELECTED])
        axis.invert_yaxis()
        axis.set(
            xlim=(0, 1.05),
            xlabel="Доля точно восстановленных I",
            title=f"Ковариация входов: condition={condition:g}",
        )
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle("Одинаковые 32 функции ×128 примеров; шум0.2; 6 геометрий")
    save_figure(fig, output, "input_covariance")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for i, balanced in enumerate((True, False)):
        for j, noise in enumerate((0.2, 1.0)):
            axis = axes[i, j]
            chosen = [
                r
                for r in select(
                    rows, condition=100.0, excitation="weak", noise=noise, tasks=128
                )
                if (r["family"] < 4200) == balanced
            ]
            sizes = sorted({r["n"] for r in chosen})
            for method in SELECTED:
                values = [select(chosen, method=method, n=n) for n in sizes]
                exact = [
                    np.mean([v["recovery"]["exact_I"] for v in group])
                    for group in values
                ]
                axis.plot(
                    sizes, exact, marker="o", label=LABELS[method], color=COLORS[method]
                )
            axis.set(
                xscale="log",
                ylim=(-0.03, 1.03),
                xlabel="Примеров на source-функцию",
                ylabel="Доля точных I",
                title=f"{'14/14/14 связей' if balanced else '48/12/3 связи'}; шум={noise:g}",
            )
            axis.set_xticks(sizes, list(map(str, sizes)))
            axis.xaxis.set_minor_formatter(NullFormatter())
            axis.grid(alpha=0.2)
    axes[1, 1].legend(fontsize=8)
    fig.suptitle(
        "Слабый третий код: std1/0.3/0.03; 128 функций; covariance condition100"
    )
    save_figure(fig, output, "weak_group_robustness")

    path = root / "showcases" / "family_4201_n8192.npz"
    with np.load(path) as archive:
        state = dict(archive)
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    weights = state["source_codes_weight_matrix"].reshape(128, -1)
    truth_spectrum = np.linalg.eigvalsh(weights.T @ weights / len(weights))[::-1]
    axis.plot(
        np.arange(1, 9),
        np.maximum(truth_spectrum[:8], 1e-7),
        "k--o",
        label="Истинный сигнальный спектр: только аудит",
    )
    for method in ("ols_pca", "ols_cross_spectrum", "ols_noise_debiased_spectrum"):
        spectrum = state[f"{method}_spectrum"][::-1]
        axis.plot(
            np.arange(1, 9),
            np.maximum(spectrum[:8], 1e-7),
            marker="o",
            color=COLORS[method],
            label=LABELS[method],
        )
    axis.axvline(3.5, color="gray", linestyle=":")
    axis.set(
        yscale="log",
        xlabel="Номер направления",
        ylabel="Eigenvalue, значения<1e−7 обрезаны",
        title="Первое несбалансированное семейство4201: слабый код скрыт шумом",
    )
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    save_figure(fig, output, "weak_direction_spectrum")

    cmap = ListedColormap(
        ["#9e0142", "#f46d43", "#fee08b", "#f7f7f7", "#abdda4", "#66c2a5", "#3288bd"]
    )
    norm = BoundaryNorm(np.arange(-3.5, 4), cmap.N)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for axis, method in zip(axes.flat, ("truth", *SELECTED)):
        if method == "truth":
            index, label, errors = state["truth"], "Истинный I", 0
        else:
            index, label = state[f"{method}_aligned_indices"], LABELS[method]
            errors = select(
                rows,
                family=4201,
                condition=100.0,
                excitation="weak",
                noise=1.0,
                n=8192,
                tasks=128,
                method=method,
            )[0]["recovery"]["categorical_errors"]
        image = axis.imshow(
            index, cmap=cmap, norm=norm, interpolation="nearest", aspect="auto"
        )
        axis.set(title=f"{label}\nошибок I: {errors}", xlabel="Вход", ylabel="Выход")
    fig.colorbar(image, ax=axes.ravel().tolist(), ticks=np.arange(-3, 4), shrink=0.8)
    fig.suptitle(
        "Первое несбалансированное семейство4201; 128 функций ×8192 примера, шум1"
    )
    save_figure(fig, output, "diagnostic_indexators")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for method in SELECTED:
        row = select(
            rows,
            family=4201,
            condition=100.0,
            excitation="weak",
            noise=1.0,
            n=8192,
            tasks=128,
            method=method,
        )[0]
        loss = np.asarray(row["fit_loss_by_step"])
        axes[0].plot(
            np.arange(len(loss)),
            loss / max(loss[0], 1e-20),
            color=COLORS[method],
            label=LABELS[method],
        )
        axes[1].plot(
            np.arange(len(loss)),
            row["mask_recall_by_step"],
            color=COLORS[method],
            label=LABELS[method],
        )
    axes[0].set(
        xlabel="Шаг декодирования",
        ylabel="Ошибка профилей / ошибка первого шага",
        title="Реально измеренный fit декодера",
    )
    axes[1].set(
        xlabel="Шаг декодирования",
        ylabel="Recall ненулевых связей",
        ylim=(-0.02, 1.02),
        title="Recall не используется для выбора",
    )
    axes[1].legend(fontsize=7)
    for axis in axes:
        axis.grid(alpha=0.2)
    save_figure(fig, output, "decoder_loss_recall")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    paths = sorted((root / "cases").glob("job_*.json"))
    if len(paths) != 84:
        raise RuntimeError(f"Expected84 registered jobs, found{len(paths)}")
    rows = [r for path in paths for r in json.loads(path.read_text())["records"]]
    output = root / "report"
    output.mkdir(exist_ok=True)
    regimes = {
        "ordinary": summarize(
            select(rows, condition=1.0, excitation="full", noise=0.2, n=128, tasks=32)
        ),
        "correlated": summarize(
            select(rows, condition=100.0, excitation="full", noise=0.2, n=128, tasks=32)
        ),
        "weak_2048": summarize(
            select(
                rows, condition=100.0, excitation="weak", noise=1.0, n=2048, tasks=128
            )
        ),
        "weak_8192": summarize(
            select(
                rows, condition=100.0, excitation="weak", noise=1.0, n=8192, tasks=128
            )
        ),
        "unidentifiable": summarize(
            select(rows, excitation="collision", noise=0.0, n=2048, tasks=128)
        ),
    }
    save_json(
        output / "summary.json",
        {
            "regimes": regimes,
            "records": len(rows),
            "cases": len(rows) // len(LABELS),
            "max_OLS_identity_error": max(
                r["normal_equation_relative_error"] for r in rows
            ),
            "max_optimum_gradient_ratio": max(
                r["optimum_to_initial_gradient_norm"] for r in rows
            ),
        },
    )
    with (output / "comparison.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "family",
                "condition",
                "excitation",
                "noise",
                "n",
                "tasks",
                "method",
                "exact_I",
                "I_errors",
                "mask_f1",
                "subspace_error",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    *[
                        row[k]
                        for k in (
                            "family",
                            "condition",
                            "excitation",
                            "noise",
                            "n",
                            "tasks",
                            "method",
                        )
                    ],
                    row["recovery"]["exact_I"],
                    row["recovery"]["categorical_errors"],
                    row["recovery"]["mask_f1"],
                    row["spectral"]["subspace_error"],
                )
            )
    figures(root, output, rows)
    lines = [
        "# G1: информация о структуре в градиентах и спектре",
        "",
        "Диагностический этап активной исследовательской цели. 876 условий,9 базовых способов извлечения,7884 результата;5 свободных V1000/1/2/3/6. Это не подтверждение нового метода и не сравнение с обученными Yeh/MSR/WSR. Теория записана до запуска в THEORY.md.",
        "",
        "Шесть независимых геометрий:3 скрытые переставленные свёртки и3 произвольных знаковых структуры. Последние имеют группы48/12/3; их число связей не передаётся алгоритмам. Известны shape14×16 иK3. Raw-gradient, обычный МНК, PCA, cross-split и шумовая коррекция — контроли, не заявления о новизне. Популяционное предобусловливание получает известную covariance и является отдельной oracle-design диагностикой.",
        "",
        "## Основные срезы",
        "",
        "| Режим | Метод | Точный I | F1 маски | Ошибка подпространства |",
        "|---|---|---:|---:|---:|",
    ]
    for regime, summary in regimes.items():
        for method, row in summary.items():
            lines.append(
                f"| {regime} | {LABELS[method]} | {row['exact']}/{row['cases']} | {row['mask_f1']:.3f} | {row['subspace_error']:.3f} |"
            )
    lines += [
        "",
        "ordinary: condition1, noise0.2,32 функции×128 примеров. correlated отличается только condition100. weak2048/8192: std кодов1/0.3/0.03, noise1,128функций, condition100. unidentifiable: первые два кода равны во всех source-функциях,2048примеров×128функций, без шума.",
        "",
        "## Диагноз",
        "",
        "1. Сырой градиент содержит ковариацию входов. При condition100 его профили/PCA не восстанавливают ни одной геометрии ordinary-среза, тогда как МНК-варианты дают6/6. Усечение спектра само эту неоднозначность не исправляет.",
        "2. Полные обычные профили являются сильным контролем. При достаточной вариации кодов простая оценка весов уже извлекает I; выдавать это за новый градиентный метод нельзя.",
        "3. Редкая и слабо возбуждённая группа может выпасть изtop-K. В сложном8192-срезе полные МНК-профили/PCA/cross-split дают3/6, шумовая коррекция5/6. Оставшееся семейство4201 имеет cosine слабого направления около0.14 даже после коррекции; это прежде всего ошибка оценивания подпространства, а не только ближайшего центроида.",
        "4. Слепое whitening по найденным eigenvalues усиливает шумовой третий вектор: у4201 профильный декодер делает21ошибку, PCA59, коррекция69. В двух других несбалансированных геометриях коррекция действительно извлекла весь I. Один общий агрегат скрывал бы эту разницу.",
        "5. В collision-контроле ни один способ не восстановил истинный I. Это ожидаемая неидентифицируемость групп, а не доказательство слабости конкретного оптимизатора. Больше наблюдений тех же функций не создадут различия совпадающих кодов.",
        "6. Предобусловленный эмпирический градиент совпал с МНК, а средний градиент в МНК-оптимуме исчез. Численные максимумы сохранены в summary.json. Новое направление должно учитывать эти отрицательные результаты.",
        "",
        "## Ограничения и следующий шаг",
        "",
        "Данные этого этапа открыты для диагностики и выбора гипотез, поэтому не будут использоваться как независимое подтверждение будущего метода. Новых целевых функций с resetC здесь не оценивали: эта проверка обязательна в следующих этапах. Нет гарантии, что локальныйLloyd найдёт оптимальную дискретную проекцию. Cross-split и noise-debiasing имеют известные статистические аналоги; их самостоятельная новизна не утверждается.",
        "",
        "Следующий механизм должен устранять измеренную потерю слабой/редкой группы, а не просто переименовывать PCA или менять градиентный шаг. Перед реализацией нужны теория группового остатка и аудит близких gradient-untying/heteroscedastic-PCA методов.",
        "",
        "![Ковариация входов](input_covariance.png)",
        "",
        "![Устойчивость слабой группы](weak_group_robustness.png)",
        "",
        "![Спектр](weak_direction_spectrum.png)",
        "",
        "![Индексаторы](diagnostic_indexators.png)",
        "",
        "![Loss и recall](decoder_loss_recall.png)",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    save_json(
        root / "complete.json",
        {
            "stage": "G1 only; overall goal remains active",
            "job_sha256": {str(p.relative_to(root)): digest(p) for p in paths},
        },
    )
    print(
        json.dumps(
            {
                name: {
                    method: f"{r['exact']}/{r['cases']}"
                    for method, r in summary.items()
                }
                for name, summary in regimes.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
