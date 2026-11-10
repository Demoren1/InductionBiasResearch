"""Audit and report the general synthetic extension without replacing old results."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from experiments.sharing_structure_research_v1.blind_contract import save_json

from .report import (
    LABELS,
    METHODS,
    PROJECT,
    freeze,
    relative_reduction,
    score,
)


def render(base):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output, root = base / "report", base / "confirmation"
    scored = json.loads((output / "scoring.json").read_text())
    transfer, recovery, costs = [scored[k] for k in ("transfer", "recovery", "costs")]
    generation = json.loads((root / "private/generation.json").read_text())
    metadata = {row["name"]: row for row in generation["cases"]}
    noise = generation["noise"]
    datasets = list(dict.fromkeys(row["dataset"] for row in transfer))
    groups = [(dataset, condition) for dataset in datasets for condition in (0, 1)]
    colors = dict(zip(METHODS, ("#444444", "#d28e24", "#206db0", "#8888aa")))
    artifact = PROJECT / "artifacts" / "g34_general_2026-11-08"
    artifact.mkdir(parents=True, exist_ok=True)
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    labels = {dataset: dataset.removeprefix("general_") for dataset in datasets}
    primary = [
        row for row in transfer if row["stage"] == "Adam+GN" and row["support"] == 1024
    ]

    def selected(rows, dataset, condition, method=None):
        return [
            row
            for row in rows
            if row["dataset"] == dataset
            and row["condition"] == condition
            and (method is None or row["method"] == method)
        ]

    def values(dataset, condition, method, budget=1024, field="mse"):
        return np.asarray(
            [
                row[field]
                for row in selected(transfer, dataset, condition, method)
                if row["stage"] == "Adam+GN" and row["support"] == budget
            ]
        )

    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(plots / name, dpi=170)
        plt.close(fig)

    fig, axes = plt.subplots(
        len(datasets), 2, figsize=(11, 3.5 * len(datasets)), sharey="row"
    )
    for ax, (dataset, condition) in zip(axes.flat, groups):
        means = [values(dataset, condition, method).mean() for method in METHODS]
        ax.bar(
            range(4),
            np.maximum(means, 1e-16),
            color=[colors[method] for method in METHODS],
        )
        ax.set_xticks(range(4), [LABELS[m] for m in METHODS], rotation=15)
        ax.set_yscale("log")
        ax.set_ylabel("Clean target MSE, N=1024")
        ax.set_title(f"{labels[dataset]}, source noise {noise[condition]:g}")
    for row, dataset in enumerate(datasets):
        means = [
            values(dataset, condition, method).mean()
            for condition in (0, 1)
            for method in METHODS
        ]
        axes[row, 0].set_ylim(min(means) * 0.5, max(means) * 1.5)
    finish(fig, "target_quality.png")

    fig, axes = plt.subplots(len(datasets), 2, figsize=(11, 3.5 * len(datasets)))
    for ax, (dataset, condition) in zip(axes.flat, groups):
        cases = sorted({r["case"] for r in selected(primary, dataset, condition)})
        for method in METHODS:
            histories = []
            for case in cases:
                key = f"{case}_c{0 if method == 'dense' else condition}"
                with np.load(root / "targets" / key / method / "curves.npz") as curve:
                    histories.append(
                        (
                            curve["train_history_1024"].mean(axis=(1, 2)),
                            curve["validation_history_1024"].mean(axis=(1, 2)),
                        )
                    )
                    steps = curve["steps_1024"]
            train, validation = np.mean(histories, axis=0)
            ax.plot(steps, train, color=colors[method], label=LABELS[method])
            ax.plot(steps, validation, color=colors[method], linestyle="--", alpha=0.7)
        ax.set_yscale("log")
        ax.set_xlabel("Adam step")
        ax.set_ylabel("Normalized MSE")
        ax.set_title(f"{labels[dataset]}, source noise {noise[condition]:g}")
    axes[0, 0].legend(fontsize=8)
    finish(fig, "target_loss.png")

    fig, axes = plt.subplots(len(datasets), 2, figsize=(11, 3.5 * len(datasets)))
    for ax, (dataset, condition) in zip(axes.flat, groups):
        cases = sorted({r["case"] for r in selected(primary, dataset, condition)})
        for method in METHODS[:3]:
            train_histories, selection_histories = [], []
            for case in cases:
                folder = root / "fits" / f"{case}_c{condition}"
                path = (
                    folder / "joint_fitted.json"
                    if method == "current"
                    else folder / method / "fitted.json"
                )
                for chunk in json.loads(path.read_text())["history"]:
                    train_histories.append(np.asarray(chunk["train"]))
                    selection_histories.append(np.asarray(chunk["validation"]))
            train = np.concatenate(train_histories, axis=1).mean(1)
            validation = np.concatenate(selection_histories, axis=1).mean(1)
            ax.plot(train, color=colors[method], label=LABELS[method])
            ax.plot(validation, color=colors[method], linestyle="--", alpha=0.7)
        ax.set_yscale("log")
        ax.set_xlabel("Gauss-Newton step")
        ax.set_ylabel("Source refit MSE")
        ax.set_title(f"{labels[dataset]}, source noise {noise[condition]:g}")
    axes[0, 0].legend(fontsize=8)
    finish(fig, "source_loss.png")

    fig, axes = plt.subplots(len(datasets), 2, figsize=(11, 3.5 * len(datasets)))
    for ax, (dataset, condition) in zip(axes.flat, groups):
        for offset, field, alpha, label in (
            (-0.18, "mask_precision", 0.45, "Precision"),
            (0.18, "mask_recall", 1, "Recall"),
        ):
            means = [
                np.mean(
                    [r[field] for r in selected(recovery, dataset, condition, method)]
                )
                for method in METHODS[:3]
            ]
            ax.bar(
                np.arange(3) + offset,
                means,
                width=0.36,
                alpha=alpha,
                color=[colors[m] for m in METHODS[:3]],
                label=label,
            )
        ax.set_xticks(range(3), [LABELS[m] for m in METHODS[:3]], rotation=15)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Selected support precision / recall")
        ax.set_title(f"{labels[dataset]}, source noise {noise[condition]:g}")
    axes[0, 0].legend(fontsize=8)
    finish(fig, "structural_recall.png")

    audit = json.loads((output / "index_audit.json").read_text())
    fig, axes = plt.subplots(6, 5, figsize=(16, 18))
    for row, case in enumerate(metadata):
        item = audit[f"{case}_c1"]
        for col, name in enumerate(("truth", "current", "diversity", "theory")):
            picture = axes[row, col].imshow(item[name], cmap="PiYG", vmin=-4, vmax=4)
            axes[row, col].set_title(
                f"{case}: {metadata[case]['variant']}\n"
                f"{'True I' if name == 'truth' else LABELS[name]}",
                fontsize=9,
            )
            fig.colorbar(picture, ax=axes[row, col], fraction=0.046)
        picture = axes[row, 4].imshow(item["dense_source_t0_weight"], cmap="coolwarm")
        axes[row, 4].set_title("Dense source t0 W (own order)", fontsize=9)
        fig.colorbar(picture, ax=axes[row, 4], fraction=0.046)
    finish(fig, "indexators.png")

    lines = [
        "# G34 на более сложной общей синтетике",
        "",
        "8 ноября 2026 года. Расширение проверки теоретических предложений.",
        "",
    ]
    interpretation = output / "interpretation.md"
    if interpretation.exists():
        lines += [interpretation.read_text().strip(), ""]
    lines += [
        "## Что представляет собой датасет",
        "",
        "Шесть семейств сетей 16→16 ReLU→1/2 с четырьмя независимыми числами "
        "кодбука первого слоя. В каждом семействе один общий signed-индексатор I; "
        "коды, hidden biases, свободные выходные веса и offsets меняются между "
        "16 source-задачами и восемью новыми target-функциями.",
        "",
        "- **local**: восемь повторяющихся блоков 2×2; входы и hidden-нейроны переставлены.",
        "- **quartet_mix**: две четвёрки двухмерных точек, полные блоки связей "
        "внутри каждой четвёрки, диагональные блоки противоположного знака.",
        "- **random**: независимая вероятность связи 0.55; случайные коды и знаки.",
        "",
        "Варианты local и DeepSets могут иметь одну геометрию первого слоя, "
        "но здесь выходные веса каждого нейрона свободны и не повторяют pooled-голову. "
        "Это существенное отличие генератора функций. Во второй репликации "
        "каждого варианта дополнительно свободны истинные hidden biases.",
        "",
        "| Вариант | Case | Выходов | Teacher hidden bias | Source задач | Target функций |",
        "|---|---|---:|---|---:|---:|",
    ]
    for case, item in metadata.items():
        lines.append(
            f"| {item['variant']} | {case} | {item['replicate'] + 1} | "
            f"{item['bias_style']} | 16 | 8 |"
        )
    lines += [
        "",
        "Learner знает только D/H/K/O и наблюдения, не получает вариант, "
        "маску, роли входов, teacher-параметры и дисперсию шума. Один ответ "
        "на пример. True I и query-ответы открыты только после общего freeze.",
        "",
        "## Основное качество переноса",
        "",
        "Clean target MSE1024 после Adam+GN, selection256, query2048. "
        "В каждой ячейке среднее по двум указанным стратам, по восемь функций "
        "в каждом. Source noise 0/.2; targets общие для обоих условий. "
        "Положительный процент означает снижение MSE.",
        "",
        "| Вариант | Source σ | G34 | Diversity | Theory | Dense | Снижение MSE Theory |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    summary = []
    for dataset, condition in groups:
        means = {m: float(values(dataset, condition, m).mean()) for m in METHODS}
        change = relative_reduction(means["current"], means["theory"])
        display = "—" if change is None else f"{change:+.2f}%"
        lines.append(
            f"| {labels[dataset]} | {noise[condition]:g} | {means['current']:.5g} | "
            f"{means['diversity']:.5g} | {means['theory']:.5g} | "
            f"{means['dense']:.5g} | {display} |"
        )
        summary.append(
            {
                "dataset": dataset,
                "condition": condition,
                "source_noise": noise[condition],
                **means,
                "theory_improvement_percent": change,
            }
        )
    lines += [
        "",
        "Масштабы выходов различаются: общий raw-MSE по трём вариантам "
        "не используется для вывода о победителе. При MSE G34 ниже1e-12 "
        "относительный процент не приводится.",
        "",
        f"![Качество]({artifact / 'target_quality.png'})",
        "",
        "## Парные результаты по каждому семейству",
        "",
        "| Вариант | Case | σ | G34 MSE | Theory MSE | Снижение | Победы/равенства/поражения |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset, condition in groups:
        cases = sorted({r["case"] for r in selected(primary, dataset, condition)})
        for case in cases:
            rows = [
                r for r in primary if r["case"] == case and r["condition"] == condition
            ]
            old = np.asarray([r["mse"] for r in rows if r["method"] == "current"])
            new = np.asarray([r["mse"] for r in rows if r["method"] == "theory"])
            change = relative_reduction(float(old.mean()), float(new.mean()))
            display = "—" if change is None else f"{change:+.2f}%"
            wins, losses = (
                int((new < old - 1e-12).sum()),
                int((new > old + 1e-12).sum()),
            )
            lines.append(
                f"| {labels[dataset]} | {case} | {noise[condition]:g} | "
                f"{old.mean():.5g} | {new.mean():.5g} | {display} | "
                f"{wins}/{len(old) - wins - losses}/{losses} |"
            )
    lines += [
        "",
        "Равенство означает абсолютную разность не больше1e-12. Восемь target-задач "
        "разделяют извлечённый I и не являются восемью независимыми структурными "
        "подтверждениями. Два страта различаются размером выхода и bias-моделью; "
        "выделить причинный эффект каждого изменения по ним нельзя.",
        "",
        "## Меньшие выборки и нормированная ошибка",
        "",
        "| Вариант | σ | N | G34 MSE | Theory MSE | G34 NMSE | Theory NMSE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, condition in groups:
        for budget in (64, 256, 1024):
            cell = [
                float(values(dataset, condition, m, budget, field).mean())
                for field in ("mse", "nmse")
                for m in ("current", "theory")
            ]
            lines.append(
                f"| {labels[dataset]} | {noise[condition]:g} | {budget} | "
                + " | ".join(f"{value:.5g}" for value in cell)
                + " |"
            )
    lines += [
        "",
        "NMSE делит MSE каждой задачи на дисперсию её query-ответов, "
        "затем усредняется по задачам. "
        "Adam-only результаты также сохранены в CSV; основной критерий — Adam+GN.",
        "",
        "## Структура и покрытие банка",
        "",
        "| Вариант | σ | Метод | Exact I | True I в pool | True I в refit | Ошибок позиций | Mask P/R | Sharing ARI |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, condition in groups:
        for method in METHODS[:3]:
            rows = selected(recovery, dataset, condition, method)
            lines.append(
                f"| {labels[dataset]} | {noise[condition]:g} | {LABELS[method]} | "
                f"{sum(r['exact'] for r in rows)}/{len(rows)} | "
                f"{sum(r['true_in_proposal_pool'] for r in rows)}/{len(rows)} | "
                f"{sum(r['true_in_refitted_bank'] for r in rows)}/{len(rows)} | "
                f"{sum(r['signed_entry_errors'] for r in rows)} | "
                f"{100 * np.mean([r['mask_precision'] for r in rows]):.2f}% / "
                f"{100 * np.mean([r['mask_recall'] for r in rows]):.2f}% | "
                f"{np.mean([r['sharing_ari'] for r in rows]):.4f} |"
            )
    lines += [
        "",
        "Sharing ARI считается на общих ненулевых позициях, не оценивает "
        "знаки и пропущенные связи. Exact I учитывает нули, группы и относительные "
        "знаки с допустимыми глобальными gauge/hidden-перестановками. "
        "Pool и refit — разные бюджеты, истинный I открывается только при аудите.",
        "",
        f"![Precision и recall]({artifact / 'structural_recall.png'})",
        "",
        "## Кривые и стоимость",
        "",
        "Target train/selection усредняются по всем18 Adam starts и задачам; "
        "selection пунктиром. Ось X — шаги, GN входит в итоговые метрики.",
        "",
        f"![Target loss]({artifact / 'target_loss.png'})",
        "",
        "Source refit-кривые усредняют все подгоняемые предложения и старты; "
        "это не loss только выбранного I.",
        "",
        f"![Source loss]({artifact / 'source_loss.png'})",
        "",
        "| Вариант | σ | Метод | Source/search с | Target с, все бюджеты/8 задач | Peak CUDA MiB |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for dataset, condition in groups:
        for method in METHODS[:3]:
            rows = selected(costs, dataset, condition, method)
            lines.append(
                f"| {labels[dataset]} | {noise[condition]:g} | {LABELS[method]} | "
                f"{np.mean([r['source_seconds'] for r in rows]):.2f} | "
                f"{np.mean([r['target_seconds_all_budgets'] for r in rows]):.2f} | "
                f"{max(r['target_peak_mib'] for r in rows):.2f} |"
            )
    lines += [
        "",
        "Равен лимит32 новых refit, а не wall time. Source/search включает общие "
        "dense fits и исходный pool; новые варианты дополнительно оплачивают "
        "полный theory pool. Для diversity это верхний учёт общей стадии, не "
        "отдельный standalone-профиль. Target время — packed throughput с "
        "profiler overhead. Torch FLOPs в сырых metrics неполны и исключают "
        "NumPy/SciPy. Development на новом наборе не выполнялся.",
        "",
        "## Индексаторы и настоящие dense-веса",
        "",
        "Все шесть noisy-семейств. Выбранные I выровнены к истине только "
        "для отображения после freeze. Dense W — реальные веса source t0 "
        "в собственном порядке нейронов.",
        "",
        f"![Индексаторы]({artifact / 'indexators.png'})",
        "",
        "## Границы вывода и воспроизведение",
        "",
        "Конфигурация bank.py/variant.py буквально совпадает с предыдущей "
        "регистрацией Pattern/DeepSets. Пороги и квоты после известных "
        "результатов той проверки не менялись. Старые и новые данные "
        "не объединяются в независимое подтверждение. Теоретические error "
        "bounds не сертифицированы; жадные proposals не дают полного покрытия. "
        "Причинные эффекты relations/log/sign ветвей отдельно не изолированы.",
        "",
        f"[Первый отчёт]({PROJECT / 'G34_THEORY_EXPERIMENT_REPORT_2026-11-08.md'}), "
        f"[теория]({PROJECT / 'IDENTIFIABILITY_AND_BANK_DESIGN_2026-11-07.md'}), "
        f"[протокол]({artifact / 'protocol.md'}), "
        f"[регистрация]({artifact / 'method_registration.json'}), "
        f"[freeze]({artifact / 'frozen_results.json'}).",
        "",
        f"Таблицы: [transfer]({artifact / 'transfer.csv'}), "
        f"[recovery]({artifact / 'recovery.csv'}), [costs]({artifact / 'costs.csv'}). "
        "Компактная папка содержит evidence и рисунки; полный prediction-аудит "
        f"требует локального каталога {base}.",
    ]
    report = PROJECT / "G34_GENERAL_SYNTHETIC_REPORT_2026-11-08.md"
    diagnostic_path = output / "theory_diagnostic.json"
    if diagnostic_path.exists():
        diagnostic = json.loads(diagnostic_path.read_text())["rows"]
        detail = [
            "## Проверка достаточных условий из теории",
            "",
            "Это post-freeze диагностика с истинной маской и выравниванием "
            "dense-нейронов к teacher. Используются истинный ранг и только "
            "истинные активные позиции. Learner этих подсказок не получает; "
            "измерение не заменяет проверку всего алгоритма.",
            "",
            "По матрице логарифмических контрастов собственных dense-решений "
            "строится пространство истинного ранга. δP — операторная ошибка "
            "его проектора. Используются консервативные достаточные границы "
            "из теории; n_max — максимальное число появлений одного кода, "
            "Z — идеальные контрасты 16 teacher-задач.",
            "",
            "$$",
            r"\begin{aligned}",
            r"\delta_P &= \|\widehat P-P\|_2"
            r" < \frac{1}{8\sqrt{n_{\max}}},\\",
            r"\varepsilon &= \|\widehat Z-Z\|_2"
            r" < \frac{\sigma_r(Z)}{16\sqrt{n_{\max}}}.",
            r"\end{aligned}",
            "$$",
            "",
            "Последний столбец — измеренная спектральная ошибка, "
            "делённая на достаточный шумовой порог.",
            "",
            "| Вариант | Case | σ | Ранг | Все W(j) связны | δP | Порог δP | Ошибка / шумовой порог |",
            "|---|---|---:|---:|---|---:|---:|---:|",
        ]
        for row in diagnostic:
            detail.append(
                f"| {row['variant']} | {row['case']} | {row['source_noise']:g} | "
                f"{row['contrast_rank']} | "
                f"{'да' if row['all_pair_graphs_connected'] else 'нет'} | "
                f"{row['oracle_aligned_projector_error']:.4g} | "
                f"{row['sufficient_projector_bound']:.4g} | "
                f"{row['error_to_noise_bound']:.4g} |"
            )
        detail += [
            "",
            "Граф W(j) соединяет нейроны, содержащие код j и ещё один общий код. "
            "Его связность относится к восстановлению unsigned-групп при "
            "правильной маске и известных размерах. Знаковые профили и остальные "
            "показатели сохранены отдельно. Логарифмы ограничены снизу на1e-12 "
            "только в диагностике. Неисполнение достаточной границы "
            "не доказывает невозможность восстановления.",
            "",
            f"[Полная таблица условий]({artifact / 'theory_conditions.csv'}).",
            "",
        ]
        position = lines.index("## Кривые и стоимость")
        lines[position:position] = detail
    report.write_text("\n".join(lines) + "\n")
    save_json(output / "summary.json", {"primary": summary, "report": str(report)})
    for name in (
        "transfer.csv",
        "recovery.csv",
        "costs.csv",
        "summary.json",
        "index_audit.json",
        "frozen_results.json",
    ):
        shutil.copy2(output / name, artifact / name)
    for path in plots.glob("*.png"):
        shutil.copy2(path, artifact / path.name)
    for name, path in (
        ("protocol.md", base / "PROTOCOL.md"),
        ("method_registration.json", root / "theory_registration.json"),
        ("data_commitments.json", root / "data_commitments.json"),
    ):
        shutil.copy2(path, artifact / name)
    if interpretation.exists():
        shutil.copy2(interpretation, artifact / "interpretation.md")
    for name in ("theory_conditions.csv", "theory_diagnostic.json"):
        if (output / name).exists():
            shutil.copy2(output / name, artifact / name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--phase", choices=("freeze", "score", "render"), required=True)
    args = parser.parse_args()
    base = args.base.resolve()
    if args.phase == "freeze":
        freeze(base)
    elif args.phase == "score":
        score(base, render_report=False)
        from .general_diagnostic import audit

        audit(base)
        render(base)
    else:
        render(base)


if __name__ == "__main__":
    main()
