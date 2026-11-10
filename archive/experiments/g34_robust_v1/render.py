"""Render measured losses, structures, and the paired calibration report."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from experiments.sharing_structure_research_v1.blind_contract import save_json

from .report import ALL_METHODS, LABELS, METHODS, model_folder
from .run import PROJECT


COLORS = {
    "current": "#444444",
    "guard": "#d28e24",
    "irls": "#206db0",
    "relative": "#26845b",
    "dense": "#8888aa",
}
SHORT_LABELS = {
    "current": "G34",
    "guard": "Guard",
    "irls": "IRLS floor",
    "relative": "IRLS rel.",
    "dense": "Dense",
}
ARTIFACT_NAME = "g34_robust_2026-11-08"
REPORT_NAME = "G34_ROBUST_CALIBRATION_REPORT_2026-11-08.md"


def render(base):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = base / "report"
    scored = json.loads((output / "scoring.json").read_text())
    reference = Path(
        json.loads((base / "registration.json").read_text())["reference"]
    )
    transfer, recovery, costs = [
        scored[name] for name in ("transfer", "recovery", "costs")
    ]
    variants = ("local", "quartet_mix", "random")
    groups = [(variant, c) for variant in variants for c in (0, 1)]
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    artifact = PROJECT / "artifacts" / ARTIFACT_NAME
    artifact.mkdir(parents=True, exist_ok=True)

    def rows(collection, variant, condition, method=None):
        return [
            row for row in collection
            if row["variant"] == variant
            and row["condition"] == condition
            and (method is None or row["method"] == method)
        ]

    def values(variant, condition, method, budget=1024, stage="Adam+GN"):
        return np.asarray([
            row["mse"]
            for row in rows(transfer, variant, condition, method)
            if row["support"] == budget and row["stage"] == stage
        ])

    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(plots / name, dpi=170)
        plt.close(fig)

    fig, axes = plt.subplots(
        3, 2, figsize=(11, 10.5), sharey="row"
    )
    for ax, (variant, condition) in zip(axes.flat, groups):
        means = [
            values(variant, condition, method).mean()
            for method in ALL_METHODS
        ]
        ax.bar(
            range(len(ALL_METHODS)), np.maximum(means, 1e-16),
            color=[COLORS[m] for m in ALL_METHODS],
        )
        ax.set_xticks(
            range(len(ALL_METHODS)), [SHORT_LABELS[m] for m in ALL_METHODS],
            rotation=15,
            fontsize=8,
        )
        ax.set_yscale("log")
        ax.set_title(f"{variant}, source noise {0.2 * condition:g}")
        ax.set_ylabel("Clean target MSE, N=1024")
    finish(fig, "target_quality.png")

    fig, axes = plt.subplots(3, 2, figsize=(11, 10.5))
    for ax, (variant, condition) in zip(axes.flat, groups):
        cases = sorted({
            row["case"] for row in rows(recovery, variant, condition)
        })
        for method in ALL_METHODS:
            histories = []
            for case in cases:
                folder = model_folder(
                    base, reference, case, condition, method
                )
                with np.load(folder / "curves.npz") as curve:
                    histories.append((
                        curve["train_history_1024"].mean(axis=(1, 2)),
                        curve["validation_history_1024"].mean(axis=(1, 2)),
                    ))
                    steps = curve["steps_1024"]
            train, validation = np.mean(histories, axis=0)
            ax.plot(
                steps, train, color=COLORS[method], label=LABELS[method]
            )
            ax.plot(
                steps, validation, color=COLORS[method],
                linestyle="--", alpha=0.7,
            )
        ax.set_yscale("log")
        ax.set_xlabel("Adam step")
        ax.set_ylabel("Normalized target MSE")
        ax.set_title(f"{variant}, source noise {0.2 * condition:g}")
    axes[0, 0].legend(fontsize=7)
    finish(fig, "target_loss.png")

    fig, axes = plt.subplots(3, 2, figsize=(11, 10.5))
    for ax, (variant, condition) in zip(axes.flat, groups):
        cases = sorted({
            row["case"] for row in rows(recovery, variant, condition)
        })
        for method in ALL_METHODS[:-1]:
            training, selection = [], []
            for case in cases:
                root = (
                    reference if method == "current" else base / method
                )
                path = (
                    root / "fits" / f"{case}_c{condition}"
                    / "joint_fitted.json"
                )
                for chunk in json.loads(path.read_text())["history"]:
                    training.append(np.asarray(chunk["train"]))
                    selection.append(np.asarray(chunk["validation"]))
            train = np.concatenate(training, axis=1).mean(1)
            validation = np.concatenate(selection, axis=1).mean(1)
            ax.plot(train, color=COLORS[method], label=LABELS[method])
            ax.plot(
                validation, color=COLORS[method], linestyle="--",
                alpha=0.7,
            )
        ax.set_yscale("log")
        ax.set_xlabel("Gauss-Newton step")
        ax.set_ylabel("Source refit MSE")
        ax.set_title(f"{variant}, source noise {0.2 * condition:g}")
    axes[0, 0].legend(fontsize=7)
    finish(fig, "source_loss.png")

    fig, axes = plt.subplots(3, 2, figsize=(11, 10.5))
    for ax, (variant, condition) in zip(axes.flat, groups):
        cases = sorted({
            row["case"] for row in rows(recovery, variant, condition)
        })
        for method in METHODS:
            histories = []
            for case in cases:
                runs = json.loads((
                    base / method / "fits" / f"{case}_c{condition}"
                    / "optimization.json"
                ).read_text())["runs"]
                histories.extend(np.asarray(run["q"]) for run in runs)
            length = max(len(curve) for curve in histories)
            padded = np.stack([
                np.pad(curve, (0, length - len(curve)), mode="edge")
                for curve in histories
            ])
            lo, median, hi = np.percentile(padded, (25, 50, 75), axis=0)
            ax.plot(
                median, color=COLORS[method], label=LABELS[method]
            )
            ax.fill_between(
                np.arange(length), lo, hi, color=COLORS[method],
                alpha=0.12,
            )
        ax.set_yscale("symlog", linthresh=1e-8)
        ax.set_xlabel("Calibration / assignment step")
        ax.set_ylabel("Full Q, median and quartiles")
        ax.set_title(f"{variant}, source noise {0.2 * condition:g}")
    axes[0, 0].legend(fontsize=7)
    finish(fig, "geometry_q.png")

    fig, axes = plt.subplots(3, 2, figsize=(11, 10.5))
    for ax, (variant, condition) in zip(axes.flat, groups):
        for offset, field, alpha, label in (
            (-0.18, "mask_precision", 0.45, "Precision"),
            (0.18, "mask_recall", 1.0, "Recall"),
        ):
            means = [
                np.mean([
                    row[field]
                    for row in rows(recovery, variant, condition, method)
                ])
                for method in ALL_METHODS[:-1]
            ]
            ax.bar(
                np.arange(4) + offset, means, width=0.36, alpha=alpha,
                color=[COLORS[m] for m in ALL_METHODS[:-1]], label=label,
            )
        ax.set_xticks(
            range(4), [SHORT_LABELS[m] for m in ALL_METHODS[:-1]],
            rotation=15, fontsize=8,
        )
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{variant}, source noise {0.2 * condition:g}")
        ax.set_ylabel("Selected support precision / recall")
    axes[0, 0].legend(fontsize=7)
    finish(fig, "structural_recall.png")

    audit = json.loads((output / "index_audit.json").read_text())
    fig, axes = plt.subplots(6, 6, figsize=(19, 18))
    for row, (case, metadata) in enumerate(scored["metadata"].items()):
        item = audit[f"{case}_c1"]
        for column, name in enumerate(
            ("truth", "current", "guard", "irls", "relative")
        ):
            picture = axes[row, column].imshow(
                item[name], cmap="PiYG", vmin=-4, vmax=4
            )
            axes[row, column].set_title(
                f"{case}: {metadata['variant']}\n"
                f"{'True I' if name == 'truth' else LABELS[name]}",
                fontsize=8,
            )
            fig.colorbar(
                picture, ax=axes[row, column], fraction=0.046
            )
        picture = axes[row, 5].imshow(
            item["dense_source_t0_weight"], cmap="coolwarm"
        )
        axes[row, 5].set_title("Actual dense source t0 W", fontsize=8)
        fig.colorbar(picture, ax=axes[row, 5], fraction=0.046)
    finish(fig, "indexators.png")

    summary = []
    lines = [
        "# Согласованная робастная калибровка G34",
        "",
        "8 ноября 2026. Парная ретроспективная абляция на общей синтетике.",
        "",
    ]
    interpretation = output / "interpretation.md"
    if interpretation.exists():
        lines.extend([interpretation.read_text().strip(), ""])
    lines.extend([
        "## Качество на новых функциях",
        "",
        "Clean query MSE после Adam+GN: 1024 support-примера, 256 отдельных "
        "selection-примеров, 2048 query-примеров. В каждой строке две "
        "структуры × восемь новых функций. Δ = (IRLS/G34 − 1) × 100%; "
        "отрицательное значение означает улучшение. В колонке Δ ниже "
        "используется дополнительный relative-вариант; результаты исходной "
        "IRLS-ветви с полом также показаны отдельно.",
        "",
        "| Геометрия | Source σ | G34 | Guard | IRLS floor | "
        "IRLS relative | Dense | Δ relative |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for variant, condition in groups:
        means = {
            method: float(values(variant, condition, method).mean())
            for method in ALL_METHODS
        }
        change = (
            100 * (means["irls"] / means["current"] - 1)
            if means["current"] > 1e-12 else None
        )
        relative_change = (
            100 * (means["relative"] / means["current"] - 1)
            if means["current"] > 1e-12 else None
        )
        summary.append({
            "variant": variant, "condition": condition, **means,
            "irls_change_percent": change,
            "relative_change_percent": relative_change,
        })
        cells = " | ".join(
            f"{means[m]:.7g}" for m in ALL_METHODS
        )
        delta = (
            f"{relative_change:+.2f}%"
            if relative_change is not None else "—"
        )
        lines.append(
            f"| {variant} | {0.2 * condition:g} | {cells} | {delta} |"
        )
    lines.extend([
        "",
        "Две структуры на геометрию не дают надёжной оценки разброса по "
        "популяции структур. Восемь функций с одним I не являются "
        "восемью независимыми структурами. Старые результаты этого набора "
        "были известны до опыта; это не fresh confirmation.",
        "",
        "![Качество](artifacts/" + ARTIFACT_NAME + "/target_quality.png)",
        "",
        "## Меньшие бюджеты адаптации",
        "",
        "Те же query-метрики после Adam+GN, при N=64/256. "
        "Все numerical selection-примеры дополнительные к N.",
        "",
        "| Геометрия | σ | N | G34 | IRLS floor | IRLS relative | Dense |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for variant, condition in groups:
        for budget in (64, 256):
            means = [
                values(variant, condition, method, budget).mean()
                for method in ("current", "irls", "relative", "dense")
            ]
            lines.append(
                f"| {variant} | {0.2 * condition:g} | {budget} | "
                + " | ".join(f"{value:.7g}" for value in means) + " |"
            )
    lines.extend([
        "",
        "## Отдельные структуры и восстановление I",
        "",
        "| Case | σ | MSE G34 | MSE guard | MSE IRLS floor | "
        "MSE IRLS relative | Ошибки I: G34 / guard / floor / relative |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    case_summary = []
    for case, metadata in scored["metadata"].items():
        for condition in (0, 1):
            means, errors = [], []
            for method in ALL_METHODS[:-1]:
                chosen = [
                    row["mse"] for row in transfer
                    if row["case"] == case
                    and row["condition"] == condition
                    and row["method"] == method
                    and row["support"] == 1024
                    and row["stage"] == "Adam+GN"
                ]
                means.append(float(np.mean(chosen)))
                metrics = next(
                    row for row in recovery
                    if row["case"] == case
                    and row["condition"] == condition
                    and row["method"] == method
                )
                errors.append(metrics["signed_entry_errors"])
            case_summary.append({
                "case": case, "variant": metadata["variant"],
                "condition": condition,
                **dict(zip(ALL_METHODS[:-1], means)),
                "irls_minus_current": means[2] - means[0],
                "relative_minus_current": means[3] - means[0],
            })
            lines.append(
                f"| {case} ({metadata['variant']}) | {0.2 * condition:g} | "
                + " | ".join(f"{value:.7g}" for value in means)
                + " | " + " / ".join(map(str, errors)) + " |"
            )
    lines.extend([
        "",
        "Ошибки I измерены после оптимального переименования кодов, смены "
        "их знаков и перестановки hidden-нейронов. Низкая функциональная "
        "ошибка не означает точное восстановление I.",
        "",
        "| Source σ | Метод | Exact I | True I в proposal pool | "
        "True I в доступном банке | True I реально отправлен в refit |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    for condition in (0, 1):
        for method in ALL_METHODS[:-1]:
            selected = [
                row for row in recovery
                if row["condition"] == condition and row["method"] == method
            ]
            counts = [
                sum(row[field] for row in selected)
                for field in (
                    "exact", "true_in_proposal_pool",
                    "true_in_refitted_bank", "true_joint_refit_requested",
                )
            ]
            lines.append(
                f"| {0.2 * condition:g} | {LABELS[method]} | "
                + " | ".join(f"{value}/6" for value in counts) + " |"
            )
    lines.extend([
        "",
        "Доступный банк включает исходные seed-состояния. Наличие true I "
        "в нём не означает, что этот I вошёл в первые 32 новых refit-слота. "
        "В recovery.csv также сохранена его фактическая source meta-ошибка.",
        "",
        "![Recall](artifacts/" + ARTIFACT_NAME + "/structural_recall.png)",
        "",
        "![Индексаторы и настоящие dense W](artifacts/"
        + ARTIFACT_NAME + "/indexators.png)",
        "",
        "## Фактическая оптимизация полного Q",
        "",
        "| Метод | Запусков | Шагов Q | Увеличений Q > 1e-10 | "
        "Линейных решений | Принятых полных GN fallback |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for method in METHODS:
        selected = [
            row for row in scored["geometry"] if row["method"] == method
        ]
        counts = [
            sum(row[field] for row in selected)
            for field in (
                "runs", "q_steps", "q_increases_above_1e-10",
                "linear_solves", "jacobian_fallback_accepted",
            )
        ]
        lines.append(
            f"| {LABELS[method]} | " + " | ".join(map(str, counts)) + " |"
        )
    lines.extend([
        "",
        "Число линейных решений учитывает итерации после общей исходной "
        "WLS-инициализации. Шаги Q относятся к принятым или сохранённым "
        "состояниям; отклонённые backtracking-предложения в эту "
        "траекторию не входят.",
        "",
        "Кривые ниже — медиана и квартильный диапазон действительного Q "
        "по всем стартам. После остановки кривая продолжается последним "
        "значением только для агрегирования, дополнительные шаги не "
        "считаются. Это не train/query loss функции. Relative использует "
        "другой знаменатель: абсолютные уровни его Q и Q с полом "
        "не следует сравнивать как одну и ту же целевую функцию.",
        "",
        "![Полный критерий](artifacts/" + ARTIFACT_NAME + "/geometry_q.png)",
        "",
        "## Стоимость",
        "",
        "| Геометрия | σ | Geometry G34, s | Geometry guard, s | "
        "IRLS floor, s | IRLS relative, s | "
        "Полная source-стоимость relative/G34 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for variant, condition in groups:
        geometry_costs = [
            np.mean([
                row["geometry_seconds"]
                for row in rows(costs, variant, condition, method)
            ])
            for method in ALL_METHODS[:-1]
        ]
        totals = [
            np.mean([
                row["source_total_seconds"]
                for row in rows(costs, variant, condition, method)
            ])
            for method in ("current", "relative")
        ]
        lines.append(
            f"| {variant} | {0.2 * condition:g} | "
            + " | ".join(f"{value:.1f}" for value in geometry_costs)
            + f" | {totals[1] / totals[0]:.2f}× |"
        )
    lines.extend([
        "",
        "Это измеренное wall time, не полный FLOP-счётчик. Geometry "
        "использует NumPy/SciPy на CPU. Общая стоимость включает ранее "
        "обученные dense-источники и seeds; их фактически не переобучали. "
        "Функциональный refit имеет прежний бюджет, но численный "
        "projection solver и остановка отличаются от исходного G34.",
        "",
        "![Source train/selection](artifacts/"
        + ARTIFACT_NAME + "/source_loss.png)",
        "",
        "![Target train/selection](artifacts/"
        + ARTIFACT_NAME + "/target_loss.png)",
        "",
        "Сплошные кривые — train, пунктир — отдельная numerical selection. "
        "Source-кривые усредняют refit-кандидаты и старты; target-кривые "
        "усредняют Adam-старты до GN. Они отличаются от loss выбранного "
        "лучшего запуска; таблица качества использует выбранные Adam+GN "
        "состояния.",
        "",
        "## Протокол и файлы",
        "",
        "Данные: local/quartet_mix/random, D=H=16, K=4, O=1/2; по две "
        "структуры. Local содержит повторяющиеся блоки 2×2; quartet_mix "
        "смешивает четвёрки двухмерных точек; random имеет плотность 0.55. "
        "Выходные веса нейронов свободны; во второй структуре каждой "
        "геометрии свободны и hidden biases. Один общий I, разные численные "
        "параметры у 16 source и 8 target-функций.",
        "",
        "Source: 4096 train, 512 numerical selection, 1024 meta. Те же "
        "4 anchors, 4 support masks и source-derived starts. "
        "Refit первых 32 кандидатов: 4 старта × 48 GN. Source meta "
        "выбирает I. На target переносится только I: 18 свежих Adam-стартов "
        "× 2000 шагов, 4 GN-старта × 48 шагов, N=64/256/1024, "
        "дополнительная numerical selection N/4. Все числа свежие.",
        "",
        "Контроль current использует сохранённые исходные результаты; "
        "guard/IRLS/relative рассчитаны заново. Исходные две ветви "
        "после запуска не менялись. Relative добавлен отдельным amendment "
        "после теоретического анализа пола, до любого нового target-fit "
        "или scoring; первые source-результаты уже были доступны. "
        "В середине расчёта пользователь "
        "разрешил все карты: изменены только ресурсы 6→8 GPU, один job "
        "на карту, что записано в resource_amendment.json.",
        "",
        "Исходный G34 core и рукопись не менялись. В диспетчер fleet.py "
        "добавлен явный список GPU и подхват процессов из данного плана; "
        "его default остаётся GPU0–5. Software tests не создавались и "
        "не запускались.",
        "",
    ])
    freeze_record = json.loads(
        (output / "frozen_results.json").read_text()
    )
    lines.append(
        f"Глобальный freeze: {freeze_record['prediction_count']} файлов "
        f"предсказаний, из них {freeze_record['new_prediction_count']} новых; "
        f"{len(freeze_record['receipts'])} access receipts, нарушений 0. "
        "Новые selected I и предсказания зафиксированы до нового scoring."
    )
    lines.extend([
        "",
        "Подробные численные результаты для всех бюджетов: "
        f"[transfer.csv](artifacts/{ARTIFACT_NAME}/transfer.csv); "
        f"[recovery.csv](artifacts/{ARTIFACT_NAME}/recovery.csv); "
        f"[geometry.csv](artifacts/{ARTIFACT_NAME}/geometry.csv); "
        f"[парные запуски Q](artifacts/{ARTIFACT_NAME}/paired_geometry.csv); "
        f"[состояния калибровки]"
        f"(artifacts/{ARTIFACT_NAME}/calibration_states.csv); "
        f"[costs.csv](artifacts/{ARTIFACT_NAME}/costs.csv).",
        "",
        "Формулы и условия гарантий: "
        "[SOLVER_NOTES.md](experiments/g34_robust_v1/SOLVER_NOTES.md). "
        "Предварительно зафиксированный "
        f"[протокол](artifacts/{ARTIFACT_NAME}/PROTOCOL.md).",
        "",
    ])
    save_json(output / "summary.json", {
        "groups": summary, "cases": case_summary,
    })
    for path in plots.glob("*.png"):
        shutil.copy2(path, artifact / path.name)
    for path in output.glob("*.csv"):
        shutil.copy2(path, artifact / path.name)
    for name in (
        "summary.json", "index_audit.json", "frozen_results.json",
    ):
        shutil.copy2(output / name, artifact / name)
    for name in (
        "PROTOCOL.md", "registration.json", "resource_amendment.json",
        "evaluation_registration.json", "relative_amendment.json",
    ):
        shutil.copy2(base / name, artifact / name)
    (PROJECT / REPORT_NAME).write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "report": str(PROJECT / REPORT_NAME),
        "artifacts": str(artifact),
    }))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    render(args.base.resolve())


if __name__ == "__main__":
    main()
