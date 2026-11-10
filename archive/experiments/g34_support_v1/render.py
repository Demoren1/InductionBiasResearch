"""Render fixed versus free support, retaining the same fixed zero symbol."""

import json
import shutil
from pathlib import Path

import numpy as np

from experiments.sharing_structure_research_v1.blind_contract import save_json

from .report import METHODS, model_folder
from .run import PROJECT


COLORS = {"current": "#333333", "free": "#2878b5", "dense": "#999999"}
SHORT = {"current": "G34 fixed mask", "free": "Free mask, same zero", "dense": "Dense"}


def render(base):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = base / "report"
    scored = json.loads((output / "scoring.json").read_text())
    reference = Path(
        json.loads((base / "registration.json").read_text())["reference"]
    )
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    artifact = PROJECT / "artifacts/g34_support_2026-11-08"
    artifact.mkdir(parents=True, exist_ok=True)
    groups = [(v, c) for v in ("local", "quartet_mix", "random") for c in (0, 1)]

    def subset(collection, variant=None, condition=None, method=None):
        return [
            r for r in scored[collection]
            if (variant is None or r["variant"] == variant)
            and (condition is None or r["condition"] == condition)
            and (method is None or r.get("method") == method)
        ]

    def loss(variant, condition, method, budget=1024, stage="Adam+GN"):
        return np.asarray([
            r["mse"] for r in subset("transfer", variant, condition, method)
            if r["support"] == budget and r["stage"] == stage
        ])

    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(plots / name, dpi=150)
        plt.close(fig)

    primary = []
    fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharey="row")
    for ax, (variant, condition) in zip(axes.flat, groups):
        means = {m: float(loss(variant, condition, m).mean()) for m in METHODS}
        primary.append({
            "variant": variant, "condition": condition, **means,
            "free_change_percent": 100 * (means["free"] / means["current"] - 1),
        })
        for method in METHODS:
            ax.plot(
                (64, 256, 1024),
                [max(float(loss(variant, condition, method, n).mean()), 1e-16)
                 for n in (64, 256, 1024)],
                marker="o", label=SHORT[method], color=COLORS[method],
            )
        ax.set_xscale("log", base=4)
        ax.set_xticks((64, 256, 1024), ("64", "256", "1024"))
        ax.set_yscale("log")
        ax.set_title(f"{variant}, source noise {0.2 * condition:g}")
        ax.set_xlabel("Target fitting examples")
        ax.set_ylabel("Clean target MSE, Adam+GN")
    axes[0, 0].legend(fontsize=8)
    finish(fig, "target_quality.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for method in METHODS:
        train, selection, steps = [], [], None
        for case in scored["metadata"]:
            for condition in (0, 1):
                folder = model_folder(base, reference, case, condition, method)
                curve = np.load(folder / "curves.npz")
                train.append(curve["train_history_1024"].mean(axis=(1, 2)))
                selection.append(
                    curve["validation_history_1024"].mean(axis=(1, 2))
                )
                steps = curve["steps_1024"]
        for ax, values in zip(axes, (train, selection)):
            ax.plot(steps, np.maximum(np.mean(values, axis=0), 1e-16),
                    color=COLORS[method], label=SHORT[method])
            ax.set_yscale("log")
            ax.set_xlabel("Adam step")
            ax.set_ylabel("Normalized MSE, all numerical starts")
    axes[0].set_title("Target train, N=1024")
    axes[1].set_title("Target numerical selection, N=256")
    axes[0].legend(fontsize=8)
    finish(fig, "target_curves.png")

    structural = ("current", "free")
    audit = json.loads((output / "index_audit.json").read_text())
    for condition in (0, 1):
        fig, axes = plt.subplots(6, 4, figsize=(14, 18))
        for row, (case, metadata) in enumerate(scored["metadata"].items()):
            item = audit[f"{case}_c{condition}"]
            for column, name in enumerate(("truth", *structural)):
                picture = axes[row, column].imshow(
                    item[name], cmap="PiYG", vmin=-4, vmax=4
                )
                axes[row, column].set_title(
                    f"{case}: {metadata['variant']}\n"
                    f"{'True I' if name == 'truth' else SHORT[name]}",
                    fontsize=8,
                )
                fig.colorbar(picture, ax=axes[row, column], fraction=0.046)
            picture = axes[row, 3].imshow(
                item["dense_source_t0_weight"], cmap="coolwarm"
            )
            axes[row, 3].set_title("Actual dense source t0 W; own order", fontsize=8)
            fig.colorbar(picture, ax=axes[row, 3], fraction=0.046)
        finish(fig, f"indexators_c{condition}.png")

    per_case = []
    for case, metadata in scored["metadata"].items():
        for condition in (0, 1):
            row = {
                "case": case, "variant": metadata["variant"],
                "condition": condition,
            }
            for method in METHODS:
                row[method] = float(np.mean([
                    r["mse"] for r in scored["transfer"]
                    if r["case"] == case and r["condition"] == condition
                    and r["method"] == method and r["stage"] == "Adam+GN"
                    and r["support"] == 1024
                ]))
            pair = [
                audit[f"{case}_c{condition}"][m] for m in ("current", "free")
            ]
            row["same_selected_index_after_truth_alignment"] = pair[0] == pair[1]
            per_case.append(row)
    save_json(output / "summary.json", {"primary": primary, "cases": per_case})
    lines = [
        "# Нужна ли G34 фиксированная бинарная support-маска",
        "",
        "8 ноября 2026. Нулевой символ сохранён в обоих сравниваемых вариантах.",
        "",
    ]
    interpretation = output / "interpretation.md"
    if interpretation.exists():
        lines.extend((interpretation.read_text().strip(), ""))
    lines.extend([
        "## Точная абляция",
        "",
        "| | Исходный G34 | Free mask |",
        "|---|---|---|",
        "| Свободные знаковые коды | Четыре | Те же четыре |",
        "| Фиксированный нулевой символ | Сохранён | Сохранён |",
        "| Нулевые позиции внутри joint projection | Заданы фиксированной M | Выбираются совместно с группами |",
        "| Символы на позиции | Только ноль либо только ±1,…,±4 согласно M | 0, ±1,…,±4 на каждой позиции |",
        "| Штраф, WLS, anchors, source descriptors, refit | Исходные | Те же |",
        "",
        "В free бинарная маска используется для начального состояния, затем "
        "не ограничивает обновления. Нули могут появляться и исчезать. "
        "Threshold-based alignment, исходные source anchors и четыре варианта "
        "инициализации сохранены. Проверяется необходимость фиксировать "
        "support в цикле, а не возможность удалить из всей процедуры "
        "любые пороговые инициализаторы.",
        "",
        "Оба банка сохраняют исходные anchors. Поэтому отдельно проверяется, "
        "выбрана новая проекция или унаследованная структура. Dense — только "
        "готовый контекстный контроль. Варианты удаления нулевого символа "
        "исключены по уточнению пользователя до новых target запусков.",
        "",
        "## Перенос на новые функции",
        "",
        "Clean query MSE после Adam+GN: N=1024 fitting, 256 отдельных "
        "numerical-selection, 2048 query примеров. Каждая строка — две "
        "структуры × восемь новых функций. Все численные параметры свежие; "
        "переносится только I. Плюс в Δ означает ухудшение относительно G34.",
        "",
        "| Геометрия | Source σ | G34 fixed mask | Free mask | Δ MSE | Dense |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in primary:
        lines.append(
            f"| {row['variant']} | {0.2 * row['condition']:g} | "
            f"{row['current']:.6g} | {row['free']:.6g} | "
            f"{row['free_change_percent']:+.2f}% | {row['dense']:.6g} |"
        )
    lines.extend([
        "",
        f"![Качество и размер target-выборки]({artifact / 'target_quality.png'})",
        "",
        "| N | Геометрия | Source σ | G34 fixed mask | Free mask | Dense |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    for budget in (64, 256):
        for variant, condition in groups:
            cells = " | ".join(
                f"{loss(variant, condition, m, budget).mean():.6g}" for m in METHODS
            )
            lines.append(f"| {budget} | {variant} | {0.2 * condition:g} | {cells} |")
    lines.extend([
        "",
        "## Восстановление структуры",
        "",
        "Mask metrics допускают лучшую перестановку hidden units; exact signed I "
        "дополнительно допускает знаки/перестановки кодов. Для маски и полного I "
        "выравнивания оптимизируются отдельно.",
        "",
        "| Source σ | Метод | Exact I | Exact mask | Precision | Recall | True I в proposals | True I запрошен в refit | Выбран исходный anchor |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for condition in (0, 1):
        for method in structural:
            rows = subset("recovery", condition=condition, method=method)
            n = len(rows)
            lines.append(
                f"| {0.2 * condition:g} | {SHORT[method]} | "
                f"{sum(r['exact'] for r in rows)}/{n} | "
                f"{sum(r['mask_exact'] for r in rows)}/{n} | "
                f"{np.mean([r['mask_precision'] for r in rows]):.2%} | "
                f"{np.mean([r['mask_recall'] for r in rows]):.2%} | "
                f"{sum(r['true_in_proposal_pool'] for r in rows)}/{n} | "
                f"{sum(r['true_joint_refit_requested'] for r in rows)}/{n} | "
                f"{sum(r['selected_initial_anchor'] for r in rows)}/{n} |"
            )
    lines.extend([
        "",
        "Полная разбивка по структурам, наличие истинной маски в банке, ARI "
        "шэринга, размеры proposal pool и принадлежность выбранного I "
        "новым proposals сохранены в recovery.csv.",
        "",
        "## Действительно ли свободный цикл меняет маску",
        "",
        "| Геометрия | Source σ | Запусков с изменённой конечной маской | Начальных связей | Конечных связей | Событий смены нуля |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for variant, condition in groups:
        rows = subset("geometry", variant, condition, "free")
        values = {
            field: np.mean([r[field] for r in rows])
            for field in (
                "fraction_changed_support", "mean_initial_active",
                "mean_final_active", "mean_support_flip_events",
            )
        }
        lines.append(
            f"| {variant} | {0.2 * condition:g} | "
            f"{values['fraction_changed_support']:.2%} | "
            f"{values['mean_initial_active']:.1f} | "
            f"{values['mean_final_active']:.1f} | "
            f"{values['mean_support_flip_events']:.1f} |"
        )
    lines.extend([
        "",
        "Значения усреднены по всем геометрическим стартам, не только победителю.",
        "",
        "## Стоимость и численная сходимость",
        "",
        "Данные, рестарты и шаги совпадают: до32 source refits×4×48GN; "
        "target18Adam×2000+4GN×48, N64/256/1024. Время измерено с операторным "
        "профилированием; отдельный hardware benchmark не повторялся. "
        "First-layer координат4 у обоих структурных вариантов и256 у Dense; "
        "сырые численные координаты не равны числу степеней свободы после gauges.",
        "",
        "| Метод | Среднее extraction после seeds | Среднее target время, три N | Среднее target TFLOPs |",
        "|---|---:|---:|---:|",
    ])
    for method in METHODS:
        rows = subset("costs", method=method)
        lines.append(
            f"| {SHORT[method]} | "
            f"{np.mean([r['new_extraction_seconds'] for r in rows]):.1f} с | "
            f"{np.mean([r['target_seconds_all_budgets'] for r in rows]):.1f} с | "
            f"{np.mean([r['target_torch_operator_flops'] for r in rows]) / 1e12:.3f} |"
        )
    lines.extend([
        "",
        f"![Train и numerical selection]({artifact / 'target_curves.png'})",
        "",
        "Кривые усредняют все Adam-рестарты, восемь функций и условия; это "
        "train/numerical-selection, не query-loss на каждом шаге. Итоговые "
        "независимые query-ошибки после GN показаны выше.",
        "",
        "## Индексаторы и реальные плотные веса",
        "",
        "Индексаторы выровнены только для аудита. Последняя колонка — настоящие "
        "плотные source-веса в собственном порядке нейронов.",
        "",
        f"![Без source noise]({artifact / 'indexators_c0.png'})",
        "",
        f"![Source noise 0.2]({artifact / 'indexators_c1.png'})",
        "",
        "## Теория и ограничения",
        "",
        "При фиксированных числах можно добавить ноль в обычное categorical "
        "assignment и выбирать его по тому же робастному cost. Хранить M "
        "отдельно не требуется: она определяется нулевыми символами I. "
        "Пространство со свободными нулевыми назначениями содержит все "
        "фиксированные маски; его глобальный минимум одного критерия "
        "не хуже. Это не гарантия конечной чередующейся процедуры G34.",
        "",
        f"Доказательства и разбор порогов: [THEORY_NOTES.md]({artifact / 'THEORY_NOTES.md'}).",
        "",
        "Ретроспективный известный cohort: две local структуры с2×2 блоками "
        "(32связи), две quartet_mix с четырёхмерными группами (128связей), "
        "две random с плотностью генерации0.55. D=H16,O1/2; во второй "
        "структуре каждой семьи независимые hidden biases; readout независим "
        "между задачами. Source noise0/.2; target query чистые. Это не свежая "
        "confirmation и не реальные данные. Две структуры на семейство "
        "ограничивают общность выводов.",
        "",
        "Learner не открывал teacher, source test или target query responses. "
        "Все новые I и predictions глобально зафиксированы перед scoring; "
        "source/target receipts проверены. Ядро G34 и рукопись не менялись; "
        "software tests не запускались.",
        "",
        "## Артефакты",
        "",
        f"- [Активный протокол]({artifact / 'ACTIVE_PROTOCOL.md'}), "
        f"[уточнение объёма]({artifact / 'scope_amendment.json'}), "
        f"[фиксация]({artifact / 'frozen_results.json'}).",
        f"- [Target MSE]({artifact / 'transfer.csv'}), "
        f"[структура]({artifact / 'recovery.csv'}), "
        f"[изменения масок]({artifact / 'geometry.csv'}), "
        f"[стоимость]({artifact / 'costs.csv'}).",
        f"- [Сводка]({artifact / 'summary.json'}), "
        f"[матрицы]({artifact / 'index_audit.json'}).",
        f"- Raw: {base}. Код: {Path(__file__).parent}.",
        "",
        "Для повторения использовать zero-preserving entrypoint "
        "experiments.g34_support_v1.resume: --phase setup с новой --base и "
        "--reference, затем --phase complete с той же --base. Исходная "
        "расширенная run.complete заменена этим entrypoint после уточнения "
        "пользователя. Отдельные фазы report: freeze, score, render.",
    ])
    for path in (
        *output.glob("*.csv"), output / "summary.json",
        output / "index_audit.json", output / "frozen_results.json",
        base / "ACTIVE_PROTOCOL.md", base / "scope_amendment.json",
        base / "registration.json",
        Path(__file__).with_name("THEORY_NOTES.md"), *plots.glob("*.png"),
    ):
        shutil.copy2(path, artifact / path.name)
    report = PROJECT / "G34_SUPPORT_MASK_REPORT_2026-11-08.md"
    report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"report": str(report), "artifacts": str(artifact)}))
