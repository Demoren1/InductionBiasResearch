"""Render compact evidence and figures without repeating any fitting."""

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

from .extract import group_diagnostics

LABELS = {
    "ratio": "Direct one-task ratios",
    "current": "G34",
    "dense": "Dense",
    "signature": "Weighted SVD",
}


def run(root, output):
    summary = json.loads((output / "summary.json").read_text())
    selection = json.loads((output / "models/ratio/selected.json").read_text())
    codes = selection["source_selection"]["codes"]
    origin = json.loads(
        (output / f"source_selection/k{codes}/selected.json").read_text()
    )["origin"]
    index = np.load(output / "models/ratio/model.npz")["index"]
    diagnostics = group_diagnostics(index, codes)
    save_json(output / "structure_diagnostics.json", diagnostics)
    extraction = json.loads((output / "extraction/metrics.json").read_text())
    source_metrics = [
        json.loads(path.read_text())
        for path in (output / "source_selection").glob("k*/metrics.json")
    ]
    dense_metrics = json.loads(
        (root / "source/dense/metrics.json").read_text()
    )
    validation_metrics = json.loads(
        (output / "development/metrics.json").read_text()
    )
    costs = {
        "common_dense_source_seconds": dense_metrics["wall_seconds"],
        "ratio_extraction_seconds": extraction["wall_seconds"],
        "source_calibration_sum_gpu_seconds": sum(
            r["wall_seconds"] for r in source_metrics
        ),
        "source_calibration_longest_job_seconds": max(
            r["wall_seconds"] for r in source_metrics
        ),
        "development_selection_seconds": validation_metrics["wall_seconds"],
        "target_adaptation_seconds": json.loads(
            (output / "models/ratio/metrics.json").read_text()
        )["wall_seconds"],
        "historical_g34_extraction_seconds": 215.59477520175278,
        "g34_time_provenance": (
            "12 November log, same source-model hash, unchanged real_project; "
            "single historical timing, not a fresh timing replicate"
        ),
        "timing_scope": "Single runs; preparation and HPO reported separately",
    }
    save_json(output / "costs.json", costs)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    order = ("dense", "current", "signature", "ratio")
    methods = {r["method"]: r for r in summary["methods"]}
    colors = ("#8b98a5", "#d88a31", "#7a72ac", "#2787aa")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4), layout="constrained")
    means = [100 * methods[m]["accuracy"]["mean"] for m in order]
    errors = [
        100
        * (methods[m]["accuracy"]["upper"] - methods[m]["accuracy"]["mean"])
        for m in order
    ]
    labels = ("Dense", "G34", "Weighted\nSVD", "Direct one-task\nratios")
    axes[0].bar(labels, means, yerr=errors, capsize=4, color=colors)
    for position, mean in enumerate(means):
        axes[0].text(
            position,
            mean + errors[position] + 0.6,
            f"{mean:.2f}%",
            ha="center",
            fontsize=10,
        )
    axes[0].set(
        ylabel="Query accuracy (%)",
        ylim=(0, max(means) + 6),
        title="Same 1000 Aug-Omniglot episodes",
    )
    axes[0].grid(axis="y", alpha=0.2)
    pairs = [
        summary["paired"][f"ratio_minus_{m}"]["accuracy"]
        for m in ("current", "dense", "signature")
    ]
    delta = [100 * r["mean"] for r in pairs]
    half = [100 * (r["upper"] - r["mean"]) for r in pairs]
    axes[1].errorbar(
        delta, np.arange(3), xerr=half, fmt="o", color=colors[-1], capsize=4
    )
    axes[1].set_yticks(np.arange(3), ("vs G34", "vs Dense", "vs weighted SVD"))
    axes[1].axvline(0, color=".5", linewidth=1)
    axes[1].set(
        xlabel="Paired accuracy difference (percentage points)",
        title="Direct ratios minus comparator; 95% CI",
    )
    axes[1].grid(axis="x", alpha=0.2)
    fig.suptitle(
        "5-way 5-shot · structure-only transfer · retrospective cohort"
    )
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"comparison.{suffix}", dpi=180)
    plt.close(fig)

    sizes = np.sort(diagnostics["group_sizes"])[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), layout="constrained")
    axes[0].bar(np.arange(1, codes + 1), sizes, color="#2787aa")
    axes[0].set(
        yscale="log",
        xlabel="Groups ordered by size",
        ylabel="Active connections (log scale)",
        title=f"Selected direct-ratio index: {codes} groups",
    )
    axes[0].text(
        0.97,
        0.95,
        f"Largest group: {100 * sizes[0] / sizes.sum():.2f}%\n"
        f"Singleton groups: {diagnostics['singleton_groups']}",
        ha="right",
        va="top",
        transform=axes[0].transAxes,
    )
    records = selection["all_budget_choices"]
    axes[1].plot(
        [r["codes"] for r in records],
        [r["validation_mean_loss"] for r in records],
        "o-",
        color="#2787aa",
        label="Direct ratios",
    )
    for method, color in (("current", colors[1]), ("dense", colors[0])):
        axes[1].axhline(
            methods[method]["development_ce"],
            color=color,
            linestyle="--",
            label=LABELS[method],
        )
    axes[1].set(
        xlabel="Code budget",
        ylabel="Development cross-entropy",
        title="Budget and numerical rule chosen on development",
    )
    axes[1].set_xticks([8, 16, 32, 64])
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"structure_and_development.{suffix}", dpi=180)
    plt.close(fig)

    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "method",
                "accuracy_percent",
                "ce",
                "codes",
                "task_parameters",
                "active_edges",
                "updates",
            )
        )
        for method in order:
            row = methods[method]
            writer.writerow(
                (
                    method,
                    100 * row["accuracy"]["mean"],
                    row["cross_entropy"]["mean"],
                    row["independent_first_weights"],
                    row["task_parameters"],
                    row["active_edges"],
                    row["numeric_updates"],
                )
            )
    table = []
    for method in order:
        row = methods[method]
        accuracy = row["accuracy"]
        half = 100 * (accuracy["upper"] - accuracy["mean"])
        table.append(
            f"| {LABELS[method]} | "
            f"{100 * accuracy['mean']:.2f} ± {half:.2f} | "
            f"{row['cross_entropy']['mean']:.4f} | "
            f"{row['independent_first_weights']} | {row['task_parameters']} | "
            f"{row['active_edges']} | {row['numeric_updates']} |"
        )
    paired = []
    for method in ("current", "dense", "signature"):
        value = summary["paired"][f"ratio_minus_{method}"]["accuracy"]
        paired.append(
            f"- Прямой метод − {LABELS[method]}: "
            f"{100 * value['mean']:+.3f} п.п., 95% CI "
            f"[{100 * value['lower']:+.3f}, {100 * value['upper']:+.3f}]."
        )
    artifact_relative = "../../artifacts/omniglot_single_task_2026-11-14"
    table_header = (
        "| Метод | Accuracy, % и 95% CI | CE | Коды / свободные W1 | "
        "Все task-параметры | Active edges | Updates |"
    )
    synthetic_report = "SINGLE_TASK_RECOVERY_EXPERIMENT_2026-11-14.md"
    new_accuracy = 100 * methods["ratio"]["accuracy"]["mean"]
    reference_accuracy = 100 * methods["current"]["accuracy"]["mean"]
    loss_to_reference = reference_accuracy - new_accuracy
    proposal_speedup = (
        costs["historical_g34_extraction_seconds"]
        / costs["ratio_extraction_seconds"]
    )
    code_check = ""
    diagnostic_path = output / "dominant_code_diagnostic.json"
    if diagnostic_path.exists():
        diagnostic = json.loads(diagnostic_path.read_text())
        target = diagnostic["target_fits"]
        magnitude = target["effective_absolute_weight_quantiles_05_50_95"][1]
        energy = target[
            "first_layer_squared_weight_energy_share_quantiles_05_50_95"
        ][1]
        code_check = f"""### Проверка большой группы после scoring

Из {diagnostic['total_connections']} связей явно обнулены маской
{diagnostic['explicit_zero_connections']} (50%). Большой ненулевой код
используют {diagnostic['dominant_group_size']} связей: 99.76% активных,
49.88% всех. Знаки различаются: +1 у
{diagnostic['dominant_positive_index']}, −1 у
{diagnostic['dominant_negative_index']} связей; остальные 15 групп — singleton.

На {target['tasks']} сохранённых target fits этот код оказался ровно нулём
{target['dominant_code_exactly_zero']} раз. Медианный модуль реального веса
большой группы {magnitude:.6f}; медианная доля в сумме квадратов весов
первого слоя {100 * energy:.2f}%. Последняя величина описывает сохранённую
параметризацию, а не долю вклада в logits. На 128 source fits код также
ни разу не равен нулю. Таким образом, большая группа не является вторым
нулевым кодом.

Почти все активные веса имеют одну общую амплитуду и фиксированные
относительные знаки. Сильное ограничение относительных модулей — вероятное
объяснение снижения качества. Сам размер группы не доказывает ошибку:
для подходящего семейства такая группа могла бы быть правильной.
Причинный вклад дисбаланса отдельно не изолирован. Возможный механизм
вырождения — случайные близкие scalar ratios и их транзитивное объединение;
для fitted-весов точные premises теоремы не удостоверены. Новых обучений
или изменений выбора при этой диагностике не проводилось.

"""
    text = f"""# Прямые отношения по одной задаче на Omniglot

14 ноября 2026. Контролируемое применение конструктивного правила к
реальным fitted-весам. Подбор выполнен только на source и development;
основной G34 и общие численные адаптеры не изменены.

**Главный вывод:** прямой метод получил {new_accuracy:.2f}% против
{reference_accuracy:.2f}% у G34, отставание {loss_to_reference:.2f} п.п.
Proposal-стадия быстрее примерно в {proposal_speedup:.2f} раза по двум
одиночным измерениям, но функциональное качество снизилось. Этот опыт
не поддерживает замену G34. От Dense нет установленного различия по
accuracy: парный интервал включает ноль; CE прямого метода лучше.

## Результаты

{table_header}
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

{chr(10).join(paired)}

Интервалы относятся к эпизодам и условны на одном source-ансамбле.
Это расширение уже оценённого 12 ноября cohort: новые предсказания
зафиксированы до нового scoring, но 1000 эпизодов не являются новым
независимым подтверждением. Исторические 37.73% G34 из исходной таблицы
7 ноября относятся к другому численному запуску; здесь baseline 41.02%.

## Как подбирались настройки

Сеть 784–16–5 ReLU; 5-way 5-shot, 25 support / 75 query, query-only
augmentation. Прежние 128 source и 128 development-эпизодов. F/H/M source
панели побайтово воспроизвелись. Весь dense-ансамбль также совпал с входом
12 ноября: e14a76b02530e0e88669a2b96359de7ea93ab58f411944669a744e741a37dad5.

Каждый proposal использует ровно одну source-задачу, 60 или 107, и
dispersion четырёх её fits; совместная SVD не применяется. Проверены
семь масок на anchor: full, 1/2/3×uncertainty, top25/50/75% в нейроне.
Масштаб нейрона исключён разностями log-модулей. Оба правила слияния
исполнены полностью через точный sorted scalar forest; контроль на двух
сохранённых source-наблюдениях дал одинаковые разбиения 8/8 с исходной
буквальной реализацией. Каждый threshold считается независимо.

Бюджеты 8/16/32/64; t=0 и 97 geometric thresholds 1e-12…1. Дополнительная
24-шаговая bisection используется только для поиска порога и не изменяет
группы. Глобальная монотонность не предполагается; это конечный поиск,
не исчерпывание всех возможных порогов. Все четыре бюджета получены;
неполные разбиения не достраивались. Извлечено и оценено 32 индексатора,
по восемь на бюджет, с разнообразием масок и robust geometric score.

Для каждого индексатора исходные численные параметры подогнаны на всех
128 source-задачах: 18 starts, 2000 Adam, один общий dense warm start.
H выбирает численные состояния; независимая M выбирает одну структуру
на бюджет. Поэтому «одна задача» описывает построение каждого кандидата,
а не весь объём данных для выбора банка. Development затем выбирает
бюджет, LR/quantile/update count, усредняя обе свежие инициализации.

Выбран m={codes}, anchor={origin["anchor"]}, mask={origin["mask"]},
threshold={origin["threshold"]:.10g}; development CE
{selection["validation_mean_loss"]:.6f}.
LR={selection["global_setting"]["lr"]},
activity quantile={selection["global_setting"]["quantile"]},
{selection["numeric_updates"]} updates. На target переданы только I/basis
и это общее правило; code values, bias и readout обучены заново по support.

## Структурная диагностика и границы вывода

Активных связей {int(np.count_nonzero(index))}; наибольшая группа содержит
{100 * diagnostics["largest_group_fraction"]:.2f}% всех активных связей,
singleton-групп {diagnostics["singleton_groups"]} из {codes}.
Все сохранённые кандидаты уже при source-only извлечении имели одну
крупную группу и небольшие остатки. Совпадения скалярных отношений и их
транзитивное замыкание на fitted-весах не удостоверяют истинный sharing.
У Omniglot нет известного истинного I; условия разделения и bound ошибки
наблюдаемых логарифмов здесь не проверены. Строгая теорема для точных
наблюдений не гарантирует успех на таком dense-ансамбле.

{code_check}\
## Стоимость и фиксация

- Общая source Dense-подготовка: {costs["common_dense_source_seconds"]:.2f} s.
- Ratio proposals: {costs["ratio_extraction_seconds"]:.2f} s CPU.
- Исторический G34 proposal run на том же dense SHA: 215.59 s;
  это одно старое измерение, без независимых timing repeats.
- Source calibration/selection: сумма GPU jobs
  {costs["source_calibration_sum_gpu_seconds"]:.2f} s; четыре бюджета шли
  параллельно, самая длинная job
  {costs["source_calibration_longest_job_seconds"]:.2f} s.
- Development selection: {costs["development_selection_seconds"]:.2f} s.
- Target adaptation на 1000 эпизодах:
  {costs["target_adaptation_seconds"]:.2f} s с FLOP-инструментированием.

Нельзя переносить ускорение одной proposal-стадии на весь pipeline или
объявлять измеренное качество универсальной гарантией. Новые predictions
зафиксированы {summary["prediction_freeze_utc"]}; ответы открыты для
scoring {summary["answers_first_opened_for_new_scoring_utc"]}.
Access receipts: {summary["access_receipts"]}; violations:
{len(summary["access_violations"])}. Software tests не создавались и не
запускались. Эпизоды и target inputs сверены с историческими hashes.

## Артефакты

- [Протокол](../../experiments/omniglot_single_task_v1/PROTOCOL.md).
- [Сводка и пер-эпизодные данные]({artifact_relative}/README.md).
- [Accuracy и парные разности]({artifact_relative}/figures/comparison.png).
- [Группы и подбор]({artifact_relative}/figures/structure_and_development.png).
- [Предшествующая синтетическая проверка]({synthetic_report}).
"""
    report = Path("paper/theory/OMNIGLOT_SINGLE_TASK_2026-11-14.md")
    report.write_text(text)
    (output / "REPORT.md").write_text(
        text.replace(
            "../../artifacts/omniglot_single_task_2026-11-14/", "./"
        ).replace(
            f"({synthetic_report})", f"(../../paper/theory/{synthetic_report})"
        )
    )
    (output / "README.md").write_text(
        "# Direct one-task ratios on Aug-Omniglot\n\n"
        "Completed retrospective comparison on the same 1000 episodes as "
        "the 12 November G34/Dense/SVD run.\n\n"
        "- [Research report]"
        "(../../paper/theory/OMNIGLOT_SINGLE_TASK_2026-11-14.md).\n"
        "- `summary.json`, `comparison.csv`, `per_episode.npz`: "
        "paired results.\n"
        "- `extraction/`: exact candidates, descriptors, scans and receipts.\n"
        "- `source_selection/`, `development/`, `models/ratio/`: "
        "choices and states.\n"
        "- `figures/`: comparison and structure PNG/PDF.\n"
        "- `costs.json`, `access_audit.json`, freeze files "
        "and code snapshots.\n\n"
        "Original source images and dense states remain in "
        "`artifacts/local/omniglot_single_task_2026-11-14/`. "
        "No new training is needed to inspect or redraw these results.\n"
    )
    common = output / "common_source"
    common.mkdir(exist_ok=True)
    for name in (
        "registered.json",
        "metrics.json",
        "access_receipt.json",
        "executed_code.py",
    ):
        shutil.copy2(root / "source/dense" / name, common / name)
    shutil.copy2(root / "protocol.json", output / "protocol.json")
    for path in Path(__file__).parent.iterdir():
        if path.is_file():
            shutil.copy2(path, output / "code_snapshot" / path.name)
    reference_files = json.loads(
        (output / "unchanged_reference_code.json").read_text()
    )
    for path in reference_files:
        destination = output / "code_snapshot/reference" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    save_json(
        output / "compact_manifest.json",
        {
            "files": {
                str(path.relative_to(output)): digest(path)
                for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "compact_manifest.json"
            }
        },
    )
    print(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
