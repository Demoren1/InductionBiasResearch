"""Export the latent-search experiment without publishing banks or weights."""

import csv
import json
import os
from pathlib import Path
import shutil

import numpy as np

from .core import save_json
from .settings import DIRECTORY


LABELS = {
    "gradient_z": "Optimize z + refit",
    "random_z": "Random z + refit",
    "prior_sampling": "Prior sampling, 64 starts",
}
COLORS = {
    "gradient_z": "#147d92",
    "random_z": "#d18d28",
    "prior_sampling": "#7161a8",
}


def publish(run):
    run = Path(run)
    plan = json.loads((run / "plan.json").read_text())
    split = "test" if (run / "TEST_COMPLETE.json").exists() else "validation"
    if not (run / f"{split.upper()}_COMPLETE.json").exists():
        raise RuntimeError("Finish a complete evaluation split before publishing")
    records = [json.loads(p.read_text()) for p in (run / split).glob("*.json")]
    destination = DIRECTORY / "results" / run.name
    destination.mkdir(parents=True, exist_ok=True)
    save_json(destination / "plan.json", plan)
    save_json(destination / "records.json", records)
    source = destination / "source"
    source.mkdir(exist_ok=True)
    for path in (run / "source").glob("*.py"):
        shutil.copy2(path, source / path.name)
    groups = {
        method: [r for r in records if r["method"] == method]
        for method in LABELS
    }
    summary = {"split": split, "results": {}, "by_seed": {}}
    for method, values in groups.items():
        searches = sum(r["searches"] for r in values)
        success = sum(r["success"] for r in values)
        summary["results"][method] = {
            "success": success,
            "searches": searches,
            "rate": success / searches,
            "mean_selected_population_mse": float(np.average(
                [r["mean_selected_population_mse"] for r in values],
                weights=[r["searches"] for r in values],
            )),
            "recall_at_2": float(np.average(
                [r["mean_selected_recall_at_2"] for r in values],
                weights=[r["searches"] for r in values],
            )),
            "seconds": sum(r["seconds"] for r in values),
        }
        summary["by_seed"][method] = {
            str(seed): {
                "success": sum(r["success"] for r in values if r["model_seed"] == seed),
                "searches": sum(r["searches"] for r in values if r["model_seed"] == seed),
            }
            for seed in plan["config"]["model_seeds"]
        }
    gradients = [
        h for r in groups["gradient_z"] for h in r["history"]
        if "gradient_norm_mean" in h
    ]
    summary["gradient_diagnostics"] = {
        "mean_gradient_norm": float(np.mean([
            h["gradient_norm_mean"] for h in gradients
        ])),
        "mean_nonzero_gradient_fraction": float(np.mean([
            h["nonzero_gradient_fraction"] for h in gradients
        ])),
        "frozen_vae": all(not r["vae_has_gradients"] for r in records),
    }
    save_json(destination / "summary.json", summary)
    with (destination / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["method", *next(iter(summary["results"].values()))]
        )
        writer.writeheader()
        for method, values in summary["results"].items():
            writer.writerow({"method": method, **values})

    os.environ.setdefault("MPLCONFIGDIR", str(run / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for method, values in groups.items():
        histories = [r["history"] for r in values]
        x = [h["network_fits"] for h in histories[0]]
        color = COLORS[method]
        for key, axis in (
            ("selected_population_mse", axes[0, 0]),
            ("selected_recall_at_2", axes[1, 0]),
        ):
            mean = np.mean([[h[key] for h in history] for history in histories], axis=0)
            axis.plot(x, mean, "o-", color=color, label=LABELS[method], markersize=3)
        if method == "gradient_z":
            for key, label, style in (
                ("train_mse", "Train MSE", "--"),
                ("query_mse", "Outer query MSE", "-"),
            ):
                mean = np.mean([[h[key] for h in history] for history in histories], axis=0)
                axes[0, 1].plot(x, mean, style, label=label)
    axes[0, 0].set_title("Selected population MSE (diagnostic)")
    axes[0, 1].set_title("Loss during latent optimization")
    axes[1, 0].set_title("Selected mask Recall@2 (diagnostic)")
    axes[1, 0].set_ylim(0, 1.03)
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xlabel("Network fits per search")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    methods = list(LABELS)
    rates = [100 * summary["results"][m]["rate"] for m in methods]
    bars = axes[1, 1].bar(
        ["Optimize z\n+ refit", "Random z\n+ refit", "Prior\nsampling"],
        rates, color=[COLORS[m] for m in methods],
    )
    axes[1, 1].bar_label(bars, labels=[f"{r:.2f}%" for r in rates], padding=3)
    axes[1, 1].set_ylim(0, max(5, max(rates) * 1.25))
    axes[1, 1].set_ylabel("Successful searches (%)")
    axes[1, 1].set_title("Population MSE < 0.1; equal SGD sample budget")
    figure.suptitle(f"Frozen partwise VAE: latent search ({split})")
    figure.savefig(destination / "curves.png", dpi=160)
    plt.close(figure)

    table = []
    for method, values in summary["results"].items():
        table.append(
            f"| {LABELS[method]} | {values['success']}/{values['searches']} | "
            f"{100 * values['rate']:.2f}% | "
            f"{values['mean_selected_population_mse']:.5f} | "
            f"{values['recall_at_2']:.3f} | {values['seconds']:.1f} |"
        )
    report = f"""# Замороженный VAE: градиентный поиск латентного кода

Оценка: **{split}**. Использованы три сохранённых partwise VAE исходного
sine-эксперимента; encoder и decoder не обучаются. На новой задаче
обучаются код z и веса отдельной синусоидальной сети.

## Постановка и протокол

Данные, сеть и порог успеха совпадают с sine v1. Оптимизируется обычная
эмпирическая MSE предсказаний. Маска во всех прямых проходах бинарная,
содержит ровно две из 16 связей. Для производной используется straight-through
оценка с последовательной RelaxedTopK: это приближённый градиент дискретного
выбора. Градиент проходит через все 1024 шага обучения весов; аналитические
производные каждого SGD-обновления дифференцируются PyTorch.

Четыре кода z из нормального prior, восемь раундов обучения кандидатов,
семь обновлений z алгоритмом Adam (lr 0.1, температура 1.0). На каждом раунде
веса сети обучаются заново из фиксированной для данного рестарта инициализации.
Градиент считается по отдельной query-выборке из 4096 примеров. Независимая
выборка из 4096 примеров выбирает лучшую маску среди всех раундов. Затем
выбранная маска получает ещё 32 независимых старта весов. Выбирается лучший
кандидат из всех 64 обучений по эмпирической ошибке selection-выборки.

Контроль random_z имеет те же четыре начальные точки весов и данные,
но вместо градиентного обновления z сэмплирует новые коды; финальные
32 старта весов у обоих методов одинаковы. Контроль prior_sampling
использует 64 независимых кода и инициализации весов без дообучения z.

У всех методов 64 обучения по 1024 шага и batch 64: **4 194 304 обработанных
обучающих примера на поиск**. Обратный проход гиперградиента требует
дополнительных вычислений; равенство FLOPs или времени не заявляется.
Query/selection-оценки также считаются отдельно. Общий IID-поток обучающих
минибатчей повторяется между внешними раундами и методами; query и selection
независимы от него. Популяционная MSE и правильная поддержка используются
только при измерении результатов и диагностического Recall@2.

Параметры зафиксированы до оценки; validation используется для диагностики.
Итоговые функции 301/302 новые, четыре структурных test-пары уже изучались
в исходном v1. Это подтверждение на новых функциях и стартах, а не новый
слепой структурный holdout. Все три seed сохранены отдельно.

## Результаты

| Метод | Успехи | Доля | MSE выбранного | Recall@2 | Время, с |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

Средняя норма градиента z: {summary['gradient_diagnostics']['mean_gradient_norm']:.6g}.
Доля кодов с ненулевым градиентом:
{summary['gradient_diagnostics']['mean_nonzero_gradient_fraction']:.2%}.
VAE заморожен: {summary['gradient_diagnostics']['frozen_vae']}.

![Loss, Recall@2 и вероятность успеха](curves.png)

В summary.json находятся агрегаты и результаты по seed, в records.json —
все контексты и истории внешних раундов. plan.json фиксирует настройки,
SHA исходных checkpoint и снимка реализации. Полные маски, коды и ошибки
кандидатов сохранены локально в `{run}`. Исходный v1 не изменён.
"""
    (destination / "REPORT.md").write_text(report)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Report: {destination / 'REPORT.md'}", flush=True)
