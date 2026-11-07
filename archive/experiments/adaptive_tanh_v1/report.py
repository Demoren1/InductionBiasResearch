"""Aggregate independent banks and save loss, recall and mechanism figures."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.adaptive_tanh_v1.fit import selected_models


LABELS = {
    "fixed_raw": "Fixed I, raw SGDM",
    "fixed_normalized": "Fixed I, normalized SGDM",
    "adaptive": "Gradient splits",
    "random_matched": "Random splits, matched sizes",
    "dense_sgd": "Dense SGDM",
    "dense_adam": "Dense Adam",
}
COLORS = {
    "fixed_raw": "#98a6b3",
    "fixed_normalized": "#2667a0",
    "adaptive": "#008c78",
    "random_matched": "#d28c28",
    "dense_sgd": "#a64b6d",
    "dense_adam": "#6554a4",
}


def write_csv(path, rows):
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="experiments/adaptive_tanh_v1/results/pilot_v1"
    )
    args = parser.parse_args()
    output = Path(args.output)
    plan = json.loads((output / "protocol.json").read_text())
    complete = json.loads((output / "complete.json").read_text())
    if not complete["complete"]:
        raise RuntimeError("Pilot has not completed")
    frozen_path = output / "frozen_selection.json"
    frozen = json.loads(frozen_path.read_text())
    frozen_hash = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    if frozen_hash != complete["global_selection_sha256"]:
        raise RuntimeError("Frozen global selection changed")
    if (
        hashlib.sha256((output / "protocol.json").read_bytes()).hexdigest()
        != frozen["protocol_sha256"]
    ):
        raise RuntimeError("Frozen protocol changed")
    for bank in range(plan["evaluation_banks"]):
        folder = output / f"evaluation_bank_{bank}"
        manifest = json.loads(
            (folder / "frozen_before_target.json").read_text()
        )
        if manifest["global_selection_sha256"] != frozen_hash:
            raise RuntimeError("Evaluation used another frozen selection")
        actual = hashlib.sha256((folder / "bank.npz").read_bytes()).hexdigest()
        if manifest["source_bank_sha256"] != actual:
            raise RuntimeError("Source indexator changed after freezing")
    with (output / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in [
            "bank",
            "task",
            "task_row",
            "groups",
            "selected_start",
            "selected_step",
        ]:
            row[field] = int(row[field])
        for field in [
            "residual_ratio",
            "query_nmse",
            "train_nmse",
            "test_mse",
            "test_nmse",
            "test_positive_recall",
        ]:
            row[field] = float(row[field])
        row["residual_ratio"] = round(row["residual_ratio"], 6)
    buckets = defaultdict(list)
    for row in rows:
        key = (
            row["bank"],
            row["residual_ratio"],
            row["method"],
            row["selection"],
        )
        buckets[key].append(row)
    bank_rows = []
    for (bank, ratio, method, selection), values in sorted(buckets.items()):
        bank_rows.append(
            {
                "bank": bank,
                "residual_ratio": ratio,
                "method": method,
                "selection": selection,
                "tasks": len(values),
                "test_nmse": float(np.mean([v["test_nmse"] for v in values])),
                "test_mse": float(np.mean([v["test_mse"] for v in values])),
                "groups": float(np.mean([v["groups"] for v in values])),
                "test_positive_recall": float(
                    np.mean([v["test_positive_recall"] for v in values])
                ),
                "success_fraction_nmse_below_0_001": float(
                    np.mean([v["test_nmse"] < 0.001 for v in values])
                ),
            }
        )
    write_csv(output / "bank_summary.csv", bank_rows)
    aggregated = defaultdict(list)
    for row in bank_rows:
        aggregated[
            (row["residual_ratio"], row["method"], row["selection"])
        ].append(row)
    summary = []
    for (ratio, method, selection), values in sorted(aggregated.items()):
        errors = np.array([v["test_nmse"] for v in values])
        summary.append(
            {
                "residual_ratio": ratio,
                "method": method,
                "selection": selection,
                "banks": len(values),
                "mean_nmse": float(errors.mean()),
                "bank_min_nmse": float(errors.min()),
                "bank_max_nmse": float(errors.max()),
                "bank_std_nmse": float(errors.std(ddof=1)),
                "mean_groups": float(np.mean([v["groups"] for v in values])),
                "mean_positive_recall": float(
                    np.mean([v["test_positive_recall"] for v in values])
                ),
                "success_fraction": float(
                    np.mean(
                        [
                            v["success_fraction_nmse_below_0_001"]
                            for v in values
                        ]
                    )
                ),
            }
        )
    write_csv(output / "summary.csv", summary)
    lookup = {
        (r["residual_ratio"], r["method"], r["selection"]): r for r in summary
    }
    pairs = []
    for ratio in plan["residual_ratios"]:
        for selection in ["selected", "k_48", "k_64", "k_96"]:
            for bank in range(plan["evaluation_banks"]):
                adaptive = buckets[(bank, ratio, "adaptive", selection)]
                random = buckets[(bank, ratio, "random_matched", selection)]
                random_by_task = {r["task"]: r for r in random}
                differences = [
                    random_by_task[r["task"]]["test_nmse"] - r["test_nmse"]
                    for r in adaptive
                ]
                pairs.append(
                    {
                        "bank": bank,
                        "residual_ratio": ratio,
                        "selection": selection,
                        "random_minus_adaptive_nmse": float(
                            np.mean(differences)
                        ),
                        "adaptive_task_win_fraction": float(
                            np.mean(np.array(differences) > 0)
                        ),
                        "adaptive_nmse": float(
                            np.mean([r["test_nmse"] for r in adaptive])
                        ),
                        "random_nmse": float(
                            np.mean([r["test_nmse"] for r in random])
                        ),
                    }
                )
    write_csv(output / "paired_splits.csv", pairs)

    assets = Path("mds/assets/2026-11-02/adaptive_tanh")
    assets.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    ratios = plan["residual_ratios"]
    x = np.arange(len(ratios))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for axis, selection, title in zip(
        axes,
        ["selected", "k_96"],
        [
            "Validation-selected checkpoint and K",
            "Gradient vs random splits: K = 96",
        ],
    ):
        for method in plan["methods"]:
            key = (
                selection
                if method in {"adaptive", "random_matched"}
                else "selected"
            )
            values = [lookup[(r, method, key)] for r in ratios]
            errors = np.array([v["mean_nmse"] for v in values])
            axis.plot(
                x,
                np.maximum(errors, 1e-12),
                "o-",
                color=COLORS[method],
                label=LABELS[method],
            )
            axis.fill_between(
                x,
                [max(v["bank_min_nmse"], 1e-12) for v in values],
                [max(v["bank_max_nmse"], 1e-12) for v in values],
                color=COLORS[method],
                alpha=0.09,
            )
        axis.set_yscale("log")
        axis.set_xticks(x, [f"{100 * r:g}%" for r in ratios])
        axis.set_xlabel("Residual weight norm / base weight norm")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Held-out NMSE (lower is better)")
    axes[1].legend(fontsize=8)
    fig.suptitle("3 independent source banks; shading = min/max bank mean")
    fig.tight_layout()
    fig.savefig(assets / "comparison.png", dpi=170)
    plt.close(fig)

    histories = defaultdict(list)
    for bank in range(plan["evaluation_banks"]):
        folder = output / f"evaluation_bank_{bank}"
        for method in plan["methods"]:
            result = np.load(folder / f"{method}.npz")
            _, models = selected_models(result)
            history = json.loads(
                (folder / f"{method}_history.json").read_text()
            )
            for ratio_id, ratio in enumerate(ratios):
                ids = models[
                    ratio_id * plan["evaluation_tasks"] : (ratio_id + 1)
                    * plan["evaluation_tasks"]
                ]
                curves = {"step": [h["step"] for h in history]}
                for metric in [
                    "train_nmse",
                    "query_nmse",
                    "train_positive_recall",
                    "query_positive_recall",
                    "conflict_energy",
                    "conflict_fraction",
                ]:
                    curves[metric] = [
                        float(np.mean(np.asarray(h[metric])[ids]))
                        for h in history
                    ]
                histories[(ratio, method)].append(curves)
    for kind, metrics, filename in [
        ("NMSE", ["train_nmse", "query_nmse"], "loss_curves.png"),
        (
            "Positive-sign recall",
            ["train_positive_recall", "query_positive_recall"],
            "recall_curves.png",
        ),
    ]:
        fig, axes = plt.subplots(
            2, len(ratios), figsize=(17, 7), sharex=True, sharey=True
        )
        for column, ratio in enumerate(ratios):
            for method in plan["methods"]:
                values = histories[(ratio, method)]
                for row, metric in enumerate(metrics):
                    curve = np.mean([v[metric] for v in values], axis=0)
                    if kind == "NMSE":
                        curve = np.maximum(curve, 1e-12)
                        axes[row, column].set_yscale("log")
                    axes[row, column].plot(
                        values[0]["step"],
                        curve,
                        color=COLORS[method],
                        label=LABELS[method],
                    )
            axes[0, column].set_title(f"Residual {100 * ratio:g}%")
            for row in range(2):
                axes[row, column].grid(alpha=0.2)
                for step, _ in plan["split_schedule"]:
                    axes[row, column].axvline(
                        step, color="black", alpha=0.12, linestyle=":"
                    )
            axes[1, column].set_xlabel("Gradient updates")
        axes[0, 0].set_ylabel(f"Train {kind}")
        axes[1, 0].set_ylabel(f"Validation {kind}")
        axes[0, -1].legend(fontsize=7)
        fig.suptitle(
            "Trajectories of validation-selected starts; mean across tasks then banks"
        )
        fig.tight_layout()
        fig.savefig(assets / filename, dpi=170)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for method in ["adaptive", "random_matched"]:
        axes[0].plot(
            x,
            [lookup[(r, method, "selected")]["mean_groups"] for r in ratios],
            "o-",
            color=COLORS[method],
            label=LABELS[method],
        )
    axes[0].axhline(
        plan["groups"],
        color=COLORS["fixed_normalized"],
        linestyle="--",
        label="Fixed I",
    )
    axes[0].set_ylabel("Validation-selected K")
    axes[0].legend(fontsize=8)
    for bank in range(plan["evaluation_banks"]):
        differences = [
            next(
                p["random_minus_adaptive_nmse"]
                for p in pairs
                if p["bank"] == bank
                and p["residual_ratio"] == r
                and p["selection"] == "k_96"
            )
            for r in ratios
        ]
        axes[1].plot(x, differences, "o-", label=f"Bank {bank + 1}")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Random − adaptive NMSE, K=96")
    axes[1].legend(fontsize=8)
    for bank in range(plan["evaluation_banks"]):
        conflicts = []
        for ratio in ratios:
            curves = histories[(ratio, "adaptive")][bank]
            point = curves["step"].index(plan["split_schedule"][0][0])
            conflicts.append(curves["conflict_energy"][point])
        axes[2].plot(
            x, np.maximum(conflicts, 1e-20), "o-", label=f"Bank {bank + 1}"
        )
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Gradient conflict energy before first split")
    for axis in axes:
        axis.set_xticks(x, [f"{100 * r:g}%" for r in ratios])
        axis.set_xlabel("Residual ratio")
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(assets / "mechanism.png", dpi=170)
    plt.close(fig)

    source_rows = []
    folders = sorted(output.glob("*_bank_*"))
    fig, axes = plt.subplots(1, len(folders), figsize=(19, 4), sharey=True)
    for axis, folder in zip(axes, folders):
        diagnostic = json.loads(
            (folder / "source_diagnostic.json").read_text()
        )
        source_rows.append({"bank": folder.name, **diagnostic})
        source = np.load(folder / "bank.npz")
        history = json.loads((folder / "source_history.json").read_text())
        for metric, label in [
            ("train_nmse_first_1024", "Train subset"),
            ("query_nmse", "Validation"),
        ]:
            curve = [
                np.mean(np.asarray(h[metric])[source["selected"]])
                for h in history
            ]
            axis.plot(
                [h["step"] for h in history],
                np.maximum(curve, 1e-14),
                label=label,
            )
        axis.set_yscale("log")
        axis.set_title(folder.name.replace("_", " "))
        axis.set_xlabel("Gradient updates")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Source NMSE, selected starts")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(assets / "source_loss.png", dpi=160)
    plt.close(fig)
    (output / "source_summary.json").write_text(
        json.dumps(source_rows, indent=2)
    )

    source_seconds = sum(r["seconds"] for r in source_rows)
    calibration_seconds = sum(
        r["seconds"] for r in complete["calibration_timings"].values()
    )
    target_seconds = sum(
        r["seconds"] for r in complete["evaluation_timings"].values()
    )
    interpretation_path = output / "interpretation.json"
    interpretation = (
        json.loads(interpretation_path.read_text())
        if interpretation_path.exists()
        else {}
    )
    lines = [
        "# Адаптивные индексаторы: GPU-пилот на tanh",
        "",
        *interpretation.get(
            "intro", ["Численные результаты этого запуска.", ""]
        ),
        "## Протокол",
        "",
        "MLP 32 → 16 tanh → сумма / 4; фиксированные нулевые bias и readout. "
        "В каждом банке новый скрытый индексатор из 32 групп на 512 весах. "
        "Шесть исходных функций × четыре старта обучены плотным Adam; структура "
        "извлечена из весов через k-means и согласование перестановок, затем "
        "голосование лучших исходных решений. Истинный I служит только генератору "
        "и диагностике восстановления.",
        "",
        "Два отдельных банка использованы для выбора learning rates. После "
        "заморозки получены три новых банка, по шесть новых функций и четыре "
        "силы отклонения весов. Это 18 задач на условие, но лишь три независимых "
        "повтора банка. Отклонение ортогонально исходным равенствам; его норма "
        "равна 0%, 5%, 15% или 30% нормы исходных весов. Это контролируемый "
        "сдвиг от точной source-структуры, а не наблюдательный шум.",
        "",
        "Каждый метод получает 128 support и ещё 1024 validation labels, четыре "
        "старта и 4096 полных градиентных шагов. 8192 отложенных примера "
        "используются после выбора модели. Итоговая NMSE нормирована на "
        "дисперсию отложенных меток; train/validation NMSE — на support variance. "
        "Источники и калибровка оплачены отдельно. Градиентные шаги совпадают, "
        "время и FLOPs не уравнены.",
        "",
        "На шагах 1024/2048/3072 число групп увеличивается до 48/64/96. "
        "Градиентный split максимизирует эмпирический квадрат контраста "
        "градиентов. Случайный split повторяет его выбор родительской группы "
        "и размеры детей, но случайно распределяет позиции внутри своего "
        "родителя. Поэтому совпадают K и размеры всех групп у каждой пары "
        "стартов. Коды и старый момент копируются, функция при split сохраняется. "
        "Независимое подтверждение направления и теоретическая граница шума "
        "в этом пилоте не реализованы. После split выполняется обычный "
        "SGDM по кодам, а не теоретический шаг с известной константой "
        "гладкости; гарантия безопасного шага из теории сюда не переносится.",
        "",
        "## Ошибка после выбора по validation",
        "",
        "Средние NMSE по банкам; меньше — лучше. Для split разрешён выбор "
        "checkpoint из любой фазы, включая исходные 32 группы.",
        "",
        "| Метод | 0% | 5% | 15% | 30% |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in plan["methods"]:
        values = [lookup[(r, method, "selected")]["mean_nmse"] for r in ratios]
        lines.append(
            "| "
            + LABELS[method]
            + " | "
            + " | ".join(
                f"{v:.2e}" if v < 0.001 else f"{v:.4f}" for v in values
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Сравнение при одинаковом K=96",
            "",
            "Checkpoint и старт выбираются только внутри последней фазы. "
            "В таблице также показана разность random − adaptive по каждому банку; "
            "положительное значение означает преимущество градиентного split.",
            "",
            "| Отклонение | Adaptive | Random | Разности по трём банкам |",
            "|---|---:|---:|---|",
        ]
    )
    for ratio in ratios:
        a = lookup[(ratio, "adaptive", "k_96")]["mean_nmse"]
        r = lookup[(ratio, "random_matched", "k_96")]["mean_nmse"]
        delta = [
            p["random_minus_adaptive_nmse"]
            for p in pairs
            if p["residual_ratio"] == ratio and p["selection"] == "k_96"
        ]
        lines.append(
            f"| {100 * ratio:g}% | {a:.4g} | {r:.4g} | "
            + ", ".join(f"{v:+.4g}" for v in delta)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Восстановление source-структуры",
            "",
            "| Банк | Верных позиций | Лучшая source NMSE |",
            "|---|---:|---:|",
        ]
    )
    for row in source_rows:
        lines.append(
            f"| {row['bank']} | {row['correct_assignments']}/{row['positions']} | {row['source_best_mean_nmse']:.2e} |"
        )
    lines.extend(
        [
            "",
            "## Стоимость и артефакты",
            "",
            f"Tesla V100, CUDA float32. Завершившийся запуск после исправления: {complete['seconds'] / 60:.2f} мин; "
            f"source fit/extraction: {source_seconds / 60:.2f} мин; "
            f"калибровка: {calibration_seconds / 60:.2f} мин; "
            f"целевые fit: {target_seconds / 60:.2f} мин. "
            "Подробные времена split и числа моделей сохранены отдельно.",
            "",
            *interpretation.get("run_notes", []),
            "Выбранные rates: `" + json.dumps(frozen["selected_rates"]) + "`.",
            "",
            "[Код и протокол](../experiments/adaptive_tanh_v1/README.md), "
            "[результаты](../experiments/adaptive_tanh_v1/results/pilot_v1/summary.csv), "
            "[парные разности](../experiments/adaptive_tanh_v1/results/pilot_v1/paired_splits.csv). "
            "Сохранены модели всех стартов и фаз, решения split, кривые и хеши. "
            "Тесты не добавлялись и не запускались.",
            "",
            "![Отложенная ошибка](assets/2026-11-02/adaptive_tanh/comparison.png)",
            "",
            "![Train и validation loss](assets/2026-11-02/adaptive_tanh/loss_curves.png)",
            "",
            "![Число групп и конфликт градиентов](assets/2026-11-02/adaptive_tanh/mechanism.png)",
            "",
            "![Вспомогательный positive-sign recall](assets/2026-11-02/adaptive_tanh/recall_curves.png)",
            "",
            "![Обучение исходных сетей](assets/2026-11-02/adaptive_tanh/source_loss.png)",
            "",
            "Recall здесь — доля правильно предсказанных положительных знаков "
            "среди положительных меток; это вспомогательная диагностика регрессии.",
            "",
            *interpretation.get("discussion", []),
            "## Ограничения",
            "",
            "Три банка недостаточны для сильного статистического обобщения. "
            "Все функции принадлежат узкому синтетическому семейству с известной "
            "архитектурой и бюджетом исходных групп. Учёт исходной стоимости важен "
            "для практического переноса. Эксперимент не проверяет необходимость "
            "авторегрессионного генератора и не доказывает превосходство над любым SGD.",
            "",
        ]
    )
    Path("mds/ADAPTIVE_TANH_PILOT_2026-11-02.md").write_text("\n".join(lines))
    print(
        json.dumps(
            {
                "report": "mds/ADAPTIVE_TANH_PILOT_2026-11-02.md",
                "assets": str(assets),
                "rows": len(rows),
                "summary": summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
