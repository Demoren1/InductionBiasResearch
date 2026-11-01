"""Bank-level analysis of five independently rebuilt source banks."""

import argparse
import csv
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from experiments.dynamic_signal_v1.indexator_audit import align
from experiments.dynamic_signal_v1.repeat_banks import BASE, SEEDS
from experiments.dynamic_signal_v1.report import LABELS, best_rows, path_for

OUTPUT = BASE / "results/bank_repeats"
FIGURES = BASE / "figures/bank_repeats"


def export_bank(root, destination):
    """Retain source extraction evidence and complete individual target metrics."""
    for source in sorted(root.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(root)
        if (
            relative.parts[0] in ["source", "target"]
            and source.name != "teacher_audit.npz"
        ):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".npz":
            with np.load(source) as raw:
                kept = {
                    key: raw[key]
                    for key in raw.files
                    if not key.startswith("parameter_")
                }
                if relative.parts[0] in ["banks", "banks_large_tuned"]:
                    if source.name in ["round_4096.npz", "round_256.npz"]:
                        kept.update(
                            {
                                key: raw[key]
                                for key in raw.files
                                if key.startswith("parameter_")
                            }
                        )
                elif "target" in str(relative) and "parameter_0" in raw.files:
                    ids = best_rows(raw)
                    kept.update(
                        {
                            "selected_" + key: raw[key][ids]
                            for key in raw.files
                            if key.startswith("parameter_")
                        }
                    )
                if "indices" in kept:
                    kept["indices"] = kept["indices"].astype(np.uint8)
                np.savez_compressed(target, **kept)
        elif source.suffix in [".json", ".gz"]:
            shutil.copyfile(source, target)
        elif source.suffix == ".pt":
            target.with_suffix(".sha256").write_text(
                hashlib.sha256(source.read_bytes()).hexdigest() + "\n"
            )
    # pathlib does not recurse through the source-bank symlinks.
    chosen = json.loads((root / "bank_selection/selection.json").read_text())[
        "chosen_id"
    ]
    for noise in range(3):
        target = destination / "banks_relations" / f"noise_{noise}"
        target.mkdir(parents=True, exist_ok=True)
        for name in ["bank.npz", "extracted.npz", "complete.json"]:
            shutil.copyfile(
                root / f"candidate_banks/noise_{chosen}" / name, target / name
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-export", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    records, banks, recovery, accuracy = [], [], [], {}
    roots = []
    for replica, seed in enumerate(SEEDS, 1):
        raw_root = BASE / "runs" / f"bank_repeat_{replica}"
        root = (
            raw_root
            if not args.from_export and (raw_root / "complete.json").exists()
            else OUTPUT / f"bank_{replica}"
        )
        if not (root / "complete.json").exists():
            raise RuntimeError(f"Replica {replica} has not completed")
        roots.append(root)
        selection = json.loads((root / "bank_selection/selection.json").read_text())
        chosen = selection["chosen_id"]
        bank_path = root / f"candidate_banks/noise_{chosen}"
        bank = np.load(bank_path / "bank.npz")
        info = json.loads((bank_path / "complete.json").read_text())
        banks.append(
            {
                "bank": replica,
                "seed": seed,
                "source_noise": [0.15, 0.25, 0.4][chosen],
                "rows": len(bank["indices"]),
                "unique": len(np.unique(bank["indices"], axis=0)),
                "training_unique": len(
                    np.unique(bank["indices"][bank["role"] == 0], axis=0)
                ),
                "training_rows": int((bank["role"] == 0).sum()),
                "source_query_accuracy": info["retained_query_accuracy"],
            }
        )
        permutation = np.load(root / "target/teacher_audit.npz")["permutation"]
        oracle = np.zeros((128, 32), int)
        for group in range(16):
            coords = permutation[group * 8 : (group + 1) * 8]
            oracle[coords, 2 * group] = np.arange(1, 9)
            oracle[coords, 2 * group + 1] = np.arange(9, 17)
        for method in LABELS:
            path = path_for(root, method, 1, 2048)
            with np.load(path) as arrays:
                ids = best_rows(arrays)
                accuracy[replica, method] = arrays["test_accuracy"][ids]
                records.append(
                    {
                        "bank": replica,
                        "seed": seed,
                        "method": method,
                        "accuracy": float(arrays["test_accuracy"][ids].mean()),
                        "test_bce": float(arrays["test_loss"][ids].mean()),
                        "recall_positive": float(
                            arrays["test_recall_positive"][ids].mean()
                        ),
                        "recall_negative": float(
                            arrays["test_recall_negative"][ids].mean()
                        ),
                    }
                )
                if method in ["ar", "frequency", "consensus", "random_topology"]:
                    for task, row in enumerate(ids):
                        aligned = align(arrays["indices"][row], oracle)
                        support, truth = aligned > 0, oracle > 0
                        intersection = (support & truth).sum()
                        recovery.append(
                            {
                                "bank": replica,
                                "method": method,
                                "task": task,
                                "support_precision": intersection
                                / max(1, support.sum()),
                                "support_recall": intersection / truth.sum(),
                                "active_index_accuracy": (
                                    (aligned == oracle) & truth
                                ).sum()
                                / truth.sum(),
                            }
                        )
        if root == raw_root:
            frozen = json.loads((root / "frozen_manifest.json").read_text())
            mismatches = []
            for name, expected in frozen["hashes"].items():
                original = Path(name) if name.startswith(str(BASE)) else root / name
                # benchmark.py rewrites its aggregate manifest when target
                # observations are added. Preserve the pre-target snapshot.
                if name == "data_manifest.json":
                    original = root / "source_data_manifest.json"
                if hashlib.sha256(original.read_bytes()).hexdigest() != expected:
                    mismatches.append(name)
            if mismatches:
                raise RuntimeError(
                    f"Frozen inputs changed in bank {replica}: {mismatches}"
                )
            (root / "freeze_audit.json").write_text(
                json.dumps(
                    {
                        "checked": len(frozen["hashes"]),
                        "mismatches": mismatches,
                        "aggregate_manifest_snapshot": "source_data_manifest.json",
                    },
                    indent=2,
                )
                + "\n"
            )
            export_bank(root, OUTPUT / f"bank_{replica}")
    for name, data in [
        ("comparison", records),
        ("banks", banks),
        ("indexator_recovery", recovery),
    ]:
        with (OUTPUT / f"{name}.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)

    # Independent sampling units are banks. The 16 tasks share a fitted bank.
    differences = (
        np.stack(
            [
                accuracy[bank, "ar"] - accuracy[bank, "mask_unshared"]
                for bank in range(1, 6)
            ]
        )
        * 100
    )
    rng = np.random.default_rng(7111481)
    bank_ids = rng.integers(5, size=(20000, 5))
    task_ids = rng.integers(16, size=(20000, 5, 16))
    resampled = differences[bank_ids[:, :, None], task_ids].mean((1, 2))
    summary = {
        "condition": {
            "sigma": 0.25,
            "support": 2048,
            "query": 1024,
            "test_per_task": 16384,
        },
        "independent_banks": 5,
        "tasks_per_bank": 16,
        "difference_pp": float(differences.mean()),
        "difference_by_bank_pp": differences.mean(1).tolist(),
        "between_bank_sd_pp": float(differences.mean(1).std(ddof=1)),
        "hierarchical_bootstrap_95_pp": np.quantile(resampled, [0.025, 0.975]).tolist(),
        "caution": "Only five independent banks; interval is descriptive and does not establish robustness over arbitrary structure families.",
        "positive_task_differences": int((differences > 0).sum()),
        "methods": {},
    }
    for method in LABELS:
        values = np.array([100 * accuracy[bank, method].mean() for bank in range(1, 6)])
        summary["methods"][method] = {
            "mean_accuracy_percent": float(values.mean()),
            "between_bank_sd_pp": float(values.std(ddof=1)),
            "per_bank_accuracy_percent": values.tolist(),
        }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    displayed = [
        "ar",
        "consensus",
        "frequency",
        "mask_unshared",
        "transfer",
        "adam",
        "sgd",
        "wsr",
        "adam_long",
        "sgd_long",
        "oracle",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    means = [summary["methods"][name]["mean_accuracy_percent"] for name in displayed]
    deviations = [summary["methods"][name]["between_bank_sd_pp"] for name in displayed]
    y = np.arange(len(displayed))
    axes[0].barh(y, means, xerr=deviations, color="#8ec7c4", capsize=3)
    for bank in range(1, 6):
        axes[0].scatter(
            [100 * accuracy[bank, name].mean() for name in displayed],
            y,
            s=13,
            color="#243857",
            alpha=0.7,
        )
    axes[0].set_yticks(y, [LABELS[name] for name in displayed])
    axes[0].invert_yaxis()
    axes[0].set_xlim(max(0, min(means) - max(deviations) - 3), 100)
    axes[0].set_xlabel("Точность, %; точки — банки, погрешности — SD банков")
    axes[0].grid(axis="x", alpha=0.2)
    axes[1].axhline(0, color="black", lw=1)
    for bank in range(5):
        axes[1].scatter(np.full(16, bank + 1), differences[bank], s=18, color="#aeb7c5")
    axes[1].plot(
        range(1, 6),
        differences.mean(1),
        "o-",
        color="#008b80",
        label="Среднее по 16 задачам",
    )
    axes[1].set_xticks(range(1, 6))
    axes[1].set_xlabel("Независимый исходный банк")
    axes[1].set_ylabel("AR − маска с независимыми весами, п.п.")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.suptitle("Пять новых банков: шум 0,25; 2048 обучающих примеров + 1024 query")
    fig.tight_layout()
    fig.savefig(FIGURES / "comparison.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(15, 9), sharex=True, sharey=True)
    palette = ListedColormap(["white", *list(plt.get_cmap("tab20").colors[:16])])
    for bank, root in enumerate(roots, 1):
        permutation = np.load(root / "target/teacher_audit.npz")["permutation"]
        oracle = np.zeros((128, 32), int)
        for group in range(16):
            coords = permutation[group * 8 : (group + 1) * 8]
            oracle[coords, 2 * group] = np.arange(1, 9)
            oracle[coords, 2 * group + 1] = np.arange(9, 17)
        with np.load(path_for(root, "ar", 1, 2048)) as arrays:
            found = align(arrays["indices"][best_rows(arrays)[0]], oracle)
        for row, value in enumerate([oracle, found]):
            im = axes[row, bank - 1].imshow(
                value[permutation],
                vmin=0,
                vmax=16,
                cmap=palette,
                interpolation="nearest",
                aspect="auto",
            )
        axes[0, bank - 1].set_title(f"Банк {bank}")
        axes[1, bank - 1].set_xlabel("32 скрытых нейрона")
    axes[0, 0].set_ylabel("Аналитический I: 128 входов")
    axes[1, 0].set_ylabel("AR, первая целевая задача")
    fig.suptitle(
        "Проверка после обучения: согласованы эквивалентные перестановки групп и названий кодов"
    )
    fig.tight_layout()
    fig.subplots_adjust(right=0.92)
    color_axis = fig.add_axes((0.94, 0.15, 0.012, 0.7))
    colorbar = fig.colorbar(im, cax=color_axis, ticks=[0, 4, 8, 12, 16])
    colorbar.set_label("Номер общего веса; 0 — связь выключена")
    fig.savefig(FIGURES / "structures.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for bank, root in enumerate(roots, 1):
        training = json.loads(
            (root / "priors/relations/noise_1/training.json").read_text()
        )
        ax.plot(
            [point["step"] for point in training["history"]],
            [point["validation_nll"] for point in training["history"]],
            label=f"Банк {bank}",
        )
    ax.set_xlabel("Шаги обучения MADE")
    ax.set_ylabel("Средняя NLL индекса на исходной validation")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "prior_loss.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    for field, ylabel, filename in [
        ("loss", "BCE", "loss.png"),
        ("recall_positive", "Recall положительного класса", "recall.png"),
    ]:
        fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True, sharey=True)
        for bank, root in enumerate(roots, 1):
            for method in ["ar", "mask_unshared", "adam", "sgd", "transfer"]:
                path = path_for(root, method, 1, 2048)
                ids = best_rows(np.load(path))
                history_path = path.with_name(path.stem + "_history.json.gz")
                with gzip.open(history_path, "rt") as stream:
                    history = json.load(stream)
                for row, split in enumerate(["train", "query"]):
                    axes[row, bank - 1].plot(
                        [point["step"] for point in history],
                        [
                            np.asarray(point[split][field])[ids].mean()
                            for point in history
                        ],
                        label=LABELS[method],
                    )
                    axes[row, bank - 1].grid(alpha=0.2)
            axes[0, bank - 1].set_title(f"Банк {bank}")
            axes[1, bank - 1].set_xlabel("Шаги / батчи")
        axes[0, 0].set_ylabel("Train: " + ylabel)
        axes[1, 0].set_ylabel("Query: " + ylabel)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=5)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(FIGURES / filename, dpi=160, bbox_inches="tight")
        plt.close(fig)

    lines = [
        "# Перенос структуры: пять независимых банков",
        "",
        (
            "Повторили основное сравнение при шуме 0,25: 2048 обучающих примеров, "
            "1024 query для выбора старта и остановки, 16 384 тестовых примера на задачу. "
            "Каждый повтор использует новую скрытую перестановку входов, новый исходный "
            "банк и 16 новых целевых задач. Всего пять банков и 80 целевых задач."
        ),
        "",
        (
            "В каждом повторе заново обучили исходные MLP, выполнили разреживание, "
            "извлекли индексаторы из весов и обучили MADE. Один банк из трёх уровней "
            "шума выбирали по отложенным исходным задачам. Настройки целевых методов "
            "перенесли из предыдущей независимой калибровки без нового подбора. "
            "Генератор и настройки заморозили до создания целевых данных. "
            "У каждого метода восемь стартов и 4096 шагов; у длинных Adam/SGD — 16 384 шага. "
            "Сбор банка требует дополнительных исходных вычислений: одинаков здесь "
            "целевой бюджет, а не полная стоимость всех этапов."
        ),
        "",
        (
            "Это повтор основного заранее выбранного условия; остальные 11 сочетаний "
            "шума и размера выборки остаются в [первом отчёте](DYNAMIC_SIGNAL_STRUCTURES_2026-11-01.md). "
            "Там же приведены постановка задачи и аналитический индексатор."
        ),
        "",
        "## Результат по каждому банку",
        "",
        "| Банк | AR, % | Маска без общих весов, % | Adam, % | SGD, % | AR − маска, п.п. |",
        "| :-- | --: | --: | --: | --: | --: |",
    ]
    for bank in range(1, 6):
        values = [
            100 * accuracy[bank, method].mean()
            for method in ["ar", "mask_unshared", "adam", "sgd"]
        ]
        lines.append(
            f"| {bank} | "
            + " | ".join(
                f"{value:.2f}" for value in [*values, differences[bank - 1].mean()]
            )
            + " |"
        )
    low, high = summary["hierarchical_bootstrap_95_pp"]
    lines.extend(
        [
            "",
            (
                f"Средний выигрыш AR над выбранным ранее основным конкурентом: "
                f"**{summary['difference_pp']:.2f} п.п.**; между банками SD = "
                f"{summary['between_bank_sd_pp']:.2f} п.п. Иерархический bootstrap даёт "
                f"описательный 95% интервал [{low:.2f}; {high:.2f}] п.п.: сначала "
                "пересэмплируются банки, затем задачи внутри них. Независимых банков "
                "всего пять; 80 задач нельзя считать 80 независимыми проверками обучения банка."
            ),
            "",
            "## Все методы",
            "",
            "| Метод | Средняя точность, % | SD между банками, п.п. |",
            "| :-- | --: | --: |",
        ]
    )
    for method in sorted(
        LABELS, key=lambda name: -summary["methods"][name]["mean_accuracy_percent"]
    ):
        result = summary["methods"][method]
        lines.append(
            f"| {LABELS[method]} | {result['mean_accuracy_percent']:.2f} | {result['between_bank_sd_pp']:.2f} |"
        )
    lines.extend(
        [
            "",
            (
                "Контроль «Маска без общих весов» получает опорную маску выбранного банка "
                "и обучает каждую связь независимо. «Один индексатор» использует ту же "
                "структуру с общими коэффициентами. Эта пара выделяет пользу совместного "
                "использования весов. AR и частоты могут выдавать другие индексаторы того "
                "же банка, поэтому их сравнение с маской проверяет весь способ переноса. "
                "WSR — субградиентный вариант регуляризации первого слоя, с ранее "
                "подобранными штрафами; полный проксимальный метод статьи не воспроизводился."
            ),
            "",
            "![Сравнение по банкам](../experiments/dynamic_signal_v1/figures/bank_repeats/comparison.png)",
            "",
            "## Что удалось восстановить",
            "",
            "| Банк | Строк в банке | Уникальных I | Уникальных I в обучении | Правильные связи AR, % | Правильные активные индексы AR, % |",
            "| :-- | --: | --: | --: | --: | --: |",
        ]
    )
    for bank in banks:
        found = [
            row
            for row in recovery
            if row["bank"] == bank["bank"] and row["method"] == "ar"
        ]
        lines.append(
            f"| {bank['bank']} | {bank['rows']} | {bank['unique']} | {bank['training_unique']} | "
            f"{100 * np.mean([row['support_recall'] for row in found]):.2f} | "
            f"{100 * np.mean([row['active_index_accuracy'] for row in found]):.2f} |"
        )
    lines.extend(
        [
            "",
            (
                "Сравнение с аналитическим индексатором выполняется после обучения, "
                "с согласованием эквивалентных перестановок групп и названий коэффициентов. "
                "Истинная перестановка не используется для извлечения или отбора банка. "
                "Извлекатель заранее знает бюджеты: 16 групп, восемь ролей и два фильтра "
                "на группу. Проверяется восстановление состава групп и назначений коэффициентов "
                "при этих предположениях."
            ),
            "",
            "![Аналитические и найденные индексаторы](../experiments/dynamic_signal_v1/figures/bank_repeats/structures.png)",
            "",
            "## Как шло обучение",
            "",
            (
                "Кривые усреднены по 16 задачам каждого банка. Для каждой задачи показан "
                "старт, выбранный по query, без выбора по test. Train loss и recall "
                "измерены на текущем батче, query — на полной отложенной выборке."
            ),
            "",
            "![Train и query loss](../experiments/dynamic_signal_v1/figures/bank_repeats/loss.png)",
            "",
            "![Train и query recall](../experiments/dynamic_signal_v1/figures/bank_repeats/recall.png)",
            "",
            "![Обучение распределения индексаторов](../experiments/dynamic_signal_v1/figures/bank_repeats/prior_loss.png)",
            "",
            (
                "Пять повторов проверяют перенос одной общей структуры внутри каждого "
                "семейства задач. Близость AR, частот и одного индексатора не даёт "
                "оснований приписывать выигрыш именно авторегрессии. Малое число уникальных "
                "индексаторов также сохраняется: генерация новых сложных сочетаний этим "
                "опытом не проверена. Неполное восстановление в части банков нужно отличать "
                "от полного восстановления в первоначальном опыте."
            ),
            "",
            (
                "Код воспроизведения: `experiments/dynamic_signal_v1/repeat_banks.py`; "
                "таблицы, индивидуальные метрики, исходные веса извлечения, индексаторы, "
                "настройки и проверка хешей — в `experiments/dynamic_signal_v1/results/bank_repeats/`. "
                "Seeds зафиксированы: 2111481, 3111481, 4111481, 5111481, 6111481."
            ),
        ]
    )
    Path("mds/DYNAMIC_SIGNAL_FIVE_BANKS_2026-11-01.md").write_text(
        "\n".join(lines) + "\n"
    )
    hashes = {
        str(path.relative_to(OUTPUT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in OUTPUT.rglob("*")
        if path.is_file() and path.name != "export_manifest.json"
    }
    (OUTPUT / "export_manifest.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
