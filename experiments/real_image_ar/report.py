"""Read finished real-image experiments, export evidence and draw comparisons."""

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

from experiments.dynamic_signal_v1.report import best_rows
from experiments.real_image_ar.common import BASE, load_plan

LABELS = {
    "ar": "Авторегрессия",
    "frequency": "Частоты",
    "consensus": "Одна маска по частотам",
    "bank": "Целая исходная маска",
    "random": "Случайная маска",
    "untrained_ar": "Необученный MADE",
    "adam": "Плотный Adam",
    "sgd": "Плотный SGD",
    "transfer": "Перенос весов",
    "frozen_transfer": "Замороженное тело",
    "linear": "Линейная модель",
    "adam_long": "Adam: вчетверо больше шагов",
    "sgd_long": "SGD: вчетверо больше шагов",
}
RESULTS = BASE / "results/v1"
FIGURES = BASE / "figures/v1"


def target_file(root, size, method):
    directory = root / "target" / f"n{size}" / method
    metadata = json.loads((directory / "complete.json").read_text())
    return directory / metadata["files"][0]


def export(root, destination):
    """Preserve all numeric outcomes and source evidence without raw images."""
    for source in sorted(root.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(root)
        if source.name == "source.npz" or (
            source.name.startswith(("calibration_", "target_"))
            and source.suffix == ".npz"
            and not source.name.endswith("_ids.npz")
        ):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".npz":
            with np.load(source) as arrays:
                kept = {
                    key: arrays[key]
                    for key in arrays.files
                    if not key.startswith("parameter_")
                }
                if "target" in relative.parts and "parameter_0" in arrays:
                    ids = best_rows(arrays)
                    kept.update(
                        {
                            "selected_" + key: arrays[key][ids]
                            for key in arrays.files
                            if key.startswith("parameter_")
                        }
                    )
                if relative.parts[0] == "calibration" and "selected_inputs" in kept:
                    unique, inverse = np.unique(
                        kept.pop("selected_inputs"), axis=0, return_inverse=True
                    )
                    kept["selected_inputs_unique"] = unique.astype(np.uint8)
                    kept["selected_inputs_inverse"] = inverse
                np.savez_compressed(target, **kept)
        elif source.suffix == ".pt":
            target.with_suffix(".sha256").write_text(
                hashlib.sha256(source.read_bytes()).hexdigest() + "\n"
            )
        elif relative.parts[0] == "calibration" and source.name.endswith(
            "_history.json.gz"
        ):
            with gzip.open(source, "rt") as stream:
                history = json.load(stream)
            metadata = json.loads(
                source.with_name(
                    source.name.removesuffix("_history.json.gz") + ".json"
                ).read_text()
            )
            configs = len(metadata["configs"])
            means = [
                {
                    "step": point["step"],
                    **{
                        split: {
                            key: np.asarray(value)
                            .reshape(configs, -1)
                            .mean(-1)
                            .tolist()
                            for key, value in point[split].items()
                        }
                        for split in ["train", "query"]
                    },
                }
                for point in history
            ]
            target.unlink(missing_ok=True)
            with gzip.open(
                target.with_name(
                    target.name.replace(
                        "_history.json.gz", "_history_by_config.json.gz"
                    )
                ),
                "wt",
            ) as stream:
                json.dump(means, stream)
        elif source.suffix in [".json", ".gz"]:
            shutil.copyfile(source, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-export", action="store_true")
    args = parser.parse_args()
    plan = load_plan()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    metrics, per_pair, selected, roots = [], [], {}, {}
    source_summary, freeze_audits, examples = {}, {}, {}
    for dataset in plan["datasets"]:
        root = RESULTS / dataset if args.from_export else Path(plan["run"]) / dataset
        roots[dataset] = root
        frozen = json.loads((root / "frozen.json").read_text())
        if not args.from_export:
            mismatches = [
                path
                for path, expected in frozen["hashes"].items()
                if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected
            ]
            if mismatches:
                raise RuntimeError(f"Frozen source inputs changed: {mismatches}")
            (root / "freeze_audit.json").write_text(
                json.dumps(
                    {"checked": len(frozen["hashes"]), "mismatches": mismatches},
                    indent=2,
                )
                + "\n"
            )
        freeze_audits[dataset] = json.loads((root / "freeze_audit.json").read_text())
        source_summary[dataset] = {
            "bank": json.loads((root / "bank_audit/complete.json").read_text()),
            "prior": json.loads((root / "prior.json").read_text()),
        }
        cases = json.loads((root / "target_cases.json").read_text())
        for size in plan["support_per_class"]:
            for method in plan["methods"] + (
                ["adam_long", "sgd_long"] if size == 16 else []
            ):
                path = target_file(root, size, method)
                with np.load(path) as arrays:
                    ids = best_rows(arrays)
                    values = {
                        key: arrays[key][ids]
                        for key in [
                            "test_accuracy",
                            "test_loss",
                            "test_recall_positive",
                            "test_recall_negative",
                            "query_loss",
                        ]
                    }
                    selected[dataset, size, method] = {
                        "path": path,
                        "ids": ids,
                        **values,
                    }
                    masks = np.zeros((len(ids), plan["inputs"]), np.uint8)
                    if method not in ["frozen_transfer"]:
                        np.put_along_axis(
                            masks, arrays["selected_inputs"][ids], 1, axis=-1
                        )
                    if size == 16 and method in [
                        "ar",
                        "frequency",
                        "consensus",
                        "bank",
                        "random",
                    ]:
                        examples[dataset + "_" + method] = masks[0].reshape(14, 14)
                    metrics.append(
                        {
                            "dataset": dataset,
                            "support_per_class": size,
                            "method": method,
                            "accuracy": float(values["test_accuracy"].mean()),
                            "bce": float(values["test_loss"].mean()),
                            "recall_positive": float(
                                values["test_recall_positive"].mean()
                            ),
                            "recall_negative": float(
                                values["test_recall_negative"].mean()
                            ),
                            "selected_unique_masks": len(np.unique(masks, axis=0))
                            if method
                            in [
                                "ar",
                                "frequency",
                                "consensus",
                                "bank",
                                "random",
                                "untrained_ar",
                            ]
                            else None,
                        }
                    )
                    for pair_id in range(6):
                        rows = slice(
                            pair_id * plan["target_draws"],
                            (pair_id + 1) * plan["target_draws"],
                        )
                        per_pair.append(
                            {
                                "dataset": dataset,
                                "support_per_class": size,
                                "method": method,
                                "pair": "-".join(
                                    map(
                                        str,
                                        cases[pair_id * plan["target_draws"]]["pair"],
                                    )
                                ),
                                "accuracy": float(values["test_accuracy"][rows].mean()),
                            }
                        )
        if not args.from_export:
            data = np.load(root / "target_16.npz")
            examples[dataset + "_image"] = data["test_x"][0, 0].reshape(14, 14)
            export(root, RESULTS / dataset)
    for name, rows in [("comparison", metrics), ("per_pair", per_pair)]:
        with (RESULTS / f"{name}.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if args.from_export:
        stored_examples = np.load(RESULTS / "examples.npz")
        examples.update({key: stored_examples[key] for key in stored_examples.files})
    else:
        np.savez_compressed(RESULTS / "examples.npz", **examples)

    primary = plan["primary"]
    name, size = primary["dataset"], primary["support_per_class"]
    competitor = json.loads((roots[name] / "frozen.json").read_text())[
        "primary_competitor"
    ]
    difference = (
        selected[name, size, "ar"]["test_accuracy"]
        - selected[name, size, competitor]["test_accuracy"]
    ).reshape(6, plan["target_draws"]) * 100
    rng = np.random.default_rng(plan["seed"] + 9000)
    groups = rng.integers(6, size=(20000, 6))
    draws = rng.integers(plan["target_draws"], size=(20000, 6, plan["target_draws"]))
    bootstrap = difference[groups[:, :, None], draws].mean((1, 2))
    summary = {
        "primary": dict(primary, competitor=competitor),
        "difference_pp": float(difference.mean()),
        "difference_by_pair_pp": difference.mean(-1).tolist(),
        "descriptive_cluster_bootstrap_95_pp": np.quantile(
            bootstrap, [0.025, 0.975]
        ).tolist(),
        "statistical_limit": "Six pairs share four classes and finite official test pools; this interval is descriptive, not evidence from independent dataset families or independent source banks.",
        "source": source_summary,
        "freeze_audits": freeze_audits,
        "results": metrics,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    shown = [
        "ar",
        "frequency",
        "consensus",
        "bank",
        "random",
        "adam",
        "sgd",
        "transfer",
        "linear",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for axis, dataset in zip(axes, plan["datasets"]):
        for method in shown:
            rows = [
                row
                for row in metrics
                if row["dataset"] == dataset and row["method"] == method
            ]
            axis.plot(
                [row["support_per_class"] for row in rows],
                [100 * row["accuracy"] for row in rows],
                "o-",
                label=LABELS[method],
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks(
            plan["support_per_class"],
            [str(value) for value in plan["support_per_class"]],
        )
        axis.set_title(dataset)
        axis.set_xlabel("Обучающих изображений на класс (+64 query)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Точность на official test, %")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.19, 1, 1))
    fig.savefig(FIGURES / "comparison.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    methods = list(LABELS)
    for axis, dataset in zip(axes, plan["datasets"]):
        means = [
            100 * selected[dataset, 16, method]["test_accuracy"].mean()
            for method in methods
        ]
        axis.barh(np.arange(len(methods)), means, color="#9ecbc6")
        for position, method in enumerate(methods):
            values = (
                selected[dataset, 16, method]["test_accuracy"]
                .reshape(6, plan["target_draws"])
                .mean(-1)
                * 100
            )
            axis.scatter(values, np.full(6, position), s=11, color="#233858", alpha=0.6)
        axis.set_yticks(np.arange(len(methods)), [LABELS[method] for method in methods])
        axis.set_xlim(50, 100)
        axis.set_xlabel("Точность, %; точки — средние по парам классов")
        axis.set_title(dataset)
        axis.grid(axis="x", alpha=0.2)
    axes[0].invert_yaxis()
    fig.suptitle("Все методы: 16 обучающих изображений на класс +64 query")
    fig.tight_layout()
    fig.savefig(FIGURES / "primary_comparison.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    for field, ylabel in [
        ("loss", "BCE"),
        ("recall_positive", "Recall положительного класса"),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
        for column, dataset in enumerate(plan["datasets"]):
            for method in ["ar", "frequency", "consensus", "adam", "sgd", "transfer"]:
                item = selected[dataset, 16, method]
                path = item["path"]
                with gzip.open(
                    path.with_name(path.stem + "_history.json.gz"), "rt"
                ) as stream:
                    history = json.load(stream)
                for row, split in enumerate(["train", "query"]):
                    axes[row, column].plot(
                        [point["step"] for point in history],
                        [
                            np.asarray(point[split][field])[item["ids"]].mean()
                            for point in history
                        ],
                        label=LABELS[method],
                    )
                    axes[row, column].grid(alpha=0.2)
            axes[0, column].set_title(dataset)
            axes[1, column].set_xlabel("Шаги / батчи")
        axes[0, 0].set_ylabel("Train: " + ylabel)
        axes[1, 0].set_ylabel("Query: " + ylabel)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        fig.savefig(FIGURES / f"{field}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for column, dataset in enumerate(plan["datasets"]):
        chosen = source_summary[dataset]["prior"]["chosen"]
        histories = json.loads((roots[dataset] / "prior_history.json").read_text())
        history = next(
            item["history"]
            for item in histories
            if item["config"]["width"] == chosen["width"]
            and item["config"]["lr"] == chosen["learning_rate"]
            and item["config"]["weight_decay"] == chosen["weight_decay"]
        )
        for row, field in enumerate(
            ["nll_per_position", "teacher_forced_support_recall"]
        ):
            for split in ["train", "validation"]:
                axes[row, column].plot(
                    [point["step"] for point in history],
                    [point[split][field] for point in history],
                    label=split,
                )
            axes[row, column].axvline(
                chosen["step"], color="black", ls=":", label="Выбранное состояние"
            )
            axes[row, column].grid(alpha=0.2)
        axes[0, column].set_title(dataset)
        axes[1, column].set_xlabel("Шаги обучения MADE")
    axes[0, 0].set_ylabel("NLL индекса с учётом бюджета")
    axes[1, 0].set_ylabel("Recall связей, teacher forcing")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "prior_training.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for column, dataset in enumerate(plan["datasets"]):
        info = json.loads((roots[dataset] / "pretrain.json").read_text())
        row_id = info["chosen_row"]
        history = json.loads((roots[dataset] / "pretrain_history.json").read_text())
        for row, suffix in enumerate(["loss", "macro_recall"]):
            for split in ["train", "validation"]:
                axes[row, column].plot(
                    [point["step"] for point in history],
                    [point[split + "_" + suffix][row_id] for point in history],
                    label=split,
                )
            axes[row, column].axvline(info["best_steps"][row_id], ls=":", color="black")
            axes[row, column].grid(alpha=0.2)
        axes[0, column].set_title(dataset)
        axes[1, column].set_xlabel("Шаги исходного предобучения")
    axes[0, 0].set_ylabel("Кросс-энтропия по шести исходным классам")
    axes[1, 0].set_ylabel("Macro recall исходных классов")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "pretraining.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for column, dataset in enumerate(plan["datasets"]):
        histories = []
        for path in sorted((roots[dataset] / "bank_audit").glob("*_history.json.gz")):
            with gzip.open(path, "rt") as stream:
                histories.append(json.load(stream))
        for row, field in enumerate(["loss", "recall_positive"]):
            for split in ["train", "query"]:
                values = [
                    np.concatenate(
                        [history[step][split][field] for history in histories]
                    ).mean()
                    for step in range(len(histories[0]))
                ]
                axes[row, column].plot(
                    [point["step"] for point in histories[0]], values, label=split
                )
            axes[row, column].grid(alpha=0.2)
        axes[0, column].set_title(dataset)
        axes[1, column].set_xlabel("Шаги нового обучения весов исходных масок")
    axes[0, 0].set_ylabel("BCE: все исходные старты и настройки")
    axes[1, 0].set_ylabel("Recall положительного класса")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "bank_audit_training.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 6, figsize=(13, 5))
    for row, dataset in enumerate(plan["datasets"]):
        axes[row, 0].imshow(examples[dataset + "_image"], cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(dataset + "\nкласс 6")
        for column, method in enumerate(
            ["ar", "frequency", "consensus", "bank", "random"], 1
        ):
            axes[row, column].imshow(
                examples[dataset + "_" + method], cmap="Blues", vmin=0, vmax=1
            )
            axes[row, column].set_title(
                {
                    "ar": "AR",
                    "frequency": "Частоты",
                    "consensus": "Top-64 частот",
                    "bank": "Маска банка",
                    "random": "Случайная",
                }[method]
            )
        for axis in axes[row]:
            axis.axis("off")
    fig.suptitle("Первая задача 6/7, первый набор; маски моделей выбраны по query")
    fig.tight_layout()
    fig.savefig(FIGURES / "masks.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for axis, dataset in zip(axes, plan["datasets"]):
        for method in ["ar", "frequency", "consensus", "adam", "sgd", "transfer"]:
            rows = [
                row
                for row in per_pair
                if row["dataset"] == dataset
                and row["support_per_class"] == 16
                and row["method"] == method
            ]
            axis.plot(
                [row["pair"] for row in rows],
                [100 * row["accuracy"] for row in rows],
                "o-",
                label=LABELS[method],
            )
        axis.set_title(dataset)
        axis.set_xlabel("Новая пара классов")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Точность при 16 обучающих изображениях на класс, %")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(FIGURES / "per_pair.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# Авторегрессионный перенос масок на реальных изображениях",
        "",
        "Проверили основной авторегрессионный генератор на MNIST и Fashion-MNIST. На входе одно реальное изображение с исходной меткой, уменьшенное до 14×14. Отвлекающих картинок и скрытой перестановки нет. Задача бинарная: различить выбранные два класса. Исходные классы — 0–5, новые — 6–9.",
        "",
        "## Что и как переносили",
        "",
        "Индексатор выбирает 64 из 196 входов. После фиксации маски заново обучается MLP 64→16→1. Все ненулевые коэффициенты независимы: это проверка масок, частного случая индексатора, а не переноса равенств ненулевых весов. У модели 1057 активных параметров; у плотной MLP — 3169.",
        "",
        "В прежнем исходном банке каждого датасета было 832 маски. Каждую заново проверили при двух скоростях обучения, по четыре старта численных весов и 2048 шагов: всего восемь обучений на маску. Оставили лучшие 75% внутри исходной задачи и роли ветви поиска: 624 маски, из них 312 для обучения, по 156 для выбора генератора и аудита. Порог почти идеальной точности не требовался.",
        "",
        "MADE получает только маски. Порядок позиций случайный и фиксирован заранее. Известен только бюджет 64: при декодировании оставшиеся биты принудительно задаются, когда бюджет исчерпан или нужны все оставшиеся позиции. Это же ограничение применяется к частотам. Корреляции пикселей, координатная геометрия и функциональная полезность входов не используются.",
        "",
        "Численные параметры, температура и генератор выбирались на исходных данных. Отложенные исходные пары — 0/5 и 2/3. Все настройки заморожены до создания целевых случаев. На новой задаче — восемь стартов по 4096 шагов; состояние и старт выбираются по query BCE. Для Adam/SGD при малом support дополнительно проверен четырёхкратный бюджет.",
        "",
        "## Основная таблица",
        "",
        "В каждой ячейке — средняя test-точность по шести новым парам классов и восьми наборам обучения на пару. На класс дополнительно доступны 64 query-изображения для выбора, test содержит 512 изображений из официальной тестовой части. Поэтому 16 support на класс не означает только 16 доступных размеченных изображений.",
        "",
        "| Метод | MNIST: 16 | 64 | 256 | Fashion: 16 | 64 | 256 |",
        "| :-- | --: | --: | --: | --: | --: | --: |",
    ]
    for method, label in LABELS.items():
        values = []
        for dataset in plan["datasets"]:
            for size in plan["support_per_class"]:
                found = [
                    row
                    for row in metrics
                    if row["dataset"] == dataset
                    and row["support_per_class"] == size
                    and row["method"] == method
                ]
                values.append(f"{100 * found[0]['accuracy']:.2f}%" if found else "—")
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "![Сравнение по объёму обучения](../experiments/real_image_ar/figures/v1/comparison.png)",
            "",
            "![Все новые пары классов](../experiments/real_image_ar/figures/v1/per_pair.png)",
            "",
            "## Предварительно выбранное основное сравнение",
            "",
            f"Fashion-MNIST, 16 обучающих изображений на класс: AR против «{LABELS[competitor]}», выбранного по исходной калибровке. Разница AR − конкурент: **{difference.mean():.2f} п.п.** По шести парам: "
            + "; ".join(f"{value:+.2f}" for value in difference.mean(-1))
            + " п.п.",
            "",
            "Описательный bootstrap пересэмплирует пары, затем наборы внутри пары. Его 95% интервал: ["
            + "; ".join(
                f"{value:.2f}"
                for value in summary["descriptive_cluster_bootstrap_95_pp"]
            )
            + "] п.п. Пары разделяют четыре класса и конечные тестовые изображения. Это один исходный банк на датасет, а не 48 независимых семейств и не проверка устойчивости между банками.",
            "",
            "## Что выучил генератор",
            "",
        ]
    )
    for dataset in plan["datasets"]:
        prior = source_summary[dataset]["prior"]
        chosen = prior["chosen"]
        lines.append(
            f"**{dataset}:** ширина {chosen['width']}, выбран шаг {chosen['step']}. NLL на отложенных исходных масках: AR {prior['held_source_audit']['nll_per_position']:.4f}, частоты {prior['frequency_held_source_audit']['nll_per_position']:.4f}."
        )
        lines.append("")
    lines.extend(
        [
            "Оба генератора выбрали раннее состояние, шаг 50 из 2000. NLL улучшилась слабо, а затем заметно росла на validation при снижении train NLL. Близость частот и AR не подтверждает полезного обучения сложных зависимостей. Consensus отдельно выбирает 64 наиболее частых входа; это другой способ декодирования, чем последовательное сэмплирование с бюджетом.",
            "",
            "![Обучение генератора](../experiments/real_image_ar/figures/v1/prior_training.png)",
            "",
            "## Примеры структур и обучение чисел",
            "",
            "Аналитического правильного индексатора у этих изображений нет. Ниже реальные маски выбранных моделей на первой фиксированной задаче 6/7. Изображение показано для объяснения; оно не использовалось для выбора примера или маски по test.",
            "",
            "![Изображения и маски](../experiments/real_image_ar/figures/v1/masks.png)",
            "",
            "Кривые ниже усреднены по выбранным по query стартам всех 48 случаев. Train измеряется на текущем батче, query — на полной отложенной выборке. Продолжение кривой после лучшего состояния показывает дальнейшее обучение, а не состояние возвращённой модели.",
            "",
            "![Train/query loss](../experiments/real_image_ar/figures/v1/loss.png)",
            "",
            "![Train/query recall](../experiments/real_image_ar/figures/v1/recall_positive.png)",
            "",
            "Контроли: частоты из того же банка; одна top-64 маска; целая исходная маска; равномерная случайная маска; необученный MADE; плотные Adam/SGD; линейная модель; градиентное предобучение на том же союзе исходных изображений с дальнейшим обучением всех весов либо только новой головы. Ненулевые веса всех масочных методов обучаются заново.",
            "",
            "Это новая оценка авторегрессионного подхода на ранее изучавшихся датасетах. Совпадают целевые бюджеты обновлений, но стоимость исходного сбора, число параметров и FLOPs различны. Группировка входов и эталонная маска не сообщаются. Протокол, индивидуальные метрики, source-банк, выбранные численные состояния, кривые, конфигурации и аудит неизменности источников сохранены в `experiments/real_image_ar/results/v1/`. Команды: `experiments/real_image_ar/README.md`.",
        ]
    )
    if all(
        selected[dataset, size, "ar"]["test_accuracy"].mean()
        < selected[dataset, size, "sgd"]["test_accuracy"].mean()
        for dataset in plan["datasets"]
        for size in plan["support_per_class"]
    ):
        lines[2:2] = [
            "**Устойчивого преимущества метода в этой серии не получили.** AR уступил плотному SGD по средней test-точности во всех шести сочетаниях датасета и размера обучения. Частоты и случайные маски дают близкое к AR качество. Это отрицательный результат для данного банка и варианта масок, а не доказательство невозможности структурного переноса.",
            "",
        ]
    position = (
        next(
            i for i, text in enumerate(lines) if text.startswith("Индексатор выбирает")
        )
        + 1
    )
    lines[position:position] = [
        "",
        "Обозначим маску через m, а список включённых входов через S. Модель задаётся формулами:",
        "",
        "$$",
        r"\begin{aligned}",
        r"m&\in\{0,1\}^{196},\quad \sum_i m_i=64,\quad S=\{i:m_i=1\},\\",
        r"f(x)&=a^\top\operatorname{ReLU}(C^\top x_S+b)+o,\quad C\in\mathbb R^{64\times16},\quad a,b\in\mathbb R^{16}.",
        r"\end{aligned}",
        "$$",
        "",
        "Маска фиксируется, а C, a, b и o обучаются по бинарной кросс-энтропии; положительный логит означает второй класс пары. У каждого активного веса C собственное численное значение.",
    ]
    position = next(
        i for i, text in enumerate(lines) if text.startswith("![Сравнение по объёму")
    )
    lines[position:position] = [
        "![Все методы при малом объёме обучения](../experiments/real_image_ar/figures/v1/primary_comparison.png)",
        "",
    ]
    lines.extend(
        [
            "",
            "Плотность 64/196 зафиксирована по семейству готового банка; другие бюджеты масок в этой серии не подбирались. Результат ограничен данным банком и отбором входов. Для оценки подхода с повторениями ненулевых весов нужен другой эксперимент.",
            "Калибровочные истории в компактном архиве усреднены по конфигурациям (`*_history_by_config.json.gz`), итоговые метрики всех стартов сохранены. Калибровочные списки входов восстанавливаются как `selected_inputs_unique[selected_inputs_inverse].astype('int64')`. Целевые истории и выбранные входы сохранены полностью.",
            "",
            "Дополнительные исходные кривые: [аудит масок](../experiments/real_image_ar/figures/v1/bank_audit_training.png), [градиентное предобучение](../experiments/real_image_ar/figures/v1/pretraining.png). Аудит усреднён по всем исходным стартам и двум скоростям; предобучение показывает конфигурацию и старт, выбранные по исходной validation.",
        ]
    )
    Path("mds/REAL_IMAGE_AUTOREGRESSIVE_2026-11-02.md").write_text(
        "\n".join(lines) + "\n"
    )
    hashes = {
        str(path.relative_to(RESULTS)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in RESULTS.rglob("*")
        if path.is_file() and path.name != "export_manifest.json"
    }
    (RESULTS / "export_manifest.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: summary[key]
                for key in [
                    "primary",
                    "difference_pp",
                    "difference_by_pair_pp",
                    "descriptive_cluster_bootstrap_95_pp",
                    "results",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
