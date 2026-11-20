"""Summarize the fixed-budget linear and real-data transfer experiments."""

import argparse
import json

import numpy as np
from scipy.stats import t

from experiments.sharing_structure_research_v1.blind_contract import save_json

from .common import CONFIG, METHODS, OUT

NAMES = {
    "dense": "Dense",
    "wsr": "WSR",
    "lutq": "LUT-Q",
    "yeh_map": "Yeh hard",
    "yeh_soft": "Yeh soft",
    "msr": "MSR-FC",
    "asrr": "Ward-aSRR",
    "shuffled": "Перестановка",
    "oracle": "Истинные группы",
}
DATASETS = {
    "mnist_torus": "MNIST на торе",
    "aug_omniglot": "Aug-Omniglot",
    "fc100": "FC100",
    "har": "UCI HAR",
    "wisdm": "WISDM",
}


def confidence(values):
    mean = float(np.mean(values))
    margin = (
        t.ppf(0.975, len(values) - 1)
        * np.std(values, ddof=1)
        / np.sqrt(len(values))
    )
    return {"mean": mean, "ci95": [float(mean - margin), float(mean + margin)]}


def linear_summary(final):
    config = CONFIG["synthetic"]
    stages = ("baselines", "ward") if final else ("baselines",)
    geometries = [
        OUT / "linear" / f"geometry_{i:02d}"
        for i in range(config["geometries"])
    ]
    avg, structure = {}, {}
    for g in geometries:
        for stage in stages:
            info = json.loads((g / f"{stage}_results.json").read_text())
            for key, value in info["mse"].items():
                avg.setdefault(key, []).append(value)
            for key, record in info["structure"].items():
                structure.setdefault(key, []).append(record)
    result = {
        "mse": {
            key: confidence(observations) for key, observations in avg.items()
        }
    }
    result["signed_pairs"] = {
        key: {
            "precision": float(
                np.mean(
                    [
                        r["correct_signed_ties"] / max(1, r["proposed_ties"])
                        for r in observations
                    ]
                )
            ),
            "recall": float(
                np.mean(
                    [
                        r["correct_signed_ties"] / r["teacher_ties"]
                        for r in observations
                    ]
                )
            ),
            "exact": sum(
                r["correct_signed_ties"]
                == r["teacher_ties"]
                == r["proposed_ties"]
                for r in observations
            ),
        }
        for key, observations in structure.items()
    }
    if final:
        result["paired_mse_reduction"] = {
            f"{method}_{shot}": confidence(
                np.asarray(avg[f"{method}_{shot}"])
                - np.asarray(avg[f"asrr_{shot}"])
            )
            for method in (*METHODS, "shuffled", "oracle")
            for shot in config["shots"]
        }
    return result


def grouped(values, groups):
    return np.asarray(
        [values[groups == unit].mean() for unit in np.unique(groups)]
    )


def real_summary(final):
    methods = (*METHODS, "asrr", "shuffled") if final else METHODS
    results = {}
    for dataset in DATASETS:
        first = None
        by_query = {}
        choices = {}
        for method in methods:
            path = OUT / dataset / f"{method}_choice.json"
            if path.exists():
                choices[method] = json.loads(path.read_text())
            with np.load(
                OUT / dataset / f"{method}_test.npz", allow_pickle=False
            ) as data:
                if first is None:
                    first = data["units"]
                if not np.array_equal(first, data["units"]):
                    raise ValueError(
                        f"Misaligned episode order in {dataset} {method}"
                    )
                for key in (
                    name for name in data.files if name.endswith("_acc")
                ):
                    by_query.setdefault(key.removesuffix("_acc"), {})[
                        method
                    ] = grouped(data[key], first)
        comparisons = {}
        acc = {}
        for key, observations in by_query.items():
            acc[key] = {
                method: confidence(value)
                for method, value in observations.items()
            }
            if final:
                comparisons[key] = {
                    method: confidence(observations["asrr"] - value)
                    for method, value in observations.items()
                    if method != "asrr"
                }
        results[dataset] = {
            "codes": 128
            if dataset == "har"
            else CONFIG["real"][dataset]["codes"],
            "support_per_class": 3
            if dataset == "har"
            else CONFIG["real"][dataset]["shot"],
            "episodes": len(first),
            "independent_units": len(np.unique(first)),
            "accuracy": acc,
            "asrr_minus_method": comparisons,
            "validation_choices": choices,
        }
    return results


def num(value, digits=3):
    return f"{value:.{digits}g}".replace(".", ",")


def percent(value):
    return f"{100 * value:.2f}".replace(".", ",")


def difference(statistic):
    def signed(value):
        return f"{100 * value:+.2f}".replace(".", ",").replace("-", "−")

    lower, upper = statistic["ci95"]
    return f"{signed(statistic['mean'])} [{signed(lower)}, {signed(upper)}]"


def markdown(synthetic, real, final):
    phase = (
        "Сравнение переноса структуры"
        if final
        else "Сравнение Dense и конкурентов"
    )
    lines = [
        f"# {phase}",
        "",
        "## 1. Восстановление на частичной одномерной кросс-корреляции",
        "",
        "Все методы используют один и тот же каркас из 204 связей для 70 входов "
        "и 68 наблюдаемых выходов. Для каждого из трёх смещений фильтра 51 позиция "
        "делит общий коэффициент, 17 имеют независимые коэффициенты. "
        "В каждом из шести расположений исключений используются 96 исходных функций "
        "с 96 примерами, 48 валидационных и 160 тестовых функций. "
        "Числа генерируются заново для каждой функции. Сила гребневой регрессии "
        "и представление выбираются на валидации отдельно для одного и пяти примеров.",
        "",
        "| Метод | Коэффициенты | Точность знаковых пар | Полнота | Точное восстановление "
        "| MSE, 1 пример | MSE, 5 примеров |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    linear_methods = (
        (*METHODS, "asrr", "shuffled", "oracle") if final else METHODS
    )
    for method in linear_methods:
        summary = synthetic["signed_pairs"].get(f"{method}_1", {})
        recovery = ["—", "—", "—"]
        if summary:
            recovery = [
                percent(summary["precision"]) + "%",
                percent(summary["recall"]) + "%",
                f"{summary['exact']}/{CONFIG['synthetic']['geometries']}",
            ]
        mse = [
            num(synthetic["mse"][f"{method}_{shot}"]["mean"], 4)
            for shot in CONFIG["synthetic"]["shots"]
        ]
        lines.append(
            "| "
            + " | ".join(
                [
                    NAMES[method],
                    "204" if method == "dense" else "54",
                    *recovery,
                    *mse,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Восстановление оценивается по индексаторам, выбранным для одного "
            "опорного примера. Непрерывные базисы оцениваются по MSE. "
            "На синтетике MSR-FC обучает непрерывный базис с дифференцированием "
            "через точную гребневую подгонку. У Yeh soft та же подгонка используется "
            "с вероятностной матрицей назначений. Yeh hard максимизирует сумму "
            "вероятностей назначений при заданном числе занятых кодов.",
            "",
            "## 2. Классификация на реальных данных",
            "",
            "Точность в процентах. Для Aug-Omniglot в каждой ячейке указаны "
            "чистые и преобразованные запросы в указанном порядке.",
            "",
        ]
    )
    methods = (*METHODS, "asrr", "shuffled") if final else METHODS
    headers = [
        "Набор",
        "Примеров на класс",
        "Коды",
        *[NAMES[m] for m in methods],
    ]
    if final:
        headers.append("Ward минус перестановка, п.п. [95% ДИ]")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|---|" + "---:|" * (len(headers) - 1))
    for dataset, row in real.items():
        queries = (
            ("clean", "query") if dataset == "aug_omniglot" else ("query",)
        )
        values = [
            " / ".join(percent(row["accuracy"][q][m]["mean"]) for q in queries)
            for m in methods
        ]
        cols = [
            DATASETS[dataset],
            str(row["support_per_class"]),
            str(row["codes"]),
            *values,
        ]
        if final:
            cols.append(
                " / ".join(
                    difference(row["asrr_minus_method"][q]["shuffled"])
                    for q in queries
                )
            )
        lines.append("| " + " | ".join(cols) + " |")
    lines.extend(
        [
            "",
            "Коды в таблице — число заново подгоняемых коэффициентов первого слоя "
            "у методов шэринга. Dense использует свободные веса на том же каркасе. "
            "На четырёх панелях число шагов Adam и скорость обучения выбираются "
            "по валидационной точности, при равенстве — по кросс-энтропии. "
            "Кандидаты предварительно сравниваются при скоростях 0,003 и 0,01 "
            "после 75 и 150 шагов. До трёх кандидатов на метод проходят полную "
            "сетку из четырёх скоростей и четырёх чисел шагов. "
            "Валидируем на двух численных стартах и усредняем точность двух "
            "новых стартов при оценке. Для Aug-Omniglot выбор относится "
            "к преобразованным запросам. UCI HAR пересчитан на исходных 180 эпизодах "
            "с зафиксированными правилами подгонки и усреднением 16 стартов.",
            "",
        *(
            [
                "Контроль на реальных данных усредняет четыре перестановки назначений. "
                "Они сохраняют разрешённые связи, знаки и размеры групп. "
                "Численное правило и начальные значения совпадают с Ward-aSRR.",
                "",
            ]
            if final
            else []
        ),
            "На MNIST на торе использованы 25 исходных, 10 валидационных и 10 "
            "тестовых пар цифр. Исходные изображения разделены до циклических "
            "сдвигов и общей перестановки пикселей. На FC100 повторно использованы "
            "16 исходных задач и сгенерированы новые эпизоды для валидации и теста "
            "на отложенных суперклассах.",
            "",
            "WSR строит кандидаты с изотоническим проксимальным шагом, "
            "LUT-Q использует обучаемый словарь, выбор ближайшего кода и обновление центров. "
            "Yeh обучает блочные вероятностные назначения с дифференцированием "
            "через три внутренних шага, MSR-FC использует факторизованный базис "
            "и из ранних обученных состояний переносит только пространство весов. "
            "При переносе все численные коэффициенты и выходные веса подгоняются заново. "
            "Это адаптации конкурентов к одной и той же архитектуре внутри каждого набора.",
            "",
            "На WISDM и UCI HAR интервалы считаются по участникам. На MNIST на торе "
            "единица сравнения — отложенная пара цифр. Для FC100 и Aug-Omniglot "
            "интервалы относятся к эпизодам из фиксированных пулов тестовых классов. "
            "На синтетике интервалы из JSON считаются по шести независимым геометриям.",
            "",
            "[Общий протокол](protocol.json) · "
            f"[Численные сравнения и интервалы](summary{'' if final else '_baselines'}.json)",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines", action="store_true")
    args = parser.parse_args()
    final = not args.baselines
    synthetic = linear_summary(final)
    real = real_summary(final)
    suffix = "" if final else "_baselines"
    save_json(
        OUT / f"summary{suffix}.json", {"linear": synthetic, "real": real}
    )
    report = OUT / f"REPORT{suffix}.md"
    report.write_text(markdown(synthetic, real, final), encoding="utf-8")
    print(report)
