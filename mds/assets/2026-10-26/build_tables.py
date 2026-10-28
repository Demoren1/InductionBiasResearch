"""Refresh marked report tables from the saved experiment summaries."""

import json
from pathlib import Path
import re

import numpy as np


ASSETS = Path(__file__).resolve().parent
ROOT = ASSETS.parents[2]
RESULTS = ROOT / "paper_vae_solver/results"
REPORTS = [
    ROOT / "final_mds/FINAL_REPORT_2026-10-26.md",
    ROOT / "final_mds/FINAL_REPORT_2026-10-26_APPENDIX.md",
]


def load(name):
    return json.loads((RESULTS / name).read_text())


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in rows],
        ]
    )


def count(row):
    successes = row.get("success", row.get("successes"))
    searches = row.get("searches", row.get("n"))
    return f"{successes}/{searches} ({100 * successes / searches:.1f}%)"


def dimensional(rows, methods, **filters):
    selected = {
        (row["method"], row["dimension"]): row
        for row in rows
        if all(row[key] == value for key, value in filters.items())
    }
    return [
        [label]
        + [
            f"{selected[key, dimension]['rate'] * 100:.1f}%"
            for dimension in [16, 32, 48]
        ]
        for key, label in methods
    ]


def main():
    summary = load("summary.json")
    codebook = load("codebook_summary.json")
    hidden = load("hidden_structure_summary.json")
    robustness = load("robustness_twolayer_summary.json")
    example = json.loads((ASSETS / "solution_example.json").read_text())
    tables = {}
    labels = [
        ("oracle", "Правильная маска задана заранее"),
        ("enumerate_roles", "Перебор 16 допустимых масок"),
        ("uniform_roles", "Случайная маска с учётом двух выходов"),
        ("vae", "VAE с раздельным обучением частей"),
        ("product", "Выбор частей по исходным частотам"),
        ("dense_adam", "Плотный Adam"),
        ("uniform_global", "Две случайные связи из всех 16"),
        ("vanilla", "Обычный VAE"),
        ("dense_momentum", "Плотный SGD с моментом"),
        ("dense_sgd", "Плотный SGD"),
        ("replay", "Выбор готовой исходной маски"),
    ]
    masks = {row["method"]: row for row in summary["matched_v2"]}
    tables["masks"] = table(
        ["Метод", "Успех", "Средняя точная MSE ↓"],
        [
            [label, count(masks[key]), f"{masks[key]['mean_mse']:.4f}"]
            for key, label in labels
        ],
    )
    replicated = {row["method"]: row for row in summary["replication_v2"]}
    tables["replication"] = table(
        ["Метод", "Успех", "Средняя MSE", "Банки 751 / 752 / 753, %"],
        [
            [
                label,
                count(replicated[key]),
                f"{replicated[key]['mean_mse']:.4f}",
                " / ".join(
                    f"{value * 100:.1f}"
                    for value in replicated[key]["seed_rates"].values()
                ),
            ]
            for key, label in labels
            if key in replicated
        ],
    )
    scaling = {
        (row["method"], row["dimension"]): row for row in summary["scaling"]
    }
    tables["mask_scaling"] = table(
        ["Размерность", "VAE", "Выбор по частотам", "Adam", "SGD"],
        [
            [dimension]
            + [
                f"{scaling[key, dimension]['rate'] * 100:.1f}%"
                for key in ["vae", "product", "dense_adam", "dense_sgd"]
            ]
            for dimension in [8, 16, 32, 64]
        ],
    )
    code_methods = [
        ("code_vae", "VAE, новые значения"),
        ("code_blocks", "Исходные блоки, новые значения"),
        ("code_bits", "Независимые биты"),
        ("code_replay", "Выбор готового исходного разбиения"),
        ("code_transfer", "Блоки, начальные значения от исходных задач"),
        ("code_fixed", "Блоки, исходные значения без обучения"),
        ("code_raw_init", "Блоки без нормировки инициализации"),
        ("code_sum_gradient", "Блоки с суммированием градиентов"),
        ("code_oracle", "Правильные назначения, новые значения"),
        ("code_oracle_transfer", "Верное разбиение, исходные значения"),
        ("dense_sgd", "Плотный SGD"),
        ("dense_adam", "Плотный Adam"),
        ("dense_replay", "Перенос всего вектора исходных весов"),
        ("sparse_two", "Только две ненулевые связи"),
    ]
    tables["codebook"] = table(
        ["Метод", "16 весов", "32 веса", "48 весов"],
        dimensional(codebook["rows"], code_methods),
    )
    tables["coverage"] = table(
        [
            "Метод",
            "Весов",
            "Верное разбиение встретилось",
            "Верное разбиение выбрано",
            "Успех",
        ],
        [
            [label, dimension]
            + [
                f"{row[field] * 100:.1f}%"
                for field in [
                    "correct_assignment_seen",
                    "correct_assignment_chosen",
                    "rate",
                ]
            ]
            for key, label in code_methods[:2]
            for dimension in [16, 32, 48]
            for row in codebook["rows"]
            if row["method"] == key and row["dimension"] == dimension
        ],
    )
    hidden_methods = [
        ("code_vae", "VAE без заданных групп"),
        ("code_components", "Восстановленные группы"),
        ("code_known_blocks", "Границы блоков заданы заранее"),
        ("code_wrong_blocks", "Неверные границы"),
        ("code_bits", "Независимый выбор группы каждой координаты"),
        ("code_replay", "Выбор готового исходного разбиения"),
        ("code_uniform", "Равномерные назначения"),
        ("code_enumerate", "Перебор сочетаний восстановленных групп"),
        ("code_cem", "Адаптивный поиск по группам (CEM)"),
        ("code_cem_bits", "Адаптивный поиск по координатам"),
        ("code_oracle", "Правильное разбиение задано заранее"),
        ("dense_sgd", "Плотный SGD"),
        ("dense_adam", "Плотный Adam"),
    ]
    tables["hidden"] = table(
        ["Метод", "16 весов", "32 веса", "48 весов"],
        dimensional(
            hidden["rows"],
            hidden_methods,
            source_size="all",
            condition="shared_permutation",
        ),
    )
    spectral_methods = [
        ("spectral_components", "Подбор частоты по восстановленным группам"),
        ("spectral_uniform", "Подбор частоты по случайным разбиениям"),
        ("moment_radial", "Оценка направления по данным + подбор частоты"),
    ]
    full_hidden = {
        (row["method"], row["dimension"]): row
        for row in hidden["rows"]
        if row["source_size"] == "all"
        and row["condition"] == "shared_permutation"
    }
    tables["spectral"] = table(
        ["Метод", "16 весов", "32 веса", "48 весов"],
        [
            [label] + [count(full_hidden[key, dim]) for dim in [16, 32, 48]]
            for key, label in spectral_methods
        ],
    )
    control_rows = {
        (
            row["method"],
            row["dimension"],
            row["source_size"],
            row["condition"],
        ): row
        for row in hidden["rows"]
    }
    tables["hidden_controls"] = table(
        [
            "Метод",
            "8 источников, 32 веса",
            "8 источников, 48 весов",
            "Новый порядок входов у цели, 48 весов",
        ],
        [
            [label]
            + [
                count(control_rows[key, dim, size, condition])
                for dim, size, condition in [
                    (32, "8", "shared_permutation"),
                    (48, "8", "shared_permutation"),
                    (48, "all", "new_target_permutation"),
                ]
            ]
            for key, label in [
                hidden_methods[0],
                hidden_methods[1],
                hidden_methods[7],
                hidden_methods[10],
                hidden_methods[-1],
                spectral_methods[0],
            ]
        ],
    )
    example_labels = dict(hidden_methods + spectral_methods)
    tables["example"] = table(
        [
            "Метод",
            "MSE первого поиска",
            "Средняя MSE всех запусков на этом примере",
            "Амплитуда первого решения",
        ],
        [
            [
                example_labels[key],
                f"{row['population_mse']:.6g}",
                f"{np.mean(row['all_trial_population_mse']):.6g}",
                f"{row['amplitude']:.6g}",
            ]
            for key, row in example["methods"].items()
        ],
    )
    robust_methods = [
        "components_hard",
        "components_release",
        "vae_release",
        "dense_adam",
    ]
    tables["residual"] = table(
        [
            "Норма остатка",
            "Группы, веса связаны",
            "Группы, затем свободные веса",
            "VAE, затем свободные веса",
            "Adam",
        ],
        [
            [epsilon]
            + [
                count(robustness["robustness"][f"noise_0/e{epsilon:g}"][key])
                for key in robust_methods
            ]
            for epsilon in [0, 0.1, 0.3, 0.6, 1, 2]
        ],
    )
    tables["source_quality"] = table(
        [
            "Состояние источников",
            "Группы, веса связаны",
            "Группы, затем свободные веса",
            "VAE, затем свободные веса",
            "Adam",
        ],
        [
            [label]
            + [
                count(robustness["robustness"][condition][key])
                for key in robust_methods
            ]
            for condition, label in [
                ("noise_0/e0", "Исходный чистый банк"),
                ("early_128/e0", "Исходные задачи: 128 шагов"),
                ("early_32/e0", "Исходные задачи: 32 шага"),
                ("early_8/e0", "Исходные задачи: 8 шагов"),
                ("noise_0.25/e0", "Уровень шума 0,25"),
                ("noise_0.5/e0", "Уровень шума 0,5"),
                ("noise_1/e0", "Уровень шума 1"),
                ("noise_0.5/e0.3", "Уровень шума 0,5; остаток 0,3"),
            ]
        ],
    )
    tanh_labels = [
        ("dense_adam", "Плотный Adam"),
        ("numerical_transfer", "Перенос численных параметров + Adam"),
        ("oracle", "Правильная группировка"),
        ("components_release", "Группы, затем свободные веса"),
        ("components", "Восстановленные группы"),
        ("vae", "VAE"),
        ("dense_sgd", "Плотный SGD"),
        ("replay", "Выбор готового исходного разбиения"),
        ("uniform", "Равномерная группировка"),
    ]
    tanh = robustness["twolayer"]["two_layer_tanh"]
    tables["twolayer"] = table(
        [
            "Метод",
            "Успех",
            "Средняя MSE после нормировки",
            "Диапазон успеха по банкам, %",
        ],
        [
            [
                label,
                count(tanh[key]),
                f"{tanh[key]['mean_mse']:.6f}",
                f"{min(tanh[key]['bank_rates'].values()) * 100:.1f}–"
                f"{max(tanh[key]['bank_rates'].values()) * 100:.1f}",
            ]
            for key, label in tanh_labels
        ],
    )
    population = {
        (row["dimension"], row["mode"]): row
        for row in summary["population"]
        if row["step"] == 4096
    }
    dimensions = sorted({dimension for dimension, _ in population})
    tables["population"] = table(
        [
            "Размерность",
            "Постоянный масштаб старта",
            "Масштаб уменьшается с размерностью",
            "Точная маска",
        ],
        [
            [dimension]
            + [
                f"{population[dimension, key]['success']}/8192 "
                f"({population[dimension, key]['success'] / 8192 * 100:.2f}%)"
                for key in ["dense_fixed", "dense_fanin", "oracle"]
            ]
            for dimension in dimensions
        ],
    )
    texts = {path: path.read_text() for path in REPORTS}
    for name, content in tables.items():
        pattern = rf"(<!-- table:{name} -->).*?(<!-- /table:{name} -->)"
        replacements = 0
        for path, text in texts.items():
            texts[path], count_replaced = re.subn(
                pattern,
                lambda match: f"{match[1]}\n\n{content}\n\n{match[2]}",
                text,
                flags=re.DOTALL,
            )
            replacements += count_replaced
        if replacements != 1:
            raise ValueError(f"Expected one report table marker: {name}")
    for path, text in texts.items():
        path.write_text(text)
    print(f"Updated {len(tables)} tables in {len(REPORTS)} documents")


if __name__ == "__main__":
    main()
