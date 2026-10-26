"""Refresh marked report tables from the saved experiment summaries."""

import json
from pathlib import Path
import re

import numpy as np


ASSETS = Path(__file__).resolve().parent
ROOT = ASSETS.parents[2]
RESULTS = ROOT / "paper_vae_solver/results"
REPORT = ROOT / "mds/FINAL_REPORT_2026-10-26.md"


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
        ("oracle", "Правильная маска (oracle)"),
        ("enumerate_roles", "Перебор 16 ролевых масок"),
        ("uniform_roles", "Равномерно по известным ролям"),
        ("vae", "Partwise VAE"),
        ("product", "Эмпирический product-prior"),
        ("dense_adam", "Плотный Adam"),
        ("uniform_global", "Равномерно по всем двум связям"),
        ("vanilla", "Обычный VAE"),
        ("dense_momentum", "Плотный momentum"),
        ("dense_sgd", "Плотный SGD"),
        ("replay", "Replay исходных масок"),
    ]
    masks = {row["method"]: row for row in summary["matched_v2"]}
    tables["masks"] = table(
        ["Метод", "Успех", "Средняя population MSE ↓"],
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
        ["Размерность", "VAE", "Product", "Adam", "SGD"],
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
        ("code_blocks", "Эмпирические блоки, новые значения"),
        ("code_bits", "Независимые биты"),
        ("code_replay", "Replay целых назначений"),
        ("code_transfer", "Блоки, перенос source-значений"),
        ("code_fixed", "Блоки, фиксированные source-значения"),
        ("code_raw_init", "Блоки без нормировки инициализации"),
        ("code_sum_gradient", "Блоки с суммированием градиентов"),
        ("code_oracle", "Правильные назначения, новые значения"),
        ("code_oracle_transfer", "Правильные назначения, source-значения"),
        ("dense_sgd", "Плотный SGD"),
        ("dense_adam", "Плотный Adam"),
        ("dense_replay", "Перенос полных source-весов"),
        ("sparse_two", "Двухсвязная маска (контроль представимости)"),
    ]
    tables["codebook"] = table(
        ["Метод", "16 весов", "32 веса", "48 весов"],
        dimensional(codebook["rows"], code_methods),
    )
    tables["coverage"] = table(
        [
            "Метод",
            "Весов",
            "Верное I встретилось",
            "Верное I выбрано",
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
        ("code_vae", "Полновекторный VAE"),
        ("code_components", "Восстановленные компоненты"),
        ("code_known_blocks", "Известные блоки (привилегия)"),
        ("code_wrong_blocks", "Неверные границы"),
        ("code_bits", "Независимые source-биты"),
        ("code_replay", "Replay назначений"),
        ("code_uniform", "Равномерные назначения"),
        ("code_enumerate", "Перебор восстановленной поддержки"),
        ("code_cem", "CEM по компонентам"),
        ("code_cem_bits", "Побитовый CEM"),
        ("code_oracle", "Правильные назначения (oracle)"),
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
        ("spectral_components", "Спектральный / восстановленная поддержка"),
        ("spectral_uniform", "Спектральный / равномерные назначения"),
        ("moment_radial", "Первый момент + подбор частоты"),
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
            "Новая target-перестановка, 48 весов",
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
            "Средняя MSE этого контекста",
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
            "Компоненты, hard",
            "Компоненты + release",
            "VAE + release",
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
            "Компоненты, hard",
            "Компоненты + release",
            "VAE + release",
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
                ("early_128/e0", "128 source-шагов"),
                ("early_32/e0", "32 source-шага"),
                ("early_8/e0", "8 source-шагов"),
                ("noise_0.25/e0", "Шум σ=0,25"),
                ("noise_0.5/e0", "Шум σ=0,5"),
                ("noise_1/e0", "Шум σ=1"),
                ("noise_0.5/e0.3", "Шум σ=0,5; остаток 0,3"),
            ]
        ],
    )
    tanh_labels = [
        ("dense_adam", "Плотный Adam"),
        ("numerical_transfer", "Перенос численных параметров + Adam"),
        ("oracle", "Правильная группировка"),
        ("components_release", "Компоненты + release"),
        ("components", "Компоненты"),
        ("vae", "VAE"),
        ("dense_sgd", "Плотный SGD"),
        ("replay", "Replay source-группировки"),
        ("uniform", "Равномерная группировка"),
    ]
    tanh = robustness["twolayer"]["two_layer_tanh"]
    tables["twolayer"] = table(
        [
            "Метод",
            "Успех",
            "Средняя normalized MSE",
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
            "Плотный fixed-scale",
            "Плотный fan-in",
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
    text = REPORT.read_text()
    for name, content in tables.items():
        pattern = rf"(<!-- table:{name} -->).*?(<!-- /table:{name} -->)"
        text, replacements = re.subn(
            pattern,
            lambda match: f"{match[1]}\n\n{content}\n\n{match[2]}",
            text,
            flags=re.DOTALL,
        )
        if replacements != 1:
            raise ValueError(f"Expected one report table marker: {name}")
    REPORT.write_text(text)
    print(f"Updated {len(tables)} tables in {REPORT}")


if __name__ == "__main__":
    main()
