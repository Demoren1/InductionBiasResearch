"""Draw separate comparison charts for the main report from saved results."""

import matplotlib.pyplot as plt

from build_figures import RESULTS, read_summary, save_figure, success_panel


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
        }
    )
    masks = read_summary(RESULTS / "summary.json")
    codebook = read_summary(RESULTS / "codebook_summary.json")
    hidden = read_summary(RESULTS / "hidden_structure_summary.json")
    robustness = read_summary(RESULTS / "robustness_twolayer_summary.json")
    panels = [
        (
            "masks_comparison",
            "Выбор маски · 8 входов",
            {row["method"]: row for row in masks["matched_v2"]},
            [
                ("enumerate_roles", "Перебор 16 масок", "simple"),
                (
                    "uniform_roles",
                    "Случайная маска с учётом выходов",
                    "simple",
                ),
                ("vae", "VAE с раздельными частями", "vae"),
                ("product", "Выбор по частотам", "simple"),
                ("dense_adam", "Плотный Adam", "dense"),
                ("dense_sgd", "Плотный SGD", "dense"),
            ],
        ),
        (
            "codebook_comparison",
            "Два общих значения · 32 веса",
            {
                row["method"]: row
                for row in codebook["rows"]
                if row["dimension"] == 32
            },
            [
                ("code_vae", "VAE, новые значения", "vae"),
                ("code_blocks", "Исходные блоки, новые значения", "simple"),
                ("dense_adam", "Плотный Adam", "dense"),
                ("code_transfer", "Блоки, старт с исходных значений", "other"),
                ("code_replay", "Готовое исходное разбиение", "other"),
            ],
        ),
        (
            "hidden_comparison",
            "Неизвестные группы · 32 веса",
            {
                row["method"]: row
                for row in hidden["rows"]
                if row["dimension"] == 32
                and row["source_size"] == "all"
                and row["condition"] == "shared_permutation"
            },
            [
                ("code_components", "Восстановленные группы", "simple"),
                ("code_vae", "VAE", "vae"),
                ("dense_adam", "Плотный Adam", "dense"),
                ("code_replay", "Готовое исходное разбиение", "other"),
            ],
        ),
        (
            "tanh_comparison",
            "Двухслойная tanh-сеть",
            robustness["twolayer"]["two_layer_tanh"],
            [
                ("dense_adam", "Плотный Adam", "dense"),
                ("numerical_transfer", "Перенос параметров + Adam", "other"),
                (
                    "components_release",
                    "Группы, затем свободные веса",
                    "simple",
                ),
                ("components", "Восстановленные группы", "simple"),
                ("vae", "VAE", "vae"),
            ],
        ),
    ]
    for name, title, rows, methods in panels:
        fig, ax = plt.subplots(figsize=(10, 4.2), layout="constrained")
        success_panel(ax, title, rows, methods)
        save_figure(fig, name)
    print(f"Saved {len(panels)} comparison charts as PNG and SVG")


if __name__ == "__main__":
    main()
