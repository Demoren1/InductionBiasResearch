"""Composition examples and pilot results from sealed CUDA artifacts."""

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from ..io import checked_load, save_json
from ..masks import structural_metrics
from ..problem import reference

COLORS = ["#ffffff", "#3973ac", "#d28a32", "#8d9ca8"]
LABELS = {
    "correct_mask": "Аналитическая маска",
    "dense": "MLP без маски",
    "imp": "IMP на новой функции",
    "gumbel": "W/G Gumbel",
    "evolution": "W/G evolution",
    "random_budget": "W/G случайные z",
    "recombined_bank": "Сборка двух IMP-масок",
    "training_bank": "Выбор одной IMP-маски",
    "best_initial": "Лучший исходный z",
}


def panel(axis, values, title, colored=False):
    axis.imshow(
        values,
        cmap=ListedColormap(COLORS) if colored else "Blues",
        vmin=0,
        vmax=3 if colored else 1,
        aspect="auto",
        interpolation="nearest",
    )
    axis.axhline(15.5, color="#a4adb5", linewidth=0.8)
    axis.axhline(31.5, color="#c97575", linewidth=0.8)
    axis.axvline(3.5, color="#444444", linewidth=0.8)
    axis.set(xticks=[0, 3, 4, 7], yticks=[0, 16, 24, 32, 47])
    axis.set_title(title, fontsize=10)
    axis.set_xlabel("Скрытые нейроны")


def design(root, output):
    payload = json.loads((root / "design.json").read_text())
    definitions = payload["family_definitions"]
    structure = payload["family_structure"]
    output.mkdir(parents=True, exist_ok=True)
    grid = np.zeros((6, 6))
    for family in definitions:
        grid[family["a"], family["b"]] = {
            "train": 0,
            "validation": 1,
            "test": 2,
        }[family["split"]]
    fig, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    axis.imshow(
        grid,
        cmap=ListedColormap(["#79a5cc", "#e4a24c", "#c6ced4"]),
        vmin=0,
        vmax=2,
    )
    for a in range(6):
        for b in range(6):
            axis.text(
                b,
                a,
                ["train", "пилот", "test"][int(grid[a, b])],
                ha="center",
                va="center",
            )
    axis.set(
        xticks=range(6),
        yticks=range(6),
        xlabel="Вариант блока B",
        ylabel="Вариант блока A",
    )
    axis.set_title(
        "36 сочетаний известных блоков\n24 train; 6 новых сочетаний в пилоте; 6 test не используются"
    )
    fig.savefig(output / "composition_split.png", dpi=170)
    plt.close(fig)

    examples = structure["decompositions"][:3]
    fig, axes = plt.subplots(3, 5, figsize=(14, 14), constrained_layout=True)
    for row, example in enumerate(examples):
        target_family = definitions[example["family"]]
        parent_a = definitions[example["parent_a"]]
        parent_b = definitions[example["parent_b"]]
        left = reference(parent_a, "cpu").numpy()
        right = reference(parent_b, "cpu").numpy()
        target = reference(target_family, "cpu").numpy()
        union = np.where((left > 0) & (right > 0), 3, left + 2 * right)
        selected = np.zeros_like(left)
        selected[:, :4] = left[:, :4]
        selected[:, 4:] = 2 * right[:, 4:]
        panels = [
            (
                left,
                f"Train: A{parent_a['a']} + B{parent_a['b']}\n32 связи",
                False,
            ),
            (
                2 * right,
                f"Train: A{parent_b['a']} + B{parent_b['b']}\n32 связи",
                True,
            ),
            (
                union,
                f"Объединение родителей\n{example['union_edges']} уникальных связей",
                True,
            ),
            (
                selected,
                "Блок A слева + блок B справа\n32 связи из объединения",
                True,
            ),
            (
                target,
                f"Новая маска A{target_family['a']} + B{target_family['b']}\nВ train полного совпадения нет",
                False,
            ),
        ]
        for column, (values, title, colored) in enumerate(panels):
            panel(axes[row, column], values, title, colored)
        axes[row, 0].set_ylabel("Входы: общие / переменные / шум")
    fig.legend(
        handles=[
            Patch(color=COLORS[1], label="Первый родитель"),
            Patch(color=COLORS[2], label="Второй родитель"),
            Patch(color=COLORS[3], label="Связь есть у обоих"),
        ],
        loc="outside lower center",
        ncol=3,
    )
    fig.suptitle(
        "Новые маски целиком собираются из частей двух обучающих\nРодители показаны для объяснения конструкции; поиск их идентификаторы не получает"
    )
    fig.savefig(output / "composition_examples.png", dpi=170)
    plt.close(fig)
    save_json(output / "design.json", payload)


def aligned(mask, target):
    values, orders = structural_metrics(mask.float()[None], target)
    return mask[:, orders[0]].float(), values[0]["iou"]


def results(root, output):
    evaluation = root / "evaluation"
    summary = json.loads((evaluation / "summary.json").read_text())
    payload = json.loads((root / "design.json").read_text())
    definitions = payload["family_definitions"]
    directories = sorted(evaluation.glob("wg_*/family_*/COMPLETE.json"))
    all_rows = [
        row
        for marker in directories
        for row in json.loads((marker.parent / "results.json").read_text())
    ]
    seeds = sorted({row["decoder_seed"] for row in all_rows})
    methods = [name for name in LABELS if name in summary["summary"]]
    table = []
    for method in methods:
        rows = [row for row in all_rows if row["method"] == method]
        unique = {}
        for row in rows:
            if method == "imp":
                key = (row["family"], row["function"], row["replica"])
            elif method in ("correct_mask", "dense"):
                key = (row["family"],)
            else:
                key = (row["family"], row["decoder_seed"])
            unique[key] = row
        record = summary["summary"][method]
        table.append(
            {
                "method": method,
                "label": LABELS[method],
                "test_excess_bce": record["test_excess_bce"]["mean"],
                "test_excess_bce_sem": record["test_excess_bce"]["family_sem"],
                "iou": record["iou"]["mean"],
                "variable_iou": record["variable_iou"]["mean"],
                "exact": sum(row["exact_mask"] for row in unique.values()),
                "unique_masks": len(unique),
            }
        )
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ("test_excess_bce", "iou"),
        ("Ошибка на новых функциях ↓", "Совпадение с новой маской ↑"),
    ):
        values = [
            np.nan
            if metric == "iou" and row["method"] == "dense"
            else row[metric]
            for row in table
        ]
        errors = [
            summary["summary"][row["method"]][metric]["family_sem"]
            for row in table
        ]
        axis.barh(
            [row["label"] for row in table],
            values,
            xerr=errors,
            color=[
                "#679a86"
                if row["method"] in ("correct_mask", "dense", "imp")
                else "#547dae"
                for row in table
            ],
        )
        axis.invert_yaxis()
        axis.set_title(title)
        axis.set_xlabel(
            "Test BCE − энтропия учителя"
            if metric == "test_excess_bce"
            else "IoU после сопоставления колонок"
        )
        axis.grid(axis="x", alpha=0.2)
        if metric == "iou":
            axis.set_xlim(0, 1.05)
    fig.suptitle(
        f"Композиционный пилот: 6 новых сочетаний × {len(seeds)} W/G-декодера\nВыбор маски на функциях 4/5; оценка на новых функциях 6/7, три инициализации MLP\nУсы: SEM по шести новым сочетаниям"
    )
    fig.savefig(output / "composition_quality.png", dpi=170)
    plt.close(fig)

    chosen = [24, 26]
    seed = seeds[0]
    fig, axes = plt.subplots(2, 7, figsize=(17, 11), constrained_layout=True)
    for row, family_id in enumerate(chosen):
        family = definitions[family_id]
        target = reference(family, "cpu")
        directory = evaluation / f"wg_{seed}" / f"family_{family_id:02d}"
        selection = checked_load(directory / "selection.pt")
        imp = checked_load(
            evaluation / "imp_controls" / f"family_{family_id:02d}.pt"
        )
        imp_batch = imp["masks"].flatten(0, 1)
        imp_values, imp_orders = structural_metrics(imp_batch, target)
        mean_imp = torch.stack(
            [mask[:, order] for mask, order in zip(imp_batch, imp_orders)]
        ).mean(0)
        examples = [
            ("Аналитическая", target, 1.0),
            ("MLP без маски\n384 связи", torch.ones_like(target), None),
        ]
        for method in (
            "gumbel",
            "evolution",
            "random_budget",
            "recombined_bank",
        ):
            mask, iou = aligned(selection["masks"][method], target)
            examples.append((LABELS[method], mask, iou))
        examples.append(
            (
                "Средний IMP\n6 независимых сетей",
                mean_imp,
                np.mean([value["iou"] for value in imp_values]),
            )
        )
        for column, (label, values, iou) in enumerate(examples):
            suffix = "" if iou is None else f"\nIoU={iou:.3f}"
            panel(axes[row, column], values, label + suffix)
        axes[row, 0].set_ylabel(f"A{family['a']} + B{family['b']}\nВходы")
    fig.suptitle(
        f"Что найдено на новых сочетаниях: декодер {seed}\nСредний IMP — частота связей после сопоставления с эталоном; эталон используется только для рисунка"
    )
    fig.savefig(output / "composition_methods.png", dpi=170)
    plt.close(fig)
    for name in ("summary.json", "results.csv"):
        shutil.copyfile(
            evaluation / name,
            output
            / ("transfer_summary.json" if name.endswith("json") else name),
        )
    shutil.copyfile(root / "bank_quality.json", output / "bank_quality.json")
    shutil.copyfile(
        root / "source_reconstruction.json",
        output / "source_reconstruction.json",
    )
    shutil.copyfile(root / "audit.json", output / "audit.json")
    diagnostics = [
        json.loads((marker.parent / "diagnostics.json").read_text())
        for marker in directories
    ]
    result = {
        "families": summary["families"],
        "decoder_seeds": seeds,
        "methods": table,
        "mean_random_pool_max_iou": float(
            np.mean([row["random_pool_max_iou"] for row in diagnostics])
        ),
        "random_pools_with_exact_masks": sum(
            row["random_pool_exact_candidates"] > 0 for row in diagnostics
        ),
        "manifest_sha256": json.loads(
            (root / "vae" / f"wg_{seeds[0]}" / "protocol.json").read_text()
        )["bank_sha256"],
        "scope": "Validation pilot, six held-out combinations; six test combinations remain unused. IMP repeats across decoder seeds are counted once.",
    }
    save_json(output / "pilot_summary.json", result)
    with (output / "methods.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("design", "results"), default="results"
    )
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    design(args.root, args.output)
    if args.phase == "results":
        results(args.root, args.output)


if __name__ == "__main__":
    main()
