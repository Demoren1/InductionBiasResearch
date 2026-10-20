"""Summarize known-mask transfer using families as the independent units."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..io import DEFAULT_BANK, checked_load, save_json, sha256
from ..masks import structural_metrics
from ..problem import reference

METHODS = [
    "correct_mask",
    "dense",
    "imp",
    "training_bank",
    "gumbel",
    "evolution",
    "best_initial",
    "random_budget",
    "zero_z",
    "source_frequency",
]
LABELS = [
    "Правильная маска",
    "MLP без маски",
    "IMP",
    "Выбор из банка",
    "W/G Gumbel",
    "W/G эволюция",
    "Лучший исходный z",
    "Случайные z",
    "z = 0",
    "Частотная маска",
]


def report(root, bank=DEFAULT_BANK):
    directories = sorted(
        path.parent for path in root.glob("wg_*/family_*/COMPLETE.json")
    )
    rows, diagnostics, complete = [], [], []
    for directory in directories:
        seal = json.loads((directory / "selection_seal.json").read_text())
        if seal["sha256"] != sha256(directory / "selection.pt"):
            raise ValueError("A sealed selection changed")
        rows.extend(json.loads((directory / "results.json").read_text()))
        diagnostics.append(
            {
                "run": str(directory.relative_to(root)),
                **json.loads((directory / "diagnostics.json").read_text()),
            }
        )
        complete.append(json.loads((directory / "COMPLETE.json").read_text()))
    if not rows:
        raise ValueError("No completed results")
    with (root / "results.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    families = sorted({row["family"] for row in rows})
    metrics = [
        "test_excess_bce",
        "test_bce",
        "iou",
        "variable_iou",
        "core_recall",
        "variable_recall",
        "exact_mask",
        "noise_edges",
    ]
    summary = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        family_values = {
            metric: np.array(
                [
                    np.mean(
                        [
                            float(row[metric])
                            for row in selected
                            if row["family"] == family
                        ]
                    )
                    for family in families
                ]
            )
            for metric in metrics
        }
        summary[method] = {
            metric: {
                "mean": float(values.mean()),
                "family_sem": float(values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else 0,
                "family_values": values.tolist(),
            }
            for metric, values in family_values.items()
        }
    payload = {
        "jobs": len(directories),
        "rows": len(rows),
        "families": families,
        "summary": summary,
        "diagnostics": diagnostics,
        "peak_reserved_gib": max(
            item["cuda_peak_reserved_gib"] for item in complete
        ),
        "total_worker_seconds": sum(
            item["elapsed_seconds"] for item in complete
        ),
        "scope": "Family means average decoder seeds, two new functions, and three MLP initializations. Cached IMP and dense/oracle controls are not independent across decoder seeds.",
    }
    save_json(root / "summary.json", payload)
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ["test_excess_bce", "variable_iou"],
        [
            "Ошибка на новых функциях (меньше — лучше)",
            "IoU переменных связей (больше — лучше)",
        ],
    ):
        means = [summary[method][metric]["mean"] for method in METHODS]
        errors = [summary[method][metric]["family_sem"] for method in METHODS]
        colors = [
            "#7b9e87"
            if method in ["correct_mask", "dense", "imp"]
            else "#5379ac"
            for method in METHODS
        ]
        axis.barh(LABELS, means, xerr=errors, color=colors, alpha=0.9)
        axis.invert_yaxis()
        axis.set_title(title)
        axis.set_xlabel(
            "Test BCE − энтропия учителя"
            if metric == "test_excess_bce"
            else "IoU; для плотной MLP сравнение структуры условно"
        )
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle(
        f"Знакомые маски, новые функции: {len(families)} семейств, 3 W/G-декодера\nПогрешности: SEM по семействам; выбор маски на функциях 4/5, оценка на 6/7"
    )
    fig.savefig(figures / "quality.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for method, color in [("gumbel", "#2166ac"), ("evolution", "#cf6a28")]:
        histories = [
            json.loads((directory / f"{method}_history.json").read_text())
            for directory in directories
        ]
        # The history stores each start's hard-query score at every search step.
        key = next(
            (
                key
                for key in ["best_query_bce", "best_bce", "best"]
                if key in histories[0][0]
            ),
            None,
        )
        if key:
            traces = np.array(
                [[row[key] for row in history] for history in histories]
            )
            axes[0].plot(
                np.arange(len(traces[0])),
                traces.mean(0),
                label=method,
                color=color,
            )
        x = np.array([summary[method]["iou"]["family_values"]]).flatten()
        y = np.array(summary[method]["test_excess_bce"]["family_values"])
        axes[1].scatter(x, y, label=method, color=color, alpha=0.8)
    axes[0].set(
        title="Поиск по адаптационным функциям",
        xlabel="Шаг поиска",
        ylabel="Query BCE",
    )
    if axes[0].lines:
        axes[0].legend()
    axes[1].set(
        title="Каждая точка — семейство",
        xlabel="IoU полной маски",
        ylabel="Test excess BCE",
    )
    axes[1].legend()
    fig.savefig(figures / "search.png", dpi=170)
    plt.close(fig)

    directory = directories[0]
    protocol = json.loads((directory / "protocol.json").read_text())
    manifest = json.loads((bank / "manifest.json").read_text())
    target = reference(
        manifest["family_definitions"][protocol["family"]], "cpu"
    )
    selection = checked_load(directory / "selection.pt")
    imp = checked_load(
        root / "imp_controls" / f"family_{protocol['family']:02d}.pt"
    )
    examples = {
        "Аналитическая маска": target,
        "W/G Gumbel": selection["masks"]["gumbel"],
        "W/G эволюция": selection["masks"]["evolution"],
        "Выбор из банка": selection["masks"]["training_bank"],
        f"Средний IMP ({imp['masks'].shape[0] * imp['masks'].shape[1]} сетей)": imp[
            "masks"
        ].flatten(0, 1),
    }
    fig, axes = plt.subplots(
        1, len(examples), figsize=(13, 6), constrained_layout=True
    )
    for axis, (label, mask) in zip(axes, examples.items()):
        batch = mask if mask.ndim == 3 else mask[None]
        values, orders = structural_metrics(batch, target)
        aligned = (
            torch.stack([item[:, order] for item, order in zip(batch, orders)])
            .float()
            .mean(0)
        )
        axis.imshow(
            aligned.numpy(),
            cmap="Blues",
            vmin=0,
            vmax=1,
            aspect="auto",
            interpolation="nearest",
        )
        axis.axhline(15.5, color="#d29b32", linewidth=1)
        axis.axhline(31.5, color="#cc6262", linewidth=1)
        axis.set_title(
            label + f"\nIoU={np.mean([value['iou'] for value in values]):.3f}"
        )
        axis.set_xlabel("Скрытые нейроны")
        axis.set_xticks(range(8))
    axes[0].set_ylabel("Входы: 0–15 общие; 16–31 переменные; 32–47 шум")
    fig.suptitle(
        f"Семейство {protocol['family']}, декодер {protocol['decoder_seed']}\nСтолбцы выровнены с аналитической маской только для изображения"
    )
    fig.savefig(figures / "masks.png", dpi=170)
    plt.close(fig)
    print(
        json.dumps(
            {
                "report": str(root / "summary.json"),
                "jobs": len(directories),
                "rows": len(rows),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    args = parser.parse_args()
    report(args.root, args.bank)
