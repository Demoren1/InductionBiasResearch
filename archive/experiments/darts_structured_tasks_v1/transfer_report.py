"""Plot recorded train/validation curves and heldout structural-transfer errors."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import digest, save_json
from .report import save_figure

LABELS = {
    "consensus_baseline/learned": "Frozen consensus baseline",
    "consensus_reset/learned": "Consensus + reset",
    "consensus_adapt_keep/learned": "Consensus + C-fit, keep Z",
    "consensus_adapt_reset/learned": "Consensus + C-fit + reset",
    "plain_adapt_keep/learned": "Plain C-fit, keep Z",
    "plain_adapt_reset/learned": "Plain C-fit + reset",
    "nearest/learned": "Nearest + centering",
    "task_centers/learned": "Nearest, task-only C",
    "proximal/learned": "Charged projection",
    "proximal_warm/learned": "Projection + dense warmup",
    "quadratic/learned": "Quadratic consensus",
    "damped_admm/learned": "ADMM, dual0.25",
    "admm/learned": "ADMM, dual1",
    "oracle": "True indexator",
    "dense_adam": "Dense Adam",
}


def report(output, cohort="transfer"):
    root = output / cohort
    summary = json.loads((root / "summary.json").read_text())
    methods = [name for name in summary["aggregate"] if name.endswith("/learned")]
    methods.extend(("dense_adam", "oracle"))
    curves = {name: {"train": [], "query": []} for name in methods}
    seconds, reused_seconds = 0.0, 0.0
    for task in summary["tasks"]:
        folder = root / f"task_{task['task']}"
        spec = json.loads((folder / "specification.json").read_text())
        fits = {}
        for kind in ("fixed", "dense"):
            with np.load(folder / f"{kind}_fits.npz") as archive:
                fits[kind] = dict(archive)
            if kind == "dense" and (folder / "dense_control_reuse.json").exists():
                reused_seconds += float(fits[kind]["seconds"])
            else:
                seconds += float(fits[kind]["seconds"])
        for name in methods:
            entry = task["methods"][name]
            data = fits[entry["fit"]]
            row = entry["row"]
            for split in ("train", "query"):
                curves[name][split].append(
                    data[f"history_{split}"][:, row].mean(-1) / spec["target_variance"]
                )
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.7), constrained_layout=True)
    steps = fits["fixed"]["history_steps"]
    for name in methods:
        for axis, split in zip(axes, ("train", "query")):
            axis.plot(steps, np.mean(curves[name][split], axis=0), label=LABELS[name])
    for axis, label in zip(axes, ("Train", "Query")):
        axis.set_yscale("log")
        axis.set_title(f"{label} NMSE: mean over6 tasks and8 starts")
        axis.set_xlabel("Numerical update")
        axis.grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    save_figure(figure, root / "figures/numerical_curves")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for name in methods:
        values = summary["aggregate"][name]["per_task_selected_test_nmse"]
        axes[0].plot(range(1, len(values) + 1), values, marker="o", label=LABELS[name])
    axes[0].set_xlabel("New coefficient function")
    axes[0].set_ylabel("Test NMSE")
    axes[0].set_yscale("log")
    axes[0].set_title("Frozen I; query-selected numerical restart")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    positions = np.arange(len(methods))
    for offset, field, label in (
        (-0.18, "mean_query_selected_test_nmse", "Query-selected C"),
        (0.18, "mean_all_restarts_test_nmse", "All8 fresh C"),
    ):
        axes[1].bar(
            positions + offset,
            [summary["aggregate"][name][field] for name in methods],
            width=0.36,
            label=label,
        )
    axes[1].set_xticks(
        positions, [LABELS[name] for name in methods], rotation=35, ha="right"
    )
    axes[1].set_yscale("log")
    axes[1].set_title("Mean test NMSE over6 new functions")
    axes[1].legend(fontsize=8)
    save_figure(figure, root / "figures/transfer_errors")
    marker = "## Кривые и границы вывода"
    original = (root / "REPORT.md").read_text().split(marker)[0].rstrip()
    appendix = [
        "",
        marker,
        "",
        "![Ошибки переноса](figures/transfer_errors.png)",
        "",
        "![Train/query кривые](figures/numerical_curves.png)",
        "",
        "Кривые усредняют все8 численных стартов на всех6 новых функциях при выбранной по selection численной настройке каждого метода. Train измерен на первых512 support-примерах, query — на512; ось X — реальные обновления. Они не показывают test во время обучения. В таблице выше также сохранены перестановочные контроли шэринга и подключений.",
        "",
        "Малая source-ошибка и успешные свежие C на прежней функции недостаточны для вывода о переносимой структуре. Здесь переносится только source-selected I; значения C, bias/readout/offset не переносятся. Истинный I — отдельная диагностика численной обучаемости. Шесть функций одной скрытой структуры не являются переносом между датасетами или новыми архитектурами.",
        "",
        f"Новое численное обучение переноса: {seconds / 60:.2f} мин. Повторно использованы идентичные dense-контроли стоимостью {reused_seconds / 60:.2f} мин, если указаны dense_control_reuse.json. Совпадение наблюдений, протокола, настроек и бюджета проверено перед загрузкой. Source-поиск сюда не включён.",
        "",
    ]
    (root / "REPORT.md").write_text(original + "\n" + "\n".join(appendix))
    (root / "reporting_code_snapshot.py").write_bytes(Path(__file__).read_bytes())
    save_json(
        root / "complete.json",
        {
            "sha256": {
                str(path.relative_to(root)): digest(path)
                for path in root.rglob("*")
                if path.is_file() and path.name != "complete.json"
            }
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--cohort", choices=("transfer", "transfer_confirmation"), default="transfer"
    )
    args = parser.parse_args()
    report(args.root, args.cohort)


if __name__ == "__main__":
    main()
