"""Paired comparison of known-mask transfer before and after tuning fitness."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..io import PACKAGE as HERE
from ..io import save_json


def compare(root, old, tuning):
    summaries = {
        "old": json.loads((old / "summary.json").read_text()),
        "new": json.loads((root / "summary.json").read_text()),
    }
    calibration = json.loads((tuning / "summary.json").read_text())
    calibration_families = set(calibration["calibration_families"])
    values = {}
    for tag, summary in summaries.items():
        groups = {
            "all": summary["families"],
            "calibration": sorted(calibration_families),
            "other": sorted(set(summary["families"]) - calibration_families),
        }
        values[tag] = {}
        for group, families in groups.items():
            indices = [
                summary["families"].index(family) for family in families
            ]
            values[tag][group] = {
                method: {
                    metric: float(
                        np.mean(np.array(record["family_values"])[indices])
                    )
                    for metric, record in metrics.items()
                }
                for method, metrics in summary["summary"].items()
            }
    rows = [
        row
        for path in root.glob("wg_*/family_*/results.json")
        for row in json.loads(path.read_text())
    ]
    protocols = [
        json.loads(path.read_text())
        for path in root.glob("wg_*/family_*/protocol.json")
    ]
    execution = {}
    for protocol in protocols:
        code_hash = protocol["source_hashes"].get(
            "search.py", protocol["source_hashes"].get("calibrated_fitness.py")
        )
        execution[code_hash] = execution.get(code_hash, 0) + 1
    payload = {
        "groups": {
            "calibration": sorted(calibration_families),
            "other": sorted(
                set(summaries["new"]["families"]) - calibration_families
            ),
        },
        "selected": calibration["selected"],
        "values": values,
        "execution_source_hash_counts": execution,
        "pairing": "Identical teacher observations, decoder checkpoints, latent random seeds, and final MLP seeds. Same 1608 candidate fits and same 64 inner steps; optimizer/lr/warmup differ.",
    }
    save_json(root / "comparison.json", payload)
    methods = ["gumbel", "evolution", "random_budget", "training_bank"]
    labels = ["Gumbel", "Эволюция", "Случайный поиск", "Выбор из банка"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    x = np.arange(len(methods))
    for offset, tag, label, color in [
        (-0.18, "old", "До: SGD 64, lr=0,2", "#91a2b8"),
        (0.18, "new", "После: Adam 64, lr=0,1", "#318674"),
    ]:
        for axis, metric in zip(axes, ["test_excess_bce", "exact_mask"]):
            data = [
                summaries[tag]["summary"][method][metric] for method in methods
            ]
            axis.bar(
                x + offset,
                [item["mean"] for item in data],
                0.36,
                yerr=[item["family_sem"] for item in data],
                color=color,
                label=label,
            )
    for method, label, color in [
        ("dense", "MLP без маски", "#bb5555"),
        ("imp", "Обычный IMP", "#555555"),
    ]:
        axes[0].axhline(
            summaries["new"]["summary"][method]["test_excess_bce"]["mean"],
            linestyle="--",
            color=color,
            linewidth=1.2,
            label=label,
        )
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=8)
    axes[0].set(
        title="Ошибка на новых функциях",
        ylabel="Test excess BCE: меньше — лучше",
    )
    axes[1].set(
        title="Полностью правильная структура",
        ylabel="Доля точных масок",
        ylim=(0, 1.08),
    )
    fig.suptitle(
        "Эффект подбора fitness: 24 семейства × 3 готовых W/G-декодера\nЧисло шагов и бюджет поиска сохранены; изменены оптимизатор, LR и warmup. Погрешности — SEM по семействам"
    )
    fig.savefig(root / "figures/fitness_comparison.png", dpi=170)
    plt.close(fig)
    print(
        json.dumps(
            {
                "selected": calibration["selected"],
                "all_new": values["new"]["all"],
                "other_new": values["new"]["other"],
                "rows": len(rows),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=HERE / "runs/known_mask_v2"
    )
    parser.add_argument(
        "--old", type=Path, default=HERE / "runs/known_mask_v1"
    )
    parser.add_argument(
        "--tuning", type=Path, default=HERE / "runs/fitness_tuning_v1"
    )
    args = parser.parse_args()
    compare(args.root, args.old, args.tuning)
