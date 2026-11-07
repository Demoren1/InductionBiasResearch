"""Late G24 ablation report; all values come from frozen actual searches."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.darts_structured_tasks_v1.evaluate import align_indices

from .shared_structure_bilevel_report import read_frozen_audit

LABELS = {
    "bootstrap": "Frozen H-selected bootstrap",
    "axis_single_refit": "Old axis C / single / unit+refit",
    "bootstrap_single_fixed": "Aligned C / single / fixed",
    "bootstrap_single_refit": "Aligned C / single / refit",
    "bootstrap_joint_fixed": "Aligned C / joint4 / fixed",
    "bootstrap_joint_refit": "Aligned C / joint4 / refit",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    plan = json.loads((root / "protocol.json").read_text())
    records = read_frozen_audit(root)
    report = root / "report"
    report.mkdir(exist_ok=False)
    rows = []
    for sigma in plan["noise"]:
        for method in LABELS:
            cases = [
                row
                for row in records
                if row["noise"] == sigma and row["method"] == method
            ]
            if len(cases) != len(plan["families"]):
                raise RuntimeError("Incomplete G24 comparison")
            rows.append(
                {
                    "noise": sigma,
                    "method": method,
                    "cases": len(cases),
                    "exact_I": sum(row["exact_I"] for row in cases),
                    "exact_mask": sum(row["exact_mask"] for row in cases),
                    "exact_unsigned_I": sum(row["exact_unsigned_I"] for row in cases),
                    "bank_contains_true_I": sum(
                        row["bank_contains_true_I"] for row in cases
                    ),
                    "selection_failed_with_true_I": sum(
                        row["bank_contains_true_I"] and not row["exact_I"]
                        for row in cases
                    ),
                    "no_candidate": sum(not row["complete"] for row in cases),
                }
            )
    with (report / "comparison.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with np.load(root / "common/audit_only.npz") as truth:
        fig, axes = plt.subplots(6, 7, figsize=(20, 17), constrained_layout=True)
        for family, specification in enumerate(plan["families"]):
            expected = truth[f"family_{family}__truth_I"]
            values = [(expected, "True I")]
            with np.load(
                root / "common/selected" / f"job_{2 * family + 1:03d}.npz"
            ) as selected:
                for method, label in LABELS.items():
                    record = next(
                        row
                        for row in records
                        if row["family_id"] == family
                        and row["noise"] == 2
                        and row["method"] == method
                    )
                    if record["complete"]:
                        _, aligned, _ = align_indices(
                            selected[f"{method}__common4__index"],
                            expected,
                            plan["codes"],
                        )
                        values.append(
                            (
                                aligned,
                                label + f"\n{record['categorical_errors']} errors",
                            )
                        )
                    else:
                        values.append((None, label + "\nNo complete candidate"))
            for axis, (value, title) in zip(axes[family], values, strict=True):
                if value is not None:
                    rendered = axis.imshow(value, cmap="RdBu_r", vmin=-4, vmax=4)
                axis.set_title(title, fontsize=8)
                axis.set_xticks((0, 15))
                axis.set_yticks((0, 15))
            axes[family, 0].set_ylabel(
                f"{specification['variant']} {specification['family']}"
            )
        fig.colorbar(
            rendered, ax=axes.ravel().tolist(), shrink=0.4, label="Signed index"
        )
        fig.suptitle(
            "G24 opened diagnosis, fresh SD2 prefix noise\nGroup/hidden matching only after whole-I freeze; bootstrap mask never fixed during search"
        )
        fig.savefig(report / "joint_indexators.png", dpi=150)
        plt.close(fig)
    performance = json.loads((root / "execution_measurement.json").read_text())
    searches = [
        json.loads(path.read_text())
        for path in sorted((root / "extract").glob("job_*.json"))
    ]
    save_json(
        report / "summary.json",
        {
            "comparisons": rows,
            "source_jobs": len(searches),
            "complete_searches": sum(row["complete"] for row in searches),
            "worker_seconds_sum": sum(row["seconds"] for row in searches),
            "execution_measurement": performance,
            "source_protocol_sha256": digest(root / "protocol.json"),
            "selection_freeze_sha256": digest(root / "common/selection_frozen.json"),
            "opened_diagnostic": True,
            "independent_confirmation_or_competitor_win": False,
            "overall_goal_complete": False,
        },
    )
    text = [
        "# G24: общий поиск prefix-counts при разных кодбуках",
        "",
        (
            "Открытая диагностика прежних шести G23-геометрий/четырёх source-C. "
            "Bootstrap выбран по старому numeric H, без старого meta/teacher, и "
            "зафиксирован до свежего шума. Все новые I строятся с пустого prefix; "
            "ни маска, ни отдельные назначения bootstrap не фиксируются. Четыре "
            "single-ветки объединяются в банк, joint имеет один общий поиск; "
            "число траекторий и compute не уравнены. Это не независимая confirmation."
        ),
        "",
        "|SD|Метод|Exact mask|Exact unsigned I|Exact signed I|Истина в банке|Потеря при выборе|Нет кандидата|",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        text.append(
            f"|{row['noise']}|{LABELS[row['method']]}|{row['exact_mask']}/{row['cases']}|{row['exact_unsigned_I']}/{row['cases']}|{row['exact_I']}/{row['cases']}|{row['bank_contains_true_I']}|{row['selection_failed_with_true_I']}|{row['no_candidate']}|"
        )
    text.extend(
        [
            "",
            (
                "Всего 288 новых prefix-поисков и 12 frozen-bootstrap контролей. "
                "Старые source train данные остаются численной целью; новые независимые "
                "prefix train/validation используются поиском, новые Gaussian H и meta "
                "— checkpoint/выбором. На новые наблюдения приходится 184549376 "
                "physical labels/source-function/noise condition; цена прежнего "
                "bootstrap не обнуляется. Все уникальные I проходят одинаковый "
                "fresh32signed C/GN24 fit и whole predictor freeze до meta-Y; "
                "whole selection freeze до teacher audit. Targettest G23 не использован."
            ),
            "",
            "![Истинные и выбранные индексаторы](joint_indexators.png)",
            "",
            (
                "Теория§34 доказывает равенство kernel joint-Gram пересечению kernels "
                "задач и условную устойчивость при точном global solver. Она не "
                "гарантирует правильных parents, словаря, локальной integer-сходимости "
                "или успеха всех I. Известность сумм task-loss/alternating fit/finite "
                "alphabet factorization раскрыта; приоритет нового полного алгоритма "
                "и общее превосходство ещё не установлены. Требуются независимые I, "
                "робастность и подходящие native/query контроли после заморозки схемы."
            ),
        ]
    )
    (report / "REPORT.md").write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    main()
