"""Late structural report for G23 and its separately registered T/V controls."""

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

LABELS = {
    "prefix": "Functional prefix search",
    "lc": "LC signed adaptation",
    "dense_dp": "Dense GN + DP readout",
    "lutq": "LUT-Q signed adaptation",
    "yeh": "Categorical joint ablation",
    "yeh_bilevel_direct": "Yeh T/V, direct local H inverse",
    "yeh_bilevel_literal": "Yeh T/V, literal Neumann20",
}


def read_frozen_audit(root):
    common = root / "common"
    frozen = json.loads((common / "selection_frozen.json").read_text())
    for relative, expected in frozen["sha256"].items():
        if digest(common / relative) != expected:
            raise RuntimeError("A frozen selected model changed")
    audit = json.loads((common / "audit.json").read_text())
    if audit["selection_freeze_sha256"] != digest(common / "selection_frozen.json"):
        raise RuntimeError("Structural audit belongs to another selection freeze")
    return [row for row in audit["records"] if row["scope"] == "common4"]


def comparison_rows(primary, supplement, plan):
    rows = []
    for noise in plan["noise"]:
        for method in LABELS:
            cases = [
                r
                for r in primary + supplement
                if r["noise"] == noise and r["method"] == method
            ]
            if len(cases) != len(plan["families"]):
                raise RuntimeError(f"Incomplete common4 comparison: {method}/{noise}")
            rows.append(
                {
                    "noise": noise,
                    "method": method,
                    "cases": len(cases),
                    "exact_mask": sum(r["exact_mask"] for r in cases),
                    "exact_unsigned_I": sum(r["exact_unsigned_I"] for r in cases),
                    "exact_I": sum(r["exact_I"] for r in cases),
                    "bank_contains_true_I": sum(
                        r["bank_contains_true_I"] for r in cases
                    ),
                    "bank_contains_true_I_selection_wrong": sum(
                        r["bank_contains_true_I"] and not r["exact_I"] for r in cases
                    ),
                    "wrong_mask": sum(
                        r["structural_failure_stage"] == "mask" for r in cases
                    ),
                    "wrong_sharing_given_exact_mask": sum(
                        r["structural_failure_stage"] == "sharing" for r in cases
                    ),
                    "wrong_signs_given_exact_unsigned_I": sum(
                        r["structural_failure_stage"] == "signs" for r in cases
                    ),
                    "no_candidate": sum(not r["complete"] for r in cases),
                }
            )
    return rows


def plot_indices(root, report, plan, records):
    with np.load(root / "common/audit_only.npz") as truth:
        fig, axes = plt.subplots(6, 3, figsize=(10, 18), constrained_layout=True)
        for family, specification in enumerate(plan["families"]):
            expected = truth[f"family_{family}__truth_I"]
            path = root / "common/selected" / f"job_{2 * family + 1:03d}.npz"
            with np.load(path) as selected:
                values = [(expected, "True I")]
                for method in plan["methods"]:
                    record = next(
                        r
                        for r in records
                        if r["family_id"] == family
                        and r["method"] == method
                        and r["noise"] == 2
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
                                LABELS[method]
                                + f"\n{record['categorical_errors']} errors",
                            )
                        )
                    else:
                        values.append((None, LABELS[method] + "\nno candidate"))
            for column, (value, title) in enumerate(values):
                axis = axes[family, column]
                if value is not None:
                    image = axis.imshow(value, cmap="RdBu_r", vmin=-4, vmax=4)
                else:
                    axis.text(0.5, 0.5, "No full candidate", ha="center")
                axis.set_title(title, fontsize=9)
                axis.set_xticks([0, 7, 15])
                axis.set_yticks([0, 7, 15])
            axes[family, 0].set_ylabel(
                f"{specification['variant']} {specification['family']}"
            )
        fig.colorbar(
            image, ax=axes.ravel().tolist(), shrink=0.45, label="Signed code ID"
        )
        fig.suptitle(
            "G23 supplementary bi-level native MAP indexators, SD2\nNames/hidden columns matched only after whole-I selection freeze"
        )
        fig.savefig(report / "bilevel_indexators.png", dpi=150)
        plt.close(fig)


def plot_diagnostics(root, report, plan):
    fig, axes = plt.subplots(4, 2, figsize=(13, 15), constrained_layout=True)
    for family_id in range(len(plan["families"])):
        selection = json.loads(
            (root / "common/selected" / f"job_{2 * family_id + 1:03d}.json").read_text()
        )
        bank = json.loads(
            (root / "common/banks" / f"job_{2 * family_id + 1:03d}.json").read_text()
        )
        for column, method in enumerate(plan["methods"]):
            chosen = next(
                row for row in selection["records"] if row["method"] == method
            )
            if not chosen["complete"]:
                continue
            origin = next(
                row
                for row in bank["candidates"][chosen["selected_candidate"]]["origins"]
                if row["method"] == method
            )
            source = root / "extract" / f"job_{origin['source_job']:04d}"
            metadata = json.loads(source.with_suffix(".json").read_text())
            candidate = metadata["candidates"][origin["record"]]
            scenario = candidate["scenario"]
            family = plan["families"][family_id]["family"]
            with np.load(source.with_suffix(".npz")) as arrays:
                steps = arrays["trace_step"]
                for field, style in (
                    ("soft_H_mse", ":"),
                    ("raw_hard_H_mse", "--"),
                    ("H_mse", "-"),
                ):
                    axes[0, column].plot(
                        steps,
                        arrays[f"trace_{field}"][:, scenario].mean(-1),
                        style,
                        alpha=0.65,
                        label=f"{family} {field}",
                    )
                diagnostic_steps = arrays["diagnostic_outer_state"]
                if len(diagnostic_steps):
                    axes[1, column].plot(
                        diagnostic_steps,
                        arrays["diagnostic_C_stationarity_max_task"][:, scenario],
                        label=str(family),
                    )
                    axes[2, column].plot(
                        diagnostic_steps,
                        arrays["diagnostic_actual_inverse_residual_max_task"][
                            :, scenario
                        ],
                        label=str(family),
                    )
                    axes[3, column].plot(
                        diagnostic_steps,
                        arrays["diagnostic_indirect_gradient_norm"][:, scenario],
                        label=str(family),
                    )
            axes[0, column].set_title(LABELS[method])
    for line, label in enumerate(
        (
            "Observed H MSE: soft / raw MAP / polished MAP",
            "Inner C stationarity norm",
            "Relative inverse-action residual",
            "Indirect hypergradient norm",
        )
    ):
        for axis in axes[line]:
            axis.set_yscale("symlog", linthresh=1e-14)
            axis.set_ylabel(label)
            axis.set_xlabel("Actual outer updates / pre-update state")
            axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=6, ncols=2)
    axes[1, 0].legend(fontsize=7)
    fig.suptitle(
        "Actual selected native trajectories, SD2; finite inner Adam is not a global optimum\nThe architecture/stop/meta splits are separate"
    )
    fig.savefig(report / "bilevel_optimization_diagnostics.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    plan = json.loads((root / "protocol.json").read_text())
    parent = Path(plan["parent"])
    primary, supplementary = read_frozen_audit(parent), read_frozen_audit(root)
    rows = comparison_rows(primary, supplementary, plan)
    report = root / "report"
    report.mkdir(exist_ok=False)
    with (report / "comparison.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    source_records = [
        json.loads(path.read_text())
        for path in sorted((root / "extract").glob("job_*.json"))
    ]
    failures = [row for row in source_records if row["numerical_failure"]]
    save_json(
        report / "summary.json",
        {
            "comparisons": rows,
            "source_jobs": len(source_records),
            "source_wall_seconds_sum_over_workers": sum(
                row["seconds"] for row in source_records
            ),
            "source_jobs_with_numerical_failure": [
                {"job": r["job"], "failure": r["numerical_failure"]} for r in failures
            ],
            "selection_freeze_sha256": digest(root / "common/selection_frozen.json"),
            "overall_goal_complete": False,
        },
    )
    plot_indices(root, report, plan, supplementary)
    plot_diagnostics(root, report, plan)
    text = [
        "# G23: общая структура и отдельный bi-level контроль Yeh",
        "",
        "Точный signed I — основной результат; маска и unsigned sharing считаются независимо. Все строки common4 на тех же шести независимых I, с четырьмя отдельными кодбуками. Это дополнительный pilot на прежних геометриях, не независимая confirmation.",
        "",
        "|SD|Метод|Exact mask|Exact unsigned I|Exact signed I|Банк содержит true I|Выбор потерял true I|Нет I|",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        text.append(
            f"|{row['noise']}|{LABELS[row['method']]}|{row['exact_mask']}/{row['cases']}|{row['exact_unsigned_I']}/{row['cases']}|{row['exact_I']}/{row['cases']}|{row['bank_contains_true_I']}|{row['bank_contains_true_I_selection_wrong']}|{row['no_candidate']}|"
        )
    text.extend(
        [
            "",
            "G23 categorical joint — абляция, не полный Yeh. Supplement обучает один P на четырёх задачах: C/Adam на T, implicit P gradient на architecture V, hard-MAP checkpoint/HPO на отдельном H, fresh32signedC и meta-selection только после predictor freeze. Исходный single-source native bank также выбирается по четырём задачам; новый baseline дополнительно использует все четыре уже во время поиска. До16кандидатов/метод, одинаковый numeric/meta протокол; compute не уравнен.",
            "",
            "[Статья Yeh et al., AISTATS2022](https://proceedings.mlr.press/v151/yeh22b/yeh22b.pdf), [зафиксированный авторский код](https://github.com/raymondyeh07/equivariance_discovery/tree/164fe57aba0939e7db62daf72a899b2f25ce93c7). Signed dictionary, known bias/readout, squared loss и несколько source-задач — явные адаптации. Direct4×4 inverse — стандартный численный контроль, не новизна; literal20 сохраняет конвенцию опубликованного helper. Конечные250innerAdam/minibatch steps не гарантируют стационарный или глобальный C. Численные diagnostics опубликованы, неудачные checkpoints/запуски не скрыты.",
            "",
            f"Source jobs: {len(source_records)}, numerical failure в {len(failures)}; суммарное wall-time workers {sum(r['seconds'] for r in source_records):.1f}s, не elapsed end-to-end и не matched FLOPs. Границы: known b/readout, Gaussian noise/means, высокий inherited physical-label budget порядка275.2bn/sourcecondition; новых labels0. Перенос C не выполнялся; отдельно требуется fixed-I fresh-C transfer и независимая проверка устойчивости/новизны. Общая цель остаётся активна.",
            "",
            "![Истинные и native MAP индексаторы](bilevel_indexators.png)",
            "![Реальные ошибки и гиперградиенты](bilevel_optimization_diagnostics.png)",
            "",
            "Протокол, snapshot, все actual checkpoints и scalar diagnostics сохранены. Source1 в этом дополнении не запускался и не считается отказом. Метрики не объединяются с более ранними G21/G22 как будто это один benchmark.",
        ]
    )
    (report / "REPORT.md").write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    main()
