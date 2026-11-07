"""Frozen G16 report: separate pseudo-target coverage from nonlinear bias."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.darts_structured_tasks_v1.evaluate import align_indices

from .independent_score import METHODS
from .nonlinear_moments import project

LABELS = {
    "scalar_dp": "Scalar DP",
    "zero_student_dp": "Zero-Student + DP",
    "signed_student_graph": "Full Student graph",
    "global_interval": "Global interval solver",
}


def generate(root):
    report = root / "report"
    report.mkdir(exist_ok=False)
    plan = json.loads((root / "protocol.json").read_text())
    frozen = json.loads((root / "models_frozen.json").read_text())
    records, base, meta = [], [], []
    for i, job in enumerate(plan["jobs"]):
        path = root / "models" / f"job_{i:03d}"
        for suffix in (".npz", ".json"):
            candidate = path.with_suffix(suffix)
            if digest(candidate) != frozen["sha256"][str(candidate.relative_to(root))]:
                raise RuntimeError("Frozen G16 output changed")
        model = json.loads(path.with_suffix(".json").read_text())
        audit = json.loads((path.parent / (path.name + "_audit.json")).read_text())
        meta.append(model)
        with np.load(path.with_suffix(".npz")) as a:
            theta0 = a["reference_theta"]
        with np.load(path.parent / (path.name + "_audit_only.npz")) as a:
            truth = a["truth_indices"]
        index0 = project(torch.tensor(theta0[:-2].reshape(truth.shape)), plan["codes"])[
            0
        ]
        metric, aligned, _ = align_indices(index0, truth, plan["codes"])
        base.append(
            {"job": job, "recovery": metric, "aligned": aligned, "index": index0}
        )
        for source, after in zip(model["records"], audit["records"], strict=True):
            records.append(
                {
                    "job_id": i,
                    "family": job["family"],
                    "initialization": job["initialization"],
                    "group": "cold" if job["initialization"] == "cold" else "warm",
                    **source,
                    **after,
                }
            )
    rows = []
    for group in ("cold", "warm"):
        for n in plan["observations"]:
            selected = [
                row for row in records if row["group"] == group and row["n"] == n
            ]
            for method in METHODS:
                recovery = [row["recovery"].get(method) for row in selected]
                rows.append(
                    {
                        "group": group,
                        "n": n,
                        "B_labels": 2 * n,
                        "method": method,
                        "reference_states": len(selected),
                        "independent_geometries": len(plan["families"]),
                        "returned": sum(x is not None for x in recovery),
                        "exact": sum(x is not None and x["exact_I"] for x in recovery),
                        "wrong_issued": sum(
                            x is not None and not x["exact_I"] for x in recovery
                        ),
                        "true_CI_all_cover": sum(
                            row["all_true_weight_intervals_cover"] for row in selected
                        ),
                        "pseudo_CI_all_cover": sum(
                            row["all_pseudo_weight_intervals_cover"] for row in selected
                        ),
                        "median_max_bias_SE": float(
                            np.median(
                                [
                                    row["max_standardized_linearization_bias"]
                                    for row in selected
                                ]
                            )
                        ),
                        "mean_validation_mse": float(
                            np.mean([row["validation_mse"] for row in selected])
                        ),
                    }
                )
    with (report / "comparison.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "comparisons": rows,
        "reference_scalar_exact": {
            group: sum(
                row["recovery"]["categorical_errors"] == 0
                for row in base
                if (row["job"]["initialization"] == "cold") == (group == "cold")
            )
            for group in ("cold", "warm")
        },
        "all_pseudo_coverage": sum(
            row["all_pseudo_weight_intervals_cover"] for row in records
        ),
        "all_true_coverage": sum(
            row["all_true_weight_intervals_cover"] for row in records
        ),
        "states": len(records),
        "cohort_sha256": digest(root / "models_frozen.json"),
        "scope": "Known one-step control,4 opened geometries, correlated warm references, exact CI for pseudo-target only. Additional B data/replicate and warm-moment cost explicit. No native competitor win, new method, confidence theorem for nonlinear truth, or transfer claim.",
    }
    save_json(report / "summary.json", summary)
    plot_precision(report, records, plan)
    plot_indexators(root, report, plan, base)
    plot_weights(root, report, plan)
    plot_recovery(report, rows, plan)
    write_report(report, rows, summary)


def plot_precision(report, records, plan):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    for job_id in range(len(plan["jobs"])):
        selected = [row for row in records if row["job_id"] == job_id]
        cold = selected[0]["group"] == "cold"
        color = "#bd795b" if cold else "#467b89"
        x = [2 * row["n"] for row in selected]
        for ax, field in (
            (axes[0], "max_standardized_linearization_bias"),
            (axes[1], "max_standardized_pseudo_error"),
        ):
            ax.plot(
                x, [row[field] for row in selected], color=color, alpha=0.65, marker="."
            )
        axes[2].plot(
            x,
            [row["validation_mse"] for row in selected],
            color=color,
            alpha=0.65,
            marker=".",
        )
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Additional B labels = 2 × independent inputs")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Max absolute linearization bias / marginal SE")
    axes[0].set_title("Bias persists while stochastic uncertainty shrinks")
    axes[1].set_ylabel("Max stochastic pseudo-target error / marginal SE")
    axes[1].set_title("Noise alone stays in its expected scale")
    axes[2].set_ylabel("Observed independent validation MSE")
    axes[2].set_title("Function fit includes noise variance 0.04")
    fig.suptitle(
        "G16: orange = cold; blue = warm; 16 references on 4 opened geometries"
    )
    fig.savefig(report / "independent_score_bias_noise_loss.png", dpi=180)
    plt.close(fig)


def plot_recovery(report, rows, plan):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, group, total in ((axes[0], "warm", 12), (axes[1], "cold", 4)):
        for method in METHODS:
            selected = [
                row for row in rows if row["group"] == group and row["method"] == method
            ]
            ax.plot(
                [row["B_labels"] for row in selected],
                [row["exact"] for row in selected],
                marker="o",
                label=LABELS[method],
            )
        ax.set_xscale("log", base=2)
        ax.set_ylim(-0.3, total + 0.6)
        ax.set_yticks(range(0, total + 1, 2 if total == 12 else 1))
        ax.set_xlabel("Additional B labels (2 per input)")
        ax.set_ylabel(f"Exact I / {total} reference states")
        ax.set_title(
            f"{group.capitalize()}; {len(plan['families'])} underlying geometries"
        )
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "One step with explicit readouts; not native DARTS or a matched competitor win"
    )
    fig.savefig(report / "independent_score_recovery.png", dpi=180)
    plt.close(fig)


def plot_indexators(root, report, plan, base):
    n = plan["observations"][1]
    fig, axes = plt.subplots(4, 5, figsize=(13, 11), constrained_layout=True)
    for line, family in enumerate(plan["families"]):
        cold = line * len(plan["initializations"])
        warm = cold + 1
        images = []
        for job_id, method in (
            (cold, "scalar_dp"),
            (warm, "scalar_dp"),
            (warm, "signed_student_graph"),
        ):
            with np.load(root / "models" / f"job_{job_id:03d}_audit_only.npz") as a:
                if not images:
                    truth = a["truth_indices"]
                images.append(a[f"n{n}__{method}__aligned"])
        data = [truth, base[cold]["aligned"], *images]
        for col, (array, title) in enumerate(
            zip(
                data,
                [
                    "True I",
                    "Cold reference + DP",
                    "Cold one-step + DP",
                    "Warm one-step + DP",
                    "Warm one-step + graph",
                ],
                strict=True,
            )
        ):
            im = axes[line, col].imshow(array, cmap="RdBu_r", vmin=-4, vmax=4)
            if line == 0:
                axes[line, col].set_title(title, fontsize=9)
            axes[line, col].set_xticks([0, 7, 15])
            axes[line, col].set_yticks([0, 7, 15])
            if col == 0:
                axes[line, col].set_ylabel(f"Family {family}\nInput coordinate")
            if line == 3:
                axes[line, col].set_xlabel("Hidden neuron (audit aligned)")
    fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        shrink=0.65,
        ticks=range(-4, 5),
        label="Signed code ID; 0 = disconnected",
    )
    fig.suptitle(
        f"G16 selected I: fixed first warm reference; B inputs {n:,}, labels {2 * n:,}\n"
        "All families shown; names/signs/hidden columns aligned only for audit"
    )
    fig.savefig(report / "independent_score_indexators.png", dpi=180)
    plt.close(fig)


def plot_weights(root, report, plan):
    n = plan["observations"][1]
    fig, axes = plt.subplots(4, 4, figsize=(12, 11), constrained_layout=True)
    entries = plan["inputs"] * plan["hidden"]
    for line, family in enumerate(plan["families"]):
        images = []
        for job_id in (
            line * len(plan["initializations"]),
            line * len(plan["initializations"]) + 1,
        ):
            with np.load(root / "models" / f"job_{job_id:03d}.npz") as a:
                reference = a["reference_theta"][:entries].reshape(
                    plan["inputs"], plan["hidden"]
                )
                after = a[f"n{n}__theta"][:entries].reshape(reference.shape)
            with np.load(root / "models" / f"job_{job_id:03d}_audit_only.npz") as a:
                truth = a["truth_weights"]
            costs = ((reference[:, :, None] - truth[:, None, :]) ** 2).sum(0)
            source, target = linear_sum_assignment(costs)
            order = source[np.argsort(target)]
            if not images:
                images.extend([truth, reference[:, order], after[:, order]])
            else:
                images.append(after[:, order])
        limit = max(float(np.abs(array).max()) for array in images)
        for col, (array, title) in enumerate(
            zip(
                images,
                ["True W", "Cold reference W", "Cold one-step W", "Warm one-step W"],
                strict=True,
            )
        ):
            im = axes[line, col].imshow(array, cmap="RdBu_r", vmin=-limit, vmax=limit)
            if line == 0:
                axes[line, col].set_title(title)
            axes[line, col].set_xticks([0, 7, 15])
            axes[line, col].set_yticks([0, 7, 15])
        axes[line, 0].set_ylabel(f"Family {family}\nInput coordinate")
        fig.colorbar(im, ax=axes[line].tolist(), shrink=0.8, label="Actual weight")
    fig.suptitle(
        f"Actual numerical W before/after one independent score; B inputs {n:,}\n"
        "Hidden neurons matched using reference W, audit only; no code-name scaling"
    )
    fig.savefig(report / "independent_score_weights.png", dpi=180)
    plt.close(fig)


def write_report(report, rows, summary):
    lines = [
        "# G16: независимый one-step и его нелинейное смещение",
        "",
        "После независимого шага все12 warm reference-состояний дают точный I через простой scalar-DP уже на первом B-бюджете; полный Student-граф даёт6/12, затем12/12. Два из четырёх cold reference остаются ошибочными при любом бюджете. Это известный one-step контроль, не новая конкурентная победа.",
        "",
        "## Данные и границы сравнения",
        "",
        "THEORY.md§23 до запуска.4 ранее открытые нелинейные DeepSets геометрии8401..8404,16inputs/16ReLU, общий positive bias+offset,258 чисел,4 signed-кода+фиксированный0. На геометрию frozen G9 cold и3 moment-warm reference. Эти12 warm не являются12 независимыми структурами: всего4. Reference не меняется между n. Все16 новых моделей/48 budget-состояний заморожены до teacher-аудита.",
        "",
        "Новые независимые B:262144/1048576/4194304 Gaussian входов, по2 noisy labels sigma.2 на каждый. **B-label budget=2n**, плюс прежний reference-fit262144 и стоимость выбранного moment-предобучения (warm8.39m/67.1m/536.9m пар). Это не равный по общему бюджету benchmark против прежнего G9 или конкурентов. Все reference одной геометрии видят одинаковые nested B. Дополнительная observed validation16384 независима. Нет выбора по teacher, новых target-C или переноса.",
        "",
        "Дисперсия оценена разностью репликат, независимо от их среднего. Условно на reference/x Student-интервалы точны для дизайн-зависимой **псевдоцели**. Для истинных параметров остаётся nonlinear bias. Номинальныйalpha.05 делится по4 reference×3look иE/E² контрастам отдельно для каждого метода/геометрии; совместная гарантия на весь benchmark не заявляется. Старый G9-gate/abstentions не изменены, новая one-step процедура не требует его stationarity, поскольку оценивает другой conditional target.",
        "",
        "## Точная структура",
        "",
        "В таблице exact/returned/total; отсутствие выдачи считается неуспехом. Все192 readout-оценки учтены. Native DARTS здесь не обучается; DP/zero+DP/graph/interval объявлены явно.",
        "",
        "|Reference|Метод|B262144 (524288 labels)|B1048576 (2097152 labels)|B4194304 (8388608 labels)|",
        "|---|---|---:|---:|---:|",
    ]
    for group in ("cold", "warm"):
        for method in METHODS:
            selected = [
                row for row in rows if row["group"] == group and row["method"] == method
            ]
            values = [
                f"{row['exact']}/{row['returned']}/{row['reference_states']}"
                for row in selected
            ]
            lines.append(f"|{group}|{LABELS[method]}|{'|'.join(values)}|")
    lines.extend(
        [
            "",
            f"Для прежних reference W без нового B scalar-DP был точен{summary['reference_scalar_exact']['cold']}/4 cold и{summary['reference_scalar_exact']['warm']}/12 warm. Эти числа описывают starting point; разница после шага включает дополнительную информацию2n новых labels и не приписывается только optimizer-трюку.",
            "",
            "## Шум против смещения",
            "",
            f"После freeze auditor фиксирует hidden-permutation по reference-to-teacher W и точно разлагает ошибку на projected Gaussian noise и nonlinear bias. Все весовые CI покрывают псевдоцель в{summary['all_pseudo_coverage']}/{summary['states']} budget-состояний, истинные W — в{summary['all_true_coverage']}/{summary['states']}. Наблюдённое покрытие не доказывает неизвестную true-parameter гарантию. Structural scores отдельно учитывают допустимые hidden/code/sign переименования.",
            "",
            "У warm median max-bias/SE по256 весам растёт .247→.479→.938 при увеличении B. При4.19m максимум по warm около1.42SE: bias не исчезает, хотя консервативные интервалы во всех этих наблюдённых случаях ещё покрывают true-W. У cold8401/8402 max-bias достигает484/385SE, RMS-bias≈.128/.0383; простое добавление B не исправляет начальную локальную конфигурацию. Остальные два cold точны. Чистая случайная ошибка псевдоцели остаётся нормального масштаба — это отличает bias от плохой оценки noise/covariance.",
            "",
            "Ошибочных graph/interval выдач в этой серии не наблюдалось, но finite true-I гарантия отсюда не следует: Student-теорема относится к pseudo-target, который вообще не обязан иметь истинный шэринг. Успех требует достаточно хорошего reference либо нового контроля/устранения remainder. Сам one-step известен, в том числе для MLP: [Meskini et al.2024](https://link.springer.com/article/10.1007/s11063-024-11637-6). Их гладкийsigmoid аргумент не переносится автоматически на ReLU.",
            "",
            "## Следующий вопрос",
            "",
            "Для плохих cold reference требуется механизм смены ошибочной конфигурации, а не ещё большая оценочная B-выборка при фиксированном Jacobian. Сначала диагностировать, какие нейроны/равенства ошибочны и сохраняется ли residual-функциональная информация; затем теоретически обосновать структурное исправление и сравнить его с сильным native LC/Yeh/LUT-Q/numerical baseline. Для хороших reference дальнейший вопрос — уменьшение/контроль bias при обновлении reference на независимых данных. Ни один из этих следующих методов вG16 не заявлен реализованным.",
            "",
            "![Структурное восстановление](independent_score_recovery.png)",
            "",
            "![Истинные и выбранные I](independent_score_indexators.png)",
            "",
            "![Шум, bias и validation](independent_score_bias_noise_loss.png)",
            "",
            "![Численные веса до и после](independent_score_weights.png)",
        ]
    )
    (report / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    generate(args.root)


if __name__ == "__main__":
    main()
