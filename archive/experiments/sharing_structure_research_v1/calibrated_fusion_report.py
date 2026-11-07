"""Report G13 matched weight and lambda-resolution experiments."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.darts_structured_tasks_v1.data import save_json
from experiments.sharing_benchmark_v1.evaluate import alignment

from .gaussian_stats import sample_index

LABELS = {
    "raw": "Raw adaptive PACS",
    "raw_shrinking_floor": "Raw + shrinking floor",
    "debiased": "Noise-corrected distances",
    "heat": "Known Gaussian-kernel weights",
    "lc_reference": "LC, frozen prior pilot settings",
}


def report(root):
    plan = json.loads((root / "protocol.json").read_text())
    coarse = json.loads((root / "audit.json").read_text())["records"]
    fine_root = root / "lambda_resolution"
    fine = json.loads((fine_root / "audit.json").read_text())["records"]
    fine_choices = json.loads((fine_root / "cohort_frozen.json").read_text())["choices"]
    raw = json.loads((Path(plan["raw_control"]) / "pilot_diagnosis.json").read_text())[
        "records"
    ]
    for row in raw:
        row["job"] = {"weight_metric": "raw", "n": row["n"]}
    old = Path(plan["g4_root"])
    old_plan = json.loads((old / "protocol.json").read_text())
    old_choices = json.loads((old / "hpo_frozen.json").read_text())["choices"]
    lc_row = old_choices["lc__native_or_decoded"]["row"]
    old_source = json.loads(
        (Path(old_plan["source_root"]) / "protocol.json").read_text()
    )
    lc = []
    for i, job in enumerate(old_plan["jobs"]):
        if job["method"] != "lc":
            continue
        family = old_source["jobs"][job["source_job"]]["family"]
        truth = sample_index(family["id"], family["kind"], old_plan, "cpu").numpy()
        with np.load(old / "pilot" / f"job_{i:03d}.npz") as a:
            recovery, _ = alignment(
                a["native_or_decoded__indices"][lc_row], truth, plan["codes"]
            )
            score = float(a["native_or_decoded__selection_mse"][lc_row])
        lc.append(
            {
                "job": {"weight_metric": "lc_reference", "n": job["n"]},
                "family": family,
                "recovery": recovery,
                "validation_mse": score,
            }
        )
    rows = []
    for protocol, source in (
        ("coarse7", coarse + raw + lc),
        (
            "fine43_fixed_weight_settings",
            fine + [c for c in lc if c["job"]["n"] == 8192],
        ),
    ):
        for metric in LABELS:
            for n in (8192, 131072):
                selected = [
                    c
                    for c in source
                    if c["job"]["weight_metric"] == metric and c["job"]["n"] == n
                ]
                if not selected:
                    continue
                rows.append(
                    {
                        "protocol": protocol,
                        "method": metric,
                        "n": n,
                        "families": len(selected),
                        "exact": sum(c["recovery"]["exact_I"] for c in selected),
                        "mean_I_errors": float(
                            np.mean(
                                [c["recovery"]["categorical_errors"] for c in selected]
                            )
                        ),
                        "mean_mask_f1": float(
                            np.mean([c["recovery"]["mask_f1"] for c in selected])
                        ),
                    }
                )
    output = root / "report"
    output.mkdir(exist_ok=True)
    save_json(
        output / "summary.json",
        {
            "rows": rows,
            "lc_prior_pilot_reference": lc,
            "scope": "Four already-opened pilot geometries, not independent confirmation. All new source choices frozen before audits. LC reference uses previously frozen global HPO row and its prior pilot numerical seed, not a new best-of3 competitor run.",
            "additional_cohort": "Registered only, not launched; deferred while candidate remains below existing strong controls.",
        },
    )
    with (output / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure, ax = plt.subplots(figsize=(9.8, 4.5))
    methods = list(LABELS)
    for i, protocol in enumerate(("coarse7", "fine43_fixed_weight_settings")):
        heights = [
            next(
                r["exact"] / r["families"]
                for r in rows
                if r["protocol"] == protocol
                and r["method"] == metric
                and r["n"] == 8192
            )
            for metric in methods
        ]
        ax.bar(
            np.arange(len(methods)) + (i - 0.5) * 0.35,
            heights,
            width=0.35,
            label="7 lambda values"
            if i == 0
            else "43 lambda values; fixed weight settings",
        )
    ax.set_xticks(
        np.arange(len(methods)),
        [LABELS[m].replace(" ", "\n", 1) for m in methods],
        fontsize=8,
    )
    ax.set(
        ylabel="Exact I fraction, source n8192",
        ylim=(0, 1.05),
        title="G13: four shared pilot geometries, same source/validation data",
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "calibrated_fusion_recovery.png", dpi=170)
    plt.close(figure)
    figure, axes = plt.subplots(2, 4, figsize=(14, 7))
    for row_number, family in enumerate((6101, 6201)):
        for column, metric in enumerate(("raw", "debiased", "heat")):
            case = next(
                c
                for c in fine
                if c["family"]["id"] == family and c["job"]["weight_metric"] == metric
            )
            with np.load(
                fine_root / "pilot" / f"job_{case['job_id']:03d}_audit_only.npz"
            ) as a:
                index, truth = a["index"], a["truth"]
            axes[row_number, column].imshow(index, vmin=-3, vmax=3, cmap="coolwarm")
            axes[row_number, column].set_title(
                f"{family}: {LABELS[metric]}\nI errors={case['recovery']['categorical_errors']}",
                fontsize=8,
            )
        axes[row_number, 3].imshow(truth, vmin=-3, vmax=3, cmap="coolwarm")
        axes[row_number, 3].set_title(f"{family}: true I (audit only)", fontsize=9)
    figure.suptitle(
        "Fixed registered illustrations: first conv and first signed family; fine lambda path"
    )
    figure.tight_layout()
    figure.savefig(output / "calibrated_fusion_indexators.png", dpi=170)
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for metric in ("raw", "debiased", "heat"):
        case = next(
            c
            for c in fine
            if c["family"]["id"] == 6201 and c["job"]["weight_metric"] == metric
        )
        meta = json.loads(
            (fine_root / "pilot" / f"job_{case['job_id']:03d}.json").read_text()
        )
        row = fine_choices[metric]["row"]
        steps = [h["step"] for h in meta["history"]]
        axes[0].plot(
            steps, [h["source_mse"][row] for h in meta["history"]], label=LABELS[metric]
        )
        axes[1].plot(
            steps,
            [
                np.nan if h["validation_mse"][row] is None else h["validation_mse"][row]
                for h in meta["history"]
            ],
            marker=".",
            linestyle="none",
            label=LABELS[metric],
        )
        axes[2].plot(steps, case["trace_mask_recall"], label=LABELS[metric])
    for ax, ylabel in zip(
        axes,
        (
            "Observed source MSE",
            "Independent source-validation MSE",
            "Native constraint mask recall",
        ),
    ):
        ax.set(xlabel="Actual ADMM steps along lambda path", ylabel=ylabel)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    axes[1].set_yscale("log")
    figure.tight_layout()
    figure.savefig(output / "calibrated_fusion_loss_recall.png", dpi=170)
    plt.close(figure)
    lines = [
        "# G13: коррекция adaptive fusion-весов",
        "",
        "Полный matched pilot на прежних четырёх геометриях6101/6102/6201/6202,128source-функций, n8192/131072. Это диагностическая серия, не independent confirmation. Source-validation, shape/K и solver одинаковы. Истинные I не участвуют в выборе. Все24 новых source-состояния сначала заморожены, затем открыты для аудита.",
        "",
        "THEORY§20 до запуска выводит unbiased squared-profile-distance и достаточный asymptotic KKT-аргумент при bounded-curvature/separation. Веса всегда фиксированные и неотрицательные; прежняя выпуклая цель и native zero-prox readout сохранены. Это не finite-sample F-гарантия, и приоритет специализации не установлен.",
        "",
        "| Метод | n8192, 7λ | n8192, 43λ | n131072, 7λ |",
        "|---|---:|---:|---:|",
    ]
    for metric in methods:

        def count(protocol, n, method=metric):
            r = next(
                r
                for r in rows
                if r["protocol"] == protocol and r["method"] == method and r["n"] == n
            )
            return f"{r['exact']}/{r['families']}"

        lines.append(
            f"| {LABELS[metric]} | {count('coarse7', 8192)} | {count('fine43_fixed_weight_settings', 8192)} | {count('coarse7', 131072)} |"
        )
    lines.extend(
        [
            "",
            "## Что из этого следует",
            "",
            "Уменьшение floor и известные Gaussian-kernel weights не исправили слабую группу. Noise-correction восстановила оба signed случая, но оба conv остались неточными. Это отдельный положительный эффект против raw fusion, а не преимущество над сильными конкурентами. Сохранённый LC на тех же четырёх задачах и с ранее выбранными настройками даёт3/4 приn8192; он не переобучался и не получал нового выбора по структурной истине.",
            "",
            "После уплотнения7→43 lambda значений raw остаётся0/4, corrected2/4. Весовые gamma/bandwidth в этой дополнительной проверке фиксированы по предыдущей source-validation; это не полный повторный joint gamma×lambda HPO. Поэтому нельзя объявлять отрицание всех PACS/heat-kernel вариантов. Главные ограничения — оставшиеся conv ошибки, малый уже открытый pilot, отсутствие независимого nonlinear сравнения и неустановленный приоритет.",
            "",
            "Заранее зарегистрированное расширение на G4 cohort ещё не запущено: кандидат сначала должен стать конкурентоспособнее на диагностике. Перенос нового corrected-I здесь не измерялся; G12 raw transfer находится в отдельном отчёте. Никакие target-метрики не использовались для настройки.",
            "",
            "- [Точное I и разрешение lambda](calibrated_fusion_recovery.png)",
            "- [Выбранные индексаторы и истинные](calibrated_fusion_indexators.png)",
            "- [Измеренные loss/recall кривые](calibrated_fusion_loss_recall.png)",
            "- [Таблица](comparison.csv), [summary](summary.json)",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"rows": rows, "output": str(output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    report(parser.parse_args().root)


if __name__ == "__main__":
    main()
