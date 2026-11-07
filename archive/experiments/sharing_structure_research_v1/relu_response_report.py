"""Summarize frozen G15 observed single-step ReLU responses."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.darts_structured_tasks_v1.data import digest, save_json


def generate(root):
    report = root / "report"
    report.mkdir(exist_ok=False)
    plan = json.loads((root / "protocol.json").read_text())
    frozen = json.loads((root / "models_frozen.json").read_text())
    rows, summary = [], []
    for i in range(len(plan["jobs"])):
        path = root / "models" / f"job_{i:03d}.json"
        if digest(path) != frozen["sha256"][str(path.relative_to(root))]:
            raise RuntimeError("Frozen G15 response changed")
        row = json.loads(path.read_text())
        rows.append(row)
        gn = [point for point in row["records"] if point["relative_ridge"] == 0]
        best = max(row["records"], key=lambda point: point["actual_improvement"])
        best_gn = max(gn, key=lambda point: point["actual_improvement"])
        summary.append(
            {
                "job_id": i,
                "family": row["job"]["family"],
                "initialization": row["job"]["initialization"],
                "decrement": row["scaled_newton_decrement"],
                "linear_shift_bound_SE": row["linear_contrast_shift_bound_se"],
                "GN_full_scaled_actual": gn[0]["scaled_actual_improvement"],
                "GN_full_scaled_prediction": gn[0]["scaled_quadratic_improvement"],
                "GN_full_gate_cross_fraction": gn[0]["gate_cross_fraction"],
                "best_GN_fraction": best_gn["fraction"],
                "best_relative_ridge": best["relative_ridge"],
                "best_fraction": best["fraction"],
                "best_scaled_actual": best["scaled_actual_improvement"],
                "best_actual_MSE_improvement": best["actual_improvement"],
                "best_validation_MSE_improvement": best["validation_improvement"],
            }
        )
    with (report / "response.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    warm = [row for row in summary if row["initialization"] != "cold"]
    information = {
        "warm_negative_full_GN": sum(row["GN_full_scaled_actual"] < 0 for row in warm),
        "warm_count": len(warm),
        "warm_median_linear_shift_bound_SE": float(
            np.median([row["linear_shift_bound_SE"] for row in warm])
        ),
        "warm_best_GN_fraction_range": [
            min(row["best_GN_fraction"] for row in warm),
            max(row["best_GN_fraction"] for row in warm),
        ],
        "warm_best_MSE_improvement_range": [
            min(row["best_actual_MSE_improvement"] for row in warm),
            max(row["best_actual_MSE_improvement"] for row in warm),
        ],
        "maximum_identity_error": max(
            row["maximum_quadratic_identity_error"] for row in rows
        ),
        "scope": "Opened nonlinear single-step diagnosis; no new I, transfer, confidence or novelty claim. Original G9 gate/results unchanged.",
        "cohort_frozen_sha256": digest(root / "models_frozen.json"),
    }
    save_json(report / "summary.json", information)
    plot_curves(report, rows)
    plot_summary(report, summary)
    lines = [
        "# G15: реальный ReLU-loss против fixed-gate квадратичного прогноза",
        "",
        "Во всех12 noisy warm-состояниях полный GN-шаг **увеличивает** реальный source-loss, хотя локальная квадратичная модель предсказывает уменьшение. Точный измеренный остаток объясняет разницу. Это ограничение локальной fixed-gate модели около текущих весов, а не доказательство отсутствия информации о структуре.",
        "",
        "## Протокол",
        "",
        "THEORY.md§22 до запуска:16 замороженных G9 состояний,4 ранее открытые геометрии×cold/три warm moment-старта, noise.2,262144 прежних source-наблюдения,8192 прежних validation. Входы16, ReLU-нейроны16, общий positive bias и offset,258 параметров. Наблюдения восстановлены с точным прежним SHA256. Teacher используется только генератором для воспроизведения этих наблюдений; teacher-градиенты, новые labels, структура и C в направления не входят.",
        "",
        "На каждом состоянии fullGN и relative-ridge1e−6/1e−4/1e−2, fractions2^−k,k=0..16:68 single-step ответов. Это не68 новых optimizer-шагов и не новая training trajectory. Все1088 responses сохранены; cohort заморожен. Шесть свободных GPU0/1/2/3/6/7; программные тесты не создавались и не запускались.",
        "",
        "## Результаты",
        "",
        "Scaled improvement = n·(oldMSE−newMSE)/sigmahat²; положительное число означает улучшение. D=n·gᵀH⁻¹g/sigmahat². Best fraction выбирается только по фактическому source-loss для диагностики; validation-change показывается отдельно и не называется новым независимым выигрышем.",
        "",
        "|Геометрия/старт|D|Full GN: прогноз|Full GN: факт|Лучшая GN fraction|Лучший факт (все ridge)|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        label = row["initialization"].replace("moment_pairs", "warm ")
        lines.append(
            f"|{row['family']} {label}|{row['decrement']:.5f}|"
            f"{row['GN_full_scaled_prediction']:.5f}|{row['GN_full_scaled_actual']:.5f}|"
            f"{row['best_GN_fraction']:.6f}|{row['best_scaled_actual']:.6f}|"
        )
    lo, hi = information["warm_best_MSE_improvement_range"]
    lines.extend(
        [
            "",
            f"У warm-пусков лучшая GN fraction находится между1/1024 и1/64. Даже лучший из всех68 откликов уменьшает source-MSE лишь на{lo:.2e}..{hi:.2e} при исходной MSE≈.04. Median sqrt(D)={information['warm_median_linear_shift_bound_SE']:.3f}: это верхняя граница сдвига любого контраста до **локального линейного** optimum в единицах его SE. Она не является границей истинной ошибки theta или неизвестного нелинейного bias.",
            "",
            f"Максимальная ошибка точного равенства actual=quadratic+measured_remainder составляет{information['maximum_identity_error']:.3e}. При полном GN доля сменивших gate событий у warm-пусков около7e−5..2e−4: малая доля не делает remainder пренебрежимым, когда предсказанное улучшение уже очень мало. Это научная проверка вычислимого тождества, не гарантия coverage.",
            "",
            "Cold8402 отличается: D≈5.20, лучшая GN fraction=.5, scaled actual≈2.73. Там остаётся численная недооптимизация. Её нельзя объяснять тем же почти стационарным режимом, что warm. Validation-change лучшего source-шага отрицательный; это не доказательство улучшения переноса. Остальные cold также показаны отдельно вCSV/фигуре.",
            "",
            "## Что меняет диагноз",
            "",
            "Строгий G9-gate1e−4 не достигался, однако blanket-вывод «кодбук/сеть не обучена» для warm необоснован. В статистическом масштабе шаг мал; fixed-gate GN прогноз уже расходится с истинным loss из-за нелинейного остатка. При этом формула covariance остаётся лишь локальной/asymptotic мотивацией. Первичные G9 abstention/порог не изменены; post-hoc его ослабление не названо исправлением.",
            "",
            "Новый алгоритм в G15 не обучался, новый I и transfer не оценивались. Следующий обоснованный вопрос — независимая one-step score оценка с контролем ReLU remainder/coverage, либо полноценный сильный native nonlinear baseline; сама one-step идея известна. Для общей цели всё ещё нужны дискретный shared-I across functions, сравнение с сильными конкурентами и содержательное новое отличие.",
            "",
            "![Ответ на полный и сокращённый шаг](relu_quadratic_response.png)",
            "",
            "![Нелинейный остаток и масштаб ошибки](relu_response_summary.png)",
        ]
    )
    (report / "REPORT.md").write_text("\n".join(lines) + "\n")


def plot_curves(report, rows):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, job_id in zip(axes.ravel(), (1, 5, 9, 13), strict=True):
        row = rows[job_id]
        gn = sorted(
            [point for point in row["records"] if point["relative_ridge"] == 0],
            key=lambda point: point["fraction"],
        )
        x = [point["fraction"] for point in gn]
        ax.plot(
            x,
            [point["scaled_quadratic_improvement"] for point in gn],
            label="Fixed-gate quadratic prediction",
            lw=2,
        )
        ax.plot(
            x,
            [point["scaled_actual_improvement"] for point in gn],
            label="Actual ReLU loss decrease",
            lw=2,
            marker=".",
        )
        ax.plot(
            x,
            [point["scaled_remainder_contribution"] for point in gn],
            label="Measured nonlinear remainder",
            lw=1.5,
            ls="--",
        )
        ax.axhline(0, color="black", lw=0.7)
        ax.set_xscale("log", base=2)
        ax.set_xlim(2**-16, 1)
        ax.set_xlabel("Fraction of one full GN direction (not optimizer steps)")
        ax.set_ylabel("Scaled source-loss decrease; positive = better")
        ax.set_title(f"Family {row['job']['family']}; first frozen warm start")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "G15: small gate-crossing remainder reverses an already tiny GN benefit"
    )
    fig.savefig(report / "relu_quadratic_response.png", dpi=180)
    plt.close(fig)


def plot_summary(report, rows):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    colors = [
        "#bd795b" if row["initialization"] == "cold" else "#467b89" for row in rows
    ]
    x = np.arange(len(rows))
    axes[0].bar(x, [row["linear_shift_bound_SE"] for row in rows], color=colors)
    axes[0].set_ylabel("sqrt(D): upper bound to local linear optimum, SE")
    axes[0].set_title("Local numerical scale; not true parameter error")
    axes[1].scatter(
        [row["GN_full_scaled_prediction"] for row in rows],
        [row["GN_full_scaled_actual"] for row in rows],
        c=colors,
    )
    axes[1].axhline(0, color="black", lw=0.7)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Predicted scaled decrease for full GN")
    axes[1].set_ylabel("Actual scaled decrease for full GN")
    axes[1].set_title("Full-step prediction versus actual")
    axes[2].scatter(x, [row["best_actual_MSE_improvement"] for row in rows], c=colors)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Best actual source-MSE decrease in 68 responses")
    axes[2].set_title("Warm best-step gain is tiny; one cold differs")
    for ax in (axes[0], axes[2]):
        ax.set_xticks([1.5, 5.5, 9.5, 13.5], ["8401", "8402", "8403", "8404"])
        ax.set_xlabel("Family; each has cold + three warm starts")
    fig.suptitle("G15: blue = warm; orange = cold; all 16 frozen noisy G9 states")
    fig.savefig(report / "relu_response_summary.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    generate(args.root)


if __name__ == "__main__":
    main()
