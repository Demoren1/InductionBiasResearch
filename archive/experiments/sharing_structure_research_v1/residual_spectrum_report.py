"""Report the frozen G14 conditional-score diagnostic without retuning."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.darts_structured_tasks_v1.data import digest, save_json

from .residual_spectrum import METHODS

LABELS = {
    "coarse_native": "Joint fusion: native I",
    "dense_profiles": "OLS profiles + readout",
    "dense_pca": "PCA + readout",
    "dense_corrected_pca": "Noise-corrected PCA + readout",
    "dense_whitened_pca": "Whitening-PCA + readout",
    "subset_pca": "Coarse-zero subset: PCA",
    "subset_corrected_pca": "Coarse-zero subset: corrected PCA",
    "subset_whitened_pca": "Coarse-zero subset: whitening-PCA",
    "restricted_pca": "Conditional score: PCA",
    "restricted_corrected_pca": "Conditional score: corrected PCA",
    "restricted_whitened_pca": "Conditional score: whitening-PCA",
    "residual_selected": "Validation-selected conditional spectra",
    "dense_selected": "Validation-selected dense spectra",
}


def generate(root):
    report = root / "report"
    report.mkdir(exist_ok=False)
    plan = json.loads((root / "protocol.json").read_text())
    rows = json.loads((root / "audit.json").read_text())["records"]
    summaries = []
    for regime, n in (
        ("primary", 8192),
        ("primary", 131072),
        ("low_noise", 131072),
        ("high_noise", 131072),
    ):
        for method in (*METHODS, "residual_selected", "dense_selected"):
            selected = [
                row
                for row in rows
                if row["method"] == method
                and row["job"]["regime"] == regime
                and row["job"]["n"] == n
            ]
            summaries.append(
                {
                    "regime": regime,
                    "n": n,
                    "method": method,
                    "exact": sum(row["recovery"]["exact_I"] for row in selected),
                    "cases": len(selected),
                    "mean_I_errors": float(
                        np.mean(
                            [row["recovery"]["categorical_errors"] for row in selected]
                        )
                    ),
                    "mean_mask_recall": float(
                        np.mean([row["recovery"]["mask_recall"] for row in selected])
                    ),
                    "mean_source_mse": float(
                        np.mean([row["source_mse"] for row in selected])
                    ),
                    "mean_validation_mse": float(
                        np.mean([row["validation_mse"] for row in selected])
                    ),
                }
            )
    with (report / "comparison.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    model_meta = [
        json.loads((root / "models" / f"job_{i:03d}.json").read_text())
        for i in range(len(plan["jobs"]))
    ]
    ratios = [
        row["details"]["median_variance_ratio"]
        for row in model_meta
        if row["job"]["n"] == 8192
    ]
    save_json(
        report / "summary.json",
        {
            "audit_sha256": digest(root / "audit.json"),
            "summaries": summaries,
            "low_n_variance_ratio_median": float(np.median(ratios)),
            "low_n_variance_ratio_range": [min(ratios), max(ratios)],
            "status": "Progress only; conditional score ties known whitening-PCA. Opened post-selection diagnostic, no new independent confirmation or finite confidence claim.",
        },
    )
    comparison_figure(report, summaries)
    indexator_figure(root, report, model_meta)
    variance_figure(root, report, model_meta, rows)
    main_report(report, summaries, ratios)


def comparison_figure(report, summaries):
    names = [
        "coarse_native",
        "dense_profiles",
        "dense_pca",
        "dense_corrected_pca",
        "dense_whitened_pca",
        "subset_corrected_pca",
        "restricted_pca",
        "restricted_corrected_pca",
        "restricted_whitened_pca",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    for ax, regime, n, title in (
        (axes[0], "primary", 8192, "Noise 1; n = 8,192 per source function"),
        (axes[1], "high_noise", 131072, "Noise 2; n = 131,072 per source function"),
    ):
        counts = [
            next(
                row["exact"]
                for row in summaries
                if row["method"] == name and row["regime"] == regime and row["n"] == n
            )
            for name in names
        ]
        bars = ax.barh(
            np.arange(len(names)),
            counts,
            color=["#777777"] * 5 + ["#a9bac9"] + ["#327a71"] * 3,
        )
        ax.set_yticks(np.arange(len(names)), [LABELS[name] for name in names])
        ax.invert_yaxis()
        ax.set_xlim(0, 13.8)
        ax.set_xlabel("Exactly recovered indexators / 12 geometries")
        ax.set_title(title)
        for bar, count in zip(bars, counts, strict=True):
            ax.text(
                count + 0.12,
                bar.get_y() + bar.get_height() / 2,
                f"{count}/12",
                va="center",
            )
    fig.suptitle(
        "G14: conditional residual information helps fusion, ties whitening-PCA"
    )
    fig.savefig(report / "residual_spectrum_comparison.png", dpi=180)
    plt.close(fig)


def indexator_figure(root, report, metadata):
    fig, axes = plt.subplots(2, 5, figsize=(15, 7), constrained_layout=True)
    names = [
        "coarse_native",
        "dense_pca",
        "dense_whitened_pca",
        "restricted_whitened_pca",
    ]
    for line, job_id in enumerate((0, 24)):
        with np.load(root / "models" / f"job_{job_id:03d}_audit_only.npz") as a:
            images = [
                a["truth"],
                *[a["aligned"][METHODS.index(name)] for name in names],
            ]
        titles = [
            "True I",
            "Fusion native I",
            "PCA + readout",
            "Whitening-PCA + readout",
            "Conditional score + readout",
        ]
        for col, (data, title) in enumerate(zip(images, titles, strict=True)):
            ax = axes[line, col]
            im = ax.imshow(
                data, cmap="RdBu_r", vmin=-3, vmax=3, interpolation="nearest"
            )
            ax.set_title(title, fontsize=10)
            ax.set_xticks([0, 5, 10, 15])
            ax.set_yticks([0, 6, 13])
            if col == 0:
                family = metadata[job_id]["job"]["family"]
                ax.set_ylabel(f"{family['kind']} {family['id']}\nOutput coordinate")
            ax.set_xlabel("Input coordinate")
    fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        shrink=0.75,
        ticks=np.arange(-3, 4),
        label="Signed code ID; 0 = disconnected",
    )
    fig.suptitle(
        "Fixed examples: first conv and signed geometry; n = 8,192, noise 1\n"
        "Code names/signs aligned only for audit; hidden coordinate roles were unknown"
    )
    fig.savefig(report / "residual_spectrum_indexators.png", dpi=180)
    plt.close(fig)


def variance_figure(root, report, metadata, rows):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    low = [row for row in metadata if row["job"]["n"] == 8192]
    axes[0].bar(
        np.arange(len(low)),
        [row["details"]["median_variance_ratio"] for row in low],
        color=["#6688aa"] * 6 + ["#558877"] * 6,
    )
    axes[0].axhline(1, color="black", ls="--", lw=1)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Median marginal variance: conditional / unrestricted")
    axes[0].set_xticks(
        np.arange(len(low)),
        [str(row["job"]["family"]["id"]) for row in low],
        rotation=60,
    )
    axes[0].set_title("Full nuisance covariance; same source data")
    methods = [
        "coarse_native",
        "dense_pca",
        "dense_whitened_pca",
        "restricted_corrected_pca",
        "restricted_whitened_pca",
    ]
    for ax, field, title in (
        (axes[1], "source_mse", "Source fit (including noise)"),
        (axes[2], "validation_mse", "New-C source-validation fit"),
    ):
        for i, name in enumerate(methods):
            chosen = [
                row[field]
                for row in rows
                if row["method"] == name and row["job"]["n"] == 8192
            ]
            ax.scatter(np.full(len(chosen), i), chosen, s=22, alpha=0.8)
            ax.scatter(i, np.mean(chosen), marker="_", s=400, color="black")
        ax.set_xticks(
            range(len(methods)),
            ["Fusion", "PCA", "Wh-PCA", "Cond-corr", "Cond-wh"],
            rotation=35,
        )
        ax.set_ylabel("Observed MSE")
        ax.set_title(title)
    axes[2].set_yscale("log")
    fig.suptitle(
        "G14: numerical precision improves; exact-I superiority is still absent"
    )
    fig.savefig(report / "residual_spectrum_variance_loss.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for line, job_id in enumerate((0, 24)):
        with np.load(root / "models" / f"job_{job_id:03d}.npz") as a:
            arrays = [
                a["diagnostics__dense_ols"][0],
                a["diagnostics__restricted_raw"][0],
                a["weights"][METHODS.index("restricted_whitened_pca"), 0],
            ]
        limit = max(float(np.max(np.abs(x))) for x in arrays)
        for col, (data, title) in enumerate(
            zip(
                arrays,
                ["Unrestricted OLS W", "Conditional raw W", "Discrete I + refitted C"],
                strict=True,
            )
        ):
            im = axes[line, col].imshow(data, cmap="RdBu_r", vmin=-limit, vmax=limit)
            axes[line, col].set_title(title)
            axes[line, col].set_xlabel("Input coordinate")
        axes[line, 0].set_ylabel(
            f"Family {metadata[job_id]['job']['family']['id']}\nOutput coordinate"
        )
        fig.colorbar(
            im,
            ax=axes[line].tolist(),
            shrink=0.8,
            label="Actual weight, source function 0",
        )
    fig.suptitle("Numerical weights before and after the explicit spectral readout")
    fig.savefig(report / "residual_spectrum_weights.png", dpi=180)
    plt.close(fig)


def main_report(report, summaries, ratios):
    lines = [
        "# G14: остаточный градиент с найденными равенствами",
        "",
        "Условный GLS/score с corrected или whitening-PCA исправляет пропущенную слабую группу:12/12 точных I приn8192/noise1 вместо0/12 у native joint-fusion. **Прямой известный whitening-PCA на тех же данных также даёт12/12.** Преимущество собственного уточнения перед сильным контролем не найдено.",
        "",
        "## Постановка и честность сравнения",
        "",
        "THEORY.md§21 записан до запуска.48 ранее открытых G4 условий,12 геометрий (6 скрытых conv/6 signed),16 входов/14 выходов,128 функций,3 знаковых ненулевых кода плюс фиксированный0; std(C)=(1,.3,.03), condition100. Одни и те же sample-Gram/cross, все R=coarse-zero позиции свободны. Ни маска, ни размеры/места групп не даны. Coarse-I взят из замороженного G12; полная nuisance-covariance вычислена в линейной restricted-модели. Все48 новых состояний и source-only selections заморожены до нового аудита. Шесть GPU0/1/2/3/6/7 использованы; чужие процессы не затрагивались.",
        "",
        "Это post-selection диагностика на уже открытых данных: правильность ранних равенств не гарантирована, поэтому Gaussian covariance формула не превращена в finite-sample сертификат после их выбора. Дальнейшая A/B независимая проверка пока не выполнена. Четыре режима повторяют одни12 геометрий и не являются48 независимыми структурами.",
        "",
        "## Точное восстановление I",
        "",
        "|Метод|Noise1 n8192|Noise1 n131072|Noise.2 n131072|Noise2 n131072|",
        "|---|---:|---:|---:|---:|",
    ]
    for name in (*METHODS, "residual_selected", "dense_selected"):
        values = [
            f"{row['exact']}/{row['cases']}"
            for row in summaries
            if row["method"] == name
        ]
        lines.append(f"|{LABELS[name]}|{'|'.join(values)}|")
    lines.extend(
        [
            "",
            "PCA-контроли имеют явно обозначенный дополнительный signed readout. Для одного остаточного кода он решён exact scalar-DP; для rankK используется тот же32-start signed-vector Lloyd с выбором по собственной distortion. Native fusion не переименован в native spectral learner. Отдельные validation-selected строки используют только прежние32 full-C source-функции support4/query128; teacher-I не выбирает метод или restart.",
            "",
            "## Что объясняет улучшение",
            "",
            f"В low-n серии median отношения marginal variance conditional/unrestricted равна{np.median(ratios):.3f}, диапазон по12 геометриям{min(ratios):.3f}..{max(ratios):.3f}. Учитываются nuisance-кодбуки и все off-diagonal ковариации. Corrected subset PCA даёт10/12, corrected conditional PCA12/12: pooling найденных равенств помогает слабому направлению. Но whitening subset и прямой whitening-PCA уже дают12/12; отдельный выигрыш по exact-I отсутствует. Уменьшение variance нельзя объявить конкурентной победой.",
            "",
            "При большом n/noise.2 исходный coarse-I уже верен. Еслиg=K, residual-строки возвращают этот I без нового шага; их12/12 в этих строках не является независимым повторным успехом поиска. Приnoise2 большинство coarse-I теряли третью группу, но известные прямые профили/PCA также решают12/12.",
            "",
            "Все численные W (unrestricted, conditional raw и I+C), индивидуальные спектры, source/query losses, selections и audit-aligned I сохранены вmodels. Веса после уточнения всё ещё требуют явного дискретного readout; это не исправление native DARTS. Нет выдуманной SGD-кривой: уточнение — closed-form solve/eigendecomposition. Actual-step G12-кривые остаются в его отдельном отчёте. Новое target transfer в G14 не запускалось; перенос G4/G12 не выдается за новый эксперимент.",
            "",
            "## Вывод для дальнейшей работы",
            "",
            "Слабая группа присутствует в наблюдениях; её потеря в raw fusion не является информационной невозможностью и не объясняется только незавершённым solver. Корректная геометрия шума уже достаточно её выделяет. Продолжать подбор scalar penalties на этой лёгкой для whitening серии ради заявления превосходства не обосновано. Нужна более показательная диагностика исходной нелинейной DeepSets и сильных native сравнений, а не новое имя GLS/PCA. Общая исследовательская цель остаётся active: преимущества, новая general процедура, независимое nonlinear подтверждение и устойчивость к нарушению noise-модели ещё не закрыты.",
            "",
            "![Сравнение](residual_spectrum_comparison.png)",
            "",
            "![Выбранные индексаторы](residual_spectrum_indexators.png)",
            "",
            "![Дисперсия и ошибки](residual_spectrum_variance_loss.png)",
            "",
            "![Численные веса](residual_spectrum_weights.png)",
        ]
    )
    (report / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    generate(args.root)


if __name__ == "__main__":
    main()
