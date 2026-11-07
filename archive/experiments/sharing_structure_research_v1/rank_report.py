"""Report the registered collinear-code diagnostic without retraining models."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.sharing_benchmark_v1.report import save_figure

from .diagnostic_report import categorical_map

LABELS = {
    "ols_profiles": "OLS: полные профили",
    "ols_pca": "PCA: нормированный базис K=3",
    "ols_pca_scores": "PCA: сохранены амплитуды",
    "ols_noise_debiased_pca": "Noise-corrected PCA: базис",
    "ols_cross_pca": "Cross-split PCA: базис",
    "noise_whitened_pca": "Noise-whitened PCA: базис",
    "diagonal_noise_whitened_pca": "Diagonal whitening PCA: базис",
    "corrected_kernel_coordinate": "Полный corrected-kernel",
    "signed_contrast_graph": "Полный F-граф",
    "codebook_crossfit": "Cross-fit кодбуков + граф",
    "codebook_crossfit_only": "Cross-fit без полного fallback",
    "codebook_union_crossfit": "Cross-fit объединения направлений",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root, grouped, records = args.root, defaultdict(list), []
    plan = json.loads((root / "protocol.json").read_text())
    for job_id in range(len(plan["jobs"])):
        for folder in (root / "cases", root / "projected"):
            case = json.loads((folder / f"job_{job_id:03d}.json").read_text())
            for row in case["records"]:
                item = {"family": case["job"]["family"]["id"], **row}
                records.append(item)
                grouped[row["method"], row["n"]].append(item)
    summaries = []
    for (method, n), values in sorted(grouped.items()):
        summaries.append(
            {
                "method": method,
                "n": n,
                "exact": sum(
                    v.get("recovery", {}).get("exact_I", False) for v in values
                ),
                "total": len(values),
                "mean_I_errors_when_issued": float(
                    np.mean(
                        [
                            v["recovery"]["categorical_errors"]
                            for v in values
                            if "recovery" in v
                        ]
                    )
                ),
            }
        )
    output = root / "report"
    output.mkdir(exist_ok=True)
    save_json(output / "summary.json", {"summary": summaries})
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    table = [
        "|Метод|" + "|".join(str(n) for n in plan["observations"]) + "|",
        "|---|" + "---:|" * len(plan["observations"]),
    ]
    for method, label in LABELS.items():
        selected = {s["n"]: s for s in summaries if s["method"] == method}
        table.append(
            "|"
            + label
            + "|"
            + "|".join(
                f"{selected[n]['exact']}/{selected[n]['total']}"
                for n in plan["observations"]
            )
            + "|"
        )

    job_id, n = 2, 131072  # First arbitrary signed pilot geometry, fixed in report.
    selections = (
        "noise_whitened_pca",
        "ols_pca_scores",
        "ols_profiles",
        "signed_contrast_graph",
        "codebook_crossfit",
    )
    arrays = {}
    for path in (
        root / "showcases" / f"job_{job_id:03d}_n{n}.npz",
        root / "projected" / f"job_{job_id:03d}_n{n}.npz",
    ):
        with np.load(path) as archive:
            arrays.update({key: archive[key] for key in archive.files})
    matrices = [arrays["truth_audit_only"]]
    labels = ["Истинный I: 3 группы, rank(C)=1"]
    for method in selections:
        matrices.append(arrays[f"{method}_aligned_indices"])
        row = next(
            r
            for r in records
            if r["family"] == 6201 and r["n"] == n and r["method"] == method
        )
        labels.append(
            f"{LABELS[method]}\nошибок I: {row['recovery']['categorical_errors']}"
        )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    cmap, norm = categorical_map()
    for axis, matrix, label in zip(axes.flat, matrices, labels):
        artist = axis.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
        axis.set(title=label, xlabel="Вход", ylabel="Выход")
    fig.colorbar(
        artist, ax=list(axes.flat), ticks=np.arange(-3, 4), shrink=0.8, label="Код"
    )
    fig.suptitle("Коллинеарные, но различные коды; геометрия6201; n=131072")
    save_figure(fig, output, "collinear_indexators")
    report = """# G5: различимость групп не равна рангу межзадачных весов

Это заранее описанная в THEORY.md§12 диагностика на четырёх уже открытых pilot-геометриях, а не независимое подтверждение. Изменено только возбуждение кодов: c_t=s_t(1,2,3). Три группы имеют разные ненулевые профили и различные относительные знаки; матрица коэффициентов при этом ранга1. T=128, condition100, Gaussian noise1, пять вложенных бюджетов. Размеры групп и роли координат декодерам неизвестны, K=3 известно точно.

Полные профили, PCA с сохранёнными амплитудами и контрастные графы восстанавливают4/4 структуры уже при512 наблюдениях. Принудительное выделение трёх нормированных направлений добавляет два шумовых направления и не восстанавливает ни одной структуры даже при131072. Это не провал PCA вообще: его вариант с амплитудами здесь успешен. Граф проверяет различимость профилей; он не требует rank(C)=K.

При фиксированном кодбуке3s и нуле мягкие коэффициенты1/3,2/3,1 точно воспроизводят все три положительных числа s,2s,3s. Поэтому идеальный source-loss допускает неправильное число жёстких групп после argmax. Диагностика демонстрирует эту теоретическую возможность и различие спектральных декодеров; она не является новым запуском DARTS/Yeh или доказательством их обязательного провала. Модель с произвольным непрерывным базисом также не обязана сохранить все ненаблюдённые направления будущих независимых C.

## Точные индексаторы / четыре геометрии

""" + "\n".join(table)
    report += """

## Хитмапы

![Индексаторы до и после выбора представления](collinear_indexators.png)

Никаких teacher-значений декодеры не получали; выравнивание названий кодов применяется только для аудита рисунка. Это линейный пример. Содержательная новизна и преимущество собственного codebook-компонента перед сильными профильно-статистическими контролями по нему не установлены.
"""
    path = output / "REPORT.md"
    path.write_text(report)
    save_json(
        output / "report_manifest.json",
        {
            "report_sha256": digest(path),
            "protocol_sha256": digest(root / "protocol.json"),
            "input_sha256": {
                str(p.relative_to(root)): digest(p)
                for folder in (root / "cases", root / "projected")
                for p in sorted(folder.glob("job_*.json"))
            },
        },
    )


if __name__ == "__main__":
    main()
