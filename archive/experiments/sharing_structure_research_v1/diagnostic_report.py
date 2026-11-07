"""G2/G3 evidence, actual optimizer heatmaps, and registered pilot ablations."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.sharing_benchmark_v1.evaluate import alignment
from experiments.sharing_benchmark_v1.report import save_figure

from .curvature_control import reduced_fit


def categorical_map():
    cmap = ListedColormap(
        ["#a50045", "#f46d43", "#fee08b", "#f6f6f6", "#abdda4", "#66c2a5", "#3288bd"]
    )
    return cmap, BoundaryNorm(np.arange(-3.5, 4.0), 7)


def g2_figures(root, output):
    with np.load(root / "kernel_control/family_4201.npz") as archive:
        truth = archive["truth_audit_only"]
        dense = archive["ols_profiles"][:, 0].reshape(truth.shape)
        gram, cross = torch.tensor(archive["gram"]), torch.tensor(archive["cross_yx"])
        indices = [truth]
        names = ["Истинный I"]
        with np.load(root / "showcases/family_4201_n8192.npz") as z:
            metrics, aligned = alignment(z["ols_profiles_indices"], truth, 3)
            indices.append(aligned)
            names.append(f"Профили МНК\nошибок I: {metrics['categorical_errors']}")
        kernel = json.loads((root / "kernel_control/family_4201.json").read_text())
        chosen = kernel["selected_start"]["corrected"]
        indices.append(archive[f"corrected__{chosen}_aligned_indices"])
        names.append("Полный corrected-kernel\nошибок I: 68")
    with np.load(root / "curvature_control/family_4201.npz") as archive:
        indices.extend(
            [
                archive["ols_profiles_aligned_indices"],
                archive["noise_whitened_pca_aligned_indices"],
            ]
        )
        names.extend(
            ["Полная кривизна из МНК\nошибок I: 15", "Whitening-PCA\nошибок I: 0"]
        )
        whitened = torch.tensor(archive["noise_whitened_pca_indices"])
    cmap, norm = categorical_map()
    fig, axes = plt.subplots(
        1, len(indices), figsize=(21, 4.3), constrained_layout=True
    )
    for axis, index, name in zip(axes, indices, names):
        artist = axis.imshow(index, cmap=cmap, norm=norm, aspect="auto")
        axis.set(title=name, xlabel="Вход", ylabel="Выход")
    fig.colorbar(
        artist, ax=axes, ticks=np.arange(-3, 4), shrink=0.75, label="Знаковой код"
    )
    fig.suptitle(
        "Один и тот же случай 4201: 128 функций × 8192 примера, шум 1, condition 100"
    )
    save_figure(fig, output, "g2_indexators")

    # A real dense Adam fit on the same observed source task, not an oracle W.
    generator = torch.Generator().manual_seed(261105771)
    weight = (
        0.01 * torch.randn(truth.shape, generator=generator, dtype=torch.float64)
    ).requires_grad_()
    optimizer = torch.optim.Adam([weight], lr=0.03)
    loss_trace = []
    for step in range(2001):
        loss = ((weight @ gram[0] * weight).sum() - 2 * (weight * cross[0]).sum()) / (
            8192 * len(weight)
        )
        if step % 20 == 0:
            loss_trace.append([step, float(loss.detach())])
        if step < 2000:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    coefficient, _ = reduced_fit(whitened, gram, cross, 3)
    fitted_code = coefficient[0][whitened.abs().clamp_min(1) - 1] * whitened.sign()
    with np.load(root / "showcases/family_4201_n8192.npz") as archive:
        true_weight = archive["source_codes_weight_matrix"][0]
    matrices = [true_weight, weight.detach().numpy(), dense, fitted_code.numpy()]
    names = [
        "Истинные веса: аудит",
        "Плотный Adam: 2000 шагов",
        "Плотный МНК",
        "Найденный I + новый fit C",
    ]
    bound = max(abs(matrix).max() for matrix in matrices)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4), constrained_layout=True)
    for axis, matrix, name in zip(axes, matrices, names):
        artist = axis.imshow(
            matrix, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto"
        )
        axis.set(title=name, xlabel="Вход", ylabel="Выход")
    fig.colorbar(artist, ax=axes, shrink=0.8, label="Численное значение")
    fig.suptitle(
        "Веса source-функции №0; у всех оценок одни наблюдения; I не меняется при fit C"
    )
    save_figure(fig, output, "g2_dense_adam_weights")
    np.savez_compressed(
        output / "dense_adam_actual.npz",
        weights=weight.detach().numpy(),
        ols=dense,
        codebook_fit=fitted_code.numpy(),
        trace=np.array(loss_trace),
    )
    return {
        "adam_to_ols_relative": float(
            np.linalg.norm(weight.detach().numpy() - dense) / np.linalg.norm(dense)
        ),
        "dense_adam_steps": 2000,
    }


def g3_report(root, output):
    plan = json.loads((root / "protocol.json").read_text())
    jobs = [
        json.loads(path.read_text())
        for path in sorted((root / "cases").glob("job_*.json"))
    ]
    if len(jobs) != len(plan["jobs"]):
        raise RuntimeError("G3 pilot incomplete")
    projected_files = sorted((root / "projected").glob("job_*.json"))
    projected = [json.loads(path.read_text()) for path in projected_files]
    if projected and len(projected) != len(plan["jobs"]):
        raise RuntimeError("Projection ablation incomplete")
    summary = defaultdict(lambda: {"returned": 0, "exact": 0, "cases": 0})
    hard = []
    for number, job in enumerate(jobs):
        condition = job["job"]
        is_hard = (
            condition["condition"] == 100
            and condition["noise"] == 1
            and condition["excitation"] == "weak"
        )
        rows = job["records"] + (projected[number]["records"] if projected else [])
        for row in rows:
            key = ("hard" if is_hard else "all", row["n"], row["method"])
            value = summary[key]
            value["cases"] += 1
            value["returned"] += row["recovery"] is not None
            value["exact"] += row["recovery"] is not None and row["recovery"]["exact_I"]
        if is_hard:
            entry = {
                "family": condition["family"]["id"],
                "full_graph": job["first_graph_return"],
            }
            if projected:
                entry.update(projected[number]["first_graph_return"])
            hard.append(entry)
    methods = [
        "ols_profiles",
        "ols_pca",
        "ols_noise_debiased_pca",
        "noise_whitened_pca",
        "signed_contrast_graph",
    ]
    if projected:
        methods += [
            "codebook_crossfit",
            "codebook_crossfit_only",
            "codebook_union_crossfit",
        ]
    labels = {
        "ols_profiles": "Полные профили МНК",
        "ols_pca": "Обычный PCA",
        "ols_noise_debiased_pca": "Noise-corrected PCA",
        "noise_whitened_pca": "Noise-whitened PCA",
        "signed_contrast_graph": "Полный F-граф",
        "codebook_crossfit": "Кодбуки: cross-fit + fallback",
        "codebook_crossfit_only": "Кодбуки: только проекция",
        "codebook_union_crossfit": "Объединённые кодбуки + fallback",
    }
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    for method in methods:
        selected = [summary["hard", n, method] for n in plan["observations"]]
        axes[0].plot(
            plan["observations"],
            [value["exact"] / value["cases"] for value in selected],
            marker="o",
            label=labels[method],
        )
        if "graph" in method or "crossfit" in method:
            axes[1].plot(
                plan["observations"],
                [value["returned"] / value["cases"] for value in selected],
                marker="o",
                label=labels[method],
            )
    for axis in axes:
        axis.set(xscale="log", ylim=(-0.03, 1.03), xlabel="Примеров на функцию")
        axis.set_xticks(plan["observations"], list(map(str, plan["observations"])))
        axis.grid(alpha=0.2)
    axes[0].set(
        ylabel="Доля точно найденных I", title="Все четыре сложных pilot-геометрии"
    )
    axes[1].set(ylabel="Доля выданных I", title="Воздержание показано отдельно")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle("G3: 128 функций, std C = 1/0.3/0.03, шум 1, condition 100")
    save_figure(fig, output, "g3_recovery_and_coverage")
    rows = []
    for method in methods:
        cells = [summary["hard", n, method] for n in plan["observations"]]
        rows.append(
            "| "
            + " | ".join(
                [
                    labels[method],
                    *[f"{cell['exact']}/{cell['cases']}" for cell in cells],
                ]
            )
            + " |"
        )
    first_rows = []
    graph_methods = [
        "full_graph",
        *(
            ["codebook_crossfit", "codebook_crossfit_only", "codebook_union_crossfit"]
            if projected
            else []
        ),
    ]
    for entry in hard:
        first_rows.append(
            "| "
            + " | ".join(
                [
                    str(entry["family"]),
                    *[
                        "не выдан" if entry[name] is None else str(entry[name]["n"])
                        for name in graph_methods
                    ],
                ]
            )
            + " |"
        )
    summary_list = [
        {"regime": key[0], "n": key[1], "method": key[2], **value}
        for key, value in sorted(summary.items())
    ]
    save_json(
        output / "g3_summary.json",
        {
            "summary": summary_list,
            "hard_first_stop": hard,
            "pilot_cases": len(jobs),
            "projected_cases": len(projected),
        },
    )
    return "\n".join(rows), "\n".join(first_rows), graph_methods, jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1", type=Path, required=True)
    parser.add_argument("--g3", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.g3 / "report"
    output.mkdir(exist_ok=True)
    weights = g2_figures(args.g1, output)
    table, first, names, jobs = g3_report(args.g3, output)
    issued = [
        j["first_graph_return"] for j in jobs if j["first_graph_return"] is not None
    ]
    report = f"""# Полная кривизна, шум и контрастный граф: G2/G3

Это диагностика и pilot, **не итоговое независимое сравнение нового метода с конкурентами**.

## G2: что именно мешает

Все шесть случаев уже открыты в G1. На самом сложном4201 полный corrected-kernel улучшает свой score, но делает68 ошибок I после выбора лучшего из двух стартов. Его score превосходит score истинного I на0.0002984: здесь уже есть адаптивное переобучение шумной статистике. Точная source-MSE с полным H и новым fit C из OLS-старта оставляет15 ошибок; из предварительного whitening-PCA —0. Начало оптимизации остаётся существенно.

Стандартный noise-whitened PCA восстанавливает6/6. Вычитание noise covariance и whitening различаются: последнее учитывает её полную геометрию. Это известный контроль, а не заявляемая новизна. Полный F-граф на8192 примерах во всех шести случаях имеет5 компонент вместо7 и воздерживается.

![Индексаторы одного случая](g2_indexators.png)

![Настоящие обученные веса](g2_dense_adam_weights.png)

Плотный Adam действительно обучен2000 шагов на source-функции №0. Относительное отличие от её МНК: {weights["adam_to_ols_relative"]:.3g}. На этой линейной задаче остаётся шум оценок, а не предполагаемая нелинейная ловушка Adam. Сохранены его фактические веса и кривая целевой функции в dense_adam_actual.npz; константа Y² в кривой опущена.

## G3: протокол и полный учёт выдач

Четыре новые pilot-геометрии, два вида входной covariance, два уровня шума, слабые коды, full и collision-контроли:24 условия. На каждую функцию512/2048/8192/32768/131072 вложенных наблюдений. Использована точная симуляция достаточных статистик Gaussian регрессии, не замена эмпирических Grams на population. Bartlett sample Grams, projected Gaussian noise и residual chi-square позволяют не хранить сотни миллионов исходных x. Восемь базовых алгоритмов; размер и роли групп скрыты от поисковика.

Полный F-граф: выдал {len(issued)} индексаторов при первом допустимом останове, все {sum(r["recovery"]["exact_I"] for r in issued)} точные. В4 collision-случаях I не выдан даже на максимальном бюджете. Это наблюдение pilot, не эмпирическое доказательство номинальной гарантии0.05; сама узкая гарантия выведена в THEORY.md для точного K/общего Gaussian noise.

### Четыре самых сложных случая

Числитель — число exact I. Воздержание не записывается точным восстановлением. У всех одинаковые данные соответствующего n.

| Метод |512|2048|8192|32768|131072|
|---|---:|---:|---:|---:|---:|
{table}

![Восстановление и выдача](g3_recovery_and_coverage.png)

### Первый достаточный бюджет графовых вариантов

Все назначения проверяются после их выдачи, true I не участвует в выборе момента остановки.

| Геометрия | {" | ".join(names)} |
|---|{"---:|" * len(names)}
{first}

## Граница вклада и дальнейшая проверка

Whitening, проекционные тесты, sample splitting и fixed-confidence clustering известны. Кандидат собственного компонента: независимые направления из адаптированных кодбуков, полный covariance контрастов, знаковой граф с нулевым якорем, остановка по числу2K+1 классов, затем фиксированный I и свежий C. Это ещё не установленный приоритет или конкурентное превосходство. Спектральные контроли должны остаться в финальном сравнении.

G4 настраивает явные multi-task адаптации Yeh/LC/LUT-Q/непрерывного MSR по отдельным наблюдаемым source-validation функциям. Их результаты пока не относятся к новым confirmation-геометриям. Новая нелинейная проверка, устойчивость за пределами общего Gaussian noise, независимый перенос и расширенные конкуренты ещё требуются.
"""
    path = output / "REPORT.md"
    path.write_text(report)
    save_json(
        output / "report_manifest.json",
        {
            "report_sha256": digest(path),
            "g3_protocol_sha256": digest(args.g3 / "protocol.json"),
            "weights": weights,
        },
    )


if __name__ == "__main__":
    main()
