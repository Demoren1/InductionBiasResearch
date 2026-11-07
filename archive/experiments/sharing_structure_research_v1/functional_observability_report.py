"""Report already-frozen G18 observability diagnostics without rerunning fits."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.darts_structured_tasks_v1.data import digest, save_json

from .functional_observability import teacher

VARIANTS = {"local": "Local modules", "quartet_mix": "Orthogonally mixed modules"}
COLORS = {0.2: "tab:blue", 1.0: "tab:orange", 2.0: "tab:red"}


def generate(root):
    plan = json.loads((root / "protocol.json").read_text())
    audit = json.loads((root / "audit.json").read_text())
    frozen_path = root / "models_frozen.json"
    frozen = json.loads(frozen_path.read_text())
    if audit["cohort_frozen_sha256"] != digest(frozen_path):
        raise RuntimeError("G18 audit and frozen cohort differ")
    for relative, expected in frozen["sha256"].items():
        if digest(root / relative) != expected:
            raise RuntimeError("Frozen G18 observations changed")
    report = root / "report"
    report.mkdir(exist_ok=False)
    rows = []
    for variant in VARIANTS:
        for noise in plan["noise"]:
            for n in plan["observations"]:
                selected = [
                    row
                    for row in audit["records"]
                    if row["job"]["variant"] == variant
                    and row["noise"] == noise
                    and row["n"] == n
                ]
                rows.append(
                    {
                        "variant": variant,
                        "noise": noise,
                        "n": n,
                        "label_budget": selected[0]["label_budget"],
                        "cases": len(selected),
                        "exact_graph": sum(row["exact_graph"] for row in selected),
                        "exact_partition": sum(
                            row["exact_partition"] for row in selected
                        ),
                        "false_edges": sum(row["false_edges"] for row in selected),
                        "missing_edges": sum(row["missing_edges"] for row in selected),
                        "mean_edge_recall": float(
                            np.mean([row["edge_recall"] for row in selected])
                        ),
                    }
                )
    with (report / "comparison.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "comparisons": rows,
        "population_oracle": audit["population_oracle"],
        "models_frozen_sha256": digest(frozen_path),
        "audit_sha256": digest(root / "audit.json"),
        "scope": "Six new coordinate permutations, eight source-C functions each, paired local/mixed networks and three dependent Gaussian noise conditions. Known functional-interaction graph, not a learned weight indexator, new method or matched competitor comparison.",
        "curves": "Measured graph recall/exact recovery versus actual label budget; no optimizer, training loss, validation loss or chosen weight-I exists in this diagnostic.",
    }
    save_json(report / "summary.json", summary)
    plot_curves(report, rows)
    plot_ambiguity(report, root, plan)
    plot_graphs(report, root, plan)
    write_report(report, plan, audit, rows)


def plot_curves(report, rows):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for line, variant in enumerate(VARIANTS):
        for noise, color in COLORS.items():
            selected = [
                row
                for row in rows
                if row["variant"] == variant and row["noise"] == noise
            ]
            x = [row["label_budget"] for row in selected]
            metrics = [
                [row["mean_edge_recall"] for row in selected],
                [row["exact_graph"] / row["cases"] for row in selected],
                [row["exact_partition"] / row["cases"] for row in selected],
            ]
            for col, values in enumerate(metrics):
                axes[line, col].plot(
                    x, values, marker="o", color=color, label=f"Noise SD {noise}"
                )
        for col, metric in enumerate(
            ("Mean true-edge recall", "Exact input graph / 6", "Exact partition / 6")
        ):
            axes[line, col].set_xscale("log")
            axes[line, col].set_ylim(-0.04, 1.04)
            axes[line, col].set_ylabel(metric)
            axes[line, col].set_xlabel("Noisy labels: 2 x T x N x 137")
            axes[line, col].set_title(VARIANTS[variant])
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(
        "G18: known functional-ANOVA contrast; 6 geometries, paired noise conditions\n"
        "Input graph recovery; complete weight indexators are not inferred"
    )
    fig.savefig(report / "functional_recovery.png", dpi=180)
    plt.close(fig)


def plot_ambiguity(report, root, plan):
    family = plan["families"][0]
    with np.load(root / "models" / f"family_{family}_oracle_only.npz") as data:
        arrays = dict(data)
    local, _, _, _ = teacher(family, "local", plan, "cpu")
    mixed, _, _, _ = teacher(family, "quartet_mix", plan, "cpu")
    pairs = [
        (arrays["local_I"], arrays["mixed_I"]),
        (local[0].numpy(), mixed[0].numpy()),
        (arrays["local_m2"][0], arrays["mixed_m2"][0]),
        (
            arrays["local_gradient_anisotropic"][0],
            arrays["mixed_gradient_anisotropic"][0],
        ),
    ]
    titles = [
        "True signed I (32 versus 128 active)",
        "True numerical W, first source function",
        "Population E[Y(XX^T - Id)]",
        "Population Cov(-YX) - E[Y^2] Id",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for col, (values, title) in enumerate(zip(pairs, titles, strict=True)):
        limit = max(np.max(np.abs(value)) for value in values)
        for line, value in enumerate(values):
            im = axes[line, col].imshow(value, cmap="RdBu_r", vmin=-limit, vmax=limit)
            axes[line, col].set_title(title, fontsize=10)
            axes[line, col].set_xticks([0, 7, 15])
            axes[line, col].set_yticks([0, 7, 15])
            if col == 0:
                axes[line, col].set_ylabel(("Local", "Quartet mix")[line] + "\nInput")
            axes[line, col].set_xlabel("Hidden neuron" if col < 2 else "Input")
        fig.colorbar(im, ax=axes[:, col].tolist(), shrink=0.7)
    fig.suptitle(
        f"G18 oracle construction, family {family}: different I and functions, equal low moments\n"
        "Truth/population illustrations after whole-cohort freeze; no recovered weight-I"
    )
    fig.savefig(report / "moment_ambiguity.png", dpi=180)
    plt.close(fig)
    np.savez_compressed(
        report / "illustration_oracle_only.npz",
        **arrays,
        local_W=local.numpy(),
        mixed_W=mixed.numpy(),
    )


def plot_graphs(report, root, plan):
    family = plan["families"][0]
    conditions = [(0.2, 8192), (1.0, 131072), (2.0, 131072)]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    for line, variant in enumerate(VARIANTS):
        job = plan["jobs"].index({"family": family, "variant": variant})
        _, index, _, _ = teacher(family, variant, plan, "cpu")
        active = (index != 0).astype(int)
        truth = active @ active.T > 0
        np.fill_diagonal(truth, False)
        with np.load(root / "models" / f"job_{job:03d}.npz") as data:
            graphs = [truth] + [data[f"noise{s}_n{n}__graph"] for s, n in conditions]
        titles = ["True input interaction graph"] + [
            f"Selected edges: SD={s}, N={n:,}" for s, n in conditions
        ]
        for col, (graph, title) in enumerate(zip(graphs, titles, strict=True)):
            axes[line, col].imshow(graph, cmap="Blues", vmin=0, vmax=1)
            axes[line, col].set_title(title, fontsize=9)
            axes[line, col].set_xticks([0, 7, 15])
            axes[line, col].set_yticks([0, 7, 15])
            axes[line, col].set_xlabel("Input coordinate")
            if col == 0:
                axes[line, col].set_ylabel(VARIANTS[variant] + "\nInput coordinate")
    fig.suptitle(
        f"G18, first registered family {family}: observed graphs, not weight masks\n"
        "Missed edges remain under noise; no false edges observed across all 180 saved looks"
    )
    fig.savefig(report / "functional_graphs.png", dpi=180)
    plt.close(fig)


def write_report(report, plan, audit, rows):
    oracle = audit["population_oracle"]
    moment_names = list(oracle[0]["max_moment_absolute_errors"])
    moment_rows = "\n".join(
        "|{}|{:.2e}|".format(
            name, max(row["max_moment_absolute_errors"][name] for row in oracle)
        )
        for name in moment_names
    )
    recovery_rows = "\n".join(
        "|{}|{}|{:,}|{:,}|{}/6|{}/6|{:.2f}%|{}|".format(
            row["variant"],
            row["noise"],
            row["n"],
            row["label_budget"],
            row["exact_graph"],
            row["exact_partition"],
            row["mean_edge_recall"] * 100,
            row["false_edges"],
        )
        for row in rows
    )
    function_errors = [row["same_x_function_mse"] for row in oracle]
    coupled_error = max(row["orthogonal_coupled_max_error"] for row in oracle)
    text = f"""# G18: какие наблюдения различают структуру

5 ноября 2026. **Диагностика завершена; новый полный алгоритм извлечения I не получен.**
После сильного LC/dense-контроля G17 проверяем достаточность статистик до разработки новой схемы. THEORY.md §25 записан до протокола и запусков. Все 12 моделей и их 180 budget/noise-состояний заморожены до oracle-аудита. Код: functional_observability.py, functional_moments.py; протокол, snapshot, freeze и отдельный audit находятся на уровень выше.

## Что установлено

Для двух разных signed-индексаторов с четырьмя обучаемыми кодами существуют **точно одинаковые Gaussian input-моменты степени не выше двух**, включая covariance линейного sample-gradient при нулевой инициализации. Это не недостаток числа примеров: бесконечный объём таких статистик не определяет I в данном классе. Нельзя переносить вывод на произвольные градиенты нелинейной сети, полные данные или higher-order признаки.

Конструкция: 8 повторённых двухмерных модулей, 16 ReLU, common bias 0.15, положительный sum-readout /sqrt(16). Исходный W=Id8 tensor B(C) имеет 32 активных веса. Q4=J4/2-Id4 ортогональна и сохраняет ones; смешивание двух четвёрок даёт W'=VW=WV, 128 активных весов и значения из 0/±C/2. Shape и число кодов совпадают, маски и функции различны. На шести новых перестановках координат и восьми fresh Gaussian source-C общий I внутри каждой сети неизменен.

Обоснование: для любой функции psi совместных source-outputs перестановочная симметрия задаёт E[psi(Y)X]=ones8 tensor v и E[psi(Y)XX^T]=Id8 tensor A+J8 tensor D. V сохраняет оба выражения. В частности, совпадают E[Y], E[YX], E[Y(XX^T-Id)] и E[Y²XX^T]. Больше разных C само по себе эту неоднозначность не устраняет. Это классический symmetry/commutant аргумент; приоритет именно finite-code специализации не установлен.

|Population-величина|Максимальная численная разница двух сетей, 48 source-функций|
|---|---:|
{moment_rows}

Label-square формула вычислена детерминированной Gaussian quadrature; остаток до 3.78e-10 — её численная точность, а не измеренная статистическая разница. Теоретическое равенство следует из симметрии. Третий Hermite-момент различается: относительная Frobenius-разница 1.2247 во всех 48 функциях. MSE двух функций на одних и тех же новых Gaussian x составляет {min(function_errors):.4f}–{max(function_errors):.4f} по геометриям. После известного ортогонального преобразования x максимальная разница {coupled_error:.2e}. Эти числа получены oracle-аудитом после freeze, не служили learner-признаками или выбором параметров.

![Разные истинные I и W, одинаковые низкие моменты](moment_ambiguity.png)

## Дополнительный наблюдаемый сигнал

Использован **известный** mixed functional-ANOVA/Sobol контраст: delta_ij f=f(Xij)-f(Xi)-f(Xj)+f(X), где заменённые координаты берутся из независимого Z. E[delta²]=4 sum_(u contains i,j) V_u. Он равен нулю между аддитивными компонентами, а положительная энергия выявляет взаимодействие. Источники: [Owen 2013](https://arxiv.org/abs/1205.1774), [Borgonovo, Plischke, Rabitti 2022](https://onlinelibrary.wiley.com/doi/10.1111/sjos.12560); [Hooker 2004](https://doi.org/10.1145/1014052.1014122) ранее обнаруживал additive black-box structure. Наша реализация не объявлена неизменённым авторским кодом или новым принципом.

Доступ к active/pick-freeze query — отдельное условие; такие panels нельзя получить из произвольного готового IID банка. Для D16 нужны Q=137 query/base (base+singles+pairs), T8 source-функций и две независимые шумные метки/query. Noise SD 0.2/1/2; все пять зарегистрированных N сохраняются. Входы и unit noise парные между сетями/noise-режимами; это зависимые условия, шесть единиц геометрии, не 36 независимых подтверждений. Learner не знает истинные компоненты, активные позиции, численные C или population-моменты.

Разность двух меток независимо оценивает sigma², средняя используется в контрасте. При delta f=0 статистика F=[sum(delta ybar)²/(2TN)]/sigmahat² имеет точное F(TN,TNQ) распределение при iid homoscedastic Gaussian measurement noise. Alpha 0.05 делится на 120 пар × 5 looks **внутри family/network/noise run**. Гарантия относится только к ложным рёбрам. Она не сертифицирует отсутствие пропусков, компоненты или полный weight-I; общего benchmark-level покрытия не заявляем. Для heteroscedastic/heavy-tail noise эта гарантия не доказана.

## Результат графа входных взаимодействий

|Сеть|Noise SD|N base|Шумных меток|Точный граф|Точное разбиение входов|Mean edge recall|Ложных рёбер, сумма|
|---|---:|---:|---:|---:|---:|---:|---:|
{recovery_rows}

При SD0.2 исходный граф точен 6/6 уже при N2048 (4 489 216 меток), mixed-граф 6/6 при N8192 (17 956 864 метки). При SD1 на максимальном бюджете исходный граф 4/6, mixed 0/6, но его разбиение входов уже 5/6: для связности не нужны все рёбра. При SD2 соответственно 1/6 и 0/6; мощности явно недостаточно. Во всех 180 сохранённых состояниях ложных рёбер не наблюдалось, но это зависимые состояния и не доказательство покрытия вне заявленной модели.

![Кривые measured recall и точного графа по числу меток](functional_recovery.png)

![Истинные и выбранные графы, первая заранее заданная геометрия](functional_graphs.png)

## Стоимость и границы вывода

Каждая строка максимального N131072 стоит 287 309 824 виртуальные noisy labels; raw queries/labels генерировались поточно и не сохранялись. Три noise-режима повторно используют clean responses и unit noise, но стоимость одного режима явно включает обе метки на каждом query. Время и настройки каждого из 12 GPU-job находятся в models/job_*.json; использованы свободные GPU 0,1,2,3,6,7. Чужие 4/5 не затрагивались. Новые software tests не создавались и не запускались; выполнены необходимый математический/provenance аудит, минимальный Ruff и просмотр фигур.

У этого диагноза нет оптимизатора, train/validation loss или обученных численных W. Поэтому показаны настоящие measured graph-recall curves по label budget, а не выдуманная SGD-кривая. На heatmap отдельно подписаны истинные W/I и выбранный **input interaction graph**. Это не найденная маска весов и не полный шэринг. Ни дообучение кодбука, ни перенос на новый C в G18 не выполнялись.

Простой rectangle-F дорог и слаб при большом шуме. Известные bilinear Sobol estimators требуют D+1 query/base; их variance/cost нужно проверить на том же label budget до конкурентной заявки. Также нужны native LC/Yeh/LUT-Q на тех же panels, переход от функциональных взаимодействий к полному I без истинного шаблона, независимые геометрии, misspecified noise и transfer только I. Граф не разрешает автоматически равенства отдельных весов или ориентации нейронов. Возможный functional-I compiler пока гипотеза, не выполненный метод. Общая цель остаётся active; превосходство и новизна не установлены.
"""
    (report / "REPORT.md").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    generate(args.root)


if __name__ == "__main__":
    main()
