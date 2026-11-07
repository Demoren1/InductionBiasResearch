"""Report G21 frozen fits, numerical continuations and separate C-oracles."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.darts_structured_tasks_v1.evaluate import align_indices

LABELS = {
    "free_ls": "Free LS + rounding (G20)",
    "euclidean": "Feasible Euclidean",
    "qp": "Function QP",
    "qp_integer": "QP + integer moves",
    "qp_integer_refit": "QP + moves + C refit",
}
COLORS = dict(zip(LABELS, ("0.4", "tab:blue", "tab:orange", "tab:green", "tab:red")))
VARIANTS = ("all", "local", "quartet_mix", "random")


def read_json(path):
    return json.loads(path.read_text())


def verify(root):
    frozen = read_json(root / "models_frozen.json")
    for relative, expected in frozen["sha256"].items():
        if digest(root / relative) != expected:
            raise RuntimeError(f"Frozen source changed: {root / relative}")
    audit = read_json(root / "audit.json")
    if audit["models_frozen_sha256"] != digest(root / "models_frozen.json"):
        raise RuntimeError("Audit belongs to a different freeze")
    return audit


def load_cases(root, stages, plan):
    parent = Path(plan["parent"])
    parent_audit = verify(parent)
    parent_scores = {(r["job_id"], r["key"]): r for r in parent_audit["records"]}
    scores = {
        (r["job_id"], r["key"]): r
        for r in read_json(stages[-1] / "audit.json")["records"]
    }
    cases = []
    for job_id, job in enumerate(plan["jobs"]):
        records, arrays = {}, {}
        for stage in stages:
            path = stage / "models" / f"job_{job_id:03d}"
            records.update(
                {r["key"]: r for r in read_json(path.with_suffix(".json"))["records"]}
            )
            with np.load(path.with_suffix(".npz")) as data:
                arrays.update(dict(data))
        source = parent / "models" / f"job_{job_id:03d}"
        previous = {
            r["key"]: r for r in read_json(source.with_suffix(".json"))["records"]
        }
        with np.load(source.with_suffix(".npz")) as data:
            baseline_arrays = dict(data)
        merged = {}
        for panel_id, panel in enumerate(plan["panels"]):
            reference = records[f"panel_{panel_id}__euclidean"]
            key = reference["parent_reference_key"]
            baseline = previous[key]
            index = baseline_arrays[key + "__indices"]
            merged[(panel_id, "free_ls")] = {
                "record": {**baseline, "final_C": baseline["codebook"]},
                "indices": index,
                "codebooks": np.tile(baseline["codebook"], (len(index), 1)),
                "score": {
                    **parent_scores[(job_id, key)],
                    "method": "free_ls",
                    "panel": panel,
                    "final_C_max_error": parent_scores[(job_id, key)]["code_max_error"],
                    "full_available_source_labels": reference[
                        "full_available_source_labels"
                    ],
                    "last_reason": baseline["trace"][-1]["reason"],
                },
            }
            for method in plan["methods"]:
                current = f"panel_{panel_id}__{method}"
                merged[(panel_id, method)] = {
                    "record": records[current],
                    "indices": arrays[current + "__indices"],
                    "codebooks": arrays[current + "__codebooks"],
                    "score": scores[(job_id, current)],
                }
        cases.append(
            {
                "job_id": job_id,
                "job": job,
                "order": baseline_arrays["order"],
                "states": merged,
            }
        )
    return cases


def comparisons(cases, plan):
    rows = []
    for panel_id, panel in enumerate(plan["panels"]):
        for variant in VARIANTS:
            selected = [
                c for c in cases if variant == "all" or c["job"]["variant"] == variant
            ]
            for method in LABELS:
                states = [c["states"][(panel_id, method)] for c in selected]
                scores = [s["score"] for s in states]
                rows.append(
                    {
                        "variant": variant,
                        **panel,
                        "method": method,
                        "cases": len(scores),
                        "exact": sum(r["exact_I"] for r in scores),
                        "complete": sum(r["complete"] for r in scores),
                        "wrong_complete": sum(
                            r["complete"] and not r["exact_I"] for r in scores
                        ),
                        "rank_failures": sum(
                            r["last_reason"] == "rank_deficient_dictionary"
                            for r in scores
                        ),
                        "median_C_max_error": float(
                            np.median([r["final_C_max_error"] for r in scores])
                        ),
                        "median_prefix_seconds": None
                        if method == "free_ls"
                        else float(np.median([s["record"]["seconds"] for s in states])),
                        "full_available_source_labels": scores[0][
                            "full_available_source_labels"
                        ],
                    }
                )
    return rows


def oracle_comparisons(root, final, plan):
    records = []
    hashes = {}
    for job_id, job in enumerate(plan["jobs"]):
        path = final / "models" / f"job_{job_id:03d}_oracle.json"
        for item in (path, path.with_suffix(".npz")):
            hashes[str(item.relative_to(root))] = digest(item)
        meta = read_json(path)
        if meta["models_frozen_sha256"] != digest(final / "models_frozen.json"):
            raise RuntimeError("Oracle does not match final source freeze")
        records.extend({**r, "job": job} for r in meta["records"])
        previous = read_json(
            Path(plan["g19_parent"]) / "models" / f"job_{job_id:03d}_audit.json"
        )
        for row in previous["records"]:
            if (
                row["condition"]["noise"] not in (0.2, 2.0)
                or row["condition"]["replicates"] != 4096
            ):
                continue
            old = row["oracle_codes_observed"]
            records.append(
                {
                    **old,
                    "job": job,
                    "noise": row["condition"]["noise"],
                    "method": "free_ls",
                    "exact_I": old["complete"] and old.get("categorical_errors") == 0,
                }
            )
    save_json(root / "report" / "oracle_artifact_hashes.json", hashes)
    rows = []
    for sigma in (0.2, 2.0):
        for variant in VARIANTS:
            for method in tuple(LABELS)[:-1]:
                selected = [
                    r
                    for r in records
                    if r["noise"] == sigma
                    and r["method"] == method
                    and (variant == "all" or r["job"]["variant"] == variant)
                ]
                rows.append(
                    {
                        "noise": sigma,
                        "variant": variant,
                        "method": method,
                        "cases": len(selected),
                        "exact": sum(r["exact_I"] for r in selected),
                        "complete": sum(r["complete"] for r in selected),
                        "wrong_complete": sum(
                            r["complete"] and not r["exact_I"] for r in selected
                        ),
                    }
                )
    return rows


def plot_recovery(report, rows, oracle):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    panels = ((0.2, 4096), (0.2, 16777216), (2.0, 16777216))
    names = (
        "SD 0.2, low axis precision",
        "SD 0.2, high axis precision",
        "SD 2, high axis precision",
    )
    for ax, (sigma, budget), name in zip(axes, panels, names, strict=True):
        selected = [
            r
            for r in rows
            if r["variant"] == "all" and r["noise"] == sigma and r["budget"] == budget
        ]
        for x, row in enumerate(selected):
            ax.bar(x, row["exact"], color=COLORS[row["method"]])
            ax.bar(
                x,
                row["wrong_complete"],
                bottom=row["exact"],
                color=COLORS[row["method"]],
                alpha=0.25,
                hatch="//",
            )
            ax.text(x, row["exact"] + 0.3, str(row["exact"]), ha="center", fontsize=10)
            if budget == 16777216 and row["method"] != "qp_integer_refit":
                control = next(
                    r
                    for r in oracle
                    if r["variant"] == "all"
                    and r["noise"] == sigma
                    and r["method"] == row["method"]
                )
                ax.plot(x, control["exact"], "k_", markersize=17)
        ax.set_title(name)
        ax.set_ylim(0, 19.3)
        ax.set_yticks([0, 6, 12, 18])
        ax.set_xticks(
            range(5),
            ("Free LS", "Euclidean", "QP", "+ moves", "+ C refit"),
            rotation=25,
        )
        ax.set_ylabel("Exact I / 18 (solid); wrong full I (hatched)")
    fig.suptitle(
        "G21: full function metric and integer moves; all final QP gaps certified\nBlack markers: true-C diagnostics; opened networks, known bias/readout, huge inherited axis budget"
    )
    fig.savefig(report / "count_recovery.png", dpi=170)
    plt.close(fig)


def chosen_cases(cases):
    jobs = (
        {"family": 8801, "variant": "local"},
        {"family": 8801, "variant": "quartet_mix"},
        {"family": 8905, "variant": "random"},
    )
    return [next(c for c in cases if c["job"] == job) for job in jobs]


def plot_prefixes(report, cases, truth, plan):
    chosen = chosen_cases(cases)
    fig, axes = plt.subplots(3, 4, figsize=(17, 11), constrained_layout=True)
    for line, case in enumerate(chosen):
        expected = truth[f"job_{case['job_id']}__truth_I"]
        for method, label in LABELS.items():
            state = case["states"][(4, method)]
            trace = [p for p in state["record"]["trace"] if p["accepted"]]
            for col, field in enumerate(("slice_mse_integer", "prefix_validation_mse")):
                points = [p for p in trace if field in p]
                axes[line, col].plot(
                    [p["prefix_length"] for p in points],
                    [max(p[field], 1e-30) for p in points],
                    marker=".",
                    color=COLORS[method],
                    label=label,
                )
            recall, accuracy = [], []
            for step, index in enumerate(state["indices"], 1):
                target = expected.copy()
                target[case["order"][step:]] = 0
                metric, _, _ = align_indices(index, target, plan["codes"])
                recall.append(metric["mask_recall"])
                accuracy.append(
                    1 - metric["categorical_errors"] / (step * plan["hidden"])
                )
            axes[line, 2].plot(
                range(1, len(recall) + 1), recall, marker=".", color=COLORS[method]
            )
            axes[line, 3].plot(
                range(1, len(accuracy) + 1), accuracy, marker=".", color=COLORS[method]
            )
        for col, ylabel in enumerate(
            (
                "Training slice MSE",
                "Independent prefix-val MSE",
                "Mask recall (audit)",
                "Categorical accuracy (audit)",
            )
        ):
            ax = axes[line, col]
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Accepted coordinate step (not SGD epochs)")
            ax.set_xticks([1, 4, 8, 12, 16])
            if col < 2:
                ax.set_yscale("log")
                ax.axhline(4 / 4096, color="gray", ls="--", lw=0.8)
            else:
                ax.set_ylim(-0.03, 1.04)
        axes[line, 0].set_title(f"{case['job']['variant']} {case['job']['family']}")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(
        "G21 measured losses/recall, SD2 / high axis precision\nFixed first local/mixed family and random8905; incomplete curves are abstentions; baseline has no new prefix-val trajectory"
    )
    fig.savefig(report / "count_loss_recall.png", dpi=170)
    plt.close(fig)


def plot_heatmaps(report, cases, truth, plan, numerical=False):
    selected = [c for c in cases if c["job"]["variant"] == "random"]
    if numerical:
        selected = [chosen_cases(cases)[0], selected[2], selected[4]]
    fig, axes = plt.subplots(
        len(selected), 6, figsize=(17, 3.0 * len(selected)), constrained_layout=True
    )
    for line, case in enumerate(selected):
        expected = truth[f"job_{case['job_id']}__truth_I"]
        true_C = truth[f"job_{case['job_id']}__true_C"]
        actual = np.r_[0.0, true_C][np.abs(expected)] * np.sign(expected)
        images, titles = (
            [actual if numerical else expected],
            ["True numerical W" if numerical else "True signed I"],
        )
        for method, label in LABELS.items():
            state = case["states"][(4, method)]
            indices = state["indices"]
            index = indices[-1] if len(indices) else np.zeros_like(expected)
            _, aligned, columns = align_indices(index, expected, plan["codes"])
            values = np.r_[0.0, state["record"]["final_C"]][np.abs(index)] * np.sign(
                index
            )
            images.append(values[:, columns] if numerical else aligned)
            titles.append(
                label.replace(" (G20)", "")
                + f"\n{len(indices)}/16; "
                + (
                    "exact"
                    if state["score"]["exact_I"]
                    else "wrong"
                    if state["score"]["complete"]
                    else "partial only"
                )
            )
        limit = (
            max(float(np.abs(a).max()) for a in images) if numerical else plan["codes"]
        )
        for col, (data, title) in enumerate(zip(images, titles, strict=True)):
            im = axes[line, col].imshow(data, cmap="RdBu_r", vmin=-limit, vmax=limit)
            axes[line, col].set_title(title, fontsize=9)
            axes[line, col].set_xticks([0, 7, 15])
            axes[line, col].set_yticks([0, 7, 15])
            axes[line, col].set_xlabel("Hidden neuron")
        axes[line, 0].set_ylabel(
            f"{case['job']['variant']} {case['job']['family']}\nInput coordinate"
        )
        if numerical:
            fig.colorbar(
                im, ax=axes[line].tolist(), shrink=0.7, label="Actual numerical weight"
            )
    if not numerical:
        fig.colorbar(
            im,
            ax=axes.ravel().tolist(),
            shrink=0.5,
            label="Signed code ID; 0 absent/unprocessed",
        )
    fig.suptitle(
        "G21 actual numerical weights; only hidden columns aligned"
        if numerical
        else "G21 true and recovered I: all six random networks\nSD2 / high axis precision; names aligned only for audit; wrong full I shown explicitly"
    )
    fig.savefig(
        report / ("count_weights.png" if numerical else "count_indexators.png"), dpi=170
    )
    plt.close(fig)


def plot_refit(report, cases):
    chosen = chosen_cases(cases)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=True)
    for line, panel_id in enumerate((1, 2)):
        for ax, case in zip(axes[line], chosen, strict=True):
            state = case["states"][(panel_id, "qp_integer_refit")]
            accepted = [p for p in state["record"]["trace"] if p["accepted"]]
            title = f"{case['job']['variant']} {case['job']['family']}"
            if not accepted:
                ax.text(
                    0.5,
                    0.5,
                    "No accepted prefix; no GN steps recorded\n"
                    + state["record"]["trace"][-1]["reason"],
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                )
            else:
                point = accepted[-1]
                fitted = point["code_refit"]
                for field, label in (
                    ("train_mse", "Train"),
                    ("validation_mse", "Independent validation"),
                ):
                    ax.plot(
                        [p["step"] for p in fitted["records"]],
                        [max(p[field], 1e-30) for p in fitted["records"]],
                        marker="o",
                        label=label,
                    )
                selected = fitted["records"][fitted["selected"]]
                ax.plot(
                    selected["step"],
                    max(selected["validation_mse"], 1e-30),
                    "k*",
                    markersize=13,
                    label="Selected checkpoint",
                )
                title += f"\nLast accepted prefix {point['prefix_length']}/16"
                ax.set_yscale("log")
            ax.set_title(title)
            ax.set_xlabel("Actual GN step within this fixed prefix")
            ax.set_ylabel(("Low" if line == 0 else "High") + " axis precision; MSE")
    axes[1, 0].legend(fontsize=8)
    fig.suptitle(
        "G21 C-refit: actual optimizer steps, SD0.2 / low and high axis precision\nFixed three networks; empty panel means no recorded fit; validation selects checkpoint"
    )
    fig.savefig(report / "count_refit_steps.png", dpi=170)
    plt.close(fig)


def write_report(report, rows, oracle, stages):
    aggregate = [r for r in rows if r["variant"] == "all"]
    table = "\n".join(
        f"|{r['noise']}|{r['budget']}|{r['method']}|{r['exact']}/18|{r['wrong_complete']}|{18 - r['complete']}|{r['median_C_max_error']:.2e}|"
        for r in aggregate
    )
    oracle_table = "\n".join(
        f"|{r['noise']}|{r['method']}|{r['exact']}/18|{r['wrong_complete']}|"
        for r in oracle
        if r["variant"] == "all"
    )
    costs = "\n".join(
        f"|{r['noise']}|{r['budget']}|{r['full_available_source_labels']:,}|"
        for r in aggregate
        if r["method"] == "euclidean"
    )
    numerical = "\n".join(
        f"|{r['name']}|{r['complete']}|{r['exact']}|{r['wrong_complete']}|{r['qp_failures']}|{r['rank_failures']}|"
        for r in stages
    )
    diagnosis = ""
    if (report / "failure_diagnosis.json").exists():
        failures = read_json(report / "failure_diagnosis.json")["records"]
        entries = "\n".join(
            f"|{r['job']['family']} mixed|{r['first_numeric_wrong_prefix']}|{r['observed_half_mse_found_minus_truth']:.2e}|{r['validation_mse_found_minus_truth']:.2e}|{r['integer_metadata']['minimum_next_change']:.2e}|"
            for r in failures
        )
        diagnosis = f"""## Измеренный остаточный сбой integer decoder

THEORY §28.8 до диагностического расчёта зафиксировал сравнение уже сохранённых ошибочных known-C структур, без нового fit и без правокI. В обоих случаях выбранные counts проигрывают teacher по observed и independent-validation loss, хотя ни один допустимый одиночный перенос не улучшает их. Это локальный integer барьер, не недостаточность teacher-C или доказанное noise misranking.

|Сеть|Первый неверный numeric prefix|Observed halfMSE found−true|Validation MSE found−true|Лучший одиночный ход: изменение halfMSE|
|---|---:|---:|---:|---:|
{entries}

У mixed8803 переход прямо к teacher требует4одноединичных переносов, у mixed8806 —2; это не доказательство минимального радиуса выхода из локальной ловушки. Максимальная ошибка тождества direct difference = gᵀd+.5dᵀGd равна{max(r["identity_absolute_error"] for r in failures):.2e}. Полная квадратичная метрика рассчитана правильно; следующий проверяемый механизм — согласованные дискретные ходы/более сильный integer solve, с новой проспективной абляцией, а не подстановка teacher. Диагноз не меняет первичные16/18/12/18. Артефакт failure_diagnosis.json, код count_failure_diagnosis.py.

"""
    (
        report / "REPORT.md"
    ).write_text(f"""# G21: допустимые counts по полной ошибке функции

5 ноября 2026. **Полная метрика функции и целочисленные ходы исправляют значительную часть LS-rounding failure. При SD0.2/highaxis точный I:14/18 у frozen G20 freeLS,15/18 у Euclidean,16/18 у QP,18/18 у QP+integer. При SD2/highaxis:0,0,3,12 из18. Это отдельный парный алгоритмический эффект при одинаковых C и slices; конкурентное превосходство нового метода не установлено.**

## Почему проверен этот критерий

G19/G20 отделили неверный C от неустойчивого unrestricted LS количеств. THEORY.md §28 записан до fit: при фиксированных C/префиксах count→function линейно, L(n)=||Fn−y0||²/(2N). Полная G=FᵀF/N связывает разные продолжения, а gᵀd+.5dᵀGd даёт точное изменение loss. Это известная task-metric integer LS, а не новая общая формула или доказанная новизна. Возможное отличие — функциональная prefix-факторизация без исходной плотной W; её приоритет ещё требует проверки.

Euclidean вариант точно проектирует LS-counts в целый simplex каждого родителя. QP сначала минимизирует continuous task-loss с nonnegativity/capacity, затем использует ту же integer проекцию. Integer moves переносят одну единицу внутри родителя по точному task-loss с полным cross-gradient. Это лишь локальный integer optimum. C-refit фиксирует уже hard-prefix и уточняет4числа по предыдущим training slices; independent prefix-validation выбирает numerical checkpoint. C не сортируется относительно принятых labels.

## Полная таблица: точный I и ошибочные выдачи

|SD|Coarse-equivalent axis budget|Метод|Точный I|Неверный полный I|Нет полного I|Median max C error|
|---:|---:|---|---:|---:|---:|---:|
{table}

Все regimes сохранены; HPO/выбор метода/бюджета по teacher не делали. Полный выход не является сертификатом правильности: при SD2/low все выданные полные I неверны. В отличие от прежнего отказа freeLS, feasible projection может уверенно закончить ошибочную структуру. Это не скрыто и не считается успехом.

## Численный контроль до teacher

Все18исходныхjobs/360состояний заморожены. Первичный SLSQP часто сообщал success при FWgap выше неизменного1e−8:121остановка. До teacher-аудита THEORY §28.6 зарегистрировал продолжение только этих состояний с feasible active-set QP. Оно дало67полных выходов,2rank failure и52QP failure. THEORY §28.7 до второго продолжения исправил проверку стационарности на грани: reduced gradient вместо чрезвычайно малого шага при condition1e8–1e9. Все52достигли прежнего полного FWgap/feasibility и завершили prefix loop. Это стандартная численная поправка, не новый структурный механизм. Принятые предыдущие I/C/history не пересчитывались; raw root/child/grandchild snapshots и freezes сохранены. Полные структуры открыты только после всех source-freezes.

|Состояние cohort|Полных I|Точных I|Неверных полных|QP failures|Rank failures|
|---|---:|---:|---:|---:|---:|
{numerical}

Первичные незавершённые QP нельзя выдавать за качество optimum. У итоговых принятых QP проверены прежние absolute FWgap≤1e−8 и primal≤1e−7. Эти numerical certificates относятся к continuous QP, а не к истинному I или global integer optimum. Время каждой numerical continuation и прежнего fit сохранено; больше вычислений раскрыто.

## Факторный oracle: истинный C, те же noisy slices

Только после wholefreeze, один раз наjob/noise, без повторов по axis precision и без C-refit. Frozen G19 known-C freeLS прочитан, не перезапущен.

|SD|Count decoder, true-C diagnosis|Точный I|Неверный полный I|
|---:|---|---:|---:|
{oracle_table}

При SD2 trueC/Euclidean даёт0/18, QP4/18, integer16/18 против0/18 у старого freeLS. Следовательно mismatch метрики — реальная причина, которую не исправляет одно знание C. Остаток16/18 против18/18 требует диагностики локальной дискретной оптимизации/различимости. Разница primary12/18 и oracle16/18 показывает сохраняющийся вклад ошибки C, но это не разложение независимых причин: prefix пути различаются. C-refit особенно полезен при SD0.2/low (3/18→8/18), а при SD2/high не увеличивает12/18. Не объявлять любое уточнение C универсально полезным.

{diagnosis}## Данные, затраты и ограничения

18ранее открытых G19/G20 underlying I/C:6local,6quartet_mix,6random. D=H16,K4, known commonb=.15, readout1/sqrt(H),offset0, одна функция наI. Initial C всегда G20 continuous_dp_start/grid1025 с обоими сохранёнными precision. Все G19 slice means/order совпали по hash/массивам; noisy r4096. Новых axis наблюдений нет. На каждый16prefix добавлены512independent validation points, noisy r4096:33,554,432labels. Baseline frozen и не использует их для ретроспективногоHPO; его старые slices/C и результаты прочитаны. Таблица ниже считает весь доступный inherited axis+slice+новый prefix-validation panel; отдельный8192IIDvalidation показан лишь как report, не selection.

|SD|Axis budget|Все доступные source labels, включая prefix-validation|
|---:|---:|---:|
{costs}

Высокий режим по-прежнему около275.58миллиардаlabels/условие, низкий около234.98миллиона. Gaussian r-label means симулируются точно без материализацииrnoise, но статистическая стоимость считается полностью. Это не доказательство эффективности или победы над native LC/Yeh/LUT-Q. Все6свободныхV1000/1/2/3/6/7 использованы, чужие4/5не затрагивались; все fit/audit/oracle workers terminal. Software tests не писали/не запускали; только необходимая provenance/math проверка, Ruff и просмотр фигур.

## Фигуры и воспроизводимость

![Точный I, ошибки и true-C контроль](count_recovery.png)
![Измеренные training/validation loss и recall](count_loss_recall.png)
![Все шесть random: истинный и выбранный I](count_indexators.png)
![Настоящие численные W](count_weights.png)
![Фактические GN-шаги C-refit](count_refit_steps.png)

Loss/recall по реальным обработанным координатам, а не выдуманным SGD эпохам. GN-рисунок использует сохранённые фактические train/validation точки на одном фиксированном prefix. I можно переназвать для аудита; numerical W рассчитана из исходных I/C, выровнены только hidden columns. Неполная W не объявлена полной моделью.

Код count_projection.py/count_metric.py/count_qp_active.py/count_numeric_continuation.py, отдельный count_metric_report.py. Исторические executed_code_snapshot, protocol.json, theory_at_registration.md и models_frozen.json у каждого numerical stage сохраняют фактически выполненные версии. comparison.csv/oracle_comparison.csv/summary.json и oracle_artifact_hashes.json содержат машинные результаты и provenance.

Большая цель остаётся активна: нужны commonI разных source-C, новые независимые геометрии, свободныеbias/readout, misspecifiednoise, strong matched-query native конкуренты, перенос толькоI со свежимиC и аудит приоритета. Из QP/локального integer search не следует автоматически новая general structure-learning работа.
""")


def generate(root):
    stages = [
        root,
        root / "numeric_continuation",
        root / "numeric_continuation/face_stationarity",
    ]
    plan = read_json(root / "protocol.json")
    audits = [verify(stage) for stage in stages]
    for stage, parent in zip(stages[1:], stages, strict=False):
        protocol = read_json(stage / "protocol.json")
        if protocol["primary_freeze_sha256"] != digest(parent / "models_frozen.json"):
            raise RuntimeError("Numerical continuation parent mismatch")
    cases = load_cases(root, stages, plan)
    report = root / "report"
    report.mkdir(exist_ok=False)
    rows = comparisons(cases, plan)
    oracle = oracle_comparisons(root, stages[-1], plan)
    numerical = []
    for stage, audit in zip(stages, audits, strict=True):
        scores = audit["records"]
        numerical.append(
            {
                "name": "initial" if stage == root else str(stage.relative_to(root)),
                "complete": sum(r["complete"] for r in scores),
                "exact": sum(r["exact_I"] for r in scores),
                "wrong_complete": sum(
                    r["complete"] and not r["exact_I"] for r in scores
                ),
                "qp_failures": sum(
                    r["last_reason"] == "qp_not_converged" for r in scores
                ),
                "rank_failures": sum(
                    r["last_reason"] == "rank_deficient_dictionary" for r in scores
                ),
            }
        )
    for name, values in (("comparison.csv", rows), ("oracle_comparison.csv", oracle)):
        with (report / name).open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    reasons = Counter(
        s["score"]["last_reason"]
        for c in cases
        for (_, m), s in c["states"].items()
        if m != "free_ls"
    )
    save_json(
        report / "summary.json",
        {
            "comparisons": rows,
            "oracle": oracle,
            "numerical_stages": numerical,
            "final_reasons": dict(reasons),
            "source_freezes": {
                str(s): digest(s / "models_frozen.json") for s in stages
            },
            "scope": "Opened known-bias/readout one-function/I diagnostic; no new native competitors or multi-source fresh-C transfer; full outputs can be wrong.",
        },
    )
    with np.load(root / "audit_only.npz") as data:
        truth = dict(data)
    plot_recovery(report, rows, oracle)
    plot_prefixes(report, cases, truth, plan)
    plot_heatmaps(report, cases, truth, plan)
    plot_heatmaps(report, cases, truth, plan, numerical=True)
    plot_refit(report, cases)
    write_report(report, rows, oracle, numerical)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    generate(args.root)


if __name__ == "__main__":
    main()
