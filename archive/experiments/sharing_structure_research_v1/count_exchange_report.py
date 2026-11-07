"""Report G22 and cached G21 controls without repeating any source search."""

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.darts_structured_tasks_v1.evaluate import align_indices

from .count_metric_report import LABELS as OLD_LABELS
from .count_metric_report import load_cases, read_json, verify

LABELS = {
    **OLD_LABELS,
    "pair": "Pair transfers",
    "parent": "Exact parent",
    "parent_pair": "Parent + pair",
    "parent_pair_refit": "Parent + pair + C refit",
}
COLORS = {
    "qp_integer": "0.4",
    "pair": "tab:blue",
    "parent": "tab:orange",
    "parent_pair": "tab:green",
    "parent_pair_refit": "tab:red",
}


def generate(root):
    plan = read_json(root / "protocol.json")
    audit = verify(root)
    previous = Path(plan["g21_parent"])
    stages = [
        previous,
        previous / "numeric_continuation",
        previous / "numeric_continuation/face_stationarity",
    ]
    for stage in stages:
        verify(stage)
    cases = load_cases(previous, stages, read_json(previous / "protocol.json"))
    scores = {(r["job_id"], r["key"]): r for r in audit["records"]}
    new_records, oracle_cases, hashes = [], {}, {}
    for case in cases:
        job_id = case["job_id"]
        path = root / "models" / f"job_{job_id:03d}"
        meta = read_json(path.with_suffix(".json"))
        new_records.extend(meta["records"])
        with np.load(path.with_suffix(".npz")) as arrays:
            for record in meta["records"]:
                panel_id = plan["panels"].index(record["panel"])
                key = record["key"]
                case["states"][(panel_id, record["method"])] = {
                    "record": record,
                    "indices": arrays[key + "__indices"],
                    "codebooks": arrays[key + "__codebooks"],
                    "score": scores[(job_id, key)],
                }
        oracle_cases[job_id] = {}
        for base in (stages[-1], root):
            path = base / "models" / f"job_{job_id:03d}_oracle"
            oracle_meta = read_json(path.with_suffix(".json"))
            if oracle_meta["models_frozen_sha256"] != digest(
                base / "models_frozen.json"
            ):
                raise RuntimeError("Oracle freeze mismatch")
            with np.load(path.with_suffix(".npz")) as arrays:
                for record in oracle_meta["records"]:
                    oracle_cases[job_id][(record["noise"], record["method"])] = {
                        "record": record,
                        "indices": arrays[record["key"] + "__indices"],
                    }
            for suffix in (".json", ".npz"):
                item = path.with_suffix(suffix)
                hashes[str(item)] = digest(item)
    with np.load(previous / "audit_only.npz") as data:
        truth = dict(data)
    rows = []
    for panel_id, panel in enumerate(plan["panels"]):
        for variant in ("all", "local", "quartet_mix", "random"):
            for method in LABELS:
                selected = [
                    c["states"][(panel_id, method)]
                    for c in cases
                    if variant == "all" or c["job"]["variant"] == variant
                ]
                metrics = [s["score"] for s in selected]
                rows.append(
                    {
                        "variant": variant,
                        **panel,
                        "method": method,
                        "cases": len(selected),
                        "exact": sum(r["exact_I"] for r in metrics),
                        "wrong_complete": sum(
                            r["complete"] and not r["exact_I"] for r in metrics
                        ),
                        "incomplete": sum(not r["complete"] for r in metrics),
                        "median_C_max_error": float(
                            np.median([r["final_C_max_error"] for r in metrics])
                        ),
                        "median_prefix_seconds": None
                        if method == "free_ls"
                        else float(
                            np.median([s["record"]["seconds"] for s in selected])
                        ),
                        "full_available_source_labels": metrics[0][
                            "full_available_source_labels"
                        ],
                    }
                )
    oracle = []
    for sigma in (0.2, 2.0):
        for method in (
            "euclidean",
            "qp",
            "qp_integer",
            "pair",
            "parent",
            "parent_pair",
        ):
            selected = [o[(sigma, method)]["record"] for o in oracle_cases.values()]
            oracle.append(
                {
                    "noise": sigma,
                    "method": method,
                    "exact": sum(r["exact_I"] for r in selected),
                    "wrong_complete": sum(
                        r["complete"] and not r["exact_I"] for r in selected
                    ),
                    "cases": len(selected),
                }
            )
    paired = []
    for panel_id, panel in enumerate(plan["panels"]):
        pairs = [
            (
                c["states"][(panel_id, "qp_integer_refit")]["score"],
                c["states"][(panel_id, "parent_pair_refit")]["score"],
            )
            for c in cases
        ]
        paired.append(
            {
                **panel,
                "gains_over_G21_refit": sum(
                    b["exact_I"] and not a["exact_I"] for a, b in pairs
                ),
                "losses_against_G21_refit": sum(
                    a["exact_I"] and not b["exact_I"] for a, b in pairs
                ),
            }
        )
    integer = [p["integer"] for r in new_records for p in r["trace"] if "integer" in p]
    statistics = {
        "states": len(audit["records"]),
        "exact": sum(r["exact_I"] for r in audit["records"]),
        "wrong_complete": sum(
            r["complete"] and not r["exact_I"] for r in audit["records"]
        ),
        "reasons": dict(Counter(r["last_reason"] for r in audit["records"])),
        "integer_prefixes": len(integer),
        "neighbourhood_caps": sum(not r["local_stationary"] for r in integer),
        "largest_candidate_evaluations_per_prefix": max(
            r["candidates_evaluated"] for r in integer
        ),
        "total_candidates_evaluated": sum(r["candidates_evaluated"] for r in integer),
        "max_updates_per_prefix": max(r["moves"] for r in integer),
        "new_source_labels": 0,
    }
    report = root / "report"
    report.mkdir(exist_ok=False)
    for name, values in (("comparison.csv", rows), ("oracle_comparison.csv", oracle)):
        with (report / name).open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    save_json(
        report / "summary.json",
        {
            "comparisons": rows,
            "oracle": oracle,
            "paired_against_G21_refit": paired,
            "statistics": statistics,
            "source_freeze_sha256": digest(root / "models_frozen.json"),
            "oracle_hashes": hashes,
        },
    )
    plot_recovery(report, rows, oracle)
    plot_oracle(report, oracle_cases, truth, plan)
    plot_prefixes(report, cases, truth, plan)
    plot_weights(report, cases, truth, plan)
    plot_updates(report, oracle_cases)
    write_report(report, rows, oracle, paired, statistics)


def plot_recovery(report, rows, oracle):
    methods = (
        "qp_integer",
        "pair",
        "parent",
        "parent_pair",
        "qp_integer_refit",
        "parent_pair_refit",
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for ax, (sigma, budget) in zip(
        axes, ((0.2, 4096), (0.2, 16777216), (2.0, 16777216)), strict=True
    ):
        for position, method in enumerate(methods):
            row = next(
                r
                for r in rows
                if r["variant"] == "all"
                and r["noise"] == sigma
                and r["budget"] == budget
                and r["method"] == method
            )
            color = COLORS.get(method, "tab:purple")
            ax.bar(position, row["exact"], color=color)
            ax.bar(
                position,
                row["wrong_complete"],
                bottom=row["exact"],
                color=color,
                alpha=0.25,
                hatch="//",
            )
            ax.text(position, row["exact"] + 0.3, str(row["exact"]), ha="center")
            if budget == 16777216 and not method.endswith("refit"):
                marker = next(
                    r for r in oracle if r["noise"] == sigma and r["method"] == method
                )
                ax.plot(position, marker["exact"], "k_", markersize=15)
        ax.set_ylim(0, 19.2)
        ax.set_yticks([0, 6, 12, 18])
        ax.set_xticks(
            range(6),
            ("G21 unit", "Pair", "Parent", "Both", "G21 + C", "Both + C"),
            rotation=30,
        )
        ax.set_title(f"SD{sigma}, {'low' if budget == 4096 else 'high'} axis precision")
        ax.set_ylabel("Exact I / 18; hatched = wrong full I")
    fig.suptitle(
        "G22: stronger integer optimization fixes true-C diagnosis, not the primary C-estimation problem\nBlack markers = true-C controls; all primary conditions retained; opened known-bias/readout networks"
    )
    fig.savefig(report / "exchange_recovery.png", dpi=170)
    plt.close(fig)


def plot_oracle(report, cases, truth, plan):
    fig, axes = plt.subplots(2, 5, figsize=(14, 6.5), constrained_layout=True)
    for line, job_id in enumerate((5, 11)):
        expected = truth[f"job_{job_id}__truth_I"]
        images, titles = [expected], ["True signed I"]
        for method in ("qp_integer", "pair", "parent", "parent_pair"):
            state = cases[job_id][(2.0, method)]
            index = state["indices"][-1]
            _, aligned, _ = align_indices(index, expected, plan["codes"])
            images.append(aligned)
            titles.append(
                LABELS[method]
                + ("\nexact" if state["record"]["exact_I"] else "\nwrong")
            )
        for col, (value, title) in enumerate(zip(images, titles, strict=True)):
            im = axes[line, col].imshow(value, cmap="RdBu_r", vmin=-4, vmax=4)
            axes[line, col].set_title(title, fontsize=9)
            axes[line, col].set_xlabel("Hidden neuron")
            axes[line, col].set_xticks([0, 7, 15])
            axes[line, col].set_yticks([0, 7, 15])
        axes[line, 0].set_ylabel(
            f"Mixed {plan['jobs'][job_id]['family']}\nInput coordinate"
        )
    fig.colorbar(
        im, ax=axes.ravel().tolist(), shrink=0.7, label="Signed code ID (audit aligned)"
    )
    fig.suptitle(
        "G22 true-C diagnostic only, SD2: the two G21 failure cases\nPair and conditional-parent neighbourhoods are complementary; primary C estimates are not replaced"
    )
    fig.savefig(report / "exchange_oracle_indexators.png", dpi=170)
    plt.close(fig)


def plot_prefixes(report, cases, truth, plan):
    fig, axes = plt.subplots(3, 4, figsize=(17, 10), constrained_layout=True)
    for line, job_id in enumerate((0, 5, 11)):
        case = cases[job_id]
        expected = truth[f"job_{job_id}__truth_I"]
        for method, color in COLORS.items():
            state = case["states"][(4, method)]
            trace = [p for p in state["record"]["trace"] if p["accepted"]]
            for col, field in enumerate(("slice_mse_integer", "prefix_validation_mse")):
                axes[line, col].plot(
                    [p["prefix_length"] for p in trace],
                    [max(p[field], 1e-30) for p in trace],
                    marker=".",
                    color=color,
                    label=LABELS[method],
                )
            recall, accuracy = [], []
            for step, index in enumerate(state["indices"], 1):
                partial_truth = expected.copy()
                partial_truth[case["order"][step:]] = 0
                metric, _, _ = align_indices(index, partial_truth, plan["codes"])
                recall.append(metric["mask_recall"])
                accuracy.append(
                    1 - metric["categorical_errors"] / (step * plan["hidden"])
                )
            axes[line, 2].plot(
                range(1, len(recall) + 1), recall, color=color, marker="."
            )
            axes[line, 3].plot(
                range(1, len(accuracy) + 1), accuracy, color=color, marker="."
            )
        axes[line, 0].set_title(f"{case['job']['variant']} {case['job']['family']}")
        for col, label in enumerate(
            (
                "Training slice MSE",
                "Independent prefix-val MSE",
                "Mask recall (audit)",
                "Categorical accuracy (audit)",
            )
        ):
            axes[line, col].set_ylabel(label)
            axes[line, col].set_xlabel("Accepted coordinate step")
            axes[line, col].set_xticks([1, 4, 8, 12, 16])
            if col < 2:
                axes[line, col].set_yscale("log")
            else:
                axes[line, col].set_ylim(-0.03, 1.04)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(
        "G22 primary estimates, SD2 / high axis precision: measured loss and recall\nFirst local and the two preidentified diagnostic mixed networks; no teacher hyperparameter selection"
    )
    fig.savefig(report / "exchange_loss_recall.png", dpi=170)
    plt.close(fig)


def plot_weights(report, cases, truth, plan):
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), constrained_layout=True)
    for line, job_id in enumerate((0, 16)):
        case = cases[job_id]
        expected, centers = (
            truth[f"job_{job_id}__truth_I"],
            truth[f"job_{job_id}__true_C"],
        )
        images = [np.r_[0.0, centers][np.abs(expected)] * np.sign(expected)]
        titles = ["True numerical W"]
        for method in ("qp_integer", "parent_pair", "parent_pair_refit"):
            state = case["states"][(4, method)]
            index = (
                state["indices"][-1]
                if len(state["indices"])
                else np.zeros_like(expected)
            )
            _, _, columns = align_indices(index, expected, plan["codes"])
            weights = np.r_[0.0, state["record"]["final_C"]][np.abs(index)] * np.sign(
                index
            )
            images.append(weights[:, columns])
            titles.append(
                LABELS[method]
                + f"\n{len(state['indices'])}/16; "
                + (
                    "exact I"
                    if state["score"]["exact_I"]
                    else "wrong I"
                    if state["score"]["complete"]
                    else "partial only"
                )
            )
        limit = max(float(np.abs(a).max()) for a in images)
        for col, (value, title) in enumerate(zip(images, titles, strict=True)):
            im = axes[line, col].imshow(value, cmap="RdBu_r", vmin=-limit, vmax=limit)
            axes[line, col].set_title(title, fontsize=9)
            axes[line, col].set_xlabel("Hidden neuron")
            axes[line, col].set_xticks([0, 7, 15])
            axes[line, col].set_yticks([0, 7, 15])
        axes[line, 0].set_ylabel(
            f"{case['job']['variant']} {case['job']['family']}\nInput coordinate"
        )
        fig.colorbar(
            im, ax=axes[line].tolist(), shrink=0.8, label="Actual numerical weight"
        )
    fig.suptitle(
        "G22 primary actual W = raw I and learned C; only hidden columns aligned\nSD2 / high axis precision; structure correctness and numerical fitting are separate"
    )
    fig.savefig(report / "exchange_weights.png", dpi=170)
    plt.close(fig)


def plot_updates(report, cases):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, (job_id, step) in zip(axes, ((5, 2), (11, 15)), strict=True):
        for method in ("pair", "parent", "parent_pair"):
            point = cases[job_id][(2.0, method)]["record"]["trace"][step - 1]
            integer = point["integer"]
            initial = (
                point["slice_mse_integer_before_refit"] / 2 - integer["half_mse_change"]
            )
            history = integer["history"]
            ax.plot(
                [0] + [p["update"] for p in history],
                [initial]
                + [initial + p["cumulative_half_mse_change"] for p in history],
                marker="o",
                color=COLORS[method],
                label=LABELS[method],
            )
        old = cases[job_id][(2.0, "qp_integer")]["record"]["trace"][step - 1]
        ax.axhline(
            old["slice_mse_integer_before_refit"] / 2,
            color="0.4",
            ls="--",
            label="G21 unit endpoint only",
        )
        ax.set_yscale("log")
        ax.set_title(f"Mixed {8803 if job_id == 5 else 8806}, prefix {step}/16")
        ax.set_xlabel("Actual accepted joint-count update")
        ax.set_ylabel("Training halfMSE, exact quadratic history")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "G22 true-C diagnosis: saved integer update histories\nLoss reconstructed from actual exact quadratic deltas; old unit method has no invented trajectory"
    )
    fig.savefig(report / "exchange_integer_steps.png", dpi=170)
    plt.close(fig)


def write_report(report, rows, oracle, paired, statistics):
    table = "\n".join(
        f"|{r['noise']}|{r['budget']}|{r['method']}|{r['exact']}/18|{r['wrong_complete']}|{r['incomplete']}|{r['median_prefix_seconds']:.2f}|"
        if r["median_prefix_seconds"] is not None
        else f"|{r['noise']}|{r['budget']}|{r['method']}|{r['exact']}/18|{r['wrong_complete']}|{r['incomplete']}|—|"
        for r in rows
        if r["variant"] == "all"
    )
    oracle_table = "\n".join(
        f"|{r['noise']}|{r['method']}|{r['exact']}/18|{r['wrong_complete']}|"
        for r in oracle
    )
    pairs = "\n".join(
        f"|{r['noise']}|{r['budget']}|{r['gains_over_G21_refit']}|{r['losses_against_G21_refit']}|"
        for r in paired
    )
    (
        report / "REPORT.md"
    ).write_text(f"""# G22: согласованные дискретные изменения индексатора

5 ноября 2026. **Остаточные known-C ловушки G21 устранены сочетанием exact-parent и pair ходов:18/18 при SD2 против16/18 у одиночных ходов. Каждый отдельный механизм даёт17/18. Основной метод с оценённым C остаётся12/18; при SD0.2/low stronger search + C-refit ухудшает8/18 до4/18. Поэтому это подтверждённое исправление solver failure, но не общее улучшение извлечения структуры.**

## Диагностика и теория до запуска

G21 показал два конкретных local integer barriers: teacher counts имеют меньшие observed/validation loss, но одиночные изменения ухудшают objective. THEORY §29 до запуска вывел pair-delta = delta1+delta2+d1ᵀGd2 и conditional-parent delta по полному g/G. Перебираются все допустимые пары unit moves или все compositions counts одного parent; teacher assignment/counts не сообщаются.

Exact parent solve условный, при fixed других блоках. Максимум735471распределение приH16/K4, GPUchunk16384, таблицы зависят только отH/K. Greedy лучший ход из всех parents/pairs, строгий decrease1e−12, cap512updates. Это не global integer MLE: при cross-block G могут остаться более дальние локальные ловушки. Ни block descent, ни pair updates не объявлены новыми: [CDQuant §3.1/Algorithms1–2](https://arxiv.org/html/2406.17542v3) уже рассматривает greedy и совместные дискретные обновления по полной квадратичной форме. Отличие исследуемой постановки — counts наблюдаемых ReLU atoms без заданной плотнойW; приоритет именно её пока не установлен.

## Полная primary таблица

18открытых G21 underlyingI/C, все5panels и4новыхварианта,360states whole-frozen до нового teacher-I audit. Все старые G20/G21controls прочитаны, не повторялись. Cinit/slices/means/order и prefix-validation совпали поhash/массивам. Новыхsource labels0; precision/метод не выбирались поteacher. Время — median seconds только prefix phase; axis-init уже выполнен ранее, старыйfreeLS не имел отдельного такого таймера. Это не matched-compute native comparison.

|SD|Axis precision budget|Метод|ТочныйI|НеверныйполныйI|НетполногоI|Median prefix seconds|
|---:|---:|---|---:|---:|---:|---:|
{table}

Итог новых primary states: {statistics["exact"]}точных, {statistics["wrong_complete"]}неверныхполных, {statistics["states"] - statistics["exact"] - statistics["wrong_complete"]}отказов. Полная допустимая структура не является сертификатом истины. QP numerical gap/primal прежние, допускинеослаблялись. Rank/остальныепричины: {statistics["reasons"]}. Integer neighbourhood caps:{statistics["neighbourhood_caps"]}; проверено{statistics["integer_prefixes"]}prefix solves, max{statistics["max_updates_per_prefix"]}acceptedupdates/prefix. Всего{statistics["total_candidates_evaluated"]:,}candidate evaluations; max{statistics["largest_candidate_evaluations_per_prefix"]:,}/prefix. Это дополнительныйcompute, не новые наблюдения.

Парное сравнение parent_pair_refit против G21unit_refit:

|SD|Axis budget|Исправлено ранее неверныхI|Потеряно ранее точныхI|
|---:|---:|---:|---:|
{pairs}

## True-C diagnosis отдельно от основного метода

После wholefreeze/source-audit запущены108known-C diagnostic states на тех же noisy slices. Никаких повтора поprecision, C-refit, teacher правок primary или отбора только двух известных failurecases.

|SD|Decoder с истиннымC|ТочныйI|НеверныйполныйI|
|---:|---|---:|---:|
{oracle_table}

Два механизма дополняют друг друга: pair решает межродительский совместный ход, exact-parent — многокоординатное перераспределение внутри parent. Успех18/18 относится к true-C diagnosis и не заменяет primary12/18. Численный convex QP решён; остаток с оценённымC нельзя объяснять только незавершённым QP или ровно двумя known-C integer ловушками. Сильнее минимизировать критерий с ошибочной dictionary недостаточно: нужны работа с nuisance C, несколько source-задач/независимый выбор общей структуры и полноценные сильные контроли.

## Фигуры

![Primary и oracle](exchange_recovery.png)
![Истинные и найденныеI двух diagnostic cases](exchange_oracle_indexators.png)
![Measured train/validation loss и recall](exchange_loss_recall.png)
![Настоящие численныеW](exchange_weights.png)
![Сохранённые discrete-update trajectories](exchange_integer_steps.png)

Никаких выдуманных SGD/Adam траекторий. Count-loss восстанавливается из фактически сохранённых точных quadratic deltas; старый unit control имеет только endpoint. Prefix losses/recall измерены на реальных coordinate steps. NumericalW изrawI/C с одним hidden-column alignment; categorical group renaming используется лишь дляI audit. ДиагностическиеI не смешаны сprimaryW.

## Границы и продолжение

Known commonb=.15/readout1/sqrt(H)/offset0, D=H16/K4, однафункция/I, открытые геометрии. Highcost275,582,549,920labels/условие, low234,979,360; эти затраты наследуются полностью, хотя новыхlabelsнет. Gaussian means точно симулированы без materialization всехlabels. Все6свободныхV1000/1/2/3/6/7 использованы, чужие4/5не затрагивались. Software tests не писали/не запускали; минимальныйRuff/provenance/math/visual.

Следующий шаг должен выйти из opened known-bias solver ablations: независимые I и разные source-C с одним общимI, сильные nativeLC/Yeh/LUT-Q/denseGN на тех же observations, перенос толькоI со свежимиC. При выборе whole-I кандидата по нескольким задачам нужен отдельный meta-selection split после всех numerical C-fits; validation, уже использованный дляfitC, не может служить независимым доказательством. Цель остаётся активна; собственной конкурентной победы/новизны этот этап не доказывает.

Код count_exchanges.py/count_exchange_search.py/count_exchange_report.py; protocol/theory/snapshots/freeze/audit раздельны. comparison.csv/oracle_comparison.csv/summary.json содержат все conditions, pairing, реальныйcompute иoracle hashes.
""")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    generate(args.root)


if __name__ == "__main__":
    main()
