"""Freeze all predictions before scoring the prospective comparison."""

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from experiments.paper_structure_benchmark_v1.reporting.evaluate import (
    adjusted_rand,
    align_index,
    write_csv,
)
from experiments.sharing_structure_research_v1.blind_contract import (
    canonical_index,
    digest,
    save_json,
)


METHODS = ("current", "diversity", "theory", "dense")
LABELS = {
    "current": "G34",
    "diversity": "G34 diversity",
    "theory": "G34 theory",
    "dense": "Dense",
}
PROJECT = Path(__file__).resolve().parents[2]


def relative_reduction(old, new):
    if old < 1e-12:
        return None
    return 100 * (old - new) / old


def expected(base):
    root = base / "confirmation"
    protocol = json.loads((root / "protocol.json").read_text())
    paths = []
    for case in protocol["cases"]:
        for condition in protocol["conditions"]:
            for method in METHODS:
                if method == "dense" and condition:
                    continue
                paths.append(
                    root
                    / "targets"
                    / f"{case['name']}_c{condition}"
                    / method
                    / "frozen_predictions.npz"
                )
    return paths


def freeze(base):
    output = base / "report"
    output.mkdir(exist_ok=True)
    manifest_path = output / "frozen_results.json"
    if manifest_path.exists():
        raise RuntimeError("Global prediction freeze already exists")
    registration = json.loads((base / "registered_protocol.json").read_text())
    if digest(base / "PROTOCOL.md") != registration["protocol_sha256"]:
        raise RuntimeError("Registered protocol changed")
    for name in registration.get(
        "plan_names", ("baseline", "development", "variants", "target")
    ):
        state = json.loads((base / f"{name}_plan.finished.json").read_text())
        if state["completed"] != state["total"] or state["failures"]:
            raise RuntimeError(f"Incomplete {name} plan")
    for name, value in registration["core_sha256"].items():
        if digest(PROJECT / name) != value:
            raise RuntimeError(f"Reference G34 changed: {name}")
    files = set(expected(base))
    numerical_rules = set()
    for prediction in expected(base):
        numerical = json.loads(prediction.with_name("registered.json").read_text())
        numerical_rules.add(
            (
                numerical["adapter_sha256"],
                numerical["numeric_sha256"],
                numerical["adam_steps"],
                numerical["restarts"],
                numerical["gauss_newton_starts"],
                numerical["gauss_newton_steps"],
                tuple(numerical["budgets"]),
            )
        )
        metric = json.loads(prediction.with_name("metrics.json").read_text())
        if metric["status"] != "complete":
            raise RuntimeError(f"Incomplete target fit: {prediction}")
        if metric["predictions_sha256"] != digest(prediction):
            raise RuntimeError(f"Prediction hash mismatch: {prediction}")
    if len(numerical_rules) != 1:
        raise RuntimeError("Target numerical rules differ between methods")
    root = base / "confirmation"
    method_registration = json.loads((root / "theory_registration.json").read_text())
    for name, value in method_registration["code_sha256"].items():
        if digest(Path(__file__).with_name(name)) != value:
            raise RuntimeError(f"Registered method changed: {name}")
    for owner in ("fits", "source", "targets"):
        files.update((root / owner).rglob("*.npz"))
        files.update((root / owner).rglob("*.json"))
    files.update((root / "public").rglob("*.npz"))
    files.update((root / "public").glob("*.json"))
    files.update((root / "metrics").glob("*.json"))
    files.update((root / "theory_code_snapshot").glob("*.py"))
    files.add(root / "theory_registration.json")
    files.add(base / "registered_protocol.json")
    files.add(base / "PROTOCOL.md")
    commitments = root / "data_commitments.json"
    if commitments.exists():
        for name, value in json.loads(commitments.read_text())[
            "public_sha256"
        ].items():
            if digest(root / name) != value:
                raise RuntimeError(f"Committed observation data changed: {name}")
        files.add(commitments)
    evaluation = base / "evaluation_registration.json"
    if evaluation.exists():
        files.add(evaluation)
        files.update((base / "evaluation_code_snapshot").glob("*.py"))
    receipts = []
    for directory in (root / "receipts", root / "targets"):
        for path in directory.rglob("*.json"):
            if directory != root / "receipts" and "receipt" not in path.name:
                continue
            record = json.loads(path.read_text())
            if record.get("violations"):
                raise RuntimeError(f"Observation boundary violation: {path}")
            files.add(path)
            receipts.append(str(path.relative_to(base)))
    save_json(
        manifest_path,
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "files_sha256": {
                str(path.relative_to(base)): digest(path) for path in sorted(files)
            },
            "prediction_count": len(expected(base)),
            "receipts": receipts,
            "violations": 0,
            "target_query_answers_opened": False,
            "structural_truth_opened": False,
        },
    )
    print(json.dumps({"frozen": len(files), "predictions": len(expected(base))}))


def score(base, *, render_report=True):
    output, root = base / "report", base / "confirmation"
    manifest = json.loads((output / "frozen_results.json").read_text())
    for name, value in manifest["files_sha256"].items():
        if digest(base / name) != value:
            raise RuntimeError(f"Frozen file changed: {name}")
    protocol = json.loads((root / "protocol.json").read_text())
    generation = json.loads((root / "private" / "generation.json").read_text())
    case_metadata = {row["name"]: row for row in generation["cases"]}
    commitments = root / "data_commitments.json"
    if commitments.exists():
        for name, value in json.loads(commitments.read_text())[
            "private_sha256"
        ].items():
            if digest(root / "private" / name) != value:
                raise RuntimeError(f"Committed teacher/target data changed: {name}")
    transfer, recovery, costs = [], [], []
    index_audit = {}
    for sizes in protocol["cases"]:
        case, codes = sizes["name"], sizes["codes"]
        kind = case_metadata[case]["dataset"]
        dataset = {
            "pattern": "Pattern",
            "deepsets": "DeepSets",
        }.get(kind, kind)
        with np.load(root / "private" / f"{case}_targets.npz") as archive:
            answer = np.stack([archive[f"query_y_{t}"] for t in range(8)])
        truth = np.load(root / "private" / f"{case}.npz")["index"]
        canonical = canonical_index(truth, codes)
        for condition in protocol["conditions"]:
            key = f"{case}_c{condition}"
            target = root / "targets" / key
            structures = dict(np.load(target / "structures.npz"))
            dense_state = np.load(root / "source" / f"{key}_t0" / "learned.npz")[
                "dense_parameters"
            ]
            index_audit[key] = {
                "truth": truth.tolist(),
                "dense_source_t0_weight": dense_state[
                    : sizes["inputs"] * sizes["hidden"]
                ]
                .reshape(sizes["inputs"], sizes["hidden"])
                .tolist(),
            }
            for method in METHODS:
                model = (
                    root
                    / "targets"
                    / (f"{case}_c0" if method == "dense" else key)
                    / method
                )
                prediction = np.load(model / "frozen_predictions.npz")
                for budget in (64, 256, 1024):
                    for stage, field in (
                        ("Adam", f"adam_prediction_{budget}"),
                        ("Adam+GN", f"prediction_{budget}"),
                    ):
                        error = np.mean((prediction[field] - answer) ** 2, axis=(1, 2))
                        variance = np.maximum(np.var(answer, axis=(1, 2)), 1e-12)
                        for task, value in enumerate(error):
                            transfer.append(
                                {
                                    "dataset": dataset,
                                    "case": case,
                                    "condition": condition,
                                    "method": method,
                                    "stage": stage,
                                    "support": budget,
                                    "task": task,
                                    "mse": float(value),
                                    "nmse": float(value / variance[task]),
                                }
                            )
                metric = json.loads((model / "metrics.json").read_text())
                if method == "dense":
                    continue
                fitted = root / "fits" / key
                common_cost = sum(
                    json.loads((root / "metrics" / f"{key}_{phase}.json").read_text())[
                        "wall_seconds"
                    ]
                    for phase in ("source", "seeds", "proposals")
                )
                source_cost = common_cost
                if method == "current":
                    source_cost += sum(
                        json.loads(
                            (root / "metrics" / f"{key}_{phase}.json").read_text()
                        )["wall_seconds"]
                        for phase in ("fit", "select")
                    )
                    bank = np.load(fitted / "joint_fitted.npz")["indices"]
                    selected = json.loads((fitted / "joint_selected.json").read_text())
                    meta_mse = selected["methods"]["joint_functional"]["meta_mse"]
                    origin = "original"
                else:
                    source_cost += json.loads(
                        (root / "metrics" / f"{key}_theory_proposals.json").read_text()
                    )["wall_seconds"]
                    source_cost += sum(
                        json.loads(
                            (
                                root / "metrics" / f"{key}_{method}_{phase}.json"
                            ).read_text()
                        )["wall_seconds"]
                        for phase in ("fit", "select")
                    )
                    bank = np.load(fitted / method / "fitted.npz")["indices"]
                    selected = json.loads(
                        (fitted / method / "selected.json").read_text()
                    )
                    meta_mse = selected["meta_mse"]
                    origin = selected["origin"]["family"]
                aligned, metrics = align_index(structures[method], truth, codes)
                index_audit[key][method] = aligned.tolist()
                common = (truth != 0) & (aligned != 0)
                metrics["sharing_ari"] = adjusted_rand(
                    np.abs(truth[common]), np.abs(aligned[common])
                )
                metrics["true_in_refitted_bank"] = any(
                    np.array_equal(candidate, canonical) for candidate in bank
                )
                proposal_path = fitted / (
                    "joint_bank.npz"
                    if method in ("current", "diversity")
                    else "theory_pool.npz"
                )
                proposals = np.load(proposal_path)["indices"]
                metrics["true_in_proposal_pool"] = any(
                    np.array_equal(candidate, canonical) for candidate in proposals
                )
                metrics["refitted_bank_size"] = len(bank)
                recovery.append(
                    {
                        "dataset": dataset,
                        "case": case,
                        "condition": condition,
                        "method": method,
                        "origin": origin,
                        "source_meta_mse": meta_mse,
                        **metrics,
                    }
                )
                costs.append(
                    {
                        "dataset": dataset,
                        "case": case,
                        "condition": condition,
                        "method": method,
                        "source_seconds": source_cost,
                        "target_seconds_all_budgets": metric["wall_seconds"],
                        "target_peak_mib": metric["cuda_peak_allocated_bytes"] / 2**20,
                    }
                )
    write_csv(output / "transfer.csv", transfer)
    write_csv(output / "recovery.csv", recovery)
    write_csv(output / "costs.csv", costs)
    save_json(output / "index_audit.json", index_audit)
    save_json(
        output / "scoring.json",
        {
            "scored_utc": datetime.now(UTC).isoformat(),
            "transfer": transfer,
            "recovery": recovery,
            "costs": costs,
        },
    )
    if render_report:
        render(base, transfer, recovery, costs)


def render(base, transfer, recovery, costs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output, root = base / "report", base / "confirmation"
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    groups = [(d, c) for d in ("Pattern", "DeepSets") for c in (0, 1)]
    colors = dict(zip(METHODS, ("#444444", "#d28e24", "#206db0", "#8888aa")))
    primary = [
        row for row in transfer if row["stage"] == "Adam+GN" and row["support"] == 1024
    ]

    def values(dataset, condition, method, budget=1024, stage="Adam+GN"):
        return np.asarray(
            [
                row["mse"]
                for row in transfer
                if row["dataset"] == dataset
                and row["condition"] == condition
                and row["method"] == method
                and row["support"] == budget
                and row["stage"] == stage
            ]
        )

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (dataset, condition) in zip(axes.flat, groups):
        means = [values(dataset, condition, method).mean() for method in METHODS]
        ax.bar(
            range(len(METHODS)),
            np.maximum(means, 1e-16),
            color=[colors[m] for m in METHODS],
        )
        ax.set_xticks(range(len(METHODS)), [LABELS[m] for m in METHODS], rotation=15)
        ax.set_yscale("log")
        ax.set_title(f"{dataset}, source noise {0.1 * condition:g}")
        ax.set_ylabel("Clean target MSE, N=1024")
    fig.tight_layout()
    fig.savefig(plots / "target_quality.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (dataset, condition) in zip(axes.flat, groups):
        cases = sorted({r["case"] for r in primary if r["dataset"] == dataset})
        for method in METHODS:
            curves = []
            for case in cases:
                folder = (
                    root
                    / "targets"
                    / (f"{case}_c0" if method == "dense" else f"{case}_c{condition}")
                    / method
                )
                curve = np.load(folder / "curves.npz")
                train = curve["train_history_1024"].mean(axis=(1, 2))
                validation = curve["validation_history_1024"].mean(axis=(1, 2))
                curves.append((train, validation))
                steps = curve["steps_1024"]
            train, validation = np.mean(curves, axis=0)
            ax.plot(steps, train, color=colors[method], label=LABELS[method])
            ax.plot(
                steps,
                validation,
                color=colors[method],
                linestyle="--",
                alpha=0.7,
            )
        ax.set_yscale("log")
        ax.set_title(f"{dataset}, source noise {0.1 * condition:g}")
        ax.set_xlabel("Adam step")
        ax.set_ylabel("Normalized MSE")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "target_loss.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (dataset, condition) in zip(axes.flat, groups):
        cases = sorted({r["case"] for r in primary if r["dataset"] == dataset})
        for method in METHODS[:3]:
            train_histories, selection_histories = [], []
            for case in cases:
                folder = root / "fits" / f"{case}_c{condition}"
                path = (
                    folder / "joint_fitted.json"
                    if method == "current"
                    else folder / method / "fitted.json"
                )
                for chunk in json.loads(path.read_text())["history"]:
                    train_histories.append(np.asarray(chunk["train"]))
                    selection_histories.append(np.asarray(chunk["validation"]))
            train = np.concatenate(train_histories, axis=1).mean(1)
            selection = np.concatenate(selection_histories, axis=1).mean(1)
            steps = np.arange(len(train))
            ax.plot(steps, train, color=colors[method], label=LABELS[method])
            ax.plot(
                steps,
                selection,
                color=colors[method],
                linestyle="--",
                alpha=0.7,
            )
        ax.set_yscale("log")
        ax.set_title(f"{dataset}, source noise {0.1 * condition:g}")
        ax.set_xlabel("Source Gauss–Newton iteration")
        ax.set_ylabel("Mean source train/selection MSE")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "source_loss.png", dpi=170)
    plt.close(fig)
    audit = json.loads((output / "index_audit.json").read_text())
    fig, axes = plt.subplots(2, 5, figsize=(13, 6))
    for row, case in enumerate(("case_00", "case_03")):
        record = audit[f"{case}_c1"]
        fields = (
            "truth",
            "current",
            "diversity",
            "theory",
            "dense_source_t0_weight",
        )
        for col, field in enumerate(fields):
            value = np.asarray(record[field])
            ax = axes[row, col]
            maximum = np.abs(value).max() if col == 4 else (2 if row == 0 else 4)
            ax.imshow(
                value,
                cmap="coolwarm",
                vmin=-maximum,
                vmax=maximum,
                interpolation="nearest",
                aspect="auto",
            )
            ax.set_title(("Truth", "G34", "Diversity", "Theory", "Dense source W")[col])
            ax.set_xlabel("Hidden neuron")
            if col == 0:
                ax.set_ylabel("Pattern" if row == 0 else "DeepSets")
    fig.tight_layout()
    fig.savefig(plots / "indexators.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (dataset, condition) in zip(axes.flat, groups):
        methods = METHODS[:3]
        recall, precision = [
            [
                np.mean(
                    [
                        r[field]
                        for r in recovery
                        if r["dataset"] == dataset
                        and r["condition"] == condition
                        and r["method"] == method
                    ]
                )
                for method in methods
            ]
            for field in ("mask_recall", "mask_precision")
        ]
        ax.bar(
            np.arange(3) - 0.18,
            precision,
            width=0.36,
            color=[colors[m] for m in methods],
            alpha=0.45,
            label="Precision",
        )
        ax.bar(
            np.arange(3) + 0.18,
            recall,
            width=0.36,
            color=[colors[m] for m in methods],
            label="Recall",
        )
        ax.set_xticks(range(3), [LABELS[m] for m in methods], rotation=15)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{dataset}, source noise {0.1 * condition:g}")
        ax.set_ylabel("Selected support precision / recall")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "structural_recall.png", dpi=170)
    plt.close(fig)
    lines = [
        "# Проверка улучшения G34 на основе теории",
        "",
        "8 ноября 2026 года.",
        "",
        "Проверка на новых synthetic-семействах; параметры метода зафиксированы "
        "после source-only development и до confirmation selection/target scoring. "
        "Переносится только индексатор, все числа на targets обучаются заново.",
        "",
        "## Качество на новых задачах",
        "",
        "Основная метрика — clean MSE после Adam+GN при support1024 и "
        "numerical selection256. В каждой ячейке три независимые source/numerical "
        "репликации текущего генератора, "
        "по восемь target-задач; noise относится только к source observations. "
        "Для двух source-noise условий используются одни и те же targets.",
        "",
        "| Семья | Source noise | G34 | Diversity | Theory | Dense | Изменение Theory |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    summary = []
    for dataset, condition in groups:
        means = {m: float(values(dataset, condition, m).mean()) for m in METHODS}
        change = relative_reduction(means["current"], means["theory"])
        display = "—" if change is None else f"{change:+.2f}%"
        lines.append(
            f"| {dataset} | {0.1 * condition:g} | {means['current']:.3g} | "
            f"{means['diversity']:.3g} | {means['theory']:.3g} | "
            f"{means['dense']:.3g} | {display} |"
        )
        summary.append(
            {
                "dataset": dataset,
                "condition": condition,
                **means,
                "theory_improvement_percent": change,
            }
        )
    lines += [
        "",
        "## Дополнительные размеры support",
        "",
        "Вторичные метрики после того же Adam+GN; primary endpoint остаётся N1024.",
        "",
        "| Семья | Source noise | Support | G34 | Theory | Изменение Theory |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset, condition in groups:
        for budget in (64, 256):
            old = float(values(dataset, condition, "current", budget).mean())
            new = float(values(dataset, condition, "theory", budget).mean())
            change = relative_reduction(old, new)
            display = "—" if change is None else f"{change:+.2f}%"
            lines.append(
                f"| {dataset} | {0.1 * condition:g} | {budget} | {old:.3g} | "
                f"{new:.3g} | {display} |"
            )
    lines += [
        "",
        "Положительный процент означает снижение MSE. Общая средняя по двум "
        "семьям не используется для выбора победителя: масштабы выходов различаются.",
        "При исходной MSE ниже 1e-12 относительное изменение не приводится: "
        "около машинного нуля оно неинформативно. Абсолютные MSE сохранены.",
        "",
        f"![Качество]({plots / 'target_quality.png'})",
        "",
        "## Парное сравнение по независимым семействам",
        "",
        "| Семья | Case | Noise | G34 | Theory | Изменение | Победы/равенства/поражения по задачам |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset, condition in groups:
        cases = sorted({r["case"] for r in primary if r["dataset"] == dataset})
        for case in cases:
            rows = [
                r for r in primary if r["case"] == case and r["condition"] == condition
            ]
            old = np.asarray([r["mse"] for r in rows if r["method"] == "current"])
            new = np.asarray([r["mse"] for r in rows if r["method"] == "theory"])
            change = relative_reduction(float(old.mean()), float(new.mean()))
            display = "—" if change is None else f"{change:+.2f}%"
            wins, losses = (
                int((new < old - 1e-12).sum()),
                int((new > old + 1e-12).sum()),
            )
            lines.append(
                f"| {dataset} | {case} | {0.1 * condition:g} | {old.mean():.3g} | "
                f"{new.mean():.3g} | {display} | {wins}/{8 - wins - losses}/{losses} |"
            )
    lines += [
        "",
        "Три репликации текущего генератора не покрывают произвольные графы. Восемь задач "
        "внутри семейства разделяют извлечённую структуру; они не используются "
        "как восемь независимых подтверждений её восстановления.",
        "",
        "## Восстановление и покрытие",
        "",
        "| Семья | Noise | Метод | Exact выбранного I | True I в refit-банке | Precision маски | Recall маски |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for dataset, condition in groups:
        for method in METHODS[:3]:
            rows = [
                r
                for r in recovery
                if r["dataset"] == dataset
                and r["condition"] == condition
                and r["method"] == method
            ]
            lines.append(
                f"| {dataset} | {0.1 * condition:g} | {LABELS[method]} | "
                f"{sum(r['exact'] for r in rows)}/3 | "
                f"{sum(r['true_in_refitted_bank'] for r in rows)}/3 | "
                f"{100 * np.mean([r['mask_precision'] for r in rows]):.2f}% | "
                f"{100 * np.mean([r['mask_recall'] for r in rows]):.2f}% |"
            )
    lines += [
        "",
        "True I открыт только после freeze предсказаний. Его наличие в банке "
        "и выбор по source loss — отдельные показатели; exact recovery "
        "не гарантирует наилучшую численную адаптацию.",
        "",
        f"![Structural recall]({plots / 'structural_recall.png'})",
        "",
        "## Обучение и стоимость",
        "",
        "Train/selection curves ниже — измеренные значения, усреднённые по "
        "всем18 Adam starts и задачам, а не loss только итогового выбранного старта. "
        "Ось X — шаги; selection показана пунктиром. GN входит в итоговое качество.",
        "",
        f"![Train и selection loss]({plots / 'target_loss.png'})",
        "",
        "Source refit-кривые усредняют все кандидаты и старты. Они показывают "
        "фактические численные итерации; это не loss только выбранной структуры.",
        "",
        f"![Source loss]({plots / 'source_loss.png'})",
        "",
        "| Семья | Noise | Метод | Source/search с | Target с, все бюджеты/8 задач | Peak CUDA MiB |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for dataset, condition in groups:
        for method in METHODS[:3]:
            rows = [
                r
                for r in costs
                if r["dataset"] == dataset
                and r["condition"] == condition
                and r["method"] == method
            ]
            lines.append(
                f"| {dataset} | {0.1 * condition:g} | {LABELS[method]} | "
                f"{np.mean([r['source_seconds'] for r in rows]):.2f} | "
                f"{np.mean([r['target_seconds_all_budgets'] for r in rows]):.2f} | "
                f"{max(r['target_peak_mib'] for r in rows):.2f} |"
            )
    lines += [
        "",
        "Source/search включает общий dense solver, anchors и исходный pool. "
        "Для новых вариантов включена полная измеренная стоимость нового pool; "
        "diversity использует только его исходную часть, поэтому это верхний "
        "учёт общей стадии, а не отдельный профиль standalone diversity. "
        "Бюджет одинаков по32 новым refit, но wall time не уравнен. "
        "Target seconds относятся к packed throughput и включают profiler overhead.",
        "",
        "Разработка использовала готовые source fits предыдущего benchmark. "
        "Её новые расходы сохранены отдельно в development/metrics, включая "
        "повтор построения proposals после оптимизации вычисления пространств. "
        "Эти development-расходы не включены в среднюю стоимость одного "
        "confirmation-семейства в таблице.",
        "",
        "## Что реализовано и ограничения",
        "",
        f"Основание изменений — [теоретический отчёт]("
        f"{PROJECT / 'IDENTIFIABILITY_AND_BANK_DESIGN_2026-11-07.md'}): "
        "условие повторяющихся пар, положительные масштабы ReLU, знаковое "
        "разнообразие и сохранение остаточной неоднозначности банка.",
        "",
        "Для первого семейства каждого вида показаны noisy-source структуры "
        "после аудита. Индексаторы выровнены к истине только для отображения; "
        "Dense W — настоящие веса первого source, в исходном порядке нейронов.",
        "",
        f"![Индексаторы и dense weights]({plots / 'indexators.png'})",
        "",
        "- Повторяющиеся пары логарифмических отношений соединяют локальные "
        "группы; противоречащие локальному разбиению объединения запрещены.",
        "- Дополнительная калибровка логарифмов допускает только положительные "
        "масштабы; знаковые профили дают отдельные предложения.",
        "- Support-гипотезы используют подтверждение несколькими source-задачами. "
        "Разные знаковые продолжения внутри одного пространства сохраняются.",
        "- Пороги и uncertainty являются proposal-эвристиками; достаточные "
        "теоретические границы ошибок не сертифицированы.",
        "- Ветвь отношений строит ограниченное число жадных продолжений "
        "частичных групп. Полного перебора, требуемого общей гарантией "
        "покрытия, нет.",
        "- Поиск по длинным циклам и разделителям графа остаётся дальнейшим шагом.",
        "- Omniglot и MNIST в эту проверку не входят. Из результата нельзя "
        "заключать об улучшении G34 на произвольных реальных данных.",
        "",
        "CSV с полной точностью: "
        f"[transfer]({output / 'transfer.csv'}), "
        f"[recovery]({output / 'recovery.csv'}), "
        f"[costs]({output / 'costs.csv'}). "
        f"[Протокол]({base / 'PROTOCOL.md'}), "
        f"[регистрация метода]({root / 'theory_registration.json'}), "
        f"[freeze]({output / 'frozen_results.json'}).",
    ]
    report = PROJECT / "G34_THEORY_EXPERIMENT_REPORT_2026-11-08.md"
    interpretation = output / "interpretation.md"
    if interpretation.exists():
        lines[4:4] = [interpretation.read_text().strip(), ""]
    report.write_text("\n".join(lines) + "\n")
    save_json(output / "summary.json", {"primary": summary, "report": str(report)})
    artifact = PROJECT / "artifacts" / "g34_theory_2026-11-08"
    artifact.mkdir(parents=True, exist_ok=True)
    for name in (
        "transfer.csv",
        "recovery.csv",
        "costs.csv",
        "summary.json",
        "index_audit.json",
        "frozen_results.json",
    ):
        shutil.copy2(output / name, artifact / name)
    for path in plots.glob("*.png"):
        shutil.copy2(path, artifact / path.name)
    for name, path in (
        ("protocol.md", base / "PROTOCOL.md"),
        ("method_registration.json", root / "theory_registration.json"),
    ):
        shutil.copy2(path, artifact / name)
    if interpretation.exists():
        shutil.copy2(interpretation, artifact / "interpretation.md")
    compact = report.read_text().replace(str(plots) + "/", str(artifact) + "/")
    for name in ("transfer.csv", "recovery.csv", "costs.csv"):
        compact = compact.replace(str(output / name), str(artifact / name))
    report.write_text(compact)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--phase", choices=("freeze", "score"), required=True)
    args = parser.parse_args()
    operation = freeze if args.phase == "freeze" else score
    operation(args.base.resolve())


if __name__ == "__main__":
    main()
