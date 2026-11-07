"""Report frozen G23 source structure, fresh-C transfer, and actual trajectories."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.darts_structured_tasks_v1.evaluate import align_indices

from .shared_structure_data import teacher

LABELS = {
    "prefix": "Functional prefix",
    "lc": "LC adaptation",
    "dense_dp": "Dense GN + DP",
    "lutq": "LUT-Q adaptation",
    "yeh": "Yeh-style MAP",
    "oracle_true_I": "True-I numeric control",
}
COLORS = {
    "prefix": "tab:blue",
    "lc": "tab:orange",
    "dense_dp": "tab:green",
    "lutq": "tab:red",
    "yeh": "tab:purple",
    "oracle_true_I": "black",
}


def read_json(path):
    return json.loads(path.read_text())


def verify_freeze(directory, name):
    path = directory / name
    frozen = read_json(path)
    for relative, expected in frozen["sha256"].items():
        if digest(directory / relative) != expected:
            raise RuntimeError(f"Frozen G23 artifact changed: {directory / relative}")
    return digest(path)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )


def load_selected(root, job_id):
    path = root / "common/selected" / f"job_{job_id:03d}"
    with np.load(path.with_suffix(".npz")) as arrays:
        values = dict(arrays)
    return read_json(path.with_suffix(".json")), values


def selected_fit(root, job_id, method, scope, phase):
    if phase == "common":
        meta, _ = load_selected(root, job_id)
        row = next(
            r for r in meta["records"] if r["method"] == method and r["scope"] == scope
        )
        candidate = row.get("selected_candidate")
    else:
        bank = read_json(root / phase / "banks" / f"job_{job_id:03d}.json")
        if method == "oracle_true_I":
            candidate = bank["oracle_reference"]
        else:
            row = next(
                r
                for r in bank["records"]
                if r["method"] == method and r["scope"] == scope
            )
            candidate = row.get("candidate")
    if candidate is None:
        return None, None
    bank = read_json(root / phase / "banks" / f"job_{job_id:03d}.json")
    path = (
        root
        / phase
        / "fits"
        / f"job_{job_id:03d}"
        / bank["candidates"][candidate]["signature"]
    )
    with np.load(path.with_suffix(".npz")) as arrays:
        values = dict(arrays)
    return read_json(path.with_suffix(".json")), values


def comparisons(plan, structural, transfer, selected):
    target = {
        (r["family_id"], r["noise"], r["method"], r["scope"]): r for r in transfer
    }
    rows = []
    for sigma in plan["noise"]:
        for scope in ("source1", "common4"):
            for variant in ("all", "local", "quartet_mix", "random"):
                for method in plan["methods"]:
                    cases = [
                        r
                        for r in structural
                        if r["noise"] == sigma
                        and r["method"] == method
                        and r["scope"] == scope
                        and (variant == "all" or r["family"]["variant"] == variant)
                    ]
                    complete = [r for r in cases if r["complete"]]
                    targets = (
                        [
                            target[(r["family_id"], sigma, method, scope)]
                            for r in complete
                        ]
                        if transfer
                        else []
                    )
                    risks = [r["mean_target_clean_mse"] for r in targets]
                    excess = [r["mean_excess_mse_over_true_I"] for r in targets]
                    intervals = [
                        selected[(r["family_id"], sigma, method, scope)][
                            "risk_intervals"
                        ]
                        for r in complete
                    ]
                    rows.append(
                        {
                            "noise": sigma,
                            "scope": scope,
                            "variant": variant,
                            "method": method,
                            "independent_geometries": len(cases),
                            "exact_I": sum(r["exact_I"] for r in cases),
                            "exact_mask": sum(r["exact_mask"] for r in cases),
                            "exact_unsigned_I": sum(
                                r["exact_unsigned_I"] for r in cases
                            ),
                            "wrong_mask": sum(
                                r["structural_failure_stage"] == "mask" for r in cases
                            ),
                            "wrong_sharing_given_exact_mask": sum(
                                r["structural_failure_stage"] == "sharing"
                                for r in cases
                            ),
                            "wrong_signs_given_exact_unsigned_I": sum(
                                r["structural_failure_stage"] == "signs" for r in cases
                            ),
                            "complete": len(complete),
                            "wrong_full_I": sum(not r["exact_I"] for r in complete),
                            "no_candidate": len(cases) - len(complete),
                            "bank_contains_true_I": sum(
                                r["bank_contains_true_I"] for r in cases
                            ),
                            "bank_contains_true_I_but_selection_wrong": sum(
                                r["bank_contains_true_I"] and not r["exact_I"]
                                for r in cases
                            ),
                            "strict_best_fitted_design_risk": sum(
                                r["strictly_best_design_risk_certified"]
                                for r in intervals
                            ),
                            "mean_mask_recall_complete_only": (
                                float(np.mean([r["mask_recall"] for r in complete]))
                                if complete
                                else None
                            ),
                            "target_functions_with_output": len(targets)
                            * plan["targets"],
                            "target_functions_total": len(cases) * plan["targets"],
                            "mean_target_clean_mse_complete_only": (
                                float(np.mean(risks)) if risks else None
                            ),
                            "mean_target_excess_over_true_I_complete_only": (
                                float(np.mean(excess)) if excess else None
                            ),
                        }
                    )
    return rows


def plot_comparison(report, rows, plan):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    positions = np.arange(len(plan["methods"]))
    for column, sigma in enumerate(plan["noise"]):
        for shift, scope in ((-0.2, "source1"), (0.2, "common4")):
            current = [
                next(
                    r
                    for r in rows
                    if r["noise"] == sigma
                    and r["scope"] == scope
                    and r["variant"] == "all"
                    and r["method"] == method
                )
                for method in plan["methods"]
            ]
            x = positions + shift
            exact = [r["exact_I"] for r in current]
            axes[0, column].bar(x, exact, width=0.36, label=scope)
            axes[0, column].bar(
                x,
                [r["wrong_full_I"] for r in current],
                width=0.36,
                bottom=exact,
                facecolor="none",
                edgecolor="0.5",
                hatch="//",
            )
            axes[0, column].plot(
                x,
                [r["bank_contains_true_I"] for r in current],
                "k_",
                ms=14,
            )
            for place, value in zip(x, exact, strict=True):
                axes[0, column].text(place, value + 0.1, str(value), ha="center")
            axes[1, column].plot(
                positions,
                [
                    r["mean_target_clean_mse_complete_only"]
                    if r["complete"]
                    else np.nan
                    for r in current
                ],
                marker="o",
                label=scope + " (available outputs)",
            )
        axes[0, column].set_ylim(0, len(plan["families"]) + 0.7)
        axes[0, column].set_yticks(range(len(plan["families"]) + 1))
        axes[0, column].set_ylabel(
            "Exact I; hatched = wrong full I; black = bank recall"
        )
        axes[0, column].set_title(f"Noise SD {sigma}; six independent I")
        axes[1, column].set_yscale("log")
        axes[1, column].set_ylabel("Fresh-target clean MSE (coverage in CSV)")
        for ax in axes[:, column]:
            ax.set_xticks(positions, [LABELS[m] for m in plan["methods"]], rotation=20)
    axes[0, 0].legend()
    axes[1, 0].legend()
    fig.suptitle(
        "G23 independent pilot: one-source vs common-I selection across four tasks\n"
        "Same observed panels and outer selection; known bias/readout; high query cost"
    )
    fig.savefig(report / "shared_recovery_transfer.png", dpi=170)
    plt.close(fig)


def plot_indices(root, report, plan, structural, truth):
    lookup = {(r["family_id"], r["method"]): r for r in structural if r["noise"] == 2}
    fig, axes = plt.subplots(6, 6, figsize=(18, 18), constrained_layout=True)
    for family_id, family in enumerate(plan["families"]):
        expected = truth[f"family_{family_id}__truth_I"]
        _, arrays = load_selected(root, 2 * family_id + 1)
        images = [(expected, "True signed I")]
        for method in plan["methods"]:
            score = lookup[(family_id, method)]
            if score["complete"]:
                _, aligned, _ = align_indices(
                    arrays[f"{method}__common4__index"], expected, plan["codes"]
                )
                images.append(
                    (
                        aligned,
                        LABELS[method]
                        + (
                            "\nexact"
                            if score["exact_I"]
                            else f"\n{score['categorical_errors']} errors"
                        ),
                    )
                )
            else:
                images.append((None, LABELS[method] + "\nno full candidate"))
        for column, (value, title) in enumerate(images):
            ax = axes[family_id, column]
            if value is not None:
                im = ax.imshow(value, cmap="RdBu_r", vmin=-4, vmax=4)
            else:
                ax.text(0.5, 0.5, "No output", ha="center", va="center")
            ax.set_title(title, fontsize=9)
            ax.set_xticks([0, 7, 15])
            ax.set_yticks([0, 7, 15])
        axes[family_id, 0].set_ylabel(
            f"{family['variant']} {family['family']}\nInput row"
        )
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5, label="Signed code ID")
    fig.suptitle(
        "G23 common4: all six true and selected indexators, noise SD2\n"
        "Group/sign names and hidden columns matched only after selection freeze"
    )
    fig.savefig(report / "shared_indexators.png", dpi=170)
    plt.close(fig)


def numeric_weights(index, codes):
    return np.r_[0.0, codes][np.abs(index)] * np.sign(index)


def numeric_columns(value, expected):
    costs = ((value[:, :, None] - expected[:, None, :]) ** 2).sum(0)
    source, target = linear_sum_assignment(costs)
    return source[np.argsort(target)]


def plot_source_weights(root, report, plan):
    fig, axes = plt.subplots(3, 6, figsize=(18, 10), constrained_layout=True)
    for line, family_id in enumerate((0, 2, 4)):
        family = plan["families"][family_id]
        index, codes = teacher(plan, family)
        expected = numeric_weights(index, codes[0])
        images = [(expected, "True numerical W")]
        for method in plan["methods"]:
            number = next(
                n
                for n, job in enumerate(plan["extract_jobs"])
                if job
                == {"family_id": family_id, "task": 0, "noise": 2.0, "method": method}
            )
            path = root / "extract" / f"job_{number:04d}"
            meta = read_json(path.with_suffix(".json"))
            if not meta["selected_records"]:
                images.append((None, LABELS[method] + "\nno full candidate"))
                continue
            record = meta["records"][meta["selected_records"][0]]
            key = record["key"]
            with np.load(path.with_suffix(".npz")) as arrays:
                if method == "prefix":
                    value = numeric_weights(
                        arrays[key + "__index"], arrays[key + "__centers"]
                    )
                    title = "Prefix hard W + C polish"
                else:
                    value = arrays[key + "__trace_latent_weights"][
                        record["selected_checkpoint"]
                    ]
                    title = {
                        "lc": "LC actual latent W",
                        "dense_dp": "Dense GN actual W",
                        "lutq": "LUT-Q Adam latent Z",
                        "yeh": "Yeh Adam actual soft W",
                    }[method]
            images.append((value[:, numeric_columns(value, expected)], title))
        limit = max(
            float(np.abs(value).max()) for value, _ in images if value is not None
        )
        for column, (value, title) in enumerate(images):
            ax = axes[line, column]
            if value is not None:
                im = ax.imshow(value, cmap="RdBu_r", vmin=-limit, vmax=limit)
            else:
                ax.text(0.5, 0.5, "No output", ha="center", va="center")
            ax.set_title(title, fontsize=9)
            ax.set_xticks([0, 7, 15])
            ax.set_yticks([0, 7, 15])
        axes[line, 0].set_ylabel(f"{family['variant']} {family['family']}\nInput row")
        fig.colorbar(
            im, ax=axes[line].tolist(), shrink=0.7, label="Actual numerical value"
        )
    fig.suptitle(
        "G23 matched source0 / SD2: actual source-fit-validation-selected states\n"
        "Only hidden columns matched; LUT-Q Z is latent, not its quantized forward W"
    )
    fig.savefig(report / "shared_source_weights.png", dpi=170)
    plt.close(fig)


def plot_numeric_fits(root, report, plan, phases=("common", "transfer")):
    task = 0
    for phase in phases:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        methods = plan["methods"] + (["oracle_true_I"] if phase == "transfer" else [])
        for column, family_id in enumerate((0, 2, 4)):
            family = plan["families"][family_id]
            job_id = 2 * family_id + 1
            for method in methods:
                meta, _ = selected_fit(root, job_id, method, "common4", phase)
                if meta is None:
                    continue
                fit = meta["numeric_fits"][task]
                start = fit["selected_start"]
                for line, field in enumerate(("train_mse", "validation_mse")):
                    history = fit["history"]
                    axes[line, column].plot(
                        [r["step"] for r in history],
                        [max(r[field][start], 1e-30) for r in history],
                        label=LABELS[method],
                        color=COLORS[method],
                    )
                    selected = history[fit["selected_step"]]
                    axes[line, column].plot(
                        selected["step"],
                        max(selected[field][start], 1e-30),
                        marker="o",
                        color=COLORS[method],
                        ms=4,
                    )
            axes[0, column].set_title(f"{family['variant']} {family['family']} / SD2")
            for line, title in enumerate(
                ("Actual training MSE", "Independent fit-validation MSE")
            ):
                axes[line, column].set_yscale("log")
                axes[line, column].set_xlabel("Actual fresh-C Gauss–Newton step")
                axes[line, column].set_ylabel(title)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(
            f"G23 {phase}: selected start's real signed-C optimizer trajectory\n"
            "Selected checkpoint marked; source0 or fresh target0, fixed selected I, no numeric transfer"
        )
        fig.savefig(report / f"shared_{phase}_numeric_fits.png", dpi=170)
        plt.close(fig)


def plot_search_curves(root, report, plan, truth):
    fig, axes = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)
    for line, family_id in enumerate((0, 2, 4)):
        expected = truth[f"family_{family_id}__truth_I"]
        with np.load(root / "data" / f"family_{family_id:03d}_source.npz") as data:
            order = data["order"]
        for method in plan["methods"]:
            number = next(
                n
                for n, job in enumerate(plan["extract_jobs"])
                if job
                == {"family_id": family_id, "task": 0, "noise": 2.0, "method": method}
            )
            path = root / "extract" / f"job_{number:04d}"
            meta = read_json(path.with_suffix(".json"))
            chosen = meta["selected_records"][0] if meta["selected_records"] else 0
            record = meta["records"][chosen]
            with np.load(path.with_suffix(".npz")) as data:
                indices = data[record["key"] + "__trace_indices"]
            if method == "prefix":
                history = [r for r in record["history"] if r["accepted"]]
                steps = [r["prefix_length"] for r in history]
                train = [r["slice_mse_integer"] for r in history]
                validation = [r["prefix_validation_mse"] for r in history]
                for column, values in enumerate((train, validation)):
                    axes[line, column].plot(
                        steps, values, ".-", color=COLORS[method], label="Prefix"
                    )
                    axes[line, column].set_xlabel("Accepted input coordinate")
                accuracy = []
                recall = []
                for step, index in enumerate(indices, 1):
                    target = expected.copy()
                    target[order[step:]] = 0
                    metric, _, _ = align_indices(index, target, plan["codes"])
                    accuracy.append(
                        1 - metric["categorical_errors"] / (step * plan["hidden"])
                    )
                    recall.append(metric["mask_recall"])
                twin = axes[line, 1].twinx()
                twin.plot(steps, accuracy, "k-", label="Prefix I accuracy")
                twin.plot(steps, recall, "k--", label="Prefix mask recall")
                twin.set_ylabel("Prefix accuracy / recall (audit)")
                twin.set_ylim(-0.03, 1.03)
            else:
                history = record["history"]
                steps = [
                    r["l_solves"] if method in ("lc", "dense_dp") else r["step"]
                    for r in history
                ]
                metric = [
                    align_indices(index, expected, plan["codes"])[0]
                    for index in indices
                ]
                accuracy = [1 - r["categorical_errors"] / expected.size for r in metric]
                column = 2 if method in ("lc", "dense_dp") else 3
                axes[line, column].plot(
                    steps, accuracy, color=COLORS[method], label=LABELS[method]
                )
                axes[line, column].plot(
                    steps,
                    [r["mask_recall"] for r in metric],
                    color=COLORS[method],
                    ls="--",
                    alpha=0.7,
                )
                axes[line, column].set_xlabel(
                    "Actual GN solves"
                    if column == 2
                    else "Actual Adam steps after GN warmup"
                )
                axes[line, column].set_ylabel(
                    "I accuracy (solid); mask recall (dashed)"
                )
                axes[line, column].set_ylim(-0.03, 1.03)
        for column, title in enumerate(
            (
                "Prefix training slice MSE",
                "Prefix independent validation MSE",
                "GN methods",
                "Adam methods",
            )
        ):
            axes[line, column].set_title(title)
            if column < 2:
                axes[line, column].set_yscale("log")
            else:
                axes[line, column].legend(fontsize=8)
        family = plan["families"][family_id]
        axes[line, 0].set_ylabel(f"{family['variant']} {family['family']}\nMSE")
    fig.suptitle(
        "G23 source0 / SD2: measured search trajectories of fit-validation-selected runs\n"
        "Prefix coordinate and native optimizer axes differ; true I used only for late audit"
    )
    fig.savefig(report / "shared_search_curves.png", dpi=170)
    plt.close(fig)


def generate(root):
    plan = read_json(root / "protocol.json")
    freezes = {}
    for phase, names in (
        (root, ("data_frozen.json", "extract_frozen.json")),
        (root / "common", ("fit_frozen.json", "selection_frozen.json")),
        (root / "transfer", ("data_frozen.json", "fit_frozen.json")),
    ):
        for name in names:
            freezes[str((phase / name).relative_to(root))] = verify_freeze(phase, name)
    source_audit = read_json(root / "common/audit.json")
    target_audit = read_json(root / "transfer/audit.json")
    if (
        source_audit["selection_freeze_sha256"]
        != freezes["common/selection_frozen.json"]
    ):
        raise RuntimeError("Source audit refers to another I selection")
    if target_audit["target_fit_freeze_sha256"] != freezes["transfer/fit_frozen.json"]:
        raise RuntimeError("Target audit refers to another numeric fit")
    if (
        target_audit["source_selection_freeze_sha256"]
        != freezes["common/selection_frozen.json"]
    ):
        raise RuntimeError("Target audit refers to another source I selection")
    structural = source_audit["records"]
    transfer = target_audit["records"]
    selected = {}
    for job_id in range(len(plan["families"]) * len(plan["noise"])):
        meta, _ = load_selected(root, job_id)
        for row in meta["records"]:
            selected[
                (meta["family_id"], meta["noise"], row["method"], row["scope"])
            ] = row
    rows = comparisons(plan, structural, transfer, selected)
    timing = defaultdict(lambda: {"jobs": 0, "trajectories": 0, "worker_seconds": 0.0})
    partial = Counter()
    for job_id in range(len(plan["extract_jobs"])):
        meta = read_json(root / "extract" / f"job_{job_id:04d}.json")
        method = meta["job"]["method"]
        timing[method]["jobs"] += 1
        timing[method]["trajectories"] += len(meta["records"])
        timing[method]["worker_seconds"] += meta["seconds"]
        for row in meta["records"]:
            if method == "prefix" and not row["complete"]:
                partial[row["history"][-1]["reason"]] += 1
    source_cost = read_json(root / "data/family_000.json")
    target_cost = read_json(root / "transfer/data/family_000.json")
    numeric = {}
    for phase in ("common", "transfer"):
        fits = [read_json(p) for p in (root / phase / "fits").glob("job_*/*.json")]
        numeric[phase] = {
            "unique_I_fits": len(fits),
            "task_C_fits": sum(len(r["numeric_fits"]) for r in fits),
            "worker_seconds": sum(r["seconds"] for r in fits),
            "reuse": "Same canonical I / data / starts reused across methods; not a matched-FLOPs claim",
        }
    report = root / "report"
    report.mkdir(exist_ok=False)
    write_csv(report / "comparison.csv", rows)
    write_csv(report / "source_family_metrics.csv", structural)
    write_csv(report / "target_family_metrics.csv", transfer)
    summary = {
        "comparison": rows,
        "source_timing": dict(timing),
        "numeric_timing": numeric,
        "prefix_incomplete_reasons": dict(partial),
        "source_cost": source_cost,
        "target_cost": target_cost,
        "independent_geometries": len(plan["families"]),
        "underlying_source_functions": len(plan["families"]) * plan["sources"],
        "underlying_target_functions": len(plan["families"]) * plan["targets"],
        "paired_noise_conditions": len(plan["noise"]),
        "frozen_inputs": freezes,
        "report_code_sha256": digest(Path(__file__)),
    }
    save_json(report / "summary.json", summary)
    with np.load(root / "common/audit_only.npz") as arrays:
        truth = dict(arrays)
    plot_comparison(report, rows, plan)
    plot_indices(
        root, report, plan, [r for r in structural if r["scope"] == "common4"], truth
    )
    plot_source_weights(root, report, plan)
    plot_numeric_fits(root, report, plan)
    plot_search_curves(root, report, plan, truth)
    write_report(report, summary, plan)


def generate_source(root):
    """Publish exact-I evidence as soon as source selection has been audited."""
    plan = read_json(root / "protocol.json")
    freezes = {}
    for directory, names in (
        (root, ("data_frozen.json", "extract_frozen.json")),
        (root / "common", ("fit_frozen.json", "selection_frozen.json")),
    ):
        for name in names:
            freezes[str((directory / name).relative_to(root))] = verify_freeze(
                directory, name
            )
    audit = read_json(root / "common/audit.json")
    if audit["selection_freeze_sha256"] != freezes["common/selection_frozen.json"]:
        raise RuntimeError("Source audit refers to another I selection")
    selected = {}
    for job_id in range(len(plan["families"]) * len(plan["noise"])):
        meta, _ = load_selected(root, job_id)
        for row in meta["records"]:
            selected[
                (meta["family_id"], meta["noise"], row["method"], row["scope"])
            ] = row
    rows = [
        {key: value for key, value in row.items() if "target" not in key}
        for row in comparisons(plan, audit["records"], [], selected)
    ]
    report = root / "structure_report"
    report.mkdir(exist_ok=False)
    write_csv(report / "comparison.csv", rows)
    write_csv(report / "family_metrics.csv", audit["records"])
    save_json(
        report / "summary.json",
        {
            "comparison": rows,
            "frozen_inputs": freezes,
            "primary_criterion": "Exact signed I modulo group/sign names and hidden-column permutations",
            "target_results_included": False,
            "report_code_sha256": digest(Path(__file__)),
        },
    )
    with np.load(root / "common/audit_only.npz") as arrays:
        truth = dict(arrays)
    plot_indices(
        root,
        report,
        plan,
        [r for r in audit["records"] if r["scope"] == "common4"],
        truth,
    )
    plot_source_weights(root, report, plan)
    plot_numeric_fits(root, report, plan, phases=("common",))
    plot_search_curves(root, report, plan, truth)
    lines = [
        "# G23: сравнение восстановления структуры",
        "",
        (
            "Основной критерий — точный signed индексатор: нули, группы шэринга и знаки, "
            "с точностью до глобального переименования групп/знаков и перестановки hidden columns. "
            "Входные строки не переставляются. Шесть новых независимых I, четыре source-C на I, "
            "два парных шумовых условия. Качество кодбука не подменяет критерий структуры."
        ),
        "",
        "| Шум | Выбор | Метод | Точная маска / 6 | Верные unsigned группы / 6 | Точный signed I / 6 | Истина в банке / 6 | Ошибка выбора при наличии истины | Нет кандидата |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        if row["variant"] == "all":
            lines.append(
                f"| {row['noise']} | {row['scope']} | {LABELS[row['method']]} | "
                f"{row['exact_mask']}/6 | {row['exact_unsigned_I']}/6 | "
                f"{row['exact_I']}/6 | {row['bank_contains_true_I']}/6 | "
                f"{row['bank_contains_true_I_but_selection_wrong']} | "
                f"{row['no_candidate']} |"
            )
    lines.extend(
        [
            "",
            (
                "До teacher-аудита заморожены все source-кандидаты, все численные fits до "
                "meta-Y и все выбранные I. Общая meta-selection обёртка одинакова для методов. "
                "LC/LUT-Q/Yeh-style — objective-level адаптации; dense GN имеет явный DP-readout. "
                "Прямой functional-prefix метод строит назначения из функциональных counts. "
                "Полные результаты по каждой геометрии и группам задач сохранены в CSV/JSON."
            ),
            "",
            (
                "Точная маска допускает оптимальную перестановку hidden columns независимо от "
                "номеров групп. Unsigned I дополнительно проверяет равенство модулей весов с "
                "глобальным переименованием групп. Signed I также требует верных знаков "
                "с точностью до одного глобального переворота каждой группы. Поэтому "
                "точный signed I подразумевает точный unsigned I, а тот — точную маску. "
                "CSV отдельно содержит mask failures, sharing failures при верной маске "
                "и sign failures при верном unsigned I."
            ),
            "",
            "![Все шесть индексаторов](shared_indexators.png)",
            "",
            "![Настоящие source W/Z](shared_source_weights.png)",
            "",
            "![Реальные шаги поиска и recall](shared_search_curves.png)",
            "",
            "![Дополнительный численный контроль C](shared_common_numeric_fits.png)",
            "",
            (
                "Пилот использует известные common bias/readout и активные запросы. Полная цена "
                "наблюдений — 275,196,674,048 labels на source-функцию/шум. "
                "Перенос на новые задачи проверяется отдельно и ещё не включён в эту таблицу. "
                "Эта публикация не является утверждением о завершении общей цели, "
                "конкурентном превосходстве или доказанном приоритете нового метода."
            ),
            "",
        ]
    )
    (report / "REPORT.md").write_text("\n".join(lines))


def write_report(report, summary, plan):
    lines = [
        "# G23: общий индексатор нескольких задач и перенос только структуры",
        "",
        (
            "Независимый пилот: шесть новых I, по две геометрии local/mixed/random, "
            "четыре независимых source-C и восемь новых target-C на I. "
            "24 исходные и 48 целевых функций наблюдаются в двух парных шумовых условиях. "
            "Это шесть независимых структур, не 48 или 96 независимых геометрий."
        ),
        "",
        (
            "| Шум | Выбор | Метод | Точный I / 6 | Истина в банке / 6 | Неверный полный I | "
            "Нет кандидата | Target coverage | Clean target MSE* | Excess к true-I* |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in summary["comparison"]:
        if row["variant"] != "all":
            continue
        risk = row["mean_target_clean_mse_complete_only"]
        excess = row["mean_target_excess_over_true_I_complete_only"]
        lines.append(
            f"| {row['noise']} | {row['scope']} | {LABELS[row['method']]} | "
            f"{row['exact_I']}/6 | {row['bank_contains_true_I']}/6 | {row['wrong_full_I']} | "
            f"{row['no_candidate']} | {row['target_functions_with_output']}/48 | "
            + (f"{risk:.6g} | {excess:.6g} |" if risk is not None else "— | — |")
        )
    lines.extend(
        [
            "",
            (
                "*MSE и excess усреднены только по структурам с полным выходом; "
                "доля доступных задач приведена отдельно, отказы не объявлены нулевой ошибкой. "
                "Excess сравнивает с обучением свежего C на истинном I на тех же support/validation "
                "и с теми же 32 стартами; это численный контроль, не перенос истинного C."
            ),
            "",
            "## Протокол выбора и границы интерпретации",
            "",
            (
                "Каждый метод извлекает до четырёх уникальных I из каждой source-задачи по "
                "fit-validation. Общий банк объединяет четыре задачи. Для каждого I все четыре "
                "кодбука обучаются заново из 32 signed стартов. Все предсказатели заморожены до "
                "открытия отдельной meta-selection Y; затем заморожены все выбранные I и только "
                "потом открыт структурный аудит. Source1 ограничивает и происхождение, и оценку "
                "задачей0. Все методы используют одинаковые панели и общий внешний протокол."
            ),
            "",
            (
                "На переносе I неизменен, C каждого нового задания обучается с нуля. Ни source-C, "
                "ни W не переносятся. Все target-модели заморожены до clean-test аудита. "
                "Gaussian интервалы относятся к различию design-risk фактически fitted "
                "предсказателей внутри одного фиксированного банка; это не сертификаты "
                "истинного I, population-risk или глобального численного optimum C."
            ),
            "",
            (
                "LC, LUT-Q и Yeh-style — раскрытые адаптации известных objectives к signed "
                "кодбуку и известному bias/readout, а не неизменённые авторские скрипты. "
                "Dense GN + DP имеет явный дополнительный проекционный readout. "
                "Fixed-I numerical polish не изменяет назначения native-метода и не подаётся "
                "обратно в его траекторию. Больший HPO бюджет конкурентов сохранён полностью."
            ),
            "",
            "## Цена данных и вычислений",
            "",
            (
                f"Source: {summary['source_cost']['labels_per_source_task_condition']:,} физических "
                "наблюдений на функцию/шум, включая все осевые, prefix, fit-validation и meta "
                "панели. На четыре source-функции одного I: "
                f"{summary['source_cost']['labels_per_family_condition']:,}. "
                "Точная симуляция Gaussian means не делает эти повторы бесплатными данными. "
                f"Target: {summary['target_cost']['labels_per_family_condition']:,} noisy labels "
                "на восемь функций одного I/шум и отдельно "
                f"{summary['target_cost']['test_true_function_evaluations']:,} clean evaluations."
            ),
            "",
            "| Метод | Source jobs | Траектории | Сумма worker seconds |",
            "|---|---|---|---|",
        ]
    )
    for method in plan["methods"]:
        row = summary["source_timing"][method]
        lines.append(
            f"| {LABELS[method]} | {row['jobs']} | {row['trajectories']} | {row['worker_seconds']:.2f} |"
        )
    lines.extend(
        [
            "",
            (
                "Worker seconds — сумма измеренных длительностей, не FLOPs и не календарное "
                "время всей параллельной серии. Численные fits одинаковых canonical I переиспользованы "
                "между методами только при одинаковых observations/initials; их общий учёт "
                "сохранён отдельно в summary.json, без заявления matched compute."
            ),
            "",
            "## Графики и артефакты",
            "",
            "![Восстановление и перенос](shared_recovery_transfer.png)",
            "",
            "![Все шесть выбранных индексаторов](shared_indexators.png)",
            "",
            "![Настоящие численные состояния source-оптимизаторов](shared_source_weights.png)",
            "",
            (
                "Последний рисунок показывает source0/SD2 и checkpoint по source-fit-validation, "
                "который может отличаться от выбранного общего I. LUT-Q показывает настоящий "
                "latent Z; он не обозначен как quantized forward W. Yeh показывает actual soft W. "
                "Численные значения не переименованы; выровнены только hidden columns."
            ),
            "",
            "![Обучение свежего C на source](shared_common_numeric_fits.png)",
            "",
            "![Обучение свежего C на target](shared_transfer_numeric_fits.png)",
            "",
            "![Реальные шаги поиска и поздний структурный аудит](shared_search_curves.png)",
            "",
            (
                "Кривые используют сохранённые реальные шаги; coordinate-prefix, GN solves и "
                "Adam steps не объединены в фиктивную общую шкалу. Численные train/validation "
                "кривые показаны для selected start и выбранного checkpoint. Полные "
                "построчные результаты: comparison.csv, source_family_metrics.csv, "
                "target_family_metrics.csv и summary.json; исходные протоколы/snapshots/freezes сохранены."
            ),
            "",
            "## Что эта серия не доказывает",
            "",
            (
                "Известны общий bias=.15, readout=1/sqrt(H), D=H16 и K4; "
                "использованы активные запросы и очень высокая точность осевых means. "
                "Шесть новых I — пилот, не окончательная confirmation. Нужны свободные bias/readout, "
                "другие шумы и независимое подтверждение. Hypothesis selection, GN, nearest-code "
                "и block coordinate descent известны; их применение само по себе не новизна. "
                "Приоритет функциональной count-факторизации и преимущество собственного "
                "компонента над подходящими сильными functional-extraction контролями "
                "ещё требуется установить. Большая цель пользователя остаётся active."
            ),
            "",
        ]
    )
    (report / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    (generate_source if args.source_only else generate)(args.root)


if __name__ == "__main__":
    main()
