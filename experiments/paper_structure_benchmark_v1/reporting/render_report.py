"""Build paper tables, scientific figures and the companion benchmark report."""

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.stats import t as student_t

from .evaluate import write_csv

ORDER = (
    "dense_adam",
    "dense",
    "random",
    "lutq",
    "yeh_map",
    "yeh_soft",
    "msr",
    "current",
    "current_functional",
    "mask_only",
    "mask_random",
    "msr_native",
)
NAMES = {
    "dense_adam": "Dense Adam",
    "dense": "Dense",
    "random": "Random I",
    "lutq": "LUT-Q I",
    "yeh_map": "Yeh MAP I",
    "yeh_soft": "Yeh soft",
    "msr": "MSR-FC U",
    "current": "Ours",
    "current_functional": "Ours + functional",
    "mask_only": "Our mask, free weights",
    "mask_random": "Random mask, free weights",
    "msr_native": "MSR + learned init",
}
DATASETS = ("Pattern 0110", "Synthetic DeepSets", "Aug-Omniglot", "Paired MNIST sums")
COLORS = dict(
    zip(
        ORDER,
        (
            "#8c8c8c",
            "#4d4d4d",
            "#c7b299",
            "#d78b24",
            "#9d66b3",
            "#6a43a1",
            "#3a83bd",
            "#16845d",
            "#c13f56",
            "#257d88",
            "#a7b5ad",
            "#1d4859",
        ),
    )
)


def load_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def metadata(path):
    return json.loads(path.read_text())


def name(method, dataset):
    if method == "dense":
        return "Dense Adam + GN" if dataset in DATASETS[:2] else "Dense Adam"
    return NAMES[method]


def short_number(value):
    if value is None:
        return "--"
    if value == 0:
        return "0"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 0.01:
        return f"{value:.3g}"
    return f"{value:.2e}"


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[
            (
                row["dataset"],
                row["method"],
                int(row["condition"]),
                int(row["support_fit"]),
            )
        ].append(row)
    result = []
    for (dataset, method, condition, support), values in groups.items():
        metric = "accuracy" if dataset == "Aug-Omniglot" else "mse"
        if dataset in DATASETS[:2]:
            by_case = defaultdict(list)
            for row in values:
                by_case[row["case"]].append(float(row[metric]))
            case_means = np.array([np.mean(items) for items in by_case.values()])
            mean, spread = case_means.mean(), case_means.std(ddof=1)
            spread_kind = "SD across three families"
        else:
            numeric = np.array([float(row[metric]) for row in values])
            mean, spread = (
                numeric.mean(),
                student_t.ppf(0.975, len(numeric) - 1)
                * numeric.std(ddof=1)
                / math.sqrt(len(numeric)),
            )
            spread_kind = (
                "95% t interval across episodes/tasks, conditional on one source cohort"
            )
        result.append(
            {
                "dataset": dataset,
                "method": method,
                "condition": condition,
                "support_fit": support,
                "metric": metric,
                "mean": float(mean),
                "spread": float(spread),
                "spread_kind": spread_kind,
                "tasks": len(values),
                "codes": int(values[0]["codes"]),
                "numeric_updates": values[0].get("numeric_updates", ""),
                "transfer_mode": values[0]["transfer_mode"],
            }
        )
    return result


def profile_sum(records):
    return {
        "profiled_offline_seconds": sum(
            item.get("wall_seconds", 0) for item in records
        ),
        "offline_torch_flops": sum(
            item.get("torch_operator_flops", 0) or 0 for item in records
        ),
        "offline_peak_mib": max(
            (item.get("cuda_peak_allocated_bytes", 0) for item in records), default=0
        )
        / 2**20,
        "offline_cpu_peak_mib": max(
            (item.get("cpu_peak_rss_bytes", 0) for item in records), default=0
        )
        / 2**20,
    }


def parameter_counts(base, dataset, method):
    """Count all target numerical coordinates, excluding the fixed structure."""
    if dataset in DATASETS[:2]:
        case = "case_00" if dataset == DATASETS[0] else "case_03"
        root = base / "synthetic_confirmation" / "targets" / f"{case}_c0"
        owner = "dense" if method == "dense_adam" else method
        parameter_key = "parameters_1024"
    else:
        directory = "omniglot_pilot" if dataset == DATASETS[2] else "digit_sum_pilot"
        root = base / directory / "targets"
        owner, parameter_key = method, "parameters"
    sizes = metadata(root / "sizes.json")
    with np.load(root / owner / "frozen_predictions.npz") as archive:
        total = archive[parameter_key].shape[-1]
    hidden, outputs = sizes["hidden"], sizes["outputs"]
    other = hidden + hidden * outputs + outputs
    optimizer = (
        sizes["inputs"] * hidden + other
        if method in ("mask_only", "mask_random")
        else total
    )
    return {
        "target_first_layer_parameters": total - other,
        "target_other_parameters": other,
        "target_numerical_parameters": total,
        "target_optimizer_coordinates": optimizer,
        "parameter_count_scope": "Per target task; codebook/free active first weights plus hidden biases, free output matrix and output offset; fixed indexator/basis excluded",
    }


def cost_rows(base):
    rows = []
    synthetic = base / "synthetic_confirmation"
    source = load_csv(synthetic / "source_report" / "source_selection.csv")
    labels = {
        "current": "Current",
        "lutq": "LUT-Q + refit",
        "yeh_map": "Yeh MAP + refit",
        "yeh_soft": "Yeh soft",
        "msr": "MSR-FC",
    }
    for dataset, case in zip(DATASETS[:2], ("case_00", "case_03")):
        for method in ORDER:
            if method in (
                "current_functional",
                "mask_only",
                "mask_random",
                "msr_native",
            ):
                continue
            owner = "dense" if method == "dense_adam" else method
            folder = synthetic / "targets" / f"{case}_c0" / owner
            measured = metadata(folder / "unprofiled_timing" / "metrics.json")
            timing = metadata(folder / "unprofiled_timing" / "timing.json")
            instrumented = metadata(folder / "metrics.json")
            seconds = (
                timing[
                    "adam_seconds_1024" if method == "dense_adam" else "seconds_1024"
                ]
                / 8
            )
            matching = [
                row
                for row in source
                if row["dataset"]
                == ("Pattern 0110" if dataset == DATASETS[0] else "DeepSets")
                and row["condition"] == "0"
                and row["method"] == labels.get(method)
            ]
            offline = (
                float(np.mean([float(row["offline_seconds"]) for row in matching]))
                if matching
                else 0.0
                if method in ("dense", "dense_adam")
                else None
            )
            operations = (
                float(np.mean([float(row["torch_operator_flops"]) for row in matching]))
                if matching
                else 0.0
                if method in ("dense", "dense_adam")
                else None
            )
            peak = max([float(row["gpu_peak_mib"]) for row in matching], default=0.0)
            target_peak = (
                timing.get("adam_peak_allocated_bytes_1024")
                if method == "dense_adam"
                else measured["cuda_peak_allocated_bytes"]
            )
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "profiled_offline_seconds": offline,
                    "offline_torch_flops": operations,
                    "offline_peak_mib": peak if method != "random" else None,
                    "target_seconds_per_task_unprofiled": seconds,
                    "target_peak_mib": None
                    if target_peak is None
                    else target_peak / 2**20,
                    "target_torch_flops_per_task_profiled": None
                    if method == "dense_adam"
                    else instrumented["torch_operator_flops"] / 8,
                    "target_operation_scope": "Three support-size sweep (64/256/1024); time is N1024 only",
                    "timing_scope": "Representative first family, source noise0, one process per GPU; no FLOP counter",
                    "numeric_parameters_transferred": False,
                }
            )
    for dataset, directory in zip(DATASETS[2:], ("omniglot_pilot", "digit_sum_pilot")):
        root = base / directory
        protocol = metadata(root / "protocol.json")
        for method in ORDER:
            if method == "dense_adam" or (
                method in ("current_functional", "mask_only", "mask_random")
                and directory != "digit_sum_pilot"
            ):
                continue
            folder = root / "targets" / method
            target = metadata(folder / "metrics.json")
            measured = metadata(folder / "unprofiled_timing" / "metrics.json")
            common = metadata(root / "source" / "dense" / "metrics.json")
            if method in ("dense", "random"):
                records = [metadata(root / "validation_fit" / method / "metrics.json")]
            elif method in ("mask_only", "mask_random"):
                records = [
                    common,
                    metadata(root / "source" / "current_proposals" / "metrics.json"),
                    *[
                        metadata(root / "fits" / f"current_k{k}" / "metrics.json")
                        for k in protocol["independent_code_grid"]
                    ],
                    metadata(root / "validation_fit" / "current" / "metrics.json"),
                    metadata(root / "validation_fit" / method / "metrics.json"),
                ]
            elif method == "msr_native":
                records = [
                    common,
                    *[
                        metadata(root / "source" / f"msr_k{k}" / "metrics.json")
                        for k in protocol["independent_code_grid"]
                    ],
                ]
                records.append(
                    {
                        "wall_seconds": metadata(folder / "prepare_selected.json")[
                            "wall_seconds"
                        ]
                    }
                )
            else:
                owner = "yeh" if method.startswith("yeh") else method
                if method == "current_functional":
                    records = [
                        common,
                        metadata(
                            root / "source" / "current_proposals" / "metrics.json"
                        ),
                        *[
                            metadata(root / "fits" / f"current_k{k}" / "metrics.json")
                            for k in protocol["independent_code_grid"]
                        ],
                        metadata(root / "validation_fit" / "current" / "metrics.json"),
                        metadata(root / "validation_fit" / "dense" / "metrics.json"),
                        metadata(
                            root
                            / "source"
                            / "current_functional_proposals"
                            / "metrics.json"
                        ),
                        metadata(
                            root / "fits" / "current_functional_k128" / "metrics.json"
                        ),
                        metadata(root / "validation_fit" / method / "metrics.json"),
                    ]
                else:
                    records = [common]
                    if method == "current":
                        records.append(
                            metadata(
                                root / "source" / "current_proposals" / "metrics.json"
                            )
                        )
                    else:
                        records.extend(
                            metadata(root / "source" / f"{owner}_k{k}" / "metrics.json")
                            for k in protocol["independent_code_grid"]
                        )
                    records.extend(
                        metadata(root / "fits" / f"{method}_k{k}" / "metrics.json")
                        for k in protocol["independent_code_grid"]
                    )
                    records.append(
                        metadata(root / "validation_fit" / method / "metrics.json")
                    )
            combined = profile_sum(records)
            if method in ("random", "mask_random"):
                combined.update(
                    profiled_offline_seconds=None,
                    offline_torch_flops=None,
                    offline_peak_mib=None,
                )
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    **combined,
                    "target_seconds_per_task_unprofiled": measured["wall_seconds"]
                    / target["tasks"],
                    "target_peak_mib": measured["cuda_peak_allocated_bytes"] / 2**20,
                    "target_torch_flops_per_task_profiled": target[
                        "torch_operator_flops"
                    ]
                    / target["tasks"],
                    "target_operation_scope": "One prescribed support budget, complete target cohort",
                    "timing_scope": "Complete target cohort, one process per GPU; no FLOP counter",
                    "numeric_parameters_transferred": method == "msr_native",
                }
            )
    for row in rows:
        row.update(parameter_counts(base, row["dataset"], row["method"]))
    return rows


def table_quality(paper, summary):
    lines = [
        r"\begin{table}[htbp]",
        r"\centering\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Pattern ($0$) & Pattern ($0.1$) & DeepSets ($0$) & DeepSets ($0.1$)\\",
        r"\midrule",
    ]
    for method in ORDER:
        cells = []
        for dataset, condition in (
            (DATASETS[0], 0),
            (DATASETS[0], 1),
            (DATASETS[1], 0),
            (DATASETS[1], 1),
        ):
            row = next(
                (
                    item
                    for item in summary
                    if item["dataset"] == dataset
                    and item["method"] == method
                    and item["condition"] == condition
                    and item["support_fit"] == 1024
                ),
                None,
            )
            cells.append(
                "--"
                if row is None
                else r"$<10^{-12}$"
                if row["mean"] + row["spread"] < 1e-12
                else f"{row['mean']:.2e}"
            )
        if any(cell != "--" for cell in cells):
            lines.append(
                name(method, DATASETS[0]).replace("_", r"\_")
                + " & "
                + " & ".join(cells)
                + r"\\"
            )
    lines += [
        r"\bottomrule\end{tabular}",
        r"\caption{Mean target MSE over three families, each with eight new numerical tasks. Parentheses in headings give source noise; target data are clean. All models use 1024 fitting and 256 numerical-selection observations. Family-level spreads and all smaller support budgets are in the accompanying CSV.}",
        r"\label{tab:benchmark-synthetic}",
        r"\end{table}",
        "",
        r"\begin{table}[htbp]",
        r"\centering\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Omniglot accuracy (\%) & $m$ & MNIST MSE & $m$\\",
        r"\midrule",
    ]
    for method in ORDER:
        if method == "dense_adam":
            continue
        cells = []
        for dataset in DATASETS[2:]:
            row = next(
                (
                    item
                    for item in summary
                    if item["dataset"] == dataset and item["method"] == method
                ),
                None,
            )
            if row is None:
                cells += ["--", "--"]
            else:
                cells += [
                    f"{100 * row['mean']:.2f} $\\pm$ {100 * row['spread']:.2f}"
                    if dataset == "Aug-Omniglot"
                    else f"{row['mean']:.3f}",
                    str(row["codes"]) if row["codes"] else "--",
                ]
        if method in ("mask_only", "msr_native"):
            lines.append(r"\midrule")
        lines.append(name(method, DATASETS[2]) + " & " + " & ".join(cells) + r"\\")
    lines += [
        r"\bottomrule\end{tabular}",
        r"\caption{Real-data transfer with fresh task parameters. Omniglot shows 95\% $t$ intervals across episodes, conditional on one source cohort. MNIST uses 32 new digit-value tasks. Mask-only controls use more independent values, explicitly shown by $m$. The final row additionally transfers learned numerical initialization and is a separate regime. Conditional ablations are tested only on MNIST.}",
        r"\label{tab:benchmark-real}",
        r"\end{table}",
    ]
    (paper / "tables" / "benchmark_transfer.tex").write_text("\n".join(lines) + "\n")


def table_costs(paper, costs):
    lines = []
    for dataset in DATASETS:
        lines += [
            r"\begin{table}[htbp]",
            r"\centering\small\setlength{\tabcolsep}{3.5pt}",
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            r"Method & Search s$^*$ & Search TFLOPs$^*$ & Target GFLOPs$^\dagger$ & ms/task & Target MiB & Search MiB\\",
            r"\midrule",
        ]
        for method in ORDER:
            row = next(
                (
                    item
                    for item in costs
                    if item["dataset"] == dataset and item["method"] == method
                ),
                None,
            )
            if row is None:
                continue
            values = [
                row["profiled_offline_seconds"],
                None
                if row["offline_torch_flops"] is None
                else row["offline_torch_flops"] / 1e12,
                None
                if row["target_torch_flops_per_task_profiled"] is None
                else row["target_torch_flops_per_task_profiled"] / 1e9,
                row["target_seconds_per_task_unprofiled"] * 1000,
                row["target_peak_mib"],
                row["offline_peak_mib"],
            ]
            lines.append(
                name(method, dataset)
                + " & "
                + " & ".join(short_number(value) for value in values)
                + r"\\"
            )
        lines += [
            r"\bottomrule\end{tabular}",
            f"\\caption{{{dataset}: measured computation. Target timing excludes the FLOP profiler and uses one process per GPU. $^*$Search seconds include profiling overhead and shared scheduling. $^\\dagger$Target GFLOPs are per task: the complete 64/256/1024 sweep on synthetic data, one prescribed support budget on real data; synthetic time covers only 1024. These are incomplete Torch operation counts, not FLOP/s. A dash indicates unavailable measurement.}}",
            r"\end{table}",
        ]
    (paper / "tables" / "benchmark_costs.tex").write_text("\n".join(lines) + "\n")


def table_parameters(paper, costs):
    lines = [
        r"\begin{table}[htbp]\centering\small",
        r"\begin{tabular}{lrrrr}\toprule",
        r"Method & Pattern & DeepSets & Omniglot & MNIST\\\midrule",
    ]
    for method in ORDER:
        if method == "dense_adam":
            continue
        cells = []
        for dataset in DATASETS:
            row = next(
                (
                    item
                    for item in costs
                    if item["dataset"] == dataset and item["method"] == method
                ),
                None,
            )
            cells.append(
                "--" if row is None else str(row["target_numerical_parameters"])
            )
        lines.append(NAMES[method] + " & " + " & ".join(cells) + r"\\")
    lines += [
        r"\bottomrule\end{tabular}",
        r"\caption{Total numerical parameters fitted per target task, including hidden biases, the free output matrix and output offset. Dense Adam and Dense Adam plus GN have identical parameter counts. The fixed transferred structure is excluded. Mask controls count active weights; their current optimizer still allocates the full masked first-layer tensor (3201 coordinates on MNIST), distinguished in CSV. The native MSR row transfers learned initialization.}",
        r"\label{tab:benchmark-parameters}\end{table}",
    ]
    (paper / "tables/benchmark_parameters.tex").write_text("\n".join(lines) + "\n")


def table_recovery(paper, recovery):
    lines = [
        r"\begin{table}[htbp]",
        r"\centering\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Pattern $0$ & Pattern $0.1$ & DeepSets $0$ & DeepSets $0.1$\\",
        r"\midrule",
    ]
    for method in ("current", "lutq", "yeh_map", "random", "current_bank"):
        cells = []
        for dataset, condition in (
            (DATASETS[0], 0),
            (DATASETS[0], 1),
            (DATASETS[1], 0),
            (DATASETS[1], 1),
        ):
            rows = [
                row
                for row in recovery
                if row["dataset"] == dataset
                and row["method"] == method
                and int(row["condition"]) == condition
            ]
            exact = sum(str(row["exact"]).lower() == "true" for row in rows)
            cells.append(f"{exact}/{len(rows)}")
        lines.append(
            ("Our bank coverage" if method == "current_bank" else NAMES[method])
            + " & "
            + " & ".join(cells)
            + r"\\"
        )
    lines += [
        r"\bottomrule\end{tabular}",
        r"\caption{Exact generating indexator recovery under the stated code-pair/sign/hidden-unit symmetries, three cases per cell. Bank coverage is a late descriptive audit, not an oracle selection rule. Full mask precision/recall and sharing ARI are supplied in CSV. Continuous bases have no discrete recovery score.}",
        r"\label{tab:benchmark-recovery}",
        r"\end{table}",
    ]
    (paper / "tables" / "benchmark_recovery.tex").write_text("\n".join(lines) + "\n")


def save_figure(fig, paper, name):
    fig.savefig(paper / "plots" / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(paper / "plots" / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plots(base, paper, summary):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS[:2]):
        for method in ORDER:
            rows = sorted(
                [
                    row
                    for row in summary
                    if row["dataset"] == dataset
                    and row["method"] == method
                    and row["condition"] == 0
                ],
                key=lambda item: item["support_fit"],
            )
            if not rows:
                continue
            ax.plot(
                [row["support_fit"] for row in rows],
                [max(row["mean"], 1e-16) for row in rows],
                marker="o",
                label=name(method, dataset),
                color=COLORS[method],
                linewidth=2.3 if method == "current" else 1.5,
            )
        ax.set(
            xscale="log",
            yscale="log",
            title=dataset,
            xlabel="Fitting observations (plus N/4 selection)",
            ylabel="New-task query MSE (floor 1e-16)",
        )
        ax.set_xticks((64, 256, 1024), ("64", "256", "1024"))
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    save_figure(fig, paper, "benchmark_transfer")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), constrained_layout=True)
    root = base / "synthetic_confirmation"
    for ax, case, title in zip(axes, ("case_00", "case_03"), DATASETS[:2]):
        for method in ("dense", "lutq", "yeh_map", "yeh_soft", "msr", "current"):
            trace = np.load(root / "targets" / f"{case}_c0" / method / "curves.npz")
            loss = trace["validation_history_1024"]
            curve = np.minimum.accumulate(loss.min(1), axis=0).mean(1)
            ax.plot(
                trace["steps_1024"],
                np.maximum(curve, 1e-16),
                label=name(method, title),
                color=COLORS[method],
            )
        ax.set(
            yscale="log",
            xlabel="Adam updates",
            ylabel="Numerical-selection normalized MSE",
            title=title,
        )
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    save_figure(fig, paper, "benchmark_numeric_loss")
    aligned = np.load(base / "report" / "aligned_indices_for_display.npz")
    export = {key: aligned[key].tolist() for key in aligned.files}
    for case, title, codes in (
        ("case_00", "Pattern 0110", 2),
        ("case_03", "Synthetic DeepSets", 4),
    ):
        fig, axes = plt.subplots(2, 5, figsize=(14, 6.5), constrained_layout=True)
        palette = [
            plt.get_cmap("RdBu_r")((value + codes) / (2 * codes))
            for value in range(-codes, codes + 1)
        ]
        palette[codes] = (1, 1, 1, 1)
        cmap, norm = (
            ListedColormap(palette),
            BoundaryNorm(np.arange(-codes - 0.5, codes + 1.5), 2 * codes + 1),
        )
        for condition in (0, 1):
            truth = aligned[f"{case}_truth"]
            methods = ("truth", "current", "lutq", "yeh_map")
            for column, method in enumerate(methods):
                index = (
                    truth
                    if method == "truth"
                    else aligned[f"{case}_c{condition}_{method}"]
                )
                axes[condition, column].imshow(
                    index, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest"
                )
                axes[condition, column].set_title(
                    "Generating I" if method == "truth" else NAMES[method], fontsize=10
                )
                axes[condition, column].set_xlabel("Hidden unit")
                axes[condition, column].set_ylabel(
                    f"Input; source noise {0.1 * condition:g}"
                )
                export[f"{case}_c{condition}_{method}"] = index.tolist()
            dense = np.load(root / "dense" / f"{case}_c{condition}" / "fitted.npz")[
                "parameters"
            ][0]
            matrix = dense[: truth.size].reshape(truth.shape)
            axes[condition, 4].imshow(
                matrix,
                cmap="RdBu_r",
                aspect="auto",
                vmin=-np.max(np.abs(matrix)),
                vmax=np.max(np.abs(matrix)),
            )
            axes[condition, 4].set_title("Dense Adam W1 (native order)", fontsize=10)
            axes[condition, 4].set_xlabel("Hidden unit")
        fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=axes[:, :4],
            ticks=np.arange(-codes, codes + 1),
            shrink=0.65,
            label="Signed group label; white = disconnected",
        )
        fig.suptitle(
            f"{title}: native selected indexators; display alignment only after freeze",
            fontsize=12,
        )
        save_figure(
            fig,
            paper,
            "benchmark_pattern_heatmaps"
            if codes == 2
            else "benchmark_deepsets_heatmaps",
        )
    for dataset in REAL_DATA_NAMES:
        source = base / dataset
        methods = ["current", "lutq", "yeh_map", "random"] + (
            ["current_functional", "mask_only", "mask_random"]
            if dataset == "digit_sum_pilot"
            else []
        )
        for method in methods:
            export[f"{dataset}_{method}"] = np.load(
                source / "targets" / method / "model.npz"
            )["index"].tolist()
    (paper / "indexators" / "selected_indices.json").write_text(
        json.dumps(
            {
                "representation": "Signed labels, zero disconnected; group names arbitrary. Synthetic display arrays aligned using truth only after freeze. Original raw indexators remain in frozen target model archives.",
                "indices": export,
            },
            indent=2,
        )
        + "\n"
    )
    real_plots(base, paper, summary)


def real_plots(base, paper, summary):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS[2:]):
        rows = [
            next(
                item
                for item in summary
                if item["dataset"] == dataset and item["method"] == method
            )
            for method in ORDER
            if any(
                item["dataset"] == dataset and item["method"] == method
                for item in summary
            )
        ]
        multiplier = 100 if dataset == "Aug-Omniglot" else 1
        y = np.arange(len(rows))
        bars = ax.barh(
            y,
            [multiplier * row["mean"] for row in rows],
            xerr=[multiplier * row["spread"] for row in rows],
            color=[COLORS[row["method"]] for row in rows],
            capsize=3,
        )
        for bar, row in zip(bars, rows):
            if row["method"] in ("mask_only", "mask_random", "msr_native"):
                bar.set_hatch("//")
            width = bar.get_width()
            ax.text(
                width
                + multiplier * row["spread"]
                + (0.3 if multiplier == 100 else 0.01),
                bar.get_y() + bar.get_height() / 2,
                f"{width:.2f}" if multiplier == 100 else f"{width:.3f}",
                va="center",
                fontsize=9,
            )
        ax.set_yticks(y, [name(row["method"], dataset) for row in rows], fontsize=9)
        ax.invert_yaxis()
        ax.set(
            title=dataset,
            xlabel="Query accuracy (%) — higher is better"
            if multiplier == 100
            else "Query MSE — lower is better",
        )
        ax.set_xlim(0, max(multiplier * (r["mean"] + r["spread"]) for r in rows) * 1.15)
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Fresh numerical fitting; hatching marks additional transfer or larger budgets"
    )
    save_figure(fig, paper, "benchmark_real_transfer")

    root = base / "digit_sum_pilot"
    original = np.load(root / "targets/current/model.npz")["index"]
    functional = np.load(root / "targets/current_functional/model.npz")["index"]
    random_mask = np.load(root / "targets/mask_random/model.npz")["index"] != 0
    models = np.load(root / "source/dense/models.npz")
    dense = models["parameters"][0, : original.size].reshape(original.shape)
    fig, axes = plt.subplots(1, 5, figsize=(15, 5.8), constrained_layout=True)
    scale = max(np.abs(dense).max(), 1e-12)
    image = axes[0].imshow(dense, cmap="RdBu_r", vmin=-scale, vmax=scale, aspect="auto")
    axes[0].set_title("Source Dense Adam W1\n(native hidden order)", fontsize=10)
    fig.colorbar(image, ax=axes[0], shrink=0.5)
    for ax, matrix, title in (
        (axes[1], original != 0, "Learned mask\n627 active connections"),
        (axes[2], original, "Selected I\n128 independent values"),
        (axes[3], functional, "Functional refinement I\n128 independent values"),
        (axes[4], random_mask, "Random mask\nsame per-neuron degrees"),
    ):
        is_mask = matrix.dtype == np.bool_
        cmap = "Greys" if is_mask else "RdBu_r"
        bound = 1 if is_mask else max(np.abs(matrix).max(), 1)
        ax.imshow(
            matrix, cmap=cmap, vmin=0 if is_mask else -bound, vmax=bound, aspect="auto"
        )
        ax.set_title(title, fontsize=10)
    for ax in axes:
        ax.set_xlabel("Hidden unit")
        ax.set_ylabel("Input coordinate")
    fig.suptitle(
        "Paired MNIST: observed source weights and transferred constraints; no reference I"
    )
    save_figure(fig, paper, "benchmark_mnist_heatmaps")

    root = base / "omniglot_pilot"
    index = np.load(root / "targets/current/model.npz")["index"]
    parameters = np.load(root / "source/dense/models.npz")["parameters"]
    dense = parameters[0, : index.size].reshape(index.shape)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), constrained_layout=True)
    for ax, matrix, title, cmap in (
        (axes[0], dense, "Source Dense W1: native unit order", "RdBu_r"),
        (axes[1], index != 0, "Transferred binary mask", "Greys"),
        (axes[2], index, "Transferred signed I (m = 64)", "RdBu_r"),
    ):
        tiles = np.full((4 * 30, 4 * 30), np.nan)
        for unit in range(16):
            row, column = divmod(unit, 4)
            tiles[row * 30 : row * 30 + 28, column * 30 : column * 30 + 28] = matrix[
                :, unit
            ].reshape(28, 28)
        bound = max(float(np.abs(matrix).max()), 1e-12)
        image = ax.imshow(
            tiles, cmap=cmap, vmin=0 if matrix.dtype == np.bool_ else -bound, vmax=bound
        )
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, shrink=0.6)
    fig.suptitle(
        "Omniglot: each tile is one hidden unit; no reference network indexator"
    )
    save_figure(fig, paper, "benchmark_omniglot_heatmaps")


REAL_DATA_NAMES = ("omniglot_pilot", "digit_sum_pilot")


def paired_comparisons(transfer):
    rows = []
    for dataset in DATASETS[2:]:
        metric = "accuracy" if dataset == "Aug-Omniglot" else "mse"
        grouped = defaultdict(dict)
        for row in transfer:
            if row["dataset"] == dataset:
                grouped[row["method"]][int(row["task"])] = float(row[metric])
        for method, values in grouped.items():
            if method == "dense":
                continue
            delta = np.array(
                [value - grouped["dense"][key] for key, value in values.items()]
            )
            half = (
                student_t.ppf(0.975, len(delta) - 1)
                * delta.std(ddof=1)
                / np.sqrt(len(delta))
            )
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "metric": metric,
                    "difference_from_dense": delta.mean(),
                    "paired_95_lower": delta.mean() - half,
                    "paired_95_upper": delta.mean() + half,
                    "source_cohorts": 1,
                    "tasks": len(delta),
                }
            )
    return rows


def binary_table(paper, binary):
    lines = [
        r"\begin{table}[htbp]\centering\small",
        r"\begin{tabular}{lrr}\toprule",
        r"Method & Clean-source I & Noisy-source I\\\midrule",
    ]
    for method in METHODS_BINARY:
        values = [
            np.mean(
                [
                    float(row["accuracy"])
                    for row in binary
                    if row["method"] == method and int(row["condition"]) == c
                ]
            )
            for c in (0, 1)
        ]
        lines.append(
            name(method, DATASETS[2])
            + " & "
            + " & ".join(f"{100 * value:.2f}" for value in values)
            + r"\\"
        )
    lines += [
        r"\bottomrule\end{tabular}",
        r"\caption{Binary Pattern follow-up: mean query accuracy (\%) over three families, with 512 support, 128 numerical-selection and 1408 held-out binary inputs. All numerical parameters are trained afresh by binary cross entropy. These structures were extracted from real-valued source functions, so this is not extraction from binary supervision alone.}",
        r"\label{tab:benchmark-binary}\end{table}",
    ]
    (paper / "tables/benchmark_binary_pattern.tex").write_text("\n".join(lines) + "\n")


METHODS_BINARY = ("dense", "random", "lutq", "yeh_map", "yeh_soft", "msr", "current")


def companion(base, paper, summary, costs, recovery, binary):
    lines = [
        "# Сравнение переноса структуры",
        "",
        "Все численные параметры новых задач обучаются заново. Исключение — отдельно отмеченный native MSR, переносящий также обученную инициализацию. Истинные параметры, маски и группы не использовались при извлечении или выборе моделей; синтетическая структура открыта только после фиксации предсказаний.",
        "",
        "## Метод",
        "",
        "Для первого слоя ищем общий целочисленный индексатор J. На задаче τ он задаёт Wτ[d,h] = sign(J[d,h]) cτ[abs(J[d,h])], где индекс0 означает нулевой вес. Обучаемые числа cτ, сдвиги bτ, свободная выходная матрица Aτ и выходной сдвиг dτ принадлежат конкретной задаче. Используется парный кодбук(0,cτ,−cτ); m — число независимых чисел.",
        "",
        "Плотные source-модели обучаются только по наблюдениям. Их нормированные направления выравниваются по скрытым нейронам. Совместная проекция подбирает общие группы и неизвестные масштабы столбцов, проверяя согласованность по нескольким задачам. Разные маски и старты дают конечный банк. Для каждого I независимо подгоняются числа; численный checkpoint выбирает H0, а отдельная source M — общий I. На новой задаче переносится только I, все числа инициализируются заново. Полное математическое описание и два псевдоалгоритма находятся в Method рукописи.",
        "",
        "## Основные выводы",
        "",
        "Pattern: точный I найден в 6/6 случаев, как и у LUT-Q; отдельная бинарная проверка даёт 100% accuracy. Чистый DeepSets: наш exact I — 3/3, LUT-Q — 2/3, но LUT-Q имеет меньшее MSE после конечного численного обучения (0.000129 против 0.000610). Ненулевой MSE при exact I показывает оставшийся вклад численной оптимизации. При source noise 0.1 оба метода имеют exact 0/3; у нас правильного I нет ни в одном банке, а маска точна в 2/3 случаев. Это ограничение генерации кандидатов, не только выбора чисел.",
        "",
        "Omniglot: 37.73% против Dense 37.14%; парная разница +0.59 процентного пункта, 95% интервал примерно от −0.02 до +1.19. По одной source-серии нельзя утверждать устойчивое преимущество над Dense. Среди исследованных вариантов с ограниченным числом численных параметров наш результат лучший. MNIST: исходный шэринг уступает Dense, а освобождение весов в найденной маске почти закрывает разрыв при большем численном бюджете. Зарегистрированная функциональная поправка не улучшила перенос и не принята.",
        "",
        "## Итоговое качество",
        "",
        "Синтетика: среднее MSE по трём семьям, каждая содержит 8 новых функций; разброс — SD между семьями. Real: одна source-серия; интервалы — 95% t-интервалы между эпизодами/задачами. Основной support: 1024 (+256 selection) в синтетике, 25 в Omniglot и 512 в MNIST. Полные парные разницы с Dense находятся в tables/benchmark_paired_comparisons.csv.",
        "",
        "| Набор | Source noise | Метод | Качество | m |",
        "|---|---:|---|---:|---:|",
    ]
    for dataset in DATASETS:
        for method in ORDER:
            for row in [
                item
                for item in summary
                if item["dataset"] == dataset
                and item["method"] == method
                and item["support_fit"]
                == (
                    1024
                    if dataset in DATASETS[:2]
                    else 25
                    if dataset == "Aug-Omniglot"
                    else 512
                )
            ]:
                value = (
                    f"{100 * row['mean']:.2f}% ±{100 * row['spread']:.2f}п.п."
                    if row["metric"] == "accuracy"
                    else f"{row['mean']:.3e} ±{row['spread']:.2e}"
                )
                lines.append(
                    f"| {dataset} | {0.1 * row['condition']:g} | {name(method, dataset)} | {value} | {row['codes'] or '--'} |"
                )
    lines += [
        "",
        "## Число обучаемых параметров",
        "",
        "Считаем все численные параметры одной новой задачи: D — размер входа, H — число скрытых нейронов, O — число выходов, m — число независимо обучаемых значений первого слоя. Нулевой код и отрицательные копии значений не добавляют параметров. Фиксированный индексатор или непрерывный базис на target не обучается и в этот счёт не входит.",
        "",
        "$$",
        r"\begin{aligned}",
        r"N_{\mathrm{structured}} &= m + H + HO + O,\\",
        r"N_{\mathrm{dense}} &= DH + H + HO + O.",
        r"\end{aligned}",
        "$$",
        "",
        "Для маски со свободными весами вместо m берётся число активных связей. Это число координат модели, а не число одновременно обучаемых рестартов или source-кандидатов. В текущей реализации маски оптимизатор хранит полный массив с заблокированными неактивными весами: для MNIST 3201 координата буфера против 692 активных численных параметров. Оба числа записаны отдельно в CSV.",
        "",
        "| Метод | Pattern | DeepSets | Omniglot | MNIST |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in ORDER:
        if method == "dense_adam":
            continue
        cells = [
            next(
                (
                    str(item["target_numerical_parameters"])
                    for item in costs
                    if item["dataset"] == dataset and item["method"] == method
                ),
                "--",
            )
            for dataset in DATASETS
        ]
        lines.append(f"| {NAMES[method]} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Вычисления",
        "",
        "Target время измерено отдельно без FLOP-профайлера, по одному процессу на GPU. Это амортизированная стоимость packed-обучения задачи, не latency одиночного запроса. Синтетическое время — заранее выбранная первая семья и noise0; качество выше усреднено по трём семьям. Source/search секунды — сумма измеренных стадий со всеми гипервариантами; они включают накладные расходы FLOP-профайлера и совместное планирование. Torch FLOPs неполны: не учитывают NumPy/SciPy и неподдерживаемые операции. Dense не требует source-extractor; в real его подготовка включает выбор численного правила на development-задачах. Random I использует найденные у G34 бюджет/плотность и является диагностическим контролем.",
        "",
        "Target измерение включает адаптацию и предсказание на query X; память — максимум всего packed-процесса, а не размер хранимого I/C. Синтетический Adam-only контроль остановлен до GN. Пик его памяти отдельно не записан и оставлен пустым. В MNIST текущая реализация базиса материализует промежуточные признаки и требует около 1096 MiB против 71 MiB у Dense. Преимущества по памяти или универсального ускорения здесь нет.",
        "",
        "FLOPs — объём вычислений; FLOP/s — скорость их выполнения. В отчёте приведены FLOPs и отдельное время, без заявления о скорости GPU в FLOP/s. Target GFLOPs в синтетике охватывают суммарный sweep 64/256/1024, а время — только 1024; делить эти столбцы друг на друга нельзя. В real GFLOPs относятся к полной адаптации и предсказанию одной задачи при указанном support-бюджете. Все счётчики неполны: учитывают поддерживаемые операции PyTorch.",
        "",
        "| Набор | Метод | Search s (profiled) | Torch TFLOPs(search) | Target GFLOPs/task | Target ms/task(no profiler) | Target peak MiB |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in costs:
        lines.append(
            f"| {row['dataset']} | {name(row['method'], row['dataset'])} | {short_number(row['profiled_offline_seconds'])} | {short_number(None if row['offline_torch_flops'] is None else row['offline_torch_flops'] / 1e12)} | {short_number(None if row['target_torch_flops_per_task_profiled'] is None else row['target_torch_flops_per_task_profiled'] / 1e9)} | {1000 * row['target_seconds_per_task_unprofiled']:.2f} | {short_number(row['target_peak_mib'])} |"
        )
    lines += [
        "",
        "## Проверка структуры",
        "",
        "Exact означает совпадение порождающего signed I с точностью до переименования/смены знака пар кодов и перестановки скрытых нейронов. Это не полная проверка всех функциональных эквивалентностей. Непрерывные Yeh/MSR не превращались в индексаторы и не получают такой метрики. Real-данные не имеют истинного нейронного индексатора.",
        "",
        "| Набор | Noise | Метод | Exact | Mask recall | Mask precision |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for dataset in DATASETS[:2]:
        for condition in (0, 1):
            for method in ("current", "lutq", "yeh_map", "random", "current_bank"):
                selected = [
                    row
                    for row in recovery
                    if row["dataset"] == dataset
                    and int(row["condition"]) == condition
                    and row["method"] == method
                ]
                exact = sum(str(row["exact"]).lower() == "true" for row in selected)
                recall = (
                    np.mean(
                        [
                            float(row["mask_recall"])
                            for row in selected
                            if row.get("mask_recall")
                        ]
                    )
                    if method != "current_bank"
                    else None
                )
                precision = (
                    np.mean(
                        [
                            float(row["mask_precision"])
                            for row in selected
                            if row.get("mask_precision")
                        ]
                    )
                    if method != "current_bank"
                    else None
                )
                lines.append(
                    f"| {dataset} | {0.1 * condition:g} | {NAMES.get(method, 'G34 bank')} | {exact}/{len(selected)} | {short_number(recall)} | {short_number(precision)} |"
                )
    lines += [
        "",
        "## Условная поправка на MNIST",
        "",
        "На development-задачах исходный G34 уступил Dense: 1.203 против 1.082 MSE, хуже на 29/32 задачах. Поэтому до открытия тестовых ответов проверена отдельная source-only абляция: чередовать обучение чисел с изменениями hard I. Для каждой возможной замены оценивается среднее g·ΔW+0.5·diag(GGN)·ΔW²; допускается также индекс 0. Пакеты изменений принимаются только после уменьшения фактического source H0 loss. Банк затем проходит тот же численный refit и выбор по source M. Истинные параметры и координатные роли не используются.",
        "",
        "Эта поправка не принята в основной метод: development MSE стал 1.238 вместо 1.203, хотя собственный source H0 loss уменьшился с 0.574 до 0.433. Финальный тест тоже не показал улучшения: 1.192 против 1.185. Оптимизация чисел и меток на старых задачах сама по себе не гарантирует перенос шэринга.",
        "",
        "Второй диагностический контроль оставляет найденную бинарную маску и освобождает веса внутри неё. Development MSE 1.085 приблизился к Dense 1.082; финальный тест дал 1.062 против Dense 1.058. Случайная маска с теми же степенями скрытых нейронов дала 1.145 на тесте. При этом свободны 627 первых весов, а не 128 кодов (у Dense — 3136). Это свидетельство полезности подключений; оно не подтверждает правильность групп и не является выигрышем при одинаковом бюджете.",
        "",
        "Незавершённая первая попытка имела ошибку пересчёта масштаба кода и исключена. Исправленная попытка проверяет сохранение стартового predictor: оба source H0 loss равны0.574105. Исходные результаты G34 и конкурентов сохранены.",
        "",
        "## Границы сравнения",
        "",
        "Yeh,MSR,LUT-Q портированы на один fully connected слой и общий парный словарь. Это не буквальные исходные CNN/embedding архитектуры. В частности, опубликованные MSR97.7±0.2% на Aug-Omniglot относятся к CNN и переносу численной инициализации при значительно большем source-бюджете; их нельзя ставить в одну matched-строку с нашим FC/structure-only экспериментом. MNIST здесь — уменьшенное multitask-расширение, а не опубликованный InfiMNIST набор5..50 изображений. Эти эксперименты не устанавливают SOTA на исходных real benchmarks.",
        "",
        "Более точный source-контроль Yeh (100 inner steps и явный damped Hessian вместо20/Neumann20) на одном DeepSets-пилоте дал MAP source MSE0.04325 вместо0.04854, soft0.02507 вместо0.02905; G34 на нём0.001762. Это уменьшает опасение, что весь разрыв вызван облегчённым solver, но не доказывает оптимальную настройку Yeh во всех постановках.",
        "",
        "## Артефакты",
        "",
        "- `main.pdf`: рукопись с определениями, формулами и алгоритмами.",
        "- `tables/`: качество по задачам/семьям, затраты, recovery и binary Pattern контроль.",
        "- `plots/`: перенос, численные loss-кривые и heatmaps.",
        "- `indexators/selected_indices.json`: выбранные signed индексаторы.",
        "- [Компактный комплект для пересчёта метрик и рисунков](../artifacts/benchmark_2026-11-07/README.md) содержит точные предсказания, ответы для аудита, выбранные структуры и протоколы. Пересчёт не требует обучения или GPU.",
        "- Полные наблюдения, численные состояния и исходные frozen-артефакты остаются локально в experiments/paper_structure_benchmark_v1/results/20261107/ и исключены из новых коммитов.",
    ]
    (paper / "BENCHMARK_REPORT.md").write_text("\n".join(lines) + "\n")


def run(base, paper):
    if not (base / "report" / "scoring.json").exists():
        raise RuntimeError("Scoring must follow the global prediction freeze")
    for directory in ("tables", "plots", "indexators"):
        (paper / directory).mkdir(exist_ok=True)
    transfer = load_csv(base / "report/transfer_per_task.csv")
    recovery = load_csv(base / "report/structural_recovery.csv")
    binary = load_csv(base / "report/binary_classification.csv")
    summary = aggregate(transfer)
    costs = cost_rows(base)
    for row in summary:
        matched = next(
            item
            for item in costs
            if item["dataset"] == row["dataset"] and item["method"] == row["method"]
        )
        row.update(
            {
                key: value
                for key, value in matched.items()
                if key.startswith("target_") and "parameters" in key
            }
        )
    write_csv(paper / "tables/benchmark_transfer.csv", summary)
    write_csv(paper / "tables/benchmark_costs.csv", costs)
    write_csv(
        paper / "tables/benchmark_parameters.csv",
        [
            {
                key: value
                for key, value in row.items()
                if key in ("dataset", "method", "parameter_count_scope")
                or key.startswith("target_")
                and ("parameters" in key or "coordinates" in key)
            }
            for row in costs
        ],
    )
    write_csv(
        paper / "tables/benchmark_paired_comparisons.csv", paired_comparisons(transfer)
    )
    for original, new in (
        ("transfer_per_task.csv", "benchmark_per_task.csv"),
        ("structural_recovery.csv", "benchmark_recovery.csv"),
        ("binary_classification.csv", "benchmark_binary_pattern.csv"),
    ):
        shutil.copy2(base / "report" / original, paper / "tables" / new)
    table_quality(paper, summary)
    table_costs(paper, costs)
    table_parameters(paper, costs)
    table_recovery(paper, recovery)
    binary_table(paper, binary)
    plots(base, paper, summary)
    companion(base, paper, summary, costs, recovery, binary)
    print(
        json.dumps(
            {"paper": str(paper), "quality_rows": len(summary), "cost_rows": len(costs)}
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--paper", type=Path, required=True)
    args = parser.parse_args()
    run(args.base.resolve(), args.paper.resolve())


if __name__ == "__main__":
    main()
