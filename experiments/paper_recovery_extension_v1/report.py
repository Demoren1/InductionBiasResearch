"""Render the frozen comparison into concise manuscript tables and figures."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

from .prepare import METHODS, OUTPUT

LABEL = {
    "dense": "Dense",
    "g34": "G34",
    "asrr": "aSRR",
    "pair": "Pair recovery",
    "lutq": "LUT-Q",
    "yeh_soft": "Yeh (soft)",
    "yeh_map": "Yeh (MAP)",
    "msr": "MSR-FC",
}
DATASETS = (("mnist", "MNIST"), ("fashion", "Fashion-MNIST"), ("har", "UCI HAR"))
# Direct pair recovery is deferred from the manuscript pending further work.
PAPER_METHODS = tuple(method for method in METHODS if method != "pair")


def constrained(summary, method):
    codes = summary["decisions"][method]["best_constrained_codes"]
    name = "dense" if method == "dense" else f"{method}_k{codes}"
    return next(r for r in summary["all_budgets"] if r["name"] == name)


def table(dataset, title, summary, methods=PAPER_METHODS):
    lines = [
        r"\begin{table}[htbp]",
        r"    \centering\small",
        r"    \begin{tabular}{lrrrrr}",
        r"        \toprule",
        r"        Method & Budget & Parameters & Updates & Accuracy (\%) & CE\\",
        r"        \midrule",
    ]
    for method in methods:
        row = constrained(summary, method)
        marker = (
            r"$^{\dagger}$"
            if method != "dense"
            and summary["decisions"][method]["best_including_unshared"] == "unshared"
            else ""
        )
        lines.append(
            f"        {LABEL[method]}{marker} & {row['codes']:,} & "
            f"{row['task_parameters']:,} & {row['numeric_updates']} & "
            f"${100 * row['accuracy']:.2f}\\pm{100 * row['accuracy_half_ci']:.2f}$ & "
            f"{row['cross_entropy']:.3f}\\\\"
        )
    unit = "nine subject means" if dataset == "har" else "96 episodes"
    lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        f"    \\caption{{{title}: development-selected constrained budgets and the Dense reference.",
        "        Budgets are 8--512; parameter counts include all allocated numerical coordinates.",
        r"        $\dagger$ marks rows for which allowing one code per weight instead selects",
        "        the common full-budget control, whose predictions equal Dense exactly.",
        f"        Intervals use {unit}.}}",
        f"    \\label{{tab:recovery-{dataset}}}",
        r"\end{table}",
    ]
    Path(f"paper/sections/recovery_{dataset}_results.tex").write_text(
        "\n".join(lines) + "\n"
    )


def budget_figure(summaries, methods=PAPER_METHODS):
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.1), constrained_layout=False)
    colors = dict(zip(METHODS[1:], plt.cm.tab10.colors))
    colors["asrr"] = "tab:brown"
    for axis, (dataset, title), summary in zip(axes, DATASETS, summaries):
        for method in methods[1:]:
            color = colors[method]
            rows = sorted(
                (r for r in summary["all_budgets"] if r["method"] == method),
                key=lambda r: r["codes"],
            )
            axis.plot(
                [r["codes"] for r in rows],
                [100 * r["accuracy"] for r in rows],
                color=color,
                marker="o",
                ms=3,
                lw=1.2,
                label=LABEL[method],
            )
            best = constrained(summary, method)
            axis.scatter(
                best["codes"],
                100 * best["accuracy"],
                facecolor="none",
                edgecolor=color,
                s=65,
                linewidth=1.2,
                zorder=4,
            )
        dense = constrained(summary, "dense")
        axis.axhline(
            100 * dense["accuracy"],
            color="black",
            ls="--",
            lw=1,
            label="Dense / full budget",
        )
        axis.scatter(
            dense["codes"], 100 * dense["accuracy"], color="black", marker="D", s=16
        )
        axis.set_xscale("log", base=2)
        axis.set_xticks(
            [8, 32, 128, 512, dense["codes"]], ["8", "32", "128", "512", "full"]
        )
        axis.set_title(title)
        axis.set_xlabel("Code / latent budget")
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("Query accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=4 if len(methods) > 6 else 3,
        frameon=False, fontsize=10,
    )
    fig.subplots_adjust(left=0.065, right=0.99, top=0.88, bottom=0.30, wspace=0.27)
    for suffix in ("pdf", "png"):
        fig.savefig(
            Path("paper/plots") / f"recovery_budget_comparison.{suffix}", dpi=180
        )
    plt.close(fig)


def dense_quality_figure():
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
    for axis, (dataset, title) in zip(axes, DATASETS):
        model = dict(np.load(OUTPUT / dataset / "dense_curves.npz"))
        best = model["numerical_loss"].argmin(0)
        tasks = np.arange(len(best))
        for key, label, style in (
            ("train_history", "Training", "-"),
            ("validation_history", "Numerical selection", "--"),
        ):
            curve = model[key][:, best, tasks].mean(-1)
            axis.plot(model["steps"], np.maximum(curve, 1e-9), style, label=label)
        quality = json.loads((OUTPUT / dataset / "dense_quality.json").read_text())
        axis.scatter(
            model["steps"][-1],
            np.mean(quality["H_ce"]),
            marker="D",
            s=18,
            color="tab:orange",
            label="After polishing",
        )
        axis.set_yscale("log")
        axis.yaxis.set_minor_formatter(NullFormatter())
        if dataset == "fashion":
            axis.set_yticks([0.03, 0.1, 0.3], ["0.03", "0.1", "0.3"])
        axis.set_title(title)
        axis.set_xlabel("Adam updates")
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("Source cross-entropy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.87, bottom=0.30, wspace=0.27)
    for suffix in ("pdf", "png"):
        fig.savefig(Path("paper/plots") / f"recovery_dense_quality.{suffix}", dpi=180)
    plt.close(fig)


def scientific(value):
    if 1e-3 <= value < 1e3:
        return f"{value:.3g}"
    number, exponent = f"{value:.2e}".split("e")
    return f"${number}\\times10^{{{int(exponent)}}}$"


def synthetic_table(synthetic, methods=PAPER_METHODS):
    lines = [
        r"\begin{table}[htbp]",
        r"    \centering\small",
        r"    \begin{tabular}{lrr}",
        r"        \toprule",
        r"        Method & Geometry A: MSE & Geometry B: MSE\\",
        r"        \midrule",
    ]
    for method in methods:
        scores = [
            next(
                r["mse"]
                for r in synthetic["rows"]
                if r["method"] == method and r["geometry"] == n
            )
            for n in (4, 5)
        ]
        lines.append(
            f"        {LABEL[method]} & {scientific(scores[0])} & {scientific(scores[1])}\\\\"
        )
    lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        r"    \caption{Fresh numerical transfer on the same two sharing geometries.",
        "        Mean query MSE over four new functions per geometry, with 256 fitting,",
        "        64 numerical-selection and 2048 query examples per function. Each",
        "        structured model has four code or latent coordinates; continuous",
        "        competitors retain their native bases. All methods use fresh numerical parameters.}",
        r"    \label{tab:recovery-synthetic}",
        r"\end{table}",
    ]
    Path("paper/sections/recovery_synthetic_results.tex").write_text(
        "\n".join(lines) + "\n"
    )


def run():
    summaries = [
        json.loads((OUTPUT / ds / "summary.json").read_text()) for ds, _ in DATASETS
    ]
    for (dataset, title), summary in zip(DATASETS, summaries):
        table(dataset, title, summary)
    budget_figure(summaries)
    dense_quality_figure()
    synthetic = json.loads((OUTPUT / "synthetic_results.json").read_text())
    synthetic_table(synthetic)
    report = [
        "# Recovery and native competitors: expanded comparison",
        "",
        "14 November 2026. Source and development choices are fixed before drawing the new target cohort.",
        "The original three-method results remain in artifacts/paper_recovery_comparison_2026-11-14.",
        "",
        "## Applied results",
        "",
        "Constrained choices use the common grid 8/16/32/64/128/512. The full-code endpoint is executed separately and equals Dense exactly.",
        "",
        "| Dataset | Method | Constrained budget | Accuracy (%) | Full budget preferred on development |",
        "|---|---|---:|---:|---|",
    ]
    for (dataset, title), summary in zip(DATASETS, summaries):
        for method in METHODS:
            row = constrained(summary, method)
            full = summary["decisions"][method]["best_including_unshared"] == "unshared"
            report.append(
                f"|{title}|{LABEL[method]}|{row['codes']}|{100 * row['accuracy']:.2f} ± {100 * row['accuracy_half_ci']:.2f}|{'Yes' if full else 'No'}|"
            )
    report += [
        "",
        "## Synthetic transfer",
        "",
        "| Method | Geometry A MSE | Geometry B MSE |",
        "|---|---:|---:|",
    ]
    for method in METHODS:
        scores = [
            next(
                r["mse"]
                for r in synthetic["rows"]
                if r["method"] == method and r["geometry"] == n
            )
            for n in (4, 5)
        ]
        report.append(f"|{LABEL[method]}|{scores[0]:.3e}|{scores[1]:.3e}|")
    report += [
        "",
        "## Main findings",
        "",
        "On the same frozen image episodes, Pair accuracy rises from 87.00% to 91.57% on MNIST and from 92.33% to 95.27% on Fashion-MNIST when moving from 64 to 512 codes. The two image extractors use 562 allocated numerical coordinates at 512 codes, compared with 12594 for Dense.",
        "",
        "A full positive unique-code indexator (12544 image weights;8976 HAR weights) produces bit-identical Dense predictions. Matching Dense through this endpoint provides no sharing or compression.",
        "",
        "HAR development selects 32 Pair codes (79.41% test accuracy). The prespecified 512-code evaluation scores 84.89%; it is shown in the curve and does not replace the development choice. Budget selection across people is a limitation of this split. All reported choices remain frozen.",
        "",
        "G34 and Pair recover both synthetic indexators exactly and transfer with nearly zero MSE. These fixed four-code cases with one source function do not imply a general advantage on naturally occurring tasks or against published implementations.",
        "",
        "Yeh uses a 20-term Neumann inverse-Hessian approximation. At budget 512, the final recorded relative residual ranges across the 12 scenarios are 0.90–0.99 (MNIST), 0.70–0.98 (Fashion-MNIST), and 0.27–0.96 (HAR). These diagnostics do not certify an optimized implicit solution; the rows describe this finite-budget adaptation and do not establish limits of the published method.",
        "",
        "## Interpretation and scope",
        "",
        "These are matched fully connected adaptations of LUT-Q, Yeh and MSR, not unchanged author scripts or published benchmark scores. MSR transfers only its basis; learned initialization and rates are discarded. Continuous methods are never post-clustered. All source and target fitting choices are saved.",
        "",
        "Image confidence intervals describe 96 episodes conditional on the fixed class split and source cohort. HAR intervals use nine subject means, each averaging ten episodes. Neither constitutes evidence across all possible dataset environments. The public image class environments were already examined in the original comparison; the new episodes are a conditional follow-up.",
        "",
        "Budget denotes allocated code/latent coordinates. Native discrete methods may leave codes unused. Counts of zero connections and group sizes are retained separately; a larger budget can coexist with a dominant shared group. Numerical parameter counts include biases and the full output layer, but exclude the offline structure/basis storage.",
        "",
        "The finite grid does not establish a globally optimal sharing budget. Native algorithms have different source objectives and costs; shared observations and the common downstream solver do not imply equal total computation. Concurrent GPU runs do not support a wall-time speed claim.",
        "",
        "Source Dense predictive quality is checked on separate numerical-selection examples and curves; global parameter convergence is not claimed. No software tests were created or run.",
        "",
        "Reproduction: experiments/paper_recovery_extension_v1. Compact scores, freezes and development choices: artifacts/paper_recovery_extension_2026-11-14. Raw registered native states and source fits: artifacts/local/paper_recovery_extension_2026-11-14.",
        "",
    ]
    Path("paper/theory/PAPER_RECOVERY_EXTENDED_COMPARISON_2026-11-14.md").write_text(
        "\n".join(report)
    )


if __name__ == "__main__":
    run()
