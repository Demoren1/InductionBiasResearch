"""Report replication and frozen tuned results with their actual budgets."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import SETTINGS
from .replicate import OUTPUT

TITLES = {
    "mnist100": "MNIST 100+50",
    "mnist16": "MNIST 16",
    "fashion100": "Fashion 100+50",
    "fashion16": "Fashion 16",
    "har3": "HAR 3",
    "omniglot10": "Omniglot 10+5",
}
METHODS = {"dense": "Dense", "g34": "G34", "asrr": "aSRR"}


def run():
    results = json.loads((OUTPUT / "tuned_results.json").read_text())
    rows = results["rows"]
    lines = [
        "# Broad aSRR tuning and independent confirmation",
        "",
        "17 November 2026.",
        "",
        "The former successful procedures were first replicated unchanged on",
        "512 fresh episodes per dataset. A separate development search then",
        "selected wider budgets and numerical rules. Two new cohorts evaluate",
        "the frozen tuned policies, without retuning between cohorts.",
        "The primary aSRR policy improves over Dense on both cohorts for",
        "MNIST at both support sizes, HAR and Omniglot. On Fashion 100+50",
        "the paired difference includes zero; on Fashion 16 the primary",
        "aSRR policy loses a small amount of accuracy. The secondary CE",
        "policies remain separately reported rather than replacing these choices.",
        "",
        (OUTPUT / "REPLICATION.md")
        .read_text()
        .replace("# Fresh", "## Fresh", 1)
        .replace(
            "ongoing broader development search", "separate broader search"
        ),
        "",
        "## Tuned primary policies",
        "",
        "Primary selection maximizes development accuracy, breaking ties by CE",
        "and fewer codes. The following pool uses unchanged policies in both",
        "prespecified cohorts. Each image setting has 1024 episodes; HAR 180 episodes",
        "use nine subject means. Environments/classes/subjects were studied before.",
        "",
        "| Setting | Method | Accuracy, % (95% half CI) | CE | Codes | Inference ensemble | Starts trained | Max updates | Inference parameters |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for setting in SETTINGS:
        outputs = SETTINGS[setting][3]
        for method in ("asrr", "g34", "dense"):
            row = next(
                r
                for r in rows
                if r["setting"] == setting
                and r["cohort"] == "pooled_fixed"
                and r["policy"] == f"{method}_accuracy"
            )
            rule = row["rule"]
            ensemble = rule.get("ensemble", 1)
            count = ensemble * (row["codes"] + 16 + 16 * outputs + outputs)
            lines.append(
                f"|{TITLES[setting]}|{method}|{100 * row['accuracy']:.2f} ± {100 * row['accuracy_half_ci']:.2f}|"
                f"{row['ce']:.4f}|{row['codes']}|{ensemble}|{rule['numeric_starts']}|{rule['steps']}|{count}|"
            )
    lines += [
        "",
        "## Separate confirmation cohorts",
        "",
        "Accuracy-selected policies are identical in both cohorts.",
        "",
        "| Setting | Cohort | aSRR accuracy, % | Dense accuracy, % | G34 accuracy, % |",
        "|---|---|---:|---:|---:|",
    ]
    for setting in SETTINGS:
        for cohort in ("test_a", "test_b"):
            values = [
                next(
                    r
                    for r in rows
                    if r["setting"] == setting
                    and r["cohort"] == cohort
                    and r["policy"] == f"{method}_accuracy"
                )
                for method in ("asrr", "dense", "g34")
            ]
            cells = [
                f"{100 * row['accuracy']:.2f} ± {100 * row['accuracy_half_ci']:.2f}"
                for row in values
            ]
            lines.append(
                f"|{TITLES[setting]}|{cohort}|" + "|".join(cells) + "|"
            )
    lines += [
        "",
        "## Paired gains of tuned aSRR",
        "",
        "| Setting | Cohort | aSRR minus | Accuracy difference, pp (95% half CI) |",
        "|---|---|---|---:|",
    ]
    for row in results["paired"]:
        if row["criterion"] == "accuracy":
            lines.append(
                f"|{TITLES[row['setting']]}|{row['cohort']}|{row['asrr_minus']}|"
                f"{100 * row['difference']:.3f} ± {100 * row['half_ci']:.3f}|"
            )
    lines += [
        "",
        "## Separately frozen CE policies",
        "",
        "| Setting | Method | Accuracy, % | CE | Codes |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        if row["cohort"] == "pooled_fixed" and row["policy"].endswith("_ce"):
            lines.append(
                f"|{TITLES[row['setting']]}|{row['policy']}|{100 * row['accuracy']:.2f}|{row['ce']:.4f}|{row['codes']}|"
            )
    lines += [
        "",
        "## Interpretation and scope",
        "",
        "The searched code budgets extend through 8192; the result is the best",
        "development-selected procedure in this declared search, not a global",
        "optimum. Every selected bank transfers only its indexator; all numeric",
        "states are fitted afresh. Ensembles increase inference parameters and",
        "training starts are reported separately. No numerical source values transfer.",
        "Practical G34 uses retained banks and source-only refinements rather",
        "than a new exhaustive native structural search. All three methods",
        "receive the same applicable numerical search per retained candidate;",
        "their numbers of structural candidates differ. The previous LUT-Q/Yeh/MSR",
        "narrow-budget tables are a separate matched comparison.",
        "No software tests were created or run.",
        "",
    ]
    lines += [
        "## Measured execution cost",
        "",
        "| Stage | GPU jobs | Summed worker hours |",
        "|---|---:|---:|",
    ]
    for phase in ("d1", "refine", "d2", "target"):
        receipts = [
            json.loads(path.read_text())
            for path in (OUTPUT / f"{phase}_jobs").glob("*.json")
        ]
        hours = sum(row["seconds"] for row in receipts) / 3600
        lines.append(f"|{phase}|{len(receipts)}|{hours:.3f}|")
    lines += [
        "",
        "Worker time includes process startup and data loading. These stages",
        "reuse previously prepared Dense source fits; their original cost and",
        "the separate faithful replication are excluded from this subtotal.",
        "",
    ]
    report = "\n".join(lines)
    (OUTPUT / "REPORT.md").write_text(report)
    Path("paper/theory/ASRR_BROAD_TUNING_2026-11-17.md").write_text(report)
    latex = []
    for name, settings in (
        ("small", ("mnist16", "fashion16", "har3")),
        ("validated", ("mnist100", "fashion100", "omniglot10")),
    ):
        latex += [
            r"\begin{table}[htbp]",
            r"\centering\small",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"Setting & Dense & G34 & aSRR & aSRR codes & Ensembles\\",
            r"\midrule",
        ]
        for setting in settings:
            values = {
                method: next(
                    r
                    for r in rows
                    if r["setting"] == setting
                    and r["cohort"] == "pooled_fixed"
                    and r["policy"] == f"{method}_accuracy"
                )
                for method in ("dense", "g34", "asrr")
            }
            cells = [
                f"${100 * values[m]['accuracy']:.2f}\\pm{100 * values[m]['accuracy_half_ci']:.2f}$"
                for m in ("dense", "g34", "asrr")
            ]
            ensembles = "/".join(
                str(values[m]["rule"].get("ensemble", 1))
                for m in ("dense", "g34", "asrr")
            )
            latex.append(
                f"{TITLES[setting]} & "
                + " & ".join(cells)
                + f" & {values['asrr']['codes']:,} & {ensembles}\\\\"
            )
        latex += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Broader development selection: query accuracy (\%) and 95\% intervals for the frozen accuracy-selected policies.",
            "Two fixed fresh cohorts contribute 1024 episodes per image setting or 180 HAR episodes; HAR intervals use nine subject means.",
            "Setting names give fitting examples per class; counts after a plus sign are additional within-task numerical-selection examples.",
            r"Ensemble sizes are Dense/G34/aSRR; training starts and parameter counts are in Appendix~\ref{app:asrr-tuning-details}.}",
            f"\\label{{tab:asrr-tuned-{name}}}",
            r"\end{table}",
            "",
        ]
    Path("paper/sections/asrr_tuning_results.tex").write_text("\n".join(latex))
    details = [
        r"\begin{table}[htbp]",
        r"\centering\small",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Setting & Method & Codes & Starts & Updates & Ensemble & Parameters\\",
        r"\midrule",
    ]
    for setting in SETTINGS:
        outputs = SETTINGS[setting][3]
        for method in ("dense", "g34", "asrr"):
            row = next(
                r
                for r in rows
                if r["setting"] == setting
                and r["cohort"] == "pooled_fixed"
                and r["policy"] == f"{method}_accuracy"
            )
            rule = row["rule"]
            ensemble = rule.get("ensemble", 1)
            parameters = ensemble * (
                row["codes"] + 16 + 16 * outputs + outputs
            )
            details.append(
                f"{TITLES[setting]} & {METHODS[method]} & {row['codes']:,} & {rule['numeric_starts']} & "
                f"{rule['steps']} & {ensemble} & {parameters:,}\\\\"
            )
        details.append(r"\addlinespace")
    details += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Selected numerical rules for the broader comparison. The Starts column counts every fitted restart, including unused ensemble members.",
        "The update budget is the training horizon; tasks with numerical validation may retain an earlier checkpoint.",
        "Parameter counts include every inference ensemble member and exclude the fixed indexator.}",
        r"\label{tab:asrr-tuned-resources}",
        r"\end{table}",
    ]
    Path("paper/sections/asrr_tuning_resources.tex").write_text(
        "\n".join(details) + "\n"
    )
    secondary = [
        r"\begin{table}[htbp]",
        r"\centering\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Setting & Dense & G34 & aSRR & Codes & CE & Ensembles\\",
        r"\midrule",
    ]
    for setting in SETTINGS:
        values = {
            method: next(
                row
                for row in rows
                if row["setting"] == setting
                and row["cohort"] == "pooled_fixed"
                and row["policy"] == f"{method}_ce"
            )
            for method in ("dense", "g34", "asrr")
        }
        cells = [
            f"${100 * values[m]['accuracy']:.2f}\\pm{100 * values[m]['accuracy_half_ci']:.2f}$"
            for m in ("dense", "g34", "asrr")
        ]
        ensembles = "/".join(
            str(values[m]["rule"].get("ensemble", 1))
            for m in ("dense", "g34", "asrr")
        )
        secondary.append(
            f"{TITLES[setting]} & "
            + " & ".join(cells)
            + f" & {values['asrr']['codes']:,} & {values['asrr']['ce']:.3f} & {ensembles}\\\\"
        )
    secondary += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Secondary policies selected by development cross-entropy, frozen before the same two query cohorts.",
        r"Accuracy (\%) and 95\% intervals use the same units as the primary tables.",
        "Codes and CE refer to aSRR; ensemble sizes are Dense/G34/aSRR.",
        "These policies do not replace the accuracy-selected primary comparison.}",
        r"\label{tab:asrr-tuned-ce}",
        r"\end{table}",
    ]
    Path("paper/sections/asrr_tuning_ce_results.tex").write_text(
        "\n".join(secondary) + "\n"
    )
    settings = list(SETTINGS)
    fig, axis = plt.subplots(figsize=(7, 3.4))
    for offset, other, color in (
        (-0.12, "dense", "tab:blue"),
        (0.12, "g34", "tab:orange"),
    ):
        selected = [
            next(
                r
                for r in results["paired"]
                if r["setting"] == setting
                and r["cohort"] == "pooled_fixed"
                and r["criterion"] == "accuracy"
                and r["asrr_minus"] == other
            )
            for setting in settings
        ]
        axis.errorbar(
            np.arange(len(settings)) + offset,
            [100 * r["difference"] for r in selected],
            yerr=[100 * r["half_ci"] for r in selected],
            fmt="o",
            capsize=3,
            color=color,
            label=f"aSRR minus {other}",
        )
    axis.axhline(0, color="black", lw=0.8)
    axis.set_xticks(
        np.arange(len(settings)),
        [TITLES[s] for s in settings],
        rotation=20,
        ha="right",
    )
    axis.set_ylabel("Accuracy difference (percentage points)")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(OUTPUT / f"paired_tuned.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    run()
