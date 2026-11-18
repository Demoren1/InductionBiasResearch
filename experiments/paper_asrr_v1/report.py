"""Combine immutable baseline scores and the matched aSRR addition."""

import json
from pathlib import Path

from experiments.paper_recovery_extension_v1.report import (
    DATASETS,
    LABEL,
    PAPER_METHODS,
    budget_figure,
    constrained,
    synthetic_table,
    table,
)
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

from .prepare import OUTPUT, PREVIOUS


def run():
    methods = (*PAPER_METHODS[:2], "asrr", *PAPER_METHODS[2:])
    summaries, additions = [], []
    for dataset, title in DATASETS:
        summary = json.loads((PREVIOUS / dataset / "summary.json").read_text())
        added = json.loads((OUTPUT / dataset / "summary.json").read_text())
        summary["all_budgets"].extend(added["all_budgets"])
        summary["decisions"]["asrr"] = added["decision"]
        summaries.append(summary)
        additions.append(added)
        table(dataset, title, summary, methods=methods)
    budget_figure(summaries, methods=methods)
    synthetic = json.loads((PREVIOUS / "synthetic_results.json").read_text())
    synthetic_added = json.loads(
        (OUTPUT / "synthetic_results.json").read_text()
    )
    synthetic["rows"].extend(synthetic_added["rows"])
    synthetic_table(synthetic, methods=methods)
    lines = [
        "# aSRR under the manuscript's evaluation protocol",
        "",
        "17 November 2026.",
        "",
        "This adds aSRR to the previously evaluated benchmark, with the same",
        "source/development observations, budget grid, numerical solver and target",
        "episode draws. It is an added-method follow-up, not a new independent",
        "test environment. All aSRR choices were frozen on development before",
        "its query responses were scored. Baseline artifacts are unchanged.",
        "",
        "See `experiments/paper_asrr_v1/PROTOCOL.md` and the compact records in",
        "`artifacts/paper_asrr_2026-11-17` for candidate families and provenance.",
        "",
        "## Applied tasks",
        "",
        "Accuracy intervals: 96 conditional image episodes; nine HAR subject means.",
        "Constrained rows use 8/16/32/64/128/512 codes, selected on development.",
        "",
        "| Dataset | Method | Codes | Parameters | Updates | Accuracy (%) | CE | Full endpoint preferred |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for (dataset, title), summary in zip(DATASETS, summaries):
        for method in ("dense", "g34", "asrr"):
            row = constrained(summary, method)
            full = (
                summary["decisions"][method]["best_including_unshared"]
                == "unshared"
            )
            lines.append(
                f"|{title}|{LABEL[method]}|{row['codes']}|{row['task_parameters']}|"
                f"{row['numeric_updates']}|{100 * row['accuracy']:.2f} ± "
                f"{100 * row['accuracy_half_ci']:.2f}|{row['cross_entropy']:.3f}|"
                f"{'Yes' if full else 'No'}|"
            )
    lines += [
        "",
        "Paired accuracy differences use each method's development-selected",
        "constrained budget; intervals are in percentage points.",
        "",
        "| Dataset | aSRR minus | Difference (pp) | 95% half interval |",
        "|---|---|---:|---:|",
    ]
    for (_, title), added in zip(DATASETS, additions):
        for method, row in added["paired_constrained_minus"].items():
            lines.append(
                f"|{title}|{LABEL[method]}|{100 * row['accuracy_difference']:.2f}|"
                f"{100 * row['half_ci']:.2f}|"
            )
    lines += [
        "",
        "## Two fixed synthetic geometries",
        "",
        "| Geometry | aSRR query MSE | Signed exact recovery | Support errors |",
        "|---|---:|---|---:|",
    ]
    for row in synthetic_added["rows"]:
        lines.append(
            f"|{row['geometry']}|{row['mse']:.6g}|"
            f"{row['structure']['signed_exact']}|{row['structure']['mask_errors']}|"
        )
    lines += [
        "",
        "## Scope",
        "",
        "The image results apply to the 16-example-per-class setting and",
        "the compressed grid 8--512. They do not reproduce or invalidate",
        "the earlier 100-example-per-class MNIST result with 1024 codes",
        "in APPROXIMATE_SRR_RESULTS_2026-11-15.md. That study also selected",
        "families on development and used within-task numerical validation",
        "and multiple fresh restarts. Here source M first selects one",
        "candidate per budget, before development chooses the numeric rule",
        "and budget. Insufficient code budget is a hypothesis requiring",
        "a separate extension; these measurements do not isolate its effect.",
        "",
        "The Ward generator is approximate and is not covered by an exact",
        "single-task recovery guarantee. Real-data indexators are evaluated",
        "by predictive transfer, without claiming a known ground-truth grouping.",
        "The synthetic teachers have sparse supports; the full-support aSRR",
        "generator does not propose their zero connections and does not recover",
        "either signed indexator. Its MSE is higher than Dense and G34 in both cases.",
        "No software tests were created or run.",
        "",
    ]
    Path("paper/theory/ASRR_MATCHED_RESULTS_2026-11-17.md").write_text(
        "\n".join(lines)
    )
    save_json(
        OUTPUT / "manuscript_render.json",
        {
            "source_summaries_sha256": {
                str(path): digest(path)
                for path in [
                    *(OUTPUT / ds / "summary.json" for ds, _ in DATASETS),
                    OUTPUT / "synthetic_results.json",
                ]
            },
            "paper_methods": methods,
        },
    )


if __name__ == "__main__":
    run()
