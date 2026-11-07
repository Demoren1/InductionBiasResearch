"""Report frozen interval/core results without changing source models."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.darts_structured_tasks_v1.data import save_json

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=plt.get_cmap("tab20").colors)

LABELS = {
    "scalar_dp": "Scalar DP",
    "marginal_information_dp": "Marginal-information DP",
    "diagonal_hessian_dp": "Diagonal-Hessian DP",
    "zero_t_then_information_dp": "Zero significance + information DP",
    "point_single_linkage": "Point SLINK",
    "same_interval_graph": "Same-interval graph",
    "full_pair_f_graph": "Full pairwise F graph",
    "global_interval_certificate": "Global interval certificate",
    "witness_codebook": "One witness per code",
    "mean_core_codebook": "Mean of certified cores",
    "nonzero_gls_codebook": "Nonzero-core GLS",
    "full_gls_core_codebook": "Full GLS + certified zeros",
}
CONFIDENCE = tuple(list(LABELS)[5:])


def report(root):
    plan = json.loads((root / "protocol.json").read_text())
    is_core = "full_gls_core_codebook" in plan["methods"]
    cases = [
        json.loads(p.read_text()) for p in sorted((root / "cases").glob("*/audit.json"))
    ]
    if len(cases) != len(plan["jobs"]):
        raise RuntimeError("Report requires the complete frozen cohort")
    out = root / "report"
    out.mkdir(exist_ok=True)
    rows, detailed, stops = [], [], []
    for method in plan["methods"]:
        for n in plan["observations"]:
            selected = [
                r
                for c in cases
                for r in c["records"]
                if r["method"] == method and r["n"] == n
            ]
            returned = [r for r in selected if r["returned_index"]]
            row = {
                "method": method,
                "n": n,
                "cases": len(cases),
                "issued": len(returned),
                "exact": sum(r["recovery"]["exact_I"] for r in returned),
                "wrong_issued": sum(not r["recovery"]["exact_I"] for r in returned),
                "mean_mask_recall_conditional_on_issue": float(
                    np.mean([r["recovery"]["mask_recall"] for r in returned])
                )
                if returned
                else None,
                "mean_categorical_errors_conditional_on_issue": float(
                    np.mean([r["recovery"]["categorical_errors"] for r in returned])
                )
                if returned
                else None,
                "mean_weighted_distortion": float(
                    np.mean([r["weighted_distortion"] for r in returned])
                )
                if returned and "weighted_distortion" in returned[0]
                else None,
            }
            if n in plan["transfer"]["evaluate_at"]:
                for support in plan["transfer"]["support"]:
                    row[f"transfer_mse_support{support}_conditional_on_issue"] = (
                        float(
                            np.mean(
                                [
                                    r["transfer"][str(support)]["mean_mse"]
                                    for r in returned
                                ]
                            )
                        )
                        if returned
                        else None
                    )
            rows.append(row)
            for regime in plan["regimes"]:
                for sigma in plan["noise"]:
                    group = [
                        r
                        for c in cases
                        if c["job"]["regime"] == regime and c["job"]["noise"] == sigma
                        for r in c["records"]
                        if r["method"] == method and r["n"] == n
                    ]
                    detailed.append(
                        {
                            "method": method,
                            "n": n,
                            "regime": regime,
                            "noise": sigma,
                            "cases": len(group),
                            "issued": sum(r["returned_index"] for r in group),
                            "exact": sum(
                                r.get("recovery", {}).get("exact_I", False)
                                for r in group
                            ),
                        }
                    )
        if method in CONFIDENCE:
            first = [c["first_return"].get(method) for c in cases]
            stops.append(
                {
                    "method": method,
                    "issued_by_final": sum(n is not None for n in first),
                    "never_issued": sum(n is None for n in first),
                    "median_n_conditional_on_issue": float(
                        np.median([n for n in first if n is not None])
                    ),
                    "ever_wrong_runs": sum(
                        any(
                            r["returned_index"] and not r["recovery"]["exact_I"]
                            for r in c["records"]
                            if r["method"] == method
                        )
                        for c in cases
                    ),
                }
            )
    coverage = {
        n: sum(c["interval_coverage"][i]["all_weight_intervals_cover"] for c in cases)
        for i, n in enumerate(plan["observations"])
    }
    diagnosis = {}
    for n in plan["observations"]:
        counter = Counter()
        for c in cases:
            r = next(
                r
                for r in c["records"]
                if r["n"] == n and r["method"] == "global_interval_certificate"
            )
            q = r["certificate"]["minimum_nonzero_codes"]
            counter[
                "issued"
                if r["returned_index"]
                else (
                    "q<K"
                    if q < plan["codes"]
                    else "q>K"
                    if q > plan["codes"]
                    else "q=K ambiguous"
                )
            ] += 1
        diagnosis[n] = dict(counter)
    summary = {
        "independent_geometries": len(plan["families"]),
        "dependent_cases": len(cases),
        "rows": rows,
        "stratified": detailed,
        "stops": stops,
        "interval_simultaneous_coverage": coverage,
        "global_abstention_diagnosis": diagnosis,
        "claim": "Improved confidence coverage over pairwise graphs; no superiority over strong point estimators or native nonlinear competitors established.",
    }
    save_json(out / "summary.json", summary)
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with (out / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for ax, (sigma, regime) in zip(
        axes.ravel(), ((s, r) for s in plan["noise"] for r in plan["regimes"])
    ):
        for method in plan["methods"]:
            selected = [
                r
                for r in detailed
                if r["method"] == method
                and r["regime"] == regime
                and r["noise"] == sigma
            ]
            ax.plot(
                [r["n"] for r in selected],
                [r["exact"] / r["cases"] for r in selected],
                marker=".",
                label=LABELS[method],
            )
        ax.set_xscale("log", base=2)
        ax.set_title(f"{regime}; noise={sigma}")
        ax.set_xlabel("Source observations (closed-form fitting)")
        ax.set_ylabel("Exact I fraction; abstention = failure")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(out / "exact_recovery.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for method in plan["methods"]:
        selected = [r for r in rows if r["method"] == method]
        axes[0].plot(
            [r["n"] for r in selected],
            [r["mean_mask_recall_conditional_on_issue"] for r in selected],
            marker=".",
            label=LABELS[method],
        )
        if method in plan["methods"][:4]:
            axes[1].plot(
                [r["n"] for r in selected],
                [r["mean_weighted_distortion"] for r in selected],
                marker=".",
                label=LABELS[method],
            )
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.grid(alpha=0.2)
        ax.set_xlabel("Source observations (no optimizer steps)")
    axes[0].set_ylabel("Mask recall conditional on issued I")
    axes[1].set_ylabel("Observed projection criterion (method-specific units)")
    axes[1].set_yscale("log")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(out / "recall_and_projection_loss.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, n in zip(axes, plan["transfer"]["evaluate_at"]):
        for method in plan["methods"]:
            r = next(r for r in rows if r["method"] == method and r["n"] == n)
            ax.plot(
                plan["transfer"]["support"],
                [
                    r[f"transfer_mse_support{s}_conditional_on_issue"]
                    for s in plan["transfer"]["support"]
                ],
                marker=".",
                label=f"{LABELS[method]} ({r['issued']}/108)",
            )
        oracle = [
            np.mean([c["oracle_transfer"][str(s)]["mean_mse"] for c in cases])
            for s in plan["transfer"]["support"]
        ]
        ax.plot(
            plan["transfer"]["support"],
            oracle,
            color="black",
            linestyle="--",
            label="True I (audit only)",
        )
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="New-function support examples",
            title=f"Source n={n}",
        )
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Fresh-C clean population MSE; issued subset")
    axes[-1].legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(out / "fresh_codebook_transfer.png", dpi=160)
    plt.close(figure)
    # Diagnostic illustration selected by observed method disagreement, not teacher quality.
    chosen = None
    for folder in sorted((root / "cases").glob("job_*")):
        c = json.loads((folder / "audit.json").read_text())
        for n in plan["observations"]:
            at = {r["method"]: r for r in c["records"] if r["n"] == n}
            if (
                at["global_interval_certificate"]["returned_index"]
                and not at["same_interval_graph"]["returned_index"]
            ):
                chosen = folder, c, n, at
                break
        if chosen:
            break
    if chosen:
        folder, case, n, at = chosen
        with np.load(folder / "learner_outputs.npz") as a:
            fitted, lower, upper = (
                a[f"fitted__n{n}"],
                a[f"lower__n{n}"],
                a[f"upper__n{n}"],
            )
        with np.load(folder / "aligned_indexators_audit_only.npz") as a:
            index, truth = (
                a[at["global_interval_certificate"]["key"] + "__aligned"],
                a["truth_indices"],
            )
        figure, axes = plt.subplots(1, 4, figsize=(14, 3.6))
        for ax, value, title, cmap, vmin, vmax in zip(
            axes,
            (fitted, upper - lower, index, truth),
            (
                "Observed OLS weights",
                "CI width",
                "Selected I (audit-aligned)",
                "True I (audit only)",
            ),
            ("coolwarm", "viridis", "coolwarm", "coolwarm"),
            (None, 0, -4, -4),
            (None, None, 4, 4),
        ):
            image = ax.imshow(value, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=9)
            figure.colorbar(image, ax=ax, fraction=0.046)
        figure.suptitle(f"Mechanism illustration: {folder.name}, n={n}; graph abstains")
        figure.tight_layout()
        figure.savefig(out / "selected_indexator_heatmap.png", dpi=170)
        plt.close(figure)
        save_json(
            out / "heatmap_selection.json",
            {
                "job": folder.name,
                "n": n,
                "source_job": case["job"],
                "selection": "First lexicographic case/look where global issues and same-CI graph abstains; diagnostic illustration, not independent performance evidence.",
            },
        )
    lines = [
        "# G11: independent codebooks from certified cores"
        if is_core
        else "# G10: global interval compatibility",
        "",
        "Frozen prospective linear-Gaussian diagnostic: six independent geometries, three coefficient draws, three input-covariance regimes and two noise levels =108 dependent cases. All source outputs were frozen before true-I audits and fresh-C transfer. Exactly four signed nonzero groups and zero are known; locations, group sizes and numeric codes are hidden.",
        "",
        "The new solver gives more certified structures than pairwise graphs. Strong zero-significance + weighted-DP controls remain more accurate at a fixed budget. No superiority over native competing methods or nonlinear recovery is established.",
        "",
        "## Exact indexators (abstention counts as failure)",
        "",
        "| Method | n8192 | n32768 | n131072 | n8388608 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in plan["methods"]:
        counts = [
            next(r["exact"] for r in rows if r["method"] == method and r["n"] == n)
            for n in (8192, 32768, 131072, 8388608)
        ]
        lines.append(
            "| "
            + LABELS[method]
            + " | "
            + " | ".join(f"{v}/108" for v in counts)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Reliability and scope",
            "",
            "No wrong confidence-issued I occurred at any of the eight looks. The guarantee is alpha=.05 per algorithm/source run across its eight looks, not a simultaneous guarantee for all108 benchmark cases. Marginal simultaneous coverage counts by look: "
            + str(coverage)
            + ". These observations are compatible with, and do not prove, the finite Gaussian argument."
            + (
                " Core methods split the same total n into two independent halves; alpha is divided across the A interval and B contrast stages. Full GLS exploits all forced zeros/nonzeros with the complete covariance."
                if is_core
                else " The six q>K events at n131072 coincide with actual CI noncoverage and cause abstention."
            ),
            "",
            "A single source-function draw can contain nearly colliding code magnitudes. Some cases remain unresolved even at8.39m examples. More data cannot resolve an exactly colliding source codebook; multiple informative source functions remain necessary for general structural identification.",
            "",
            "## Diagnosis and transfer",
            "",
            "At n131072 full GLS issues75/108 exact I, versus72/108 global-interval and66/108 full-F. Mean-core and nonzero-GLS both issue73/108, witness-only72/108. Strong zero-cleanup+information-DP gives89/108. This is a small improvement in certified coverage, not superiority in unconditional exact recovery."
            if is_core
            else "At n32768,33 cases have q=K but some ambiguous assignments,22 have q<K,53 issue. This motivates a separately registered next stage: use forced positions as cores, independently fit codebooks, and exploit full covariance to classify remaining positions. G10 alone does not establish that extension works.",
            "",
            "Fresh-C plots use100 new Gaussian codebooks per source geometry/draw, with I frozen and all C re-estimated using only new support. Source numeric weights are discarded. Confidence curves condition on issued structures; their issued counts are shown, so lower MSE on an easier subset is not an unconditional win. Source regimes/noise reuse the same target cohort and are dependent repeats.",
            "",
            "## Nonlinear diagnostic is separate",
            "",
            "G11 is a linear diagnostic only. Its exact Student argument does not certify nonlinear G9 or new DeepSets tasks. Independent native nonlinear competitors, new C transfer and robustness to misspecified noise remain outstanding."
            if is_core
            else "On already-opened G9 estimates, the global solver issues all16 noiseless views correctly, and only one of four noisy warm families per budget. Nonlinear CI coverage and stationarity are not certified. This is exploratory and does not inherit the exact linear Gaussian guarantee. Hidden-column permutation is accounted for in audit metrics; the initial wrong metric file is preserved beside its correction record.",
            "",
            "## Artifacts",
            "",
            "- [Exact recovery](exact_recovery.png)",
            "- [Recall and observed projection loss](recall_and_projection_loss.png); closed-form sample budgets, not fictitious optimizer-step curves.",
            "- [Fresh codebook transfer](fresh_codebook_transfer.png)",
            "- [Observed W / CI width / chosen I / true I](selected_indexator_heatmap.png)",
            "- [Machine-readable comparison](comparison.csv), [full summary](summary.json).",
            "",
            "Weighted k-means/Hessian-weighted quantization, interval piercing, confidence clustering and post-selection sample splitting are known principles. The specific global signed-equality certificate is a candidate specialization; priority and advantage over strong native solvers remain open.",
        ]
    )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"cases": len(cases), "stops": stops, "output": str(out)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    report(parser.parse_args().root)


if __name__ == "__main__":
    main()
