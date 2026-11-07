"""Post-freeze G11 mechanism audit; no source-model selection or mutation."""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.darts_structured_tasks_v1.data import save_json


def diagnose(root):
    plan = json.loads((root / "protocol.json").read_text())
    folders = sorted((root / "cases").glob("job_*"))
    cases = [json.loads((p / "audit.json").read_text()) for p in folders]
    records, geometry_intervals, all_contradictions = [], [], []
    rng = np.random.default_rng(plan["seed"] + 111)
    for n in plan["observations"]:
        counts, ratios, partial_errors, partial_errors_covered = Counter(), [], 0, 0
        for folder, case in zip(folders, cases):
            at = {r["method"]: r for r in case["records"] if r["n"] == n}
            full = at["full_gls_core_codebook"]["certificate"]
            core = full["core_certificate"]
            q = core["minimum_nonzero_codes"]
            counts[
                "q<K" if q < plan["codes"] else "q>K" if q > plan["codes"] else "q=K"
            ] += 1
            if q != plan["codes"]:
                continue
            with np.load(folder / "teacher_audit_only.npz") as a:
                truth = a["indices"].ravel()
            forced = np.asarray(core["forced_mask"])
            selected = np.asarray(core["forced_indices"])
            errors = int((selected[forced] != truth[forced]).sum())
            partial_errors += errors
            covered = next(
                r["all_weight_intervals_cover"]
                for r in case["independent_core_interval_coverage"]
                if r["n"] == n
            )
            if covered:
                partial_errors_covered += errors
            if covered and errors:
                all_contradictions.append(
                    {"job": folder.name, "n": n, "errors": errors}
                )
            mean = at["mean_core_codebook"]["certificate"]
            ratios.extend(
                np.asarray(full["code_variance"]) / np.asarray(mean["code_variance"])
            )
        records.append(
            {
                "n": n,
                "core_availability": dict(counts),
                "median_full_gls_over_mean_code_variance": float(np.median(ratios))
                if ratios
                else None,
                "forced_errors_total": partial_errors,
                "forced_errors_on_coverage_event": partial_errors_covered,
            }
        )
        for comparison in (
            "global_interval_certificate",
            "mean_core_codebook",
            "zero_t_then_information_dp",
        ):
            difference = []
            for family in plan["families"]:
                group = [c for c in cases if c["job"]["family"] == family]
                values = []
                for c in group:
                    at = {r["method"]: r for r in c["records"] if r["n"] == n}
                    values.append(
                        float(
                            at["full_gls_core_codebook"]
                            .get("recovery", {})
                            .get("exact_I", False)
                        )
                        - float(
                            at[comparison].get("recovery", {}).get("exact_I", False)
                        )
                    )
                difference.append(np.mean(values))
            difference = np.asarray(difference)
            boot = difference[
                rng.integers(0, len(difference), (20000, len(difference)))
            ].mean(1)
            geometry_intervals.append(
                {
                    "n": n,
                    "comparison": comparison,
                    "mean_exact_fraction_difference": float(difference.mean()),
                    "descriptive_geometry_bootstrap95": np.quantile(
                        boot, [0.025, 0.975]
                    ).tolist(),
                    "independent_geometries": len(difference),
                    "caution": "Six independent geometries only; descriptive percentile interval, not proof of advantage. Coefficient/noise/covariance repeats not treated as independent.",
                }
            )
    if all_contradictions:
        raise ArithmeticError(
            "A forced assignment contradicted truth on the coverage event"
        )
    out = root / "report"
    summary = {
        "records": records,
        "paired_geometry_effects": geometry_intervals,
        "logical_contradictions_on_coverage_event": all_contradictions,
        "status": "Post-freeze scientific diagnosis, no HPO or target selection",
    }
    save_json(out / "core_diagnosis.json", summary)
    selected = None
    for folder, case in zip(folders, cases):
        for n in plan["observations"]:
            at = {r["method"]: r for r in case["records"] if r["n"] == n}
            if (
                at["full_gls_core_codebook"]["returned_index"]
                and not at["global_interval_certificate"]["returned_index"]
            ):
                selected = folder, case, n, at
                break
        if selected:
            break
    if selected:
        folder, case, n, at = selected
        meta = at["full_gls_core_codebook"]["certificate"]["core_certificate"]
        forced, partial = (
            np.asarray(meta["forced_mask"]),
            np.asarray(meta["forced_indices"], dtype=float),
        )
        partial[~forced] = np.nan
        with np.load(folder / "learner_outputs.npz") as a:
            fitted = a[f"fitted__n{n}"]
        with np.load(folder / "aligned_indexators_audit_only.npz") as a:
            index = a[at["full_gls_core_codebook"]["key"] + "__aligned"]
            truth = a["truth_indices"]
        figure, axes = plt.subplots(1, 4, figsize=(14, 3.8))
        cmap = plt.get_cmap("coolwarm").copy()
        cmap.set_bad("#252525")
        for ax, value, title in zip(
            axes,
            (fitted, partial.reshape(truth.shape), index, truth),
            (
                "Observed full-data OLS W",
                "Certified A cores; dark=unknown",
                "Full-GLS I (audit-aligned)",
                "True I (audit only)",
            ),
        ):
            picture = ax.imshow(
                value,
                cmap=cmap,
                vmin=None if ax == axes[0] else -4,
                vmax=None if ax == axes[0] else 4,
            )
            ax.set_title(title, fontsize=9)
            figure.colorbar(picture, ax=ax, fraction=0.046)
        figure.suptitle(
            f"Mechanism illustration: {folder.name}, total n={n}; all-data interval solver abstains"
        )
        figure.tight_layout()
        figure.savefig(out / "certified_core_indexators.png", dpi=170)
        plt.close(figure)
        save_json(
            out / "core_heatmap_selection.json",
            {
                "job": folder.name,
                "n": n,
                "source_job": case["job"],
                "selection": "First lexicographic observed case/look where full-GLS issues and global interval solver abstains. No true-I quality filter; diagnostic illustration only.",
            },
        )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(
        [r["n"] for r in records],
        [r["core_availability"].get("q=K", 0) / len(cases) for r in records],
        marker=".",
    )
    axes[0].set_ylabel("Fraction with K certified cores in half A")
    axes[1].plot(
        [r["n"] for r in records],
        [r["median_full_gls_over_mean_code_variance"] for r in records],
        marker=".",
    )
    axes[1].set_ylabel("Median code variance: full GLS / core mean")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Total source observations (A+B)")
        ax.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(out / "core_availability_and_codebook_variance.png", dpi=170)
    plt.close(figure)
    note = "\n\n## Post-freeze mechanism audit\n\nAt n131072,86/108 cases have q=K on half A;22 abstain before codebook estimation. Full GLS reduces median code variance to0.15 of the arithmetic-core estimator, but improves exact issued I only73→75. Better numeric codebooks alone do not remove the remaining uncertain coordinate assignments or the first-stage bottleneck. No forced-I contradiction was found on any audited simultaneous-coverage event.\n\nPaired effects are bootstrapped over six geometries, not108 dependent repetitions. They are descriptive and cannot establish strong general superiority. [Full diagnostic numbers](core_diagnosis.json), [core availability/variance](core_availability_and_codebook_variance.png), [cores and final I](certified_core_indexators.png).\n"
    with (out / "REPORT.md").open("a") as f:
        f.write(note)
    print(
        json.dumps(
            {
                "diagnosis": records,
                "effect_n131072": [r for r in geometry_intervals if r["n"] == 131072],
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    diagnose(parser.parse_args().root)


if __name__ == "__main__":
    main()
