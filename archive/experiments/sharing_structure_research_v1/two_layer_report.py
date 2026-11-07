"""Joint structure summaries and measured scientific figures for G31/G32."""

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from experiments.darts_indexator_v1.data import digest, load_archive, save_json

from .indexed_readout import canonical_head, code_gauges
from .pattern_prefix import timestamp
from .readout_symmetry import automorphisms
from .synchronized_head_study import joint_head_metric

BASE = Path(__file__).resolve().parent
G31 = BASE / "results/20261106/guarded_two_layer_g31"
G32 = BASE / "results/20261106/synchronized_two_layer_g32"
LABELS = {
    "old_free": "Old prefixes / free head",
    "keep2_free": "2H pool / free head",
    "synchronized": "2H + synchronized J2",
    "gated": "2H + synchronized J2 + gate",
}
COLORS = {
    "old_free": "#9b5678",
    "keep2_free": "#db8a3e",
    "synchronized": "#6e8c47",
    "gated": "#277ca4",
}


def load_json(path):
    return json.loads(path.read_text())


def corrected_baseline(folder, plan):
    rows = []
    for case in plan["cases"]:
        truth = load_archive(folder / "private" / f"{case['name']}.npz")
        for condition, sigma in enumerate(plan["noise"]):
            path = folder / "source" / f"{case['name']}_{condition}"
            fitted = load_archive(path / "fitted.npz")
            metadata = load_json(path / "fitted.json")
            for row in metadata["selections"]:
                result = {
                    "case": case["name"],
                    "kind": case["kind"],
                    "style": case["style"],
                    "noise": sigma,
                    "method": row["method"],
                    "abstained": row["abstained"],
                }
                if not row["abstained"]:
                    metric, _, head, _ = joint_head_metric(
                        fitted[f"i{row['candidate']}_index"],
                        fitted[f"i{row['candidate']}_{row['variant']}_head_index"],
                        truth["index"],
                        truth["head_index"],
                        case["codes"],
                    )
                    result.update(
                        metric,
                        head=head,
                        exact=metric["categorical_errors"] == 0,
                        meta_mse=row["meta_mse"],
                    )
                rows.append(result)
    save_json(
        folder / "joint_symmetry_audit.json",
        {
            "evaluated_utc": timestamp(),
            "definition": "I1 first; when exact, head evaluated over all compatible global automorphisms, not one arbitrary hidden-column matching.",
            "rows": rows,
        },
    )
    return rows


def summarize(root):
    folder = root / "confirmation"
    plan = load_json(folder / "protocol.json")
    baseline = corrected_baseline(folder, plan)
    sync = load_json(folder / "synchronized_head_ablation/evaluation.json")["rows"]
    gate = load_json(folder / "head_gate/evaluation.json")["rows"]
    all_rows = []
    for source, wanted, label in (
        (baseline, "old_free", "old_free"),
        (baseline, "keep2_free", "keep2_free"),
        (baseline, "keep2_indexed", "unsynchronized"),
        (baseline, "keep2_forced4", "forced4"),
        (sync, "keep2_indexed", "synchronized"),
        (gate, "keep2_gated", "gated"),
    ):
        all_rows.extend(
            [{**row, "comparison": label} for row in source if row["method"] == wanted]
        )
    aggregate = []
    for kind in ("synthetic", "pattern", "sets"):
        styles = sorted({row["style"] for row in all_rows if row["kind"] == kind})
        for sigma in plan["noise"]:
            for style in styles:
                for method in (
                    "old_free",
                    "keep2_free",
                    "unsynchronized",
                    "forced4",
                    "synchronized",
                    "gated",
                ):
                    subset = [
                        row
                        for row in all_rows
                        if row["kind"] == kind
                        and row["noise"] == sigma
                        and row["style"] == style
                        and row["comparison"] == method
                    ]
                    exact = [row for row in subset if row.get("exact", False)]
                    aggregate.append(
                        {
                            "kind": kind,
                            "noise": sigma,
                            "style": style,
                            "method": method,
                            "cases": len(subset),
                            "exact_I1": len(exact),
                            "exact_both": sum(row["head"]["exact"] for row in exact),
                            "mask_exact": sum(
                                row.get("mask_f1", 0) == 1 for row in subset
                            ),
                            "total_I1_errors": sum(
                                row.get("categorical_errors", 256) for row in subset
                            ),
                            "abstained": sum(row["abstained"] for row in subset),
                        }
                    )
    summary = {
        "created_utc": timestamp(),
        "stage": plan["stage"],
        "aggregate": aggregate,
        "rows": all_rows,
        "unit": "Six synthetic geometries reused across head styles; paired noises. Pattern is one fixed0110 geometry with new numbers, DeepSets one hidden geometry per cohort. Exact-both requires I1 exact plus signed J2 relation/mask exact under the same compatible hidden alignment.",
    }
    save_json(root / "summary.json", summary)
    with (root / "structure_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    return summary


def comparison_figure(summary):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    methods = ("old_free", "keep2_free", "synchronized", "gated")
    styles = ("free", "tied2", "tied4")
    for i, sigma in enumerate((0.2, 2.0)):
        for j, measure in enumerate(("exact_I1", "exact_both")):
            ax = axes[i, j]
            for k, method in enumerate(methods):
                values = [
                    next(
                        row[measure]
                        for row in summary["aggregate"]
                        if row["kind"] == "synthetic"
                        and row["noise"] == sigma
                        and row["style"] == style
                        and row["method"] == method
                    )
                    for style in styles
                ]
                x = np.arange(3) + (k - 1.5) * 0.2
                bars = ax.bar(
                    x, values, width=0.19, color=COLORS[method], label=LABELS[method]
                )
                for bar, value in zip(bars, values, strict=True):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        value + 0.06,
                        str(value),
                        ha="center",
                        fontsize=9,
                    )
            ax.set_xticks(
                range(3), ["Free W2", "Two shared codes", "Four shared codes"]
            )
            ax.set(
                ylim=(0, 6.9),
                yticks=range(7),
                ylabel="Exact structures / 6 geometries",
                title=("I1" if j == 0 else "Both I1 and J2")
                + f", raw noise SD={sigma}",
            )
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("G32 independent confirmation: actual structure recovery")
    fig.savefig(G32 / "figures/structure_comparison.png", dpi=180)
    plt.close(fig)


def heatmaps():
    folder = G32 / "confirmation"
    selected = load_archive(folder / "head_gate/aligned_indices.npz")
    cmap = ListedColormap(
        [
            "#184b76",
            "#2683ae",
            "#6bbbd2",
            "#b2dde7",
            "#f5f5f5",
            "#edb09b",
            "#dd775d",
            "#b33a46",
            "#741c37",
        ]
    )
    norm = BoundaryNorm(np.arange(-4.5, 5), cmap.N)
    cases = (
        ("pattern_tied2", "Pattern0110"),
        ("sets_tied4", "DeepSets"),
        ("new_0_tied2", "Local"),
        ("new_4_tied4", "Random"),
    )
    fig, axes = plt.subplots(4, 4, figsize=(14, 15), layout="constrained")
    for i, (case, label) in enumerate(cases):
        truth = load_archive(folder / "private" / f"{case}.npz")
        key = f"{case}_0_keep2_gated"
        index, head = selected[key + "_I"], canonical_head(selected[key + "_J"])
        true_head = canonical_head(truth["head_index"])
        head_codes = max(int(np.max(np.abs(head))), int(np.max(np.abs(true_head))), 1)
        head_cmap = plt.get_cmap("coolwarm", 2 * head_codes + 1)
        head_norm = BoundaryNorm(
            np.arange(-head_codes - 0.5, head_codes + 1.5), head_cmap.N
        )
        titles = (
            f"{label}: true I1",
            f"Selected I1: {np.sum(index != truth['index'])} errors",
            "True J2",
            "Selected J2",
        )
        for j, matrix in enumerate((truth["index"], index, true_head, head)):
            axes[i, j].imshow(
                matrix,
                cmap=cmap if j < 2 else head_cmap,
                norm=norm if j < 2 else head_norm,
                interpolation="nearest",
                aspect="auto" if j > 1 else "equal",
            )
            axes[i, j].set(
                title=titles[j], xlabel="Hidden neuron" if j < 2 else "Output"
            )
            if j in (0, 2):
                axes[i, j].set_ylabel("Input coordinate" if j == 0 else "Hidden neuron")
            if j > 1:
                axes[i, j].set_xticks(range(matrix.shape[1]))
            for (d, h), value in np.ndenumerate(matrix):
                if value and (j > 1 or matrix.size < 100):
                    axes[i, j].text(
                        h,
                        d,
                        str(value),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if abs(value) >= 3 else "#222222",
                    )
    fig.suptitle(
        "Actual selected integer matrices, moderate noise; J2 canonicalized by first occurrence\nSame hidden alignment for both layers; zeros in I1 are white, head indices are printed"
    )
    fig.savefig(G32 / "figures/selected_indexators.png", dpi=180)
    plt.close(fig)


def prefix_figures():
    path = G31 / "pilot/synthetic_9922_0.json"
    rows = load_json(path)["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    for mode, label, color in (
        ("old", "Old selection", "#9b5678"),
        ("keep2", "2H candidate pool", "#277ca4"),
    ):
        row = next(row for row in rows if row["mode"] == mode and row["task"] == 2)
        audit = row["prefix_audit"]
        axes[0].plot(
            [r["step"] for r in audit],
            [r["recall"] for r in audit],
            "o-",
            color=color,
            label=label,
        )
        trace = [r for r in row["trace"] if "training_mse" in r]
        for ax, field in zip(axes[1:], ("training_mse", "validation_mse"), strict=True):
            ax.semilogy(
                [r["step"] for r in trace],
                [max(r[field], 1e-20) for r in trace],
                "o-",
                color=color,
                label=label,
            )
    axes[0].set(
        title="True prefix recall",
        ylabel="Recovered true prefixes / true prefixes",
        ylim=(-0.03, 1.06),
    )
    axes[1].set(title="Prefix training loss", ylabel="MSE")
    axes[2].set(title="Prefix validation loss", ylabel="MSE")
    for ax in axes:
        ax.set_xlabel("Processed coordinates")
        ax.legend(fontsize=8)
    fig.suptitle(
        "Measured exploratory G31 example: random geometry9922, task2, raw SD0.2"
    )
    fig.savefig(G31 / "figures/prefix_loss_recall.png", dpi=180)
    plt.close(fig)


def head_curves():
    folder = G32 / "confirmation"
    case = "new_0_tied2"
    path = folder / "synchronized_head_ablation" / f"{case}_0"
    synchronized = load_archive(path / "fitted.npz")
    decisions = load_json(folder / "head_gate/decisions" / f"{case}_0.json")
    selected = next(r for r in decisions["selections"] if r["method"] == "keep2_gated")
    prefix = selected["key"]
    candidate = int(prefix[1:])
    index = synchronized[prefix + "_index"]
    codes = synchronized[prefix + "_indexed_codes"]
    basepath = folder / "source" / f"{case}_0"
    initial, fitted = (
        load_archive(basepath / "bootstrap.npz"),
        load_archive(basepath / "fitted.npz"),
    )
    metadata = load_json(basepath / "fitted.json")
    detail = next(row for row in metadata["details"] if row["candidate"] == candidate)
    axis = code_gauges(initial["axis_codes"])[
        np.arange(4), detail["mapping"]["selected_mapping"]
    ]
    states = {"axis": axis, "polished": fitted[prefix + "_codes"]}
    trace = next(
        row
        for row in load_json(path / "fitted.json")["details"]
        if row["candidate"] == candidate
    )
    autos = automorphisms(index, 4)
    curve = None
    for row in trace["curves"]:
        transformed = np.stack(
            [
                states[row["state"]][t] @ autos[row["gauge"][t]]["code_matrix"]
                for t in range(4)
            ]
        )
        if np.allclose(transformed, codes, rtol=1e-10, atol=1e-12):
            curve = row["path"]
            break
    if curve is None:
        raise ValueError("Cannot reconstruct saved selected head trajectory")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    groups = curve[0]["groups"]
    steps = [groups - row["groups"] for row in curve]
    for field, label in (("train_mse", "Train"), ("validation_mse", "H1 validation")):
        axes[0].plot(
            steps,
            [row[field] - curve[0][field] for row in curve],
            "o-",
            markersize=3,
            label=label,
        )
    axes[0].set_yscale("symlog", linthresh=1e-6)
    axes[0].set(
        xlabel="Group merges / removals",
        ylabel="MSE minus initial free-head MSE",
        title="Actual selected gauge / numerical state",
    )
    axes[0].legend()
    gate = load_json(folder / "head_gate/evaluation.json")["rows"]
    for style, marker, color in (
        ("free", "o", "#9b5678"),
        ("tied2", "s", "#277ca4"),
        ("tied4", "^", "#6e8c47"),
    ):
        subset = [
            row
            for row in gate
            if row["kind"] == "synthetic"
            and row["noise"] == 0.2
            and row["style"] == style
            and row["method"] == "keep2_gated"
        ]
        ratio = [
            row["gate_gain"] / row["gate_standard_error"]
            if row["gate_standard_error"] > 0
            else 0
            for row in subset
        ]
        axes[1].scatter(
            range(len(subset)), ratio, marker=marker, color=color, label=style
        )
    axes[1].axhline(2, color="#333333", linestyle="--", label="Accept threshold")
    axes[1].set(
        xlabel="Fresh geometry index",
        ylabel="Independent gate gain / paired SE",
        title="Head-sharing evidence, moderate noise",
    )
    axes[1].legend(fontsize=8)
    fig.savefig(G32 / "figures/head_loss_and_gate.png", dpi=180)
    plt.close(fig)


def provenance():
    verified = {}

    def verify(path, expected):
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"Frozen artifact changed: {path}")
        verified[str(path.relative_to(BASE))] = actual

    for root in (G31, G32):
        folder = root / "confirmation"
        plan = load_json(folder / "protocol.json")
        for name, expected in plan["code_sha256"].items():
            verify(BASE / name, expected)
            verify(folder / "code_snapshot" / name, expected)
        for name, expected in load_json(folder / "data_frozen.json")["sha256"].items():
            verify(folder / name, expected)
        for source in ("source", "synchronized_head_ablation"):
            for path in (folder / source).glob("*/fitted.json"):
                metadata = load_json(path)
                verify(path.parent / "fitted.npz", metadata["sha256"])
        for name, expected in load_json(folder / "head_gate/data_frozen.json")[
            "sha256"
        ].items():
            verify(folder / "head_gate/data" / name, expected)
        for phase in ("predictors", "decisions"):
            freeze = load_json(folder / "head_gate" / f"{phase}_frozen.json")
            for name, expected in freeze["sha256"].items():
                path = folder / "head_gate" / phase / name
                verify(path, expected)
                verify(path.with_suffix(".npz"), load_json(path)["sha256"])
        registration = load_json(folder / "head_gate/registration.json")
        verify(BASE / "sharing_evidence_gate.py", registration["code_sha256"])
        if root == G32:
            if not registration["before_any_new_cohort_bootstrap"]:
                raise ValueError("G32 gate registered after learner started")
            earliest = min(
                load_json(path)["frozen_utc"]
                for path in (folder / "source").glob("*/bootstrap.json")
            )
            if registration["registered_utc"] >= earliest:
                raise ValueError(
                    "G32 gate protocol time is not before bootstrap freeze"
                )
        if (
            load_json(folder / "head_gate/decisions_frozen.json")["frozen_utc"]
            >= load_json(folder / "head_gate/evaluation.json")["evaluated_utc"]
        ):
            raise ValueError("Teacher audit preceded final decision freeze")
    # New synthetic geometries really differ, modulo the permitted label gauge.
    p31, p32 = (
        load_json(G31 / "confirmation/protocol.json"),
        load_json(G32 / "confirmation/protocol.json"),
    )
    from .shared_structure_numeric import canonical_index

    keys31 = {
        canonical_index(
            load_archive(G31 / "confirmation/private" / f"{case['name']}.npz")["index"],
            4,
        ).tobytes()
        for case in p31["cases"]
        if case["kind"] == "synthetic"
    }
    keys32 = {
        canonical_index(
            load_archive(G32 / "confirmation/private" / f"{case['name']}.npz")["index"],
            4,
        ).tobytes()
        for case in p32["cases"]
        if case["kind"] == "synthetic"
    }
    if len(keys32) != 6 or keys31 & keys32:
        raise ValueError(
            "Independent synthetic geometries overlap the exploratory cohort"
        )
    return {
        "verified_utc": timestamp(),
        "verified_artifacts": len(verified),
        "fresh_synthetic_geometries": len(keys32),
        "overlap_with_G31": len(keys31 & keys32),
        "sha256": verified,
    }


def main():
    for root in (G31, G32):
        (root / "figures").mkdir(exist_ok=True)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    summarize(G31)
    final = summarize(G32)
    comparison_figure(final)
    heatmaps()
    prefix_figures()
    head_curves()
    audit = provenance()
    save_json(G32 / "provenance_audit.json", audit)
    print(
        json.dumps(
            {
                "verified": audit["verified_artifacts"],
                "G32_primary": [
                    row for row in final["aggregate"] if row["method"] == "gated"
                ],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
