"""Summarize frozen G29/G30 structure searches without changing their choices."""

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from experiments.darts_indexator_v1.data import digest, load_archive, save_json

from .pattern_prefix import timestamp

BASE = Path(__file__).resolve().parent
PILOT = BASE / "results/20261106/unknown_readout_g29"
SYNTHETIC = BASE / "results/20261106/unknown_readout_synthetic_g30"


def read(path):
    return json.loads(path.read_text())


def summarize():
    source_rows, aggregate = [], []
    for root in (PILOT, SYNTHETIC):
        rows = read(root / "structure_audit.json")["rows"]
        source_rows.extend(rows)
        for kind in dict.fromkeys(row["kind"] for row in rows):
            for noise in sorted({row["noise"] for row in rows}):
                subset = [r for r in rows if r["kind"] == kind and r["noise"] == noise]
                aggregate.append(
                    {
                        "kind": kind,
                        "noise": noise,
                        "exact": sum(row["exact"] for row in subset),
                        "cases": len(subset),
                        "exact_masks": sum(row["mask_f1"] == 1 for row in subset),
                        "mean_mask_f1": float(np.mean([r["mask_f1"] for r in subset])),
                        "total_errors": sum(
                            row["categorical_errors"] for row in subset
                        ),
                    }
                )
    evaluation = read(PILOT / "target/evaluation.json")
    transfer = []
    for kind in ("pattern", "sets"):
        for noise in (0.0, 0.2):
            for method in ("selected", "oracle", "random"):
                subset = [
                    row
                    for row in evaluation["rows"]
                    if row["kind"] == kind
                    and row["noise"] == noise
                    and row["method"] == method
                ]
                transfer.append(
                    {
                        "kind": kind,
                        "noise": noise,
                        "method": method,
                        "functions": len(subset),
                        "mean_nmse": float(np.mean([row["nmse"] for row in subset])),
                        "worst_nmse": max(row["nmse"] for row in subset),
                    }
                )
    diagnostic = read(SYNTHETIC / "diagnosis/summary.json")["rows"]
    ridge = read(SYNTHETIC / "ridge_ablation/summary.json")["rows"]
    controls = []
    for noise_index, noise in enumerate((0.2, 2.0)):
        for label in diagnostic[0]["exact_tasks"]:
            subset = [
                row["exact_tasks"][label]
                for row in diagnostic
                if row["condition"] == noise_index
            ]
            controls.append(
                {
                    "noise": noise,
                    "control": label,
                    "exact_individual": sum(subset),
                    "total_individual": 24,
                    "true_in_bank": sum(value > 0 for value in subset),
                    "total_banks": 6,
                }
            )
        for multiplier in (0.1, 1.0, 10.0):
            subset = [
                row["exact_tasks"][str(multiplier)]
                for row in ridge
                if row["condition"] == noise_index
            ]
            controls.append(
                {
                    "noise": noise,
                    "control": f"ridge_{multiplier}",
                    "exact_individual": sum(subset),
                    "total_individual": 24,
                    "true_in_bank": sum(value > 0 for value in subset),
                    "total_banks": 6,
                }
            )
    return {
        "created_utc": timestamp(),
        "primary_metric": "Exact signed I modulo global code relabel/sign and hidden permutation; input order fixed. Numerical losses are secondary.",
        "structure": aggregate,
        "source_cases": source_rows,
        "transfer": transfer,
        "pattern_binary": evaluation["pattern_binary"],
        "diagnosis": controls,
        "g30_target_numeric": "Not run: user prioritized I; frozen G29 adaptation is secondary evidence.",
    }


def structure_figures(summary):
    output = PILOT / "figures"
    output.mkdir(exist_ok=True)
    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    )
    rows = summary["structure"]
    labels = [
        "Pattern\n0",
        "Pattern\n0.2",
        "DeepSets\n0",
        "DeepSets\n0.2",
        "Old synthetic\n0.2",
        "Old synthetic\n2",
    ]
    fig, ax = plt.subplots(figsize=(10.5, 4.5), layout="constrained")
    values = [row["exact"] / row["cases"] for row in rows]
    bars = ax.bar(labels, values, color=["#2475a6"] * 4 + ["#cc7844"] * 2)
    for bar, row in zip(bars, rows, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{row['exact']}/{row['cases']}",
            ha="center",
            weight="bold",
        )
    ax.set(
        ylim=(0, 1.18),
        ylabel="Exact selected I / cases",
        xlabel="Task and raw noise SD (answers averaged)",
    )
    ax.set_title("Unknown W2: exact structure, not numerical fit")
    fig.savefig(output / "exact_indexators.png", dpi=180)
    plt.close(fig)

    colors = [
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
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-4.5, 5), cmap.N)
    diffmap = ListedColormap(["#f5f5f5", "#c93d45"])
    chosen = load_archive(PILOT / "aligned_indices.npz")
    fig, axes = plt.subplots(2, 3, figsize=(13, 9), layout="constrained")
    for i, (name, label) in enumerate(
        (("pattern_0", "Pattern 0110"), ("sets_0", "DeepSets"))
    ):
        truth = load_archive(PILOT / "private" / f"{name}.npz")["index"]
        found = chosen[name + "_1"]
        for j, (matrix, title) in enumerate(
            (
                (truth, "True I"),
                (found, "Selected I, raw SD=0.2"),
                (truth != found, "Different entries: 0"),
            )
        ):
            ax = axes[i, j]
            if j == 2:
                ax.imshow(matrix, cmap=diffmap, vmin=0, vmax=1, interpolation="nearest")
            else:
                im = ax.imshow(matrix, cmap=cmap, norm=norm, interpolation="nearest")
                for (d, h), value in np.ndenumerate(matrix):
                    if value:
                        ax.text(
                            h,
                            d,
                            str(value),
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="white" if abs(value) >= 3 else "#222222",
                        )
            ax.set(title=title, xlabel="Aligned hidden neuron")
            if j == 0:
                ax.set_ylabel(label + "\nInput coordinate")
            ax.set_xticks(range(0, truth.shape[1], 2))
            ax.set_yticks(range(0, truth.shape[0], 2))
    fig.colorbar(
        im, ax=axes[:, :2], ticks=range(-4, 5), shrink=0.7, label="Signed group label"
    )
    fig.suptitle(
        "Primary selected integer indexators; input coordinates are not reordered"
    )
    fig.savefig(output / "pattern_deepsets_indices.png", dpi=180)
    plt.close(fig)

    output = SYNTHETIC / "figures"
    output.mkdir(exist_ok=True)
    chosen = load_archive(SYNTHETIC / "aligned_indices.npz")
    fig, axes = plt.subplots(6, 4, figsize=(13, 18), layout="constrained")
    for row, family in enumerate((9901, 9902, 9911, 9912, 9921, 9922)):
        name = f"synthetic_{family}"
        truth = load_archive(SYNTHETIC / "private" / f"{name}.npz")["index"]
        first, second = chosen[name + "_0"], chosen[name + "_1"]
        values = (truth, first, second, first != truth)
        titles = (
            f"{family}: true I",
            f"SD=0.2: {np.sum(first != truth)} errors",
            f"SD=2: {np.sum(second != truth)} errors",
            "SD=0.2: different entries",
        )
        for j, (matrix, title) in enumerate(zip(values, titles, strict=True)):
            axes[row, j].imshow(
                matrix,
                cmap=diffmap if j == 3 else cmap,
                norm=None if j == 3 else norm,
                interpolation="nearest",
                **({"vmin": 0, "vmax": 1} if j == 3 else {}),
            )
            axes[row, j].set(title=title, xticks=[], yticks=[])
    fig.suptitle("Same G23 I and C, new unknown signed W2; all six geometries")
    fig.savefig(output / "synthetic_indices.png", dpi=180)
    plt.close(fig)


def diagnosis_figure(summary):
    rows = read(SYNTHETIC / "diagnosis/summary.json")["rows"]
    labels = [
        "estimated_observed",
        "true_observed",
        "estimated_clean",
        "true_clean",
        "clean_estimated_clean",
    ]
    titles = [
        "Estimated C\nNoisy prefixes",
        "True C\nNoisy prefixes",
        "Noisy-axis C\nClean prefixes",
        "True C\nClean prefixes",
        "Clean-axis C\nClean prefixes",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    for condition, ax in enumerate(axes):
        subset = [r for r in rows if r["condition"] == condition]
        matrix = np.array(
            [[r["exact_tasks"][label] for label in labels] for r in subset]
        )
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=4, aspect="auto")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(
                j,
                i,
                f"{value}/4",
                ha="center",
                va="center",
                color="white" if value >= 3 else "#222222",
            )
        ax.set_xticks(range(5), titles, rotation=25, ha="right")
        ax.set_yticks(range(6), [r["case"].split("_")[-1] for r in subset])
        ax.set(title=f"Raw noise SD={[0.2, 2][condition]}", ylabel="G23 geometry")
    fig.suptitle(
        "Exact individual I: paired post-freeze diagnostics, W2 always unknown\nTrue C and clean responses are oracle controls, not primary results"
    )
    fig.savefig(SYNTHETIC / "figures/prefix_noise_diagnosis.png", dpi=180)
    plt.close(fig)
    with (SYNTHETIC / "diagnosis_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary["diagnosis"][0]))
        writer.writeheader()
        writer.writerows(summary["diagnosis"])


def numeric_curves():
    fitted = load_archive(PILOT / "target_pattern/pattern_fits.npz")
    frozen = read(PILOT / "target_pattern/pattern_frozen.json")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    for kind, color in (
        ("shared", "#2878a5"),
        ("free", "#d78037"),
        ("random_free", "#9f4666"),
    ):
        row = next(r for r in frozen["rows"] if r["head"] == kind and r["seed"] == 0)
        key, winner = row["key"], row["winner"]
        steps = fitted[key + "_steps"]
        axes[0].semilogy(
            steps, fitted[key + "_train"][:, winner], color=color, label=kind + " train"
        )
        axes[0].semilogy(
            steps,
            fitted[key + "_validation"][:, winner],
            "--",
            color=color,
            label=kind + " validation",
        )
        axes[1].plot(steps, fitted[key + "_recall"][:, winner], color=color, label=kind)
    axes[0].set(
        title="Pattern binary: selected start, seed0",
        xlabel="Gradient evaluations",
        ylabel="BCE",
    )
    axes[0].legend(fontsize=8)
    axes[1].set(
        title="Pattern: positive-class recall",
        xlabel="Gradient evaluations",
        ylabel="Validation recall",
        ylim=(0, 1.05),
    )
    axes[1].legend(fontsize=8)
    fit = load_archive(PILOT / "source/sets_0_1/fits_1.npz")
    candidate = read(PILOT / "source/sets_0_1/selection.json")["candidate"]
    train, val = (
        fit[f"candidate_{candidate}_train"],
        fit[f"candidate_{candidate}_validation"],
    )
    starts = val.min(0).reshape(4, 64).argmin(1) + 64 * np.arange(4)
    steps = fit[f"candidate_{candidate}_steps"]
    axes[2].semilogy(steps, train[:, starts].mean(1), label="Train")
    axes[2].semilogy(steps, val[:, starts].mean(1), "--", label="Validation")
    axes[2].set(
        title="DeepSets: joint C/W2 numeric fit",
        xlabel="GN step",
        ylabel="Mean MSE, four source functions",
    )
    axes[2].legend()
    fig.suptitle(
        "Secondary numeric diagnostics; recall here is classification recall, not structure recall"
    )
    fig.savefig(PILOT / "figures/secondary_numeric_curves.png", dpi=180)
    plt.close(fig)


def verify_frozen():
    checked = {}

    def verify(path, expected):
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"Frozen artifact changed: {path}")
        checked[str(path.relative_to(BASE))] = actual

    for root in (PILOT, SYNTHETIC):
        plan = read(root / "protocol.json")
        for name, expected in plan["code_sha256"].items():
            verify(BASE / name, expected)
            verify(root / "code_snapshot" / name, expected)
        data = read(root / "data_frozen.json")
        for name, expected in data["sha256"].items():
            verify(root / "data" / name, expected)
        for name, expected in data["private_sha256"].items():
            verify(root / "private" / name, expected)
        frozen = read(root / "source_frozen.json")
        for name, expected in frozen["sha256"].items():
            verify(root / name, expected)
            selection = read(root / name)
            if not selection["abstained"]:
                verify((root / name).parent / "selected.npz", selection["sha256"])
        if frozen["frozen_utc"] >= read(root / "structure_audit.json")["evaluated_utc"]:
            raise ValueError("Source structural audit preceded freeze")
        for folder in (root / "source").iterdir():
            for phase in (0, 1):
                trace = read(folder / f"fit_{phase}.json")
                if not trace["abstained"]:
                    verify(folder / f"fits_{phase}.npz", trace["sha256"])
    target = read(PILOT / "target/data_frozen.json")
    for name, expected in target["sha256"].items():
        verify(PILOT / "target" / name, expected)
    verify(
        BASE / "unknown_readout_transfer.py",
        target["generator_after_closure_refactor_sha256"],
    )
    cohort = read(PILOT / "target/cohort_frozen.json")
    for name, expected in cohort["sha256"].items():
        verify(PILOT / name, expected)
        trace = read(PILOT / name)
        if name.startswith("target_pattern"):
            verify(PILOT / "target_pattern/pattern_fits.npz", trace["sha256"])
        else:
            verify(
                PILOT / name.replace("frozen_", "fits_").replace(".json", ".npz"),
                trace["sha256"],
            )
    if cohort["frozen_utc"] >= read(PILOT / "target/evaluation.json")["evaluated_utc"]:
        raise ValueError("Target test preceded cohort freeze")
    for folder, runner in (
        ("diagnosis", "weighted_prefix_diagnosis.py"),
        ("ridge_ablation", "prefix_ridge_ablation.py"),
    ):
        registration = read(SYNTHETIC / folder / "registration.json")
        verify(BASE / runner, registration["code_sha256"])
        if (
            registration["registered_utc"]
            <= read(SYNTHETIC / "source_frozen.json")["frozen_utc"]
        ):
            raise ValueError("Control not registered after primary freeze")
    return {"verified_utc": timestamp(), "artifacts": len(checked), "sha256": checked}


def main():
    summary = summarize()
    provenance = verify_frozen()
    save_json(PILOT / "summary.json", summary)
    save_json(SYNTHETIC / "summary.json", summary)
    save_json(PILOT / "provenance_audit.json", provenance)
    with (PILOT / "structure_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary["source_cases"][0]))
        writer.writeheader()
        writer.writerows(summary["source_cases"])
    structure_figures(summary)
    diagnosis_figure(summary)
    numeric_curves()
    print(
        json.dumps(
            {
                "structure": summary["structure"],
                "diagnosis": summary["diagnosis"],
                "verified": provenance["artifacts"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
