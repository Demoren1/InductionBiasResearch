"""Score immutable predictions after opening the separate evaluation answers."""

import csv
import json
import os
from itertools import combinations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from scipy.stats import t

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json

from .run import OUTPUT, ROOT, stamp

LABELS = {
    "dense": "Dense",
    "fixed_g34": "Fixed G34",
    "adapt": "G34: dense adaptation",
    "prior": "G34: discrete prior",
    "single": "Single-task Lloyd",
    "msr_structure": "MSR: structure only",
    "msr_native": "MSR: learned initialization",
    "positive_prior": "G34: positive penalty",
}
COLORS = (
    "#6b7280",
    "#0284c7",
    "#e8790c",
    "#7c3aed",
    "#059669",
    "#be123c",
    "#111827",
    "#a855f7",
)


def interval(values):
    values = np.asarray(values)
    mean = float(values.mean())
    half = float(
        t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    )
    return {"mean": mean, "lower": mean - half, "upper": mean + half}


def anchor_disagreement(anchor, indices):
    """Match code labels and code signs in the already aligned neuron order."""
    old = anchor.ravel()
    codes = int(np.abs(old).max())
    disagreements = []
    for row in indices.reshape(-1, len(old)):
        active = (old != 0) & (row != 0)
        key = (np.abs(old[active]).astype(int) - 1) * codes + np.abs(row[active]) - 1
        counts = np.bincount(key, minlength=codes * codes).reshape(codes, codes)
        votes = np.bincount(
            key,
            weights=np.sign(old[active]) * np.sign(row[active]),
            minlength=codes * codes,
        ).reshape(codes, codes)
        score = (counts + np.abs(votes)) / 2
        first, second = linear_sum_assignment(-score)
        matches = score[first, second].sum() + np.sum((old == 0) & (row == 0))
        disagreements.append(1 - matches / len(old))
    return interval(disagreements)


def run():
    frozen = json.loads((OUTPUT / "predictions_frozen.json").read_text())
    if digest(OUTPUT / "choices_frozen.json") != frozen["choices_sha256"]:
        raise RuntimeError("Global choices changed after predictions were frozen")
    if digest(ROOT / "target_inputs.npz") != frozen["inputs_sha256"]:
        raise RuntimeError("Target inputs changed")
    for name, expected in frozen["predictions"].items():
        if digest(ROOT / "test" / name / "fit.npz") != expected:
            raise RuntimeError(f"Predictions changed: {name}")
    native = json.loads((OUTPUT / "native_predictions_frozen.json").read_text())
    if digest(ROOT / "test/msr_native/fit.npz") != native["predictions_sha256"]:
        raise RuntimeError("Native MSR predictions changed")
    if digest(OUTPUT / "native_choices_frozen.json") != native["choices_sha256"]:
        raise RuntimeError("Native MSR global settings changed")
    if digest(ROOT / "target_inputs.npz") != native["inputs_sha256"]:
        raise RuntimeError("Native MSR target inputs changed")
    positive = json.loads((OUTPUT / "positive_predictions_frozen.json").read_text())
    if digest(ROOT / "test/positive_prior/fit.npz") != positive["predictions_sha256"]:
        raise RuntimeError("Positive-penalty predictions changed")
    if digest(OUTPUT / "positive_choices_frozen.json") != positive["choices_sha256"]:
        raise RuntimeError("Positive-penalty global settings changed")
    if digest(ROOT / "target_inputs.npz") != positive["inputs_sha256"]:
        raise RuntimeError("Positive-penalty target inputs changed")
    created = json.loads((OUTPUT / "test_created.json").read_text())
    answers = ROOT / "private/query_answers.npz"
    if digest(answers) != created["answers_sha256"]:
        raise RuntimeError("Target evaluation answers changed")
    if not (OUTPUT / "scoring_started.json").exists():
        save_json(OUTPUT / "scoring_started.json", stamp(predictions_verified=True))
    with np.load(answers) as arrays:
        truth = arrays["y"]
    with np.load(ROOT / "target_inputs.npz") as arrays:
        train_y, valid_y = arrays["train_y"], arrays["valid_y"]
    choices = json.loads((OUTPUT / "choices_frozen.json").read_text())["choices"]
    positive_choice = json.loads((OUTPUT / "positive_choices_frozen.json").read_text())[
        "chosen"
    ]
    native_choice = json.loads((OUTPUT / "native_choices_frozen.json").read_text())[
        "chosen"
    ]
    LABELS["prior"] = f"G34: selected penalty {choices['prior']['penalty']:g}"
    LABELS["positive_prior"] = f"G34: penalty {positive_choice['penalty']:g}"
    with np.load(ROOT / "source_bank.npz") as arrays:
        anchor = arrays[choices["common"]["name"]]
    metrics, traces, rows = {}, {}, []
    for name, label in LABELS.items():
        with np.load(ROOT / "test" / name / "fit.npz") as arrays:
            logits = arrays["prediction"]
            accuracy = (logits.argmax(-1) == truth.argmax(-1)).mean(1)
            ce = (logsumexp(logits, axis=-1) - (logits * truth).sum(-1)).mean(1)
            train = (arrays["train_prediction"].argmax(-1) == train_y.argmax(-1)).mean(
                1
            )
            valid = (arrays["valid_prediction"].argmax(-1) == valid_y.argmax(-1)).mean(
                1
            )
            codes = int(arrays["codes"])
            selected_step = arrays["selected_step"]
            row = {
                "method": name,
                "label": label,
                "accuracy": interval(accuracy),
                "ce": interval(ce),
                "train_accuracy": interval(train),
                "valid_accuracy": interval(valid),
                "codes": codes,
                "median_selected_step": float(np.median(selected_step)),
            }
        for task in range(len(accuracy)):
            rows.append(
                {
                    "method": name,
                    "task": task,
                    "accuracy": accuracy[task],
                    "ce": ce[task],
                    "train_accuracy": train[task],
                    "valid_accuracy": valid[task],
                }
            )
        path = ROOT / "test" / name / "changes.json"
        if path.exists():
            changes = json.loads(path.read_text())
            row["assignment_change_fraction"] = interval(changes["fraction"])
            row["zero_fraction"] = interval(changes["zero_fraction"])
            with np.load(ROOT / "test" / name / "index.npz") as arrays:
                row["anchor_disagreement_after_signed_code_matching"] = (
                    anchor_disagreement(anchor, arrays["index"])
                )
        metrics[name] = {**row, "accuracy_values": accuracy, "ce_values": ce}
        traces[name] = json.loads((ROOT / "test" / name / "trace.json").read_text())
    comparisons = []
    for first, second in combinations(LABELS, 2):
        comparisons.append(
            {
                "first": first,
                "second": second,
                "accuracy_difference": interval(
                    metrics[first]["accuracy_values"]
                    - metrics[second]["accuracy_values"]
                ),
                "ce_difference": interval(
                    metrics[first]["ce_values"] - metrics[second]["ce_values"]
                ),
            }
        )
    summary = [
        {key: value for key, value in row.items() if not key.endswith("_values")}
        for row in metrics.values()
    ]
    save_json(
        OUTPUT / "summary.json",
        {"methods": summary, "paired": comparisons, "choices": choices},
    )
    with (OUTPUT / "per_episode.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (OUTPUT / "summary.csv").open("w") as stream:
        fields = (
            "method",
            "accuracy",
            "accuracy_ci_half",
            "ce",
            "codes",
            "train_accuracy",
            "valid_accuracy",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            writer.writerow(
                {
                    "method": row["method"],
                    "accuracy": row["accuracy"]["mean"],
                    "accuracy_ci_half": row["accuracy"]["upper"]
                    - row["accuracy"]["mean"],
                    "ce": row["ce"]["mean"],
                    "codes": row["codes"],
                    "train_accuracy": row["train_accuracy"]["mean"],
                    "valid_accuracy": row["valid_accuracy"]["mean"],
                }
            )
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/ibr-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    displayed = [row for row in summary if row["method"] != "prior"]
    positions = np.arange(len(displayed))
    palette = dict(zip(LABELS, COLORS))
    for ax, field, multiplier, title in zip(
        axes,
        ("accuracy", "ce"),
        (100, 1),
        ("Evaluation accuracy / macro recall (%)", "Evaluation cross-entropy"),
    ):
        means = [row[field]["mean"] * multiplier for row in displayed]
        errors = [
            (row[field]["upper"] - row[field]["mean"]) * multiplier for row in displayed
        ]
        ax.barh(
            positions,
            means,
            xerr=errors,
            color=[palette[row["method"]] for row in displayed],
            capsize=3,
            alpha=0.9,
        )
        ax.set_yticks(positions, [row["label"] for row in displayed])
        ax.invert_yaxis()
        ax.set_xlabel(title)
        ax.grid(axis="x", alpha=0.2)
    figure.suptitle(
        "Omniglot · 784–16–5 ReLU · 128 fresh 5-way tasks\n10 train + 5 validation + 5 evaluation drawings per class; 95% episode CIs"
    )
    for suffix in ("png", "pdf"):
        figure.savefig(OUTPUT / f"comparison.{suffix}", dpi=180)
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
    for (name, label), color in zip(LABELS.items(), COLORS):
        if name == "prior":
            continue
        trace = traces[name]
        for ax, field, scale in zip(
            axes, ("train", "validation", "recall"), (1, 1, 100)
        ):
            ax.plot(
                trace["step"],
                np.asarray(trace[field]) * scale,
                label=label,
                color=color,
            )
    for ax, title, ylabel in zip(
        axes,
        ("Training", "Numerical validation", "Numerical validation"),
        ("Cross-entropy", "Cross-entropy", "Accuracy / macro recall (%)"),
    ):
        ax.set(title=title, xlabel="Optimizer step", ylabel=ylabel)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    figure.suptitle(
        "Fresh models: mean of six numerical fits; native MSR: selected SGD rule\nEvaluation answers are not used for checkpoint selection"
    )
    for suffix in ("png", "pdf"):
        figure.savefig(OUTPUT / f"learning_curves.{suffix}", dpi=180)
    plt.close(figure)
    lines = [
        "# Omniglot: dense-first adaptation with a discrete prior",
        "",
        "15 November 2026. Fresh 128-task cohort, conditional on the retained source ensemble.",
        "",
        "784–16–5 ReLU; five distinct characters per task, each with 10 fitting, 5 numerical-validation and 5 evaluation drawings. No query augmentation. This differs from the earlier augmented 5-shot cohort and from the native MSR CNN benchmark.",
        "",
        "| Method | Accuracy, % (95% CI) | CE | Codes | Train accuracy, % | Validation accuracy, % |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        mean = row["accuracy"]["mean"] * 100
        half = (row["accuracy"]["upper"] - row["accuracy"]["mean"]) * 100
        lines.append(
            f"| {row['label']} | {mean:.2f} ± {half:.2f} | {row['ce']['mean']:.4f} | {row['codes']} | {row['train_accuracy']['mean'] * 100:.2f} | {row['valid_accuracy']['mean'] * 100:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Global development choices",
            "",
            f"Selected common indexator: `{choices['common']['name']}`.",
            f"Ordinary adaptation: {choices['plain']}; prior adaptation: {choices['prior']}.",
            f"Single-task code budget: {choices['single']['codes']}.",
            f"Strictly positive penalty control: {positive_choice}.",
            f"Retained native MSR adaptation rule: {native_choice}.",
            "",
            "The common structure and budget were selected first; the penalty is compared on that same structure. The unshared 12544-weight control was also evaluated on development. The MSR structure-only row transfers its retained continuous basis with fresh numerical fitting. The separate learned-initialization row retains its 8-dimensional basis, source numerical initialization and learned signed per-group SGD rates; development selects a rate multiplier and step count. Both are retained MLP controls, not a new reproduction of the native CNN benchmark.",
            "",
            "## Paired comparisons",
            "",
        ]
    )
    for row in comparisons:
        if row["first"] in ("fixed_g34", "adapt") and row["second"] == "prior":
            value = row["accuracy_difference"]
            lines.append(
                f"{LABELS[row['first']]} minus prior: {value['mean'] * 100:+.2f} pp, 95% CI [{value['lower'] * 100:+.2f}, {value['upper'] * 100:+.2f}]."
            )
    lines.extend(
        [
            "",
            "The two main adaptation policies include a depth-zero candidate. A separate control selects among strictly positive penalties. The selected-penalty policy chose zero, so its predictions equal ordinary adaptation; the duplicate is omitted from plots. Fresh numerical parameters are fitted for all structure-only methods using the same 50 fitting and 25 validation labels per task. The native MSR control instead uses its learned initialization and the development-selected SGD rule.",
            "",
            "Intervals summarize sampled episodes on one retained source ensemble. They do not measure uncertainty across independently learned source banks. There is no known true Omniglot indexator: assignment changes are not structural recovery accuracy. Additional source proposals use a smaller finite source search than the original retained 64-code models.",
            "",
            "![Comparison](comparison.png)",
            "",
            "![Learning curves](learning_curves.png)",
            "",
        ]
    )
    (OUTPUT / "REPORT.md").write_text("\n".join(lines))
    save_json(OUTPUT / "complete.json", stamp(methods=len(summary), tasks=len(truth)))
    print(json.dumps({"methods": summary, "paired": comparisons}, indent=2))


if __name__ == "__main__":
    run()
