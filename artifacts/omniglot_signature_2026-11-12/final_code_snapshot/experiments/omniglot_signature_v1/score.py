"""Open target answers after the common freeze and publish paired results."""

import argparse
import csv
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
from scipy.special import logsumexp
from scipy.stats import t

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json

from .pipeline import METHODS

LABELS = {
    "dense": "Dense",
    "current": "G34",
    "signature": "Weighted SVD: geometric top-one",
    "signature_random": "Shuffled signature structure",
}


def interval(values):
    mean = float(np.mean(values))
    half = float(
        t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / np.sqrt(len(values))
    )
    return {"mean": mean, "lower": mean - half, "upper": mean + half}


def run(root, output):
    freeze = json.loads((output / "predictions_frozen.json").read_text())
    models = json.loads((output / "development_choices_frozen.json").read_text())[
        "models"
    ]
    for method in METHODS:
        path = root / "targets" / method / "frozen_predictions.npz"
        if digest(path) != freeze["methods"][method]:
            raise RuntimeError(f"Frozen prediction mismatch: {method}")
        for name, expected in models[method].items():
            if digest(root / "targets" / method / name) != expected:
                raise RuntimeError(f"Frozen model mismatch: {method}/{name}")
    commitment = json.loads((output / "target_data_commitment.json").read_text())
    if digest(root / "private" / "target.npz") != commitment["private_target_sha256"]:
        raise RuntimeError("Target observations changed after their commitment")
    opened = datetime.now(UTC).isoformat()
    target = np.load(root / "private" / "target.npz")
    answers = target["query_y"]
    truth = answers.argmax(-1)
    rows, by_task, losses = [], {}, {}
    receipts, violations = 0, []
    receipt_paths = sorted(
        set(root.rglob("*access_receipt.json"))
        | set(output.glob("*access_receipt.json"))
    )
    for path in receipt_paths:
        receipt = json.loads(path.read_text())
        receipts += 1
        violations.extend(receipt.get("violations", []))
    for method in METHODS:
        folder = root / "targets" / method
        arrays = np.load(folder / "frozen_predictions.npz")
        prediction = arrays["prediction"]
        accuracy = (prediction.argmax(-1) == truth).mean(1)
        ce = (logsumexp(prediction, axis=-1) - (prediction * answers).sum(-1)).mean(1)
        by_task[method], losses[method] = accuracy, ce
        selection = json.loads((folder / "selected.json").read_text())
        index = np.load(folder / "model.npz")["index"]
        codes = 12544 if method == "dense" else selection["source_selection"]["codes"]
        row = {
            "method": method,
            "label": LABELS[method],
            "accuracy": interval(accuracy),
            "cross_entropy": interval(ce),
            "independent_first_weights": codes,
            "total_task_parameters": int(arrays["parameters"].shape[-1]),
            "active_edges": 12544
            if method == "dense"
            else int(np.count_nonzero(index)),
            "numeric_updates": selection["numeric_updates"],
            "global_setting": selection["global_setting"],
            "development_ce": selection["validation_mean_loss"],
        }
        rows.append(row)
        destination = output / "models" / method
        destination.mkdir(parents=True, exist_ok=True)
        for name in (
            "selected.json",
            "model.npz",
            "access_receipt.json",
            "metrics.json",
        ):
            shutil.copy2(folder / name, destination / name)
        np.savez_compressed(destination / "predictions.npz", prediction=prediction)
    paired = {}
    for reference in ("dense", "current", "signature_random"):
        paired[f"signature_minus_{reference}"] = {
            "accuracy": interval(by_task["signature"] - by_task[reference]),
            "cross_entropy": interval(losses["signature"] - losses[reference]),
        }
    ranks = list((output / "ranks").rglob("r*.json"))
    complete = []
    for path in ranks:
        record = json.loads(path.read_text())
        complete.extend(
            {"space": record["key"], **item} for item in record["candidates"]
        )
    summary = {
        "target_answers_first_opened_utc": opened,
        "prediction_freeze_utc": freeze["created_utc"],
        "target_episodes": len(truth),
        "queries_per_episode": truth.shape[1],
        "source_episodes": 128,
        "development_episodes": 128,
        "methods": rows,
        "paired": paired,
        "rank_jobs_completed": len(ranks),
        "complete_partitions": complete,
        "access_receipts": receipts,
        "access_violations": violations,
        "ci_scope": "95% t intervals over target episodes, conditional on one fixed source cohort/training",
        "neural_ground_truth_indexator_available": False,
    }
    save_json(output / "summary.json", summary)
    np.savez_compressed(
        output / "per_episode.npz",
        **{f"{m}_accuracy": by_task[m] for m in METHODS},
        **{f"{m}_ce": losses[m] for m in METHODS},
    )
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "method",
                "accuracy_percent",
                "ce",
                "codes",
                "parameters",
                "edges",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "method": row["method"],
                    "accuracy_percent": 100 * row["accuracy"]["mean"],
                    "ce": row["cross_entropy"]["mean"],
                    "codes": row["independent_first_weights"],
                    "parameters": row["total_task_parameters"],
                    "edges": row["active_edges"],
                }
            )
    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    labels = [
        "Dense",
        "G34",
        "Weighted SVD\ngeometric top-one",
        "Shuffled\nsignature I",
    ]
    means = np.array([row["accuracy"]["mean"] for row in rows]) * 100
    errors = (
        np.array([row["accuracy"]["upper"] - row["accuracy"]["mean"] for row in rows])
        * 100
    )
    ax.bar(
        labels,
        means,
        yerr=errors,
        capsize=4,
        color=["#7e8d9e", "#d48827", "#337faa", "#aeb9c4"],
    )
    for number, value in enumerate(means):
        ax.text(number, value + errors[number] + 0.5, f"{value:.2f}%", ha="center")
    ax.set_ylim(0, max(means + errors) + 5)
    ax.set_ylabel("Target query accuracy (%)")
    ax.set_title("Aug-Omniglot · FC 784–16–5 · 5-way 5-shot · 1000 new episodes")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "accuracy.png", dpi=180)
    fig.savefig(output / "accuracy.pdf")
    plt.close(fig)
    text = [
        "# Weighted SVD: first geometric candidate on Aug-Omniglot",
        "",
        (
            "The signature structure for each budget was selected only by the frozen "
            "geometric score. No source-functional reranking was used. Disjoint "
            "development tasks selected the code budget and one common support-only "
            "numerical adaptation rule. All target predictions were frozen before "
            "opening the new episode answers."
        ),
        "",
        "## Comparison",
        "",
        "| Method | Accuracy (95% CI) | Cross-entropy | First-layer free values | All task parameters | Active edges |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        acc = row["accuracy"]
        text.append(
            f"| {row['label']} | {100 * acc['mean']:.2f}% "
            f"[{100 * acc['lower']:.2f}, {100 * acc['upper']:.2f}] | "
            f"{row['cross_entropy']['mean']:.4f} | {row['independent_first_weights']} | "
            f"{row['total_task_parameters']} | {row['active_edges']} |"
        )
    text.extend(["", "## Paired differences", ""])
    for reference in ("dense", "current", "signature_random"):
        acc = paired[f"signature_minus_{reference}"]["accuracy"]
        text.append(
            f"- Signature − {LABELS[reference]}: {100 * acc['mean']:+.3f} "
            f"percentage points; 95% CI [{100 * acc['lower']:+.3f}, {100 * acc['upper']:+.3f}]."
        )
    text.extend(
        [
            "",
            "## Scope and diagnostics",
            "",
            "- Same fully connected architecture and query-only augmentation as the historical comparison; no pretrained encoder.",
            "- Source panels reproduce their historical SHA-256 commitments; development uses the historical generator and seed. Dense models and G34 were rerun; final target episodes use a new seed on the same test class split.",
            "- One source cohort, 128 source tasks, 128 development tasks; confidence intervals measure episode variation, not independent source retraining.",
            f"- Completed rank jobs: {len(ranks)}; complete unsigned partitions: {len(complete)}. Missing code budgets were not filled by forced clustering.",
            f"- Access receipts: {receipts}; violations: {len(violations)}.",
            "- There is no known true neural grouping on Omniglot. Predictive transfer cannot establish exact structural recovery.",
            "",
            "![Accuracy comparison](accuracy.png)",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(text))
    print(json.dumps({"methods": rows, "paired": paired}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
