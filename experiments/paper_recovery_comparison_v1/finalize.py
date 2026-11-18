"""Audit frozen scientific records and retain the compact comparison."""

import argparse
import hashlib
import itertools
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from torchvision.datasets import MNIST, FashionMNIST

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json


def image_hashes(images):
    return Counter(hashlib.sha256(row.tobytes()).digest() for row in images)


def run(root, output, raw, dataset):
    state = json.loads((output / "workflow.json").read_text())
    if state["status"] != "complete":
        raise RuntimeError("The registered workflow is not complete")
    registration = json.loads((output / "registration.json").read_text())
    expected = registration["code_sha256"]
    snapshots_match = all(
        digest(output / "code_snapshot" / Path(name).name) == value
        for name, value in expected.items()
    )
    source = root / "source"
    dense_sha = digest(source / "dense/models.npz")
    g34 = json.loads((source / "current_proposals/registered.json").read_text())
    pair = json.loads((source / "ratio_proposals/registered.json").read_text())
    shared_dense = dense_sha == g34["dense_sha256"] == pair["dense_sha256"]
    current = Path("experiments/paper_structure_benchmark_v1/methods")
    g34_unchanged = (
        digest(current / "real_project.py") == g34["code_sha256"]
        and digest(current / "scalable_projection.py") == g34["projection_sha256"]
    )
    loader = MNIST if dataset == "mnist" else FashionMNIST
    data = loader(str(raw), train=True, download=False)
    images = data.data.numpy().reshape(-1, 784)
    labels = data.targets.numpy()
    rng = np.random.default_rng(26111491013)
    ids = {c: rng.permutation(np.flatnonzero(labels == c)) for c in range(5)}
    disjoint = all(
        not (set(a) & set(b))
        for c in range(5)
        for a, b in itertools.combinations(
            (ids[c][:4000], ids[c][4000:4500], ids[c][4500:5000]), 2
        )
    )
    data_checks = []
    for panel, start, stop in (
        ("support", 0, 4000),
        ("numerical", 4000, 4500),
        ("structure", 4500, 5000),
    ):
        path = root / "public" / f"{panel}.npz"
        if digest(path) != registration["source_sha256"][panel]:
            raise RuntimeError("Source observations changed after registration")
        expected_hashes = {
            c: image_hashes(images[ids[c][start:stop]]) for c in range(5)
        }
        with np.load(path) as archive:
            x, y = archive["x"], archive["y"].argmax(-1)
            for task, classes in enumerate(registration["source_pairs"]):
                for label, c in enumerate(classes):
                    actual = x[task, y[task] == label]
                    data_checks.append(
                        len(actual) == stop - start
                        and image_hashes(actual) == expected_hashes[c]
                    )
    frozen = json.loads((output / "predictions_frozen.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    refit_checks = []
    for method in ("g34", "pair"):
        for codes in (8, 16, 32, 64):
            folder = root / "fits" / f"{method}_k{codes}"
            record = json.loads((folder / "registered.json").read_text())
            selected = json.loads((folder / "selected.json").read_text())
            refit_checks.append(
                record["steps"] == 2000
                and record["starts"] == 18
                and len(record["candidates"]) <= 8
                and selected["codes"] == codes
                and abs(
                    selected["source_M_loss"] - min(selected["all_source_M_losses"])
                )
                < 1e-8
            )
    tuning_checks = []
    for method in ("g34", "pair", "dense"):
        folder = root / "validation_fit" / method
        selected = json.loads((folder / "selected.json").read_text())
        with np.load(folder / "curves.npz") as curves:
            for item in selected["all_budget_choices"]:
                codes = item["codes"]
                errors = curves[f"k{codes}_validation_history"]
                scores = errors.mean(-1).reshape(-1, 9, 2).mean(-1)
                checkpoint, _ = np.unravel_index(scores.argmin(), scores.shape)
                tuning_checks.append(
                    abs(float(scores.min()) - item["validation_mean_loss"]) < 1e-8
                    and curves[f"k{codes}_steps"][checkpoint] == item["numeric_updates"]
                )
        tuning_checks.append(
            selected["support_count"] == 32
            and selected["validation_mean_loss"]
            == min(
                row["validation_mean_loss"] for row in selected["all_budget_choices"]
            )
        )
    freeze_before_labels = datetime.fromisoformat(
        frozen["created_utc"]
    ) < datetime.fromisoformat(summary["query_answers_opened_utc"])
    score_checks = []
    with np.load(output / "scoring.npz") as archive:
        truth = archive["truth"]
        for row in summary["methods"]:
            method = row["method"]
            value = archive[f"{method}_prediction"]
            acc = (value.argmax(-1) == truth.argmax(-1)).mean(1)
            ce = (logsumexp(value, axis=-1) - (value * truth).sum(-1)).mean(1)
            score_checks.append(
                abs(acc.mean() - row["accuracy"]) < 1e-10
                and abs(ce.mean() - row["cross_entropy"]) < 1e-7
            )
            file = root / "targets" / method / "frozen_predictions.npz"
            score_checks.append(digest(file) == frozen["predictions_sha256"][method])
            with np.load(file) as raw:
                score_checks.append(
                    np.array_equal(raw["prediction"], value)
                    and raw["parameters"].shape == (96, row["task_parameters"])
                )
    checks = {
        "workflow_complete": True,
        "registered_code_snapshots_match": snapshots_match,
        "same_dense_archive_for_both_extractors": shared_dense,
        "unchanged_g34_implementation": g34_unchanged,
        "source_F_H_M_original_ids_disjoint": disjoint,
        "source_panels_match_registered_original_images_and_labels": all(data_checks),
        "audited_source_class_panels": len(data_checks),
        "all_predictions_frozen_before_query_answers": freeze_before_labels,
        "published_scores_and_prediction_hashes_match": all(score_checks),
        "both_methods_all_budgets_refitted_under_common_budget": all(refit_checks),
        "development_selection_matches_two_restart_averaged_curves": all(tuning_checks),
        "access_violations": summary["access_violations"],
        "same_budget_grid": [8, 16, 32, 64],
        "software_tests_created_or_run": False,
    }
    required = [
        value
        for key, value in checks.items()
        if isinstance(value, bool) and key != "software_tests_created_or_run"
    ]
    if not all(required) or checks["access_violations"]:
        raise RuntimeError(f"Scientific record audit failed: {checks}")
    shutil.copytree(source / "current_proposals", output / "g34/extraction")
    shutil.copy2(source / "dense/models.npz", output / "shared_dense_models.npz")
    shutil.copy2(source / "dense/polish.json", output / "dense_polish.json")
    receipts = output / "receipts"
    receipts.mkdir()
    for file in root.rglob("*access_receipt.json"):
        shutil.copy2(file, receipts / "__".join(file.relative_to(root).parts))
    save_json(output / "audit.json", checks)
    shutil.copy2(__file__, output / "finalize.py")
    save_json(
        output / "compact_manifest.json",
        {
            str(p.relative_to(output)): digest(p)
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "compact_manifest.json"
        },
    )
    print(json.dumps({"dataset": dataset, "audit": checks}))


def main():
    parser = argparse.ArgumentParser()
    for name in ("root", "output", "raw"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--dataset", choices=("mnist", "fashion"), required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve(), args.raw.resolve(), args.dataset)


if __name__ == "__main__":
    main()
