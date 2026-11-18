"""Freeze source choices, export target inputs and score frozen predictions."""

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import t

from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

REFERENCE = Path("artifacts/omniglot_signature_2026-11-12")


def freeze_source(root, output):
    extraction = json.loads((output / "extraction/metrics.json").read_text())
    budgets = extraction["available_budgets"]
    if not budgets:
        raise RuntimeError("Direct ratio recovery produced no complete budget")
    choices = {
        str(k): digest(root / "fits" / f"ratio_k{k}" / "selected.npz")
        for k in budgets
    }
    save_json(
        root / "source_choices_frozen.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "ratio": choices,
            "candidates_sha256": digest(
                root / "source/ratio_proposals/candidates.npz"
            ),
            "target_answers_used": False,
            "development_data_used": False,
        },
    )
    path = root / "protocol.json"
    protocol = json.loads(path.read_text())
    protocol["requested_code_grid"] = [8, 16, 32, 64]
    protocol["independent_code_grid"] = budgets
    save_json(path, protocol)
    shutil.copy2(
        root / "source_choices_frozen.json",
        output / "source_choices_frozen.json",
    )


def export_targets(root, output):
    validation = root / "validation_fit/ratio"
    if not (validation / "selected.json").exists():
        raise RuntimeError("Development choice must be frozen first")
    targets = root / "targets"
    targets.mkdir()
    private_path = root / "private/target.npz"
    expected_private = json.loads(
        (REFERENCE / "target_data_commitment.json").read_text()
    )["private_target_sha256"]
    if digest(private_path) != expected_private:
        raise RuntimeError("Target observations differ from historical cohort")
    with np.load(private_path) as arrays:
        np.savez_compressed(
            targets / "support.npz",
            x=arrays["support_x"],
            y=arrays["support_y"],
        )
        np.savez_compressed(targets / "query_x.npz", x=arrays["query_x"])
    expected = json.loads(
        (REFERENCE / "development_choices_frozen.json").read_text()
    )
    actual = {
        "support_sha256": digest(targets / "support.npz"),
        "query_inputs_sha256": digest(targets / "query_x.npz"),
    }
    if any(value != expected[key] for key, value in actual.items()):
        raise RuntimeError(
            "Exported target input hashes differ from reference"
        )
    protocol = json.loads((root / "protocol.json").read_text())
    save_json(
        targets / "sizes.json",
        {
            key: protocol[key]
            for key in ("inputs", "hidden", "outputs", "loss")
        },
    )
    destination = targets / "ratio"
    destination.mkdir()
    for name in ("model.npz", "selected.json"):
        shutil.copy2(validation / name, destination / name)
    save_json(
        output / "development_choices_frozen.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "models": {
                "ratio": {
                    name: digest(destination / name)
                    for name in ("model.npz", "selected.json")
                }
            },
            **actual,
            "private_target_sha256": expected_private,
            "query_answers_exported": False,
            "matches_historical_cohort": True,
        },
    )
    compact = output / "development"
    shutil.copytree(validation, compact)
    for name in (
        "prepare.py",
        "extract.py",
        "bank_fit.py",
        "scalar_pairs.py",
        "finish.py",
    ):
        path = Path(__file__).with_name(name)
        shutil.copy2(path, output / "code_snapshot" / name)


def freeze_predictions(root, output):
    folder = root / "targets/ratio"
    models = json.loads(
        (output / "development_choices_frozen.json").read_text()
    )["models"]["ratio"]
    for name, expected in models.items():
        if digest(folder / name) != expected:
            raise RuntimeError(
                f"Changed model after development freeze: {name}"
            )
    registry = json.loads((output / "registration.json").read_text())
    for path, expected in registry["baseline_files"].items():
        if digest(Path(path)) != expected:
            raise RuntimeError(f"Changed comparator: {path}")
    save_json(
        output / "predictions_frozen.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "ratio_sha256": digest(folder / "frozen_predictions.npz"),
            "baseline_files": registry["baseline_files"],
            "new_target_predictions_scored": False,
            "historical_cohort_already_scored_on_12_november": True,
        },
    )
    destination = output / "models/ratio"
    shutil.copytree(folder, destination)


def interval(values):
    mean = float(np.mean(values))
    half = float(
        t.ppf(0.975, len(values) - 1)
        * np.std(values, ddof=1)
        / np.sqrt(len(values))
    )
    return {"mean": mean, "lower": mean - half, "upper": mean + half}


def score(root, output):
    freeze = json.loads((output / "predictions_frozen.json").read_text())
    model_freeze = json.loads(
        (output / "development_choices_frozen.json").read_text()
    )
    new_path = root / "targets/ratio/frozen_predictions.npz"
    if digest(new_path) != freeze["ratio_sha256"]:
        raise RuntimeError("New predictions differ from frozen predictions")
    for name, expected in model_freeze["models"]["ratio"].items():
        if digest(root / "targets/ratio" / name) != expected:
            raise RuntimeError(f"New model differs from frozen model: {name}")
    for path, expected in freeze["baseline_files"].items():
        if digest(Path(path)) != expected:
            raise RuntimeError(f"Changed comparator: {path}")
    private = root / "private/target.npz"
    if digest(private) != model_freeze["private_target_sha256"]:
        raise RuntimeError("Target observations changed after freeze")
    opened = datetime.now(UTC).isoformat()
    with np.load(private) as arrays:
        answers = arrays["query_y"]
    truth = answers.argmax(-1)
    predictions = {"ratio": np.load(new_path)["prediction"]}
    for method in ("current", "dense", "signature"):
        predictions[method] = np.load(
            REFERENCE / "models" / method / "predictions.npz"
        )["prediction"]
    accuracy, ce, rows = {}, {}, []
    for method, prediction in predictions.items():
        if prediction.shape != answers.shape:
            raise RuntimeError(f"Wrong prediction dimensions for {method}")
        accuracy[method] = (prediction.argmax(-1) == truth).mean(1)
        ce[method] = (
            logsumexp(prediction, axis=-1) - (prediction * answers).sum(-1)
        ).mean(1)
        folder = (
            (output if method == "ratio" else REFERENCE) / "models" / method
        )
        selection = json.loads((folder / "selected.json").read_text())
        index = np.load(folder / "model.npz")["index"]
        codes = (
            12544
            if method == "dense"
            else selection["source_selection"]["codes"]
        )
        rows.append(
            {
                "method": method,
                "accuracy": interval(accuracy[method]),
                "cross_entropy": interval(ce[method]),
                "independent_first_weights": codes,
                "task_parameters": codes + 101,
                "active_edges": 12544
                if method == "dense"
                else int(np.count_nonzero(index)),
                "numeric_updates": selection["numeric_updates"],
                "global_setting": selection["global_setting"],
                "development_ce": selection["validation_mean_loss"],
            }
        )
    paired = {
        f"ratio_minus_{method}": {
            "accuracy": interval(accuracy["ratio"] - accuracy[method]),
            "cross_entropy": interval(ce["ratio"] - ce[method]),
        }
        for method in ("current", "dense", "signature")
    }
    receipts = []
    for path in root.rglob("*access_receipt.json"):
        receipt = json.loads(path.read_text())
        receipts.append({"path": str(path), **receipt})
    violations = [v for r in receipts for v in r.get("violations", [])]
    summary = {
        "prediction_freeze_utc": freeze["created_utc"],
        "answers_first_opened_for_new_scoring_utc": opened,
        "target_episodes": len(truth),
        "queries_per_episode": truth.shape[1],
        "comparison": "Retrospective matched-cohort extension",
        "methods": rows,
        "paired": paired,
        "access_receipts": len(receipts),
        "access_violations": violations,
        "ci_scope": "95% episode t intervals conditional on one source cohort",
        "true_sharing_index_observed": False,
    }
    save_json(output / "summary.json", summary)
    np.savez_compressed(
        output / "per_episode.npz",
        **{f"{k}_accuracy": v for k, v in accuracy.items()},
        **{f"{k}_ce": v for k, v in ce.items()},
    )
    np.savez_compressed(output / "scoring_answers.npz", query_y=answers)
    save_json(output / "access_audit.json", {"receipts": receipts})
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=(
            "freeze-source",
            "export-targets",
            "freeze-predictions",
            "score",
        ),
        required=True,
    )
    args = parser.parse_args()
    operation = {
        "freeze-source": freeze_source,
        "export-targets": export_targets,
        "freeze-predictions": freeze_predictions,
        "score": score,
    }[args.stage]
    operation(args.root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
