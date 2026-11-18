"""Freeze added-method choices, reproduce the original episodes, then score."""

import argparse
import itertools
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import t

from experiments.paper_recovery_extension_v1 import har
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

from .prepare import ORIGINAL, OUTPUT, PREVIOUS, ROOT


def freeze(dataset):
    root, output = ROOT / dataset, OUTPUT / dataset
    protocol = json.loads((root / "protocol.json").read_text())
    folder = root / "validation_fit/asrr"
    selected = json.loads((folder / "selected.json").read_text())
    dense = json.loads(
        (root / "validation_fit/dense/selected.json").read_text()
    )
    decision = {
        "best_constrained_codes": selected["codes"],
        "best_constrained_development_ce": selected["validation_mean_loss"],
        "best_including_unshared": "unshared"
        if dense["validation_mean_loss"] < selected["validation_mean_loss"]
        else "asrr",
        "common_unshared_codes": protocol["inputs"] * protocol["hidden"],
    }
    models = {}
    for record in selected["all_budget_choices"]:
        codes = record["codes"]
        origin = folder / "budgets" / f"k{codes}"
        target = root / "targets" / f"asrr_k{codes}"
        target.mkdir()
        for name in ("model.npz", "selected.json"):
            shutil.copy2(origin / name, target / name)
        models[f"asrr_k{codes}"] = {
            name: digest(target / name)
            for name in ("model.npz", "selected.json")
        }
    save_json(
        output / "choices_frozen.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "models": models,
            "decision": decision,
            "query_answers_opened": False,
            "target_cohort": "Exact original episode draw, already evaluated for baseline methods",
        },
    )
    if dataset == "har":
        har.RAW = ROOT / "raw/uci_har.zip"
        x, y, subjects = har.read_split("test")
        panel, ids = har.episodes(
            x, y, subjects, np.unique(subjects), 90, 2611141125
        )
    else:
        from torchvision.datasets import MNIST, FashionMNIST

        loader = MNIST if dataset == "mnist" else FashionMNIST
        raw = (
            Path("artifacts/local/mnist_dense_sample_size_2026-11-14/raw")
            if dataset == "mnist"
            else Path(
                "artifacts/local/paper_recovery_comparison_2026-11-14/raw"
            )
        )
        train, test = [
            loader(str(raw), train=flag, download=False)
            for flag in (True, False)
        ]
        x, qx = [data.data.numpy().reshape(-1, 784) for data in (train, test)]
        y, qy = train.targets.numpy(), test.targets.numpy()
        rng = np.random.default_rng(26111498231)
        pairs = list(itertools.combinations((7, 8, 9), 2))
        arrays = {
            k: [] for k in ("support_x", "support_y", "query_x", "query_y")
        }
        ids = []
        for episode in range(96):
            classes = pairs[episode % 3]
            support = np.concatenate(
                [
                    rng.choice(np.flatnonzero(y == c), 16, replace=False)
                    for c in classes
                ]
            )
            query = np.concatenate(
                [
                    rng.choice(np.flatnonzero(qy == c), 100, replace=False)
                    for c in classes
                ]
            )
            for key, value in zip(
                arrays,
                (
                    x[support],
                    np.eye(2, dtype=np.float32)[
                        (y[support] == classes[1]).astype(int)
                    ],
                    qx[query],
                    np.eye(2, dtype=np.float32)[
                        (qy[query] == classes[1]).astype(int)
                    ],
                ),
            ):
                arrays[key].append(value)
            ids.append(
                {
                    "classes": classes,
                    "support_ids": support.tolist(),
                    "query_ids": query.tolist(),
                }
            )
        panel = {k: np.stack(v) for k, v in arrays.items()}
    np.savez_compressed(root / "private/target.npz", **panel)
    save_json(root / "private/target_ids.json", ids)
    np.savez_compressed(
        root / "targets/support.npz",
        x=panel["support_x"],
        y=panel["support_y"],
    )
    np.savez_compressed(root / "targets/query_x.npz", x=panel["query_x"])
    save_json(
        root / "targets/sizes.json",
        {k: protocol[k] for k in ("inputs", "hidden", "outputs", "loss")},
    )
    save_json(
        output / "observations_frozen.json",
        {
            "private_sha256": digest(root / "private/target.npz"),
            "support_sha256": digest(root / "targets/support.npz"),
            "query_inputs_sha256": digest(root / "targets/query_x.npz"),
        },
    )


def half_interval(values):
    return float(
        t.ppf(0.975, len(values) - 1)
        * values.std(ddof=1)
        / np.sqrt(len(values))
    )


def score(dataset):
    root, output = ROOT / dataset, OUTPUT / dataset
    frozen = json.loads((output / "choices_frozen.json").read_text())
    predictions = {}
    for name, hashes in frozen["models"].items():
        folder = root / "targets" / name
        for filename, expected in hashes.items():
            if digest(folder / filename) != expected:
                raise RuntimeError(f"Changed frozen input {name}/{filename}")
        receipt = json.loads((folder / "access_receipt.json").read_text())
        if receipt["violations"]:
            raise RuntimeError(f"Held-out data violation: {name}")
        predictions[name] = digest(folder / "frozen_predictions.npz")
    save_json(
        output / "predictions_frozen.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "predictions_sha256": predictions,
            "query_answers_opened_for_asrr_scoring": False,
        },
    )
    observations = json.loads(
        (output / "observations_frozen.json").read_text()
    )
    if digest(root / "private/target.npz") != observations["private_sha256"]:
        raise RuntimeError("Changed target observations")
    truth = np.load(root / "private/target.npz")["query_y"]
    with np.load(PREVIOUS / dataset / "scoring.npz") as original:
        if not np.array_equal(truth, original["truth"]):
            raise RuntimeError("Original target labels do not match")
        baseline_arrays = {
            k: original[k]
            for k in original.files
            if k.endswith("_accuracy") or k.endswith("_ce")
        }
    original_frozen = json.loads(
        (PREVIOUS / dataset / "development_frozen.json").read_text()
    )
    same_observations = all(
        observations[k] == original_frozen[k]
        for k in ("private_sha256", "support_sha256", "query_inputs_sha256")
    )
    if not same_observations:
        raise RuntimeError(
            "Original episode archives differ; inspect before comparing"
        )
    rows, arrays = [], {"truth": truth}
    for name in predictions:
        folder = root / "targets" / name
        data = dict(np.load(folder / "frozen_predictions.npz"))
        value = data["prediction"]
        if not np.isfinite(value).all():
            raise RuntimeError(f"Nonfinite predictions: {name}")
        accuracy = (value.argmax(-1) == truth.argmax(-1)).mean(1)
        ce = (logsumexp(value, axis=-1) - (value * truth).sum(-1)).mean(1)
        record = json.loads((folder / "selected.json").read_text())
        unit_accuracy = (
            accuracy.reshape(10, 9).mean(0) if dataset == "har" else accuracy
        )
        row = {
            "name": name,
            "method": "asrr",
            "codes": record["codes"],
            "task_parameters": data["parameters"].shape[-1],
            "numeric_updates": record["numeric_updates"],
            "accuracy": float(accuracy.mean()),
            "accuracy_half_ci": half_interval(unit_accuracy),
            "cross_entropy": float(ce.mean()),
            "development_ce": record["validation_mean_loss"],
        }
        rows.append(row)
        arrays.update(
            {
                f"{name}_prediction": value,
                f"{name}_accuracy": accuracy,
                f"{name}_ce": ce,
            }
        )
    decision = frozen["decision"]
    selected = next(
        r for r in rows if r["codes"] == decision["best_constrained_codes"]
    )
    original_summary = json.loads(
        (PREVIOUS / dataset / "summary.json").read_text()
    )
    paired = {}
    for method in ("dense", "g34"):
        codes = original_summary["decisions"][method]["best_constrained_codes"]
        name = "dense" if method == "dense" else f"{method}_k{codes}"
        delta = (
            arrays[f"{selected['name']}_accuracy"]
            - baseline_arrays[f"{name}_accuracy"]
        )
        if dataset == "har":
            delta = delta.reshape(10, 9).mean(0)
        paired[method] = {
            "accuracy_difference": float(delta.mean()),
            "half_ci": half_interval(delta),
        }
    np.savez_compressed(output / "scoring.npz", **arrays)
    save_json(
        output / "summary.json",
        {
            "selected_constrained": selected,
            "all_budgets": rows,
            "decision": decision,
            "paired_constrained_minus": paired,
            "same_original_observations": same_observations,
            "query_scored_utc": datetime.now(UTC).isoformat(),
        },
    )


def score_synthetic():
    inputs = {}
    for number in (4, 5):
        folder = ROOT / f"synthetic_{number}/targets/case_{number:02d}_c0/asrr"
        receipt = json.loads((folder / "access_receipt.json").read_text())
        if receipt["violations"]:
            raise RuntimeError("Synthetic target access violation")
        inputs[str(number)] = digest(folder / "frozen_predictions.npz")
    save_json(
        OUTPUT / "synthetic_predictions_frozen.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "sha256": inputs,
            "query_answers_opened_for_asrr_scoring": False,
        },
    )
    from experiments.single_task_recovery_v1.audit import index_metrics

    rows, arrays = [], {}
    original = np.load(PREVIOUS / "synthetic_scoring.npz")
    for number in (4, 5):
        root, output = (
            ROOT / f"synthetic_{number}",
            OUTPUT / f"synthetic_{number}",
        )
        truth = np.load(root / "private/answers.npz")["y"]
        if not np.array_equal(truth, original[f"case{number}_truth"]):
            raise RuntimeError("Different synthetic target functions")
        value = np.load(
            root / f"targets/case_{number:02d}_c0/asrr/frozen_predictions.npz"
        )["prediction_256"]
        if not np.isfinite(value).all():
            raise RuntimeError("Nonfinite synthetic predictions")
        index = np.load(root / "fits/structures.npz")["index"]
        teacher = np.load(ORIGINAL / f"private/case_{number:02d}.npz")["index"]
        mse = ((value - truth) ** 2).mean((1, 2))
        rows.append(
            {
                "geometry": number,
                "method": "asrr",
                "mse": float(mse.mean()),
                "per_task_mse": mse.tolist(),
                "structure": index_metrics(index, teacher),
            }
        )
        arrays.update(
            {f"case{number}_truth": truth, f"case{number}_prediction": value}
        )
    np.savez_compressed(OUTPUT / "synthetic_scoring.npz", **arrays)
    save_json(OUTPUT / "synthetic_results.json", {"rows": rows})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=("freeze", "score", "score-synthetic")
    )
    parser.add_argument("--dataset")
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze(args.dataset)
    elif args.phase == "score":
        score(args.dataset)
    else:
        score_synthetic()
