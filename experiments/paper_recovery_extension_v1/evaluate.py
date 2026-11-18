"""Freeze all budgets, draw one common target cohort, and score it once."""

import argparse
import itertools
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import t

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json

from .prepare import METHODS, OUTPUT, ROOT


def freeze(dataset):
    root, output = ROOT / dataset, OUTPUT / dataset
    protocol = json.loads((root / "protocol.json").read_text())
    d, h, o = [protocol[k] for k in ("inputs", "hidden", "outputs")]
    models, decisions = {}, {}
    dense = json.loads((root / "validation_fit/dense/selected.json").read_text())
    for method in METHODS:
        folder = root / "validation_fit" / method
        selected = json.loads((folder / "selected.json").read_text())
        decisions[method] = {
            "best_constrained_codes": selected["codes"],
            "best_constrained_development_ce": selected["validation_mean_loss"],
            "best_including_unshared": "unshared"
            if dense["validation_mean_loss"] < selected["validation_mean_loss"]
            else method,
            "common_unshared_codes": d * h,
        }
        for record in selected["all_budget_choices"]:
            codes = record["codes"]
            name = "dense" if method == "dense" else f"{method}_k{codes}"
            origin = folder / "budgets" / f"k{codes}"
            target = root / "targets" / name
            target.mkdir()
            for filename in ("model.npz", "selected.json"):
                shutil.copy2(origin / filename, target / filename)
            models[name] = {
                f: digest(target / f) for f in ("model.npz", "selected.json")
            }
    # Execute the full-budget indexator as well as Dense, with identical RNG.
    target = root / "targets/unshared"
    target.mkdir()
    np.savez_compressed(
        target / "model.npz",
        index=np.arange(1, d * h + 1, dtype=np.int16).reshape(d, h),
        codes=np.asarray(d * h),
        scale=np.asarray(1.0),
    )
    save_json(
        target / "selected.json",
        {
            **dense,
            "method": "unshared",
            "codes": d * h,
            "source_selection": {"codes": d * h},
            "unshared": True,
            "model_sha256": digest(target / "model.npz"),
            "selection_provenance": "Dense development rule, reused for the identical full-code model class",
        },
    )
    models["unshared"] = {f: digest(target / f) for f in ("model.npz", "selected.json")}
    save_json(
        output / "choices_frozen_before_target_draw.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "models": models,
            "decisions": decisions,
            "query_answers_opened": False,
            "protocol_sha256": digest(root / "protocol.json"),
        },
    )
    if dataset == "har":
        from .har import episodes, read_split

        x, y, subject = read_split("test")
        panel, ids = episodes(x, y, subject, np.unique(subject), 90, 2611141125)
    else:
        from torchvision.datasets import MNIST, FashionMNIST

        loader = MNIST if dataset == "mnist" else FashionMNIST
        raw = (
            Path("artifacts/local/mnist_dense_sample_size_2026-11-14/raw")
            if dataset == "mnist"
            else Path("artifacts/local/paper_recovery_comparison_2026-11-14/raw")
        )
        train, test = [
            loader(str(raw), train=flag, download=False) for flag in (True, False)
        ]
        x, qx = [data.data.numpy().reshape(-1, 784) for data in (train, test)]
        y, qy = train.targets.numpy(), test.targets.numpy()
        rng = np.random.default_rng(26111498231)
        pairs = list(itertools.combinations((7, 8, 9), 2))
        arrays = {k: [] for k in ("support_x", "support_y", "query_x", "query_y")}
        ids = []
        for episode in range(96):
            classes = pairs[episode % 3]
            support_ids = np.concatenate(
                [rng.choice(np.flatnonzero(y == c), 16, replace=False) for c in classes]
            )
            query_ids = np.concatenate(
                [
                    rng.choice(np.flatnonzero(qy == c), 100, replace=False)
                    for c in classes
                ]
            )
            for key, value in zip(
                arrays,
                (
                    x[support_ids],
                    np.eye(2, dtype=np.float32)[
                        (y[support_ids] == classes[1]).astype(int)
                    ],
                    qx[query_ids],
                    np.eye(2, dtype=np.float32)[
                        (qy[query_ids] == classes[1]).astype(int)
                    ],
                ),
            ):
                arrays[key].append(value)
            ids.append(
                {
                    "classes": classes,
                    "support_ids": support_ids.tolist(),
                    "query_ids": query_ids.tolist(),
                }
            )
        panel = {k: np.stack(v) for k, v in arrays.items()}
    np.savez_compressed(root / "private/target.npz", **panel)
    save_json(root / "private/target_ids.json", ids)
    np.savez_compressed(
        root / "targets/support.npz", x=panel["support_x"], y=panel["support_y"]
    )
    np.savez_compressed(root / "targets/query_x.npz", x=panel["query_x"])
    save_json(
        root / "targets/sizes.json",
        {k: protocol[k] for k in ("inputs", "hidden", "outputs", "loss")},
    )
    save_json(
        output / "development_frozen.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "models": models,
            "decisions": decisions,
            "private_sha256": digest(root / "private/target.npz"),
            "support_sha256": digest(root / "targets/support.npz"),
            "query_inputs_sha256": digest(root / "targets/query_x.npz"),
            "query_answers_opened": False,
        },
    )


def score(dataset):
    root, output = ROOT / dataset, OUTPUT / dataset
    frozen = json.loads((output / "development_frozen.json").read_text())
    predictions = {}
    for name, files in frozen["models"].items():
        folder = root / "targets" / name
        for filename, sha in files.items():
            if digest(folder / filename) != sha:
                raise RuntimeError(f"Changed frozen model {name}/{filename}")
        receipt = json.loads((folder / "access_receipt.json").read_text())
        if receipt["violations"]:
            raise RuntimeError(f"Held-out violation: {name}")
        predictions[name] = digest(folder / "frozen_predictions.npz")
    save_json(
        output / "predictions_frozen.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "predictions_sha256": predictions,
            "query_answers_opened": False,
        },
    )
    if digest(root / "private/target.npz") != frozen["private_sha256"]:
        raise RuntimeError("Changed target observations")
    dense_value, full_value = [
        np.load(root / "targets" / method / "frozen_predictions.npz")["prediction"]
        for method in ("dense", "unshared")
    ]
    exact_endpoint = np.array_equal(dense_value, full_value)
    maximum_difference = float(np.max(np.abs(dense_value - full_value)))
    save_json(
        output / "endpoint_check_before_scoring.json",
        {
            "exact_predictions": exact_endpoint,
            "maximum_logit_difference": maximum_difference,
            "query_answers_opened": False,
        },
    )
    if not np.allclose(dense_value, full_value, rtol=1e-5, atol=1e-5):
        raise RuntimeError("Full-budget endpoint differs materially from Dense")
    truth = np.load(root / "private/target.npz")["query_y"]
    protocol = json.loads((root / "protocol.json").read_text())
    h, o = protocol["hidden"], protocol["outputs"]
    rows, arrays = [], {"truth": truth}
    for name, sha in predictions.items():
        folder = root / "targets" / name
        if digest(folder / "frozen_predictions.npz") != sha:
            raise RuntimeError(f"Changed frozen predictions {name}")
        data = dict(np.load(folder / "frozen_predictions.npz"))
        value = data["prediction"]
        acc = (value.argmax(-1) == truth.argmax(-1)).mean(1)
        ce = (logsumexp(value, axis=-1) - (value * truth).sum(-1)).mean(1)
        selected = json.loads((folder / "selected.json").read_text())
        codes = data["parameters"].shape[-1] - h - h * o - o
        unit_accuracy = acc.reshape(10, 9).mean(0) if dataset == "har" else acc
        half = float(
            t.ppf(0.975, len(unit_accuracy) - 1)
            * unit_accuracy.std(ddof=1)
            / np.sqrt(len(unit_accuracy))
        )
        rows.append(
            {
                "name": name,
                "method": name.split("_k")[0],
                "codes": codes,
                "task_parameters": data["parameters"].shape[-1],
                "numeric_updates": selected["numeric_updates"],
                "accuracy": float(acc.mean()),
                "accuracy_half_ci": half,
                "cross_entropy": float(ce.mean()),
                "development_ce": selected["validation_mean_loss"],
            }
        )
        arrays[f"{name}_prediction"] = value
        arrays[f"{name}_accuracy"] = acc
        arrays[f"{name}_ce"] = ce
    main = []
    for method, decision in frozen["decisions"].items():
        best = (
            "dense"
            if method == "dense"
            else "unshared"
            if decision["best_including_unshared"] == "unshared"
            else f"{method}_k{decision['best_constrained_codes']}"
        )
        main.append(
            {
                "method": method,
                "evaluated_model": best,
                **next(r for r in rows if r["name"] == best),
            }
        )
        main[-1]["method"] = method
    paired = {}
    for other in ("g34", "dense", "lutq", "yeh_map", "yeh_soft", "msr"):
        left = next(r for r in main if r["method"] == "pair")["evaluated_model"]
        right = next(r for r in main if r["method"] == other)["evaluated_model"]
        delta = arrays[f"{left}_accuracy"] - arrays[f"{right}_accuracy"]
        if dataset == "har":
            delta = delta.reshape(10, 9).mean(0)
        paired[other] = {
            "mean": float(delta.mean()),
            "half_ci": float(
                t.ppf(0.975, len(delta) - 1) * delta.std(ddof=1) / np.sqrt(len(delta))
            ),
        }
    np.savez_compressed(output / "scoring.npz", **arrays)
    save_json(
        output / "summary.json",
        {
            "methods": main,
            "all_budgets": rows,
            "decisions": frozen["decisions"],
            "pair_minus": paired,
            "full_code_endpoint_equals_dense_exactly": exact_endpoint,
            "endpoint_maximum_logit_difference": maximum_difference,
            "query_answers_opened_utc": datetime.now(UTC).isoformat(),
            "ci_scope": (
                "Nine held-out subject means, each averaging ten episodes"
                if dataset == "har"
                else "Conditional episodes, fixed class split and source cohort"
            ),
        },
    )
    for method in METHODS:
        shutil.copytree(
            root / "validation_fit" / method, output / "development" / method
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("freeze", "score"), required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    globals()[args.phase](args.dataset)
