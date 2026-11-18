"""Develop on separate classes, freeze choices, predict, then score once."""

import argparse
import heapq
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp

from experiments.omniglot_prior_v1.model import (
    align,
    calibration,
    numerical_fit,
)
from experiments.omniglot_prior_v1.run import episodes as omniglot_episodes
from experiments.paper_recovery_extension_v1.adapter import fit_adam, values
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json

from .extract import DEST, ROOT

OLD_OMNI = Path("artifacts/local/omniglot_prior_2026-11-15")
OLD_REAL = Path("artifacts/local/paper_recovery_extension_2026-11-14")
MNIST_RAW = Path("artifacts/local/mnist_dense_sample_size_2026-11-14/raw")


def mnist_episodes(split, count, seed):
    """Draw fresh binary tasks, preserving class and official data splits."""
    from torchvision.datasets import MNIST

    train, test = [
        MNIST(str(MNIST_RAW), train=flag, download=False) for flag in (True, False)
    ]
    images, query_images = [
        item.data.numpy().reshape(-1, 784).astype(np.float32) / 255
        for item in (train, test)
    ]
    labels, query_labels = train.targets.numpy(), test.targets.numpy()
    pairs = ((5, 6),) if split == "development" else ((7, 8), (7, 9), (8, 9))
    rng = np.random.default_rng(seed)
    panels = {
        f"{part}_{key}": []
        for part in ("train", "valid", "query")
        for key in ("x", "y")
    }
    provenance = []
    for episode in range(count):
        classes = pairs[episode % len(pairs)]
        train_ids = [
            rng.choice(np.flatnonzero(labels == value), 150, replace=False)
            for value in classes
        ]
        query_ids = [
            rng.choice(np.flatnonzero(query_labels == value), 100, replace=False)
            for value in classes
        ]
        for part, ids, data in (
            ("train", [row[:100] for row in train_ids], images),
            ("valid", [row[100:] for row in train_ids], images),
            ("query", query_ids, query_images),
        ):
            panels[f"{part}_x"].append(data[np.concatenate(ids)])
            panels[f"{part}_y"].append(
                np.repeat(np.eye(2, dtype=np.float32), len(ids[0]), axis=0)
            )
        provenance.append(
            {
                "classes": classes,
                "train_and_valid_ids": [row.tolist() for row in train_ids],
                "query_ids": [row.tolist() for row in query_ids],
            }
        )
    return {key: np.stack(rows) for key, rows in panels.items()}, provenance


def refinements(anchor, weights, uncertainty):
    """Extend G34 capacity using only calibrated source weight trajectories."""
    aligned, error = align(anchor, weights, uncertainty)
    _, scales = calibration(np.broadcast_to(anchor, aligned.shape), aligned, error)
    deviation = np.median(error * scales, axis=1).clip(1e-6)
    calibrated = aligned * scales[:, None] * np.sign(anchor)[None]
    features = (calibrated / deviation[:, None, None]).reshape(len(weights), -1).T
    groups = {
        int(group): np.flatnonzero(np.abs(anchor).ravel() == group)
        for group in np.unique(np.abs(anchor))
        if group
    }
    queue = []

    def push(group):
        positions = groups[group]
        if len(positions) > 1:
            score = np.square(features[positions] - features[positions].mean(0)).sum()
            heapq.heappush(queue, (-float(score), group))

    for group in groups:
        push(group)
    next_group = max(groups) + 1
    result = {}
    for codes in (1024, 2048, 4096, 8192):
        while len(groups) < codes:
            _, group = heapq.heappop(queue)
            positions = groups.pop(group)
            axis = np.var(features[positions], axis=0).argmax()
            order = np.argsort(features[positions, axis], kind="stable")
            half = len(positions) // 2
            groups[group], groups[next_group] = (
                positions[order[:half]],
                positions[order[half:]],
            )
            push(group)
            push(next_group)
            next_group += 1
        labels = np.zeros(anchor.size, dtype=np.int16)
        for label, group in enumerate(sorted(groups), 1):
            labels[groups[group]] = label
        result[f"g34_refined_m{codes}"] = labels.reshape(anchor.shape) * np.sign(anchor)
    return result


def metric(prediction, truth):
    return {
        "ce": float((logsumexp(prediction, -1) - (prediction * truth).sum(-1)).mean()),
        "accuracy": float((prediction.argmax(-1) == truth.argmax(-1)).mean()),
    }


def prepare():
    if not (DEST / "feasibility.json").exists():
        raise RuntimeError("Complete the four-dataset recovery stage first")
    registration = DEST / "predictive_registered.json"
    if registration.exists():
        raise RuntimeError("Predictive follow-up is already registered")
    save_json(
        registration,
        {
            "amendment_sha256": digest(
                Path(__file__).with_name("PREDICTIVE_AMENDMENT.md")
            ),
            "code_sha256": digest(Path(__file__)),
            "source_feasibility_sha256": digest(DEST / "feasibility.json"),
            "fresh_test_generated": False,
        },
    )
    jobs, models = [], {}
    for dataset in ("omniglot", "mnist"):
        folder = ROOT / dataset
        (folder / "models").mkdir()
        (folder / "panels").mkdir()
        (folder / "fits/development").mkdir(parents=True)
        if dataset == "omniglot":
            with np.load(OLD_OMNI / "development.npz") as data:
                panels = dict(data)
            provenance = json.loads(
                (OLD_OMNI / "development_provenance.json").read_text()
            )
            with np.load(OLD_OMNI / "source_bank.npz") as data:
                bank = {f"g34_{key}": data[key] for key in data.files}
            with np.load(OLD_OMNI / "budget_followup/bank.npz") as data:
                bank.update({f"g34_refined_{key}": data[key] for key in data.files})
            msr_path = Path(
                "artifacts/benchmark_2026-11-07/omniglot_pilot/targets/msr/model.npz"
            )
            msr_paths = {"msr_retained_m64": msr_path}
        else:
            panels, provenance = mnist_episodes("development", 24, 3511158600)
            bank = {}
            msr_paths = {}
            for codes in (8, 16, 32, 64, 128, 512):
                with np.load(
                    OLD_REAL / f"mnist/targets/g34_k{codes}/model.npz"
                ) as data:
                    bank[f"g34_m{codes}"] = data["index"]
                msr_paths[f"msr_m{codes}"] = (
                    OLD_REAL / f"mnist/targets/msr_k{codes}/model.npz"
                )
            with np.load(folder / "descriptors.npz") as data:
                bank.update(
                    refinements(bank["g34_m512"], data["normal"], data["uncertainty"])
                )
        np.savez_compressed(folder / "panels/development.npz", **panels)
        save_json(folder / "panels/development_provenance.json", provenance)
        with np.load(folder / "candidate_bank.npz") as data:
            summary = json.loads((DEST / dataset / "summary.json").read_text())
            for key in data.files:
                codes = int(key[1:])
                if codes >= 2 and summary["choices"][str(codes)]["weight_accepted"]:
                    bank[f"srr_point_{key}"] = data[key]
        for name, path in msr_paths.items():
            with np.load(path) as data:
                np.savez_compressed(folder / f"models/{name}.npz", basis=data["basis"])
        for name, index in bank.items():
            np.savez_compressed(folder / f"models/{name}.npz", index=index)
        bank["dense"] = None
        entries = []
        for name in (*bank, *msr_paths):
            family = "dense" if name == "dense" else name.split("_")[0]
            codes = 12544
            path = folder / f"models/{name}.npz"
            if name in msr_paths:
                with np.load(path) as data:
                    codes = data["basis"].shape[-1]
            elif bank[name] is not None:
                codes = int(np.abs(bank[name]).max())
            entry = {
                "dataset": dataset,
                "name": name,
                "family": family,
                "codes": codes,
                "model_sha256": digest(path) if path.exists() else None,
            }
            if dataset == "omniglot" and family != "srr":
                if name == "dense":
                    old = OLD_OMNI / "development/dense/fit.npz"
                elif name.startswith("g34_refined_"):
                    old = (
                        OLD_OMNI
                        / f"budget_followup/development/shared_m{codes}/fit.npz"
                    )
                elif family == "msr":
                    old = OLD_OMNI / "development/msr_structure/fit.npz"
                else:
                    old = OLD_OMNI / f"development/{name[4:]}/fit.npz"
                with np.load(old) as data:
                    prediction = data["prediction"]
                destination = folder / f"fits/development/{name}.npz"
                np.savez_compressed(destination, prediction=prediction)
                entry["reused_development"] = {"path": str(old), "sha256": digest(old)}
            else:
                jobs.append({"dataset": dataset, "name": name, "cohort": "development"})
            entries.append(entry)
        models[dataset] = entries
    save_json(DEST / "models_frozen.json", models)
    save_json(DEST / "development_jobs.json", jobs)
    print(
        json.dumps(
            {
                "development_jobs": len(jobs),
                "models": {key: len(value) for key, value in models.items()},
            }
        ),
        flush=True,
    )


def worker(dataset, name, cohort, device, *, study_root=ROOT, numerical_seed=None):
    torch.set_num_threads(2)
    folder = study_root / dataset
    output = folder / f"fits/{cohort}/{name}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite completed predictions: {output}")
    with np.load(
        folder / f"panels/{cohort}_inputs.npz"
        if cohort == "test"
        else folder / "panels/development.npz"
    ) as data:
        panels = {
            key: data[key]
            for key in ("train_x", "train_y", "valid_x", "valid_y", "query_x")
        }
    path = folder / f"models/{name}.npz"
    index, basis = None, None
    if not name.startswith("dense"):
        with np.load(path) as data:
            if "index" in data.files:
                index = data["index"]
            else:
                basis = data["basis"]
    torch.zeros(1, device=device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    if dataset == "omniglot":
        fitted = numerical_fit(
            panels,
            index=index,
            basis=basis,
            precompute=False,
            device=device,
            seed=3511158401 if numerical_seed is None else numerical_seed,
        )
        trace = fitted.pop("trace")
        fitted.pop("ensemble_parameters")
    else:
        x, y, vx, vy, qx = [
            torch.as_tensor(panels[key], dtype=torch.float32, device=device)
            for key in ("train_x", "train_y", "valid_x", "valid_y", "query_x")
        ]
        representation = None
        if index is not None:
            representation = {
                "index": torch.as_tensor(index, dtype=torch.long, device=device),
                "codes": int(np.abs(index).max()),
                "scale": 1.0,
            }
        elif basis is not None:
            representation = torch.as_tensor(basis, dtype=torch.float32, device=device)
        settings = [
            {"lr": rate, "quantile": 0.5, "restart": restart}
            for rate in (0.003, 0.01, 0.03)
            for restart in range(2)
        ]
        parameter, trace = fit_adam(
            x,
            y,
            vx,
            vy,
            representation,
            16,
            2,
            steps=200,
            loss="cross_entropy",
            seed=3511158603 if numerical_seed is None else numerical_seed,
            keep=1,
            settings=settings,
        )
        with torch.no_grad():
            fitted = {
                "prediction": values(qx, representation, parameter, 16, 2)[0]
                .cpu()
                .numpy(),
                "weight_parameters": parameter[0].cpu().numpy(),
                "valid_prediction": values(vx, representation, parameter, 16, 2)[0]
                .cpu()
                .numpy(),
            }
    np.savez_compressed(output, **fitted)
    np.savez_compressed(
        output.with_suffix(".trace.npz"),
        **{key: np.asarray(value) for key, value in trace.items() if key != "settings"},
    )
    torch.cuda.synchronize(device)
    save_json(
        output.with_suffix(".complete.json"),
        {
            "dataset": dataset,
            "name": name,
            "cohort": cohort,
            "seconds": time.perf_counter() - started,
            "prediction_sha256": digest(output),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "test_query_labels_read": False,
        },
    )
    print(
        json.dumps(
            {
                "dataset": dataset,
                "name": name,
                "cohort": cohort,
                "seconds": time.perf_counter() - started,
            }
        ),
        flush=True,
    )


def freeze():
    models = json.loads((DEST / "models_frozen.json").read_text())
    choices, jobs = {}, []
    for dataset, entries in models.items():
        folder = ROOT / dataset
        truth = np.load(folder / "panels/development.npz")["query_y"]
        rows = []
        for entry in entries:
            path = folder / f"fits/development/{entry['name']}.npz"
            with np.load(path) as data:
                score = metric(data["prediction"], truth)
            rows.append(
                {**entry, **score, "development_predictions_sha256": digest(path)}
            )
        selected = {}
        for family in ("dense", "g34", "srr", "msr"):
            options = [row for row in rows if row["family"] == family]
            selected[family] = (
                min(options, key=lambda row: (row["ce"], row["codes"]))
                if options
                else None
            )
        choices[dataset] = {
            "selected": selected,
            "development": rows,
            "test_query_labels_opened": False,
        }
    save_json(DEST / "choices_frozen.json", choices)
    for dataset, decision in choices.items():
        folder = ROOT / dataset
        selected = decision["selected"]
        if dataset == "omniglot":
            panels, provenance = omniglot_episodes("test", 128, 3511158601)
        else:
            panels, provenance = mnist_episodes("test", 128, 3511158602)
        (folder / "private").mkdir()
        np.savez_compressed(
            folder / "private/test_query_answers.npz", y=panels.pop("query_y")
        )
        np.savez_compressed(folder / "panels/test_inputs.npz", **panels)
        save_json(folder / "private/test_provenance.json", provenance)
        test_models = {
            family: row["name"] for family, row in selected.items() if row is not None
        }
        np.savez_compressed(
            folder / "models/unshared.npz",
            index=np.arange(1, 12545, dtype=np.int16).reshape(784, 16),
        )
        test_models["unshared"] = "unshared"
        # Also compare G34 at the selected SRR budget when that bank exists.
        if selected["srr"] is not None:
            matched = [
                row
                for row in decision["development"]
                if row["family"] == "g34" and row["codes"] == selected["srr"]["codes"]
            ]
            if matched:
                test_models["g34_matched"] = min(matched, key=lambda row: row["ce"])[
                    "name"
                ]
        save_json(
            DEST / dataset / "test_registered.json",
            {
                "choices_sha256": digest(DEST / "choices_frozen.json"),
                "inputs_sha256": digest(folder / "panels/test_inputs.npz"),
                "answers_sha256": digest(folder / "private/test_query_answers.npz"),
                "models": test_models,
                "query_labels_opened": False,
            },
        )
        for name in dict.fromkeys(test_models.values()):
            jobs.append({"dataset": dataset, "name": name, "cohort": "test"})
    save_json(DEST / "test_jobs.json", jobs)
    print(
        json.dumps(
            {
                "test_jobs": len(jobs),
                "choices": {
                    key: {
                        family: None if row is None else row["name"]
                        for family, row in value["selected"].items()
                    }
                    for key, value in choices.items()
                },
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "worker", "freeze"))
    parser.add_argument("--dataset", choices=("omniglot", "mnist"))
    parser.add_argument("--name")
    parser.add_argument("--cohort", choices=("development", "test"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.action == "worker":
        worker(args.dataset, args.name, args.cohort, args.device)
    else:
        globals()[args.action]()


if __name__ == "__main__":
    main()
