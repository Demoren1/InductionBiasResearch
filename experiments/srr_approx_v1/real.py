"""Source hierarchies, development-guided A1, and frozen real-task comparison."""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from operator import itemgetter
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp

from experiments.omniglot_prior_v1.run import episodes as omniglot_episodes
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from experiments.srr_real_v1.extract import SOURCES
from experiments.srr_real_v1.predictive import mnist_episodes
from experiments.srr_real_v1.predictive import worker as numerical_worker

from .core import (
    canonical,
    input_forest,
    local_forest,
    signed_forest,
    synchronize_profiles,
)
from .projection import project_chunked

ROOT = Path("artifacts/local/srr_approx_real_2026-11-15")
DEST = Path("artifacts/srr_approx_real_2026-11-15")
PREVIOUS = Path("artifacts/local/srr_real_2026-11-15")
OLD_DEST = Path("artifacts/srr_real_2026-11-15")
BUDGETS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 12544)
EXTRA_SEEDS = (3511158811, 3511158812, 3511158813)


def source_profiles(dataset):
    path, dimensions = SOURCES[dataset]
    with np.load(path) as data:
        ranking = data["numerical_cross_entropy"].min(0).argsort()
        ranking = ranking[:16] if dataset == "omniglot" else ranking
        selected = (
            data["parameters"][ranking, : dimensions * 16]
            .reshape(-1, dimensions, 16)
            .astype(float)
        )
        ensembles = (
            data["ensemble_parameters"][ranking, :, : dimensions * 16]
            .reshape(len(ranking), 4, dimensions, 16)
            .astype(float)
        )
    normal = selected / np.maximum(
        np.linalg.norm(selected, axis=1, keepdims=True), 1e-12
    )
    consensus, uncertainty = [], []
    for reference, ensemble in zip(normal, ensembles):
        aligned = []
        for weight in ensemble:
            unit = weight / np.maximum(np.linalg.norm(weight, axis=0), 1e-12)
            left, right = linear_sum_assignment(-reference.T @ unit)
            aligned.append(unit[:, right[np.argsort(left)]])
        aligned = np.stack(aligned)
        consensus.append(aligned.mean(0))
        uncertainty.append(np.sqrt(np.mean((aligned - reference) ** 2, axis=(0, 1))))
    consensus = np.stack(consensus)
    uncertainty = np.maximum(uncertainty, 0.0003)
    matched, means, deviations = [], [], []
    for reference, mean, error in zip(normal, consensus, uncertainty):
        left, right = linear_sum_assignment(-normal[0].T @ reference)
        order = right[np.argsort(left)]
        matched.append(reference[:, order])
        means.append(mean[:, order])
        deviations.append(error[order])
    return np.stack(matched), np.stack(means), np.stack(deviations), ranking


def add_model(dataset, name, family, *, index=None, basis=None, **metadata):
    folder = ROOT / dataset
    path = folder / f"models/{name}.npz"
    codes, density = 12544, None
    if index is not None:
        index = canonical(index)
        codes = int(np.abs(index).max())
        density = float(np.mean(index != 0))
        np.savez_compressed(path, index=index)
    elif basis is not None:
        codes = basis.shape[-1]
        np.savez_compressed(path, basis=basis)
    row = {
        "dataset": dataset,
        "name": name,
        "family": family,
        "codes": codes,
        "density": density,
        "model_sha256": digest(path) if path.exists() else None,
        **metadata,
    }
    save_json(folder / f"model_records/{name}.json", row)
    return row


def prepare(device):
    torch.set_num_threads(2)
    synthetic = Path("artifacts/srr_approx_synthetic_2026-11-15/results.json")
    if not synthetic.exists():
        raise RuntimeError("Finish the controlled synthetic stage first")
    ROOT.mkdir(parents=True, exist_ok=False)
    DEST.mkdir(parents=True, exist_ok=False)
    save_json(
        DEST / "registered.json",
        {
            "protocol_sha256": digest(Path(__file__).with_name("REAL_PROTOCOL.md")),
            "theory_sha256": digest(
                Path("paper/theory/APPROXIMATE_SRR_THEORY_2026-11-15.md")
            ),
            "synthetic_results_sha256": digest(synthetic),
            "source_sha256": {
                name: digest(SOURCES[name][0]) for name in ("omniglot", "mnist")
            },
            "budgets": BUDGETS,
            "test_seeds": [3511158801, 3511158802],
            "test_episodes": 256,
            "test_generated": False,
        },
    )
    previous_models = json.loads((OLD_DEST / "models_frozen.json").read_text())
    previous_choices = json.loads((OLD_DEST / "choices_frozen.json").read_text())
    jobs = []
    for dataset in ("omniglot", "mnist"):
        folder = ROOT / dataset
        for name in (
            "models",
            "model_records",
            "panels",
            "logs",
            "fits/development",
            "trees",
        ):
            (folder / name).mkdir(parents=True)
        with np.load(PREVIOUS / dataset / "panels/development.npz") as data:
            np.savez_compressed(folder / "panels/development.npz", **dict(data))
        selected, mean, error, ranking = source_profiles(dataset)
        np.savez_compressed(
            folder / "source_profiles.npz",
            selected=selected,
            consensus=mean,
            uncertainty=error,
            ranking=ranking,
        )
        print(
            json.dumps(
                {"dataset": dataset, "stage": "source_profiles", "tasks": len(mean)}
            ),
            flush=True,
        )
        global_bank, record = signed_forest(mean, BUDGETS, device=device)
        np.savez_compressed(folder / "trees/global.npz", tree=record["tree"])
        synchronized = synchronize_profiles(global_bank[256], mean, error)
        sync_bank, record = signed_forest(synchronized, BUDGETS, device=device)
        np.savez_compressed(folder / "trees/synchronized.npz", tree=record["tree"])
        families = {
            "asrr_global": global_bank,
            "asrr_sync": sync_bank,
            "asrr_local": local_forest(mean, BUDGETS),
            "asrr_input": input_forest(mean, BUDGETS),
            "asrr_selected_input": input_forest(selected, BUDGETS),
        }
        for family, bank in families.items():
            for codes, index in bank.items():
                if codes == 12544:
                    continue
                name = f"{family}_m{codes}"
                add_model(
                    dataset,
                    name,
                    family,
                    index=index,
                    nominal_codes=codes,
                    source_information="consensus"
                    if family != "asrr_selected_input"
                    else "selected",
                )
                jobs.append({"dataset": dataset, "name": name, "cohort": "development"})
        for entry in previous_models[dataset]:
            if entry["family"] != "g34":
                continue
            name = entry["name"]
            with np.load(PREVIOUS / dataset / f"models/{name}.npz") as data:
                add_model(
                    dataset,
                    name,
                    "g34",
                    index=data["index"],
                    nominal_codes=entry["codes"],
                    source_information="retained practical G34 bank",
                )
            jobs.append({"dataset": dataset, "name": name, "cohort": "development"})
        for family in ("msr", "srr"):
            entry = previous_choices[dataset]["selected"][family]
            name = f"legacy_{family}"
            with np.load(PREVIOUS / dataset / f"models/{entry['name']}.npz") as data:
                if "index" in data:
                    add_model(dataset, name, f"legacy_{family}", index=data["index"])
                else:
                    add_model(dataset, name, f"legacy_{family}", basis=data["basis"])
            jobs.append({"dataset": dataset, "name": name, "cohort": "development"})
        add_model(dataset, "dense", "dense")
        jobs.append({"dataset": dataset, "name": "dense", "cohort": "development"})
        for pool, seed in enumerate(EXTRA_SEEDS, 1):
            name = f"dense_seed{pool}"
            add_model(dataset, name, "dense_pool")
            jobs.append(
                {
                    "dataset": dataset,
                    "name": name,
                    "cohort": "development",
                    "numerical_seed": seed,
                }
            )
        print(
            json.dumps({"dataset": dataset, "stage": "source_bank_complete"}),
            flush=True,
        )
    save_json(DEST / "initial_jobs.json", jobs)
    print(json.dumps({"initial_jobs": len(jobs)}), flush=True)


def worker(job, device):
    dataset, name, cohort = (job[key] for key in ("dataset", "name", "cohort"))
    if job.get("kind") == "projection":
        folder = ROOT / dataset
        with np.load(folder / "source_profiles.npz") as data:
            weights, error = data["consensus"], data["uncertainty"]
        with np.load(folder / f"models/{job['seed_name']}.npz") as data:
            initial = data["index"]
        steps = 12 if int(np.abs(initial).max()) <= 1024 else 4
        started = time.perf_counter()
        index, trace = project_chunked(
            initial, weights, error, steps=steps, device=device
        )
        add_model(
            dataset,
            name,
            job["family"],
            index=index,
            seed_name=job["seed_name"],
            projection_trace=trace,
            projection_seconds=time.perf_counter() - started,
            nominal_codes=int(np.abs(initial).max()),
        )
    numerical_worker(
        dataset,
        name,
        cohort,
        device,
        study_root=ROOT,
        numerical_seed=job.get("numerical_seed"),
    )


def gpu_worker(item):
    device, jobs = item
    rows = []
    for number, job in jobs:
        folder = ROOT / job["dataset"]
        completed = folder / f"fits/{job['cohort']}/{job['name']}.complete.json"
        if completed.exists():
            rows.append({**job, "exit_code": 0, "reused_completed": True})
            continue
        log = folder / f"logs/{job['cohort']}_{job['name']}.log"
        command = [
            sys.executable,
            "-m",
            "experiments.srr_approx_v1.real",
            "worker",
            "--queue",
            job["queue"],
            "--job",
            str(number),
            "--device",
            f"cuda:{device}",
        ]
        started = time.perf_counter()
        with log.open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=stream, check=False)
        row = {
            **job,
            "gpu": device,
            "exit_code": result.returncode,
            "seconds": time.perf_counter() - started,
            "log": str(log),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        if result.returncode:
            break
    return rows


def queue(name):
    jobs = json.loads((DEST / f"{name}_jobs.json").read_text())
    indexed = [(number, {**job, "queue": name}) for number, job in enumerate(jobs)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        groups = list(
            executor.map(
                gpu_worker, [(device, indexed[device::8]) for device in range(8)]
            )
        )
    rows = [row for group in groups for row in group]
    save_json(DEST / f"{name}_queue_complete.json", rows)
    if len(rows) != len(jobs) or any(row["exit_code"] for row in rows):
        raise RuntimeError("Numerical jobs failed; inspect recorded logs")


def combine_dense(dataset, cohort, *, study_root=ROOT):
    folder = study_root / dataset
    panels = np.load(
        folder / f"panels/{cohort}_inputs.npz"
        if cohort == "test"
        else folder / "panels/development.npz"
    )
    truth = panels["valid_y"]
    fitted = [
        dict(np.load(folder / f"fits/{cohort}/{name}.npz"))
        for name in ("dense", "dense_seed1", "dense_seed2", "dense_seed3")
    ]
    loss = np.stack(
        [
            (
                logsumexp(row["valid_prediction"], -1)
                - (row["valid_prediction"] * truth).sum(-1)
            ).mean(-1)
            for row in fitted
        ]
    )
    chosen = loss.argmin(0)
    tasks = np.arange(len(chosen))
    path = folder / f"fits/{cohort}/dense24.npz"
    np.savez_compressed(
        path,
        prediction=np.stack([row["prediction"] for row in fitted])[chosen, tasks],
        valid_prediction=np.stack([row["valid_prediction"] for row in fitted])[
            chosen, tasks
        ],
        selected_pool=chosen,
    )
    save_json(
        path.with_suffix(".complete.json"),
        {
            "prediction_sha256": digest(path),
            "seed_pools": 4,
            "selection": "minimum within-task numerical-validation CE",
            "test_query_labels_read": False,
        },
    )


def development():
    rows = []
    for dataset in ("omniglot", "mnist"):
        folder = ROOT / dataset
        truth = np.load(folder / "panels/development.npz")["query_y"]
        for path in sorted((folder / "model_records").glob("*.json")):
            entry = json.loads(path.read_text())
            prediction_path = folder / f"fits/development/{entry['name']}.npz"
            if not prediction_path.exists() or entry["family"] == "dense_pool":
                continue
            prediction = np.load(prediction_path)["prediction"]
            ce = float(
                (logsumexp(prediction, -1) - (prediction * truth).sum(-1)).mean()
            )
            accuracy = float((prediction.argmax(-1) == truth.argmax(-1)).mean())
            rows.append(
                {
                    **entry,
                    "ce": ce,
                    "accuracy": accuracy,
                    "fit_sha256": digest(prediction_path),
                }
            )
    return rows


def refine():
    for dataset in ("omniglot", "mnist"):
        combine_dense(dataset, "development")
        add_model(dataset, "dense24", "dense", seed_pools=4)
    rows = development()
    save_json(DEST / "initial_development.json", rows)
    jobs = []
    for dataset in ("omniglot", "mnist"):
        families = sorted(
            {
                row["family"]
                for row in rows
                if row["dataset"] == dataset and row["family"].startswith("asrr")
            }
        )
        for family in (*families, "g34"):
            best = min(
                (
                    row
                    for row in rows
                    if row["dataset"] == dataset and row["family"] == family
                ),
                key=lambda row: (row["ce"], row["codes"]),
            )
            name = f"a1_{best['name']}"
            jobs.append(
                {
                    "dataset": dataset,
                    "name": name,
                    "cohort": "development",
                    "kind": "projection",
                    "seed_name": best["name"],
                    "family": "g34" if family == "g34" else f"{family}_a1",
                }
            )
    save_json(DEST / "refinement_jobs.json", jobs)
    print(
        json.dumps(
            {
                "refinement_jobs": len(jobs),
                "seeds": [
                    {"dataset": row["dataset"], "seed": row["seed_name"]}
                    for row in jobs
                ],
            }
        ),
        flush=True,
    )


def freeze():
    rows = development()
    choices = {}
    order = itemgetter("ce", "codes")
    for dataset in ("omniglot", "mnist"):
        subset = [row for row in rows if row["dataset"] == dataset]
        dense = min((row for row in subset if row["family"] == "dense"), key=order)
        g34 = min((row for row in subset if row["family"] == "g34"), key=order)
        new = min(
            (row for row in subset if row["family"].startswith("asrr")), key=order
        )
        choices[dataset] = {
            "dense": dense,
            "g34": g34,
            "new": new,
            "hybrid": min((dense, g34, new), key=order),
            "legacy": min(
                (row for row in subset if row["family"] == "legacy_srr"), key=order
            ),
            "msr": min(
                (row for row in subset if row["family"] == "legacy_msr"), key=order
            ),
        }
    save_json(
        DEST / "choices_frozen.json",
        {"choices": choices, "development": rows, "test_generated": False},
    )
    jobs = []
    for dataset, selected in choices.items():
        folder = ROOT / dataset
        if dataset == "omniglot":
            panels, provenance = omniglot_episodes("test", 256, 3511158801)
        else:
            panels, provenance = mnist_episodes("test", 256, 3511158802)
        (folder / "private").mkdir()
        np.savez_compressed(
            folder / "private/query_answers.npz", y=panels.pop("query_y")
        )
        np.savez_compressed(folder / "panels/test_inputs.npz", **panels)
        save_json(folder / "private/test_provenance.json", provenance)
        mapping = {family: row["name"] for family, row in selected.items()}
        mapping["dense6"] = "dense"
        mapping["dense24"] = "dense24"
        seed = selected["new"]
        with np.load(folder / f"models/{seed['name']}.npz") as data:
            index = data["index"]
        rng = np.random.default_rng(3511158820)
        if "input" in seed["family"]:
            randomized = np.abs(index)[rng.permutation(len(index))] * np.sign(index)
        else:
            randomized = np.empty_like(index)
            for neuron in range(index.shape[1]):
                randomized[:, neuron] = np.abs(
                    index[rng.permutation(len(index)), neuron]
                ) * np.sign(index[:, neuron])
        add_model(
            dataset,
            "randomized_partition",
            "randomized",
            index=randomized,
            seed_name=seed["name"],
        )
        mapping["randomized"] = "randomized_partition"
        for name in dict.fromkeys(
            value for value in mapping.values() if value != "dense24"
        ):
            jobs.append({"dataset": dataset, "name": name, "cohort": "test"})
        for pool, numerical_seed in enumerate(EXTRA_SEEDS, 1):
            jobs.append(
                {
                    "dataset": dataset,
                    "name": f"dense_seed{pool}",
                    "cohort": "test",
                    "numerical_seed": numerical_seed,
                }
            )
        save_json(
            DEST / f"{dataset}_test_registered.json",
            {
                "models": mapping,
                "choices_sha256": digest(DEST / "choices_frozen.json"),
                "inputs_sha256": digest(folder / "panels/test_inputs.npz"),
                "answers_sha256": digest(folder / "private/query_answers.npz"),
                "new_model_sha256": selected["new"]["model_sha256"],
                "query_answers_opened": False,
            },
        )
    save_json(DEST / "test_jobs.json", jobs)
    print(
        json.dumps(
            {
                "choices": {
                    dataset: {key: value["name"] for key, value in selected.items()}
                    for dataset, selected in choices.items()
                },
                "test_jobs": len(jobs),
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "queue", "worker", "refine", "freeze")
    )
    parser.add_argument(
        "--queue",
        choices=("initial", "refinement", "test", "controls"),
        default="initial",
    )
    parser.add_argument("--job", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.device)
    elif args.action == "queue":
        queue(args.queue)
    elif args.action == "worker":
        jobs = json.loads((DEST / f"{args.queue}_jobs.json").read_text())
        worker(jobs[args.job], args.device)
    else:
        globals()[args.action]()


if __name__ == "__main__":
    main()
