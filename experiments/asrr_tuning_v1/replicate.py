"""Re-evaluate the original frozen successful policies on fresh episodes."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import t

from experiments.omniglot_prior_v1.run import episodes as omniglot_episodes
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)
from experiments.srr_real_v1.predictive import mnist_episodes, worker

ROOT = Path("artifacts/local/asrr_tuning_2026-11-17")
OUTPUT = Path("artifacts/asrr_tuning_2026-11-17")
OLD = Path("artifacts/local/srr_approx_real_2026-11-15")
OLD_OUTPUT = Path("artifacts/srr_approx_real_2026-11-15")
EXTRA_SEEDS = (3511158811, 3511158812, 3511158813)


def prepare(datasets):
    ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    registration = OUTPUT / "registration.json"
    if not registration.exists():
        save_json(
            registration,
            {
                "protocol_sha256": digest(
                    Path(__file__).with_name("PROTOCOL.md")
                ),
                "authorization": "User requested broad tuning and all GPUs",
                "previous_study": "artifacts/paper_asrr_2026-11-17",
            },
        )
    parent = json.loads((OLD_OUTPUT / "choices_frozen.json").read_text())[
        "choices"
    ]
    for dataset in datasets:
        root = ROOT / "replication" / dataset
        out = OUTPUT / "replication" / dataset
        root.mkdir(parents=True)
        out.mkdir(parents=True)
        for name in ("models", "panels", "private", "logs", "fits/test"):
            (root / name).mkdir(parents=True)
        policies = {
            method: parent[dataset][key]["name"]
            for method, key in (
                ("asrr", "new"),
                ("g34", "g34"),
                ("dense24", "dense"),
            )
        }
        hashes = {}
        for name in policies.values():
            p = OLD / dataset / f"models/{name}.npz"
            if p.exists():
                shutil.copy2(p, root / f"models/{name}.npz")
                hashes[name] = digest(p)
        save_json(
            out / "choices_frozen.json",
            {
                "policies": policies,
                "models_sha256": hashes,
                "original_choices_sha256": digest(
                    OLD_OUTPUT / "choices_frozen.json"
                ),
                "query_answers_generated": False,
            },
        )
        panel, provenance = (
            mnist_episodes("test", 512, 7611171101)
            if dataset == "mnist"
            else omniglot_episodes("test", 512, 7611171102)
        )
        np.savez_compressed(
            root / "private/query_answers.npz", y=panel.pop("query_y")
        )
        np.savez_compressed(root / "panels/test_inputs.npz", **panel)
        save_json(root / "private/provenance.json", provenance)
        jobs = [
            {"dataset": dataset, "name": name}
            for name in dict.fromkeys(
                [
                    policies["asrr"],
                    policies["g34"],
                    "dense",
                    "dense_seed1",
                    "dense_seed2",
                    "dense_seed3",
                ]
            )
        ]
        for pool, seed in enumerate(EXTRA_SEEDS, 1):
            next(j for j in jobs if j["name"] == f"dense_seed{pool}")[
                "seed"
            ] = seed
        save_json(out / "jobs.json", jobs)


def queue(datasets):
    jobs = [
        (dataset, number, job)
        for dataset in datasets
        for number, job in enumerate(
            json.loads(
                (OUTPUT / "replication" / dataset / "jobs.json").read_text()
            )
        )
    ]

    def gpu_worker(gpu):
        for dataset, number, job in jobs[gpu::8]:
            out = OUTPUT / "replication" / dataset
            receipt = out / f"{job['name']}.job.json"
            if receipt.exists():
                if json.loads(receipt.read_text())["exit_code"] == 0:
                    continue
                raise RuntimeError(f"Inspect failed job {receipt}")
            command = [
                sys.executable,
                "-u",
                "-m",
                "experiments.asrr_tuning_v1.replicate",
                "worker",
                "--datasets",
                dataset,
                "--job",
                str(number),
                "--device",
                f"cuda:{gpu}",
            ]
            with (
                ROOT / "replication" / dataset / f"logs/{job['name']}.log"
            ).open("w") as stream:
                result = subprocess.run(
                    command,
                    stdout=stream,
                    stderr=stream,
                    env={
                        **os.environ,
                        "OMP_NUM_THREADS": "2",
                        "OPENBLAS_NUM_THREADS": "2",
                        "MKL_NUM_THREADS": "2",
                    },
                )
            save_json(
                receipt, {**job, "gpu": gpu, "exit_code": result.returncode}
            )
            print(
                json.dumps({**job, "exit_code": result.returncode}), flush=True
            )
            if result.returncode:
                raise RuntimeError(f"Failed replication job {job}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(gpu_worker, range(8)))


def score(dataset):
    root, out = (
        ROOT / "replication" / dataset,
        OUTPUT / "replication" / dataset,
    )
    if (out / "results.json").exists():
        raise RuntimeError("Replication already scored")
    policies = json.loads((out / "choices_frozen.json").read_text())[
        "policies"
    ]
    names = [
        policies["asrr"],
        policies["g34"],
        "dense",
        "dense_seed1",
        "dense_seed2",
        "dense_seed3",
    ]
    save_json(
        out / "predictions_frozen.json",
        {name: digest(root / f"fits/test/{name}.npz") for name in names},
    )
    with np.load(root / "panels/test_inputs.npz") as data:
        valid_y = data["valid_y"]
    predictions = {}
    pools, errors = [], []
    for name in names:
        with np.load(root / f"fits/test/{name}.npz") as data:
            if name.startswith("dense"):
                pools.append(data["prediction"])
                v = data["valid_prediction"]
                errors.append(
                    (logsumexp(v, -1) - (v * valid_y).sum(-1)).mean(-1)
                )
            else:
                predictions[name] = data["prediction"]
    predictions[policies["dense24"]] = np.stack(pools)[
        np.argmin(errors, axis=0), np.arange(len(valid_y))
    ]
    truth = np.load(root / "private/query_answers.npz")["y"]
    arrays, records = {"truth": truth}, []
    outcomes = {}
    for method, name in policies.items():
        value = predictions[name]
        accuracy = (value.argmax(-1) == truth.argmax(-1)).mean(-1)
        ce = (logsumexp(value, -1) - (value * truth).sum(-1)).mean(-1)
        outcomes[method] = accuracy
        half = float(
            t.ppf(0.975, len(accuracy) - 1)
            * accuracy.std(ddof=1)
            / np.sqrt(len(accuracy))
        )
        records.append(
            {
                "method": method,
                "name": name,
                "accuracy": float(accuracy.mean()),
                "accuracy_half_ci": half,
                "ce": float(ce.mean()),
                "episodes": len(accuracy),
            }
        )
        arrays[f"{method}_prediction"] = value
    pairs = {}
    for other in ("g34", "dense24"):
        delta = outcomes["asrr"] - outcomes[other]
        pairs[other] = {
            "difference": float(delta.mean()),
            "half_ci": float(
                t.ppf(0.975, len(delta) - 1)
                * delta.std(ddof=1)
                / np.sqrt(len(delta))
            ),
        }
    np.savez_compressed(out / "scoring.npz", **arrays)
    save_json(out / "results.json", {"methods": records, "asrr_minus": pairs})
    print(
        json.dumps(
            {"dataset": dataset, "methods": records, "asrr_minus": pairs}
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=("prepare", "queue", "worker", "score")
    )
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--job", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.datasets)
    elif args.phase == "queue":
        queue(args.datasets)
    elif args.phase == "score":
        for dataset in args.datasets:
            score(dataset)
    else:
        dataset = args.datasets[0]
        job = json.loads(
            (OUTPUT / "replication" / dataset / "jobs.json").read_text()
        )[args.job]
        worker(
            dataset,
            job["name"],
            "test",
            args.device,
            study_root=ROOT / "replication",
            numerical_seed=job.get("seed"),
        )
