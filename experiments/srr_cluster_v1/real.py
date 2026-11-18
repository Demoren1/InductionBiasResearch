"""Matched real-data development followed by one fresh frozen test cohort."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp

from experiments.omniglot_prior_v1.run import episodes as omni_episodes
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from experiments.srr_approx_v1.projection import project_chunked
from experiments.srr_approx_v1.real import BUDGETS, EXTRA_SEEDS
from experiments.srr_approx_v1.real import DEST as OLD_DEST
from experiments.srr_approx_v1.real import ROOT as OLD_ROOT
from experiments.srr_real_v1.predictive import mnist_episodes
from experiments.srr_real_v1.predictive import worker as numerical_worker

from .clustering import METHODS, build_bank
from .synthetic import DEST
from .synthetic import ROOT as STUDY_ROOT

ROOT = STUDY_ROOT / "real"


def prepare():
    if not (DEST / "synthetic_results.json").exists():
        raise RuntimeError("Finish synthetic comparison before real tuning")
    ROOT.mkdir(exist_ok=False)
    previous = json.loads((OLD_DEST / "choices_frozen.json").read_text())["development"]
    baseline, jobs = [], []
    for dataset in ("omniglot", "mnist"):
        folder = ROOT / dataset
        for directory in (
            "models",
            "model_records",
            "panels",
            "logs",
            "fits/development",
            "construction",
        ):
            (folder / directory).mkdir(parents=True)
        shutil.copy2(
            OLD_ROOT / dataset / "source_profiles.npz", folder / "source_profiles.npz"
        )
        shutil.copy2(
            OLD_ROOT / dataset / "panels/development.npz",
            folder / "panels/development.npz",
        )
        for old in previous:
            if old["dataset"] != dataset:
                continue
            family = old["family"]
            if family.startswith("asrr"):
                group = "ward_previous"
            elif family in ("g34", "dense"):
                group = family
            else:
                continue
            name = old["name"]
            model = OLD_ROOT / dataset / f"models/{name}.npz"
            if model.exists():
                shutil.copy2(model, folder / f"models/{name}.npz")
            shutil.copy2(
                OLD_ROOT / dataset / f"fits/development/{name}.npz",
                folder / f"fits/development/{name}.npz",
            )
            row = {
                **old,
                "method": group,
                "baseline": True,
                "stage": "previous",
                "reused_development": True,
            }
            save_json(folder / f"model_records/{name}.json", row)
            baseline.append(row)
        jobs.extend(
            {"dataset": dataset, "method": method, "kind": "construct"}
            for method in METHODS
        )
    save_json(
        DEST / "real_registered.json",
        {
            "synthetic_sha256": digest(DEST / "synthetic_results.json"),
            "protocol_sha256": digest(Path(__file__).with_name("PROTOCOL.md")),
            "source_sha256": digest(__file__),
            "baseline_candidates": baseline,
            "test_generated": False,
        },
    )
    save_json(DEST / "construct_jobs.json", jobs)


def worker(job, device):
    dataset = job["dataset"]
    folder = ROOT / dataset
    if job["kind"] == "construct":
        torch.set_num_threads(2)
        weights = np.load(folder / "source_profiles.npz")["consensus"]
        started = time.perf_counter()
        bank, records = build_bank(
            weights, job["method"], BUDGETS, ("local", "input"), device=device
        )
        for name, index in bank.items():
            path = folder / f"models/{name}.npz"
            np.savez_compressed(path, index=index)
            save_json(
                folder / f"model_records/{name}.json",
                {
                    "dataset": dataset,
                    "name": name,
                    "method": job["method"],
                    "stage": "raw",
                    "codes": int(np.abs(index).max()),
                    "scope": "input" if "_input_" in name else "local",
                    "model_sha256": digest(path),
                    "baseline": False,
                },
            )
        save_json(
            folder / f"construction/{job['method']}.json",
            {"seconds": time.perf_counter() - started, "candidates": records},
        )
        return
    if job["kind"] == "projection":
        source = np.load(folder / "source_profiles.npz")
        initial = np.load(folder / f"models/{job['seed']}.npz")["index"]
        count = int(np.abs(initial).max())
        index, history = project_chunked(
            initial,
            source["consensus"],
            source["uncertainty"],
            steps=12 if count <= 1024 else 4,
            device=device,
        )
        name = job["name"]
        path = folder / f"models/{name}.npz"
        np.savez_compressed(path, index=index)
        save_json(
            folder / f"model_records/{name}.json",
            {
                "dataset": dataset,
                "name": name,
                "method": job["method"],
                "stage": "a1",
                "codes": count,
                "baseline": False,
                "model_sha256": digest(path),
                "projection": history,
                "seed": job["seed"],
            },
        )
    numerical_worker(
        dataset,
        job["name"],
        job["cohort"],
        device,
        study_root=ROOT,
        numerical_seed=job.get("numerical_seed"),
    )


def gpu_worker(item):
    device, indexed = item
    rows = []
    for number, job, queue_name in indexed:
        label = job.get("name", job.get("method"))
        log = ROOT / job["dataset"] / f"logs/{queue_name}_{label}.log"
        command = [
            sys.executable,
            "-m",
            "experiments.srr_cluster_v1.real",
            "worker",
            "--queue",
            queue_name,
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
            "seconds": time.perf_counter() - started,
            "exit_code": result.returncode,
            "log": str(log),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        if result.returncode:
            break
    return rows


def queue(name):
    jobs = json.loads((DEST / f"{name}_jobs.json").read_text())
    indexed = [(number, row, name) for number, row in enumerate(jobs)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        groups = list(
            pool.map(gpu_worker, [(device, indexed[device::8]) for device in range(8)])
        )
    rows = [row for group in groups for row in group]
    save_json(DEST / f"{name}_queue_complete.json", rows)
    if len(rows) != len(jobs) or any(row["exit_code"] for row in rows):
        raise RuntimeError(f"Incomplete {name} queue; inspect saved logs")


def fit_development(queue_name="development"):
    jobs = []
    for dataset in ("omniglot", "mnist"):
        folder = ROOT / dataset
        cached = {}
        for path in (folder / "models").glob("*.npz"):
            fit = folder / f"fits/development/{path.stem}.npz"
            if fit.exists():
                cached[np.load(path)["index"].tobytes()] = fit
        records = [
            json.loads(path.read_text())
            for path in sorted((folder / "model_records").glob("*.json"))
        ]
        scheduled = {}
        for row in records:
            if row["baseline"]:
                continue
            name = row["name"]
            fit = folder / f"fits/development/{name}.npz"
            if fit.exists():
                continue
            blob = np.load(folder / f"models/{name}.npz")["index"].tobytes()
            if blob in cached:
                shutil.copy2(cached[blob], fit)
            elif blob in scheduled:
                row["fit_alias"] = scheduled[blob]
                save_json(folder / f"model_records/{name}.json", row)
            else:
                scheduled[blob] = name
                jobs.append(
                    {
                        "dataset": dataset,
                        "kind": "fit",
                        "name": name,
                        "cohort": "development",
                    }
                )
    save_json(DEST / f"{queue_name}_jobs.json", jobs)
    print(json.dumps({"new_development_fits": len(jobs)}), flush=True)


def development():
    rows, losses = [], {}
    for dataset in ("omniglot", "mnist"):
        folder = ROOT / dataset
        truth = np.load(folder / "panels/development.npz")["query_y"]
        for path in sorted((folder / "model_records").glob("*.json")):
            record = json.loads(path.read_text())
            fit = folder / f"fits/development/{record['name']}.npz"
            if not fit.exists() and "fit_alias" in record:
                provider = folder / f"fits/development/{record['fit_alias']}.npz"
                if provider.exists():
                    shutil.copy2(provider, fit)
            if not fit.exists():
                continue
            prediction = np.load(fit)["prediction"]
            loss = (logsumexp(prediction, -1) - (prediction * truth).sum(-1)).mean(-1)
            row = {
                **record,
                "ce": float(loss.mean()),
                "accuracy": float((prediction.argmax(-1) == truth.argmax(-1)).mean()),
            }
            rows.append(row)
            losses[(dataset, row["name"])] = loss
    return rows, losses


def refine(queue_name="refine"):
    rows, _ = development()
    jobs = []
    for dataset in ("omniglot", "mnist"):
        for method in METHODS:
            candidates = [
                row
                for row in rows
                if row["dataset"] == dataset
                and row["method"] == method
                and row["stage"] == "raw"
            ]
            if len(candidates) != 2 * (len(BUDGETS) - 1):
                continue
            best = min(candidates, key=lambda row: row["ce"])
            name = f"a1_{best['name']}"
            if (ROOT / dataset / f"fits/development/{name}.npz").exists():
                continue
            jobs.append(
                {
                    "dataset": dataset,
                    "kind": "projection",
                    "method": method,
                    "name": name,
                    "seed": best["name"],
                    "cohort": "development",
                }
            )
    save_json(DEST / f"{queue_name}_development.json", rows)
    save_json(DEST / f"{queue_name}_jobs.json", jobs)
    print(json.dumps({"new_refinements": len(jobs)}), flush=True)


def select(candidates, losses, simple=False):
    best = min(candidates, key=lambda row: (row["ce"], row["codes"]))
    if not simple:
        return best
    eligible = []
    reference = losses[(best["dataset"], best["name"])]
    for row in candidates:
        difference = losses[(row["dataset"], row["name"])] - reference
        if (
            difference.mean()
            <= difference.std(ddof=1) / np.sqrt(len(difference)) + 1e-12
        ):
            eligible.append(row)
    return min(eligible, key=lambda row: (row["codes"], row["ce"]))


def freeze():
    if (DEST / "real_choices_frozen.json").exists():
        raise RuntimeError("Real policies already frozen")
    rows, losses = development()
    policies, selected = {}, {}
    for dataset in ("omniglot", "mnist"):
        candidates = [row for row in rows if row["dataset"] == dataset]
        dense = [row for row in candidates if row["method"] == "dense"]
        local = {}
        for method in METHODS:
            raw = [
                row
                for row in candidates
                if row["method"] == method and row["stage"] == "raw"
            ] + dense
            if len(raw) - len(dense) != 2 * (len(BUDGETS) - 1):
                raise RuntimeError(f"Incomplete development grid: {dataset}/{method}")
            if not any(
                row["dataset"] == dataset
                and row["method"] == method
                and row["stage"] == "a1"
                for row in rows
            ):
                raise RuntimeError(f"Incomplete Algorithm1 fit: {dataset}/{method}")
            final = [row for row in candidates if row["method"] == method] + dense
            local[f"raw_{method}"] = select(raw, losses)
            local[method] = select(final, losses)
        for method in ("ward_previous", "g34", "dense"):
            group = [row for row in candidates if row["method"] == method] + dense
            local[method] = select(group, losses)
            if method != "dense":
                local[f"simple_{method}"] = select(group, losses, simple=True)
        local["union"] = select(candidates, losses)
        local["simple_union"] = select(candidates, losses, simple=True)
        local["dense6"] = next(row for row in dense if row["name"] == "dense")
        local["g34_compressed"] = select(
            [row for row in candidates if row["method"] == "g34"], losses
        )
        policies[dataset] = {method: row["name"] for method, row in local.items()}
        selected[dataset] = local
    save_json(
        DEST / "real_choices_frozen.json",
        {
            "policies": policies,
            "selected": selected,
            "development": rows,
            "test_generated": False,
        },
    )
    jobs = []
    for dataset, mapping in policies.items():
        folder = ROOT / dataset
        (folder / "private").mkdir()
        (folder / "fits/test").mkdir()
        panels, provenance = (
            omni_episodes("test", 512, 3511159201)
            if dataset == "omniglot"
            else mnist_episodes("test", 512, 3511159202)
        )
        np.savez_compressed(
            folder / "private/query_answers.npz", y=panels.pop("query_y")
        )
        np.savez_compressed(folder / "panels/test_inputs.npz", **panels)
        save_json(folder / "private/provenance.json", provenance)
        jobs.extend(
            {"dataset": dataset, "kind": "fit", "name": name, "cohort": "test"}
            for name in dict.fromkeys(mapping.values())
            if name != "dense24"
        )
        if "dense" not in mapping.values():
            jobs.append(
                {"dataset": dataset, "kind": "fit", "name": "dense", "cohort": "test"}
            )
        jobs.extend(
            {
                "dataset": dataset,
                "kind": "fit",
                "name": f"dense_seed{number}",
                "cohort": "test",
                "numerical_seed": seed,
            }
            for number, seed in enumerate(EXTRA_SEEDS, 1)
        )
    save_json(DEST / "test_jobs.json", jobs)
    print(json.dumps({"policies": policies, "test_jobs": len(jobs)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("prepare", "queue", "development", "refine", "freeze", "worker"),
    )
    parser.add_argument("--queue")
    parser.add_argument("--job", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.stage == "queue":
        queue(args.queue)
    elif args.stage == "worker":
        jobs = json.loads((DEST / f"{args.queue}_jobs.json").read_text())
        worker(jobs[args.job], args.device)
    elif args.stage == "development":
        fit_development(args.queue or "development")
    elif args.stage == "refine":
        refine(args.queue or "refine")
    else:
        {
            "prepare": prepare,
            "freeze": freeze,
        }[args.stage]()


if __name__ == "__main__":
    main()
