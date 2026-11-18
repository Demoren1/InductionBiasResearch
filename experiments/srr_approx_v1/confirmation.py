"""One fixed additional cohort: frozen candidates and paired one-SE selection."""

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.special import logsumexp

from experiments.omniglot_prior_v1.run import episodes as omniglot_episodes
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from experiments.srr_real_v1.predictive import mnist_episodes
from experiments.srr_real_v1.predictive import worker as fit_worker

from .real import DEST as PARENT_DEST
from .real import EXTRA_SEEDS, combine_dense
from .real import ROOT as PARENT_ROOT
from .report import half_interval, write_csv

ROOT = PARENT_ROOT / "confirmation"
DEST = PARENT_DEST / "confirmation"


def choose_simple(dataset, candidates):
    """Minimum m within one paired standard error of the development best."""
    truth = np.load(PARENT_ROOT / dataset / "panels/development.npz")["query_y"]
    losses = {}
    for row in candidates:
        prediction = np.load(
            PARENT_ROOT / dataset / f"fits/development/{row['name']}.npz"
        )["prediction"]
        losses[row["name"]] = (
            logsumexp(prediction, -1) - (prediction * truth).sum(-1)
        ).mean(-1)
    best = min(
        candidates, key=lambda row: (float(losses[row["name"]].mean()), row["codes"])
    )
    eligible, records = [], []
    for row in candidates:
        difference = losses[row["name"]] - losses[best["name"]]
        mean = float(difference.mean())
        se = float(difference.std(ddof=1) / np.sqrt(len(difference)))
        accepted = mean <= se + 1e-12
        records.append(
            {
                "name": row["name"],
                "codes": row["codes"],
                "mean_difference": mean,
                "paired_standard_error": se,
                "eligible": accepted,
            }
        )
        if accepted:
            eligible.append(row)
    selected = min(
        eligible, key=lambda row: (row["codes"], float(losses[row["name"]].mean()))
    )
    return selected, {
        "minimum_ce_candidate": best["name"],
        "selected": selected["name"],
        "candidates": records,
    }


def prepare():
    ROOT.mkdir(parents=True, exist_ok=False)
    DEST.mkdir(parents=True, exist_ok=False)
    parent = json.loads((PARENT_DEST / "choices_frozen.json").read_text())
    save_json(
        DEST / "registered.json",
        {
            "scope": "One additional fixed cohort, no candidate retraining or budget retuning; original policies unchanged. New paired one-SE policy uses development only.",
            "main_test_previously_scored": True,
            "new_rule": "Minimum m with mean paired development CE difference <= one paired standard error from development best; tie by CE",
            "same_rule_for_g34": True,
            "code_sha256": digest(__file__),
            "theory_sha256": digest(
                "paper/theory/APPROXIMATE_SRR_THEORY_2026-11-15.md"
            ),
            "episodes_per_dataset": 512,
            "seeds": {"omniglot": 3511158901, "mnist": 3511158902},
            "query_answers_generated": False,
        },
    )
    policies, records = {}, {}
    for dataset, old in parent["choices"].items():
        candidates = [
            row
            for row in parent["development"]
            if row["dataset"] == dataset
            and (row["family"].startswith("asrr") or row["family"] in ("g34", "dense"))
        ]
        simple, trace = choose_simple(dataset, candidates)
        g34, gtrace = choose_simple(
            dataset, [row for row in candidates if row["family"] in ("g34", "dense")]
        )
        policies[dataset] = {
            "new": old["new"]["name"],
            "hybrid": old["hybrid"]["name"],
            "g34": old["g34"]["name"],
            "dense": old["dense"]["name"],
            "dense6": "dense",
            "dense24": "dense24",
            "simple_hybrid": simple["name"],
            "simple_g34": g34["name"],
        }
        records[dataset] = {"hybrid": trace, "g34": gtrace}
    save_json(
        DEST / "choices_frozen.json",
        {
            "policies": policies,
            "one_se": records,
            "parent_choices_sha256": digest(PARENT_DEST / "choices_frozen.json"),
            "test_generated": False,
        },
    )
    jobs = []
    for dataset, mapping in policies.items():
        folder = ROOT / dataset
        for directory in ("models", "panels", "private", "logs", "fits/test"):
            (folder / directory).mkdir(parents=True)
        for name in set(mapping.values()):
            path = PARENT_ROOT / dataset / f"models/{name}.npz"
            if path.exists():
                shutil.copy2(path, folder / f"models/{name}.npz")
        panels, provenance = (
            omniglot_episodes("test", 512, 3511158901)
            if dataset == "omniglot"
            else mnist_episodes("test", 512, 3511158902)
        )
        np.savez_compressed(
            folder / "private/query_answers.npz", y=panels.pop("query_y")
        )
        np.savez_compressed(folder / "panels/test_inputs.npz", **panels)
        save_json(folder / "private/provenance.json", provenance)
        for name in dict.fromkeys(
            name for name in mapping.values() if name != "dense24"
        ):
            jobs.append({"dataset": dataset, "name": name})
        for pool, seed in enumerate(EXTRA_SEEDS, 1):
            jobs.append(
                {
                    "dataset": dataset,
                    "name": f"dense_seed{pool}",
                    "numerical_seed": seed,
                }
            )
    save_json(DEST / "jobs.json", jobs)
    print(json.dumps({"policies": policies, "jobs": len(jobs)}), flush=True)


def gpu_worker(item):
    device, jobs = item
    rows = []
    for number, job in jobs:
        dataset, name = job["dataset"], job["name"]
        log = ROOT / dataset / f"logs/{name}.log"
        command = [
            sys.executable,
            "-m",
            "experiments.srr_approx_v1.confirmation",
            "worker",
            "--job",
            str(number),
            "--device",
            f"cuda:{device}",
        ]
        with log.open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=stream, check=False)
        row = {**job, "gpu": device, "exit_code": result.returncode, "log": str(log)}
        rows.append(row)
        print(json.dumps(row), flush=True)
        if result.returncode:
            break
    return rows


def run():
    jobs = list(enumerate(json.loads((DEST / "jobs.json").read_text())))
    with ThreadPoolExecutor(max_workers=8) as executor:
        groups = list(
            executor.map(gpu_worker, [(device, jobs[device::8]) for device in range(8)])
        )
    rows = [row for group in groups for row in group]
    save_json(DEST / "queue_complete.json", rows)
    if len(rows) != len(jobs) or any(row["exit_code"] for row in rows):
        raise RuntimeError("Confirmation fits incomplete")


def score():
    if (DEST / "results.json").exists():
        raise RuntimeError("Confirmation query answers already scored")
    policies = json.loads((DEST / "choices_frozen.json").read_text())["policies"]
    frozen = {}
    for dataset, mapping in policies.items():
        combine_dense(dataset, "test", study_root=ROOT)
        for method, name in mapping.items():
            path = ROOT / dataset / f"fits/test/{name}.npz"
            record = json.loads(path.with_suffix(".complete.json").read_text())
            if digest(path) != record["prediction_sha256"]:
                raise RuntimeError("Changed completed prediction")
            frozen[f"{dataset}/{method}"] = digest(path)
    save_json(
        DEST / "predictions_frozen.json",
        {
            "predictions": frozen,
            "choices_sha256": digest(DEST / "choices_frozen.json"),
            "query_answers_opened": False,
        },
    )
    rows, pairs = [], []
    for dataset, mapping in policies.items():
        truth = np.load(ROOT / dataset / "private/query_answers.npz")["y"]
        outcomes = {}
        for method, name in mapping.items():
            prediction = np.load(ROOT / dataset / f"fits/test/{name}.npz")["prediction"]
            accuracy = (prediction.argmax(-1) == truth.argmax(-1)).mean(1)
            ce = (logsumexp(prediction, -1) - (prediction * truth).sum(-1)).mean(1)
            outcomes[method] = accuracy, ce
            path = ROOT / dataset / f"models/{name}.npz"
            codes = (
                int(np.abs(np.load(path)["index"]).max()) if path.exists() else 12544
            )
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "name": name,
                    "codes": codes,
                    "episodes": len(accuracy),
                    "accuracy": float(accuracy.mean()),
                    "accuracy_half_ci": half_interval(accuracy),
                    "ce": float(ce.mean()),
                    "ce_half_ci": half_interval(ce),
                }
            )
        for first, second in (
            ("new", "dense"),
            ("new", "g34"),
            ("simple_hybrid", "dense"),
            ("simple_hybrid", "g34"),
            ("simple_hybrid", "simple_g34"),
            ("hybrid", "dense"),
            ("hybrid", "g34"),
        ):
            accuracy = outcomes[first][0] - outcomes[second][0]
            ce = outcomes[first][1] - outcomes[second][1]
            pairs.append(
                {
                    "dataset": dataset,
                    "first": first,
                    "second": second,
                    "accuracy_difference": float(accuracy.mean()),
                    "accuracy_half_ci": half_interval(accuracy),
                    "ce_difference": float(ce.mean()),
                    "ce_half_ci": half_interval(ce),
                }
            )
        np.savez_compressed(
            DEST / f"{dataset}_episode_scores.npz",
            **{
                f"{method}_{key}": value[column]
                for method, value in outcomes.items()
                for column, key in enumerate(("accuracy", "ce"))
            },
        )
    save_json(
        DEST / "results.json",
        {"comparison": rows, "paired": pairs, "policies": policies},
    )
    write_csv(DEST / "comparison.csv", rows)
    write_csv(DEST / "paired_differences.csv", pairs)
    print(json.dumps({"comparison": rows, "paired": pairs}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "worker", "score"))
    parser.add_argument("--job", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.action == "worker":
        job = json.loads((DEST / "jobs.json").read_text())[args.job]
        fit_worker(
            job["dataset"],
            job["name"],
            "test",
            args.device,
            study_root=ROOT,
            numerical_seed=job.get("numerical_seed"),
        )
    else:
        globals()[args.action]()
