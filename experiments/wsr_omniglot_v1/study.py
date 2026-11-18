"""Compare fixed WSR and aSRR groups with Dense on the same episodes."""

import argparse
import shutil

import numpy as np

from experiments.asrr_anchor_init_v1.omniglot_transfer import (
    DEVELOPMENT,
    fit_record,
    make_job,
    prepare_targets,
    score_episodes,
)
from experiments.asrr_anchor_init_v1.report import interval
from experiments.asrr_anchor_init_v1.transfer import public_panel
from experiments.asrr_tuning_v1.study import run as run_fits
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

from .source import (
    BANKS,
    BASELINE,
    CODES,
    OUTPUT,
    REGISTRATION,
    ROOT,
    read_json,
)


SETTING = "omniglot10"
STUDY = ROOT / "study" / SETTING
PANEL_FOLDER = STUDY / "panels"
CHOICES = OUTPUT / "choices_frozen.json"


def prepare_development():
    path = OUTPUT / "development_jobs.json"
    if path.exists():
        raise FileExistsError(path)
    config = read_json(REGISTRATION)
    candidates = read_json(OUTPUT / "source_fits.json")["candidates"]
    if [row["source_task"] for row in candidates] != config["source_tasks"]:
        raise ValueError("Source candidate tasks differ from registration")
    if any(row["codes"] != CODES for row in candidates):
        raise ValueError("A WSR candidate has the wrong code count")
    panel = public_panel(
        SETTING, DEVELOPMENT, directory=PANEL_FOLDER, **config["development"]
    )
    asrr = config["asrr"]
    jobs = [
        make_job(row["name"], DEVELOPMENT, row["model_sha256"], panel, asrr)
        for row in (*candidates, asrr, {"name": "dense", "model_sha256": None})
    ]
    save_json(
        OUTPUT / "development_panel.json", {**panel, **config["development"]}
    )
    save_json(path, jobs)


def freeze():
    if CHOICES.exists():
        raise FileExistsError(CHOICES)
    asrr = read_json(REGISTRATION)["asrr"]
    jobs_path = OUTPUT / "development_jobs.json"
    results = {}
    for job in read_json(jobs_path):
        _, record = fit_record(job, asrr, study=STUDY)
        results[job["name"]] = {**job, **record["metrics"][asrr["key"]]}
    sources = read_json(OUTPUT / "source_fits.json")["candidates"]
    ranked = sorted(
        [
            {**results[row["name"]], "source_task": row["source_task"]}
            for row in sources
        ],
        key=lambda row: (-row["accuracy"], row["ce"], row["source_task"]),
    )
    policies = {
        "wsr": ranked[0],
        "asrr": results[BASELINE],
        "dense": results["dense"],
    }
    models = OUTPUT / "frozen_indexators"
    models.mkdir()
    for row in (policies["wsr"], policies["asrr"]):
        model = BANKS / f"{row['name']}.npz"
        if digest(model) != row["model_sha256"]:
            raise ValueError(f"Changed selected model: {model}")
        shutil.copyfile(model, models / model.name)
    save_json(
        CHOICES,
        {
            "policies": policies,
            "development_wsr_candidates": ranked,
            "development_jobs_sha256": digest(jobs_path),
            "selection": "Development accuracy, then CE, then source task ID",
            "numeric_rule": asrr["rule"],
        },
    )
    print(f"Selected WSR source task {ranked[0]['source_task']}")


def summarize(rows):
    differences = {}
    for first, second in (
        ("wsr", "asrr"),
        ("wsr", "dense"),
        ("asrr", "dense"),
    ):
        delta = rows[first] - rows[second]
        differences[f"{first}_minus_{second}"] = {
            "mean": delta.mean(axis=0).tolist(),
            "ci95_halfwidth": [interval(metric) for metric in delta.T],
        }
    return {
        "episodes": len(rows["wsr"]),
        "metrics": {
            name: values.mean(axis=0).tolist() for name, values in rows.items()
        },
        "differences": differences,
    }


def score():
    path = OUTPUT / "results.json"
    if path.exists():
        raise FileExistsError(path)
    choices = read_json(CHOICES)["policies"]
    asrr = read_json(REGISTRATION)["asrr"]
    groups = score_episodes(OUTPUT, STUDY, asrr)
    pooled = summarize(
        {
            method: np.concatenate([rows[method] for rows in groups.values()])
            for method in choices
        }
    )
    save_json(
        path,
        {
            "columns": ["accuracy", "cross_entropy"],
            "code_counts": {"wsr": CODES, "asrr": CODES, "dense": 784 * 16},
            "wsr_source_task": choices["wsr"]["source_task"],
            "development": {
                method: [row["accuracy"], row["ce"]]
                for method, row in choices.items()
            },
            "cohorts": {
                cohort: summarize(rows) for cohort, rows in groups.items()
            },
            "pooled": pooled,
            "conditional_on": "fixed held-out Omniglot character pool",
        },
    )
    print(pooled)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "prepare-development",
            "development",
            "freeze",
            "prepare-targets",
            "target",
            "score",
        ),
    )
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    args = parser.parse_args()
    if args.action == "prepare-targets":
        prepare_targets(
            output=OUTPUT,
            directory=PANEL_FOLDER,
            **read_json(REGISTRATION)["targets"],
        )
    elif args.action in ("development", "target"):
        run_fits(args.action, args.gpus, root=ROOT, output=OUTPUT)
    else:
        {
            "prepare-development": prepare_development,
            "freeze": freeze,
            "score": score,
        }[args.action]()
