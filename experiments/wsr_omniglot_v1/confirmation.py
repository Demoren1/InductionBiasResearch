"""Evaluate the three frozen Omniglot policies on 4096 new paired episodes."""

import argparse
import shutil

import numpy as np

from experiments.asrr_anchor_init_v1.omniglot_transfer import (
    prepare_targets,
    score_episodes,
)
from experiments.asrr_anchor_init_v1.report import interval
from experiments.asrr_tuning_v1.study import run as run_fits
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

from .source import OUTPUT as FIRST_STUDY
from .source import REGISTRATION, read_json
from .study import SETTING, summarize


OUTPUT = FIRST_STUDY.with_name("wsr_omniglot_confirmation_2026-11-18")
ROOT = OUTPUT.parent / "local" / OUTPUT.name
STUDY = ROOT / "study" / SETTING
COUNT = 512
COHORTS = tuple(f"test_confirm_{i:02d}" for i in range(1, 9))
SEEDS = tuple(range(7611186400, 7611186400 + len(COHORTS)))


def prepare():
    first_choices = FIRST_STUDY / "choices_frozen.json"
    original_result = FIRST_STUDY / "results.json"
    choices = read_json(first_choices)
    banks = ROOT / "banks/omniglot/models"
    for method in ("wsr", "asrr"):
        row = choices["policies"][method]
        path = FIRST_STUDY / "frozen_indexators" / f"{row['name']}.npz"
        if digest(path) != row["model_sha256"]:
            raise ValueError(f"Changed frozen indexator: {path}")
    OUTPUT.mkdir(parents=True)
    banks.mkdir(parents=True)
    for method in ("wsr", "asrr"):
        name = choices["policies"][method]["name"]
        shutil.copyfile(
            FIRST_STUDY / "frozen_indexators" / f"{name}.npz",
            banks / f"{name}.npz",
        )
    shutil.copyfile(first_choices, OUTPUT / first_choices.name)
    save_json(
        OUTPUT / "registration.json",
        {
            "first_study": str(FIRST_STUDY),
            "first_result_sha256": digest(original_result),
            "choices_sha256": digest(first_choices),
            "cohorts": dict(zip(COHORTS, SEEDS, strict=True)),
            "episodes_per_cohort": COUNT,
            "primary_contrast": "WSR minus aSRR accuracy",
            "analysis_sample": "Fresh confirmation episodes",
        },
    )
    prepare_targets(
        output=OUTPUT,
        directory=STUDY / "panels",
        count=COUNT,
        seeds=SEEDS,
        cohorts=COHORTS,
    )


def score():
    path = OUTPUT / "results.json"
    if path.exists():
        raise FileExistsError(path)
    registration = read_json(OUTPUT / "registration.json")
    if (
        digest(OUTPUT / "choices_frozen.json")
        != registration["choices_sha256"]
    ):
        raise ValueError("Changed confirmation choices")
    record = read_json(REGISTRATION)["asrr"]
    groups = score_episodes(OUTPUT, STUDY, record)
    if tuple(groups) != COHORTS or any(
        len(row["wsr"]) != COUNT for row in groups.values()
    ):
        raise ValueError("Incomplete confirmation cohorts")
    pooled = {
        method: np.concatenate([rows[method] for rows in groups.values()])
        for method in ("wsr", "asrr", "dense")
    }
    result = summarize(pooled)
    result["accuracy_ci95_halfwidth"] = {
        method: interval(rows[:, 0]) for method, rows in pooled.items()
    }
    save_json(
        path,
        {
            "columns": ["accuracy", "cross_entropy"],
            "cohorts": {
                name: summarize(rows) for name, rows in groups.items()
            },
            "pooled": result,
            "conditional_on": "fixed held-out Omniglot character pool",
        },
    )
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "score"))
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "run":
        run_fits("target", args.gpus, root=ROOT, output=OUTPUT)
    else:
        score()
