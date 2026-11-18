"""Select an Omniglot anchor on development episodes and score frozen fits."""

import argparse
import shutil
from pathlib import Path

import numpy as np

from experiments.asrr_tuning_v1.study import run as run_fits
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

from .omniglot import (
    ANCHORS,
    BANKS,
    BASELINE,
    OUTPUT,
    ROOT,
    SETTING,
    read_json,
)
from .report import interval
from .transfer import COHORTS, public_panel


DEVELOPMENT = "development"
STUDY = ROOT / "study" / SETTING
PANEL_FOLDER = STUDY / "panels"
DEV_COUNT, TARGET_COUNT = 96, 512
DEV_SEED, TARGET_SEEDS = 7611185200, (7611185300, 7611185301)


def make_job(name, phase, model_hash, panel, baseline):
    return {
        "setting": SETTING,
        "name": name,
        "phase": phase,
        "variant": baseline["rule"]["variant"],
        "model_sha256": model_hash,
        "panel_sha256": panel["sha256"],
    }


def prepare_development():
    path = OUTPUT / "development_jobs.json"
    if path.exists():
        raise FileExistsError(path)
    config = read_json(OUTPUT / "registration.json")
    baseline = config["baseline"]
    sources = config["source_tasks"]
    candidates = {BASELINE: baseline["model_sha256"]}
    for anchor in ANCHORS:
        name = f"anchor_{anchor:02d}"
        record = read_json(OUTPUT / "source_fits" / name / "complete.json")
        others = [task for task in sources if task != sources[anchor]]
        if (
            record["anchor_task"] != sources[anchor]
            or record["adapted_tasks"] != others
            or record["models_per_bank"] != len(sources)
        ):
            raise ValueError(f"Incorrect bank lineage: {name}")
        candidates[name] = record["index_sha256"]
    panel = public_panel(
        SETTING, DEVELOPMENT, DEV_COUNT, DEV_SEED, directory=PANEL_FOLDER
    )
    jobs = [
        make_job(name, DEVELOPMENT, model_hash, panel, baseline)
        for name, model_hash in candidates.items()
    ]
    save_json(OUTPUT / "development_panel.json", {**panel, "seed": DEV_SEED})
    save_json(path, jobs)


def fit_record(job, baseline, *, study=STUDY):
    path = (
        study / "fits" / job["phase"] / f"{job['name']}__{job['variant']}.npz"
    )
    record = read_json(path.with_suffix(".json"))
    if (
        record["job"] != job
        or record["rules"][baseline["key"]] != baseline["rule"]
    ):
        raise ValueError(f"Unexpected numerical fit: {path}")
    return path, record


def freeze():
    path = OUTPUT / "choices_frozen.json"
    if path.exists():
        raise FileExistsError(path)
    config = read_json(OUTPUT / "registration.json")
    baseline = config["baseline"]
    jobs_path = OUTPUT / "development_jobs.json"
    results = []
    for job in read_json(jobs_path):
        _, record = fit_record(job, baseline)
        results.append({**job, **record["metrics"][baseline["key"]]})
    ranked = sorted(
        (r for r in results if r["name"] != BASELINE),
        key=lambda r: (-r["accuracy"], r["ce"], r["name"]),
    )
    warm = ranked[0]
    original = next(row for row in results if row["name"] == BASELINE)
    source = read_json(OUTPUT / "source_fits" / warm["name"] / "complete.json")
    frozen = {"warm": warm, "original": original}
    models = OUTPUT / "frozen_indexators"
    models.mkdir()
    for row in frozen.values():
        index_path = BANKS / f"{row['name']}.npz"
        if digest(index_path) != row["model_sha256"]:
            raise ValueError(f"Changed selected bank: {index_path}")
        shutil.copyfile(index_path, models / index_path.name)
    save_json(
        path,
        {
            "policies": frozen,
            "anchor_task": source["anchor_task"],
            "development_anchors": ranked,
            "development_jobs_sha256": digest(jobs_path),
            "selection": "Development accuracy, then cross-entropy, then anchor order",
            "numeric_rule": baseline["rule"],
        },
    )
    print(
        f"Selected anchor task {source['anchor_task']} with {warm['accuracy']:.4f}"
    )


def prepare_targets(
    *,
    output=OUTPUT,
    directory=PANEL_FOLDER,
    count=TARGET_COUNT,
    seeds=TARGET_SEEDS,
    cohorts=COHORTS,
):
    path = output / "target_jobs.json"
    if path.exists():
        raise FileExistsError(path)
    choices_path = output / "choices_frozen.json"
    choices = read_json(choices_path)
    baseline = {"rule": choices["numeric_rule"]}
    panels, jobs = {}, []
    for cohort, seed in zip(cohorts, seeds, strict=True):
        panel = public_panel(
            SETTING, cohort, count, seed, directory=directory, target=True
        )
        panels[cohort] = {**panel, "seed": seed}
        for row in choices["policies"].values():
            jobs.append(
                make_job(
                    row["name"], cohort, row["model_sha256"], panel, baseline
                )
            )
    save_json(
        output / "target_panels.json",
        {"panels": panels, "choices_sha256": digest(choices_path)},
    )
    save_json(path, jobs)


def summarize(rows):
    warm, original = (rows[name] for name in ("warm", "original"))
    delta = warm - original
    return {
        "episodes": len(delta),
        "warm": warm.mean(axis=0).tolist(),
        "original": original.mean(axis=0).tolist(),
        "difference": delta.mean(axis=0).tolist(),
        "difference_ci95_halfwidth": [interval(d) for d in delta.T],
    }


def score_episodes(output, study, baseline):
    choices_path = output / "choices_frozen.json"
    choices = read_json(choices_path)
    target = read_json(output / "target_panels.json")
    if digest(choices_path) != target["choices_sha256"]:
        raise ValueError("Changed frozen choices")
    panels = target["panels"]
    jobs = read_json(output / "target_jobs.json")
    names = {row["name"]: key for key, row in choices["policies"].items()}
    predictions, receipts = {}, []
    for job in jobs:
        probability_path, record = fit_record(job, baseline, study=study)
        policy, cohort = names[job["name"]], job["phase"]
        allowed = {
            probability_path.resolve(),
            Path(panels[cohort]["path"]).resolve(),
        }
        if job["model_sha256"] is not None:
            model = (
                study.parents[1]
                / "banks/omniglot/models"
                / f"{job['name']}.npz"
            )
            allowed.add(model.resolve())
        accessed_artifacts = {
            Path(path)
            for path in record["opened_workspace_files"]
            if Path(path).is_relative_to(output.parent.resolve())
        }
        if (
            job["model_sha256"] != choices["policies"][policy]["model_sha256"]
            or job["panel_sha256"] != panels[cohort]["sha256"]
            or record["access_violations"]
            or record["query_answers_used_for_target_fitting"]
            or not accessed_artifacts <= allowed
            or digest(probability_path) != record["prediction_sha256"]
        ):
            raise ValueError(
                f"Unverified frozen predictions: {probability_path}"
            )
        predictions[cohort, policy] = probability_path
        receipts.append({**job, "sha256": record["prediction_sha256"]})
    save_json(output / "predictions_frozen.json", receipts)

    groups = {}
    for cohort, panel in panels.items():
        if digest(panel["answer_path"]) != panel["answer_sha256"]:
            raise ValueError(f"Changed scoring labels: {cohort}")
        with np.load(panel["answer_path"]) as data:
            truth = data["y"]
        rows = {}
        for policy in names.values():
            with np.load(predictions[cohort, policy]) as data:
                p = data[baseline["key"]].astype(np.float64)
            if (
                p.shape != truth.shape
                or len(p) != panel["count"]
                or not np.isfinite(p).all()
            ):
                raise ValueError(
                    f"Unexpected prediction shape: {cohort}/{policy}"
                )
            accuracy = (p.argmax(-1) == truth.argmax(-1)).mean(-1)
            ce = -(np.log(np.maximum(p, 1e-30)) * truth).sum(-1).mean(-1)
            rows[policy] = np.column_stack((accuracy, ce))
        groups[cohort] = rows
    return groups


def score():
    path = OUTPUT / "results.json"
    if path.exists():
        raise FileExistsError(path)
    baseline = read_json(OUTPUT / "registration.json")["baseline"]
    groups = score_episodes(OUTPUT, STUDY, baseline)
    pooled = {
        policy: np.concatenate([rows[policy] for rows in groups.values()])
        for policy in ("warm", "original")
    }
    results = summarize(pooled)
    save_json(
        path,
        {
            "columns": ["accuracy", "cross_entropy"],
            "cohorts": {
                cohort: summarize(rows) for cohort, rows in groups.items()
            },
            "pooled": results,
            "conditional_on": "fixed held-out Omniglot character pool",
        },
    )
    print(results)


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
    actions = {
        "prepare-development": prepare_development,
        "freeze": freeze,
        "prepare-targets": prepare_targets,
        "score": score,
    }
    if args.action in actions:
        actions[args.action]()
    else:
        run_fits(args.action, args.gpus, root=ROOT, output=OUTPUT)
