"""Replay HAR on the article's frozen subject panels and numerical settings."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from experiments.asrr_benchmark_families_v1.linear import atomic_npz
from experiments.asrr_tuning_v1.study import run
from experiments.sharing_structure_research_v1.blind_contract import digest

from .common import METHODS, OUT, WORK, freeze, register

REFERENCE = Path("artifacts/compact_experiments_2026-11-18/har3")
CONTROL = Path("artifacts/har_group_controls_2026-11-18")
ROOT = WORK / "har"
DEST = OUT / "har"
ORIGINAL_ROOT = REFERENCE.parent.parent / "local" / REFERENCE.parent.name


def stage_jobs(stage):
    archive_jobs = json.loads((REFERENCE / "target_jobs.json").read_text())
    if stage == "baselines":
        return [
            job
            for job in archive_jobs
            if job["name"] in METHODS or job["name"].startswith("wsr_")
        ]
    control_jobs = json.loads((CONTROL / "target_jobs.json").read_text())
    return [job for job in archive_jobs if job["name"] == "asrr"] + [
        job
        for job in control_jobs
        if job["name"].startswith("har_matched_shuffle_")
    ]


def replay(stage, gpus):
    DEST.mkdir(parents=True, exist_ok=True)
    jobs = stage_jobs(stage)
    models = ROOT / "banks" / "har" / "models"
    output = ROOT / "study" / "har3"
    models.mkdir(parents=True, exist_ok=True)
    panel_folder = output / "panels"
    panel_folder.mkdir(parents=True, exist_ok=True)
    reference = json.loads((REFERENCE / "target_panels.json").read_text())[
        "panels"
    ]
    for cohort, panel in reference.items():
        path = panel_folder / f"{cohort}.npz"
        if not path.exists():
            shutil.copyfile(panel["path"], path)
        if digest(path) != panel["sha256"]:
            raise ValueError(f"Changed HAR panel {path}")
    for name in {job["name"] for job in jobs} - {"dense"}:
        if not (models / f"{name}.npz").exists():
            source = ORIGINAL_ROOT / "banks" / "har" / "models" / f"{name}.npz"
            shutil.copyfile(source, models / source.name)
    freeze(DEST / f"{stage}_jobs.json", jobs)
    run(stage, gpus, root=ROOT, output=DEST)
    prediction_hashes = {}
    records = {}
    for job in jobs:
        keyed = f"{job['name']}__{job['variant']}"
        path = output / "fits" / job["phase"] / f"{keyed}.npz"
        meta = json.loads(path.with_suffix(".json").read_text())
        if (
            meta["access_violations"]
            or digest(path) != meta["prediction_sha256"]
        ):
            raise ValueError(f"Unverified HAR fit {path}")
        prediction_hashes[str(path)] = meta["prediction_sha256"]
        records.setdefault(job["phase"], {})[job["name"]] = path
    freeze(DEST / f"{stage}_seal.json", prediction_hashes)
    subjects, metrics = [], {}
    for cohort, paths in records.items():
        answers = Path(reference[cohort]["answer_path"])
        if digest(answers) != reference[cohort]["answer_sha256"]:
            raise ValueError(f"Changed query labels {answers}")
        provenance = json.loads(
            answers.with_name(f"{cohort}_provenance.json").read_text()
        )
        subjects.extend(row["person"] for row in provenance)
        with np.load(answers, allow_pickle=False) as data:
            truth = data["y"].argmax(-1)
        for name, path in paths.items():
            with np.load(path, allow_pickle=False) as data:
                values = data["frozen"]
            acc = (values.argmax(-1) == truth).mean(-1)
            method = "wsr" if name.startswith("wsr_") else name
            metrics.setdefault(method, []).append(acc)
    subjects = np.asarray(subjects)
    rows = {name: np.concatenate(values) for name, values in metrics.items()}
    shuffled = [
        values
        for name, values in rows.items()
        if name.startswith("har_matched_shuffle_")
    ]
    if shuffled:
        rows["shuffled"] = np.mean(shuffled, axis=0)
    for name, observations in rows.items():
        if observations.shape != subjects.shape:
            raise ValueError(f"Incorrect HAR panel shape for {name}")
        atomic_npz(
            DEST / f"{name}_test.npz", units=subjects, query_acc=observations
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("baselines", "ward"))
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    args = parser.parse_args()
    register()
    replay(args.stage, args.gpus)
