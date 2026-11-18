"""Test whether learned groups matter for source-to-subject transfer on HAR."""

import argparse
from pathlib import Path

import numpy as np

from experiments.asrr_anchor_init_v1.report import interval
from experiments.asrr_tuning_v1.study import run as run_fits
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from experiments.srr_approx_v1.synthetic_random_controls import (
    shared_background,
    shuffle_groups,
)

from .source import OUTPUT, ROOT, read_json
from .transfer import save_model

REFERENCE = OUTPUT / "har3"
DEST = OUTPUT.with_name("har_group_controls_2026-11-18")
BANKS = ROOT / "banks/har/models"
STUDY = ROOT / "study/har3"
BASELINES = ("asrr", "dense", "wsr")
FAMILIES = ("matched_shuffle", "shared_background")
SEED = 7611189670
MASKS = 4


def prepare():
    if DEST.exists():
        raise FileExistsError(DEST)
    original_choices = REFERENCE / "choices_frozen.json"
    original_panels = REFERENCE / "target_panels.json"
    choices = read_json(original_choices)
    previous = read_json(original_panels)
    if digest(original_choices) != previous["choices_sha256"]:
        raise ValueError("Changed HAR choices")
    learned = choices["asrr"]
    original = BANKS / f"{learned['name']}.npz"
    if digest(original) != learned["model_sha256"]:
        raise ValueError("Changed original aSRR indexator")
    with np.load(original) as model:
        index, codes, scale = model["index"], int(model["codes"]), float(model["scale"])

    controls = {}
    generator = np.random.default_rng(SEED)
    for family in FAMILIES:
        controls[family] = []
        for number in range(MASKS):
            name = f"har_{family}_{number}"
            if (BANKS / f"{name}.npz").exists():
                raise FileExistsError(name)
            if family == "matched_shuffle":
                candidate = shuffle_groups(index, generator)
                if not np.array_equal(
                    np.sign(candidate), np.sign(index)
                ) or not np.array_equal(
                    np.sort(candidate, axis=0), np.sort(index, axis=0)
                ):
                    raise ValueError("Changed support, signs or signed group counts")
            else:
                candidate = shared_background(index.shape, codes, generator)
            row = save_model(BANKS, name, codes, index=candidate, scale=scale)
            choices[name] = {**row, "rule": learned["rule"]}
            controls[family].append(name)

    DEST.mkdir()
    control_names = [name for family in controls.values() for name in family]
    selected = {name: choices[name] for name in (*BASELINES, *control_names)}
    save_json(DEST / "choices_frozen.json", selected)
    save_json(
        DEST / "registration.json",
        {
            "choices_sha256": digest(DEST / "choices_frozen.json"),
            "reference_panels_sha256": digest(original_panels),
            "reference_choices_sha256": digest(original_choices),
            "controls": controls,
            "sampler_seed": SEED,
            "target_fitting": "Controls use the preexisting aSRR numerical rule and target seeds",
            "matched_shuffle": "Shuffle labels within each neuron and sign; preserve support, signs and group sizes",
            "shared_background": "One common code and 127 random singletons with nearly equal per-neuron allocation",
        },
    )
    evaluations, jobs = {}, []
    for cohort, methods in previous["evaluations"].items():
        rows = {name: methods[name] for name in BASELINES}
        for name, row in selected.items():
            if name in BASELINES:
                continue
            job = {
                **methods["asrr"]["job"],
                "name": row["name"],
                "model_sha256": row["model_sha256"],
            }
            rows[name] = {
                "path": str(
                    STUDY / "fits" / cohort / f"{job['name']}__{job['variant']}.npz"
                ),
                "key": methods["asrr"]["key"],
                "job": job,
            }
            jobs.append(job)
        evaluations[cohort] = rows
    save_json(
        DEST / "target_panels.json",
        {
            "panels": previous["panels"],
            "evaluations": evaluations,
        },
    )
    save_json(DEST / "target_jobs.json", jobs)


def score():
    output = DEST / "results.json"
    if output.exists():
        raise FileExistsError(output)
    registration = read_json(DEST / "registration.json")
    manifest = read_json(DEST / "target_panels.json")
    choice_path = DEST / "choices_frozen.json"
    if digest(choice_path) != registration["choices_sha256"]:
        raise ValueError("Changed frozen indexators")
    choices = read_json(choice_path)
    frozen = {}
    for cohort, methods in manifest["evaluations"].items():
        frozen[cohort] = {}
        for name, row in methods.items():
            path = Path(row["path"])
            receipt = read_json(path.with_suffix(".json"))
            sha = digest(path)
            if (
                receipt["job"] != row["job"]
                or receipt["access_violations"]
                or receipt["query_answers_used_for_target_fitting"]
                or receipt["prediction_sha256"] != sha
            ):
                raise ValueError(f"Unverified fitted predictions: {path}")
            frozen[cohort][name] = sha
    save_json(DEST / "predictions_frozen.json", frozen)

    records = {name: [] for name in choices}
    subjects = []
    for cohort, panel in manifest["panels"].items():
        answers = Path(panel["answer_path"])
        if digest(answers) != panel["answer_sha256"]:
            raise ValueError("Changed query labels")
        with np.load(answers) as data:
            truth = data["y"]
        subjects.extend(
            row["person"]
            for row in read_json(answers.with_name(f"{cohort}_provenance.json"))
        )
        for name, row in manifest["evaluations"][cohort].items():
            with np.load(row["path"]) as data:
                p = data[row["key"]].astype(np.float64)
            if p.shape != truth.shape or not np.isfinite(p).all():
                raise ValueError(f"Invalid prediction array: {name}/{cohort}")
            acc = (p.argmax(-1) == truth.argmax(-1)).mean(-1)
            ce = -(np.log(np.maximum(p, 1e-30)) * truth).sum(-1).mean(-1)
            records[name].append(np.column_stack((acc, ce)))
    subjects = np.asarray(subjects)
    persons = np.unique(subjects)
    units = {}
    for method, batches in records.items():
        episodes = np.concatenate(batches)
        units[method] = np.stack(
            [episodes[subjects == person].mean(0) for person in persons]
        )
    for family, methods in registration["controls"].items():
        units[family] = np.mean([units[name] for name in methods], axis=0)
    methods = {}
    for name, outcome in units.items():
        delta = units["asrr"][:, 0] - outcome[:, 0]
        methods[name] = {
            "accuracy": float(outcome[:, 0].mean()),
            "accuracy_ci95_halfwidth": interval(outcome[:, 0]),
            "mean_cross_entropy": float(outcome[:, 1].mean()),
            "asrr_minus_method": float(delta.mean()),
            "difference_ci95_halfwidth": interval(delta),
            "per_subject_accuracy": outcome[:, 0].tolist(),
        }
    save_json(
        output,
        {
            "episodes": len(subjects),
            "subjects": persons.tolist(),
            "codes": choices["asrr"]["codes"],
            "methods": methods,
        },
    )
    print({name: methods[name] for name in (*BASELINES, *FAMILIES)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "score"))
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "run":
        run_fits("target", args.gpus, root=ROOT, output=DEST)
    else:
        score()
