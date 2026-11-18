"""Select WSR on development and evaluate frozen width-64 representations."""

import argparse
import shutil
from itertools import combinations
from pathlib import Path

import numpy as np

from experiments.asrr_anchor_init_v1.report import interval
from experiments.asrr_tuning_v1.study import run as run_fits
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

from .source import (
    BANKS,
    CODES,
    HIDDEN,
    KEY,
    NUMERIC_SHARD,
    OUTPUT,
    PREVIOUS,
    ROOT,
    TIMING_EPISODES,
    VARIANT,
    WEIGHTS,
    read_json,
)

SETTING = "omniglot10"
STUDY = ROOT / "study" / SETTING
PANELS = STUDY / "panels"
METHODS = ("dense", "asrr", "wsr")
TIMING_PHASE = "test_a_0000"
CHOICES = OUTPUT / "choices_initial.json"
PLAN = OUTPUT / "target_plan.json"


def job(row, phase, panel_hash, *, seed, measure=False):
    return {
        "setting": SETTING,
        "name": row["name"],
        "phase": phase,
        "variant": VARIANT,
        "model_sha256": row["model_sha256"],
        "panel_sha256": panel_hash,
        "hidden": HIDDEN,
        "shard_size": NUMERIC_SHARD,
        "seed": seed,
        "measure_fit": measure,
    }


def fit_path(run):
    return STUDY / "fits" / run["phase"] / f"{run['name']}__{VARIANT}.npz"


def prepare_development():
    path = OUTPUT / "development_jobs.json"
    if path.exists():
        raise FileExistsError(path)
    PANELS.mkdir(parents=True, exist_ok=True)
    dev = read_json(PREVIOUS / "development_panel.json")
    if digest(Path(dev["path"])) != dev["sha256"]:
        raise ValueError("Changed development observations")
    shutil.copyfile(dev["path"], PANELS / "development.npz")
    rows = read_json(OUTPUT / "wsr_source.json")["candidates"]
    save_json(
        path,
        [job(row, "development", dev["sha256"], seed=7611172201) for row in rows],
    )


def write_target_queues(policies, plan, *, prefix=""):
    for stage, parts in (
        ("timing", [p for p in plan["parts"] if p["phase"] == TIMING_PHASE]),
        ("target", [p for p in plan["parts"] if p["phase"] != TIMING_PHASE]),
    ):
        jobs = [
            job(
                row,
                p["phase"],
                p["sha256"],
                seed=p["seed"],
                measure=stage == "timing",
            )
            for p in parts
            for row in policies.values()
        ]
        save_json(OUTPUT / f"{prefix}{stage}_jobs.json", jobs)


def select_development(runs, rule):
    records = []
    for run in runs:
        record = read_json(fit_path(run).with_suffix(".json"))
        if record["job"] != run or record["rules"][KEY] != rule:
            raise ValueError("Unexpected development rule")
        records.append({**run, **record["metrics"][KEY]})
    return min(records, key=lambda r: (-r["accuracy"], r["ce"], r["name"]))


def freeze():
    if CHOICES.exists():
        raise FileExistsError(CHOICES)
    rule = read_json(OUTPUT / "registration.json")["numeric_rule"]
    wsr = select_development(read_json(OUTPUT / "development_jobs.json"), rule)
    asrr = read_json(OUTPUT / "asrr_source.json")["candidates"][0]
    policies = {
        "dense": {"name": "dense", "model_sha256": None},
        "asrr": asrr,
        "wsr": {key: wsr[key] for key in ("name", "model_sha256", "accuracy", "ce")},
    }
    save_json(CHOICES, policies)
    old_panels = read_json(PREVIOUS / "target_panels.json")["panels"]
    parts = []
    for cohort_number, (cohort, panel) in enumerate(old_panels.items()):
        if digest(Path(panel["path"])) != panel["sha256"]:
            raise ValueError(f"Changed held-out observations: {cohort}")
        with np.load(panel["path"]) as data:
            inputs = dict(data)
        if "query_y" in inputs:
            raise ValueError("A target panel contains query labels")
        for start in range(0, panel["count"], TIMING_EPISODES):
            end = min(start + TIMING_EPISODES, panel["count"])
            phase = f"{cohort}_{start:04d}"
            path = PANELS / f"{phase}.npz"
            np.savez_compressed(
                path,
                **{key: value[start:end] for key, value in inputs.items()},
            )
            parts.append(
                {
                    "cohort": cohort,
                    "start": start,
                    "end": end,
                    "phase": phase,
                    "sha256": digest(path),
                    "seed": 7611173301 + cohort_number * 1024 + start,
                }
            )
    plan = {
        "panels": old_panels,
        "parts": parts,
        "choices_sha256": digest(CHOICES),
    }
    save_json(PLAN, plan)
    write_target_queues(policies, plan)
    print(
        f"Selected {wsr['name']} on development ({wsr['accuracy']:.4f})",
        flush=True,
    )


def prepare_added_method(method):
    source = read_json(OUTPUT / f"{method}_source.json")
    choice = source["candidates"][0]
    path = BANKS / f"{choice['name']}.npz"
    if digest(path) != choice["model_sha256"]:
        raise ValueError(f"Changed source representation: {method}")
    save_json(OUTPUT / f"choices_{method}.json", choice)
    write_target_queues({method: choice}, read_json(PLAN), prefix=f"{method}_")


def score(*, added_methods=(), result_file="results_initial.json"):
    plan = read_json(PLAN)
    if digest(CHOICES) != plan["choices_sha256"]:
        raise ValueError("Changed initial policies")
    policies = read_json(CHOICES)
    for method in added_methods:
        policies[method] = read_json(OUTPUT / f"choices_{method}.json")
    path = OUTPUT / result_file
    if path.exists():
        raise FileExistsError(path)
    rule = read_json(OUTPUT / "registration.json")["numeric_rule"]
    files, timed = {}, {}
    for method, row in policies.items():
        for part in plan["parts"]:
            phase = part["phase"]
            run = job(
                row,
                phase,
                part["sha256"],
                seed=part["seed"],
                measure=phase == TIMING_PHASE,
            )
            path_npz = fit_path(run)
            record = read_json(path_npz.with_suffix(".json"))
            allowed = {path_npz.resolve(), (PANELS / f"{phase}.npz").resolve()}
            if row["model_sha256"] is not None:
                allowed.add((BANKS / f"{row['name']}.npz").resolve())
            accessed = {
                Path(p)
                for p in record["opened_workspace_files"]
                if Path(p).is_relative_to(OUTPUT.parent.resolve())
            }
            if (
                record["job"] != run
                or record["rules"][KEY] != rule
                or record["query_answers_used_for_target_fitting"]
                or record["access_violations"]
                or not accessed <= allowed
                or digest(path_npz) != record["prediction_sha256"]
            ):
                raise ValueError(f"Unverified target fit: {path_npz}")
            files[method, phase] = path_npz
            if phase == TIMING_PHASE:
                timed[method] = record["fit_timing"]
    if len({row["device_index"] for row in timed.values()}) != 1 or any(
        row["episodes"] != TIMING_EPISODES for row in timed.values()
    ):
        raise ValueError("Target timings must use the same GPU and episode count")
    save_json(
        path.with_name(path.stem + "_prediction_hashes.json"),
        {str(f): digest(f) for f in files.values()},
    )

    answers = {}
    for cohort, panel in plan["panels"].items():
        answer = Path(panel["answer_path"])
        if digest(answer) != panel["answer_sha256"]:
            raise ValueError(f"Changed query labels: {cohort}")
        with np.load(answer) as data:
            answers[cohort] = data["y"]
    units = {}
    for method in policies:
        episodes = []
        for part in plan["parts"]:
            truth = answers[part["cohort"]][part["start"] : part["end"]]
            with np.load(files[method, part["phase"]]) as data:
                p = data[KEY].astype(np.float64)
            if p.shape != truth.shape or not np.isfinite(p).all():
                raise ValueError(f"Invalid predictions: {method}, {part['phase']}")
            accuracy = (p.argmax(-1) == truth.argmax(-1)).mean(-1)
            ce = -(truth * np.log(np.maximum(p, 1e-30))).sum(-1).mean(-1)
            episodes.append(np.column_stack((accuracy, ce)))
        units[method] = np.concatenate(episodes)
    rows = {}
    for method, results in units.items():
        accuracy = results[:, 0]
        difference = accuracy - units["asrr"][:, 0]
        source = None
        codes = WEIGHTS
        if method != "dense":
            source = read_json(OUTPUT / f"{method}_source.json")
            selected = next(
                row
                for row in source["candidates"]
                if row["name"] == policies[method]["name"]
            )
            codes = selected.get("codes", CODES)
        rows[method] = {
            "codes": codes,
            "accuracy": float(accuracy.mean()),
            "accuracy_ci95_halfwidth": interval(accuracy),
            "ce": float(results[:, 1].mean()),
            "difference_vs_asrr": float(difference.mean()),
            "difference_ci95_halfwidth": interval(difference),
            "source": source,
            "target_fitting": timed[method],
        }
    pairs = []
    for reference, method in combinations(units, 2):
        difference = units[method][:, 0] - units[reference][:, 0]
        pairs.append(
            {
                "method": method,
                "reference": reference,
                "accuracy_difference": float(difference.mean()),
                "ci95_halfwidth": interval(difference),
            }
        )
    save_json(
        path,
        {
            "episodes": len(units["asrr"]),
            "dense_source_preparation": read_json(ROOT / "source/dense/metrics.json"),
            "methods": rows,
            "paired_differences": pairs,
        },
    )
    print({m: (r["accuracy"], r["target_fitting"]["seconds"]) for m, r in rows.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "prepare-development",
            "development",
            "freeze",
            "timing",
            "target",
            "benchmark",
            "score",
            "prepare-yeh",
            "yeh_timing",
            "yeh_target",
            "yeh_benchmark",
            "score-yeh",
            "prepare-g34",
            "g34_benchmark",
            "score-g34",
        ),
    )
    parser.add_argument("--gpus", nargs="+", type=int, default=[7])
    args = parser.parse_args()
    if args.action in ("benchmark", "yeh_benchmark", "g34_benchmark"):
        prefix = args.action.removesuffix("benchmark")
        run_fits(
            f"{prefix}target",
            args.gpus,
            root=ROOT,
            output=OUTPUT,
            timing_first=(f"{prefix}timing", args.gpus[-1]),
        )
    elif args.action in (
        "development",
        "timing",
        "target",
        "yeh_timing",
        "yeh_target",
    ):
        run_fits(args.action, args.gpus, root=ROOT, output=OUTPUT)
    elif args.action == "score-yeh":
        score(added_methods=("yeh",), result_file="results.json")
    elif args.action == "score-g34":
        score(added_methods=("yeh", "g34"), result_file="results_with_g34.json")
    elif args.action in ("prepare-yeh", "prepare-g34"):
        prepare_added_method(args.action.removeprefix("prepare-"))
    else:
        {
            "prepare-development": prepare_development,
            "freeze": freeze,
            "score": score,
        }[args.action]()
