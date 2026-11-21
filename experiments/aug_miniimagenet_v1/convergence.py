"""Check source-training and target-refit convergence on Aug-miniImageNet."""

import argparse
import json
from itertools import product

import torch

from experiments.asrr_benchmark_families_v1.classification import read_data
from experiments.asrr_benchmark_families_v1.linear import atomic_npz
from experiments.asrr_transfer_refinement_v1.study import ordering
from experiments.paper_transfer_v2.bilevel import train_msr, train_yeh
from experiments.paper_transfer_v2.common import freeze
from experiments.paper_transfer_v2.report import confidence
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

from .panels import CODES, LAYOUT, SEED, SHOT, WORK, device_for, prepare
from .study import (
    MODELS,
    RANK,
    add_group_shuffles,
    build_asrr_bank,
    confirm,
    fit_source_dense,
    parallel,
    source_arrays,
)
from .tuning import OUTPUT, RUNTIME, score_probes, select_probes

RESULTS = OUTPUT / "convergence"
LOCAL = RUNTIME / "convergence"
SOURCE_MODELS = LOCAL / "sources"

MSR_GRID = ((16, 0.003, 0), (16, 0.003, 5317), (16, 0.006, 0), (1, 0.006, 0))
MSR_STEPS = (12_800, 19_200, 25_600, 38_400, 51_200)
DECAY_POINT = 25_600
MSR_EXTENSION = (64_000, 76_800)
YEH_GRID = tuple(product((0.003, 0.012), (False, True)))
YEH_STEPS = (800, 1600, 3200, 6400)
DENSE_SOURCE_STEPS = 3_200
DEV_EPISODES = 1_152
DEV_SEED = SEED + 290_137
TEST_SEED = SEED + 430_399

FAST_RATES = (0.02, 0.03, 0.045, 0.06)
FAST_STEPS = (5, 8, 10, 12, 16, 25, 75)
SLOW_RATES = (0.0015, 0.003, 0.005, 0.01)
SLOW_STEPS = (75, 150, 300, 600, 1200)


def msr_run_name(factors, rate, seed, decay_after=None):
    name = f"msr_f{factors}_rate{rate:g}_seed{seed}"
    if decay_after is None:
        return name
    suffix = (
        "_halved" if decay_after == DECAY_POINT else f"_halved{decay_after}"
    )
    return name + suffix


def train_source(job):
    gpu, method, settings, checkpoints = job
    if method == "msr":
        factors, rate, seed, *decay = settings
        name = msr_run_name(*settings)
        schedule = {decay[0] + 1: rate / 2} if decay else None
    else:
        rate, warm = settings
        name = f"yeh_soft_warm{int(warm)}_rate{rate:g}"
    paths = {
        step: SOURCE_MODELS / f"{name}_step{step}.npz" for step in checkpoints
    }
    if all(path.exists() for path in paths.values()):
        return
    device = device_for(gpu)
    support, _, queries, labels = source_arrays(device)
    data = support, torch.cat(tuple(queries.values()), 1), labels.repeat(1, 2)
    if method == "msr":
        fits = train_msr(
            data,
            LAYOUT,
            factors,
            RANK,
            rate,
            SHOT,
            device,
            checkpoints=checkpoints,
            seed_offset=seed,
            state_path=SOURCE_MODELS / f"{name}.pt",
            rate_at_step=schedule,
        )
        key = "factor"
    else:
        dense = read_data(WORK / "dense_sources.npz")
        best = int(dense["ce"][:, 0].argmin())
        exemplar = dense["parameters"][best, 0, : LAYOUT.connections]
        fits = train_yeh(
            data,
            LAYOUT,
            exemplar,
            LAYOUT.sites,
            CODES,
            rate,
            warm,
            SHOT,
            device,
            checkpoints=checkpoints,
        )
        key = "basis"
    for step, basis, *_ in fits:
        atomic_npz(paths[step], **{key: basis})
        print(paths[step].stem, flush=True)


def fit_source_jobs(jobs, gpus):
    for start in range(0, len(jobs), len(gpus)):
        batch = jobs[start : start + len(gpus)]
        parallel(
            train_source,
            [(gpu, *job) for gpu, job in zip(gpus, batch)],
            len(batch),
        )


def fit_sources(gpus):
    SOURCE_MODELS.mkdir(parents=True, exist_ok=True)
    jobs = [("msr", settings, MSR_STEPS) for settings in MSR_GRID]
    jobs.extend(("yeh_soft", settings, YEH_STEPS) for settings in YEH_GRID)
    fit_source_jobs(jobs, gpus)


def snapshot_for_decay(settings, after):
    name = msr_run_name(*settings)
    destination = SOURCE_MODELS / f"{msr_run_name(*settings, after)}.pt"
    if destination.exists():
        return
    state = torch.load(
        SOURCE_MODELS / f"{name}.pt", map_location="cpu", weights_only=True
    )
    if state["step"] != after:
        raise ValueError(f"Expected an optimizer state at step {after}")
    temporary = destination.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(destination)


def fit_decayed_sources(gpus):
    checkpoints = tuple(step for step in MSR_STEPS if step > DECAY_POINT)
    for settings in MSR_GRID[:2]:
        snapshot_for_decay(settings, DECAY_POINT)
    jobs = [("msr", (*s, DECAY_POINT), checkpoints) for s in MSR_GRID[:2]]
    fit_source_jobs(jobs, gpus)


def fit_msr_extension(gpus):
    after = MSR_STEPS[-1]
    trajectories = json.loads((RESULTS / "msr_trajectories.json").read_text())
    settings = min(
        MSR_GRID,
        key=lambda s: ordering(
            next(
                row["best_refit"]
                for row in trajectories[msr_run_name(*s)]
                if row["step"] == after
            )
        ),
    )
    freeze(
        RESULTS / "msr_extension_plan.json",
        {
            "settings": list(settings),
            "resume_step": after,
            "checkpoints": list(MSR_EXTENSION),
            "decay_factor": 0.5,
        },
    )
    snapshot_for_decay(settings, after)
    fit_source_jobs(
        [("msr", s, MSR_EXTENSION) for s in (settings, (*settings, after))],
        gpus,
    )


def evaluate(name, models, gpus):
    directory = LOCAL / name / f"gpus_{len(gpus)}"
    freeze(
        RESULTS / f"{name}_grid.json",
        {
            "panel_sha256": digest(LOCAL / "dev/manifest.json"),
            "fits": [
                [key, path, list(rates), list(steps)]
                for key, path, rates, steps in models
            ],
        },
    )
    rows = score_probes(
        models, gpus, directory, panel=LOCAL / "dev", save_episodes=True
    )
    save_json(RESULTS / f"{name}_validation.json", rows)
    return rows, directory


def fit_non_msr(gpus):
    prepare(
        "dev",
        gpus,
        work=LOCAL,
        output=RESULTS,
        seed=DEV_SEED,
        episodes=DEV_EPISODES,
    )
    dense_path = fit_source_dense(
        gpus,
        directory=LOCAL / "dense_long",
        steps=DENSE_SOURCE_STEPS,
    )
    long_bank = build_asrr_bank(
        LOCAL / "asrr_long", dense_path, device_for(gpus[0])
    )
    files = json.loads((MODELS / "candidates.json").read_text())
    original = json.loads(
        (OUTPUT / "confirmation/primary_choices.json").read_text()
    )
    yeh_models = {
        original["yeh_soft"]["candidate"]: original["yeh_soft"]["model"]
    }
    yeh_models.update(
        {p.stem: str(p) for p in sorted(SOURCE_MODELS.glob("yeh_*.npz"))}
    )
    if len(yeh_models) != len(YEH_GRID) * len(YEH_STEPS) + 1:
        raise ValueError("The scheduled Yeh source checkpoints are incomplete")
    yeh_screen = [
        (name, path, (0.003, 0.01), SLOW_STEPS[:4])
        for name, path in yeh_models.items()
    ]
    screen, _ = evaluate("yeh_screen", yeh_screen, gpus)
    top_yeh = list(
        dict.fromkeys(row["candidate"] for row in sorted(screen, key=ordering))
    )[:4]
    asrr_models = {
        name: p for name, p in files.items() if name.startswith("asrr_")
    }
    asrr_models.update(
        {
            f"asrr_long_{name}": spec["path"]
            for name, spec in long_bank.items()
            if spec["group_weight"] > 0
        }
    )
    models = [
        (name, path, FAST_RATES, FAST_STEPS)
        for name, path in asrr_models.items()
    ]
    models.extend(
        (name, yeh_models[name], SLOW_RATES, SLOW_STEPS) for name in top_yeh
    )
    models.append(("dense", None, SLOW_RATES, SLOW_STEPS))
    evaluate("refit", models, gpus)
    save_json(
        RESULTS / "non_msr_models.json",
        {**asrr_models, **yeh_models, "dense": None},
    )
    source_ce = {
        "800": float(read_data(WORK / "dense_sources.npz")["ce"][:, 0].mean()),
        str(DENSE_SOURCE_STEPS): float(
            read_data(dense_path)["ce"][:, 0].mean()
        ),
    }
    save_json(RESULTS / "dense_source_ce.json", source_ce)


def fit_msr_validation(gpus, *, early_only=False):
    paths = {p.stem: str(p) for p in sorted(SOURCE_MODELS.glob("msr_*.npz"))}
    early = {
        key: path
        for key, path in paths.items()
        if int(key.rsplit("_step", 1)[1]) <= DECAY_POINT
    }
    end = MSR_STEPS[-1]
    late = {
        key: path
        for key, path in paths.items()
        if DECAY_POINT < int(key.rsplit("_step", 1)[1]) <= end
    }
    early_result = RESULTS / "msr_early_validation.json"
    if len(early) != len(MSR_GRID) * 3:
        raise ValueError("The scheduled early MSR checkpoints are incomplete")
    if not early_result.exists():
        evaluate(
            "msr_early",
            [
                (name, path, FAST_RATES, FAST_STEPS)
                for name, path in early.items()
            ],
            gpus,
        )
    if early_only:
        return
    if len(late) < 2 * (len(MSR_GRID) + len(MSR_GRID[:2])):
        raise ValueError("The scheduled late MSR checkpoints are incomplete")
    previous = json.loads(
        (OUTPUT / "confirmation/primary_choices.json").read_text()
    )["msr"]
    late[previous["candidate"]] = previous["model"]
    banks = {"msr_late": late}
    for step in MSR_EXTENSION:
        checkpoints = {
            k: p for k, p in paths.items() if k.endswith(f"_step{step}")
        }
        if len(checkpoints) == 2:
            banks[f"msr_extension_{step}"] = checkpoints
    paths = {**early}
    for stage, bank in banks.items():
        paths.update(bank)
        if not (RESULTS / f"{stage}_validation.json").exists():
            evaluate(
                stage,
                [(n, p, FAST_RATES, FAST_STEPS) for n, p in bank.items()],
                gpus,
            )
    stages = ["msr_early", *banks]
    rows = []
    for stage in stages:
        rows.extend(
            json.loads((RESULTS / f"{stage}_validation.json").read_text())
        )
    save_json(RESULTS / "msr_validation.json", rows)
    scores = {}
    for stage in stages:
        for path in (LOCAL / stage).glob("gpus_*/validation_*.npz"):
            scores.update(read_data(path))
    trajectories = {}
    for name in paths:
        arm, step = name.rsplit("_step", 1)
        trajectories.setdefault(arm, {})[int(step)] = name
    changes = {}
    for arm, steps in trajectories.items():
        parent, separator, after = arm.partition("_halved")
        if separator:
            point = int(after) if after else DECAY_POINT
            steps = {point: trajectories[parent][point], **steps}
        changes[arm] = []
        previous = None
        for step, name in sorted(steps.items()):
            accuracy = scores[f"{name}@0.045@10"]
            best = min(
                (r for r in rows if r["candidate"] == name), key=ordering
            )
            changes[arm].append(
                {
                    "step": step,
                    "fixed_refit_mean": float(accuracy.mean()),
                    "fixed_refit_change": (
                        confidence(accuracy - previous)
                        if previous is not None
                        else None
                    ),
                    "best_refit": best,
                }
            )
            previous = accuracy
    save_json(RESULTS / "msr_trajectories.json", changes)
    save_json(RESULTS / "msr_models.json", paths)


def fit_yeh_tail(gpus):
    rows = json.loads((RESULTS / "refit_validation.json").read_text())
    names = [
        row["candidate"]
        for row in sorted(rows, key=ordering)
        if row["candidate"].startswith("yeh_soft")
    ]
    paths = json.loads((RESULTS / "non_msr_models.json").read_text())
    evaluate(
        "yeh_tail",
        [
            (name, paths[name], (0.003, 0.005, 0.01), (600, 1200, 2400))
            for name in list(dict.fromkeys(names))[:3]
        ],
        gpus,
    )


def fit_dense_rates(gpus):
    evaluate(
        "dense_rates",
        [
            ("dense", None, (rate,), (25, 50, 75, 150, 300, 600))
            for rate in (0.0075, 0.015, 0.02, 0.03)
        ],
        gpus,
    )


def final_comparison(gpus):
    restart = RESULTS / "test"
    directory = LOCAL / "confirmation"
    restart.mkdir(parents=True, exist_ok=True)
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for stage in ("yeh_screen", "refit", "yeh_tail", "dense_rates", "msr"):
        rows.extend(
            json.loads((RESULTS / f"{stage}_validation.json").read_text())
        )
    paths = {}
    for family in ("non_msr", "msr"):
        paths.update(
            json.loads((RESULTS / f"{family}_models.json").read_text())
        )
    old = json.loads(
        (OUTPUT / "confirmation/primary_choices.json").read_text()
    )
    choices = select_probes(rows, paths, old, restart / "primary_choices.json")
    asrr = choices["asrr"]["candidate"]
    if asrr.startswith("asrr_long_"):
        add_group_shuffles(
            choices,
            directory,
            source_bank=LOCAL / "asrr_long",
            structure=asrr.removeprefix("asrr_long_"),
        )
    else:
        add_group_shuffles(choices, directory)
    freeze(restart / "choices.json", choices)
    if not (restart / "results.json").exists():
        confirm(choices, gpus, work=directory, output=restart, seed=TEST_SEED)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    stages = {
        "source": fit_sources,
        "decay": fit_decayed_sources,
        "msr-extend": fit_msr_extension,
        "non-msr": fit_non_msr,
        "yeh-tail": fit_yeh_tail,
        "dense-rates": fit_dense_rates,
        "msr-early": lambda gpus: fit_msr_validation(gpus, early_only=True),
        "msr": fit_msr_validation,
        "test": final_comparison,
    }
    parser.add_argument("--stage", choices=stages, default="source")
    args = parser.parse_args()
    stages[args.stage](args.gpus)
