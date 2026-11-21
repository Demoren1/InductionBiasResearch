"""Check the edges of the Aug-miniImageNet source and refit grids."""

import argparse
import json
from itertools import product

import torch

from experiments.asrr_benchmark_families_v1.classification import read_data
from experiments.asrr_benchmark_families_v1.linear import atomic_npz
from experiments.asrr_transfer_refinement_v1.study import (
    evaluate_panel,
    ordering,
    validation_rows,
)
from experiments.paper_transfer_v2.bilevel import train_msr, train_yeh
from experiments.paper_transfer_v2.common import freeze
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

from .panels import CODES, LAYOUT, OUT, SEED, SHOT, WORK, device_for
from .study import (
    METHODS,
    MODELS,
    RANK,
    SOURCE,
    TARGET,
    add_group_shuffles,
    confirm,
    parallel,
    source_arrays,
)

OUTPUT = OUT / "tuning_check"
RUNTIME = WORK / "tuning_check"
EDGE_STEPS = (5, 10, 600)
REFIT_STEPS = tuple(sorted({*TARGET["steps"], *EDGE_STEPS}))
MSR_CHECKPOINTS = {0.003: (1600, 3200), 0.006: (800, 1600, 3200)}
YEH_CROSSES = ((0.003, False), (0.012, True))


def fit_probe(job):
    method, factors, rate, warm, steps, gpu = job
    suffix = f"f{factors}" if method == "msr" else f"warm{int(warm)}"
    paths = {
        step: RUNTIME / f"{method}_{suffix}_rate{rate:g}_step{step}.npz"
        for step in steps
    }
    if not all(path.exists() for path in paths.values()):
        device = device_for(gpu)
        ordered, _, queries, qy = source_arrays(device)
        data = (
            ordered,
            torch.cat(tuple(queries.values()), dim=1),
            qy.repeat(1, len(queries)),
        )
        if method == "msr":
            for step, factor in train_msr(
                data,
                LAYOUT,
                factors,
                RANK,
                rate,
                SHOT,
                device,
                checkpoints=steps,
            ):
                atomic_npz(paths[step], factor=factor)
                print(paths[step].stem, flush=True)
        else:
            dense = read_data(WORK / "dense_sources.npz")
            best = int(dense["ce"][:, 0].argmin())
            exemplar = dense["parameters"][best, 0, : LAYOUT.connections]
            for step, basis, _ in train_yeh(
                data,
                LAYOUT,
                exemplar,
                LAYOUT.sites,
                CODES,
                rate,
                warm,
                SHOT,
                device,
            ):
                atomic_npz(paths[step], basis=basis)
                print(paths[step].stem, flush=True)
    return {path.stem: str(path) for path in paths.values()}


def score_job(job):
    shard, entries, gpu, directory, panel, save_episodes = job
    path = directory / f"validation_{shard:02d}.json"
    archive = path.with_suffix(".npz")
    if path.exists() and (not save_episodes or archive.exists()):
        return json.loads(path.read_text())
    device = device_for(gpu)
    rows = []
    grids = {}
    episode_accuracy = {}
    for name, source, rates, steps in entries:
        grids.setdefault((rates, steps), {})[name] = source
    for (rates, steps), models in grids.items():
        scores = evaluate_panel(
            panel,
            LAYOUT,
            SHOT,
            models,
            device,
            phase="dev",
            rates=rates,
            steps=steps,
            replicates=TARGET["seed_replicates"],
        )
        rows.extend(validation_rows(scores, None))
        if save_episodes:
            episode_accuracy.update(
                {
                    f"{name}@{rate:g}@{step}": (
                        0.5 * (scores["clean"][:, 0] + scores["query"][:, 0])
                    )
                    for (name, rate, step), scores in scores.items()
                }
            )
    if save_episodes:
        atomic_npz(archive, **episode_accuracy)
    save_json(path, rows)
    print(f"Extra validation shard {shard}: {len(entries)} bases", flush=True)
    return rows


def score_probes(
    models, gpus, directory, *, panel=WORK / "dev", save_episodes=False
):
    directory.mkdir(parents=True, exist_ok=True)
    groups = [models[i :: len(gpus)] for i in range(len(gpus))]
    jobs = [
        (i, rows, gpu, directory, panel, save_episodes)
        for i, (rows, gpu) in enumerate(zip(groups, gpus, strict=True))
        if rows
    ]
    return [
        row for shard in parallel(score_job, jobs, len(jobs)) for row in shard
    ]


def select_probes(rows, paths, original, destination):
    selected = {}
    for method in METHODS:
        best = min(
            (r for r in rows if r["candidate"].startswith(method)),
            key=ordering,
        )
        model = paths[best["candidate"]]
        selected[method] = {
            **best,
            "model": model,
            "sha256": digest(model) if model is not None else None,
            "original_margin": original[method]["margin"],
        }
    freeze(destination, selected)
    print(selected, flush=True)
    return selected


def run(gpus):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    original = json.loads((OUT / "choices.json").read_text())
    old_rows = json.loads((OUT / "validation.json").read_text())
    original_paths = json.loads((MODELS / "candidates.json").read_text())
    finalists = {}
    for method in METHODS:
        ordered = sorted(
            (r for r in old_rows if r["candidate"].startswith(method)),
            key=ordering,
        )
        best = list(dict.fromkeys(row["candidate"] for row in ordered))[:3]
        finalists.update({name: original_paths[name] for name in best})
    freeze(
        OUTPUT / "protocol.json",
        {
            "original_choices_sha256": digest(OUT / "choices.json"),
            "development_panel_sha256": digest(WORK / "dev/manifest.json"),
            "refit_edge_steps": list(EDGE_STEPS),
            "msr_checkpoints": {
                str(rate): list(steps)
                for rate, steps in MSR_CHECKPOINTS.items()
            },
            "yeh_crosses": [list(pair) for pair in YEH_CROSSES],
            "old_finalists": list(finalists),
        },
    )
    probes = [
        ("msr", count, rate, False, MSR_CHECKPOINTS[rate])
        for count, rate in product((1, LAYOUT.sites), MSR_CHECKPOINTS)
    ]
    probes.extend(
        ("yeh_soft", 0, rate, warm, SOURCE["checkpoints"])
        for rate, warm in YEH_CROSSES
    )
    new_paths = {}
    for start in range(0, len(probes), len(gpus)):
        batch = probes[start : start + len(gpus)]
        for result in parallel(
            fit_probe,
            [(*spec, gpu) for spec, gpu in zip(batch, gpus)],
            len(batch),
        ):
            new_paths.update(result)
    models = [
        (name, path, tuple(TARGET["rates"]), steps)
        for bank, steps in ((finalists, EDGE_STEPS), (new_paths, REFIT_STEPS))
        for name, path in bank.items()
    ]
    extra = score_probes(models, gpus, RUNTIME)
    save_json(OUTPUT / "validation.json", extra)
    paths = {**original_paths, **new_paths}
    select_probes(
        [*old_rows, *extra], paths, original, OUTPUT / "choices.json"
    )


def extend(gpus):
    destination = OUTPUT / "confirmation"
    directory = RUNTIME / "confirmation"
    destination.mkdir(parents=True, exist_ok=True)
    seed = SEED + 87_321
    checkpoints = (6_400, 12_800)
    rates = (0.003, 0.01, 0.02, 0.03, 0.045, 0.06, 0.09)
    steps = (5, 10, 15, 25, 75, 150, 300)
    fast_steps = steps[:5]
    initial = json.loads((OUT / "choices.json").read_text())
    rows = json.loads((OUT / "validation.json").read_text())
    rows.extend(json.loads((OUTPUT / "validation.json").read_text()))
    paths = json.loads((MODELS / "candidates.json").read_text())
    paths.update({p.stem: str(p) for p in RUNTIME.glob("*.npz")})
    finalists = []
    for prefix in ("asrr", "msr_f1_", "msr_f16_"):
        ordered = sorted(
            (r for r in rows if r["candidate"].startswith(prefix)),
            key=ordering,
        )
        finalists.extend(
            list(dict.fromkeys(row["candidate"] for row in ordered))[:2]
        )
    configurations = list(product((1, LAYOUT.sites), (0.003, 0.006)))
    freeze(
        destination / "audit_plan.json",
        {
            "first_audit_sha256": digest(OUTPUT / "choices.json"),
            "development_panel_sha256": digest(WORK / "dev/manifest.json"),
            "msr_factors_and_rates": [list(pair) for pair in configurations],
            "msr_checkpoints": list(checkpoints),
            "refit_rates": list(rates),
            "new_refit_steps": list(steps),
            "finalist_refit_steps": list(fast_steps),
            "finalists": finalists,
            "test_seed": seed,
        },
    )
    new = {}
    probes = [
        ("msr", factors, rate, False, checkpoints)
        for factors, rate in configurations
    ]
    for start in range(0, len(probes), len(gpus)):
        batch = probes[start : start + len(gpus)]
        for result in parallel(
            fit_probe,
            [(*spec, gpu) for spec, gpu in zip(batch, gpus)],
            len(batch),
        ):
            new.update(result)
    models = [(name, path, rates, steps) for name, path in new.items()]
    models.extend((name, paths[name], rates, fast_steps) for name in finalists)
    extra = score_probes(models, gpus, directory)
    save_json(destination / "validation.json", extra)
    choices = select_probes(
        [*rows, *extra],
        {**paths, **new},
        initial,
        destination / "primary_choices.json",
    )
    add_group_shuffles(choices, directory)
    freeze(destination / "choices.json", choices)
    if not (destination / "results.json").exists():
        confirm(choices, gpus, work=directory, output=destination, seed=seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--extend", action="store_true")
    args = parser.parse_args()
    (extend if args.extend else run)(args.gpus)
