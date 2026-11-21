"""Fit aSRR, matched Dense, Yeh soft, and MSR-FC on Aug-miniImageNet."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from multiprocessing import get_context

import numpy as np
import torch

from experiments.asrr_benchmark_families_v1.classification import (
    input_tensor,
    read_data,
)
from experiments.asrr_benchmark_families_v1.linear import atomic_npz
from experiments.asrr_transfer_refinement_v1.study import (
    evaluate_panel,
    ordering,
    validation_rows,
)
from experiments.asrr_universal_v1.fixed_blocks import model_bank
from experiments.asrr_universal_v1.structure import spectral_basis
from experiments.paper_transfer_v2.bilevel import train_msr, train_yeh
from experiments.paper_transfer_v2.common import CONFIG, freeze
from experiments.paper_transfer_v2.report import confidence
from experiments.real_group_transfer_v1.optimization import fit_dense
from experiments.real_group_transfer_v1.study import source_profiles
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

from .panels import (
    CODES,
    LAYOUT,
    OUT,
    PLAN,
    SEED,
    SHOT,
    WORK,
    device_for,
    prepare,
)

MODELS = WORK / "models"
TARGET = CONFIG["target"]
SOURCE = CONFIG["real_bilevel"]
RANK = CODES // LAYOUT.hidden
METHODS = ("asrr", "dense", "yeh_soft", "msr")


def parallel(function, jobs, workers):
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn")
    ) as pool:
        return list(pool.map(function, jobs))


def source_arrays(device, rows=slice(None)):
    data = read_data(WORK / "source.npz")
    ordered = input_tensor(data["sx"][rows], device)
    sy = torch.as_tensor(
        data["sy"][rows], dtype=torch.long, device=device
    ).flatten(1, 2)
    queries = {
        name: input_tensor(data[key][rows], device)
        for name, key in (("query", "qx"), ("clean", "qx_clean"))
    }
    qy = torch.as_tensor(data["qy"][rows], dtype=torch.long, device=device)
    return ordered, sy, queries, qy


def dense_job(job):
    shard, rows, gpu, directory, steps = job
    path = directory / f"dense_{shard:02d}.npz"
    if path.exists():
        return path
    device = device_for(gpu)
    ordered, sy, queries, qy = source_arrays(device, rows)
    mixture = torch.cat(tuple(queries.values()), dim=1)
    states, ce, seconds = fit_dense(
        (ordered.flatten(1, 2), sy, mixture, qy.repeat(1, len(queries))),
        LAYOUT,
        steps=steps,
        restarts=PLAN["dense_source"]["restarts"],
        device=device,
        seed=SEED + 53,
    )
    atomic_npz(path, parameters=states, ce=ce)
    print(f"Dense source shard {shard}: {seconds:.1f}s", flush=True)
    return path


def fit_source_dense(gpus, *, directory=WORK, steps=None):
    if steps is None:
        steps = PLAN["dense_source"]["steps"]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "dense_sources.npz"
    if path.exists():
        return path
    ids = np.array_split(np.arange(PLAN["source"]["episodes"]), len(gpus))
    jobs = [
        (i, rows, gpu, directory, steps)
        for i, (rows, gpu) in enumerate(zip(ids, gpus, strict=True))
        if len(rows)
    ]
    shards = [read_data(path) for path in parallel(dense_job, jobs, len(jobs))]
    atomic_npz(
        path,
        **{
            key: np.concatenate([shard[key] for shard in shards])
            for key in shards[0]
        },
    )
    return path


def build_asrr_bank(folder, dense_path, device):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "bank.json"
    if path.exists():
        return json.loads(path.read_text())
    ordered, sy, queries, qy = source_arrays(device)
    dense = read_data(dense_path)
    states = dense["parameters"][np.argsort(dense["ce"][:, 0])]
    profiles = source_profiles(states, LAYOUT)
    return model_bank(
        folder,
        profiles,
        (ordered.flatten(1, 2), sy, queries, qy),
        ordered,
        LAYOUT,
        RANK,
        SHOT,
    )


def train_job(job):
    method, factors, rate, warm, gpu = job
    folder = MODELS / method
    folder.mkdir(parents=True, exist_ok=True)
    device = device_for(gpu)
    if method == "asrr":
        bank = build_asrr_bank(folder, WORK / "dense_sources.npz", device)
        return {
            f"asrr_{name}": row["path"]
            for name, row in bank.items()
            if row["group_weight"] > 0
        }
    ordered, _, queries, qy = source_arrays(device)

    stem = (
        f"{factors}_rate{rate:g}"
        if method == "msr"
        else f"warm{int(warm)}_rate{rate:g}"
    )
    names = {
        step: f"{method}_{stem}_step{step}" for step in SOURCE["checkpoints"]
    }
    paths = {step: folder / f"{name}.npz" for step, name in names.items()}
    if not all(path.exists() for path in paths.values()):
        mixture = torch.cat(tuple(queries.values()), dim=1)
        data = ordered, mixture, qy.repeat(1, len(queries))
        if method == "msr":
            for step, basis in train_msr(
                data, LAYOUT, factors, RANK, rate, SHOT, device
            ):
                atomic_npz(paths[step], factor=basis)
                print(names[step], flush=True)
        else:
            dense = read_data(WORK / "dense_sources.npz")
            best = int(np.argmin(dense["ce"][:, 0]))
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
                print(names[step], flush=True)
    return {names[step]: str(path) for step, path in paths.items()}


def train(gpus):
    manifest = MODELS / "candidates.json"
    if manifest.exists():
        return json.loads(manifest.read_text())
    fit_source_dense(gpus)
    methods = [
        ("msr", factors, rate, False)
        for factors, rate in product((1, LAYOUT.sites), SOURCE["msr_rates"])
    ]
    methods.extend(
        ("yeh_soft", 0, rate, warm)
        for rate, warm in zip(SOURCE["yeh_rates"], (True, False), strict=True)
    )
    methods.append(("asrr", 0, 0, False))
    candidates = {"dense": None}
    for start in range(0, len(methods), len(gpus)):
        batch = methods[start : start + len(gpus)]
        jobs = [(*spec, gpu) for spec, gpu in zip(batch, gpus)]
        for rows in parallel(train_job, jobs, len(jobs)):
            candidates.update(rows)
    freeze(manifest, candidates)
    return candidates


def development_job(job):
    shard, candidates, gpu = job
    path = WORK / f"validation_{shard:02d}.json"
    if path.exists():
        return json.loads(path.read_text())
    device = device_for(gpu)
    scores = evaluate_panel(
        WORK / "dev",
        LAYOUT,
        SHOT,
        candidates,
        device,
        phase="dev",
        rates=TARGET["rates"],
        steps=TARGET["steps"],
        replicates=TARGET["seed_replicates"],
    )
    rows = validation_rows(scores, None)
    save_json(path, rows)
    print(
        f"Validation shard {shard}: {len(candidates)} structures", flush=True
    )
    return rows


def selection(candidates, gpus):
    path = OUT / "choices.json"
    if path.exists():
        return json.loads(path.read_text())
    partitions = [
        dict(list(candidates.items())[i :: len(gpus)])
        for i in range(len(gpus))
    ]
    jobs = [
        (i, rows, gpu)
        for i, (rows, gpu) in enumerate(zip(partitions, gpus, strict=True))
        if rows
    ]
    scores = [
        row
        for shard in parallel(development_job, jobs, len(jobs))
        for row in shard
    ]
    choices = {
        method: min(
            (row for row in scores if row["candidate"].startswith(method)),
            key=ordering,
        )
        for method in METHODS
    }
    for policy in choices.values():
        model = candidates[policy["candidate"]]
        policy["model"] = model
        policy["sha256"] = digest(model) if model else None
    add_group_shuffles(choices, MODELS)
    save_json(OUT / "validation.json", scores)
    freeze(path=OUT / "choices.json", record=choices)
    print(
        "Frozen policies",
        {
            method: (row["candidate"], row["accuracy"])
            for method, row in choices.items()
            if method in METHODS
        },
        flush=True,
    )
    return choices


def add_group_shuffles(
    choices, directory, *, source_bank=MODELS / "asrr", structure=None
):
    bank = json.loads((source_bank / "bank.json").read_text())
    chosen = choices["asrr"]
    if structure is None:
        structure = chosen["candidate"].removeprefix("asrr_")
    spec = bank[structure]
    groups = read_data(bank[spec["groups"]]["path"])["factor"]
    guide = read_data(bank[spec["guide"]]["path"])["factor"]
    guide = np.repeat(guide, len(groups) // len(guide), axis=0)
    chosen.update(
        {
            key: spec[key]
            for key in ("moment_weight", "gradient_weight", "group_weight")
        }
    )
    for i in range(TARGET["shuffle_replicates"]):
        rng = np.random.default_rng(SEED + 621_173 + i)
        path = directory / f"shuffle_{i}.npz"
        bases = [
            spectral_basis(g[rng.permutation(len(g))], u, spec["group_weight"])
            for g, u in zip(groups, guide, strict=True)
        ]
        atomic_npz(path, factor=np.asarray(bases, dtype=np.float32))
        choices[f"shuffle_{i}"] = {
            **chosen,
            "model": str(path),
            "sha256": digest(path),
        }


def test_job(job):
    method, policy, gpu, work = job
    path = work / f"test_{method}.npz"
    if path.exists():
        return path
    device = device_for(gpu)
    model = policy["model"]
    if model and digest(model) != policy["sha256"]:
        raise ValueError(f"Changed frozen representation for {method}")
    scores = evaluate_panel(
        work / "test",
        LAYOUT,
        SHOT,
        {method: model},
        device,
        phase="test",
        rates=(policy["rate"],),
        steps=(policy["steps"],),
        replicates=TARGET["seed_replicates"],
    )
    atomic_npz(path, **next(iter(scores.values())))
    print(f"Scored {method}", flush=True)
    return path


def confirm(choices, gpus, *, work=WORK, output=OUT, seed=SEED):
    prepare("test", gpus, work=work, output=output, seed=seed)
    predictions = {}
    items = list(choices.items())
    for start in range(0, len(items), len(gpus)):
        batch = items[start : start + len(gpus)]
        paths = parallel(
            test_job,
            [(*row, gpu, work) for row, gpu in zip(batch, gpus)],
            len(batch),
        )
        predictions.update(
            {
                method: read_data(path)
                for (method, _), path in zip(batch, paths, strict=True)
            }
        )
    shuffles = [
        rows
        for name, rows in predictions.items()
        if name.startswith("shuffle_")
    ]
    predictions["shuffled"] = {
        view: np.mean([row[view] for row in shuffles], axis=0)
        for view in predictions["asrr"]
    }
    summary = {}
    for view in ("clean", "query"):
        rows = {
            name: scores[view]
            for name, scores in predictions.items()
            if not name.startswith("shuffle_")
        }
        summary[view] = {
            "accuracy": {
                name: confidence(scores[:, 0]) for name, scores in rows.items()
            },
            "asrr_minus_method": {
                name: confidence(rows["asrr"][:, 0] - scores[:, 0])
                for name, scores in rows.items()
                if name != "asrr"
            },
            "mean_ce": {
                name: float(scores[:, 1].mean())
                for name, scores in rows.items()
            },
        }
    atomic_npz(
        output / "per_episode.npz",
        **{
            f"{method}_{view}": scores
            for method, row in predictions.items()
            for view, scores in row.items()
        },
    )
    manifest = work / "test/manifest.json"
    result = {
        "episodes": PLAN["test"]["episodes"],
        "support_per_class": SHOT,
        "coefficients": CODES,
        "test_panel_sha256": digest(manifest),
        "queries": summary,
    }
    save_json(output / "results.json", result)
    print(result, flush=True)


def run(gpus):
    if not gpus or len(gpus) != len(set(gpus)) or CODES % LAYOUT.hidden:
        raise ValueError(
            "Use distinct CUDA devices and a budget divisible by the hidden width"
        )
    MODELS.mkdir(parents=True, exist_ok=True)
    for phase in ("source", "dev"):
        prepare(phase, gpus)
    choices = selection(train(gpus), gpus)
    if not (OUT / "results.json").exists():
        confirm(choices, gpus)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    run(parser.parse_args().gpus)
