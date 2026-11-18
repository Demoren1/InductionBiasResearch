"""Run a matched, source-only indexator transfer comparison on real data."""

import argparse
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from itertools import repeat
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from experiments.asrr_anchor_init_v1.report import interval
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from experiments.srr_approx_v1.core import canonical, signed_forest
from experiments.srr_approx_v1.synthetic_random_controls import shuffle_groups

from .data import (
    FC100_SUPERCLASSES,
    NINAPRO_SUBJECTS,
    fc100_panel,
    ninapro_panel,
    ninapro_source_normalization,
)
from .optimization import (
    Layout,
    fit_competitor,
    fit_dense,
    fit_target,
    optimization_panel,
)

OUTPUT = Path("artifacts/fc100_ninapro_2026-11-18")
DATA = OUTPUT.parent / "local" / OUTPUT.name / "data"
LAYOUTS = {"fc100": Layout(16, 48, 8, 5), "ninapro": Layout(1, 48, 32, 12)}
CODES = {"fc100": 384, "ninapro": 128}
SEED = 7611189800
SOURCE_STEPS = {"dense": 800, "wsr": 500, "lutq": 1200}
TARGET_STEPS = 200
SHUFFLES = 4
COMPETITORS = ("wsr", "lutq")


def source_worker(gpu, panel, layout, codes):
    torch.set_num_threads(1)
    device = torch.device(f"cuda:{gpu}")
    tensors = optimization_panel(panel, device)
    dense, val, seconds = fit_dense(
        tensors,
        layout,
        steps=SOURCE_STEPS["dense"],
        restarts=3,
        device=device,
        seed=SEED,
    )
    print(f"GPU {gpu} Dense source {seconds:.2f}s", flush=True)
    fits = {"dense": (dense, val, seconds)}
    for method in COMPETITORS:
        fits[method] = fit_competitor(
            method,
            dense[:, 0],
            tensors,
            layout,
            codes,
            SOURCE_STEPS[method],
            device,
            SEED,
        )
        print(f"GPU {gpu} {method} source {fits[method][-1]:.2f}s", flush=True)
    return fits


def target_worker(gpu, panel, layout, indices, seed):
    torch.set_num_threads(1)
    device = torch.device(f"cuda:{gpu}")
    chunk_size = 48
    predictions = {name: [] for name in indices}
    ce = {name: [] for name in indices}
    seconds = {name: 0.0 for name in indices}
    memory = {name: 0 for name in indices}
    for start in range(0, len(panel.units), chunk_size):
        piece = panel.subset(slice(start, start + chunk_size))
        tensors = optimization_panel(piece, device)
        for name, index in indices.items():
            prediction, loss, elapsed, peak = fit_target(
                tensors, layout, index, device, steps=TARGET_STEPS, seed=seed
            )
            predictions[name].append(prediction)
            ce[name].append(loss)
            seconds[name] += elapsed
            memory[name] = max(memory[name], peak)
    return {
        name: (
            np.concatenate(predictions[name]),
            np.concatenate(ce[name]),
            seconds[name],
            memory[name],
        )
        for name in indices
    }


def align(reference, weights):
    matched = np.empty_like(weights)
    for site, weight in enumerate(weights):
        rows, columns = linear_sum_assignment(-reference[site].T @ weight)
        matched[site] = weight[:, columns[np.argsort(rows)]]
    return matched


def source_profiles(dense, layout):
    raw = (
        dense[..., : layout.connections]
        .reshape(*dense.shape[:2], layout.sites, layout.fan_in, layout.channels)
        .astype(float)
    )
    raw /= np.maximum(np.linalg.norm(raw, axis=-2, keepdims=True), 1e-12)
    anchor = raw[0, 0]
    means = np.stack(
        [
            align(anchor, np.mean([align(task[0], fit) for fit in task], axis=0))
            for task in raw
        ]
    )
    return means.transpose(0, 2, 1, 3).reshape(len(dense), layout.fan_in, layout.hidden)


def build_bank(pool, gpus, panel, layout, dataset):
    tasks = np.array_split(
        np.arange(len(panel.units)), min(len(gpus), len(panel.units))
    )
    codes = CODES[dataset]
    shards = list(
        pool.map(
            source_worker,
            gpus,
            (panel.subset(rows) for rows in tasks),
            repeat(layout),
            repeat(codes),
        )
    )
    dense = np.concatenate([shard["dense"][0] for shard in shards])
    started = time.perf_counter()
    profiles = source_profiles(dense, layout)
    cuts, _ = signed_forest(profiles, (codes,), device=f"cuda:{gpus[0]}")
    candidates = {"asrr": cuts[codes]}
    source_seconds = {
        method: max(shard[method][-1] for shard in shards)
        for method in ("dense", *COMPETITORS)
    }
    source_seconds["asrr"] = time.perf_counter() - started
    details = {}
    for method in COMPETITORS:
        weights = np.concatenate([shard[method][0] for shard in shards])
        losses = np.concatenate([shard[method][1] for shard in shards])
        steps = np.concatenate([shard[method][2] for shard in shards])
        for i, w in enumerate(weights):
            name = f"{method}_{i:02d}"
            index = w.reshape(layout.sites, layout.fan_in, layout.channels)
            candidates[name] = canonical(
                index.transpose(1, 0, 2).reshape(layout.fan_in, -1)
            )
            details[name] = {
                "unit": int(panel.units[i]),
                "source_ce": float(losses[i]),
                "step": int(steps[i]),
            }
    return candidates, details, source_seconds, dense


def evaluate(pool, gpus, panel, layout, indices, seed):
    shards = list(
        pool.map(
            target_worker,
            gpus,
            (
                panel.subset(rows)
                for rows in np.array_split(np.arange(len(panel.units)), len(gpus))
            ),
            repeat(layout),
            repeat(indices),
            repeat(seed),
        )
    )
    return {
        name: (
            np.concatenate([shard[name][0] for shard in shards]),
            np.concatenate([shard[name][1] for shard in shards]),
            max(shard[name][2] for shard in shards),
            max(shard[name][3] for shard in shards),
        )
        for name in indices
    }


def summarize(values, units):
    observations = np.array(
        [np.mean(values[units == unit]) for unit in np.unique(units)]
    )
    return {
        "mean": float(observations.mean()),
        "ci95_half_width": interval(observations),
        "units": len(observations),
    }


def same_offset_pairs(index):
    labels = np.abs(index)
    counts = np.bincount(labels.ravel())[1:]
    total = np.sum(counts * (counts - 1)) / 2
    if not total:
        return None
    shared = 0
    for row in labels:
        counts = np.bincount(row)[1:]
        shared += np.sum(counts * (counts - 1)) / 2
    return float(shared / total)


def run(dataset, gpus, output):
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists():
        raise FileExistsError(output / "results.json")
    layout = LAYOUTS[dataset]
    codes = CODES[dataset]
    if dataset == "fc100":
        with np.load(DATA / "fc100.npz", allow_pickle=False) as archive:
            data = dict(archive)
        source = fc100_panel(data, "source", 16, SEED + 1)
        development = fc100_panel(data, "development", 96, SEED + 2)
        split = FC100_SUPERCLASSES
    else:
        folder = DATA / "ninapro"
        mean, scale = ninapro_source_normalization(folder)
        source = ninapro_panel(folder, "source", mean, scale)
        development = ninapro_panel(folder, "development", mean, scale)
        split = {
            phase: subjects.tolist() for phase, subjects in NINAPRO_SUBJECTS.items()
        }
    save_json(
        output / "registration.json",
        {
            "dataset": dataset,
            "split": split,
            "layout": asdict(layout),
            "codes": codes,
            "seed": SEED,
            "source_steps": SOURCE_STEPS,
            "target_steps": TARGET_STEPS,
            "source_restarts": 3,
            "shuffles": SHUFFLES,
            "gpus": gpus,
            "selection": "Development mean accuracy with mean cross-entropy tiebreak",
            "target_size": 1024
            if dataset == "fc100"
            else len(NINAPRO_SUBJECTS["test"]) * 15,
        },
    )
    with ProcessPoolExecutor(
        max_workers=len(gpus), mp_context=mp.get_context("spawn")
    ) as pool:
        candidates, details, source_seconds, dense = build_bank(
            pool, gpus, source, layout, dataset
        )
        folder = DATA.parent / dataset
        folder.mkdir(exist_ok=True)
        np.savez_compressed(
            folder / "dense_fits.npz", parameters=dense, records=source.records
        )
        np.savez_compressed(output / "candidate_groups.npz", **candidates)
        dev = evaluate(pool, gpus, development, layout, candidates, SEED + 4)
        metrics = {
            name: {
                "accuracy": float(np.mean(value[0] == development.query_y)),
                "ce": float(value[1].mean()),
            }
            for name, value in dev.items()
        }
        selected = {"asrr": "asrr"}
        for method in COMPETITORS:
            names = [name for name in metrics if name.startswith(f"{method}_")]
            selected[method] = min(
                names,
                key=lambda name: (
                    -metrics[name]["accuracy"],
                    metrics[name]["ce"],
                    name,
                ),
            )
        indices = {method: candidates[name] for method, name in selected.items()}
        rng = np.random.default_rng(SEED + 5)
        for i in range(SHUFFLES):
            indices[f"shuffled_{i}"] = shuffle_groups(indices["asrr"], rng)
        if dataset == "fc100":
            kernel = np.arange(1, codes + 1, dtype=np.int16).reshape(
                layout.fan_in, layout.channels
            )
            indices["conv"] = np.tile(kernel, (1, layout.sites))
        frozen = output / "frozen_indexators.npz"
        np.savez_compressed(frozen, **indices)
        save_json(
            output / "selection.json",
            {
                "choices": selected,
                "metrics": metrics,
                "source": details,
                "indexators_sha256": digest(frozen),
            },
        )
        indices["dense"] = None
        print(f"{dataset} frozen {selected}; fitting held-out tasks", flush=True)
        if dataset == "fc100":
            target = fc100_panel(data, "test", 1024, SEED + 3)
        else:
            target = ninapro_panel(
                folder=DATA / "ninapro", phase="test", mean=mean, scale=scale
            )
        fits = evaluate(pool, gpus, target, layout, indices, SEED + 6)
    np.savez_compressed(
        output / "predictions.npz",
        answers=target.query_y,
        records=target.records,
        units=target.units,
        **{name: value[0] for name, value in fits.items()},
    )
    accuracies = {
        name: (value[0] == target.query_y).mean(-1) for name, value in fits.items()
    }
    rows = {}
    for name, index in indices.items():
        _, ce, seconds, memory = fits[name]
        rows[name] = {
            "accuracy": summarize(accuracies[name], target.units),
            "ce": float(ce.mean()),
            "codes": int(np.abs(index).max())
            if index is not None
            else layout.connections,
            "active": int(np.count_nonzero(index))
            if index is not None
            else layout.connections,
            "target_fit_seconds_max_shard": seconds,
            "target_peak_allocated_bytes": memory,
        }
        if dataset == "fc100" and index is not None:
            rows[name]["shared_pairs_same_offset_fraction"] = same_offset_pairs(index)
    accuracies["shuffled_mean"] = np.mean(
        [accuracies[f"shuffled_{i}"] for i in range(SHUFFLES)], axis=0
    )
    rows["shuffled_mean"] = {
        "accuracy": summarize(accuracies["shuffled_mean"], target.units)
    }
    paired = {
        name: summarize(accuracies["asrr"] - accuracies[name], target.units)
        for name in ("shuffled_mean", "dense", *COMPETITORS)
    }
    save_json(
        output / "results.json",
        {
            "methods": rows,
            "paired_asrr_minus": paired,
            "source_seconds_max_shard": source_seconds,
        },
    )
    for name, row in rows.items():
        result = row["accuracy"]
        print(
            f"{name:14s} {100 * result['mean']:.2f} ± {100 * result['ci95_half_width']:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=tuple(LAYOUTS))
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    args = parser.parse_args()
    run(args.dataset, args.gpus, OUTPUT / args.dataset)
