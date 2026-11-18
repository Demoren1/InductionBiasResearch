"""Prepare and time the fixed 4096-code Omniglot source constructions."""

import argparse
import resource
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from experiments.paper_asrr_v1.extract import source_profiles
from experiments.paper_structure_benchmark_v1.fitting.real_dense import (
    run as fit_dense,
)
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)
from experiments.srr_approx_v1.core import input_forest
from experiments.wsr_omniglot_v1.source import OUTPUT as PREVIOUS
from experiments.wsr_omniglot_v1.source import (
    PANELS,
    fit_groups,
    read_json,
)

OUTPUT = Path("artifacts/omniglot64_2026-11-18")
ROOT = OUTPUT.parent / "local" / OUTPUT.name
BANKS = ROOT / "banks/omniglot/models"
INPUTS, HIDDEN, CLASSES, CODES = 784, 64, 5, 4096
WEIGHTS = INPUTS * HIDDEN
VARIANT, KEY = "wide24_500", "wide24_500_top4"
NUMERIC_SHARD = 24
TIMING_EPISODES = 96


def prepare():
    original = read_json(PREVIOUS / "registration.json")
    previous_compact = OUTPUT.parent / "compact_experiments_2026-11-18"
    yeh_choice = read_json(previous_compact / "omniglot10/choices_frozen.json")[
        "yeh_soft"
    ]
    yeh_source = (
        ROOT.parent
        / previous_compact.name
        / "omniglot_source/source/yeh_k1024/registered.json"
    )
    yeh_original = read_json(yeh_source)
    yeh_settings = yeh_original["settings"][int(yeh_choice["name"].rsplit("_r", 1)[1])]
    for name, expected in original["source_panel_sha256"].items():
        if digest(PANELS / f"{name}.npz") != expected:
            raise ValueError(f"Changed source panel: {name}")
    with np.load(PANELS / "anchors.npz") as data:
        if data["task_ids"].tolist() != original["source_tasks"]:
            raise ValueError("Unexpected Omniglot source tasks")
    OUTPUT.mkdir(parents=True)
    BANKS.mkdir(parents=True)
    (ROOT / "public").mkdir()
    (ROOT / "source").mkdir()
    for name in ("support", "numerical"):
        shutil.copyfile(PANELS / f"{name}.npz", ROOT / "public" / f"{name}.npz")
    save_json(
        ROOT / "protocol.json",
        {
            "inputs": INPUTS,
            "hidden": HIDDEN,
            "outputs": CLASSES,
            "loss": "cross_entropy",
        },
    )
    save_json(
        OUTPUT / "registration.json",
        {
            "hidden": HIDDEN,
            "codes": CODES,
            "dense_weights": WEIGHTS,
            "source_tasks": original["source_tasks"],
            "source_panel_sha256": {
                k: original["source_panel_sha256"][k] for k in ("support", "numerical")
            },
            "dense_source_steps": 2500,
            "wsr_options": {**original["source_training"], "codes": CODES},
            "yeh_settings": yeh_settings,
            "yeh_outer_steps": yeh_original["steps"],
            "yeh_selection": "Configuration of the 16-hidden-neuron development winner",
            "validation_and_targets": str(PREVIOUS),
            "numeric_rule": original["asrr"]["rule"],
            "numeric_shard": NUMERIC_SHARD,
            "timing_episodes": TIMING_EPISODES,
            "timing_scope": "Fresh fitting and within-task checkpoint selection, excluding query prediction",
        },
    )


def save_groups(method, groups, started, device, *, codes=CODES):
    records = []
    for name, index, details in groups:
        if (
            index.shape != (INPUTS, HIDDEN)
            or np.any(index == 0)
            or not np.array_equal(np.unique(np.abs(index)), np.arange(1, codes + 1))
        ):
            raise ValueError(f"Wrong shape or code count: {name}")
        path = BANKS / f"{name}.npz"
        np.savez_compressed(path, index=index)
        records.append(
            {"name": name, "model_sha256": digest(path), "codes": codes, **details}
        )
    if device is not None:
        torch.cuda.synchronize(device)
    save_json(
        OUTPUT / f"{method}_source.json",
        {
            "candidates": records,
            "seconds": time.perf_counter() - started,
            "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device is not None
            else None,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
            if device is not None
            else None,
            "device": torch.cuda.get_device_name(device)
            if device is not None
            else "CPU",
        },
    )


def group(method, device):
    torch.set_num_threads(2)
    config = read_json(OUTPUT / "registration.json")
    dense = ROOT / "source/dense/models.npz"
    if digest(dense) != read_json(dense.with_name("metrics.json"))["models_sha256"]:
        raise ValueError("Changed fitted Dense source archive")
    if method == "asrr":
        started = time.perf_counter()
        _, profiles, _, _ = source_profiles(dense, INPUTS, HIDDEN)
        index = input_forest(profiles, (CODES,))[CODES]
        save_groups("asrr", [("asrr", index, {})], started, None)
        return

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    panels = []
    for name in ("support", "numerical"):
        with np.load(ROOT / "public" / f"{name}.npz") as data:
            x = torch.as_tensor(data["x"], device=device, dtype=torch.float32) / 255
            y = torch.as_tensor(data["y"], device=device, dtype=torch.float32)
        panels.append((x, y))
    with np.load(dense) as data:
        initial = data["parameters"]
    options = {**config["wsr_options"], "break_ties": True}
    save_json(OUTPUT / "wsr_source_options.json", options)
    fitted = fit_groups(
        initial,
        panels,
        (INPUTS, HIDDEN, CLASSES),
        options,
        "cross_entropy",
    )
    save_groups(
        "wsr",
        [
            (f"wsr_{task:03d}", index, {"source_task": task, **row})
            for task, (index, row) in zip(config["source_tasks"], fitted, strict=True)
        ],
        started,
        device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "dense", "asrr", "wsr"))
    parser.add_argument("--device", default="cuda:7")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "dense":
        config = read_json(OUTPUT / "registration.json")
        fit_dense(ROOT, torch.device(args.device), config["dense_source_steps"])
    else:
        group(args.action, torch.device(args.device))
