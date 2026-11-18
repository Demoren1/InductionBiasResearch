"""Build one Omniglot aSRR indexator from each source initialization."""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from experiments.paper_structure_benchmark_v1.fitting.real_dense import (
    run as fit_dense,
)
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)
from experiments.srr_approx_v1.core import input_forest

from .source import run as run_sources


OUTPUT = Path("artifacts/asrr_anchor_omniglot_2026-11-18")
ROOT = OUTPUT.parent / "local" / OUTPUT.name
SOURCE = ROOT.parent / "omniglot_single_task_2026-11-14" / "study"
DENSE = SOURCE / "source" / "dense"
PRIOR = OUTPUT.parent / "asrr_tuning_2026-11-17"
PRIOR_PROFILES = (
    ROOT.parent / "srr_approx_real_2026-11-15/omniglot/source_profiles.npz"
)
BANKS = ROOT / "banks/omniglot/models"
SETTING = "omniglot10"
ANCHORS = tuple(range(16))
INPUTS, HIDDEN, CLASSES = 784, 16, 5
BASELINE = "asrr_input_m1024"


def read_json(path):
    return json.loads(path.read_text())


def prepare():
    previous = PRIOR / "choices_frozen.json"
    rule = read_json(previous)["policies"][SETTING]["asrr_accuracy"]
    if rule["name"] != BASELINE or rule["key"] != "wide24_500_top4":
        raise ValueError("Unexpected historical Omniglot configuration")
    baseline = ROOT.parent / PRIOR.name / "banks/omniglot/models"
    baseline = baseline / f"{BASELINE}.npz"
    if digest(baseline) != rule["model_sha256"]:
        raise ValueError("Changed independently built indexator")

    source_models = DENSE / "models.npz"
    source_sha256 = digest(source_models)
    if source_sha256 != read_json(DENSE / "metrics.json")["models_sha256"]:
        raise ValueError("Changed source-trained Dense models")
    with np.load(source_models) as archive:
        h_loss = archive["numerical_loss"].min(axis=0)
        tasks = h_loss.argsort()[: len(ANCHORS)]
        parameters = archive["parameters"][tasks]
    with np.load(PRIOR_PROFILES) as archive:
        if not np.array_equal(tasks, archive["ranking"]):
            raise ValueError("Source tasks differ from the prior input bank")

    public = SOURCE / "public"
    settings = read_json(DENSE / "registered.json")
    panel_sha256 = settings["source_sha256"]
    for name, expected in panel_sha256.items():
        if digest(public / f"{name}.npz") != expected:
            raise ValueError(f"Changed source panel: {name}")

    OUTPUT.mkdir(parents=True)
    BANKS.mkdir(parents=True)
    panels = ROOT / "source_panels"
    panels.mkdir()
    np.savez_compressed(
        panels / "anchors.npz",
        parameters=parameters,
        h_loss=h_loss[tasks],
        task_ids=tasks,
    )
    for name in panel_sha256:
        with np.load(public / f"{name}.npz") as data:
            np.savez_compressed(
                panels / f"{name}.npz", x=data["x"][tasks], y=data["y"][tasks]
            )
    shutil.copyfile(baseline, BANKS / baseline.name)
    save_json(
        OUTPUT / "registration.json",
        {
            "source_tasks": tasks.tolist(),
            "original_source_models_sha256": source_sha256,
            "original_source_panels_sha256": panel_sha256,
            "selected_source_sha256": {
                path.stem: digest(path)
                for path in sorted(panels.glob("*.npz"))
            },
            "source_fit": {
                k: settings[k]
                for k in ("steps", "settings", "checkpoint_every")
            },
            "models_per_bank": len(ANCHORS),
            "neuron_correspondence": "inherited by initialization",
            "bank_family": "input",
            "bank_codes": rule["codes"],
            "previous_selection": str(previous),
            "previous_selection_sha256": digest(previous),
            "baseline": rule,
        },
    )
    print(f"Registered source tasks: {tasks.tolist()}")


def worker(anchor, device):
    started = time.perf_counter()
    name = f"anchor_{anchor:02d}"
    root = ROOT / "source_fits" / name
    output = OUTPUT / "source_fits" / name
    panels = ROOT / "source_panels"
    config = read_json(OUTPUT / "registration.json")
    for key, expected in config["selected_source_sha256"].items():
        if digest(panels / f"{key}.npz") != expected:
            raise ValueError(f"Changed selected source data: {key}")
    with np.load(panels / "anchors.npz") as archive:
        models = dict(archive)
    parameters = models["parameters"]
    initial = parameters[anchor, : (INPUTS + 1) * HIDDEN].copy()
    others = [task for task in ANCHORS if task != anchor]
    (root / "public").mkdir(parents=True)
    (root / "source").mkdir()
    output.mkdir(parents=True)
    np.save(root / "initial_hidden.npy", initial)
    save_json(
        root / "protocol.json",
        {
            "inputs": INPUTS,
            "hidden": HIDDEN,
            "outputs": CLASSES,
            "loss": "cross_entropy",
        },
    )
    for key in ("support", "numerical"):
        with np.load(panels / f"{key}.npz") as data:
            np.savez_compressed(
                root / "public" / f"{key}.npz",
                x=data["x"][others],
                y=data["y"][others],
            )
    fit_dense(
        root, device, config["source_fit"]["steps"], initial_hidden=initial
    )
    checkpoint = root / "source/dense/models.npz"
    with np.load(checkpoint) as archive:
        parameters[others] = archive["parameters"]
        models["h_loss"][others] = archive["numerical_loss"].min(axis=0)
    models_path = root / "bank_sources.npz"
    np.savez_compressed(models_path, **models)

    weights = parameters[:, : INPUTS * HIDDEN].reshape(-1, INPUTS, HIDDEN)
    normalized = weights / np.linalg.norm(weights, axis=1, keepdims=True).clip(
        min=1e-12
    )
    codes = config["bank_codes"]
    index = input_forest(normalized, (codes,))[codes]
    if not np.array_equal(np.unique(index), np.arange(1, codes + 1)):
        raise ValueError("Unexpected code count or inactive weights")
    index_path = BANKS / f"{name}.npz"
    np.savez_compressed(index_path, index=index)
    save_json(
        output / "complete.json",
        {
            "anchor": anchor,
            "anchor_task": int(models["task_ids"][anchor]),
            "adapted_tasks": models["task_ids"][others].tolist(),
            "models_per_bank": len(parameters),
            "hidden_sha256": digest(root / "initial_hidden.npy"),
            "bank_sources_sha256": digest(models_path),
            "index_sha256": digest(index_path),
            "seconds": time.perf_counter() - started,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "source", "worker"))
    parser.add_argument("--anchor", type=int, choices=ANCHORS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
    elif args.phase == "source":
        run_sources(
            args.gpus, anchors=ANCHORS, output=OUTPUT, module=__spec__.name
        )
    else:
        worker(args.anchor, torch.device(args.device))
