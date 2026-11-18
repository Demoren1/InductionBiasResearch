"""Warm-start each source from one of the ten fitted Dense first layers."""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from experiments.paper_recovery_extension_v1.source import run as fit_source
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

from . import ANCHORS, DENSE, OUTPUT, PARENT, ROOT


def prepare():
    OUTPUT.mkdir(parents=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    protocol_path = PARENT / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if len(protocol["source_pairs"]) != len(ANCHORS):
        raise ValueError("Expected the ten MNIST source pairs")
    panels = {
        name: digest(PARENT / "public" / name)
        for name in ("support.npz", "numerical.npz")
    }
    for key, value in panels.items():
        if value != protocol["source_sha256"][key.removesuffix(".npz")]:
            raise ValueError(f"Changed source observations: {key}")
    quality = json.loads((DENSE.parent / "polish.json").read_text())
    if digest(DENSE) != quality["models_sha256"]:
        raise ValueError("Changed independent source models")
    save_json(
        OUTPUT / "registration.json",
        {
            "source_pairs": protocol["source_pairs"],
            "source_panels_sha256": panels,
            "source_protocol_sha256": digest(protocol_path),
            "independent_dense_sha256": digest(DENSE),
            "initialization": "Copy fitted first-layer weights and biases only",
            "anchor_unchanged": True,
            "other_sources": "Fresh heads, all layer parameters trainable",
            "optimizer": "Existing 12-start Adam and four L-BFGS fits",
        },
    )


def worker(anchor, device):
    started = time.perf_counter()
    name = f"anchor_{anchor:02d}"
    root = ROOT / "source_fits" / name
    output = OUTPUT / "source_fits" / name
    root.mkdir(parents=True)
    output.mkdir(parents=True)
    (root / "public").mkdir()
    (root / "source").mkdir()
    registration = json.loads((OUTPUT / "registration.json").read_text())
    if digest(DENSE) != registration["independent_dense_sha256"]:
        raise ValueError("Changed anchor archive")
    protocol = json.loads((PARENT / "protocol.json").read_text())
    other = [task for task in ANCHORS if task != anchor]
    protocol["source_pairs"] = [
        protocol["source_pairs"][task] for task in other
    ]
    protocol["source_tasks"] = len(other)
    save_json(root / "protocol.json", protocol)
    for name, expected in registration["source_panels_sha256"].items():
        path = PARENT / "public" / name
        if digest(path) != expected:
            raise ValueError(f"Changed source observations: {path}")
        with np.load(path) as panel:
            np.savez(
                root / "public" / name,
                x=panel["x"][other],
                y=panel["y"][other],
            )

    with np.load(DENSE) as archive:
        original = {key: archive[key] for key in archive.files}
    width = (protocol["inputs"] + 1) * protocol["hidden"]
    initial = original["parameters"][anchor, :width].copy()
    np.save(output / "initial_hidden.npy", initial)
    print(json.dumps({"anchor": anchor, "stage": "fit_sources"}), flush=True)
    fit_source(root, output, device, initial_hidden=initial)

    checkpoint = root / "source/dense/models.npz"
    with np.load(checkpoint) as adapted:
        for key in ("parameters", "ensemble_parameters"):
            original[key][other] = adapted[key]
        for key in ("numerical_loss", "numerical_cross_entropy"):
            original[key][:, other] = adapted[key]
    combined = output / "models.npz"
    np.savez_compressed(combined, **original)
    save_json(
        output / "complete.json",
        {
            "anchor": anchor,
            "class_pair": registration["source_pairs"][anchor],
            "adapted_tasks": other,
            "hidden_sha256": digest(output / "initial_hidden.npy"),
            "models_sha256": digest(combined),
            "seconds": time.perf_counter() - started,
        },
    )


def run(gpus, *, anchors=ANCHORS, output=OUTPUT, module=None):
    logs = output / "source_logs"
    logs.mkdir(exist_ok=True)

    def fit_partition(slot):
        for anchor in anchors[slot :: len(gpus)]:
            with (logs / f"anchor_{anchor:02d}.log").open("w") as stream:
                subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        module or __spec__.name,
                        "worker",
                        "--anchor",
                        str(anchor),
                        "--device",
                        f"cuda:{gpus[slot]}",
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            print(
                json.dumps({"anchor": anchor, "status": "complete"}),
                flush=True,
            )

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(fit_partition, range(len(gpus))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "run", "worker"))
    parser.add_argument("--anchor", type=int, choices=ANCHORS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
    elif args.phase == "run":
        run(args.gpus)
    else:
        worker(args.anchor, torch.device(args.device))
