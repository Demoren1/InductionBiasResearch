"""Freeze nested MNIST source training sets and a common held-out panel."""

import argparse
import itertools
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from torchvision.datasets import MNIST

from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

COUNTS = (5, 50, 500, 4000)
CLASSES = ((0, 1, 2, 3, 4), (5, 6, 7, 8, 9), (0, 2, 4, 6, 8))
SNAPSHOTS = (0, 25, 100, 500, 1000, 1500, 2000, 2250, 2500)


def run(root, raw, output):
    root.mkdir(parents=True, exist_ok=False)
    (root / "public").mkdir()
    (root / "fits").mkdir()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "code_snapshot"
    snapshot.mkdir()
    for path in Path(__file__).parent.iterdir():
        if path.is_file():
            shutil.copy2(path, snapshot / path.name)
    data = MNIST(str(raw), train=True, download=False)
    images = data.data.numpy().reshape(-1, 784)
    labels = data.targets.numpy()
    supports = {count: {"x": [], "y": [], "ids": []} for count in COUNTS}
    heldout = {"x": [], "y": [], "ids": []}
    for task, classes in enumerate(CLASSES):
        rng = np.random.default_rng(26111473013 + task)
        class_ids = [rng.permutation(np.flatnonzero(labels == c)) for c in classes]
        panels = [(heldout, np.concatenate([ids[:500] for ids in class_ids]))]
        for count in COUNTS:
            panels.append(
                (
                    supports[count],
                    np.concatenate([ids[500 : 500 + count] for ids in class_ids]),
                )
            )
        for panel, ids in panels:
            ids = ids[rng.permutation(len(ids))]
            local_labels = np.array([classes.index(int(y)) for y in labels[ids]])
            panel["x"].append(images[ids])
            panel["y"].append(np.eye(5, dtype=np.float32)[local_labels])
            panel["ids"].append(ids)
    for name, panel in [("H", heldout)] + [
        (f"support_{count}", supports[count]) for count in COUNTS
    ]:
        np.savez_compressed(
            root / "public" / f"{name}.npz",
            **{key: np.stack(value) for key, value in panel.items()},
        )
    settings = [
        {"lr": lr, "weight_decay": decay, "restart": restart}
        for lr, decay, restart in itertools.product(
            (0.001, 0.003, 0.01), (0.0, 1e-4), range(2)
        )
    ]
    protocol = {
        "created_utc": datetime.now(UTC).isoformat(),
        "dataset": "MNIST, official training split only",
        "source_class_sets": CLASSES,
        "training_examples_per_class": COUNTS,
        "H_examples_per_class": 500,
        "inputs": 784,
        "hidden": 16,
        "outputs": 5,
        "settings": settings,
        "steps": 2500,
        "checkpoint_every": 25,
        "snapshots": SNAPSHOTS,
        "initialization_seed": 753001,
        "masks": ["full", "variable_pixels", "variable_top50"],
        "variable_pixel_H_std_threshold": 0.02,
        "budgets": [8, 16, 32, 64],
        "target_access": False,
        "new_transfer_evaluation": False,
        "code_sha256": {
            str(path): digest(path)
            for path in Path(__file__).parent.iterdir()
            if path.is_file()
        },
        "external_code_sha256": {
            str(path): digest(path)
            for path in (
                Path("experiments/constructive_dense_recovery_v1/recovery.py"),
                Path("experiments/omniglot_constructive_v1/run.py"),
                Path("experiments/paper_structure_benchmark_v1/fitting/real_dense.py"),
            )
        },
        "source_data_sha256": {
            str(path.relative_to(root)): digest(path)
            for path in sorted((root / "public").glob("*.npz"))
        },
        "raw_training_sha256": {
            path.name: digest(path)
            for path in sorted((raw / "MNIST/raw").glob("train-*-ubyte"))
        },
    }
    save_json(root / "protocol.json", protocol)
    save_json(output / "registration.json", protocol)
    print(json.dumps({"registered": str(output), "training_split_only": True}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.raw.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
