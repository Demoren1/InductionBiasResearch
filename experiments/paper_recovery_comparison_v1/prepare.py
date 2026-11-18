"""Prepare class-disjoint image-pair source and transfer observations."""

import argparse
import itertools
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from torchvision.datasets import MNIST, FashionMNIST

from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)


def packed(images, labels, ids, classes):
    target = (labels[ids] == classes[1]).astype(int)
    return images[ids], np.eye(2, dtype=np.float32)[target]


def run(root, raw, output, dataset):
    root.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("public", "private", "validation", "source", "fits", "targets"):
        (root / name).mkdir()
    (output / "code_snapshot").mkdir()
    for path in Path(__file__).parent.iterdir():
        if path.is_file():
            shutil.copy2(path, output / "code_snapshot" / path.name)
    loader = MNIST if dataset == "mnist" else FashionMNIST
    train, test = [
        loader(str(raw), train=flag, download=False) for flag in (True, False)
    ]
    images, test_images = [data.data.numpy().reshape(-1, 784) for data in (train, test)]
    labels, test_labels = train.targets.numpy(), test.targets.numpy()
    rng = np.random.default_rng(26111491013)
    source_ids = {c: rng.permutation(np.flatnonzero(labels == c)) for c in range(5)}
    pairs = list(itertools.combinations(range(5), 2))
    panels = {}
    for name, start, stop in (
        ("support", 0, 4000),
        ("numerical", 4000, 4500),
        ("structure", 4500, 5000),
    ):
        x, y = [], []
        for classes in pairs:
            ids = np.concatenate([source_ids[c][start:stop] for c in classes])
            ids = ids[rng.permutation(len(ids))]
            px, py = packed(images, labels, ids, classes)
            x.append(px)
            y.append(py)
        panels[name] = {"x": np.stack(x), "y": np.stack(y)}
        np.savez_compressed(root / "public" / f"{name}.npz", **panels[name])
    episode_pairs = {}
    for kind, count, classes in (("validation", 16, (5, 6)), ("target", 96, (7, 8, 9))):
        pair_grid = list(itertools.combinations(classes, 2))
        arrays = {key: [] for key in ("support_x", "support_y", "query_x", "query_y")}
        episode_pairs[kind] = []
        for episode in range(count):
            pair = pair_grid[episode % len(pair_grid)]
            support_ids, query_ids = [], []
            for c in pair:
                ids = rng.permutation(np.flatnonzero(labels == c))
                support_ids.extend(ids[:16])
                pool = (
                    ids[16:]
                    if kind == "validation"
                    else np.flatnonzero(test_labels == c)
                )
                query_ids.extend(rng.choice(pool, 100, replace=False))
            sx, sy = packed(images, labels, np.asarray(support_ids), pair)
            qx, qy = packed(
                images if kind == "validation" else test_images,
                labels if kind == "validation" else test_labels,
                np.asarray(query_ids),
                pair,
            )
            for key, value in zip(arrays, (sx, sy, qx, qy)):
                arrays[key].append(value)
            episode_pairs[kind].append(pair)
        directory = root / ("validation" if kind == "validation" else "private")
        np.savez_compressed(
            directory / f"{kind}.npz",
            **{key: np.stack(value) for key, value in arrays.items()},
        )
    protocol = {
        "created_utc": datetime.now(UTC).isoformat(),
        "dataset": dataset,
        "inputs": 784,
        "hidden": 16,
        "outputs": 2,
        "loss": "cross_entropy",
        "independent_code_grid": [8, 16, 32, 64],
        "source_pairs": pairs,
        "development_classes": [5, 6],
        "target_classes": [7, 8, 9],
        "source_tasks": 10,
        "validation_tasks": 16,
        "target_tasks": 96,
        "support_count": 32,
        "source_counts": [8000, 1000, 1000],
        "source_sha256": {
            name: digest(root / "public" / f"{name}.npz") for name in panels
        },
        "target_data_sha256": digest(root / "private/target.npz"),
        "protocol_sha256": digest(Path(__file__).with_name("PROTOCOL.md")),
        "code_sha256": {
            str(p): digest(p) for p in Path(__file__).parent.iterdir() if p.is_file()
        },
    }
    save_json(root / "protocol.json", protocol)
    save_json(output / "registration.json", protocol)
    save_json(root / "private/episode_pairs.json", episode_pairs)
    print(f"Prepared {dataset}: ten source pairs, class-disjoint development/target.")


def main():
    parser = argparse.ArgumentParser()
    for name in ("root", "raw", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--dataset", choices=("mnist", "fashion"), required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.raw.resolve(), args.output.resolve(), args.dataset)


if __name__ == "__main__":
    main()
