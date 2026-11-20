"""Register shared real-data episodes before fitting numerical parameters."""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np

from experiments.asrr_benchmark_families_v1.groups import LOCAL as FAMILY_DATA
from experiments.asrr_benchmark_families_v1.linear import atomic_npz
from experiments.asrr_benchmark_families_v1.panels import batches
from experiments.real_group_transfer_v1.data import (
    FC100_SUPERCLASSES,
    fc100_panel,
)
from experiments.real_group_transfer_v1.study import DATA as FC100_DATA
from experiments.real_group_transfer_v1.study import SEED as FC100_SOURCE_SEED
from experiments.sharing_structure_research_v1.blind_contract import digest

from .common import CONFIG, OUT, SEED, WORK, freeze, register

MNIST = Path(
    "artifacts/local/mnist_dense_sample_size_2026-11-14/raw/MNIST/raw"
)


def panel_directory(dataset):
    return (
        FAMILY_DATA if dataset in ("aug_omniglot", "wisdm") else WORK
    ) / dataset


def episodes(panel, outputs, shot):
    return {
        "sx": panel.support_x.reshape(
            -1, outputs, shot, *panel.support_x.shape[-2:]
        ),
        "sy": panel.support_y.reshape(-1, outputs, shot),
        "qx": panel.query_x,
        "qy": panel.query_y,
        "units": panel.units,
    }


def prepare_fc100(folder):
    path = FC100_DATA / "fc100.npz"
    with np.load(path, allow_pickle=False) as data:
        source = fc100_panel(data, "source", 16, FC100_SOURCE_SEED + 1)
        atomic_npz(
            folder / "source.npz",
            sx=source.support_x,
            sy=source.support_y,
            qx=source.query_x,
            qy=source.query_y,
            units=source.units,
        )
        atomic_npz(folder / "source_images.npz", records=source.records)
        for offset, phase in enumerate(("dev", "test")):
            partition = "development" if phase == "dev" else phase
            panel = fc100_panel(
                data,
                partition,
                CONFIG["fc100"][partition],
                SEED + 179 + offset,
            )
            batches(
                folder,
                phase,
                episodes(panel, 5, 5),
                CONFIG["target"]["batch_size"],
            )
            atomic_npz(folder / f"{phase}_images.npz", records=panel.records)
    return {
        "cache_sha256": digest(path),
        "superclasses": {
            key: list(values) for key, values in FC100_SUPERCLASSES.items()
        },
    }


def read_idx(split):
    images = np.memmap(
        MNIST / f"{split}-images-idx3-ubyte",
        dtype=np.uint8,
        mode="r",
        offset=16,
    )
    labels = np.memmap(
        MNIST / f"{split}-labels-idx1-ubyte",
        dtype=np.uint8,
        mode="r",
        offset=8,
    )
    return images.reshape(len(labels), 28, 28), labels


def balanced_pairs(rng):
    circle = rng.permutation(10).tolist()
    rounds = []
    for _ in range(9):
        rounds.append(
            [tuple(sorted(pair)) for pair in zip(circle[:5], circle[5:][::-1])]
        )
        circle = [circle[0], circle[-1], *circle[1:-1]]
    order = rng.permutation(len(rounds))
    pairs = [
        [p for i in ids for p in rounds[i]] for ids in np.split(order, [5, 7])
    ]
    if set(p for group in pairs for p in group) != set(
        combinations(range(10), 2)
    ):
        raise ValueError("The digit pair partition is incomplete")
    return pairs


def torus(images, ids, rng, permutation):
    size = len(ids)
    row = (np.arange(28)[None, :] + rng.integers(28, size=(size, 1))) % 28
    col = (np.arange(28)[None, :] + rng.integers(28, size=(size, 1))) % 28
    moved = images[ids[:, None, None], row[:, :, None], col[:, None, :]]
    return moved.reshape(size, 1, 784)[..., permutation]


def prepare_mnist(folder):
    config = CONFIG["mnist"]
    rng = np.random.default_rng(SEED + 295)
    train_x, train_y = read_idx("train")
    test_x, test_y = read_idx("t10k")
    pairs = balanced_pairs(rng)
    permutation = rng.permutation(784)
    images = [
        rng.permutation(np.flatnonzero(train_y == digit))
        for digit in range(10)
    ]
    pools = (
        [rows[: int(len(rows) * 0.9)] for rows in images],
        [rows[int(len(rows) * 0.9) :] for rows in images],
        [np.flatnonzero(test_y == digit) for digit in range(10)],
    )
    for phase_id, phase in enumerate(("source", "dev", "test")):
        source = phase == "source"
        shot = (
            config["source_per_class"]
            if source
            else CONFIG["real"]["mnist_torus"]["shot"]
        )
        query = (
            config["numerical_per_class"]
            if source
            else config["query_per_class"]
        )
        repeats = (
            1
            if source
            else config[
                f"{'development' if phase == 'dev' else 'test'}_per_pair"
            ]
        )
        rows = {"sx": [], "sy": [], "qx": [], "qy": [], "units": []}
        records = []
        pixels = test_x if phase == "test" else train_x
        for pair_id, pair in enumerate(pairs[phase_id]):
            for _ in range(repeats):
                labels = rng.permutation(pair)
                ids = np.stack(
                    [
                        rng.choice(
                            pools[phase_id][digit], shot + query, replace=False
                        )
                        for digit in labels
                    ]
                )
                sx = torus(pixels, ids[:, :shot].ravel(), rng, permutation)
                qx = torus(pixels, ids[:, shot:].ravel(), rng, permutation)
                sy = np.repeat(np.arange(2), shot)
                if not source:
                    sx, sy = sx.reshape(2, shot, 1, 784), sy.reshape(2, shot)
                for name, arr in (
                    ("sx", sx),
                    ("sy", sy),
                    ("qx", qx),
                    ("qy", np.repeat(np.arange(2), query)),
                    ("units", pair_id),
                ):
                    rows[name].append(arr)
                records.append(ids)
        panel = {key: np.stack(values) for key, values in rows.items()}
        if source:
            atomic_npz(folder / "source.npz", **panel)
        else:
            batches(folder, phase, panel, CONFIG["target"]["batch_size"])
        atomic_npz(folder / f"{phase}_images.npz", records=np.stack(records))
    return {
        "digit_pairs": {
            phase: [list(p) for p in group]
            for phase, group in zip(
                ("source", "dev", "test"), pairs, strict=True
            )
        },
        "permutation_after_circular_shift": permutation.tolist(),
        "draw_seed": SEED + 295,
        "files": {
            path.name: digest(path)
            for path in sorted(MNIST.iterdir())
            if path.suffix != ".gz"
        },
        "original_images": "Stratified 90/10 split of original train and official test",
    }


def prepare(dataset):
    folder = panel_directory(dataset)
    output = OUT / dataset
    folder.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "panels.json"
    if manifest.exists():
        return
    provenance = {}
    if dataset == "mnist_torus":
        provenance = prepare_mnist(folder)
    elif dataset == "fc100":
        provenance = prepare_fc100(folder)
    hashes = {}
    for phase in ("dev", "test"):
        files = sorted((folder / phase).glob("*.npz"))
        if not files or not (folder / phase / "answers.npz").exists():
            raise FileNotFoundError(folder / phase)
        hashes[phase] = {path.name: digest(path) for path in files}
    freeze(
        manifest,
        {
            "source_sha256": digest(folder / "source.npz"),
            "panels": str(folder),
            "files": hashes,
            "provenance": provenance,
        },
    )
    print(dataset, folder, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", choices=CONFIG["real"], nargs="+")
    args = parser.parse_args()
    register()
    for name in args.datasets:
        prepare(name)
