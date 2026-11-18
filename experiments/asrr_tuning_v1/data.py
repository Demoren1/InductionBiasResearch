"""Class- and subject-disjoint development and target task draws."""

import itertools
from pathlib import Path

import numpy as np
from torchvision.datasets import MNIST, FashionMNIST

from experiments.omniglot_prior_v1.run import episodes as omniglot_episodes
from experiments.paper_recovery_extension_v1 import har

SETTINGS = {
    "mnist100": ("mnist", 100, 50, 2),
    "mnist16": ("mnist", 16, 0, 2),
    "fashion100": ("fashion", 100, 50, 2),
    "fashion16": ("fashion", 16, 0, 2),
    "har3": ("har", 3, 0, 6),
    "omniglot10": ("omniglot", 10, 5, 5),
}


def episodes(setting, split, count, seed):
    dataset, shots, validation, outputs = SETTINGS[setting]
    if dataset == "omniglot":
        return omniglot_episodes(
            "val" if split == "development" else "test", count, seed
        )
    if dataset == "har":
        har.RAW = Path("artifacts/local/paper_asrr_2026-11-17/raw/uci_har.zip")
        x, y, subjects = har.read_split(
            "train" if split == "development" else "test"
        )
        people = (
            np.unique(subjects)[16:]
            if split == "development"
            else np.unique(subjects)
        )
        raw, provenance = har.episodes(x, y, subjects, people, count, seed)
        return {
            f"{part}_{kind}": raw[f"{old}_{kind}"] / 255
            if kind == "x"
            else raw[f"{old}_{kind}"]
            for part, old in (("train", "support"), ("query", "query"))
            for kind in ("x", "y")
        }, provenance
    raw = (
        Path("artifacts/local/mnist_dense_sample_size_2026-11-14/raw")
        if dataset == "mnist"
        else Path("artifacts/local/paper_recovery_comparison_2026-11-14/raw")
    )
    loader = MNIST if dataset == "mnist" else FashionMNIST
    train, test = [
        loader(str(raw), train=flag, download=False) for flag in (True, False)
    ]
    x, qx = [
        d.data.numpy().reshape(-1, 784).astype(np.float32) / 255
        for d in (train, test)
    ]
    y, qy = train.targets.numpy(), test.targets.numpy()
    pairs = (
        ((5, 6),)
        if split == "development"
        else tuple(itertools.combinations((7, 8, 9), 2))
    )
    rng = np.random.default_rng(seed)
    parts = ("train", "valid", "query") if validation else ("train", "query")
    panels = {f"{p}_{k}": [] for p in parts for k in ("x", "y")}
    provenance = []
    for episode in range(count):
        classes = pairs[episode % len(pairs)]
        selected = [
            rng.choice(
                np.flatnonzero(y == c), shots + validation, replace=False
            )
            for c in classes
        ]
        query_on_train = split == "development" and validation == 0
        query_labels, query_data = (y, x) if query_on_train else (qy, qx)
        query = [
            rng.choice(
                np.setdiff1d(np.flatnonzero(query_labels == c), ids)
                if query_on_train
                else np.flatnonzero(query_labels == c),
                100,
                replace=False,
            )
            for c, ids in zip(classes, selected)
        ]
        indices = {"train": [a[:shots] for a in selected], "query": query}
        if validation:
            indices["valid"] = [a[shots:] for a in selected]
        for part in parts:
            ids = indices[part]
            panels[f"{part}_x"].append(
                (query_data if part == "query" else x)[np.concatenate(ids)]
            )
            panels[f"{part}_y"].append(
                np.repeat(
                    np.eye(outputs, dtype=np.float32), len(ids[0]), axis=0
                )
            )
        provenance.append(
            {
                "classes": classes,
                "train_valid_ids": [a.tolist() for a in selected],
                "query_ids": [a.tolist() for a in query],
            }
        )
    return {k: np.stack(v) for k, v in panels.items()}, provenance
