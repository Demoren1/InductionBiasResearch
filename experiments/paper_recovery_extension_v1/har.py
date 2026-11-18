"""Prepare a subject-disjoint, six-way UCI HAR transfer task."""

import json
import zipfile
from pathlib import Path

import numpy as np

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json

from .prepare import GRID, OUTPUT, ROOT

RAW = ROOT / "raw/uci_har.zip"


def read_split(name):
    with zipfile.ZipFile(RAW) as archive:
        arrays = []
        for prefix, dtype in (("X", np.float32), ("y", int), ("subject", int)):
            path = f"UCI HAR Dataset/{name}/{prefix}_{name}.txt"
            # The official UCI archive contains a nested, unchanged dataset zip.
            if path not in archive.namelist():
                import io

                nested = next(n for n in archive.namelist() if n.endswith(".zip"))
                with zipfile.ZipFile(io.BytesIO(archive.read(nested))) as inner:
                    arrays.append(np.loadtxt(inner.open(path), dtype=dtype))
            else:
                arrays.append(np.loadtxt(archive.open(path), dtype=dtype))
    x, y, subject = arrays
    return x * 255, y - 1, subject


def class_pools(labels, subjects, people):
    # Original windows overlap by 50%; retain every second window within
    # each subject/activity before assigning disjoint observation panels.
    return [
        np.concatenate(
            [np.flatnonzero((subjects == p) & (labels == c))[::2] for p in people]
        )
        for c in range(6)
    ]


def episodes(x, y, subject, people, count, seed):
    rng = np.random.default_rng(seed)
    arrays = {k: [] for k in ("support_x", "support_y", "query_x", "query_y")}
    ids = []
    for episode in range(count):
        person = int(people[episode % len(people)])
        pools = class_pools(y, subject, [person])
        if min(map(len, pools)) < 8:
            raise RuntimeError(
                f"Subject {person} has fewer than eight nonoverlapping windows/class"
            )
        selected = [rng.permutation(pool)[:8] for pool in pools]
        support = np.concatenate([a[:3] for a in selected])
        query = np.concatenate([a[3:] for a in selected])
        for key, value in zip(
            arrays,
            (
                x[support],
                np.eye(6, dtype=np.float32)[y[support]],
                x[query],
                np.eye(6, dtype=np.float32)[y[query]],
            ),
        ):
            arrays[key].append(value)
        ids.append(
            {
                "person": person,
                "support_ids": support.tolist(),
                "query_ids": query.tolist(),
            }
        )
    return {k: np.stack(v) for k, v in arrays.items()}, ids


def prepare():
    root, output = ROOT / "har", OUTPUT / "har"
    root.mkdir()
    output.mkdir()
    for name in (
        "public",
        "source",
        "fits",
        "validation",
        "validation_fit",
        "private",
        "targets",
    ):
        (root / name).mkdir()
    x, y, subject = read_split("train")
    people = np.unique(subject)
    groups = people[:16].reshape(4, 4)
    development = people[16:]
    rng = np.random.default_rng(2611141123)
    pools = [
        [rng.permutation(pool) for pool in class_pools(y, subject, group)]
        for group in groups
    ]
    smallest = min(len(pool) for group in pools for pool in group)
    train_count, select_count = int(smallest * 0.70), int(smallest * 0.15)
    source_ids = {}
    for name, start, stop in (
        ("support", 0, train_count),
        ("numerical", train_count, train_count + select_count),
        ("structure", train_count + select_count, train_count + 2 * select_count),
    ):
        ids = np.stack(
            [np.concatenate([pool[start:stop] for pool in group]) for group in pools]
        )
        np.savez_compressed(
            root / "public" / f"{name}.npz",
            x=x[ids],
            y=np.eye(6, dtype=np.float32)[y[ids]],
        )
        source_ids[name] = ids.tolist()
    panel, episode_ids = episodes(x, y, subject, development, 20, 2611141124)
    np.savez_compressed(root / "validation/validation.npz", **panel)
    protocol = {
        "dataset": "har",
        "inputs": 561,
        "hidden": 16,
        "outputs": 6,
        "loss": "cross_entropy",
        "source_tasks": 4,
        "validation_tasks": 20,
        "target_tasks": 90,
        "support_count": 18,
        "query_count": 30,
        "source_counts": [6 * train_count, 6 * select_count, 6 * select_count],
        "source_subject_groups": groups.tolist(),
        "development_subjects": development.tolist(),
        "target_subjects": "All nine official test subjects, loaded only after model freeze",
        "independent_code_grid": list(GRID),
        "extension_code_grid": list(GRID),
        "input_scaling": "Official 561 features unchanged; stored times255 for common loader",
        "temporal_correlation": "Every second window per subject/activity retained before random partition; remaining subject dependence is not treated as independent environments",
        "archive_sha256": digest(RAW),
        "source_sha256": {n: digest(root / "public" / f"{n}.npz") for n in source_ids},
        "preparation_code_sha256": digest(Path(__file__)),
        "target_draw_seed": 2611141125,
    }
    save_json(root / "protocol.json", protocol)
    save_json(output / "registration.json", protocol)
    save_json(
        output / "observation_ids.json",
        {"source": source_ids, "development": episode_ids},
    )
    print(json.dumps(protocol), flush=True)


if __name__ == "__main__":
    prepare()
