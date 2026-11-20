"""Prepare WISDM participant splits and matched clean/augmented Omniglot episodes."""

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.msr_reproduction_v1.images_data import augment_query
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json

from .groups import LOCAL, PLAN, PLAN_PATH, ROOT
from .linear import atomic_npz

OMNIGLOT = Path("artifacts/local/msr_reproduction_2026-11-14/images/omniglot")
WISDM_MIRROR = (
    "https://huggingface.co/datasets/mnemoraorg/"
    "smartphone-and-smartwatch-activity-and-biometrics-15m6/tree/c1fbf19/watch_raw/accel"
)


def batches(folder, phase, panels, batch):
    directory = folder / phase
    directory.mkdir(exist_ok=True)
    for i in range(0, len(panels["sx"]), batch):
        arrays = {
            k: v[i : i + batch] for k, v in panels.items() if k not in ("qy", "units")
        }
        np.savez(directory / f"inputs_{i:04d}.npz", **arrays)
    atomic_npz(directory / "answers.npz", qy=panels["qy"], units=panels["units"])


def stack_panels(entries):
    names = ("sx", "sy", "qx", "qy", "units")
    return {
        name: np.stack(x)
        for name, x in zip(names, zip(*entries, strict=True), strict=True)
    }


def windows(path, classes, size):
    rows = []
    malformed = 0
    subject = int(path.name.split("_")[1])
    for line in path.read_bytes().splitlines():
        fields = line.strip().rstrip(b";").split(b",")
        try:
            user, name, stamp, x, y, z = fields
            if int(user) != subject:
                raise ValueError(f"Participant mismatch in {path}")
            rows.append(
                (
                    classes.index(name.decode("ascii")),
                    int(stamp),
                    float(x),
                    float(y),
                    float(z),
                )
            )
        except (ValueError, UnicodeError):
            malformed += 1
    data = np.asarray(rows, dtype=np.float64)
    labels, timestamp = data[:, 0].astype(int), data[:, 1]
    step = np.diff(timestamp)
    splits = np.flatnonzero((np.diff(labels) != 0) | (step <= 0) | (step > 500e6)) + 1
    groups = [[] for _ in classes]
    for chunk in np.split(data, splits):
        time = chunk[:, 1] - chunk[0, 1]
        n = (int(time[-1] // 50_000_000) + 1) // size
        if n:
            grid = np.arange(n * size) * 50_000_000
            resampled = np.stack(
                [np.interp(grid, time, chunk[:, j]) for j in (2, 3, 4)], axis=1
            )
            features = np.c_[resampled, np.linalg.norm(resampled, axis=-1)]
            groups[int(chunk[0, 0])].append(
                features.astype(np.float32).reshape(n, size, 4)
            )
    result = [
        np.concatenate(group) if group else np.empty((0, size, 4), np.float32)
        for group in groups
    ]
    return result, malformed


def wisdm():
    cfg = PLAN["wisdm"]
    folder, out = LOCAL / "wisdm", ROOT / "wisdm"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "panel_registration.json").exists():
        return
    watch = folder / "watch"
    order = np.random.default_rng(cfg["seed"]).permutation(np.arange(1600, 1651))
    a, b, _ = cfg["subject_counts"]
    groups = np.split(order, [a, a + b])
    pools, meta, excluded = {}, {}, []
    for sub in order:
        data = watch / f"data_{sub}_accel_watch.txt"
        series, skipped = windows(data, cfg["classes"], cfg["window"])
        if min(map(len, series)) < 24:
            excluded.append(int(sub))
        else:
            pools[sub] = (
                [x[: int(len(x) * 0.4)] for x in series],
                [x[int(len(x) * 0.5) :] for x in series],
            )
        meta[str(sub)] = {
            "windows": list(map(len, series)),
            "sha256": digest(data),
            "skipped_rows": skipped,
        }
    assigned = {
        phase: [int(k) for k in ids if k in pools]
        for phase, ids in zip(("source", "dev", "test"), groups, strict=True)
    }
    if min(map(len, assigned.values())) < 4:
        raise ValueError("Insufficient participants in a predefined split")
    support = [x for sub in assigned["source"] for x in pools[sub][0]]
    values = np.concatenate(support, axis=0).reshape(-1, 4)
    center = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = values.std(axis=0, dtype=np.float64).astype(np.float32).clip(1e-6)
    for sub, phase_windows in pools.items():
        pools[sub] = tuple(
            [(x - center) / scale for x in part] for part in phase_windows
        )
    rng = np.random.default_rng(cfg["seed"] + 317)
    c = len(cfg["classes"])
    window, sites = cfg["window"], cfg["sites"]
    dim = window * 4 // sites

    def draw(part, count):
        rows = np.concatenate(
            [x[rng.choice(len(x), count, replace=False)] for x in part]
        )
        order = rng.permutation(len(rows))
        return rows.reshape(-1, sites, dim)[order], np.repeat(np.arange(c), count)[
            order
        ]

    for phase, subjects in assigned.items():
        if phase == "source":
            num_support = min(28, *(len(x) for s in subjects for x in pools[s][0]))
            num_query = min(15, *(len(x) for s in subjects for x in pools[s][1]))
            entries = [
                (*draw(pools[s][0], num_support), *draw(pools[s][1], num_query), s)
                for s in subjects
            ]
            np.savez(folder / "source.npz", **stack_panels(entries))
            continue
        entries, queries = [], 12
        repetitions = (
            cfg["development_per_subject"]
            if phase == "dev"
            else cfg["test_per_subject"]
        )
        max_shots = max(cfg["shots"])
        queries = min(queries, *(len(x) for s in subjects for x in pools[s][1]))
        for s in subjects:
            for _ in range(repetitions):
                # Keep support class-major so the smaller episode is a fixed prefix per class.
                fields = [
                    x[rng.choice(len(x), max_shots, replace=False)] for x in pools[s][0]
                ]
                sx = np.stack(fields).reshape(c, max_shots, sites, dim)
                qx, qy = draw(pools[s][1], queries)
                sy = np.broadcast_to(np.arange(c)[:, None], (c, max_shots))
                entries.append((sx, sy, qx, qy, s))
        batches(folder, phase, stack_panels(entries), cfg["numeric_batch"])
    save_json(
        out / "panel_registration.json",
        {
            "plan_sha256": digest(PLAN_PATH),
            "raw_files": meta,
            "raw_revision": WISDM_MIRROR,
            "resampling": "20 Hz linear interpolation within contiguous activities; split across timestamp gaps over 0.5 s",
            "window_features": "x,y,z,Euclidean magnitude; source-only channel means and scales",
            "center": center.tolist(),
            "scale": scale.tolist(),
            "splits": assigned,
            "excluded_lt24_windows_in_any_activity": excluded,
            "time_split": "first 40 percent support, final 50 percent query; nonoverlapping 64-reading windows",
            "source_samples_per_activity": [num_support, num_query],
            "classes": cfg["classes"],
        },
    )


def augmented(x, generator):
    shape = x.shape
    images = torch.from_numpy(x.copy()).reshape(-1, 1, 28, 28).float() / 255
    # Batching only limits memory. Draw each transformation from one recorded RNG.
    result = torch.cat([augment_query(part, generator) for part in images.split(1024)])
    return (result * 255).round().to(torch.uint8).numpy().reshape(shape)


def aug_omniglot():
    torch.set_num_threads(1)
    cfg, folder, out = (
        PLAN["aug_omniglot"],
        LOCAL / "aug_omniglot",
        ROOT / "aug_omniglot",
    )
    folder.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "panel_registration.json").exists():
        return
    files = {}
    for phase_no, (phase, split) in enumerate(
        zip(("source", "dev", "test"), ("train", "val", "test"), strict=True)
    ):
        path = OMNIGLOT / f"{split}.npy"
        data = np.load(path, mmap_mode="r")
        rng = np.random.default_rng(cfg["seed"] + 12313 * phase_no)
        generator = torch.Generator().manual_seed(cfg["seed"] + 7391 * phase_no)
        entries, provenance = [], []
        for episode in range(cfg[f"{phase}_tasks"]):
            classes = rng.choice(len(data), 5, replace=False)
            turns = rng.integers(4, size=5)
            drawings = np.stack([rng.permutation(20) for _ in classes])
            images = np.stack(
                [
                    np.rot90(data[i, permutation, 0], int(r), axes=(-2, -1))
                    for i, r, permutation in zip(classes, turns, drawings, strict=True)
                ]
            ).reshape(5, 20, 1, 784)
            if phase == "source":
                orig = images[:, :5]
                transformed = augmented(images[:, 5:], generator)
                sx = np.concatenate((orig, transformed[:, :10]), axis=1).reshape(
                    75, 1, 784
                )
                qx = transformed[:, 10:].reshape(25, 1, 784)
                sy, qy = np.repeat(np.arange(5), 15), np.repeat(np.arange(5), 5)
            else:
                sx, sy = images[:, :5], np.broadcast_to(np.arange(5)[:, None], (5, 5))
                q = rng.permutation(75)
                qx, qy = (
                    images[:, 5:].reshape(75, 1, 784)[q],
                    np.repeat(np.arange(5), 15)[q],
                )
            entries.append((sx, sy, qx, qy, episode))
            provenance.append((classes, turns, drawings))
        panel = stack_panels(entries)
        if phase == "source":
            np.savez(folder / "source.npz", **panel)
        else:
            panel["qx_clean"] = panel["qx"]
            panel["qx"] = augmented(panel["qx_clean"], generator)
            batches(folder, phase, panel, cfg["numeric_batch"])
        classes, turns, drawings = map(np.stack, zip(*provenance, strict=True))
        atomic_npz(
            folder / f"{phase}_draws.npz",
            classes=classes,
            turns=turns,
            drawings=drawings,
        )
        files[phase] = {
            "images": digest(path),
            "draws": digest(folder / f"{phase}_draws.npz"),
        }
    save_json(
        out / "panel_registration.json",
        {
            "plan_sha256": digest(PLAN_PATH),
            "images": files,
            "source": "5 clean and 10 transformed samples per class for fit, 5 disjoint transformed for checkpoints",
            "evaluation": "nested clean supports; identical query drawings presented with and without MSR query augmentation",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("wisdm", "aug_omniglot"))
    if parser.parse_args().dataset == "wisdm":
        wisdm()
    else:
        aug_omniglot()
