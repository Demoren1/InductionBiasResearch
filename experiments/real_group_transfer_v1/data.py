"""Prepare disjoint FC100 class and NinaPro DB2 subject episodes."""

import argparse
import hashlib
import pickle
import subprocess
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path
from urllib.parse import quote

import numpy as np
from scipy.io import loadmat

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json

FC100_SUPERCLASSES = {
    "source": (1, 2, 3, 4, 5, 6, 9, 10, 15, 17, 18, 19),
    "development": (8, 11, 13, 16),
    "test": (0, 7, 12, 14),
}
CIFAR100_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"
NINAPRO_MOVEMENTS = tuple(range(1, 13))
NINAPRO_REPETITIONS = (1, 2, 3, 4, 5, 6)
NINAPRO_SUBJECTS = dict(
    zip(
        ("source", "development", "test"),
        np.split(np.random.default_rng(2182603).permutation(40) + 1, (16, 24)),
    )
)


@dataclass(frozen=True)
class Panel:
    support_x: np.ndarray
    support_y: np.ndarray
    query_x: np.ndarray
    query_y: np.ndarray
    units: np.ndarray
    records: np.ndarray

    def subset(self, rows):
        return Panel(
            self.support_x[rows],
            self.support_y[rows],
            self.query_x[rows],
            self.query_y[rows],
            self.units[rows],
            self.records[rows],
        )


def prepare_fc100(path, output):
    with path.open("rb") as handle:
        checksum = hashlib.file_digest(handle, "md5").hexdigest()
    if checksum != CIFAR100_MD5:
        raise ValueError(f"CIFAR-100 checksum mismatch: {checksum}")

    with tarfile.open(path, "r:gz") as archive:
        batches = [
            pickle.load(
                archive.extractfile(f"cifar-100-python/{name}"), encoding="bytes"
            )
            for name in ("train", "test")
        ]
    image = np.concatenate([batch[b"data"] for batch in batches])
    # A 4 × 4 grid of 8 × 8 RGB patches, mean-pooled to 4 × 4 within each patch.
    patch = image.reshape(-1, 3, 4, 4, 2, 4, 4, 2).mean(axis=(4, 7), dtype=np.float32)
    x = patch.transpose(0, 2, 4, 3, 5, 1).reshape(-1, 16, 48)
    coarse = np.concatenate([batch[b"coarse_labels"] for batch in batches])
    fine = np.concatenate([batch[b"fine_labels"] for batch in batches])
    np.savez(output, x=(x / 127.5 - 1).astype(np.float16), coarse=coarse, fine=fine)


def fc100_panel(data, phase, count, seed):
    rng = np.random.default_rng(seed)
    fine, coarse = data["fine"], data["coarse"]
    classes = np.unique(fine[np.isin(coarse, FC100_SUPERCLASSES[phase])])
    members = {label: np.flatnonzero(fine == label) for label in classes}
    shots, queries = (200, 50) if phase == "source" else (5, 15)
    labels = np.stack(
        [rng.choice(classes, size=5, replace=False) for _ in range(count)]
    )
    records = np.stack(
        [
            [
                rng.choice(members[label], shots + queries, replace=False)
                for label in row
            ]
            for row in labels
        ]
    )
    support = records[:, :, :shots].reshape(count, -1)
    query = records[:, :, shots:].reshape(count, -1)
    return Panel(
        data["x"][support],
        np.broadcast_to(np.repeat(np.arange(5), shots), support.shape),
        data["x"][query],
        np.broadcast_to(np.repeat(np.arange(5), queries), query.shape),
        np.arange(count),
        records,
    )


def extract_ninapro(path, subject, output):
    names = ("emg", "stimulus", "repetition")
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            matches = [
                name
                for name in archive.namelist()
                if Path(name).name.upper() == f"S{subject}_E1_A1.MAT"
            ]
            if len(matches) != 1:
                raise ValueError(f"No unique first exercise for subject {subject}")
            with archive.open(matches[0]) as handle:
                raw = loadmat(handle, variable_names=names, squeeze_me=True)
    else:
        raw = loadmat(path, variable_names=names, squeeze_me=True)

    emg = raw["emg"]
    movement, repetition = raw["stimulus"], raw["repetition"]
    if emg.shape != (len(movement), 12) or repetition.shape != movement.shape:
        raise ValueError(f"Unexpected DB2 recording shape for subject {subject}")
    windows = []
    for label in NINAPRO_MOVEMENTS:
        trials = []
        for trial in NINAPRO_REPETITIONS:
            positions = np.flatnonzero((movement == label) & (repetition == trial))
            # Use the central sixteen nonoverlapping 200 ms windows in each trial.
            length = 16 * 400
            if len(positions) < length or np.any(np.diff(positions) != 1):
                raise ValueError(
                    f"Short or discontinuous DB2 trial {subject}/{label}/{trial}"
                )
            start = int(positions[(len(positions) - length) // 2])
            x = emg[start : start + length].reshape(16, 400, 12)
            dx = np.diff(x, axis=1)
            signals = np.stack(
                [
                    np.log(np.mean(np.abs(x), axis=1) + 1e-8),
                    np.log(np.std(x, axis=1) + 1e-8),
                    np.log(np.mean(np.abs(dx), axis=1) + 1e-8),
                    np.mean(np.signbit(x[:, 1:]) != np.signbit(x[:, :-1]), axis=1),
                ],
                axis=-1,
            )
            trials.append(signals.reshape(16, 48).astype(np.float32))
        windows.append(trials)
    np.save(output, np.asarray(windows), allow_pickle=False)


def fetch_ninapro_subject(subject, cache, output):
    features = output / f"s{subject:02d}.npy"
    receipt = output / f"s{subject:02d}.json"
    if features.is_file() and receipt.is_file():
        return subject
    name = f"S{subject}_E1_A1.mat"
    path = cache / name
    member = quote(f"DB2_s{subject}/DB2_s{subject}/{name}", safe="")
    endpoint = (
        "https://www.kaggle.com/api/v1/datasets/download/"
        f"quddusikashaf/ninapro-db2/{member}?datasetVersionNumber=1"
    )
    if not path.exists():
        temporary = path.with_suffix(".part")
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--retry",
                "3",
                "--connect-timeout",
                "12",
                "--max-time",
                "240",
                "--output",
                str(temporary),
                endpoint,
            ],
            check=True,
        )
        temporary.replace(path)
    extract_ninapro(path, subject, features)
    save_json(receipt, {"url": endpoint, "mat_sha256": digest(path)})
    path.unlink()
    return subject


def ninapro_source_normalization(folder):
    recordings = [
        np.load(folder / f"s{subject:02d}.npy", allow_pickle=False)
        for subject in NINAPRO_SUBJECTS["source"]
    ]
    features = np.stack(recordings)[:, :, :4].reshape(-1, 48)
    return features.mean(0), features.std(0).clip(1e-6)


def ninapro_panel(folder, phase, mean, scale):
    subjects = NINAPRO_SUBJECTS[phase]
    values = np.stack(
        [
            np.load(folder / f"s{subject:02d}.npy", allow_pickle=False)
            for subject in subjects
        ]
    )
    labels = len(NINAPRO_MOVEMENTS)
    values = ((values - mean) / scale).clip(-8, 8)[:, :, :, :, None, :]
    variants = (
        [(0, 1, 2, 3)]
        if phase == "source"
        else [(first, second) for first in range(6) for second in range(first + 1, 6)]
    )
    records = [
        (support, tuple(i for i in range(6) if i not in support))
        for support in variants
    ]
    sx, qx = [
        np.stack(
            [
                values[:, :, row[column]].reshape(len(subjects), -1, 1, 48)
                for row in records
            ],
            axis=1,
        )
        for column in (0, 1)
    ]
    sx, qx = [array.reshape(-1, *array.shape[2:]) for array in (sx, qx)]
    sy, qy = [
        np.broadcast_to(
            np.repeat(np.arange(labels), array.shape[1] // labels),
            array.shape[:2],
        )
        for array in (sx, qx)
    ]
    return Panel(
        sx,
        sy,
        qx,
        qy,
        np.repeat(subjects, len(variants)),
        np.tile(
            np.asarray([support + query for support, query in records]),
            (len(subjects), 1),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("fc100", "ninapro", "ninapro-kaggle"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--subject", type=int)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.dataset == "fc100":
        prepare_fc100(args.input, args.output)
    elif args.dataset == "ninapro-kaggle":
        args.input.mkdir(parents=True, exist_ok=True)
        args.output.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            for subject in pool.map(
                fetch_ninapro_subject,
                range(1, 41),
                repeat(args.input),
                repeat(args.output),
            ):
                print(f"Prepared NinaPro DB2 subject {subject:02d}", flush=True)
    else:
        extract_ninapro(args.input, args.subject, args.output)
