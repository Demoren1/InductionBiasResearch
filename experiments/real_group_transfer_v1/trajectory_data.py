"""Two writer- or speaker-disjoint UCI corpora for aSRR transfer."""

import io
import json
import re
import subprocess
import zipfile
from pathlib import Path

import numpy as np

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json


PLAN = Path(__file__).with_name("trajectory_plan.json")
ROOT = Path("artifacts/asrr_digit_trajectories_2026-11-19")
LOCAL = ROOT.parent / "local" / ROOT.name
SPLITS = {"source": 0, "discovery": 1, "selection": 2, "target": 3}
CHANNELS = {"pen": 2, "spoken": 13}


def pen_records(archive, train):
    ext = "tra" if train else "tes"
    values = np.loadtxt(io.BytesIO(archive.read(f"pendigits.{ext}")), delimiter=",")
    raw = subprocess.run(
        ["gzip", "-dc"],
        input=archive.read(f"pendigits-orig.{ext}.Z"),
        capture_output=True,
        check=True,
    ).stdout
    comments = np.array(
        [
            list(map(int, line.split()[1:]))
            for line in raw.splitlines()
            if line.startswith(b".COMMENT")
        ]
    )
    if not np.array_equal(values[:, -1], comments[:, 0]):
        raise ValueError("Official coordinates and UNIPEN writer records disagree")
    return values[:, :-1], comments[:, 0], comments[:, 1]


def spoken_records(archive, train):
    n = 66 if train else 22
    path = f"{'Train' if train else 'Test'}_Arabic_Digit.txt"
    content = archive.read(path).strip()
    records = re.split(rb"\r?\n[ \t]*\r?\n(?:[ \t]*\r?\n)*", content)
    if len(records) != 10 * n * 10:
        raise ValueError(
            f"{path}: expected one record per utterance, got {len(records)}"
        )
    values = []
    for record in records:
        frames = np.fromstring(record, dtype=np.float32, sep=" ").reshape(-1, 13)
        time = np.linspace(0, len(frames) - 1, 16)
        start = time.astype(int)
        frac = (time - start)[:, None]
        sample = (1 - frac) * frames[start] + frac * frames[
            np.minimum(start + 1, len(frames) - 1)
        ]
        values.append(sample.ravel())
    return (
        np.stack(values),
        np.repeat(np.arange(10), n * 10),
        np.tile(np.repeat(np.arange(n), 10), 10),
    )


def prepare(dataset):
    plan = json.loads(PLAN.read_text())
    config = plan[dataset]
    folder, local = ROOT / dataset, LOCAL / dataset
    folder.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    manifest = folder / "data.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    path = LOCAL / f"{dataset}.zip"
    load = pen_records if dataset == "pen" else spoken_records
    with zipfile.ZipFile(path) as archive:
        x_train, y_train, u_train = load(archive, True)
        x_test, y_test, u_test = load(archive, False)
    rng = np.random.default_rng(plan["seeds"]["participants"])
    if dataset == "pen":
        strata = [np.unique(u_train)]
        sizes = config["train_writers"]
    else:
        strata = np.split(np.unique(u_train), 2)
        sizes = config["training_blocks_per_sex"]
    groups = {k: [] for k in SPLITS}
    for stratum in strata:
        if sum(sizes) != len(stratum):
            raise ValueError("Official training units do not match the planned split")
        for phase, ids in zip(
            ("source", "discovery", "selection"),
            np.split(rng.permutation(stratum), np.cumsum(sizes)[:-1]),
            strict=True,
        ):
            groups[phase].extend(ids.tolist())
    groups["target"] = (100 + np.unique(u_test)).tolist()
    x = np.concatenate([x_train, x_test]).astype(np.float32)
    y = np.concatenate([y_train, y_test])
    unit = np.concatenate([u_train, 100 + u_test])
    split = np.full(len(x), -1)
    for phase, group in groups.items():
        split[np.isin(unit, group)] = SPLITS[phase]
    if np.any(split < 0):
        raise ValueError("Unassigned data record")
    rng = np.random.default_rng(plan["seeds"]["source_split"])
    fit, validation = [], []
    nf, nv = config["source_fit_per_class"], config["source_val_per_class"]
    for group in groups["source"]:
        rows = np.stack(
            [
                rng.choice(
                    np.flatnonzero((unit == group) & (y == c)), nf + nv, replace=False
                )
                for c in range(10)
            ]
        )
        fit.append(rows[:, :nf].ravel())
        validation.append(rows[:, nf:].ravel())
    fit, validation = np.stack(fit), np.stack(validation)
    mean = x[fit.ravel()].mean(0, dtype=np.float64)
    scale = x[fit.ravel()].std(0, dtype=np.float64).clip(1e-6)
    x = ((x - mean) / scale).astype(np.float32)
    data = local / "records.npz"
    np.savez(
        data, x=x, y=y, unit=unit, split=split, source_fit=fit, source_val=validation
    )
    np.savez(folder / "source_scaling.npz", mean=mean, scale=scale)
    save_json(
        manifest,
        {
            "plan_sha256": digest(PLAN),
            "zip_sha256": digest(path),
            "records_sha256": digest(data),
            "units": groups,
            "inputs": x.shape[1],
            "counts": {p: int(np.sum(split == i)) for p, i in SPLITS.items()},
        },
    )


def episode_indices(dataset, phase):
    plan = json.loads(PLAN.read_text())
    folder = LOCAL / dataset
    panel = folder / f"{phase}_indices.npz"
    if panel.exists():
        return
    with np.load(folder / "records.npz") as f:
        unit, y, split = f["unit"], f["y"], f["split"]
    active = np.flatnonzero(split == SPLITS[phase])
    if dataset == "pen":
        pools = [active[unit[active] == u] for u in np.unique(unit[active])]
        count = plan[dataset]["episodes_per_writer"]
    else:
        pools = [active]
        count = plan[dataset]["episodes"][phase]
    rng = np.random.default_rng(plan["seeds"]["episode"][phase])
    support, query, units = [], [], []
    for pool in pools:
        labels = [pool[y[pool] == c] for c in range(10)]
        for _ in range(count):
            sample = np.stack([rng.choice(rows, 17, replace=False) for rows in labels])
            support.append(sample[:, :5])
            query.append(rng.permutation(sample[:, 5:].ravel()))
            units.append(unit[pool[0]] if dataset == "pen" else len(units) // 32)
    np.savez_compressed(
        panel, support=np.stack(support), query=np.stack(query), units=units
    )
