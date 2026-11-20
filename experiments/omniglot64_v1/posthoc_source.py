"""Prepare Omniglot panels and extract input aSRR templates from source teachers."""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import torch
from scipy.cluster.hierarchy import cut_tree, linkage

from experiments.paper_asrr_v1.extract import source_profiles
from experiments.real_group_transfer_v1.optimization import Layout, fit_dense

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "artifacts/asrr_omniglot_posthoc_budget_2026-11-19"
IMAGES = PROJECT / "artifacts/local/msr_reproduction_2026-11-14/images/omniglot"
PLAN_FILE = Path(__file__).with_name("posthoc_budget_plan.json")
PLAN = json.loads(PLAN_FILE.read_text())
D, H, C = PLAN["input"], PLAN["teacher_width"], PLAN["classes"]
WIDTHS = tuple(map(int, PLAN["budgets"]))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def prepare_panel(name, split, cohorts, shots):
    data = np.load(IMAGES / f"{split}.npy", mmap_mode="r")
    folder = ROOT / name
    folder.mkdir(exist_ok=True)
    specs = []
    for seed, count in cohorts:
        rng = np.random.default_rng(seed)
        for _ in range(count):
            specs.append(
                (
                    rng.choice(len(data), C, replace=False),
                    rng.integers(0, 4, C),
                    np.stack([rng.permutation(20) for _ in range(C)]),
                    int(rng.integers(2**31)),
                )
            )
    classes, rotations, orders, shuffle_seeds = map(np.asarray, zip(*specs))
    np.savez(
        folder / "draws.npz",
        classes=classes,
        rotations=rotations,
        drawings=orders,
        shuffle_seeds=shuffle_seeds,
    )
    for shot in shots:
        public = folder / str(shot)
        public.mkdir(exist_ok=True)
        sx = np.empty((len(specs), C * shot, D), dtype=np.uint8)
        qx = np.empty((len(specs), C * (20 - shot), D), dtype=np.uint8)
        sy = np.empty(sx.shape[:2], dtype=np.uint8)
        qy = np.empty(qx.shape[:2], dtype=np.uint8)
        for i, (labels, turns, drawings, seed) in enumerate(specs):
            images = np.stack(
                [
                    np.rot90(data[c, d, 0], int(r), axes=(-2, -1)).reshape(20, D)
                    for c, r, d in zip(labels, turns, drawings)
                ]
            )
            rng = np.random.default_rng(seed + shot)
            sp, qp = rng.permutation(C * shot), rng.permutation(C * (20 - shot))
            sx[i] = images[:, :shot].reshape(-1, D)[sp]
            qx[i] = images[:, shot:].reshape(-1, D)[qp]
            sy[i] = np.repeat(np.arange(C), shot)[sp]
            qy[i] = np.repeat(np.arange(C), 20 - shot)[qp]
        for key, arr in (("sx", sx), ("sy", sy), ("qx", qx)):
            np.save(public / f"{key}.npy", arr)
        np.save(folder / f"answers_{shot}.npy", qy)


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    cfg = PLAN["source"]
    prepare_panel(
        "source", "train", [(cfg["seed"], cfg["episodes"])], [cfg["fit_shots"]]
    )
    for name in ("teacher_dev", "policy_dev"):
        cfg = PLAN[name]
        prepare_panel(
            name, cfg["split"], [(cfg["seed"], cfg["episodes"])], PLAN["target_shots"]
        )
    cfg = PLAN["test"]
    prepare_panel(
        "test",
        cfg["split"],
        [(s, cfg["episodes_per_cohort"]) for s in cfg["seeds"]],
        PLAN["target_shots"],
    )
    character_sets = [
        set(map(tuple, json.loads((IMAGES / f"{s}_classes.json").read_text())))
        for s in ("train", "val", "test")
    ]
    if any(
        a & b for i, a in enumerate(character_sets) for b in character_sets[i + 1 :]
    ):
        raise ValueError("Overlapping base characters between splits")
    write_json(
        ROOT / "registration.json",
        {
            "plan_sha256": sha(PLAN_FILE),
            "plan": PLAN,
            "data_sha256": {
                s: sha(IMAGES / f"{s}.npy") for s in ("train", "val", "test")
            },
            "draws_sha256": {
                n: sha(ROOT / n / "draws.npz")
                for n in ("source", "teacher_dev", "policy_dev", "test")
            },
        },
    )


def source_worker(work):
    gpu, ids = work
    cfg = PLAN["source"]
    torch.set_num_threads(1)
    torch.cuda.set_device(gpu)
    device = f"cuda:{gpu}"
    folder = ROOT / "source"
    data = [
        np.load(folder / str(cfg["fit_shots"]) / f"{k}.npy", mmap_mode="r")
        for k in ("sx", "sy", "qx")
    ]
    data.append(np.load(folder / f"answers_{cfg['fit_shots']}.npy", mmap_mode="r"))
    for i in ids:
        panel = [torch.tensor(x[i : i + 1], device=device) for x in data]
        sx, sy, qx, qy = panel
        panel = (
            sx.float().unsqueeze(2) / 255,
            sy.long(),
            qx.float().unsqueeze(2) / 255,
            qy.long(),
        )
        models, ce, _ = fit_dense(
            panel,
            Layout(1, D, H, C),
            steps=cfg["updates"],
            restarts=cfg["restarts"],
            seed=400079 + 997 * i,
            device=device,
        )
        np.savez(folder / f"fit_{i:03d}.npz", parameters=models[0], ce=ce[0])


def extract_bank():
    folder = ROOT / "source"
    params, losses = [], []
    for i in range(PLAN["source"]["episodes"]):
        with np.load(folder / f"fit_{i:03d}.npz") as data:
            params.append(data["parameters"])
            losses.append(data["ce"])
    params = np.stack(params)
    np.savez(
        folder / "models.npz",
        parameters=params[:, 0],
        ensemble_parameters=params,
        numerical_cross_entropy=np.asarray(losses).T,
    )
    matched, profiles, _, ranking = source_profiles(folder / "models.npz", D, H)
    bank = {"anchor": matched[0].astype(np.float32), "source_order": ranking}
    for blocks in (1, 4):
        per = H // blocks
        trees = [
            linkage(
                profiles[:, :, i * per : (i + 1) * per]
                .transpose(1, 0, 2)
                .reshape(D, -1),
                method="ward",
            )
            for i in range(blocks)
        ]
        for width in WIDTHS:
            k = PLAN["budgets"][str(width)] // H
            labels = np.stack([cut_tree(tree, n_clusters=k).ravel() for tree in trees])
            if any(len(np.unique(row)) != k for row in labels):
                raise ValueError("Wrong number of groups")
            bank[f"w{width}_b{blocks}"] = labels.astype(np.uint8)
    np.savez(ROOT / "bank.npz", **bank)
    print("Source bank ready", sha(ROOT / "bank.npz"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    args = parser.parse_args()
    prepare()
    pending = [
        i
        for i in range(PLAN["source"]["episodes"])
        if not (ROOT / "source" / f"fit_{i:03d}.npz").exists()
    ]
    with mp.get_context("spawn").Pool(len(args.gpus)) as pool:
        pool.map(
            source_worker,
            [(g, pending[i :: len(args.gpus)]) for i, g in enumerate(args.gpus)],
        )
    extract_bank()


if __name__ == "__main__":
    main()
