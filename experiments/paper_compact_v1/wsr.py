"""Add WSR to the retained synthetic experiment and HAR source bank."""

import argparse
import shutil
import time

import numpy as np
import torch

from experiments.paper_recovery_extension_v1.regression_target import (
    run as fit_synthetic_target,
)
from experiments.paper_structure_benchmark_v1.protocol.real_boundary import (
    StudyBoundary,
)
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)
from experiments.wsr_omniglot_v1.source import (
    REGISTRATION,
    fit_groups,
    read_json,
)

from .source import BUDGETS, HAR, OUTPUT, ROOT, SYNTHETIC, SYNTHETIC_ORIGINAL


CASE = "case_04"
TARGET = f"targets/{CASE}_c0"


def fit(setting, device):
    out = OUTPUT / setting
    out.mkdir()
    root = ROOT / setting
    boundary = StudyBoundary(root)
    boundary.install()
    torch.set_num_threads(2)
    start = time.perf_counter()
    if setting == "synthetic":
        data = SYNTHETIC_ORIGINAL / "public" / f"{CASE}_c0"
        protocol = read_json(SYNTHETIC / "protocol.json")
        panels = [
            data / f"{split}.npz" for split in ("fixed_train", "selection")
        ]
        dense = SYNTHETIC_ORIGINAL / f"dense/{CASE}_c0/fitted.npz"
    else:
        protocol = read_json(HAR / "protocol.json")
        panels = [
            HAR / "public" / f"{split}.npz"
            for split in ("support", "numerical")
        ]
        dense = HAR / "source/dense/models.npz"
    arrays = []
    for path in panels:
        with np.load(path) as panel:
            if setting == "synthetic":
                x, y = panel["x_0"][None], panel["y_0"][None]
            else:
                x, y = panel["x"] / 255, panel["y"]
        arrays.append(
            tuple(
                torch.as_tensor(a, device=device, dtype=torch.float32)
                for a in (x, y)
            )
        )
    with np.load(dense) as data:
        initial = data["parameters"].astype(np.float32)
    options = {
        **read_json(REGISTRATION)["source_training"],
        "codes": BUDGETS[setting],
    }
    shape = tuple(protocol[key] for key in ("inputs", "hidden", "outputs"))
    torch.cuda.synchronize(device)
    groups = fit_groups(initial, arrays, shape, options, protocol["loss"])
    bank = root / "wsr"
    bank.mkdir(parents=True)
    rows = []
    for task, (index, metrics) in enumerate(groups):
        name = f"wsr_{task}"
        path = bank / f"{name}.npz"
        np.savez_compressed(path, index=index)
        rows.append({"name": name, **metrics, "model_sha256": digest(path)})
    save_json(
        out / "wsr_source.json",
        {
            "settings": options,
            "candidates": rows,
            "source_data_sha256": {
                p.name: digest(p) for p in (*panels, dense)
            },
            "seconds": time.perf_counter() - start,
        },
    )
    boundary.save(out / "wsr_source_access.json", "complete")


def synthetic_target(device):
    root = ROOT / "synthetic"
    target = root / TARGET
    target.mkdir(parents=True)
    for name in ("sizes.json", "support.npz", "selection.npz", "query_x.npz"):
        shutil.copyfile(SYNTHETIC / TARGET / name, target / name)
    path = root / "wsr/wsr_0.npz"
    record = read_json(OUTPUT / "synthetic/wsr_source.json")
    if digest(path) != record["candidates"][0]["model_sha256"]:
        raise ValueError("Changed synthetic WSR grouping")
    with np.load(path) as model:
        basis = np.eye(BUDGETS["synthetic"])[model["index"] - 1]
    np.savez_compressed(target / "structures.npz", wsr=basis)
    fit_synthetic_target(root, CASE, 0, "wsr", device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("synthetic", "har3", "synthetic-target")
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.action == "synthetic-target":
        synthetic_target(device)
    else:
        fit(args.action, device)
