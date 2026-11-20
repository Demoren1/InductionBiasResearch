"""Evaluate frozen refinements on freshly drawn episodes from held-out pools."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.asrr_benchmark_families_v1.classification import read_data
from experiments.asrr_benchmark_families_v1.groups import ROOT as FAMILY
from experiments.asrr_benchmark_families_v1.linear import atomic_npz
from experiments.asrr_benchmark_families_v1.panels import (
    OMNIGLOT,
    augmented,
    batches,
    stack_panels,
    windows,
)
from experiments.paper_transfer_v2.common import OUT as PAPER
from experiments.paper_transfer_v2.panels import (
    episodes,
    panel_directory,
    read_idx,
    torus,
)
from experiments.paper_transfer_v2.report import confidence, grouped
from experiments.real_group_transfer_v1.data import fc100_panel
from experiments.real_group_transfer_v1.study import DATA as FC100_DATA
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

from .common import DATASETS, OUT, SEED
from .study import evaluate, ordering

PANEL_SEED = SEED + 986721


def omniglot(rng, seed):
    data = np.load(OMNIGLOT / "test.npy", mmap_mode="r")
    entries = []
    labels = np.repeat(np.arange(5), 15)
    support = np.broadcast_to(np.arange(5)[:, None], (5, 5))
    for episode in range(768):
        classes = rng.choice(len(data), 5, replace=False)
        images = np.stack(
            [
                np.rot90(
                    data[c, rng.permutation(20), 0],
                    int(rng.integers(4)),
                    axes=(-2, -1),
                )
                for c in classes
            ]
        ).reshape(5, 20, 1, 784)
        order = rng.permutation(75)
        query = images[:, 5:].reshape(75, 1, 784)[order]
        entries.append((images[:, :5], support, query, labels[order], episode))
    panel = stack_panels(entries)
    panel["qx_clean"] = panel["qx"]
    generator = torch.Generator().manual_seed(seed + 81)
    panel["qx"] = augmented(panel["qx"], generator)
    return panel


def mnist(rng):
    config = json.loads((PAPER / "mnist_torus" / "panels.json").read_text())[
        "provenance"
    ]
    permutation = np.array(config["permutation_after_circular_shift"])
    x, y = read_idx("t10k")
    entries = []
    for pair_id, pair in enumerate(config["digit_pairs"]["test"]):
        for _ in range(200):
            digits = rng.permutation(pair)
            ids = np.stack(
                [
                    rng.choice(np.flatnonzero(y == digit), 25, replace=False)
                    for digit in digits
                ]
            )
            support = torus(x, ids[:, :5].ravel(), rng, permutation).reshape(
                2, 5, 1, 784
            )
            query = torus(x, ids[:, 5:].ravel(), rng, permutation)
            entries.append(
                (
                    support,
                    np.broadcast_to(np.arange(2)[:, None], (2, 5)),
                    query,
                    np.repeat(np.arange(2), 20),
                    pair_id,
                )
            )
    return stack_panels(entries)


def wisdm(rng):
    cfg = json.loads(
        (FAMILY / "wisdm" / "panel_registration.json").read_text()
    )
    center = np.array(cfg["center"], dtype=np.float32)
    scale = np.array(cfg["scale"], dtype=np.float32)
    classes = cfg["classes"]
    entries = []
    labels = np.repeat(np.arange(len(classes)), 12)
    support = np.broadcast_to(
        np.arange(len(classes))[:, None], (len(classes), 5)
    )
    for subject in cfg["splits"]["test"]:
        path = (
            panel_directory("wisdm")
            / "watch"
            / f"data_{subject}_accel_watch.txt"
        )
        groups, _ = windows(path, classes, 64)
        early = [(x[: int(len(x) * 0.4)] - center) / scale for x in groups]
        late = [(x[int(len(x) * 0.5) :] - center) / scale for x in groups]
        for _ in range(64):
            sx = np.stack(
                [x[rng.choice(len(x), 5, replace=False)] for x in early]
            )
            qx = np.concatenate(
                [x[rng.choice(len(x), 12, replace=False)] for x in late]
            )
            order = rng.permutation(len(qx))
            entries.append(
                (
                    sx.reshape(len(classes), 5, 4, 64),
                    support,
                    qx.reshape(-1, 4, 64)[order],
                    labels[order],
                    subject,
                )
            )
    return stack_panels(entries)


def prepare(dataset, folder, seed):
    if (folder / "test" / "answers.npz").exists():
        return
    rng = np.random.default_rng(seed + DATASETS.index(dataset))
    if dataset == "fc100":
        with np.load(FC100_DATA / "fc100.npz", allow_pickle=False) as raw:
            panel = episodes(fc100_panel(raw, "test", 2048, seed + 6), 5, 5)
    else:
        panel = {
            "mnist_torus": mnist,
            "aug_omniglot": lambda rng: omniglot(rng, seed),
            "wisdm": wisdm,
        }[dataset](rng)
    batches(folder, "test", panel, 24)


def policies(dataset, stages):
    root = OUT / dataset
    candidates = []
    for name in stages:
        stage = root / name
        record = json.loads((stage / "choices.json").read_text())
        row = record["family"][record["selected"]]
        model = json.loads((stage / "bank.json").read_text())[
            row["candidate"]
        ]["path"]
        candidates.append({**row, "model": model, "study": name})
    best = min(candidates, key=ordering)
    reference = {
        name: json.loads((PAPER / dataset / f"{name}_choice.json").read_text())
        for name in ("asrr", "yeh_soft", "msr")
    }
    return {"refined": best, **reference}


def run(dataset, stages, device, replicate):
    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    name = "confirmation" if replicate == 0 else f"confirmation_{replicate}"
    folder = OUT / dataset / name
    seed = PANEL_SEED + 25783 * replicate
    folder.mkdir(parents=True, exist_ok=True)
    chosen = policies(dataset, stages)
    for row in chosen.values():
        row["sha256"] = digest(Path(row["model"]))
    save_json(
        folder / "choices.json", {"panel_seed": seed, "policies": chosen}
    )
    prepare(dataset, folder, seed)
    panel = folder / "test"
    values = {}
    units = read_data(panel / "answers.npz")["units"]
    for name, policy in chosen.items():
        data = evaluate(
            dataset,
            "test",
            {name: Path(policy["model"])},
            device,
            rates=(policy["rate"],),
            steps=(policy["steps"],),
            replicates=2,
            folder=panel,
        )
        values[name] = {
            k: v[:, 0] for k, v in next(iter(data.values())).items()
        }
    metrics, differences = {}, {}
    for name, row in values.items():
        metrics[name] = {
            q: confidence(grouped(v, units)) for q, v in row.items()
        }
        if name != "refined":
            differences[name] = {
                q: confidence(grouped(values["refined"][q] - v, units))
                for q, v in row.items()
            }
    atomic_npz(
        folder / "per_episode.npz",
        units=units,
        **{
            f"{m}_{q}": v
            for m, rows in values.items()
            for q, v in rows.items()
        },
    )
    save_json(
        folder / "results.json",
        {
            "metrics": metrics,
            "refined_minus_method": differences,
            "episodes": len(units),
            "units": len(np.unique(units)),
        },
    )
    print(
        dataset,
        "new episodes",
        {
            name: {
                key: round(100 * val["mean"], 2) for key, val in row.items()
            }
            for name, row in metrics.items()
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--stages", nargs="+", default=["known"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--replicate", type=int, default=0)
    args = parser.parse_args()
    run(args.dataset, args.stages, torch.device(args.device), args.replicate)
