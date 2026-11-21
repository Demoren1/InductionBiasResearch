"""Repeat the frozen aSRR versus MSR protocol on disjoint WISDM subjects."""

import argparse
import json
from itertools import product

import numpy as np
import torch

from experiments.asrr_benchmark_families_v1.groups import PLAN, ROOT as FAMILY
from experiments.asrr_benchmark_families_v1.linear import atomic_npz
from experiments.asrr_benchmark_families_v1.panels import (
    batches,
    stack_panels,
    windows,
)
from experiments.asrr_transfer_refinement_v1.common import OUT, SEED
from experiments.asrr_transfer_refinement_v1.study import (
    evaluate,
    ordering,
    validation_rows,
)
from experiments.paper_transfer_v2.bilevel import train_msr
from experiments.paper_transfer_v2.common import CONFIG, freeze
from experiments.paper_transfer_v2.panels import panel_directory
from experiments.paper_transfer_v2.report import confidence, grouped
from experiments.real_group_transfer_v1.optimization import Layout, fit_dense
from experiments.real_group_transfer_v1.study import source_profiles
from experiments.sharing_structure_research_v1.blind_contract import save_json

from .fixed_blocks import STAGE, model_bank

DATASET = "wisdm"
FOLDER = OUT / DATASET / STAGE / "participant_folds"
SETTINGS = CONFIG["real"][DATASET]
LAYOUT = Layout(*SETTINGS["layout"])
RANK = SETTINGS["codes"] // LAYOUT.hidden
SHOTS = SETTINGS["shot"]


def subject_folds():
    path = FAMILY / DATASET / "panel_registration.json"
    record = json.loads(path.read_text())
    original = record["splits"]
    remaining = np.asarray(original["source"] + original["dev"])
    rng = np.random.default_rng(SEED + 90711)
    held_out = np.array_split(rng.permutation(remaining), 3)
    eligible = np.r_[remaining, original["test"]]
    folds = []
    for test in held_out:
        others = rng.permutation(np.setdiff1d(eligible, test))
        folds.append(
            {
                "test": test.tolist(),
                "train": others[:27].tolist(),
                "dev": others[27:].tolist(),
            }
        )
    return record["classes"], folds


def observations(split, classes, rng):
    folder = panel_directory(DATASET) / "watch"
    pools = {}
    for subject in sum(split.values(), []):
        series, _ = windows(
            folder / f"data_{subject}_accel_watch.txt", classes, 64
        )
        pools[subject] = (
            [row[: int(len(row) * 0.4)] for row in series],
            [row[int(len(row) * 0.5) :] for row in series],
        )
    values = np.concatenate(
        [x for subject in split["train"] for x in pools[subject][0]]
    ).reshape(-1, 4)
    center = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = values.std(axis=0, dtype=np.float64).astype(np.float32).clip(1e-6)
    for key, blocks in pools.items():
        pools[key] = tuple(
            [(x - center) / scale for x in part] for part in blocks
        )

    def draw(part, count):
        sample = np.stack(
            [x[rng.choice(len(x), count, replace=False)] for x in part]
        )
        return sample.reshape(len(classes), count, LAYOUT.sites, LAYOUT.fan_in)

    train = []
    ceiling = [
        min(cap, *(len(x) for s in split["train"] for x in pools[s][i]))
        for i, cap in enumerate((28, 15))
    ]
    for subject in split["train"]:
        x, q = [
            draw(part, count).reshape(-1, LAYOUT.sites, LAYOUT.fan_in)
            for part, count in zip(pools[subject], ceiling, strict=True)
        ]
        train.append(
            (
                x,
                np.repeat(np.arange(len(classes)), ceiling[0]),
                q,
                np.repeat(np.arange(len(classes)), ceiling[1]),
                subject,
            )
        )
    panels = {"train": stack_panels(train)}
    for phase, count in (("dev", 32), ("test", 64)):
        entries = []
        for subject in split[phase]:
            support, query = pools[subject]
            for _ in range(count):
                x = draw(support, SHOTS)
                q = draw(query, 12).reshape(-1, LAYOUT.sites, LAYOUT.fan_in)
                order = rng.permutation(len(q))
                sy = np.broadcast_to(
                    np.arange(len(classes))[:, None], (len(classes), SHOTS)
                )
                qy = np.repeat(np.arange(len(classes)), 12)[order]
                entries.append((x, sy, q[order], qy, subject))
        panels[phase] = stack_panels(entries)
    return panels


def fit_structures(folder, train, device):
    sx, sy, qx, qy = [
        torch.as_tensor(train[key], device=device)
        for key in ("sx", "sy", "qx", "qy")
    ]
    ordered = sx.reshape(
        len(sx), LAYOUT.outputs, -1, LAYOUT.sites, LAYOUT.fan_in
    )
    config = PLAN[DATASET]
    dense, error, _ = fit_dense(
        (sx, sy, qx, qy),
        LAYOUT,
        steps=config["source_steps"],
        restarts=config["source_restarts"],
        device=device,
        seed=SEED + 607,
    )
    profiles = source_profiles(dense[np.argsort(error[:, 0])], LAYOUT)
    bank = model_bank(
        folder,
        profiles,
        (sx, sy, {"query": qx}, qy),
        ordered,
        LAYOUT,
        RANK,
        SHOTS,
    )
    models = {
        f"asrr_{name}": row["path"]
        for name, row in bank.items()
        if row["group_weight"] > 0
    }
    for factors, rate in product(
        (1, LAYOUT.sites), CONFIG["real_bilevel"]["msr_rates"]
    ):
        for step, basis in train_msr(
            (ordered, qx, qy), LAYOUT, factors, RANK, rate, SHOTS, device
        ):
            name = f"msr_f{factors}_lr{rate:g}_step{step}"
            path = folder / f"{name}.npz"
            atomic_npz(path, factor=basis)
            models[name] = str(path)
    return models


def run(fold, device):
    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    folder = FOLDER / f"fold_{fold}"
    folder.mkdir(parents=True, exist_ok=True)
    classes, splits = subject_folds()
    split = splits[fold - 1]
    freeze(folder / "split.json", split)
    config = CONFIG["target"]
    manifest = folder / "models.json"
    if manifest.exists():
        models = json.loads(manifest.read_text())
    else:
        panels = observations(
            split, classes, np.random.default_rng(SEED + 5019 * fold)
        )
        for phase in ("dev", "test"):
            batches(folder, phase, panels[phase], 24)
        models = fit_structures(folder, panels["train"], device)
        freeze(manifest, models)
    choices = folder / "choices.json"
    if choices.exists():
        selected = json.loads(choices.read_text())
    else:
        rows = validation_rows(
            evaluate(
                DATASET,
                "dev",
                models,
                device,
                rates=config["rates"],
                steps=config["steps"],
                replicates=config["seed_replicates"],
                folder=folder / "dev",
            ),
            None,
        )
        selected = {
            key: min(
                (r for r in rows if r["candidate"].startswith(key)),
                key=ordering,
            )
            for key in ("asrr", "msr")
        }
        freeze(choices, selected)
    answers = folder / "test/answers.npz"
    with np.load(answers, allow_pickle=False) as arrays:
        units = arrays["units"]
    predictions = {}
    for key, choice in selected.items():
        result = evaluate(
            DATASET,
            "test",
            {key: models[choice["candidate"]]},
            device,
            rates=(choice["rate"],),
            steps=(choice["steps"],),
            replicates=config["seed_replicates"],
            folder=folder / "test",
        )
        predictions[key] = next(iter(result.values()))["query"][:, 0]
    delta = predictions["asrr"] - predictions["msr"]
    report = {
        "asrr_minus_msr": confidence(grouped(delta, units)),
        "metrics": {
            key: confidence(grouped(value, units))
            for key, value in predictions.items()
        },
        "episodes": len(units),
        "subjects": len(np.unique(units)),
    }
    atomic_npz(folder / "per_episode.npz", units=units, **predictions)
    save_json(folder / "results.json", report)
    print(f"WISDM fold {fold}", report, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fold", type=int, choices=(1, 2, 3))
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    run(args.fold, args.device)
