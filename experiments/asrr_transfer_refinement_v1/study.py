"""Select refinements on development only and pair their accuracy with baselines."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import torch

from experiments.asrr_benchmark_families_v1.classification import read_data
from experiments.asrr_benchmark_families_v1.linear import atomic_npz
from experiments.paper_transfer_v2.common import CONFIG, OUT as PAPER
from experiments.paper_transfer_v2.common import SEED as FIT_SEED
from experiments.paper_transfer_v2.panels import panel_directory
from experiments.paper_transfer_v2.report import confidence, grouped
from experiments.paper_transfer_v2.target import (
    fit_batch,
    metric,
    model_on_device,
    panel_batches,
)
from experiments.real_group_transfer_v1.optimization import Layout
from experiments.sharing_structure_research_v1.blind_contract import save_json

from .allocation import build_allocation
from .anchor import build_anchor
from .bank import build_bank
from .common import (
    DATASETS,
    OUT,
    SCREEN_EPISODES,
    SCREEN_RATES,
    SCREEN_STEPS,
    SEED,
)
from .contrasts import build_contrasts
from .conditioning import build_conditioning
from .fusion import build_fusion
from .meta import build_meta
from .reliable import build_reliable

REFERENCES = ("asrr", "dense", "yeh_soft", "msr")


def evaluate(
    dataset,
    phase,
    candidates,
    device,
    *,
    rates,
    steps,
    replicates,
    screen=False,
    folder=None,
):
    config = CONFIG["real"][dataset]
    layout = Layout(*config["layout"])
    directory = panel_directory(dataset) / phase if folder is None else folder
    return evaluate_panel(
        directory,
        layout,
        config["shot"],
        candidates,
        device,
        phase=phase,
        rates=rates,
        steps=steps,
        replicates=replicates,
        screen=screen,
    )


def evaluate_panel(
    directory,
    layout,
    shot,
    candidates,
    device,
    *,
    phase,
    rates,
    steps,
    replicates,
    screen=False,
):
    truth_path = directory / "answers.npz"
    truth = read_data(truth_path)["qy"]
    values = {}
    parts = panel_batches(
        directory, layout, shot, device, pack=6, include_clean=True
    )
    for seeds, rows, sx, sy, qx in parts:
        answers = truth[rows]
        if screen:
            grid = np.linspace(
                0,
                len(sx),
                num=round(SCREEN_EPISODES * len(sx) / len(truth)),
                endpoint=False,
                dtype=int,
            )
            if not len(grid):
                continue
            ids = torch.as_tensor(grid, device=device)
            sx, sy = sx[ids], sy[ids]
            qx = {key: x[ids] for key, x in qx.items()}
            answers = answers[grid]
            seeds = [(seeds[0][0], len(sx))]
        for name, path in candidates.items():
            structure = model_on_device(path, device)
            for rate, rep in product(rates, range(replicates)):
                offset = 8011 if phase == "test" else 0
                root = SEED if screen else FIT_SEED
                random = [
                    (n, root + 1129 * i + 10111 * rep + offset)
                    for i, n in seeds
                ]
                predictions = fit_batch(
                    sx, sy, qx, layout, structure, rate, steps, random
                )
                for step, logits in predictions.items():
                    key = (name, rate, step, rep)
                    for query, score in logits.items():
                        acc, ce = metric(score, answers)
                        values.setdefault((*key, query), []).append(
                            np.column_stack((acc, ce))
                        )
    packed = {}
    for (name, rate, step, _, query), rows in values.items():
        packed.setdefault((name, rate, step), {}).setdefault(query, []).append(
            np.concatenate(rows)
        )
    return {
        key: {query: np.mean(rows, axis=0) for query, rows in results.items()}
        for key, results in packed.items()
    }


def reference_floors(dataset, device):
    if dataset != "aug_omniglot":
        return {"query": 0.0}
    means = []
    for method in ("yeh_soft", "msr"):
        choice = json.loads(
            (PAPER / dataset / f"{method}_choice.json").read_text()
        )
        result = evaluate(
            dataset,
            "dev",
            {method: Path(choice["model"])},
            device,
            rates=(choice["rate"],),
            steps=(choice["steps"],),
            replicates=2,
        )
        row = next(iter(result.values()))
        means.append(
            {query: float(score[:, 0].mean()) for query, score in row.items()}
        )
    return {query: max(r[query] for r in means) for query in means[0]}


def validation_rows(scores, floors):
    rows = []
    for (name, rate, step), results in scores.items():
        accuracy = {
            query: float(v[:, 0].mean()) for query, v in results.items()
        }
        rows.append(
            {
                "candidate": name,
                "rate": rate,
                "steps": step,
                "accuracy": accuracy,
                "margin": (
                    float(np.mean(list(accuracy.values())))
                    if floors is None
                    else min(accuracy[key] - floors[key] for key in accuracy)
                ),
                "mean_ce": float(
                    np.mean([v[:, 1].mean() for v in results.values()])
                ),
            }
        )
    return rows


def ordering(row):
    return (
        -row["margin"],
        -np.mean(list(row["accuracy"].values())),
        row["mean_ce"],
        row["candidate"],
        row["rate"],
        row["steps"],
    )


def select(dataset, bank, folder, device, *, average_views=False):
    floors = None if average_views else reference_floors(dataset, device)
    paths = {name: Path(row["path"]) for name, row in bank.items()}
    screen = validation_rows(
        evaluate(
            dataset,
            "dev",
            paths,
            device,
            rates=SCREEN_RATES,
            steps=SCREEN_STEPS,
            replicates=1,
            screen=True,
        ),
        floors,
    )
    families = sorted({row["family"] for row in bank.values()})
    shortlisted = {}
    for family in families:
        rows = [
            row for row in screen if bank[row["candidate"]]["family"] == family
        ]
        top = list(
            dict.fromkeys(
                row["candidate"] for row in sorted(rows, key=ordering)
            )
        )[:2]
        shortlisted.update({name: paths[name] for name in top})
    cfg = CONFIG["target"]
    full = validation_rows(
        evaluate(
            dataset,
            "dev",
            shortlisted,
            device,
            rates=cfg["rates"],
            steps=cfg["steps"],
            replicates=cfg["seed_replicates"],
        ),
        floors,
    )
    chosen = {
        family: min(
            (
                row
                for row in full
                if bank[row["candidate"]]["family"] == family
            ),
            key=ordering,
        )
        for family in families
    }
    best = min(chosen, key=lambda key: ordering(chosen[key]))
    save_json(
        folder / "validation.json",
        {
            "reference_floors": floors,
            "screen": screen,
            "full": full,
        },
    )
    save_json(folder / "choices.json", {"family": chosen, "selected": best})
    print(
        dataset,
        "frozen",
        {k: (v["candidate"], v["accuracy"]) for k, v in chosen.items()},
        flush=True,
    )
    return chosen, best


def infer(dataset, bank, folder, chosen, best, device):
    key = "test"
    results, per_episode = {}, {}
    references = {
        method: read_data(PAPER / dataset / f"{method}_test.npz")
        for method in REFERENCES
    }
    units = references["asrr"]["units"]
    for family, row in chosen.items():
        name = row["candidate"]
        raw = evaluate(
            dataset,
            key,
            {name: Path(bank[name]["path"])},
            device,
            rates=(row["rate"],),
            steps=(row["steps"],),
            replicates=2,
        )
        queries = next(iter(raw.values()))
        results[family] = {}
        for query, values in queries.items():
            acc = values[:, 0]
            per_episode[f"{family}_{query}"] = acc
            results[family][query] = {
                "accuracy": confidence(grouped(acc, units)),
                "difference": {
                    method: confidence(
                        grouped(acc - reference[f"{query}_acc"], units)
                    )
                    for method, reference in references.items()
                },
            }
        print(
            dataset,
            family,
            {
                key: round(100 * val["accuracy"]["mean"], 2)
                for key, val in results[family].items()
            },
            flush=True,
        )
    atomic_npz(folder / "per_episode.npz", units=units, **per_episode)
    save_json(folder / "results.json", {"selected": best, "methods": results})


def run(dataset, stage, device):
    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    folder = OUT / dataset / stage
    if (folder / "results.json").exists():
        return
    if stage in ("hybrid", "ridge"):
        bank = build_meta(dataset, device, stage=stage)
    elif stage == "anchor":
        bank = build_anchor(dataset, device)
    elif stage == "contrasts":
        bank = build_contrasts(dataset, device)
    elif stage == "allocation":
        bank = build_allocation(dataset, device)
    elif stage == "conditioning":
        bank = build_conditioning(dataset, device)
    elif stage in ("reliable", "second_moments"):
        bank = build_reliable(dataset, device, stage=stage)
    elif stage == "fusion":
        bank = build_fusion(dataset)
    else:
        bank = build_bank(dataset, stage, device)
    path = folder / "choices.json"
    if path.exists():
        archive = json.loads(path.read_text())
        chosen, best = archive["family"], archive["selected"]
    else:
        chosen, best = select(dataset, bank, folder, device)
    infer(dataset, bank, folder, chosen, best, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument(
        "stage",
        choices=(
            "known",
            "relaxed",
            "hybrid",
            "anchor",
            "contrasts",
            "ridge",
            "allocation",
            "conditioning",
            "reliable",
            "fusion",
            "second_moments",
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.dataset, args.stage, torch.device(args.device))
