"""Score frozen supplemental rows on the unified aSRR article panels."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.asrr_benchmark_families_v1.classification import read_data
from experiments.asrr_benchmark_families_v1.linear import atomic_npz
from experiments.asrr_tuning_v1.data import episodes
from experiments.paper_transfer_v2.common import CONFIG, OUT as PAPER
from experiments.paper_transfer_v2.har import REFERENCE, ROOT as HAR_ARCHIVE
from experiments.paper_transfer_v2.report import confidence, grouped
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

from .common import DATASETS, OUT, SEED
from .study import evaluate
from .unified_global import har_test

SUPPLEMENT = ("dense", "wsr", "lutq", "yeh_map")
DATA_ORDER = (*DATASETS, "har")


def phase(dataset):
    return (
        OUT / dataset / ("unified" if dataset == "har" else "confirmation_2")
    )


def classification(dataset, folder, device):
    queries = read_data(folder / "per_episode.npz")
    results = {"units": queries["units"]}
    for method in SUPPLEMENT:
        rule = json.loads(
            (PAPER / dataset / f"{method}_choice.json").read_text()
        )
        model = None if method == "dense" else Path(rule["model"])
        values = evaluate(
            dataset,
            "test",
            {method: model},
            device,
            rates=(rule["rate"],),
            steps=(rule["steps"],),
            replicates=2,
            folder=folder / "test",
        )
        for view, score in next(iter(values.values())).items():
            results[f"{method}_{view}"] = score[:, 0]
    if dataset == "aug_omniglot":
        rule = json.loads((folder / "choices.json").read_text())["policies"][
            "refined"
        ]
        model = read_data(Path(rule["model"]))
        rng = np.random.default_rng(SEED + 25120)
        directory = folder / "shuffled"
        directory.mkdir(exist_ok=True)
        paths = {}
        for i in range(4):
            name = f"shuffle_{i}"
            paths[name] = directory / f"{name}.npz"
            factor = np.stack(
                [rng.permutation(row, axis=0) for row in model["factor"]]
            )
            atomic_npz(paths[name], **{**model, "factor": factor})
        values = evaluate(
            dataset,
            "test",
            paths,
            device,
            rates=(rule["rate"],),
            steps=(rule["steps"],),
            replicates=2,
            folder=folder / "test",
        )
        for view in ("query", "clean"):
            shuffle = np.stack([row[view][:, 0] for row in values.values()])
            results[f"shuffled_{view}"] = shuffle.mean(0)
    return results


def har(folder, device):
    seed = json.loads((folder / "choices.json").read_text())["panel_seed"]
    count = len(read_data(folder / "per_episode.npz")["units"])
    observations, provenance = episodes("har3", "test", count, seed)
    units = np.asarray([row["person"] for row in provenance])
    inputs = [observations[k] for k in ("train_x", "train_y", "query_x")]
    archive = HAR_ARCHIVE / "banks" / "har" / "models"
    rules = json.loads((REFERENCE / "choices_frozen.json").read_text())
    prediction = {}
    for method in SUPPLEMENT:
        rule = rules[method]
        path = None if method == "dense" else archive / f"{rule['name']}.npz"
        if path is not None and digest(path) != rule["model_sha256"]:
            raise ValueError(f"Changed frozen HAR baseline {path}")
        prediction[method] = har_test(inputs, path, rule["rule"], device)
    truth = np.asarray(observations["query_y"]).argmax(-1)
    metrics = {
        f"{key}_query": (out == truth).mean(-1)
        for key, out in prediction.items()
    }
    return {"units": units, **metrics}


def complete(dataset, device):
    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    folder = phase(dataset)
    destination = folder / "article_remaining.npz"
    if destination.exists():
        return
    result = (
        har(folder, device)
        if dataset == "har"
        else classification(dataset, folder, device)
    )
    atomic_npz(destination, **result)
    print(
        dataset,
        {
            k: round(float(v.mean()) * 100, 2)
            for k, v in result.items()
            if k != "units"
        },
        flush=True,
    )


def summarize():
    records = {}
    for dataset in DATA_ORDER:
        folder = phase(dataset)
        is_har = dataset == "har"
        models = json.loads((folder / "choices.json").read_text())["policies"]
        original = read_data(folder / "per_episode.npz")
        additional = read_data(folder / "article_remaining.npz")
        units = original["units"]
        if not np.array_equal(units, additional["units"]):
            raise ValueError(f"Mismatched target episodes for {dataset}")
        comparison = {
            "asrr": "refined",
            "ward": "asrr",
            "yeh_soft": "yeh_soft",
            "msr": "msr",
        }
        views = ("clean", "query") if dataset == "aug_omniglot" else ("query",)
        data = {}
        for view in views:
            scores = {m: additional[f"{m}_{view}"] for m in SUPPLEMENT}
            for dest, src in comparison.items():
                scores[dest] = original[src if is_har else f"{src}_{view}"]
            scores["shuffled"] = (
                additional[f"shuffled_{view}"]
                if dataset == "aug_omniglot"
                else read_data(folder / "shuffled" / "per_episode.npz")[
                    "accuracy"
                ].mean(0)
            )
            data[view] = {
                "accuracy": {
                    m: confidence(grouped(v, units)) for m, v in scores.items()
                },
                "asrr_minus": {
                    m: confidence(grouped(scores["asrr"] - v, units))
                    for m, v in scores.items()
                    if m != "asrr"
                },
            }
        records[dataset] = {
            "episodes": len(units),
            "units": len(np.unique(units)),
            "codes": 128 if is_har else CONFIG["real"][dataset]["codes"],
            "refit_choice": models["refined"],
            "queries": data,
        }
    linear = json.loads(
        (OUT / "synthetic" / "unified" / "results.json").read_text()
    )
    save_json(
        PAPER / "unified_summary.json", {"linear": linear, "real": records}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=(*DATA_ORDER, "summary"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.task == "summary":
        summarize()
    else:
        complete(args.task, torch.device(args.device))
