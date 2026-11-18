"""New controlled structures, matched cluster banks, then frozen truth audit."""

import argparse
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from experiments.heterogeneous_index_v1.model import scalar_seed
from experiments.paper_structure_benchmark_v1.methods.scalable_projection import assign
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from experiments.srr_approx_v1.core import (
    canonical,
    local_forest,
    signed_forest,
    synchronize_profiles,
)
from experiments.srr_approx_v1.synthetic import (
    BUDGETS,
    NOISE,
    SCENARIOS,
    cases,
    numerical,
)
from experiments.srr_approx_v1.synthetic import prepare as generate

from .clustering import METHODS, VENDOR, build_bank

ROOT = Path("artifacts/local/srr_clustering_2026-11-15")
DEST = Path("artifacts/srr_clustering_2026-11-15")
DATA = ROOT / "synthetic"


def prepare():
    DEST.mkdir(exist_ok=False)
    save_json(
        DEST / "registered.json",
        {
            "protocol_sha256": digest(Path(__file__).with_name("PROTOCOL.md")),
            "source_sha256": {
                str(path): digest(path)
                for path in [*Path(__file__).parent.glob("*.py"), *VENDOR.glob("*.py")]
            },
            "official_faiss_tag": "v1.16.0",
            "methods": METHODS,
            "synthetic_seed_base": 3511159100,
            "real_test_seeds": {"omniglot": 3511159201, "mnist": 3511159202},
            "real_test_episodes": 512,
            "real_started": False,
        },
    )
    generate(destination=DATA, seed_base=3511159100)


def learner(item):
    name, _scenario, _rank, replicate, _level = item
    torch.set_num_threads(1)
    started = time.perf_counter()
    data = dict(np.load(DATA / "public" / f"{name}.npz"))
    weights, error = data["weights"], data["uncertainty"]
    auxiliary, _ = signed_forest(weights, [24])
    synchronized = synchronize_profiles(auxiliary[24], weights, error)
    dense = np.arange(1, weights[0].size + 1, dtype=np.int16).reshape(weights.shape[1:])
    banks, records = {}, []
    for method in METHODS:
        bank, details = build_bank(
            weights, method, BUDGETS, ("global", "local", "input")
        )
        sync_bank, sync_details = build_bank(synchronized, method, BUDGETS, ("global",))
        bank.update(
            {key.replace("global", "sync"): value for key, value in sync_bank.items()}
        )
        banks[method] = bank
        records.extend(details)
        records.extend({**row, "scope": "sync"} for row in sync_details)
    banks["ward_adaptive"] = {
        f"adaptive_local_m{count}": index
        for count, index in local_forest(weights, BUDGETS).items()
        if count != dense.size
    }
    banks["g34"] = {}
    rng = np.random.default_rng(3511159150 + replicate)
    for count in BUDGETS[:-1]:
        for anchor in (0, 1):
            seed = scalar_seed(weights[anchor].ravel()[:, None], count).reshape(
                dense.shape
            )
            for restart in (0, 1):
                initial = (
                    seed
                    if restart == 0
                    else rng.integers(1, count + 1, dense.shape)
                    * np.sign(weights[anchor])
                )
                answer = assign(
                    initial.astype(np.int16),
                    np.ones(dense.shape, bool),
                    torch.as_tensor(weights),
                    torch.as_tensor(error),
                    count,
                    steps=12,
                )
                if answer is not None:
                    index, _, scales, _, _ = answer
                    banks["g34"][f"a{anchor}_r{restart}_m{count}"] = canonical(
                        index * np.where(scales[0] >= 0, 1, -1)[None]
                    )
    dense_dev, _ = numerical(dense, data, "development")
    dense_test, penalty = numerical(dense, data, "test")
    dev_truth = data["development_query_y"]
    dense_loss = float(np.mean((dense_dev - dev_truth) ** 2))
    arrays = {"dense_index": dense, "dense_prediction": dense_test}
    selected = {"dense": {"name": "dense", "codes": dense.size, "penalties": penalty}}
    choices, cache = [], {dense.tobytes(): (dense_loss, dense_test, penalty)}
    for method, bank in banks.items():
        candidates = [(dense_loss, "dense", dense)]
        for key, index in bank.items():
            blob = index.tobytes()
            if blob not in cache:
                dev_prediction, _ = numerical(index, data, "development")
                loss = float(np.mean((dev_prediction - dev_truth) ** 2))
                cache[blob] = (loss, None, None)
            loss = cache[blob][0]
            candidates.append((loss, key, index))
            choices.append(
                {
                    "method": method,
                    "name": key,
                    "codes": int(np.abs(index).max()),
                    "development_mse": loss,
                }
            )
        loss, key, index = min(candidates, key=lambda row: row[0])
        blob = index.tobytes()
        if cache[blob][1] is None:
            prediction, penalty = numerical(index, data, "test")
            cache[blob] = (loss, prediction, penalty)
        _, prediction, penalty = cache[blob]
        arrays[f"{method}_index"] = index
        arrays[f"{method}_prediction"] = prediction
        selected[method] = {
            "name": key,
            "codes": int(np.abs(index).max()),
            "development_mse": loss,
            "penalties": penalty,
        }
    folder = DATA / "predictions" / name
    folder.mkdir(exist_ok=False)
    np.savez_compressed(folder / "frozen.npz", **arrays)
    save_json(
        folder / "frozen.json",
        {
            "selected": selected,
            "development": choices,
            "sha256": digest(folder / "frozen.npz"),
            "seconds": time.perf_counter() - started,
            "private_opened": False,
        },
    )
    save_json(folder / "clustering.json", records)
    return {"case": name, "seconds": time.perf_counter() - started}


def run():
    pending = []
    for item in cases():
        folder = DATA / "predictions" / item[0]
        if (folder / "frozen.json").exists() and (folder / "clustering.json").exists():
            record = json.loads((folder / "frozen.json").read_text())
            if digest(folder / "frozen.npz") != record["sha256"]:
                raise RuntimeError("Incomplete or changed synthetic prediction")
            continue
        if folder.exists():
            shutil.rmtree(folder)
        pending.append(item)
    with ProcessPoolExecutor(max_workers=24) as pool:
        for row in pool.map(learner, pending):
            print(json.dumps(row), flush=True)
    save_json(
        DEST / "synthetic_predictions_frozen.json",
        {
            name: digest(DATA / "predictions" / name / "frozen.npz")
            for name, *_ in cases()
        },
    )


def score():
    if (DEST / "synthetic_results.json").exists():
        raise RuntimeError("Synthetic audit already completed")
    frozen = json.loads((DEST / "synthetic_predictions_frozen.json").read_text())
    rows = []
    for name, scenario, rank, replicate, level in cases():
        path = DATA / "predictions" / name / "frozen.npz"
        if digest(path) != frozen[name]:
            raise RuntimeError("Prediction changed after freeze")
        arrays = np.load(path)
        truth = np.load(DATA / "private" / f"{name}.npz")
        selected = json.loads(path.with_name("frozen.json").read_text())["selected"]
        true = truth["index"].ravel()
        equal = np.abs(true[:, None]) == np.abs(true[None])
        np.fill_diagonal(equal, False)
        for method, record in selected.items():
            index = arrays[f"{method}_index"].ravel()
            proposed = np.abs(index[:, None]) == np.abs(index[None])
            np.fill_diagonal(proposed, False)
            signed = np.sign(index[:, None] * index[None]) == np.sign(
                true[:, None] * true[None]
            )
            both = int(np.sum(proposed & equal & signed))
            rows.append(
                {
                    "case": name,
                    "scenario": scenario,
                    "rank": rank,
                    "replicate": replicate,
                    "noise": NOISE[level],
                    "method": method,
                    **record,
                    "mse": float(
                        np.mean(
                            (arrays[f"{method}_prediction"] - truth["test_query_clean"])
                            ** 2
                        )
                    ),
                    "signed_precision": both / int(proposed.sum())
                    if proposed.any()
                    else None,
                    "signed_recall": both / int(equal.sum()),
                }
            )
    aggregate = []
    for scenario in SCENARIOS:
        for noise in NOISE:
            for method in (*METHODS, "ward_adaptive", "g34", "dense"):
                group = [
                    row
                    for row in rows
                    if (row["scenario"], row["noise"], row["method"])
                    == (scenario, noise, method)
                ]
                aggregate.append(
                    {
                        "scenario": scenario,
                        "noise": noise,
                        "method": method,
                        "mse": float(np.mean([row["mse"] for row in group])),
                        "signed_recall": float(
                            np.mean([row["signed_recall"] for row in group])
                        ),
                        "mean_codes": float(np.mean([row["codes"] for row in group])),
                    }
                )
    save_json(DEST / "synthetic_results.json", {"cases": rows, "aggregate": aggregate})
    for row in aggregate:
        if row["noise"] == 0.04:
            print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run", "score"))
    stage = parser.parse_args().stage
    {"prepare": prepare, "run": run, "score": score}[stage]()


if __name__ == "__main__":
    main()
