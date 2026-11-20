"""Fit source networks and construct the indexators of manuscript Algorithm 1."""

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
import torch

from experiments.paper_asrr_v1.extract import source_profiles
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from experiments.srr_approx_v1.core import input_forest, signed_forest

from .optimization import Layout, fit_dense
from .trajectory_data import LOCAL, PLAN, ROOT, prepare


def fit_source(gpu, tasks, dataset):
    plan = json.loads(PLAN.read_text())
    network = plan["network"]
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    with np.load(LOCAL / dataset / "records.npz") as data:
        fit, validation = data["source_fit"], data["source_val"]
        rows = np.unique(np.r_[fit.ravel(), validation.ravel()])
        x = torch.as_tensor(data["x"][rows, None, :], device=device)
        y = torch.as_tensor(data["y"][rows], device=device)
    fit, validation = np.searchsorted(rows, fit), np.searchsorted(rows, validation)
    layout = Layout(1, x.shape[-1], network["hidden"], network["classes"])
    results = []
    for task in tasks:
        pooled = task == len(fit)
        a, b = (
            (fit.ravel(), validation.ravel())
            if pooled
            else (fit[task], validation[task])
        )
        panel = x[a][None], y[a][None], x[b][None], y[b][None]
        params, ce, _ = fit_dense(
            panel,
            layout,
            steps=network["pooled_steps" if pooled else "source_steps"],
            restarts=network["source_restarts"],
            device=device,
            seed=plan["seeds"]["source_fit"] + 19 * task,
        )
        results.append((task, params[0], ce[0]))
    return results


def construct(dataset, gpus):
    plan = json.loads(PLAN.read_text())
    folder = ROOT / dataset
    if (folder / "source.json").exists():
        raise FileExistsError(folder / "source.json")
    if not (folder / "data.json").exists():
        prepare(dataset)
    data = json.loads((folder / "data.json").read_text())
    if data["plan_sha256"] != digest(PLAN):
        raise ValueError("The registered specification has changed")
    n = len(data["units"]["source"])
    with ProcessPoolExecutor(
        max_workers=len(gpus), mp_context=mp.get_context("spawn")
    ) as pool:
        shards = pool.map(
            fit_source,
            gpus,
            [list(range(i, n + 1, len(gpus))) for i in range(len(gpus))],
            repeat(dataset),
        )
        results = sorted(
            (row for shard in shards for row in shard), key=lambda row: row[0]
        )
    _, pooled, pooled_ce = results.pop()
    np.savez_compressed(
        folder / "pooled.npz", parameter=pooled[0], validation_ce=pooled_ce
    )
    params, ce = np.stack([r[1] for r in results]), np.stack([r[2] for r in results])
    source = folder / "source_fits.npz"
    np.savez_compressed(
        source,
        parameters=params[:, 0],
        ensemble_parameters=params,
        numerical_cross_entropy=ce.T,
    )
    hidden, inputs = plan["network"]["hidden"], data["inputs"]
    torch.set_num_threads(1)
    _, z, _, ranking = source_profiles(source, inputs, hidden)
    budgets = [inputs * hidden * f // 8 for f in plan["network"]["fractions_eighths"]]
    global_maps, _ = signed_forest(z, budgets, device=f"cuda:{gpus[0]}")
    input_maps = input_forest(z, budgets)
    bank = {
        f"{family}_{budget}": maps[budget]
        for family, maps in (("global", global_maps), ("input", input_maps))
        for budget in budgets
    }
    if any(
        len(np.unique(abs(index))) != int(key.split("_")[1])
        for key, index in bank.items()
    ):
        raise ValueError("An indexator did not attain its planned budget")
    np.savez_compressed(folder / "asrr_bank.npz", **bank)
    np.savez_compressed(folder / "profiles.npz", profiles=z)
    save_json(
        folder / "source.json",
        {
            "plan_sha256": digest(PLAN),
            "data_sha256": digest(folder / "data.json"),
            "source_sha256": digest(source),
            "bank_sha256": digest(folder / "asrr_bank.npz"),
            "pooled_sha256": digest(folder / "pooled.npz"),
            "budgets": budgets,
            "source_tasks": n,
            "reference_task": int(ranking[0]),
        },
    )
    print(dataset, "source bank frozen", budgets, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("pen", "spoken"))
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    args = parser.parse_args()
    construct(args.dataset, args.gpus)
