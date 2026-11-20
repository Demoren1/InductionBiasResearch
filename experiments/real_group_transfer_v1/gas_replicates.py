"""Prospective Algorithm 1 comparison against replicated physical gas sensors."""

import argparse
import json
import multiprocessing as mp
import zipfile
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import numpy as np
import torch

from experiments.paper_asrr_v1.extract import source_profiles
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from experiments.srr_approx_v1.core import canonical, input_forest, signed_forest

from .compatible_study import score_predictions
from .optimization import Layout, fit_dense, fit_target_logits, losses, predict
from .trajectory_data import LOCAL, ROOT


PLAN = Path(__file__).with_name("gas_replicates_plan.json")
FOLDER, CACHE = ROOT / "gas", LOCAL / "gas"
LAYOUT = Layout(1, 128, 32, 5)
PERIODS = {"selection": (6, 7), "target": (8, 9, 10)}


def data():
    FOLDER.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    plan = json.loads(PLAN.read_text())
    path = LOCAL / "gas.zip"
    x, y, period = [], [], []
    feature_ids = list(range(1, 129))
    with zipfile.ZipFile(path) as archive:
        for t in range(1, 11):
            for line in archive.open(f"batch{t}.dat"):
                label, *tokens = line.split()
                gas = int(label.split(b";")[0])
                if gas == 6:
                    continue
                cols = [token.split(b":") for token in tokens]
                if [int(i) for i, _ in cols] != feature_ids:
                    raise ValueError("Unexpected gas feature layout")
                x.append([float(v) for _, v in cols])
                y.append(gas - 1)
                period.append(t)
    x, y, period = np.asarray(x), np.asarray(y), np.asarray(period)
    rng = np.random.default_rng(plan["seeds"]["data"])
    fit = np.zeros(len(x), bool)
    for t in range(1, 6):
        for c in range(5):
            rows = rng.permutation(np.flatnonzero((period == t) & (y == c)))
            fit[rows[: int(0.75 * len(rows))]] = True
    x = np.sign(x) * np.log1p(abs(x))
    mean, scale = x[fit].mean(0), x[fit].std(0).clip(1e-6)
    x = ((x - mean) / scale).clip(-8, 8).astype(np.float32)
    np.savez(CACHE / "records.npz", x=x, y=y, period=period, fit=fit)
    np.savez(FOLDER / "source_scaling.npz", mean=mean, scale=scale)
    save_json(
        FOLDER / "data.json",
        {
            "plan_sha256": digest(PLAN),
            "zip_sha256": digest(path),
            "records_sha256": digest(CACHE / "records.npz"),
            "records_per_period": [int((period == i).sum()) for i in range(1, 11)],
        },
    )


def source_worker(gpu, periods):
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    seed = json.loads(PLAN.read_text())["seeds"]["source"]
    with np.load(CACHE / "records.npz") as f:
        rows = f["period"] <= 5
        period, fit = f["period"][rows], f["fit"][rows]
        x = torch.as_tensor(f["x"][rows, None], device=device)
        y = torch.as_tensor(f["y"][rows], device=device)
    result = []
    for t in periods:
        select = np.ones(len(x), bool) if t == 0 else period == t
        a, b = np.flatnonzero(select & fit), np.flatnonzero(select & ~fit)
        inputs = x[a][None], y[a][None], x[b][None], y[b][None]
        pars, ce, _ = fit_dense(
            inputs,
            LAYOUT,
            steps=1200 if t == 0 else 800,
            restarts=3,
            device=device,
            seed=seed + 19 * t,
        )
        result.append((t, pars[0], ce[0]))
    return result


def construct(gpus):
    if (FOLDER / "source.json").exists():
        raise FileExistsError(FOLDER / "source.json")
    data()
    plan = json.loads(PLAN.read_text())
    with ProcessPoolExecutor(
        max_workers=len(gpus), mp_context=mp.get_context("spawn")
    ) as pool:
        shards = pool.map(
            source_worker,
            gpus,
            [list(range(i, 6, len(gpus))) for i in range(len(gpus))],
        )
        results = sorted((v for group in shards for v in group), key=lambda r: r[0])
    _, warm, warm_ce = results.pop(0)
    np.savez_compressed(FOLDER / "pooled.npz", parameter=warm[0], ce=warm_ce)
    pars, ce = np.stack([r[1] for r in results]), np.stack([r[2] for r in results])
    path = FOLDER / "source_fits.npz"
    np.savez_compressed(
        path,
        parameters=pars[:, 0],
        ensemble_parameters=pars,
        numerical_cross_entropy=ce.T,
    )
    _, z, _, ranking = source_profiles(path, LAYOUT.fan_in, LAYOUT.hidden)
    budgets = plan["budgets"]
    torch.set_num_threads(1)
    gs, _ = signed_forest(z, budgets, device=f"cuda:{gpus[0]}")
    ins = input_forest(z, budgets)
    bank = {
        f"{family}_{m}": v
        for family, maps in (("global", gs), ("input", ins))
        for m, v in maps.items()
    }
    group = ((np.arange(16) % 8) // 2)[:, None] * 8 + np.arange(8)
    bank["hardware"] = canonical(
        group.reshape(-1, 1) * LAYOUT.hidden + np.arange(1, LAYOUT.hidden + 1)
    )
    np.savez_compressed(FOLDER / "source_bank.npz", **bank)
    save_json(
        FOLDER / "source.json",
        {
            "plan_sha256": digest(PLAN),
            "bank_sha256": digest(FOLDER / "source_bank.npz"),
            "data_sha256": digest(FOLDER / "data.json"),
            "source_fits_sha256": digest(path),
            "ref_period": int(ranking[0] + 1),
            "pooled_sha256": digest(FOLDER / "pooled.npz"),
        },
    )
    print("gas source bank frozen", flush=True)


def panels(phase):
    public = CACHE / phase
    public.mkdir(exist_ok=True)
    receipt = public / "answers.npz"
    if receipt.exists():
        return public
    plan = json.loads(PLAN.read_text())
    rng = np.random.default_rng(
        plan["seeds"]["development" if phase == "selection" else "target"]
    )
    with np.load(CACHE / "records.npz") as f:
        x, y, periods = f["x"], f["y"], f["period"]
    support, query, units = [], [], []
    for t in PERIODS[phase]:
        rows = [np.flatnonzero((periods == t) & (y == c)) for c in range(5)]
        for _ in range(256 if phase == "selection" else 512):
            draw = np.stack([rng.choice(source, 17, replace=False) for source in rows])
            support.append(draw[:, :5])
            query.append(rng.permutation(draw[:, 5:].ravel()))
            units.append(t)
    support, query = np.stack(support), np.stack(query)
    for b, start in enumerate(range(0, len(units), 32)):
        s, q = support[start : start + 32], query[start : start + 32]
        np.savez(public / f"public_{b}.npz", sx=x[s], sy=y[s], qx=x[q])
    np.savez_compressed(public / "indices.npz", support=support, query=query)
    np.savez_compressed(
        receipt, query_y=y[query], units=units, batch=np.arange(len(units)) // 32
    )
    return public


def adapt_pretrained(sx, sy, qx, warm, steps):
    state = (
        torch.as_tensor(warm, device=sx.device)[None, None]
        .expand(len(sx), -1, -1)
        .clone()
        .requires_grad_()
    )
    optimizer = torch.optim.Adam([state], lr=0.01)
    size = LAYOUT.connections
    result = {}
    for i in range(max(steps) + 1):
        if i in steps:
            with torch.no_grad():
                result[i] = (
                    predict(qx, state[..., :size], state[..., size:], LAYOUT)[:, 0]
                    .softmax(-1)
                    .cpu()
                    .numpy()
                )
        if i == max(steps):
            break
        error = losses(predict(sx, state[..., :size], state[..., size:], LAYOUT), sy)
        optimizer.zero_grad(set_to_none=True)
        error.mean(-1).sum().backward()
        optimizer.step()
    return result


def predict_worker(gpu, batches, stage, phase, shots, manifest):
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    with np.load(FOLDER / manifest["bank"]) as f:
        maps = dict(f)
    with np.load(FOLDER / "pooled.npz") as f:
        warm = f["parameter"]
    for b in batches:
        path = CACHE / stage / f"pred_{b}_0.npz"
        if path.exists():
            continue
        with np.load(CACHE / phase / f"public_{b}.npz") as f:
            x = f["sx"][:, :, :shots]
            sx = torch.as_tensor(x.reshape(len(x), -1, 1, 128), device=device)
            sy = torch.as_tensor(
                f["sy"][:, :, :shots].reshape(len(x), -1), device=device
            )
            qx = torch.as_tensor(f["qx"][:, :, None, :], device=device)
        probs = {}
        for name, spec in manifest["methods"].items():
            if name == "pooled":
                for step, p in adapt_pretrained(
                    sx, sy, qx, warm, spec["steps"]
                ).items():
                    probs[f"{name}__{step}"] = p
                continue
            layout = Layout(1, 128, spec["width"], 5)
            index = None if spec["dense"] else maps[name]
            for step in spec["steps"]:
                pred, _, _ = fit_target_logits(
                    sx,
                    sy,
                    qx,
                    layout,
                    index,
                    device,
                    steps=step,
                    seed=manifest["numeric_seed"] + b,
                )
                probs[f"{name}__{step}"] = pred[:, 0].softmax(-1).cpu().numpy()
        tmp = path.with_suffix(".partial.npz")
        np.savez(tmp, **probs)
        tmp.replace(path)


def evaluate(phase, shots, gpus, methods, bank):
    stage = "development" if phase == "selection" else f"target_{shots}"
    public = panels(phase)
    cache = CACHE / stage
    cache.mkdir(exist_ok=True)
    plan = json.loads(PLAN.read_text())
    manifest = {
        "plan_sha256": digest(PLAN),
        "bank": bank,
        "bank_sha256": digest(FOLDER / bank),
        "source_sha256": digest(FOLDER / "source.json"),
        "choice_sha256": None
        if phase == "selection"
        else digest(FOLDER / "frozen.json"),
        "indices_sha256": digest(public / "indices.npz"),
        "answers_sha256": digest(public / "answers.npz"),
        "methods": methods,
        "shots": shots,
        "numeric_seed": plan["seeds"][
            "numeric_development" if phase == "selection" else "numeric_target"
        ],
    }
    path = FOLDER / f"{stage}_manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Different gas configuration or assignments encountered")
    save_json(path, manifest)
    if (FOLDER / f"{stage}_scores.npz").exists():
        return
    nb = len(list(public.glob("public_*.npz")))
    with ProcessPoolExecutor(
        max_workers=len(gpus), mp_context=mp.get_context("spawn")
    ) as pool:
        list(
            pool.map(
                predict_worker,
                gpus,
                [list(range(i, nb, len(gpus))) for i in range(len(gpus))],
                repeat(stage),
                repeat(phase),
                repeat(shots),
                repeat(manifest),
            )
        )
    save_json(
        FOLDER / f"{stage}_predictions.json",
        {str(b): digest(cache / f"pred_{b}_0.npz") for b in range(nb)},
    )
    names = [f"{k}__{s}" for k, v in methods.items() for s in v["steps"]]
    score_predictions(FOLDER, cache, stage, [names], nb, public / "answers.npz")
    print("gas", stage, nb, "prediction batches", flush=True)


def develop(gpus):
    plan = json.loads(PLAN.read_text())
    with np.load(FOLDER / "source_bank.npz") as f:
        bank = dict(f)
    methods = {k: {"steps": [25, 75, 200], "width": 32, "dense": False} for k in bank}
    budgets = {k: 1024 if k == "hardware" else int(k.split("_")[1]) for k in bank}
    for m in plan["budgets"]:
        name = f"narrow_{m}"
        budgets[name] = m
        methods[name] = {"steps": [25, 75, 200], "width": m // 128, "dense": True}
    for name in ("pooled", "dense"):
        budgets[name] = 4096
        methods[name] = {
            "steps": ([0, 25, 75, 200] if name == "pooled" else [25, 75, 200]),
            "width": 32,
            "dense": True,
        }
    evaluate("selection", 2, gpus, methods, "source_bank.npz")
    with np.load(FOLDER / "development_scores.npz") as f:
        scores = {k: f[k].mean(0) for k in f.files if k not in ("batch", "units")}

    def best(candidates):
        def ranking(k):
            name, step = k.rsplit("__", 1)
            acc, _, brier = scores[k]
            return -acc, brier, budgets[name], int(step), name

        return min(
            (k for k in scores if k.rsplit("__", 1)[0] in candidates), key=ranking
        )

    families = [k for k in bank if k != "hardware"]
    chosen = {
        "asrr": best(families),
        "physical_budget": best([k for k in families if budgets[k] == 1024]),
    }
    winning_m = budgets[chosen["asrr"].rsplit("__", 1)[0]]
    for key, model in (
        ("dense", "dense"),
        ("narrow", f"narrow_{winning_m}"),
        ("hardware", "hardware"),
        ("pooled", "pooled"),
    ):
        chosen[key] = best([model])
    test_methods, test_bank, aliases = {}, {}, {}
    for label, setting in chosen.items():
        name, step = setting.rsplit("__", 1)
        aliases[label] = name
        test_methods[name] = dict(methods[name], steps=[int(step)])
        if name in bank:
            test_bank[name] = bank[name]
    for group in ("asrr", "physical_budget"):
        key = aliases[group]
        for i, seed in enumerate(plan["seeds"]["permutations"]):
            name = f"{key}_perm_{i}"
            aliases[f"{group}_perm_{i}"] = name
            test_bank[name] = test_bank[key][
                np.random.default_rng(seed).permutation(128)
            ]
            test_methods[name] = test_methods[key]
    np.savez_compressed(FOLDER / "frozen_indices.npz", **test_bank)
    save_json(
        FOLDER / "frozen.json",
        {
            "plan_sha256": digest(PLAN),
            "source_sha256": digest(FOLDER / "source.json"),
            "scores_sha256": digest(FOLDER / "development_scores.npz"),
            "bank_sha256": digest(FOLDER / "frozen_indices.npz"),
            "pooled_sha256": digest(FOLDER / "pooled.npz"),
            "choices": chosen,
            "budget": winning_m,
            "aliases": aliases,
            "methods": test_methods,
            "development_accuracy_ce_brier": {
                k: scores[v].tolist() for k, v in chosen.items()
            },
        },
    )
    print("gas frozen", chosen, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("source", "develop", "target", "all"))
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    args = parser.parse_args()
    if args.stage in ("source", "all"):
        construct(args.gpus)
    if args.stage in ("develop", "all"):
        develop(args.gpus)
    if args.stage in ("target", "all"):
        frozen = json.loads((FOLDER / "frozen.json").read_text())
        if (
            frozen["plan_sha256"] != digest(PLAN)
            or frozen["bank_sha256"] != digest(FOLDER / "frozen_indices.npz")
            or frozen["pooled_sha256"] != digest(FOLDER / "pooled.npz")
        ):
            raise ValueError("Gas test choices differ from frozen receipt")
        for shots in (2, 5):
            evaluate(
                "target", shots, args.gpus, frozen["methods"], "frozen_indices.npz"
            )
