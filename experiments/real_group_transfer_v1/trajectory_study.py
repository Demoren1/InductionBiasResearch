"""Select structures on development tasks before refitting on unseen cohorts."""

import argparse
import json

import numpy as np

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from experiments.srr_approx_v1.compatible_transfer import block_bank, join, residuals
from experiments.srr_approx_v1.core import canonical, input_forest

from .compatible_relations import maximal_supported, relation_intersections
from .trajectory_data import CHANNELS, PLAN, ROOT
from .trajectory_eval import evaluate


def best_setting(scores, candidates, budgets):
    def key(item):
        name, steps = item.rsplit("__", 1)
        acc, brier, _ = scores[item].mean(axis=0)
        return -acc, brier, budgets[name], int(steps), name

    name, step = min(
        (
            k
            for k in scores.files
            if k != "units" and k.rsplit("__", 1)[0] in candidates
        ),
        key=key,
    ).rsplit("__", 1)
    return {
        "model": name,
        "step": int(step),
        "accuracy_brier_ce": scores[f"{name}__{step}"].mean(axis=0).tolist(),
    }


def select_core(dataset, gpus):
    folder = ROOT / dataset
    plan, source, data = (
        json.loads(p.read_text())
        for p in (PLAN, folder / "source.json", folder / "data.json")
    )
    if source["plan_sha256"] != digest(PLAN) or source["bank_sha256"] != digest(
        folder / "asrr_bank.npz"
    ):
        raise ValueError("The source-only bank or study plan has changed")
    hidden, steps = plan["network"]["hidden"], plan["network"]["candidate_steps"]
    inputs, channels = data["inputs"], CHANNELS[dataset]
    times = inputs // channels
    with np.load(folder / "asrr_bank.npz") as f:
        bank = dict(f)
    learned = list(bank)
    budgets = {k: int(k.split("_")[1]) for k in bank}
    methods = {k: {"kind": "index", "width": hidden, "steps": steps} for k in bank}
    for budget in source["budgets"]:
        k = budget // hidden
        time_bins = k // channels
        labels = (np.arange(times) * time_bins // times)[
            :, None
        ] * channels + np.arange(channels)
        temporal = f"time_{budget}"
        bank[temporal] = canonical(
            labels.reshape(-1, 1) * hidden + np.arange(1, hidden + 1)
        )
        methods[temporal] = {"kind": "index", "width": hidden, "steps": steps}
        narrow = f"narrow_{budget}"
        methods[narrow] = {"kind": "dense", "width": budget // inputs, "steps": steps}
        budgets[temporal] = budgets[narrow] = budget
    for name in ("dense", "pooled"):
        methods[name] = {
            "kind": name,
            "width": hidden,
            "steps": ([0] + steps) if name == "pooled" else steps,
        }
        budgets[name] = hidden * inputs
    np.savez_compressed(folder / "core_bank.npz", **bank)
    evaluate(
        dataset, "selection_core", "selection", 2, methods, "core_bank.npz", gpus, None
    )
    with np.load(folder / "selection_core_scores.npz") as scores:
        choices = {"asrr": best_setting(scores, learned, budgets)}
        budget = budgets[choices["asrr"]["model"]]
        for name, key in (
            ("dense", "dense"),
            ("pooled", "pooled"),
            ("narrow", f"narrow_{budget}"),
            ("time", f"time_{budget}"),
        ):
            choices[name] = best_setting(scores, [key], budgets)
    save_json(
        folder / "core_choice.json",
        {
            "budget": budget,
            "choices": choices,
            "bank_sha256": digest(folder / "core_bank.npz"),
            "scores_sha256": digest(folder / "selection_core_scores.npz"),
        },
    )
    print(dataset, "Algorithm 1", choices["asrr"], flush=True)


def select_compatible(dataset, gpus):
    folder = ROOT / dataset
    plan = json.loads(PLAN.read_text())
    core = json.loads((folder / "core_choice.json").read_text())
    budget, step = core["budget"], core["choices"]["asrr"]["step"]
    with np.load(folder / "profiles.npz") as f:
        z = f["profiles"]
    base = join(
        [
            input_forest(block, [budget // 4])[budget // 4]
            for block in np.split(z, 4, axis=2)
        ]
    )
    hybrids, _ = block_bank(base, residuals(z, base), plan["seeds"]["hybrid"])
    with np.load(folder / "asrr_bank.npz") as f:
        proposed = {k: f[f"{k}_{budget}"] for k in ("global", "input")}
    proposed.update({f"hybrid_{k}": v for k, v in hybrids.items()})
    bank, seen = {}, set()
    for name, index in proposed.items():
        identity = canonical(index).tobytes()
        if identity not in seen:
            bank[name] = index
            seen.add(identity)
    labels = list(bank)
    np.savez_compressed(folder / "compatible_bank.npz", **bank)
    maps, extends = relation_intersections(list(bank.values()))
    hidden = plan["network"]["hidden"]
    methods = {k: {"kind": "index", "width": hidden, "steps": [step]} for k in bank}
    for phase in ("discovery", "selection"):
        evaluate(
            dataset,
            f"{phase}_compat",
            phase,
            2,
            methods,
            "compatible_bank.npz",
            gpus,
            "core_choice.json",
        )
    with np.load(folder / "discovery_compat_scores.npz") as f:
        risk = np.stack([f[f"{k}__{step}"][:, 1] for k in labels], axis=1)
        if dataset == "pen":
            units = f["units"]
            risk = np.stack([risk[units == u].mean(0) for u in np.unique(units)])
    good = risk <= risk.min(1, keepdims=True) + 0.01
    frontier = maximal_supported(good, extends, 1)
    eligible = [labels[i] for i in np.flatnonzero(extends[frontier].any(axis=0))]
    with np.load(folder / "selection_compat_scores.npz") as scores:
        budgets = dict.fromkeys(labels, budget)
        strict = best_setting(scores, eligible, budgets)
        erm = best_setting(scores, labels, budgets)
    np.savez_compressed(
        folder / "shared_relations.npz",
        maps=maps[frontier],
        extends=extends[frontier],
        models=labels,
    )
    save_json(
        folder / "compat_choice.json",
        {
            "budget": budget,
            "strict": strict,
            "erm": erm,
            "eligible": eligible,
            "bank_size": len(labels),
            "frontier_size": len(frontier),
            "discovery_tasks": len(risk),
            "median_versions": float(np.median(good.sum(1))),
            "relations_sha256": digest(folder / "shared_relations.npz"),
        },
    )


def freeze(dataset):
    folder = ROOT / dataset
    plan = json.loads(PLAN.read_text())
    core = json.loads((folder / "core_choice.json").read_text())
    comp = json.loads((folder / "compat_choice.json").read_text())
    choices, budget = core["choices"], core["budget"]
    with np.load(folder / "core_bank.npz") as f:
        fixed = {k: f[choices[k]["model"]] for k in ("asrr", "time")}
    with np.load(folder / "compatible_bank.npz") as f:
        fixed.update({k: f[comp[k]["model"]] for k in ("strict", "erm")})
    asrr = fixed["asrr"]
    inputs, width = asrr.shape
    for j, seed in enumerate(plan["seeds"]["shuffles"]):
        fixed[f"permutation_{j}"] = asrr[
            np.random.default_rng(seed).permutation(inputs)
        ]
        selected = np.random.default_rng(seed + 1).choice(
            asrr.size, budget - 1, replace=False
        )
        single = np.ones(asrr.size, dtype=np.int16)
        single[selected] = np.arange(2, budget + 1)
        fixed[f"mask_{j}"] = single.reshape(asrr.shape)
    primary_step = choices["asrr"]["step"]
    methods, aliases, bank, previously = {}, {}, {}, {}
    for name, index in fixed.items():
        step = choices["time"]["step"] if name == "time" else primary_step
        identity = (index.tobytes(), step)
        if identity not in previously:
            previously[identity] = name
            bank[name] = index
            methods[name] = {"kind": "index", "width": width, "steps": [step]}
        aliases[name] = previously[identity]
    for k in ("dense", "pooled", "narrow"):
        methods[k] = {
            "kind": "pooled" if k == "pooled" else "dense",
            "width": budget // inputs if k == "narrow" else width,
            "steps": [choices[k]["step"]],
        }
        aliases[k] = k
    np.savez_compressed(folder / "frozen_indices.npz", **bank)
    save_json(
        folder / "frozen.json",
        {
            "plan_sha256": digest(PLAN),
            "core_sha256": digest(folder / "core_choice.json"),
            "compat_sha256": digest(folder / "compat_choice.json"),
            "bank_sha256": digest(folder / "frozen_indices.npz"),
            "pooled_sha256": digest(folder / "pooled.npz"),
            "methods": methods,
            "aliases": aliases,
        },
    )
    print(
        dataset,
        "frozen Algorithm 2 and ERM",
        comp["strict"]["model"],
        comp["erm"]["model"],
        flush=True,
    )


def target(dataset, gpus):
    folder = ROOT / dataset
    frozen = json.loads((folder / "frozen.json").read_text())
    if (
        frozen["bank_sha256"] != digest(folder / "frozen_indices.npz")
        or frozen["pooled_sha256"] != digest(folder / "pooled.npz")
        or frozen["plan_sha256"] != digest(PLAN)
    ):
        raise ValueError("The test-stage indexators, weights or plan have changed")
    for shots in (2, 5):
        evaluate(
            dataset,
            f"target_{shots}",
            "target",
            shots,
            frozen["methods"],
            "frozen_indices.npz",
            gpus,
            "frozen.json",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("pen", "spoken"))
    parser.add_argument("stage", choices=("core", "compat", "target", "all"))
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    args = parser.parse_args()
    if args.stage in ("core", "all"):
        select_core(args.dataset, args.gpus)
    if args.stage in ("compat", "all"):
        select_compatible(args.dataset, args.gpus)
        freeze(args.dataset)
    if args.stage in ("target", "all"):
        target(args.dataset, args.gpus)
