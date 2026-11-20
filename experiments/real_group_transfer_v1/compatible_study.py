"""Prospective FC100 and NinaPro evaluation of aSRR Algorithm 2."""

import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t as student

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from experiments.srr_approx_v1.compatible_transfer import NAMES, block_bank, residuals
from experiments.srr_approx_v1.core import canonical
from experiments.wsr_omniglot_v1.source import read_json as read

from .compatible_relations import maximal_supported, pair_counts, relation_intersections
from .data import (
    NINAPRO_SUBJECTS,
    fc100_panel,
    ninapro_panel,
    ninapro_source_normalization,
)
from .optimization import fit_target_logits
from .study import DATA, LAYOUTS, OUTPUT as PILOT, TARGET_STEPS, source_profiles


PLAN = Path(__file__).with_name("compatibility_plan.json")
ROOT = Path("artifacts/asrr_compatible_fc100_ninapro_2026-11-19")
LOCAL = ROOT.parent / "local" / ROOT.name
COMPENSATED = Path(
    "artifacts/asrr_compensated_merging_2026-11-19/fc100/global_indexators.npz"
)
PACK = 8


def model_key(j, policy, uniform=False):
    if policy in ("0000", "ward"):
        return policy
    return f"{'uniform' if uniform else 'bank'}{j}_{policy}"


def prepare(dataset):
    folder = ROOT / dataset
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "registration.json").exists():
        raise FileExistsError(folder / "registration.json")
    plan = read(PLAN)
    previous = PILOT / "tuning" / dataset
    config = read(previous / "registration.json")
    source = DATA.parent / dataset / "dense_fits.npz"
    if digest(source) != config["source_dense_sha256"]:
        raise ValueError("Source fits differ from the archived aSRR benchmark")
    with np.load(source) as fits:
        z = source_profiles(fits["parameters"], LAYOUTS[dataset])
    codes = plan[dataset]["codes"]
    with np.load(previous / "candidates.npz") as f:
        base = f[f"input_p4_m{codes}"]
    with np.load(previous / "frozen_indexators.npz") as f:
        ordinary = f["asrr"]
    policies = list(NAMES)
    archive = {}
    if dataset == "fc100":
        policies.append("ward")
        archive["ward"] = ordinary
        with np.load(COMPENSATED) as f:
            archive["compensated"] = f["global_covariance"]
    elif not np.array_equal(base, ordinary):
        raise ValueError("NinaPro ordinary aSRR should be the four-partition model")
    error = residuals(z, base)
    for j, seed in enumerate(plan["proposal_seeds"]):
        candidates, uniforms = block_bank(base, error, seed)
        if not np.array_equal(candidates["0000"], canonical(base)):
            raise ValueError(
                "The four unmodified blocks must reproduce the source indexator"
            )
        for name in NAMES:
            archive[model_key(j, name)] = candidates[name]
            archive[model_key(j, name, True)] = uniforms[name]
        maps, extends = relation_intersections(
            [archive[model_key(j, m)] for m in policies]
        )
        np.savez_compressed(
            folder / f"relations_{j}.npz",
            maps=maps,
            extends=extends,
            pairs=pair_counts(maps),
        )
    masks_path = PILOT / "tuning" / "shared_code" / dataset / "indexators.npz"
    with np.load(masks_path) as masks:
        for j in range(4):
            archive[f"mask_{j}"] = masks[f"mask_{j}"]
    for k, index in archive.items():
        if not np.array_equal(np.unique(abs(index)), np.arange(1, codes + 1)):
            raise ValueError(f"Incorrect occupied code budget for {k}")
    path = folder / "bank.npz"
    np.savez_compressed(path, **archive)
    save_json(
        folder / "registration.json",
        {
            "plan_sha256": digest(PLAN),
            "bank_sha256": digest(path),
            "source_fits_sha256": digest(source),
            "archived_asrr_sha256": digest(previous / "frozen_indexators.npz"),
            "archived_masks_sha256": digest(masks_path),
            "policies": policies,
            "ordinary": "ward" if dataset == "fc100" else "0000",
            "relation_sha256": [
                digest(folder / f"relations_{j}.npz")
                for j in range(len(plan["proposal_seeds"]))
            ],
            "target_optimizer_sha256": digest(
                Path(__file__).with_name("optimization.py")
            ),
            "stages": plan[dataset],
            "codes": codes,
            "steps": TARGET_STEPS,
        },
    )
    print(dataset, len(policies), "candidates per bank, registered", flush=True)


def prepare_panels(dataset, phase, config, directory):
    directory.mkdir(parents=True, exist_ok=True)
    split = "test" if phase == "target" else "development"
    if dataset == "fc100":
        with np.load(DATA / "fc100.npz") as f:
            panel = fc100_panel(
                dict(f),
                split,
                config["phase_counts"][phase],
                config["episode_seeds"][phase],
            )
        batch = config["batch"]
    else:
        source = DATA / "ninapro"
        panel = ninapro_panel(source, split, *ninapro_source_normalization(source))
        if phase != "target":
            dev = NINAPRO_SUBJECTS["development"]
            subjects = dev[:4] if phase == "discovery" else dev[4:]
            panel = panel.subset(np.isin(panel.units, subjects))
        batch = 15
    rng = np.random.default_rng(config["numeric_seeds"][phase] + 53000)
    orders, answers, batch_ids = [], [], []
    for i, start in enumerate(range(0, len(panel.units), batch)):
        rows = slice(start, start + batch)
        query = panel.query_x[rows]
        order = np.stack([rng.permutation(query.shape[1]) for _ in range(len(query))])
        select = (np.arange(len(query))[:, None], order)
        np.savez(
            directory / f"public_{i}.npz",
            sx=panel.support_x[rows],
            sy=panel.support_y[rows],
            qx=query[select],
        )
        orders.append(order)
        answers.append(panel.query_y[rows][select])
        batch_ids.extend([i] * len(query))
    np.savez_compressed(
        directory / "answers.npz",
        query_y=np.concatenate(answers),
        order=np.concatenate(orders),
        records=panel.records,
        units=panel.units,
        batch=np.asarray(batch_ids),
    )
    return len(orders)


def predict_packs(gpu, jobs, dataset, seed, bank_path, inputs, outputs):
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    layout = LAYOUTS[dataset]
    with np.load(bank_path) as archive:
        models = {
            k: archive[k]
            for k in {k for _, _, names in jobs for k in names}
            if k != "dense"
        }
    for batch, group, names in jobs:
        output = outputs / f"pred_{batch}_{group}.npz"
        if output.exists():
            continue
        with np.load(inputs / f"public_{batch}.npz") as f:
            sx = torch.as_tensor(f["sx"], dtype=torch.float32, device=device)
            sy = torch.as_tensor(f["sy"], dtype=torch.long, device=device)
            qx = torch.as_tensor(f["qx"], dtype=torch.float32, device=device)
        probs = {}
        for name in names:
            logits, _, _ = fit_target_logits(
                sx,
                sy,
                qx,
                layout,
                models.get(name),
                device,
                steps=TARGET_STEPS,
                seed=seed + batch,
            )
            probs[name] = logits[:, 0].softmax(-1).cpu().numpy()
        temporary = output.with_suffix(".partial.npz")
        np.savez(temporary, **probs)
        temporary.replace(output)


def score_predictions(folder, directory, phase, packs, batches, answers):
    with np.load(answers) as a:
        target, units, batch_ids = a["query_y"], a["units"], a["batch"]
    scores = {name: [] for pack in packs for name in pack}
    for i in range(batches):
        y = target[batch_ids == i]
        for group, pack in enumerate(packs):
            with np.load(directory / f"pred_{i}_{group}.npz") as f:
                for name in pack:
                    prob = f[name]
                    true_prob = np.take_along_axis(prob, y[..., None], axis=-1)[..., 0]
                    accuracy = (prob.argmax(-1) == y).mean(axis=1)
                    ce = -np.log(np.maximum(true_prob, 1e-30)).mean(axis=1)
                    brier = ((prob**2).sum(axis=-1) - 2 * true_prob + 1).mean(
                        axis=1
                    ) / 2
                    scores[name].append(np.stack([accuracy, ce, brier], axis=1))
    np.savez_compressed(
        folder / f"{phase}_scores.npz",
        units=units,
        batch=batch_ids,
        **{k: np.concatenate(v) for k, v in scores.items()},
    )


def evaluate_packs(
    dataset, phase, gpus, packs, batches, seed, bank_path, inputs, outputs
):
    folder = ROOT / dataset
    outputs.mkdir(parents=True, exist_ok=True)
    jobs = [(b, j, pack) for b in range(batches) for j, pack in enumerate(packs)]
    with ProcessPoolExecutor(
        max_workers=len(gpus), mp_context=mp.get_context("spawn")
    ) as pool:
        list(
            pool.map(
                predict_packs,
                gpus,
                [jobs[i :: len(gpus)] for i in range(len(gpus))],
                repeat(dataset),
                repeat(seed),
                repeat(bank_path),
                repeat(inputs),
                repeat(outputs),
            )
        )
    save_json(
        folder / f"{phase}_prediction_hashes.json",
        {f"{b}_{j}": digest(outputs / f"pred_{b}_{j}.npz") for b, j, _ in jobs},
    )
    score_predictions(folder, outputs, phase, packs, batches, inputs / "answers.npz")


def evaluate(dataset, phase, gpus):
    folder, directory = ROOT / dataset, LOCAL / dataset / phase
    reg, plan = read(folder / "registration.json"), read(PLAN)
    if (
        reg["plan_sha256"] != digest(PLAN)
        or reg["bank_sha256"] != digest(folder / "bank.npz")
        or reg["target_optimizer_sha256"]
        != digest(Path(__file__).with_name("optimization.py"))
    ):
        raise ValueError("Frozen study or numerical protocol has changed")
    banks = range(len(plan["proposal_seeds"]))
    keys = [model_key(j, m) for j in banks for m in reg["policies"]]
    chosen_hash = None
    if phase == "target":
        selection = folder / "selection.json"
        decisions = read(selection)["decisions"]
        chosen_hash = digest(selection)
        keys += [model_key(j, d["strict"], True) for j, d in enumerate(decisions)]
        keys += ["dense", *(f"mask_{j}" for j in range(4))]
        if dataset == "fc100":
            keys.append("compensated")
    keys = list(dict.fromkeys(keys))
    packs = [keys[i : i + PACK] for i in range(0, len(keys), PACK)]
    config = plan[dataset]
    batches = prepare_panels(dataset, phase, config, directory)
    frozen = folder / f"{phase}_manifest.json"
    manifest = {
        "registration_sha256": digest(folder / "registration.json"),
        "selection_sha256": chosen_hash,
        "keys": keys,
        "batches": batches,
        "answers_sha256": digest(directory / "answers.npz"),
        "pack_size": PACK,
    }
    if frozen.exists() and manifest != read(frozen):
        raise ValueError("Phase inputs differ from cached predictions")
    save_json(frozen, manifest)
    evaluate_packs(
        dataset,
        phase,
        gpus,
        packs,
        batches,
        config["numeric_seeds"][phase],
        folder / "bank.npz",
        directory,
        directory,
    )
    print(dataset, phase, len(keys), "models,", batches, "batches", flush=True)


def unit_scores(folder, phase):
    with np.load(folder / f"{phase}_scores.npz") as f:
        units = f["units"]
        scores = {k: f[k] for k in f.files if k not in ("units", "batch")}
        if folder.name == "fc100":
            return scores
        rows = [np.flatnonzero(units == s) for s in np.unique(units)]
        return {
            k: np.stack([v[ids].mean(axis=0) for ids in rows])
            for k, v in scores.items()
        }


def versions(scores, policies, j, epsilon):
    risk = np.stack([scores[model_key(j, m)][:, 2] for m in policies], axis=1)
    return risk <= risk.min(axis=1, keepdims=True) + epsilon


def select(dataset):
    folder = ROOT / dataset
    plan, reg = read(PLAN), read(folder / "registration.json")
    policies = reg["policies"]
    discovery, selection = [
        unit_scores(folder, phase) for phase in ("discovery", "selection")
    ]
    decisions, frozen = [], {}
    with np.load(folder / "bank.npz") as bank:
        for j in range(len(plan["proposal_seeds"])):
            path = folder / f"relations_{j}.npz"
            if digest(path) != reg["relation_sha256"][j]:
                raise ValueError("Physical relations differ from the registered bank")
            with np.load(path) as f:
                maps, extends, pairs = f["maps"], f["extends"], f["pairs"]
            good = versions(
                discovery, policies, j, plan["tolerance_half_multiclass_brier"]
            )

            def rank(i):
                acc, _, brier = selection[model_key(j, policies[i])].mean(axis=0)
                return -acc, brier, policies[i]

            best = min(range(len(policies)), key=rank)
            row = {
                "erm": policies[best],
                "universal_pairs": int(pairs[extends.all(axis=1)].item()),
                "median_versions": float(np.median(good.sum(axis=1))),
                "intersections": len(maps),
            }
            for rule, coverage in (
                ("strict", plan["primary_coverage"]),
                ("trimmed", plan["sensitivity_coverage"]),
            ):
                frontier = maximal_supported(good, extends, coverage)
                eligible = np.flatnonzero(extends[frontier].any(axis=0))
                chosen = min(eligible, key=rank)
                row.update(
                    {
                        rule: policies[chosen],
                        f"{rule}_frontier": frontier,
                        f"{rule}_pairs": pairs[frontier].tolist(),
                        f"{rule}_eligible": [policies[i] for i in eligible],
                    }
                )
                frozen[f"{rule}_{j}"] = bank[model_key(j, policies[chosen])]
            decisions.append(row)
    np.savez_compressed(folder / "fixed_indices.npz", **frozen)
    save_json(
        folder / "selection.json",
        {
            "decisions": decisions,
            "selection_scores_sha256": digest(folder / "selection_scores.npz"),
            "discovery_scores_sha256": digest(folder / "discovery_scores.npz"),
            "fixed_sha256": digest(folder / "fixed_indices.npz"),
        },
    )
    print(
        dataset, "strict, ERM", [(d["strict"], d["erm"]) for d in decisions], flush=True
    )


def interval(delta, confidence):
    n = len(delta)
    mean = float(np.mean(delta))
    half = float(
        student.ppf((1 + confidence) / 2, n - 1) * np.std(delta, ddof=1) / np.sqrt(n)
    )
    return [mean, mean - half, mean + half]


def paired_accuracy(a, b, ids, confidence):
    delta = a[:, 0] - b[:, 0]
    return interval(
        np.asarray([delta[ids == i].mean() for i in np.unique(ids)]), confidence
    )


def summarize(dataset):
    folder = ROOT / dataset
    plan, reg = read(PLAN), read(folder / "registration.json")
    decisions = read(folder / "selection.json")["decisions"]
    scores = unit_scores(folder, "target")
    rows = {
        "ordinary": scores[reg["ordinary"]],
        "dense": scores["dense"],
        "four_uniform": np.mean([scores[f"mask_{j}"] for j in range(4)], axis=0),
    }
    if dataset == "fc100":
        rows["compensated"] = scores["compensated"]
    for rule in ("strict", "erm", "trimmed", "uniform_selected"):
        rows[rule] = np.mean(
            [
                scores[
                    model_key(
                        j,
                        row["strict" if rule == "uniform_selected" else rule],
                        rule == "uniform_selected",
                    )
                ]
                for j, row in enumerate(decisions)
            ],
            axis=0,
        )
    diagnostics = []
    for j, row in enumerate(decisions):
        good = versions(
            scores, reg["policies"], j, plan["tolerance_half_multiclass_brier"]
        )
        with np.load(folder / f"relations_{j}.npz") as f:
            extends = f["extends"]
        diagnostics.append(
            {
                "median_versions": float(np.median(good.sum(axis=1))),
                "fixed_coverage": float(
                    good[:, reg["policies"].index(row["strict"])].mean()
                ),
                "strict_witness_coverage": (good @ extends[row["strict_frontier"]].T)
                .mean(axis=0)
                .tolist(),
                "trimmed_witness_coverage": (good @ extends[row["trimmed_frontier"]].T)
                .mean(axis=0)
                .tolist(),
            }
        )
    if dataset == "fc100":
        with np.load(folder / "target_scores.npz") as f:
            ids = f["batch"]
    else:
        ids = np.arange(len(rows["ordinary"]))

    comparisons = {
        k: paired_accuracy(rows["strict"], v, ids, 0.95) for k, v in rows.items()
    }
    result = {
        "accuracy_ce_halfbrier": {k: v.mean(axis=0).tolist() for k, v in rows.items()},
        "strict_minus_ordinary_ci975": paired_accuracy(
            rows["strict"], rows["ordinary"], ids, 0.975
        ),
        "strict_minus_controls_ci95": comparisons,
        "paired_units": len(np.unique(ids)),
        "per_bank_minus_ordinary_ci95": [
            paired_accuracy(
                scores[model_key(j, d["strict"])], rows["ordinary"], ids, 0.95
            )
            for j, d in enumerate(decisions)
        ],
        "target_compatibility": diagnostics,
    }
    save_json(folder / "results.json", result)
    print(dataset, result["strict_minus_ordinary_ci975"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "develop", "target"))
    parser.add_argument("dataset", choices=("fc100", "ninapro"))
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.dataset)
    elif args.stage == "develop":
        for phase in ("discovery", "selection"):
            evaluate(args.dataset, phase, args.gpus)
        select(args.dataset)
    else:
        evaluate(args.dataset, "target", args.gpus)
        summarize(args.dataset)
