"""Replicate the frozen selection rule with new proposal masks and new episodes."""

import argparse
from statistics import NormalDist

import numpy as np

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from experiments.srr_approx_v1.compatible import LOCAL, OUTPUT, RelationBank
from experiments.srr_approx_v1.compatible_transfer import (
    BASE,
    CHOICES,
    NAMES,
    SETTINGS,
    block_bank,
    panel_scores,
    read_json,
    source_input,
)


DEST = OUTPUT / "replication"
DATA = LOCAL / "replication"
SEEDS = (731119101, 923191001, 923191011, 923191021)


def name(j, mask, neutral=False):
    if mask == BASE:
        return BASE
    return f"{'neutral' if neutral else 'bank'}{j}_{mask}"


def prepare(setting):
    DEST.mkdir(parents=True, exist_ok=True)
    _, base, errors = source_input(setting)
    archive = {}
    for j, seed in enumerate(SEEDS):
        learned, uniform = block_bank(base, errors, seed)
        for mask in NAMES:
            archive[name(j, mask)] = learned[mask]
            archive[name(j, mask, True)] = uniform[mask]
    original = OUTPUT / f"{setting}_bank.npz"
    previous = read_json(OUTPUT / f"{setting}_selection.json")
    with np.load(original) as f:
        for k in NAMES:
            if not np.array_equal(f[k], archive[name(0, k)]):
                raise ValueError("Replication must reproduce every original candidate")
        if setting == "omniglot10":
            archive["wsr_step_75"] = f["wsr_step_75"]
    path = DEST / f"{setting}_indices.npz"
    np.savez_compressed(path, **archive)
    save_json(
        DEST / f"{setting}_plan.json",
        {
            "bank_seeds": SEEDS,
            "original_chosen": previous["fixed"],
            "original_erm": previous["erm"],
            "development_seeds": [952191121, 952191122],
            "target_seeds": list(range(952191201, 952191217)),
            "episodes_per_panel": 96,
            "coverage": 0.90,
            "tolerance": 0.01,
            "bank_sha256": digest(path),
            "primary": "episode-wise mean accuracy of three newly selected banks minus ordinary aSRR",
            "two_dataset_intervals": "97.5% marginal Bonferroni intervals for two primary differences",
            "secondary": "original frozen selection; same-bank mean Brier; uniform exceptions on identical selected blocks",
        },
    )


def interval(delta, confidence=0.95):
    critical = NormalDist().inv_cdf(0.5 + confidence / 2)
    half = critical * delta.std(ddof=1) / np.sqrt(len(delta))
    return [float(delta.mean()), float(delta.mean() - half), float(delta.mean() + half)]


def study(setting, phase, gpus):
    plan = read_json(DEST / f"{setting}_plan.json")
    file = DEST / f"{setting}_indices.npz"
    if digest(file) != plan["bank_sha256"]:
        raise ValueError("Frozen candidate bank changed")
    if phase == "development":
        keys = [name(j, m) for j in range(1, len(SEEDS)) for m in NAMES]
    else:
        policies = read_json(DEST / f"{setting}_selection.json")
        keys = [BASE]
        for j, row in enumerate(policies):
            keys.extend(
                (
                    name(j, row["fixed"]),
                    name(j, row["erm"]),
                    name(j, row["fixed"], True),
                )
            )
        if setting == "omniglot10":
            keys.append("wsr_step_75")
    with np.load(file) as f:
        bank = {k: f[k] for k in dict.fromkeys(keys)}
    rows = panel_scores(setting, phase, gpus, plan, bank, DEST, DATA)
    if phase == "development":
        lib = RelationBank([set(enumerate(row)) for row in CHOICES])
        policies = [{"fixed": plan["original_chosen"], "erm": plan["original_erm"]}]
        for j in range(1, len(SEEDS)):
            loss = np.stack([rows[name(j, m)][:, 2] for m in NAMES], axis=1)
            found = lib.infer(loss, plan["tolerance"], plan["coverage"])
            winner = found["fixed"]
            policies.append(
                {
                    "fixed": NAMES[winner],
                    "erm": NAMES[loss.mean(axis=0).argmin()],
                    "winner_coverage": found["full_coverage"][winner],
                    "maximal": [sorted(s) for s in found["maximal"]],
                }
            )
        save_json(DEST / f"{setting}_selection.json", policies)
        print(setting, [p["fixed"] for p in policies], flush=True)
    else:
        new = list(enumerate(policies))[1:]
        fixed = np.mean([rows[name(j, p["fixed"])] for j, p in new], axis=0)
        neutral = np.mean([rows[name(j, p["fixed"], True)] for j, p in new], axis=0)
        erm = np.mean([rows[name(j, p["erm"])] for j, p in new], axis=0)
        rows.update(new_selected=fixed, new_neutral_same_blocks=neutral, new_erm=erm)
        a = rows[BASE][:, 0]
        frozen = rows[name(0, policies[0]["fixed"])][:, 0]
        report = {
            "accuracy_ce_brier": {k: v.mean(axis=0).tolist() for k, v in rows.items()},
            "new_minus_baseline_ci95": interval(fixed[:, 0] - a),
            "new_minus_baseline_ci975": interval(fixed[:, 0] - a, 0.975),
            "new_minus_neutral_ci95": interval(fixed[:, 0] - neutral[:, 0]),
            "new_minus_erm_ci95": interval(fixed[:, 0] - erm[:, 0]),
            "frozen_minus_baseline_ci95": interval(frozen - a),
            "per_new_bank_minus_baseline": [
                interval(rows[name(j, p["fixed"])][:, 0] - a) for j, p in new
            ],
            "episodes": len(a),
        }
        if "wsr_step_75" in rows:
            report["new_minus_wsr_ci95"] = interval(
                fixed[:, 0] - rows["wsr_step_75"][:, 0]
            )
        save_json(DEST / f"{setting}_results.json", report)
        print(setting, report["new_minus_baseline_ci975"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "development", "target"))
    parser.add_argument("setting", choices=SETTINGS)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.setting)
    else:
        study(args.setting, args.phase, args.gpus)
