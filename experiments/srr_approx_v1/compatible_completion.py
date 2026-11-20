"""Separate support for a partial structure from selection of a transferable completion."""

import argparse

import numpy as np

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from experiments.srr_approx_v1.compatible import LOCAL, OUTPUT, RelationBank
from experiments.srr_approx_v1.compatible_replication import DEST as PREVIOUS
from experiments.srr_approx_v1.compatible_replication import interval, name
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


DEST = OUTPUT / "completion"
DATA = LOCAL / "completion"
DEPLOY_SEEDS = (61041801, 61041821, 61041841)


def prepare(setting):
    DEST.mkdir(parents=True, exist_ok=True)
    original = PREVIOUS / f"{setting}_indices.npz"
    prior_plan = read_json(PREVIOUS / f"{setting}_plan.json")
    if digest(original) != prior_plan["bank_sha256"]:
        raise ValueError("Development masks differ from the previous frozen bank")
    prior_scores = PREVIOUS / f"{setting}_development_scores.npz"
    with np.load(prior_scores) as scores:
        risk = np.stack(
            [
                np.mean([scores[name(j, m)][:, 2] for j in (1, 2, 3)], axis=0)
                for m in NAMES
            ],
            axis=1,
        )
    bank = RelationBank([set(enumerate(row)) for row in CHOICES])
    found = bank.infer(risk, 0.01, 0.90)
    eligible = [
        mask
        for mask, row in zip(NAMES, bank.full, strict=True)
        if any(s <= row for s in found["maximal"])
    ]
    _, base, errors = source_input(setting)
    deployment = {}
    for j, seed in enumerate(DEPLOY_SEEDS, start=1):
        learned, neutral = block_bank(base, errors, seed)
        for mask in NAMES:
            deployment[name(j, mask)] = learned[mask]
            deployment[name(j, mask, True)] = neutral[mask]
    if setting == "omniglot10":
        with np.load(original) as f:
            deployment["wsr_step_75"] = f["wsr_step_75"]
    path = DEST / f"{setting}_deployment.npz"
    np.savez_compressed(path, **deployment)
    save_json(
        DEST / f"{setting}_plan.json",
        {
            "discovery_scores_sha256": digest(prior_scores),
            "selection_bank_sha256": digest(original),
            "deployment_sha256": digest(path),
            "deployment_seeds": DEPLOY_SEEDS,
            "discovery_tasks": len(risk),
            "fit_seeds_for_selection": prior_plan["bank_seeds"][1:],
            "eligible": eligible,
            "maximal": [sorted(s) for s in found["maximal"]],
            "discovery_coverage_winner": NAMES[found["fixed"]],
            "tolerance": 0.01,
            "coverage": 0.90,
            "development_seeds": list(range(61041901, 61041905)),
            "target_seeds": list(range(61042001, 61042017)),
            "episodes_per_panel": 96,
            "choice": "maximize mean source-development episode accuracy over three masks; tie-break by mean half-Brier and then lexicographically",
            "primary": "mean of three unseen mask seeds versus ordinary aSRR on fresh episode accuracy, 97.5% per-domain intervals",
            "controls": "unrestricted same-bank ERM, identical blocks with uniform exceptions, single-stage coverage rule and (Omniglot) archived WSR",
        },
    )


def study(setting, phase, gpus):
    plan = read_json(DEST / f"{setting}_plan.json")
    if phase == "development":
        path = PREVIOUS / f"{setting}_indices.npz"
        keys = [name(j, mask) for j in (1, 2, 3) for mask in NAMES]
        expected = plan["selection_bank_sha256"]
    else:
        path = DEST / f"{setting}_deployment.npz"
        policy = read_json(DEST / f"{setting}_selection.json")
        keys = [BASE]
        for j in (1, 2, 3):
            keys.extend(name(j, policy[key]) for key in ("fixed", "erm", "coverage"))
            keys.append(name(j, policy["fixed"], True))
        if setting == "omniglot10":
            keys.append("wsr_step_75")
        expected = plan["deployment_sha256"]
    if digest(path) != expected:
        raise ValueError("An indexator changed after registration")
    with np.load(path) as f:
        bank = {k: f[k] for k in dict.fromkeys(keys)}
    uniform_controls = []
    if setting == "omniglot10" and phase == "target":
        # Retain the four historical random masks as secondary paired controls.
        # Their identities and file hash are recorded before target prediction.
        archive = OUTPUT / f"{setting}_bank.npz"
        references = read_json(OUTPUT / f"{setting}_plan.json")
        if digest(archive) != references["bank_sha256"]:
            raise ValueError("The archived random controls have changed")
        uniform_controls = sorted(
            k for k in references["controls"] if k.startswith("balanced_random_mask_")
        )
        if len(uniform_controls) != 4:
            raise ValueError("The comparison requires all four archived random masks")
        save_json(
            DEST / f"{setting}_uniform_controls.json",
            {
                "bank_sha256": references["bank_sha256"],
                "names": uniform_controls,
                "effect": "paired episode accuracy against the mean of the four frozen masks",
            },
        )
        with np.load(archive) as f:
            bank.update({k: f[k] for k in uniform_controls})
    rows = panel_scores(setting, phase, gpus, plan, bank, DEST, DATA)
    if phase == "development":
        means = {
            mask: np.mean([rows[name(j, mask)].mean(axis=0) for j in (1, 2, 3)], axis=0)
            for mask in NAMES
        }

        def rank(m):
            return -means[m][0], means[m][2], m

        policy = {
            "fixed": min(plan["eligible"], key=rank),
            "erm": min(NAMES, key=rank),
            "coverage": plan["discovery_coverage_winner"],
            "mean_accuracy_ce_brier": {m: v.tolist() for m, v in means.items()},
        }
        save_json(DEST / f"{setting}_selection.json", policy)
        print(setting, policy["fixed"], policy["erm"], flush=True)
    else:
        for key in ("fixed", "erm", "coverage"):
            rows[key] = np.mean([rows[name(j, policy[key])] for j in (1, 2, 3)], axis=0)
        rows["neutral"] = np.mean(
            [rows[name(j, policy["fixed"], True)] for j in (1, 2, 3)], axis=0
        )
        if uniform_controls:
            rows["four_uniform_mean"] = np.mean(
                [rows[k] for k in uniform_controls], axis=0
            )
        diff = rows["fixed"][:, 0] - rows[BASE][:, 0]
        report = {
            "episodes": len(diff),
            "fixed": policy["fixed"],
            "erm": policy["erm"],
            "accuracy_ce_brier": {k: v.mean(axis=0).tolist() for k, v in rows.items()},
            "minus_baseline_ci975": interval(diff, 0.975),
            "paired_ci95": {
                k: interval(rows["fixed"][:, 0] - v[:, 0]) for k, v in rows.items()
            },
            "per_seed_minus_baseline": [
                interval(rows[name(j, policy["fixed"])][:, 0] - rows[BASE][:, 0])
                for j in (1, 2, 3)
            ],
        }
        save_json(DEST / f"{setting}_results.json", report)
        print(setting, report["minus_baseline_ci975"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("prepare", "development", "target"))
    p.add_argument("setting", choices=SETTINGS)
    p.add_argument("--gpus", type=int, nargs="+", default=[0])
    args = p.parse_args()
    if args.phase == "prepare":
        prepare(args.setting)
    else:
        study(args.setting, args.phase, args.gpus)
