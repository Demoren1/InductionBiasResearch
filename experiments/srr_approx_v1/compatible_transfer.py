"""Fit partial block proposals on development episodes and freeze a completion."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
import torch

from experiments.asrr_anchor_init_v1.transfer import public_panel
from experiments.paper_asrr_v1.extract import source_profiles
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from experiments.srr_approx_v1.compatible import LOCAL, OUTPUT, RelationBank
from experiments.srr_approx_v1.core import canonical
from experiments.srr_approx_v1.group_transfer import (
    MNIST_BANK,
    MNIST_SOURCE,
    OMNI_BANK,
    OMNI_SOURCE,
    SETTINGS,
    metrics,
    predict,
    read_json,
    shared_with_exceptions,
)


CHOICES = tuple(product((0, 1), repeat=4))
NAMES = tuple("".join(map(str, row)) for row in CHOICES)
BASE, ALT = NAMES[0], NAMES[-1]


def residuals(z, base):
    result = np.zeros(base.shape)
    for h, ids in enumerate(base.T):
        _, group, n = np.unique(ids, return_inverse=True, return_counts=True)
        v = z[:, :, h].T
        centers = np.zeros((len(n), len(z)))
        np.add.at(centers, group, v)
        error = ((v - (centers / n[:, None])[group]) ** 2).sum(axis=1)
        result[:, h] = error / max(error.mean(), 1e-12)
    return result


def join(blocks):
    offset, parts = 0, []
    for block in blocks:
        local = canonical(block)
        parts.append(np.sign(local) * (abs(local) + offset))
        offset += int(abs(local).max())
    return canonical(np.concatenate(parts, axis=1))


def source_input(setting):
    hidden, _ = SETTINGS[setting]
    if setting == "mnist16":
        source = MNIST_BANK / "asrr_input_m2048.npz"
        with np.load(MNIST_SOURCE / "source/asrr/profiles.npz") as data:
            z = data["consensus"]
    else:
        source = OMNI_BANK / "asrr_four_partitions.npz"
        _, z, _, _ = source_profiles(
            OMNI_SOURCE / "source/dense/models.npz", 784, hidden
        )
    with np.load(source) as f:
        base = f["index"]
    return source, base, residuals(z, base)


def block_bank(base, error, seed):
    hidden = base.shape[1]
    pairs, neutral = [], []
    for j, hs in enumerate(np.split(np.arange(hidden), 4)):
        sub = base[:, hs]

        def proposal(weight):
            return shared_with_exceptions(
                sub, error[:, hs], 1, True, np.random.default_rng(seed + j), weight
            )

        pairs.append((sub, proposal(1.0)))
        neutral.append(proposal(0.0))
    bank = {
        name: join([pair[v] for pair, v in zip(pairs, row, strict=True)])
        for name, row in zip(NAMES, CHOICES, strict=True)
    }
    uniform = {
        name: join(
            [
                u if v else pair[0]
                for u, pair, v in zip(neutral, pairs, row, strict=True)
            ]
        )
        for name, row in zip(NAMES, CHOICES, strict=True)
    }
    return bank, uniform


def prepare(setting):
    hidden, codes = SETTINGS[setting]
    source, base, error = source_input(setting)
    bank, uniform = block_bank(base, error, 731119101)
    bank["uniform_block"] = uniform[ALT]
    refs = {"uniform_block": "uniform allocation in the same four-block architecture"}
    if setting == "omniglot10":
        frozen = read_json(
            Path(
                "artifacts/omniglot64_gradient_profiles_2026-11-19/choices_frozen.json"
            )
        )["policies"]
        for key in (
            "wsr_step_75",
            *(f"balanced_random_mask_{s}" for s in range(7611189541, 7611189545)),
        ):
            row = frozen[key]
            path = OMNI_BANK / f"{row['name']}.npz"
            if digest(path) != row["model_sha256"]:
                raise ValueError(f"The archived reference changed: {key}")
            with np.load(path) as f:
                bank[key] = f["index"]
            refs[key] = str(path)
    if not np.array_equal(bank[BASE], canonical(base)):
        raise ValueError("All baseline blocks must reproduce the reference")
    for index in bank.values():
        if not np.array_equal(np.unique(abs(index)), np.arange(1, codes + 1)):
            raise ValueError("Non-exhaustive transfer code budget")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"{setting}_bank.npz"
    np.savez_compressed(path, **bank)
    save_json(
        OUTPUT / f"{setting}_plan.json",
        {
            "bank_sha256": digest(path),
            "original_sha256": digest(source),
            "candidates": NAMES,
            "controls": refs,
            "hidden": hidden,
            "codes": codes,
            "development_seeds": [731119201, 731119202],
            "target_seeds": list(range(731119311, 731119319)),
            "episodes_per_panel": 96,
            "loss": "mean half multiclass Brier score in [0, 1]",
            "tolerance": 0.01,
            "coverage": 0.90,
            "rule": "retain inclusion-maximal joint partials, then freeze the most frequently acceptable completion among their extensions; break ties by mean Brier and lexicographic order",
            "primary": "fixed versus original accuracy, episode-paired",
            "same_bank_control": "minimum mean development Brier",
            "additional": "strict task coverage diagnostic at the same tolerance",
        },
    )


def panel_scores(setting, phase, gpus, plan, bank, output=OUTPUT, local=LOCAL):
    torch.set_num_threads(1)
    folder = local / setting
    development = phase == "development"
    names = list(bank)
    panels, jobs = {}, []
    for n, seed in enumerate(plan[phase + "_seeds"]):
        key = f"{phase}_{n}"
        panels[key] = public_panel(
            setting,
            key,
            plan["episodes_per_panel"],
            seed,
            directory=folder,
            target=not development,
        )
        jobs.extend((key, name, seed + 13000) for name in names)
    save_json(output / f"{setting}_{phase}_panels.json", panels)

    def worker(slot):
        device = f"cuda:{gpus[slot]}"
        torch.cuda.set_device(device)
        for key, name, seed in jobs[slot :: len(gpus)]:
            with np.load(panels[key]["path"]) as f:
                panel = {k: f[k] for k in f.files if k != "query_y"}
            prob = predict(panel, bank[name], setting, seed, device)
            np.save(folder / f"{key}_{name}.npy", prob)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(worker, range(len(gpus))))
    save_json(
        output / f"{setting}_{phase}_prediction_hashes.json",
        {
            f"{key}_{name}": digest(folder / f"{key}_{name}.npy")
            for key in panels
            for name in names
        },
    )
    rows = {n: [] for n in names}
    for key, record in panels.items():
        with np.load(record["path"] if development else record["answer_path"]) as f:
            y = f["query_y" if development else "y"]
        for name in names:
            prob = np.load(folder / f"{key}_{name}.npy")
            brier = ((prob - y) ** 2).sum(axis=-1).mean(axis=1) / 2
            rows[name].append(np.column_stack((metrics(prob, y), brier)))
    rows = {k: np.concatenate(v) for k, v in rows.items()}
    np.savez_compressed(output / f"{setting}_{phase}_scores.npz", **rows)
    return rows


def evaluate(setting, phase, gpus):
    plan = read_json(OUTPUT / f"{setting}_plan.json")
    file = OUTPUT / f"{setting}_bank.npz"
    if digest(file) != plan["bank_sha256"]:
        raise ValueError("Candidate bank differs from the frozen plan")
    development = phase == "development"
    if development:
        names = list(NAMES)
    else:
        selection = read_json(OUTPUT / f"{setting}_selection.json")
        names = list(
            dict.fromkeys(
                [selection["fixed"], BASE, ALT, selection["erm"], *plan["controls"]]
            )
        )
    with np.load(file) as f:
        bank = {k: f[k] for k in names}
    rows = panel_scores(setting, phase, gpus, plan, bank)
    if development:
        risk = np.stack([rows[name][:, 2] for name in NAMES], axis=1)
        relations = RelationBank([set(enumerate(row)) for row in CHOICES])
        found = relations.infer(risk, plan["tolerance"], plan["coverage"])
        strict = relations.infer(risk, plan["tolerance"])
        report = {
            "fixed": NAMES[found["fixed"]],
            "erm": NAMES[risk.mean(axis=0).argmin()],
            "maximal": [sorted(s) for s in found["maximal"]],
            "strict_maximal": [sorted(s) for s in strict["maximal"]],
            "necessary": sorted(found["necessary"]),
            "median_acceptable_completions": float(
                np.median(found["solutions_per_task"])
            ),
            "coverage_by_completion": dict(zip(NAMES, found["full_coverage"])),
            "mean_brier": dict(zip(NAMES, risk.mean(axis=0))),
        }
        save_json(OUTPUT / f"{setting}_selection.json", report)
    else:
        if setting == "omniglot10":
            rows["four_uniform_mean"] = np.mean(
                [v for k, v in rows.items() if k.startswith("balanced_random_mask_")],
                axis=0,
            )
        chosen = selection["fixed"]
        differences = {}
        for name, values in rows.items():
            delta = rows[chosen][:, 0] - values[:, 0]
            half = 1.96 * delta.std(ddof=1) / np.sqrt(len(delta))
            differences[name] = [delta.mean(), delta.mean() - half, delta.mean() + half]
        report = {
            "selected": chosen,
            "erm": selection["erm"],
            "episodes": len(rows[chosen]),
            "accuracy_ce_brier": {k: v.mean(axis=0).tolist() for k, v in rows.items()},
            "paired_acc_difference_ci95": differences,
        }
        save_json(OUTPUT / f"{setting}_results.json", report)
    print(setting, phase, report.get("fixed", report.get("selected")), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("prepare", "development", "target"))
    p.add_argument("setting", choices=SETTINGS)
    p.add_argument("--gpus", nargs="+", type=int, default=[0])
    args = p.parse_args()
    if args.phase == "prepare":
        prepare(args.setting)
    else:
        evaluate(args.setting, args.phase, args.gpus)
