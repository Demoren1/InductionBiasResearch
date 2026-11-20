"""Select a clean/augmented gradient mixture at the fixed Aug-Omniglot rank."""

import argparse
import json
import multiprocessing as mp
from functools import partial
from pathlib import Path

import numpy as np

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json

from .classification import SETTINGS, grid, key_for, read_data
from .gradient_groups import gradient_refine
from .gradient_transfer import CONFIG, basis_for, evaluate, source_kernels
from .groups import LOCAL, ROOT
from .linear import atomic_npz, tolerance_interval
from .panels import OMNIGLOT

DATASET = "aug_omniglot"
ROOT_OUT = ROOT / "gradient_mixture"
LOCAL_OUT = LOCAL / "gradient_mixture"
OUT = ROOT_OUT / DATASET
PREVIOUS = ROOT / "gradient_asrr" / DATASET
SHARES = (0, 25, 50, *range(100, 901, 50), 925, 950, 975, 990, 1000)
SHOTS = ("5", "1")


def clean_source_queries():
    work = LOCAL / DATASET
    base, draws = OMNIGLOT / "train.npy", work / "source_draws.npz"
    registration = json.loads((ROOT / DATASET / "panel_registration.json").read_text())
    source = registration["images"]["source"]
    if {"images": digest(base), "draws": digest(draws)} != source:
        raise ValueError("Changed Aug-Omniglot source images or draw indices")
    images = np.load(base, mmap_mode="r")
    draw = read_data(draws)
    episodes = np.stack(
        [
            np.stack(
                [
                    np.rot90(images[i, ids, 0], int(turn), axes=(-2, -1))
                    for i, turn, ids in zip(classes, turns, drawings, strict=True)
                ]
            )
            for classes, turns, drawings in zip(
                draw["classes"], draw["turns"], draw["drawings"], strict=True
            )
        ]
    ).reshape(-1, 5, 20, 1, 784)
    support = read_data(work / "source.npz")["sx"].reshape(-1, 5, 15, 1, 784)
    if not np.array_equal(episodes[:, :, :5], support[:, :, :5]):
        raise ValueError("Reconstructed clean source supports do not match")
    kept = episodes[:, :, 5:15].reshape(-1, 50, 1, 784)
    held_out = episodes[:, :, 15:].reshape(-1, 25, 1, 784)
    return np.concatenate((kept, held_out), axis=1)


def candidate(shot, share):
    return f"s{shot}_clean_{share:04d}"


def make_bank(pool, gpus):
    bank_file = OUT / "bank.npz"
    if bank_file.exists():
        return
    previous = read_data(PREVIOUS / "groups.npz")
    ward = previous["ward"]
    clean = clean_source_queries()
    kernels = pool.map(
        partial(source_kernels, clean_queries=clean),
        [(DATASET, int(s), gpus[i % len(gpus)], ward) for i, s in enumerate(SHOTS)],
        chunksize=1,
    )
    arrays, groups = {"normalized": basis_for(ward)}, {"normalized": ward}
    for shot, rows in zip(SHOTS, kernels, strict=True):
        for share in SHARES:
            alpha = share / 1000
            mixing = (1 - alpha) * rows["adapted"] + alpha * rows["clean"]
            labels = np.stack(
                [
                    gradient_refine(signal, initial)
                    for signal, initial in zip(mixing, ward, strict=True)
                ]
            )
            if share == 0 and not np.array_equal(
                labels, previous[f"s{shot}_adapted_refine"]
            ):
                raise ValueError(
                    f"Pure augmented gradient groups differ at {shot} shot"
                )
            name = candidate(shot, share)
            arrays[name], groups[name] = basis_for(labels), labels
    atomic_npz(OUT / "groups.npz", **groups)
    atomic_npz(bank_file, **arrays)
    print("Built paired-query candidates; pure augmented groups reproduced", flush=True)


def validation_rows(rows, names):
    summaries = []
    for name in names:
        for rate in SETTINGS["rates"]:
            for step in SETTINGS["steps"]:
                clean = rows[key_for(name, rate, step, "clean")].mean(0)
                aug = rows[key_for(name, rate, step, "query")].mean(0)
                summaries.append(
                    {
                        "name": name,
                        "rate": rate,
                        "step": step,
                        "clean_acc": float(clean[1]),
                        "aug_acc": float(aug[1]),
                        "ce": float((clean[0] + aug[0]) / 2),
                    }
                )
    return summaries


def best_clean(rows):
    return min(rows, key=lambda row: (-row["clean_acc"], -row["aug_acc"], row["ce"]))


def best_guarded(rows, floor):
    feasible = [row for row in rows if row["clean_acc"] >= floor - 1e-12]
    chosen = (
        min(feasible, key=lambda row: (-row["aug_acc"], -row["clean_acc"], row["ce"]))
        if feasible
        else best_clean(rows)
    )
    return {**chosen, "clean_floor": floor, "feasible": bool(feasible)}


def select(pool, gpus):
    original = json.loads((PREVIOUS / "choices.json").read_text())
    names = {shot: [candidate(shot, share) for share in SHARES] for shot in SHOTS}
    values, _ = evaluate(
        DATASET,
        "dev",
        {s: grid(["normalized", *names[s]]) for s in SHOTS},
        gpus,
        pool,
        root=ROOT_OUT,
        local_root=LOCAL_OUT,
        clean_dev=True,
    )
    all_rows = {s: validation_rows(values[s], ["normalized", *names[s]]) for s in SHOTS}
    choices, chosen_share = {}, None
    for shot, rows in all_rows.items():
        ward = [r for r in rows if r["name"] == "normalized"]
        ref = original[shot]["normalized"]
        baseline = next(
            r for r in ward if (r["rate"], r["step"]) == (ref["rate"], ref["step"])
        )
        floor = baseline["clean_acc"]
        available = (
            names[shot] if chosen_share is None else [candidate(shot, chosen_share)]
        )
        chosen = best_guarded([r for r in rows if r["name"] in available], floor)
        chosen_share = int(chosen["name"].rsplit("_", 1)[1])
        choices[shot] = {
            "mixture": {**chosen, "clean_share": chosen_share / 1000},
            "ward_guarded": best_guarded(ward, floor),
            "clean_only": best_clean(
                [r for r in rows if r["name"] == candidate(shot, 1000)]
            ),
            "normalized_reference": baseline,
        }
    save_json(OUT / "validation_grid.json", all_rows)
    save_json(OUT / "choices.json", choices)
    print("Frozen validation decisions", json.dumps(choices), flush=True)
    return choices


def predict_test(choices, gpus, pool):
    specs = {}
    for shot, rows in choices.items():
        settings = {}
        for method in ("mixture", "ward_guarded", "clean_only"):
            choice = rows[method]
            rates = settings.setdefault(choice["name"], {})
            rates.setdefault(choice["rate"], set()).add(choice["step"])
        specs[shot] = {
            name: [[rate, sorted(steps)] for rate, steps in sorted(rates.items())]
            for name, rates in settings.items()
        }
    return evaluate(
        DATASET,
        "test",
        specs,
        gpus,
        pool,
        root=ROOT_OUT,
        local_root=LOCAL_OUT,
    )


def summarize(values, units, choices):
    original = read_data(PREVIOUS / "per_episode.npz")
    if not np.array_equal(units, original["units"]):
        raise ValueError("Test episodes do not match the previous control")
    scores, raw = {}, {}
    for shot, rows in choices.items():
        scores[shot] = {}
        for kind in ("query", "clean"):
            selected = {
                method: values[shot][
                    key_for(choice["name"], choice["rate"], choice["step"], kind)
                ]
                for method, choice in rows.items()
                if method != "normalized_reference"
            }
            selected.update(
                {
                    name: original[f"s{shot}_{kind}_{key}"]
                    for name, key in (
                        ("normalized", "normalized"),
                        ("aug_only", "gradient"),
                        ("msr", "msr"),
                    )
                }
            )
            scores[shot][kind] = {
                "scores": {name: a.mean(0).tolist() for name, a in selected.items()},
                "mixture_delta_pp_95_ci": {
                    name: tolerance_interval(
                        (selected["mixture"][:, 1] - a[:, 1]) * 100
                    )
                    for name, a in selected.items()
                    if name != "mixture"
                },
            }
            raw.update({f"s{shot}_{kind}_{name}": a for name, a in selected.items()})
    save_json(OUT / "results.json", scores)
    atomic_npz(OUT / "per_episode.npz", units=units, **raw)
    print("Completed mixed-gradient comparison", flush=True)


def run(phase, gpus):
    OUT.mkdir(parents=True, exist_ok=True)
    save_json(
        OUT / "registration.json",
        {
            "code_sha256": digest(Path(__file__)),
            "kernel_code_sha256": digest(
                Path(__file__).with_name("gradient_transfer.py")
            ),
            "previous_choices_sha256": digest(PREVIOUS / "choices.json"),
            "codes": CONFIG[DATASET]["rank"] * 64,
            "shares": [share / 1000 for share in SHARES],
            "construction": "Reuse adapted Ward reference. Average paired query gradients with the given weight on clean queries, then refine Ward by the original rule.",
            "selection": "Use five-shot to pick clean share and optimizer. Require validation clean accuracy at least that of prior normalized Ward, maximize augmented accuracy. If infeasible, maximize clean accuracy. Keep share fixed for one-shot.",
            "controls": "Re-pick Ward optimizer by the same rule. Also test pure clean with its validation-best clean setting. Prior Ward, pure augmented, and MSR scores are frozen.",
        },
    )
    with mp.get_context("spawn").Pool(len(gpus)) as pool:
        if phase in ("dev", "run"):
            make_bank(pool, gpus)
            choices = select(pool, gpus)
        else:
            choices = json.loads((OUT / "choices.json").read_text())
        if phase in ("test", "run"):
            values, units = predict_test(choices, gpus, pool)
            summarize(values, units, choices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("dev", "test", "run"))
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    args = parser.parse_args()
    run(args.phase, args.gpus)
