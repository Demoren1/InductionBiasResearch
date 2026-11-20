"""Choose which Aug-Omniglot blocks use clean-query groups."""

import argparse
import json
import multiprocessing as mp
from itertools import product
from pathlib import Path

import numpy as np

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json

from .classification import grid, key_for, read_data
from .gradient_mixture import DATASET, best_guarded, candidate, validation_rows
from .gradient_mixture import OUT as MIXTURE
from .gradient_transfer import basis_for, evaluate
from .groups import LOCAL, ROOT
from .linear import atomic_npz, tolerance_interval

SHOT = "5"
ROOT_OUT = ROOT / "gradient_blocks"
LOCAL_OUT = LOCAL / "gradient_blocks"
OUT = ROOT_OUT / DATASET
ASSIGNMENTS = tuple(
    "".join(types) for types in product("ac", repeat=4) if types.count("c") in (1, 2, 3)
)


def make_bank():
    if (OUT / "bank.npz").exists():
        return
    previous = read_data(MIXTURE / "groups.npz")
    aug, clean = (previous[candidate(SHOT, share)] for share in (0, 1000))
    maps = {
        name: np.where(np.array(list(name))[:, None] == "c", clean, aug)
        for name in ASSIGNMENTS
    }
    bases = {name: basis_for(labels) for name, labels in maps.items()}
    if any(array.shape != (4, 784, 32) for array in bases.values()):
        raise ValueError("Every assignment must use four 32-group partitions")
    atomic_npz(OUT / "groups.npz", **maps)
    atomic_npz(OUT / "bank.npz", **bases)


def select(gpus, pool):
    values, _ = evaluate(
        DATASET,
        "dev",
        {SHOT: grid(ASSIGNMENTS)},
        gpus,
        pool,
        root=ROOT_OUT,
        local_root=LOCAL_OUT,
        clean_dev=True,
    )
    rows = validation_rows(values[SHOT], ASSIGNMENTS)
    old_choices = json.loads((MIXTURE / "choices.json").read_text())[SHOT]
    ward = old_choices["normalized_reference"]
    choice = best_guarded(rows, ward["clean_acc"])
    choices = {
        "selected": choice,
        "clean_blocks": [i + 1 for i, t in enumerate(choice["name"]) if t == "c"],
        "augmented_blocks": [i + 1 for i, t in enumerate(choice["name"]) if t == "a"],
        "by_clean_blocks": {
            count: best_guarded(
                [row for row in rows if row["name"].count("c") == count],
                ward["clean_acc"],
            )
            for count in range(1, 4)
        },
        "ward": ward,
        "mixture": old_choices["mixture"],
    }
    save_json(OUT / "validation_grid.json", rows)
    save_json(OUT / "choices.json", choices)
    print("Frozen validation choices", json.dumps(choices), flush=True)
    return choices


def test(choices, gpus, pool):
    chosen = choices["selected"]
    name, rate, step = chosen["name"], chosen["rate"], chosen["step"]
    values, units = evaluate(
        DATASET,
        "test",
        {SHOT: {name: [[rate, [step]]]}},
        gpus,
        pool,
        root=ROOT_OUT,
        local_root=LOCAL_OUT,
    )
    previous = read_data(MIXTURE / "per_episode.npz")
    if not np.array_equal(units, previous["units"]):
        raise ValueError("Test episodes changed")
    report, raw = {}, {"units": units}
    for kind in ("clean", "query"):
        mixed_blocks = values[SHOT][key_for(name, rate, step, kind)]
        pairs = {
            control: previous[f"s{SHOT}_{kind}_{control}"]
            for control in ("normalized", "mixture", "aug_only", "clean_only", "msr")
        }
        report[kind] = {
            "scores": {
                label: value.mean(0).tolist()
                for label, value in {"mixed_blocks": mixed_blocks, **pairs}.items()
            },
            "mixed_blocks_delta_pp_95_ci": {
                label: tolerance_interval((mixed_blocks[:, 1] - other[:, 1]) * 100)
                for label, other in pairs.items()
            },
            "episodes": len(units),
        }
        raw.update(
            {
                f"{kind}_{k}": v
                for k, v in {"mixed_blocks": mixed_blocks, **pairs}.items()
            }
        )
    save_json(OUT / "results.json", report)
    atomic_npz(OUT / "per_episode.npz", **raw)
    print("Mixed blocks complete", flush=True)


def run(phase, gpus):
    OUT.mkdir(exist_ok=True, parents=True)
    save_json(
        OUT / "registration.json",
        {
            "code_sha256": digest(Path(__file__)),
            "parent_groups_sha256": digest(MIXTURE / "groups.npz"),
            "prior_validation_choices_sha256": digest(MIXTURE / "choices.json"),
            "shot": int(SHOT),
            "codes": 2048,
            "assignments": ASSIGNMENTS,
            "selection": "Each of the 14 mixed block assignments uses the unchanged "
            "clean or augmented group partition for every corresponding block. "
            "Maximize validation augmented accuracy subject to clean accuracy "
            "at least that of normalized Ward. If infeasible, maximize clean "
            "accuracy. Tune the same two rates and four step counts.",
            "evaluation": "Evaluate only the chosen assignment, original paired "
            "clean/augmented queries. Previously frozen endpoints are controls.",
        },
    )
    with mp.get_context("spawn").Pool(len(gpus)) as pool:
        if phase in ("dev", "run"):
            make_bank()
            choices = select(gpus, pool)
        else:
            choices = json.loads((OUT / "choices.json").read_text())
        if phase in ("test", "run"):
            test(choices, gpus, pool)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("dev", "test", "run"))
    parser.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    args = parser.parse_args()
    run(args.phase, args.gpus)
