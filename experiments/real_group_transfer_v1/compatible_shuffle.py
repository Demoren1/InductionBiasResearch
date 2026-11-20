"""Refit the selected aSRR structures after a common permutation of input rows."""

import argparse
from pathlib import Path

import numpy as np

from .compatible_study import (
    LOCAL,
    PACK,
    PLAN,
    ROOT,
    digest,
    evaluate_packs,
    model_key,
    paired_accuracy,
    read,
    save_json,
    unit_scores,
)


SHUFFLE_PLAN = Path(__file__).with_name("compatible_shuffle_plan.json")
PHASE = "row_shuffle"


def prepare(dataset):
    folder = ROOT / dataset
    receipt = read(ROOT / "shuffle_registration.json")
    selection = read(folder / "selection.json")
    fixed = folder / "fixed_indices.npz"
    if (
        digest(SHUFFLE_PLAN) != receipt["spec_sha256"]
        or digest(folder / "selection.json")
        != receipt["datasets"][dataset]["selection_sha256"]
        or digest(fixed) != selection["fixed_sha256"]
    ):
        raise ValueError(
            "The permutation specification or the selected structure has changed"
        )
    manifest_path = folder / "shuffle_indices.json"
    bank_path = folder / "shuffle_bank.npz"
    if manifest_path.exists():
        manifest = read(manifest_path)
        if manifest["bank_sha256"] != digest(bank_path):
            raise ValueError("The stored permutations have changed")
        return manifest
    seeds = read(SHUFFLE_PLAN)["permutation_seeds"]
    bank, seen, by_selection = {}, {}, []
    with np.load(fixed) as saved:
        inputs = saved["strict_0"].shape[0]
        permutations = [
            np.random.default_rng(seed).permutation(inputs) for seed in seeds
        ]
        for j in range(len(selection["decisions"])):
            selected = saved[f"strict_{j}"]
            names = []
            for k, order in enumerate(permutations):
                shuffled = selected[order]
                identity = shuffled.tobytes()
                if identity not in seen:
                    key = f"shuffle_{j}_{k}"
                    bank[key] = shuffled
                    seen[identity] = key
                names.append(seen[identity])
            by_selection.append(names)
    np.savez_compressed(bank_path, **bank)
    manifest = {
        "spec_sha256": digest(SHUFFLE_PLAN),
        "fixed_sha256": digest(fixed),
        "bank_sha256": digest(bank_path),
        "permutations": [order.tolist() for order in permutations],
        "by_selection": by_selection,
        "keys": list(bank),
        "target_manifest_sha256": digest(folder / "target_manifest.json"),
    }
    save_json(manifest_path, manifest)
    return manifest


def evaluate(dataset, gpus):
    folder = ROOT / dataset
    manifest = prepare(dataset)
    primary = read(folder / "target_manifest.json")
    registration = read(folder / "registration.json")
    inputs = LOCAL / dataset / "target"
    if (
        digest(folder / "target_manifest.json") != manifest["target_manifest_sha256"]
        or digest(inputs / "answers.npz") != primary["answers_sha256"]
        or digest(PLAN) != registration["plan_sha256"]
        or digest(Path(__file__).with_name("optimization.py"))
        != registration["target_optimizer_sha256"]
    ):
        raise ValueError("Primary episodes or refitting parameters have changed")
    keys = manifest["keys"]
    packs = [keys[i : i + PACK] for i in range(0, len(keys), PACK)]
    evaluate_packs(
        dataset,
        PHASE,
        gpus,
        packs,
        primary["batches"],
        read(PLAN)[dataset]["numeric_seeds"]["target"],
        folder / "shuffle_bank.npz",
        inputs,
        LOCAL / dataset / PHASE,
    )
    print(dataset, len(keys), "permuted indexators", flush=True)


def summarize(dataset):
    folder = ROOT / dataset
    decisions = read(folder / "selection.json")["decisions"]
    manifest = read(folder / "shuffle_indices.json")
    original, shuffled = unit_scores(folder, "target"), unit_scores(folder, PHASE)
    selected = [original[model_key(j, d["strict"])] for j, d in enumerate(decisions)]
    controls = [
        np.mean([shuffled[k] for k in keys], axis=0)
        for keys in manifest["by_selection"]
    ]
    with (
        np.load(folder / "target_scores.npz") as a,
        np.load(folder / f"{PHASE}_scores.npz") as b,
    ):
        if not all(np.array_equal(a[k], b[k]) for k in ("batch", "units")):
            raise ValueError(
                "The selected structures and permutations used different tasks"
            )
        ids = a["batch"] if dataset == "fc100" else np.unique(a["units"])
    selected_mean = np.mean(selected, axis=0)
    shuffled_mean = np.mean(controls, axis=0)
    result = {
        "selected_accuracy_ce_halfbrier": selected_mean.mean(axis=0).tolist(),
        "permuted_accuracy_ce_halfbrier": shuffled_mean.mean(axis=0).tolist(),
        "selected_minus_permuted_ci95": paired_accuracy(
            selected_mean, shuffled_mean, ids, 0.95
        ),
        "per_bank_minus_permuted_ci95": [
            paired_accuracy(s, c, ids, 0.95) for s, c in zip(selected, controls)
        ],
        "unique_controls": len(manifest["keys"]),
        "paired_units": len(np.unique(ids)),
    }
    save_json(folder / "shuffle_results.json", result)
    print(dataset, result["selected_minus_permuted_ci95"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("predict", "summarize"))
    parser.add_argument("dataset", choices=("fc100", "ninapro"))
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    if args.stage == "predict":
        evaluate(args.dataset, args.gpus)
    else:
        summarize(args.dataset)
