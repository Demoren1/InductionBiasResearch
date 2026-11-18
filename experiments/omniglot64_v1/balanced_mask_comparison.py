"""Compare profile-free random masks with the archived Omniglot controls."""

import argparse
import time

import numpy as np

from experiments.asrr_anchor_init_v1.omniglot_transfer import score_episodes
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json

from .allocation_ablation import background_index, sample_exceptions
from .random_mask_confirmation import COMPARISON as REFERENCE
from .random_mask_confirmation import paired_accuracy, summarize_masks
from .source import BANKS, CODES, HIDDEN, KEY, OUTPUT, ROOT, read_json, save_groups
from .transfer import STUDY, run_fits

COMPARISON = OUTPUT.with_name("omniglot64_balanced_mask_comparison_2026-11-18")
SOURCE = "balanced_random_masks"
SEEDS = (7611189541, 7611189542, 7611189543, 7611189544)


def prepare():
    source_path = OUTPUT / f"{SOURCE}_source.json"
    if COMPARISON.exists() or source_path.exists():
        raise FileExistsError("Balanced-mask comparison already prepared")

    started = time.perf_counter()
    base, extra = divmod(CODES - 1, HIDDEN)
    counts = [base + int(neuron < extra) for neuron in range(HIDDEN)]
    groups = []
    for seed in SEEDS:
        name = f"balanced_random_mask_{seed}"
        if (BANKS / f"{name}.npz").exists():
            raise FileExistsError(name)
        mask = sample_exceptions(counts, seed)
        groups.append(
            (
                name,
                background_index(mask),
                {
                    "seed": seed,
                    "exceptions_per_neuron": counts,
                    "pixels_with_any_exception": int(mask.any(axis=1).sum()),
                },
            )
        )
    save_groups(SOURCE, groups, started, None)
    new = read_json(source_path)["candidates"]
    balanced_keys = [row["name"] for row in new]

    prior_choices = REFERENCE / "choices_frozen.json"
    prior_panels = REFERENCE / "target_panels.json"
    prior_jobs = REFERENCE / "target_jobs.json"
    choices = read_json(prior_choices)
    panels = read_json(prior_panels)
    if digest(prior_choices) != panels["choices_sha256"]:
        raise ValueError("Changed confirmation choices")
    old = read_json(prior_jobs)
    wsr = choices["policies"]["wsr_step_75"]["name"]
    templates = {run["phase"]: run for run in old if run["name"] == wsr}
    if set(templates) != set(panels["panels"]):
        raise ValueError("Incomplete reference episode panels")
    added = [
        {**run, "name": row["name"], "model_sha256": row["model_sha256"]}
        for run in templates.values()
        for row in new
    ]
    choices["policies"].update(zip(balanced_keys, new, strict=True))

    COMPARISON.mkdir()
    frozen = COMPARISON / "choices_frozen.json"
    save_json(frozen, choices)
    save_json(
        COMPARISON / "registration.json",
        {
            "choices_sha256": digest(frozen),
            "reference": str(REFERENCE),
            "reference_choices_sha256": digest(prior_choices),
            "reference_panels_sha256": digest(prior_panels),
            "reference_jobs_sha256": digest(prior_jobs),
            "balanced_source_sha256": digest(source_path),
            "balanced_masks": balanced_keys,
            "profile_count_masks": read_json(REFERENCE / "registration.json")[
                "new_masks"
            ],
            "mask_seeds": SEEDS,
            "exceptions_per_neuron": counts,
            "construction": "Mask positions and counts use only layer dimensions, code budget and RNG seeds",
            "intervals": "Paired over episodes, conditional on the fixed masks and held-out characters",
        },
    )
    save_json(
        COMPARISON / "target_panels.json",
        {**panels, "choices_sha256": digest(frozen)},
    )
    save_json(COMPARISON / "target_jobs.json", [*old, *added])
    save_json(COMPARISON / "added_jobs.json", added)
    print(
        {
            "episodes": sum(row["count"] for row in panels["panels"].values()),
            "new_jobs": len(added),
            "reused_jobs": len(old),
        },
        flush=True,
    )


def score():
    path = COMPARISON / "results.json"
    if path.exists():
        raise FileExistsError(path)
    registration = read_json(COMPARISON / "registration.json")
    choices_path = COMPARISON / "choices_frozen.json"
    if digest(choices_path) != registration["choices_sha256"]:
        raise ValueError("Changed frozen choices")
    choices = read_json(choices_path)
    groups = score_episodes(
        COMPARISON, STUDY, {"key": KEY, "rule": choices["numeric_rule"]}
    )
    pooled = {
        name: np.concatenate([rows[name] for rows in groups.values()])
        for name in choices["policies"]
    }
    balanced = registration["balanced_masks"]
    profile = registration["profile_count_masks"]
    pooled["mean_balanced"] = np.mean([pooled[name] for name in balanced], axis=0)
    pooled["mean_profile_counts"] = np.mean([pooled[name] for name in profile], axis=0)
    methods = summarize_masks(pooled)
    differences = [
        paired_accuracy(pooled, method, reference)
        for method in (*balanced, "mean_balanced")
        for reference in ("mean_profile_counts", "wsr_step_75")
    ]
    save_json(
        path,
        {
            "episodes": len(pooled["wsr_step_75"]),
            "same_episodes_as": str(REFERENCE),
            "balanced_masks": balanced,
            "profile_count_masks": profile,
            "methods": methods,
            "paired_differences": differences,
            "intervals": registration["intervals"],
        },
    )
    print(
        {
            name: methods[name]
            for name in (
                *balanced,
                "mean_balanced",
                "mean_profile_counts",
                "wsr_step_75",
            )
        },
        differences[-2:],
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "score"))
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "run":
        run_fits("added", args.gpus, root=ROOT, output=COMPARISON)
    else:
        score()
