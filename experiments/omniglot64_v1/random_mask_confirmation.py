"""Confirm the shared-code random masks on fresh common Omniglot episodes."""

import argparse
import time

import numpy as np

from experiments.asrr_anchor_init_v1.omniglot_transfer import score_episodes
from experiments.asrr_anchor_init_v1.report import interval
from experiments.asrr_anchor_init_v1.transfer import public_panel
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json

from .allocation_ablation import RANDOM, background_index, sample_exceptions
from .source import BANKS, CODES, KEY, OUTPUT, ROOT, read_json, save_groups
from .transfer import PANELS, SETTING, STUDY, job, run_fits

COMPARISON = OUTPUT.with_name("omniglot64_random_mask_confirmation_2026-11-18")
PREVIOUS = OUTPUT.with_name("omniglot64_allocation_ablation_2026-11-18")
SOURCE = "asrr_random_confirmation_masks"
MASK_SEEDS = (7611188601, 7611188602, 7611188603, 7611188604)
PANEL_SEEDS = tuple(range(7611188650, 7611188658))
EPISODES_PER_PANEL = 96
BASELINES = ("four_partitions", "wsr_step_75")


def summarize_masks(pooled):
    return {
        name: {
            "accuracy": float(values[:, 0].mean()),
            "accuracy_ci95_halfwidth": interval(values[:, 0]),
            "mean_cross_entropy": float(values[:, 1].mean()),
            "codes": CODES,
        }
        for name, values in pooled.items()
    }


def paired_accuracy(pooled, method, reference):
    delta = pooled[method][:, 0] - pooled[reference][:, 0]
    return {
        "method": method,
        "reference": reference,
        "accuracy_difference": float(delta.mean()),
        "ci95_halfwidth": interval(delta),
    }


def prepare():
    if COMPARISON.exists() or (OUTPUT / f"{SOURCE}_source.json").exists():
        raise FileExistsError("Random mask confirmation already prepared")
    previous_path = PREVIOUS / "choices_frozen.json"
    previous = read_json(previous_path)
    original = previous["policies"]["random_exceptions"]
    counts = original["exceptions_per_neuron"]
    if sum(counts) != CODES - 1:
        raise ValueError("Unexpected original random mask budget")

    policies = {key: previous["policies"][key] for key in BASELINES}
    policies["original_random"] = original
    for row in policies.values():
        if digest(BANKS / f"{row['name']}.npz") != row["model_sha256"]:
            raise ValueError(f"Changed reference assignment: {row['name']}")
    with np.load(BANKS / f"{original['name']}.npz") as model:
        if not np.array_equal((model["index"] != 1).sum(axis=0), counts):
            raise ValueError("Different per-neuron counts in the original mask")

    started = time.perf_counter()
    groups = []
    for seed in MASK_SEEDS:
        exceptions = sample_exceptions(counts, seed)
        name = f"{RANDOM}_{seed}"
        if (BANKS / f"{name}.npz").exists():
            raise FileExistsError(name)
        groups.append(
            (
                name,
                background_index(exceptions),
                {
                    "seed": seed,
                    "exceptions_per_neuron": counts,
                    "pixels_with_any_exception": int(exceptions.any(axis=1).sum()),
                },
            )
        )
    save_groups(SOURCE, groups, started, None)
    source = OUTPUT / f"{SOURCE}_source.json"
    added = read_json(source)["candidates"]
    keys = tuple(f"random_{row['seed']}" for row in added)
    policies.update(zip(keys, added, strict=True))

    COMPARISON.mkdir()
    choices_path = COMPARISON / "choices_frozen.json"
    save_json(
        choices_path,
        {
            "numeric_rule": previous["numeric_rule"],
            "policies": policies,
        },
    )
    save_json(
        COMPARISON / "registration.json",
        {
            "previous_choices_sha256": digest(previous_path),
            "choices_sha256": digest(choices_path),
            "new_masks_sha256": digest(source),
            "new_masks": keys,
            "mask_seeds": MASK_SEEDS,
            "panel_seeds": PANEL_SEEDS,
            "episodes_per_panel": EPISODES_PER_PANEL,
            "primary_summary": "Mean of four new random masks; the original mask is reported separately",
            "intervals": "Paired over episodes, conditional on the fixed masks and held-out characters",
        },
    )

    panels, jobs = {}, []
    for number, seed in enumerate(PANEL_SEEDS):
        phase = f"test_random_mask_confirm_{number:02d}"
        panel = public_panel(
            SETTING, phase, EPISODES_PER_PANEL, seed, directory=PANELS, target=True
        )
        panels[phase] = {**panel, "seed": seed}
        jobs.extend(
            job(row, phase, panel["sha256"], seed=7611188750 + number * 96)
            for row in policies.values()
        )
    save_json(
        COMPARISON / "target_panels.json",
        {
            "panels": panels,
            "choices_sha256": digest(choices_path),
        },
    )
    save_json(COMPARISON / "target_jobs.json", jobs)
    print({"episodes": len(panels) * EPISODES_PER_PANEL, "jobs": len(jobs)}, flush=True)


def score():
    path = COMPARISON / "results.json"
    if path.exists():
        raise FileExistsError(path)
    registration = read_json(COMPARISON / "registration.json")
    choices_path = COMPARISON / "choices_frozen.json"
    if digest(choices_path) != registration["choices_sha256"]:
        raise ValueError("Changed preregistered choices")
    choices = read_json(choices_path)
    groups = score_episodes(
        COMPARISON, STUDY, {"key": KEY, "rule": choices["numeric_rule"]}
    )
    pooled = {
        name: np.concatenate([rows[name] for rows in groups.values()])
        for name in choices["policies"]
    }
    keys = registration["new_masks"]
    seed_accuracies = [float(pooled[name][:, 0].mean()) for name in keys]
    pooled["mean_new_masks"] = np.mean([pooled[name] for name in keys], axis=0)
    summaries = summarize_masks(pooled)
    differences = [
        paired_accuracy(pooled, method, reference)
        for method in ("original_random", *keys, "mean_new_masks")
        for reference in BASELINES
    ]
    save_json(
        path,
        {
            "episodes": len(pooled["four_partitions"]),
            "new_mask_seeds": registration["mask_seeds"],
            "methods": summaries,
            "new_mask_accuracy_range": [min(seed_accuracies), max(seed_accuracies)],
            "new_mask_accuracy_sd": float(np.std(seed_accuracies, ddof=1)),
            "paired_differences": differences,
            "intervals": registration["intervals"],
        },
    )
    print(summaries, differences, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "score"))
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "run":
        run_fits("target", args.gpus, root=ROOT, output=COMPARISON)
    else:
        score()
