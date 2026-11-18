"""Compare a shared weight and random exceptions on the held-out pilot tasks."""

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from experiments.srr_approx_v1.synthetic_random_controls import shared_background

from .study import LAYOUTS, OUTPUT, TARGET_STEPS, evaluate, summarize
from .tuning import matching_predictions, pilot_panels

MASK_SEEDS = (7611191800, 7611191801, 7611191802, 7611191803)


def run(dataset, gpus, pool):
    reference = OUTPUT / "tuning" / dataset
    output = OUTPUT / "tuning" / "shared_code" / dataset
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists():
        raise FileExistsError(output / "results.json")

    layout = LAYOUTS[dataset]
    config = json.loads((reference / "registration.json").read_text())
    selection = json.loads((reference / "validation.json").read_text())
    codes = selection["scores"][selection["selected"]]["codes"]
    if config["target_steps"] != TARGET_STEPS:
        raise ValueError("Target fit steps differ from the reference")

    shape = (layout.fan_in, layout.hidden)
    masks = {
        f"mask_{i}": shared_background(shape, codes, np.random.default_rng(seed))
        for i, seed in enumerate(MASK_SEEDS)
    }
    indices = {**masks, "one_code": np.ones(shape, dtype=np.int16)}
    frozen = output / "indexators.npz"
    np.savez_compressed(frozen, **indices)
    old_predictions = reference / "predictions.npz"
    save_json(
        output / "registration.json",
        {
            "dataset": dataset,
            "codes_with_exceptions": codes,
            "mask_seeds": MASK_SEEDS,
            "test_episodes": config["test_episode_count"],
            "test_fit_seed": config["test_fit_seed"],
            "target_steps": TARGET_STEPS,
            "gpus": gpus,
            "indexators_sha256": digest(frozen),
            "reference_predictions_sha256": digest(old_predictions),
        },
    )
    _, load_test = pilot_panels(
        dataset, config["validation_episode_count"], config["test_episode_count"]
    )
    target = load_test()
    predictions = matching_predictions(old_predictions, target, ("asrr", "dense"))
    print(f"{dataset}: fitting {len(indices)} controls on {len(target.units)} episodes")
    fits = evaluate(pool, gpus, target, layout, indices, config["test_fit_seed"])
    predictions.update({name: fit[0] for name, fit in fits.items()})
    np.savez_compressed(
        output / "predictions.npz",
        answers=target.query_y,
        records=target.records,
        units=target.units,
        **predictions,
    )
    accuracy = {
        name: (prediction == target.query_y).mean(axis=-1)
        for name, prediction in predictions.items()
    }
    accuracy["mask_mean"] = np.mean([accuracy[name] for name in masks], axis=0)
    rows = {name: summarize(value, target.units) for name, value in accuracy.items()}
    paired = {
        f"{control}_minus_{baseline}": summarize(
            accuracy[control] - accuracy[baseline], target.units
        )
        for control in ("mask_mean", "one_code")
        for baseline in ("asrr", "dense")
    }
    save_json(output / "results.json", {"accuracy": rows, "paired": paired})
    for name, score in rows.items():
        print(
            f"{dataset} {name:10s} {100 * score['mean']:.2f} "
            f"± {100 * score['ci95_half_width']:.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", choices=tuple(LAYOUTS))
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    args = parser.parse_args()
    with ProcessPoolExecutor(
        max_workers=len(args.gpus), mp_context=mp.get_context("spawn")
    ) as pool:
        for dataset in args.datasets:
            run(dataset, args.gpus, pool)
