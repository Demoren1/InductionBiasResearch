"""Compare global Lloyd clustering with Ward on the same signed profiles."""

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
import torch

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from experiments.srr_cluster_v1.clustering import build_bank

from .study import DATA, LAYOUTS, OUTPUT, evaluate, source_profiles, summarize
from .tuning import matched_shuffles, matching_predictions, pilot_panels


def cluster_on_device(profiles, budgets, seed, gpu):
    torch.set_num_threads(1)
    return build_bank(
        profiles, "lloyd", budgets, ("global",), device=f"cuda:{gpu}", seed=seed
    )


def run(dataset, gpus):
    torch.set_num_threads(1)
    root = OUTPUT / "tuning"
    ward = root / dataset
    output = root / "global_kmeans" / dataset
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists():
        raise FileExistsError(output / "results.json")

    layout = LAYOUTS[dataset]
    registration = json.loads((ward / "registration.json").read_text())
    previous = json.loads((ward / "validation.json").read_text())
    validation, load_test = pilot_panels(
        dataset,
        registration["validation_episode_count"],
        registration["test_episode_count"],
    )
    models = DATA.parent / dataset / "dense_fits.npz"
    model_hash = digest(models)
    if model_hash != registration["source_dense_sha256"]:
        raise ValueError("Source Dense fits differ from the Ward study")
    budgets = registration["codes"]
    save_json(
        output / "registration.json",
        {
            "algorithm": "Lloyd on the sign-invariant lifted profiles used by Ward",
            "initializations_per_budget": 4,
            "max_lloyd_steps": 25,
            "initialization_selection": "Lowest source clustering error",
            "budget_selection": "Highest validation accuracy, then lowest CE, then fewer codes",
            "budgets": budgets,
            "source_dense_sha256": model_hash,
            "previous_registration_sha256": digest(ward / "registration.json"),
            "previous_validation_sha256": digest(ward / "validation.json"),
            "gpus": gpus,
        },
    )
    with np.load(models, allow_pickle=False) as archive:
        profiles = source_profiles(archive["parameters"], layout)
    with ProcessPoolExecutor(
        max_workers=len(gpus), mp_context=mp.get_context("spawn")
    ) as pool:
        started = time.perf_counter()
        shards = list(
            pool.map(
                cluster_on_device,
                repeat(profiles),
                np.array_split(budgets, min(len(gpus), len(budgets))),
                repeat(registration["partition_seed"]),
                gpus,
            )
        )
        banks, details = {}, {}
        for indices, records in shards:
            banks.update(indices)
            details.update((record["name"], record) for record in records)
        for name, index in banks.items():
            count = details[name]["codes"]
            if not np.array_equal(np.unique(np.abs(index)), np.arange(1, count + 1)):
                raise ValueError(f"{name} does not use exactly {count} codes")
        construction_seconds = time.perf_counter() - started
        np.savez_compressed(output / "candidates.npz", **banks)
        print(
            f"{dataset}: validating {len(banks)} global k-means indexators", flush=True
        )

        validation_fits = evaluate(
            pool, gpus, validation, layout, banks, registration["validation_fit_seed"]
        )
        scores = {}
        for name, (predictions, ce, _, _) in validation_fits.items():
            codes = details[name]["codes"]
            scores[name] = {
                "codes": codes,
                "accuracy": float(np.mean(predictions == validation.query_y)),
                "ce": float(ce.mean()),
                "ward_accuracy_same_codes": previous["scores"][f"global_m{codes}"][
                    "accuracy"
                ],
            }
        chosen = min(
            scores,
            key=lambda name: (
                -scores[name]["accuracy"],
                scores[name]["ce"],
                scores[name]["codes"],
            ),
        )
        chosen_codes = scores[chosen]["codes"]
        same_codes = f"global_m{chosen_codes}"
        best_ward = previous["best_by_family"]["global"]
        with np.load(ward / "candidates.npz", allow_pickle=False) as candidates:
            frozen = {
                "kmeans": banks[chosen],
                "ward_same_codes": candidates[same_codes],
                "ward_selected": candidates[best_ward],
                **matched_shuffles(banks[chosen], {"family": "global"}),
            }
        freeze_path = output / "frozen_indexators.npz"
        np.savez_compressed(freeze_path, **frozen)
        save_json(
            output / "validation.json",
            {
                "selected": chosen,
                "ward_same_codes": same_codes,
                "ward_selected": best_ward,
                "previous_winner": previous["selected"],
                "scores": scores,
                "clustering": details,
                "bank_construction_seconds": construction_seconds,
                "indexators_sha256": digest(freeze_path),
            },
        )
        print(f"{dataset}: frozen {chosen}, evaluating the pilot episodes", flush=True)

        target = load_test()
        predictions = matching_predictions(
            ward / "predictions.npz", target, ("asrr", "dense")
        )
        predictions["previous_winner"] = predictions.pop("asrr")
        to_fit = dict(frozen)
        aliases = {}
        if previous["selected"] == best_ward:
            predictions["ward_selected"] = predictions["previous_winner"]
            to_fit.pop("ward_selected")
        if same_codes == best_ward:
            aliases["ward_same_codes"] = "ward_selected"
            to_fit.pop("ward_same_codes")

        fits = evaluate(
            pool, gpus, target, layout, to_fit, registration["test_fit_seed"]
        )
    predictions.update({name: value[0] for name, value in fits.items()})
    for name, original in aliases.items():
        predictions[name] = predictions[original]
    np.savez_compressed(
        output / "predictions.npz",
        records=target.records,
        units=target.units,
        answers=target.query_y,
        **predictions,
    )
    accuracy = {
        name: (predicted == target.query_y).mean(-1)
        for name, predicted in predictions.items()
    }
    accuracy["shuffled_mean"] = np.mean(
        [values for name, values in accuracy.items() if name.startswith("shuffled_")],
        axis=0,
    )
    methods = {
        name: summarize(values, target.units) for name, values in accuracy.items()
    }
    comparisons = (
        "ward_same_codes",
        "ward_selected",
        "previous_winner",
        "dense",
        "shuffled_mean",
    )
    paired = {
        name: summarize(accuracy["kmeans"] - accuracy[name], target.units)
        for name in comparisons
    }
    save_json(
        output / "results.json",
        {
            "selected": {"name": chosen, **scores[chosen]},
            "ward_selected": previous["scores"][best_ward],
            "previous_winner": previous["scores"][previous["selected"]],
            "methods": methods,
            "paired_kmeans_minus": paired,
        },
    )
    for name in ("kmeans", *comparisons):
        score = methods[name]
        print(
            f"{dataset} {name:17s} "
            f"{100 * score['mean']:.2f} ± {100 * score['ci95_half_width']:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=tuple(LAYOUTS))
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    args = parser.parse_args()
    run(args.dataset, args.gpus)
