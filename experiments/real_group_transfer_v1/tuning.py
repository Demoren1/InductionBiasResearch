"""Select the aSRR code budget and input partitions before scoring the pilot episodes."""

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np
import torch

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from experiments.srr_approx_v1.core import canonical, signed_forest
from experiments.srr_approx_v1.synthetic_random_controls import shuffle_groups
from experiments.srr_cluster_v1.clustering import centroid_fit

from .data import fc100_panel, ninapro_panel, ninapro_source_normalization
from .study import (
    DATA,
    LAYOUTS,
    OUTPUT,
    SEED,
    SHUFFLES,
    evaluate,
    same_offset_pairs,
    source_profiles,
    summarize,
)

CODES_PER_NEURON = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40)
INPUT_PARTITIONS = (1, 2, 4, 8, 16)
FC100_VALIDATION_EPISODES = 512
PARTITION_SEED = SEED + 100


def build_candidates(profiles, gpu):
    tasks, inputs, neurons = profiles.shape
    neuron_ids = np.arange(1, neurons + 1)
    budgets = [count * neurons for count in CODES_PER_NEURON]
    ward, _ = signed_forest(profiles, budgets, device=f"cuda:{gpu}")
    bank, config = {}, {}
    for codes, index in ward.items():
        name = f"global_m{codes}"
        bank[name] = index
        config[name] = {"family": "global", "partitions": None, "codes": codes}

    for partitions in INPUT_PARTITIONS:
        size = neurons // partitions
        blocks = [
            block.transpose(1, 0, 2).reshape(inputs, tasks * size)
            for block in np.split(profiles, partitions, axis=2)
        ]
        for count in CODES_PER_NEURON:
            if count == 1 and partitions > 1:
                continue
            if count == 1:
                by_neuron = np.zeros((inputs, neurons), dtype=int)
            else:
                labels = [
                    centroid_fit(
                        block, count, "lloyd", PARTITION_SEED + i * 1009, "cpu"
                    )[0]
                    for i, block in enumerate(blocks)
                ]
                by_neuron = np.repeat(np.column_stack(labels), size, axis=1)
            codes = count * neurons
            name = f"input_p{partitions}_m{codes}"
            bank[name] = canonical(by_neuron * neurons + neuron_ids)
            config[name] = {
                "family": "input",
                "partitions": partitions,
                "codes": codes,
            }
    for name, index in bank.items():
        codes = config[name]["codes"]
        if not np.array_equal(np.unique(np.abs(index)), np.arange(1, codes + 1)):
            raise ValueError(f"{name} does not occupy exactly {codes} codes")
    return bank, config


def matched_shuffles(index, config):
    rng = np.random.default_rng(SEED + 5)
    controls = {}
    for replicate in range(SHUFFLES):
        if config["family"] == "global":
            permuted = shuffle_groups(index, rng)
        else:
            blocks = np.split(index, config["partitions"], axis=1)
            permuted = np.concatenate(
                [shuffle_groups(block, rng, shared_inputs=True) for block in blocks],
                axis=1,
            )
        controls[f"shuffled_{replicate}"] = permuted
    return controls


def pilot_panels(dataset, validation_count, test_count):
    if dataset == "fc100":
        with np.load(DATA / "fc100.npz", allow_pickle=False) as archive:
            data = dict(archive)
        validation = fc100_panel(data, "development", validation_count, SEED + 2)

        def load_test():
            return fc100_panel(data, "test", test_count, SEED + 3)
    else:
        folder = DATA / "ninapro"
        mean, scale = ninapro_source_normalization(folder)
        validation = ninapro_panel(folder, "development", mean, scale)

        def load_test():
            return ninapro_panel(folder, "test", mean, scale)

    if len(validation.units) != validation_count:
        raise ValueError("Validation episode count differs from registration")
    return validation, load_test


def matching_predictions(path, target, names):
    with np.load(path, allow_pickle=False) as archive:
        for key, expected in (
            ("records", target.records),
            ("units", target.units),
            ("answers", target.query_y),
        ):
            if not np.array_equal(archive[key], expected):
                raise ValueError(f"Test episode {key} differs from {path}")
        return {name: archive[name] for name in names}


def run(dataset, gpus):
    torch.set_num_threads(1)
    pilot = OUTPUT / dataset
    output = OUTPUT / "tuning" / dataset
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists():
        raise FileExistsError(output / "results.json")
    layout = LAYOUTS[dataset]
    previous = json.loads((pilot / "registration.json").read_text())
    if previous["seed"] != SEED or previous["layout"] != asdict(layout):
        raise ValueError("Pilot seed or network layout changed")
    models = DATA.parent / dataset / "dense_fits.npz"
    validation_count = FC100_VALIDATION_EPISODES if dataset == "fc100" else 120
    validation, load_test = pilot_panels(
        dataset, validation_count, previous["target_size"]
    )
    save_json(
        output / "registration.json",
        {
            "dataset": dataset,
            "pilot_registration_sha256": digest(pilot / "registration.json"),
            "source_dense_sha256": digest(models),
            "layout": asdict(layout),
            "codes": [count * layout.hidden for count in CODES_PER_NEURON],
            "input_partitions": INPUT_PARTITIONS,
            "include_global_signed_ward": True,
            "input_clustering": "Lloyd on normalized source input profiles in each consecutive neuron block",
            "partition_seed": PARTITION_SEED,
            "validation_episode_count": len(validation.units),
            "validation_seed": SEED + 2,
            "validation_fit_seed": SEED + 4,
            "selection": "Highest validation accuracy, then lowest CE, fewer codes, fewer partitions",
            "test_episode_count": previous["target_size"],
            "test_fit_seed": SEED + 6,
            "target_steps": previous["target_steps"],
            "shuffles": SHUFFLES,
            "gpus": gpus,
        },
    )
    started = time.perf_counter()
    with np.load(models, allow_pickle=False) as archive:
        profiles = source_profiles(archive["parameters"], layout)
    candidates, configs = build_candidates(profiles, gpus[0])
    construction_seconds = time.perf_counter() - started
    np.savez_compressed(output / "candidates.npz", **candidates)
    print(
        f"{dataset}: validating {len(candidates)} indexators on "
        f"{len(validation.units)} episodes",
        flush=True,
    )

    with ProcessPoolExecutor(
        max_workers=len(gpus), mp_context=mp.get_context("spawn")
    ) as pool:
        checks = evaluate(
            pool, gpus, validation, layout, {**candidates, "dense": None}, SEED + 4
        )
        scores = {
            name: {
                "accuracy": float(np.mean(result[0] == validation.query_y)),
                "ce": float(result[1].mean()),
                **configs.get(name, {"codes": layout.connections}),
            }
            for name, result in checks.items()
        }
        del checks
        ranking = sorted(
            candidates,
            key=lambda name: (
                -scores[name]["accuracy"],
                scores[name]["ce"],
                scores[name]["codes"],
                scores[name]["partitions"] or 0,
                name,
            ),
        )
        best = ranking[0]
        best_by_family = {
            family: next(name for name in ranking if scores[name]["family"] == family)
            for family in ("global", "input")
        }
        index = candidates[best]
        frozen = {"asrr": index, **matched_shuffles(index, configs[best])}
        frozen_path = output / "frozen_indexators.npz"
        np.savez_compressed(frozen_path, **frozen)
        save_json(
            output / "validation.json",
            {
                "selected": best,
                "best_by_family": best_by_family,
                "ranking": ranking,
                "scores": scores,
                "bank_construction_seconds": construction_seconds,
                "indexators_sha256": digest(frozen_path),
            },
        )
        print(
            f"{dataset}: frozen {best}, "
            f"validation accuracy {scores[best]['accuracy']:.4f}",
            flush=True,
        )

        target = load_test()
        pilot_predictions = pilot / "predictions.npz"
        predictions = matching_predictions(pilot_predictions, target, ("dense", "asrr"))
        predictions["pilot_asrr"] = predictions.pop("asrr")
        fits = evaluate(pool, gpus, target, layout, frozen, SEED + 6)
    predictions.update({name: result[0] for name, result in fits.items()})
    np.savez_compressed(
        output / "predictions.npz",
        answers=target.query_y,
        records=target.records,
        units=target.units,
        **predictions,
    )
    accuracies = {
        name: (prediction == target.query_y).mean(-1)
        for name, prediction in predictions.items()
    }
    accuracies["shuffled_mean"] = np.mean(
        [accuracies[f"shuffled_{i}"] for i in range(SHUFFLES)], axis=0
    )
    rows = {
        name: {"accuracy": summarize(value, target.units)}
        for name, value in accuracies.items()
    }
    for name, (_, ce, seconds, memory) in fits.items():
        rows[name].update(
            ce=float(ce.mean()),
            codes=scores[best]["codes"],
            target_fit_seconds_max_shard=seconds,
            target_peak_allocated_bytes=memory,
        )
        if dataset == "fc100":
            rows[name]["shared_pairs_same_offset_fraction"] = same_offset_pairs(
                frozen[name]
            )
    rows["dense"]["codes"] = layout.connections
    rows["pilot_asrr"]["codes"] = previous["codes"]
    rows["shuffled_mean"]["codes"] = scores[best]["codes"]
    paired = {
        name: summarize(accuracies["asrr"] - accuracies[name], target.units)
        for name in ("dense", "shuffled_mean", "pilot_asrr")
    }
    save_json(
        output / "results.json",
        {
            "selected": {"name": best, **scores[best]},
            "pilot_test_predictions_sha256": digest(pilot_predictions),
            "test_episodes": len(target.units),
            "methods": rows,
            "paired_asrr_minus": paired,
        },
    )
    for name in ("asrr", "dense", "shuffled_mean", "pilot_asrr"):
        score = rows[name]["accuracy"]
        print(
            f"{dataset} {name:13s} "
            f"{100 * score['mean']:.2f} ± {100 * score['ci95_half_width']:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=tuple(LAYOUTS))
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    args = parser.parse_args()
    run(args.dataset, args.gpus)
