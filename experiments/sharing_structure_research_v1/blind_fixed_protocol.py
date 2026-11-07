"""Generate a frozen fixed-example cohort; teacher information stays private."""

import argparse
import json
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .blind_contract import digest, save_json
from .shared_structure_data import teacher as synthetic_teacher

BASE = Path(__file__).resolve().parent


def prepare(root, *, sources=4, budget_divisor=1, seed=26110633079):
    root.mkdir(parents=True, exist_ok=False)
    for name in (
        "private",
        "public",
        "source",
        "dense",
        "fits",
        "logs",
        "receipts",
        "code_snapshot",
    ):
        (root / name).mkdir()
    definitions = []
    for number in range(2):
        definitions.append(
            {
                "dataset": "pattern_0110",
                "variant": "counts",
                "replicate": number,
                "inputs": 11,
                "hidden": 8,
                "codes": 2,
                "outputs": 1,
            }
        )
    for number in range(2):
        definitions.append(
            {
                "dataset": "deepsets",
                "variant": "pooled",
                "replicate": number,
                "inputs": 16,
                "hidden": 16,
                "codes": 4,
                "outputs": 2,
            }
        )
    for number, variant in enumerate(
        ("local", "local", "quartet_mix", "quartet_mix", "random", "random")
    ):
        definitions.append(
            {
                "dataset": "general",
                "variant": variant,
                "replicate": number,
                "inputs": 16,
                "hidden": 16,
                "codes": 4,
                "outputs": 1 + number % 2,
            }
        )
    private = {"seed": seed, "noise": [0.0, 0.2], "sources": sources, "cases": []}
    public = {
        "stage": "G33 independent fixed-example confirmation without teacher numerical priors",
        "registered_utc": datetime.now(UTC).isoformat(),
        "sources": sources,
        "conditions": [0, 1],
        "sizes_allowlist": ["inputs", "hidden", "codes", "outputs"],
        "observation_mode": "Fixed independent examples. One raw response per x. No repeat measurements, active training queries, true bias, noise, parameter ranges, roles, masks, code values or generation seeds enter the learner.",
        "learner_bias": "H independent learned hidden biases and O learned output offsets",
        "learner_head": "Free H by O numeric matrix for every source task",
        "learner_settings": {
            "adam_starts_per_source": 12,
            "adam_steps": 2500,
            "adam_learning_rates": [0.003, 0.01, 0.03],
            "dense_polish_starts": 4,
            "dense_polish_steps": 32,
            "residual_repair_rounds": 2,
            "residual_neuron_starts": 32,
            "residual_neuron_steps": 300,
            "zero_standard_error_multipliers": [3, 5],
            "numeric_starts": 2,
            "numeric_steps": 32,
            "code_norm_gauge": "RMS(C)=1, function-preserving scale change",
            "head_refit": "Profile free W2/offset on every numeric step",
        },
        "folds": {
            name: count // budget_divisor
            for name, count in {
                "fixed_train": 16384,
                "selection": 2048,
                "meta": 4096,
                "test": 8192,
            }.items()
        },
        "cases": [],
        "cohort_audit": "Freeze every selected predictor before opening any teacher structure or clean test responses. No method changes based on this cohort's structural audit.",
        "software_tests": False,
        "resource_policy": "Only free GPUs0-3; never6/7. No foreign jobs touched.",
    }
    for number, definition in enumerate(definitions):
        case = f"case_{number:02d}"
        rng = np.random.default_rng(seed + 100003 * number)
        d, h, k, o = [definition[key] for key in public["sizes_allowlist"]]
        index = np.zeros((d, h), dtype=np.int16)
        if definition["dataset"] == "pattern_0110":
            for neuron in range(h):
                index[neuron : neuron + 4, neuron] = [-1, 2, 2, -1]
            code = rng.uniform(0.8, 1.3, (sources, k))
            bias = np.repeat(rng.uniform(-3.2, -2.8, (sources, 1)), h, axis=1)
            head = np.repeat(np.exp(rng.normal(0, 0.25, (sources, 1, o))), h, axis=1)
            input_order = np.arange(d)
            bias_style = "unknown_task_common"
        elif definition["dataset"] == "deepsets":
            input_order = rng.permutation(d)
            for point in range(8):
                index[
                    np.ix_(
                        input_order[2 * point : 2 * point + 2],
                        [2 * point, 2 * point + 1],
                    )
                ] = [[1, 2], [3, 4]]
            code = rng.normal(size=(sources, k))
            channel_bias = rng.normal(size=(sources, 2)) * rng.uniform(
                0.1, 1.3, (sources, 1)
            )
            bias = np.tile(channel_bias, (1, 8))
            channel_head = rng.normal(size=(sources, 2, o))
            head = np.tile(channel_head, (1, 8, 1))
            bias_style = "unknown_two_channel_shared"
        else:
            index, code = synthetic_teacher(
                {"seed": seed, "sources": sources, "codes": k},
                {"family": 33951 + number, "variant": definition["variant"]},
            )
            if definition["replicate"] % 2:
                bias = rng.normal(0, 1.7, (sources, h)) + rng.uniform(
                    -1, 1, (sources, 1)
                )
                bias_style = "unknown_independent_neurons"
            else:
                bias = np.repeat(rng.normal(0, 0.8, (sources, 1)), h, axis=1)
                bias_style = "unknown_task_common"
            head = rng.normal(size=(sources, h, o))
            input_order = np.arange(d)
        permutation = rng.permutation(h)
        index, bias, head = (
            index[:, permutation],
            bias[:, permutation],
            head[:, permutation],
        )
        offset = rng.normal(0, 0.3, (sources, o))
        np.savez_compressed(
            root / "private" / f"{case}.npz",
            index=index,
            codes=code,
            bias=bias,
            readout=head,
            offset=offset,
            input_order=input_order,
        )
        sizes = {key: definition[key] for key in public["sizes_allowlist"]}
        save_json(root / "public" / f"{case}.json", sizes)
        public["cases"].append({"name": case, **sizes})
        private["cases"].append({"name": case, **definition, "bias_style": bias_style})
        codebook = np.c_[np.zeros(sources), code]
        weight = np.sign(index)[None] * codebook[:, np.abs(index)]
        panels = {name: {} for name in public["folds"]}
        clean_test = {}
        for task in range(sources):
            observed_rng = np.random.default_rng(
                seed + 900001 + number * 10009 + task * 100003
            )
            for name, count in public["folds"].items():
                x = observed_rng.normal(size=(count, d))
                if name in ("fixed_train", "selection"):
                    x *= np.resize(np.array([0.5, 1, 2, 4]), count)[:, None]
                clean = (
                    np.maximum(x @ weight[task] + bias[task], 0)
                    @ head[task]
                    / math.sqrt(h)
                    + offset[task]
                )
                noise = observed_rng.normal(size=clean.shape)
                panels[name][f"x_{task}"] = x
                for condition, sigma in enumerate(private["noise"]):
                    panels[name][f"y_{condition}_{task}"] = clean + sigma * noise
                if name == "test":
                    clean_test[f"y_{task}"] = clean
        np.savez_compressed(root / "private" / f"{case}_clean_test.npz", **clean_test)
        for condition in public["conditions"]:
            folder = root / "public" / f"{case}_c{condition}"
            folder.mkdir()
            for name, values in panels.items():
                arrays = {}
                for task in range(sources):
                    arrays[f"x_{task}"] = values[f"x_{task}"]
                    arrays[f"y_{task}"] = values[f"y_{condition}_{task}"]
                np.savez_compressed(folder / f"{name}.npz", **arrays)
            save_json(
                folder / "observations.json",
                {
                    "registered_utc": datetime.now(UTC).isoformat(),
                    "mode": "fixed_examples_one_response_per_input",
                    "observations_per_task": sum(public["folds"].values()),
                    "sources": sources,
                    "sha256": {p.name: digest(p) for p in folder.glob("*.npz")},
                },
            )
    save_json(root / "private" / "generation.json", private)
    names = [p.name for p in BASE.glob("blind_*.py")]
    for name in names:
        shutil.copy2(BASE / name, root / "code_snapshot" / name)
    public["code_sha256"] = {name: digest(BASE / name) for name in names}
    save_json(root / "protocol.json", public)
    save_json(
        root / "data_frozen.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "public_sha256": {
                str(p.relative_to(root)): digest(p)
                for p in (root / "public").rglob("*")
                if p.is_file()
            },
            "private_commitment_sha256": {
                p.name: digest(p) for p in (root / "private").glob("*") if p.is_file()
            },
        },
    )
    print(
        json.dumps(
            {
                "prepared": str(root),
                "cases": len(definitions),
                "sources": sources,
                "conditions": len(public["conditions"]),
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sources", type=int, default=4)
    parser.add_argument("--budget-divisor", type=int, default=1)
    parser.add_argument("--seed", type=int, default=26110633079)
    args = parser.parse_args()
    if args.sources < 1 or not 1 <= args.budget_divisor <= 512:
        parser.error("sources must be positive; budget-divisor must be in [1,512]")
    prepare(
        args.root.resolve(),
        sources=args.sources,
        budget_divisor=args.budget_divisor,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
