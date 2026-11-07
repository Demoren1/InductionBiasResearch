"""Generate observations separately from learner processes."""

import argparse
import json
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

BASE = Path(__file__).resolve().parents[1]
RESEARCH = BASE.parent / "sharing_structure_research_v1"


def numerical_tasks(rng, kind, count, hidden, outputs, codes):
    if kind == "pattern":
        code = rng.uniform(0.8, 1.3, (count, codes))
        bias = np.repeat((-2 * code.sum(1) + 0.6)[:, None], hidden, 1)
        head = np.repeat(rng.lognormal(0, 0.2, (count, 1, outputs)), hidden, 1)
    else:
        code = rng.normal(size=(count, codes))
        bias = np.tile(rng.normal(0, 0.7, (count, 2)), (1, hidden // 2))
        head = np.tile(rng.normal(size=(count, 2, outputs)), (1, hidden // 2, 1))
    offset = rng.normal(0, 0.25, (count, outputs))
    return code, bias, head, offset


def outputs_for(x, index, code, bias, head, offset):
    weight = np.sign(index) * np.r_[0.0, code][np.abs(index)]
    return np.maximum(x @ weight + bias, 0) @ head / math.sqrt(index.shape[1]) + offset


def prepare(root, stage, replicates, sources):
    root.mkdir(parents=True, exist_ok=False)
    for folder in (
        "private",
        "public",
        "source",
        "dense",
        "fits",
        "logs",
        "receipts",
        "metrics",
        "targets",
        "code_snapshot",
    ):
        (root / folder).mkdir()
    seed = 26110754013 + (10000019 if stage == "confirmation" else 0)
    conditions = [0] if stage == "pilot" else [0, 1]
    folds = {"fixed_train": 4096, "selection": 512, "meta": 1024, "test": 2048}
    protocol = {
        "created_utc": datetime.now(UTC).isoformat(),
        "stage": stage,
        "sources": sources,
        "conditions": conditions,
        "folds": folds,
        "cases": [],
        "source_settings": {
            "adam_steps": 2500,
            "adam_starts": 12,
            "adam_rates": [0.003, 0.01, 0.03],
            "gn_steps": 32,
            "revival_rounds": 2,
        },
        "allowed_gpus": [0, 1, 2, 3, 4, 5],
        "forbidden_gpus": [6, 7],
        "truth_policy": "Only prescribed architecture and codebook size enter learning. No true weights, bias, noise, masks, groups, coordinates, generation seed or target responses.",
        "selection_policy": "Support fits numbers; selection chooses numerical checkpoints; source meta chooses structures. Target query and structural truth stay sealed until every method is frozen.",
        "adaptation_policy": "Transfer only shared structure. Every target codebook, hidden bias, free readout and output offset is initialized afresh.",
        "development_policy": "Tune on pilot source validation. If ours loses, diagnose only from pilot observations and registered hypotheses. Retain original result, register changes, and confirm on new data.",
        "measurement_policy": "Synchronized wall time, CPU peak RSS, peak CUDA allocated/reserved bytes, and PyTorch registered-operator FLOPs. CPU NumPy/SciPy work is excluded from the FLOP count and this limitation must be labelled.",
        "cleanup_policy": "Remove unselected heavy checkpoints and scratch caches after paper/report/metrics/curves/selected indexators and reproducibility manifests are preserved.",
    }
    private = {"seed": seed, "cases": [], "noise": [0.0, 0.1]}
    for number in range(2 * replicates):
        case = f"case_{number:02d}"
        kind = "pattern" if number < replicates else "deepsets"
        d, h, k, o = (11, 8, 2, 1) if kind == "pattern" else (16, 16, 4, 2)
        rng = np.random.default_rng(seed + 100003 * number)
        index = np.zeros((d, h), dtype=np.int16)
        if kind == "pattern":
            for neuron in range(h):
                index[neuron : neuron + 4, neuron] = [-1, 2, 2, -1]
        else:
            coordinates = rng.permutation(d)
            for point in range(8):
                index[
                    np.ix_(
                        coordinates[2 * point : 2 * point + 2],
                        [2 * point, 2 * point + 1],
                    )
                ] = [[1, 2], [3, 4]]
        columns = rng.permutation(h)
        index = index[:, columns]
        code, bias, head, offset = numerical_tasks(rng, kind, sources, h, o, k)
        bias, head = bias[:, columns], head[:, columns]
        np.savez_compressed(
            root / "private" / f"{case}.npz",
            index=index,
            codes=code,
            bias=bias,
            readout=head,
            offset=offset,
        )
        sizes = {"inputs": d, "hidden": h, "codes": k, "outputs": o}
        save_json(root / "public" / f"{case}.json", sizes)
        protocol["cases"].append({"name": case, **sizes})
        private["cases"].append(
            {"name": case, "dataset": kind, "replicate": number % replicates}
        )
        panels = {name: {} for name in folds}
        clean_query = {}
        for task in range(sources):
            for name, count in folds.items():
                x = rng.normal(size=(count, d))
                if name in ("fixed_train", "selection"):
                    x *= np.resize([0.5, 1.0, 2.0, 4.0], count)[:, None]
                clean = outputs_for(
                    x, index, code[task], bias[task], head[task], offset[task]
                )
                noise = rng.normal(size=clean.shape)
                panels[name][f"x_{task}"] = x
                for condition in conditions:
                    panels[name][f"y_{condition}_{task}"] = (
                        clean + private["noise"][condition] * noise
                    )
                if name == "test":
                    clean_query[f"y_{task}"] = clean
        np.savez_compressed(root / "private" / f"{case}_clean_test.npz", **clean_query)
        for condition in conditions:
            folder = root / "public" / f"{case}_c{condition}"
            folder.mkdir()
            for name, panel in panels.items():
                arrays = {}
                for task in range(sources):
                    arrays[f"x_{task}"] = panel[f"x_{task}"]
                    arrays[f"y_{task}"] = panel[f"y_{condition}_{task}"]
                np.savez_compressed(folder / f"{name}.npz", **arrays)
        target_code, target_bias, target_head, target_offset = numerical_tasks(
            rng, kind, 8, h, o, k
        )
        target_bias, target_head = target_bias[:, columns], target_head[:, columns]
        target = {}
        for task in range(8):
            for name, count in (("support", 1024), ("selection", 256), ("query", 2048)):
                x = rng.normal(size=(count, d))
                target[f"{name}_x_{task}"] = x
                target[f"{name}_y_{task}"] = outputs_for(
                    x,
                    index,
                    target_code[task],
                    target_bias[task],
                    target_head[task],
                    target_offset[task],
                )
        if kind == "pattern":
            bits = (np.arange(2048)[:, None] >> np.arange(10, -1, -1)) & 1
            target["binary_x"] = (2 * bits - 1).astype(float)
            windows = np.lib.stride_tricks.sliding_window_view(bits, 4, axis=1)
            target["binary_y"] = np.any(
                np.all(windows == [0, 1, 1, 0], axis=-1), axis=1
            ).astype(float)
            target["binary_partition"] = rng.permutation(2048)
        np.savez_compressed(root / "private" / f"{case}_targets.npz", **target)
    save_json(root / "protocol.json", protocol)
    save_json(root / "private" / "generation.json", private)
    snapshots = [
        p for p in BASE.rglob("*.py") if "results" not in p.relative_to(BASE).parts
    ] + [*RESEARCH.glob("*.py")]
    for path in snapshots:
        target = root / "code_snapshot" / path.relative_to(BASE.parent.parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    save_json(
        root / "registered_sources.json",
        {
            "code_sha256": {
                str(p.relative_to(BASE.parent.parent)): digest(p) for p in snapshots
            }
        },
    )
    print(
        json.dumps(
            {"root": str(root), "cases": len(protocol["cases"]), "sources": sources}
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=("pilot", "confirmation"), default="pilot")
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--sources", type=int, default=16)
    args = parser.parse_args()
    prepare(args.root.resolve(), args.stage, args.replicates, args.sources)


if __name__ == "__main__":
    main()
