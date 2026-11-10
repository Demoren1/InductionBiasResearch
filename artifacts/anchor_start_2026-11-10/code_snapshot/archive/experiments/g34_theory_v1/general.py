"""Generate the harder general synthetic cohort for the frozen G34 variants."""

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from experiments.paper_structure_benchmark_v1.protocol.prepare import outputs_for
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

from .run import PROJECT, plan_variants


SEED = 27110864019
VARIANTS = ("local", "quartet_mix", "random")


def general_index(rng, variant):
    """Use the three geometries of the original general synthetic benchmark."""
    if variant == "random":
        index = (rng.uniform(size=(16, 16)) < 0.55) * rng.integers(1, 5, size=(16, 16))
        index *= rng.choice((-1, 1), size=(16, 16))
    else:
        index = np.zeros((16, 16), dtype=np.int16)
        for point in range(8):
            for other in range(8):
                if variant == "local":
                    sign = int(point == other)
                else:
                    sign = (-1 if point == other else 1) * int(point // 4 == other // 4)
                index[2 * point : 2 * point + 2, 2 * other : 2 * other + 2] = (
                    sign * np.arange(1, 5).reshape(2, 2)
                )
    return index[rng.permutation(16)][:, rng.permutation(16)].astype(np.int16)


def general_numbers(rng, variant, replicate, count, outputs):
    code = rng.normal(size=(count, 4))
    if variant == "quartet_mix":
        code /= 2
    if replicate == 0:
        bias = np.repeat(rng.normal(0, 0.8, (count, 1)), 16, axis=1)
    else:
        bias = rng.normal(0, 1.7, (count, 16)) + rng.uniform(-1, 1, (count, 1))
    head = rng.normal(size=(count, 16, outputs))
    offset = rng.normal(0, 0.3, (count, outputs))
    return code, bias, head, offset


def setup(base, previous):
    previous_registration = previous / "confirmation/theory_registration.json"
    method = json.loads(previous_registration.read_text())
    for name, value in method["code_sha256"].items():
        if digest(Path(__file__).with_name(name)) != value:
            raise RuntimeError(f"Previously registered method changed: {name}")
    base.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__).with_name("GENERAL_PROTOCOL.md"), base / "PROTOCOL.md")
    root = base / "confirmation"
    for name in (
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
        "theory_code_snapshot",
    ):
        (root / name).mkdir(parents=True, exist_ok=False)
    method.update(
        frozen_utc=datetime.now(UTC).isoformat(),
        extension="general local/quartet_mix/random; no retuning",
        parent_registration=str(previous_registration),
        parent_registration_sha256=digest(previous_registration),
        prior_confirmation_results_known=True,
        general_confirmation_outputs_used_to_tune=False,
    )
    save_json(root / "theory_registration.json", method)
    core = [
        *(PROJECT / "experiments/sharing_structure_research_v1").glob("*.py"),
        PROJECT / "experiments/paper_structure_benchmark_v1/current_method.py",
    ]
    files = [*Path(__file__).parent.glob("*.py"), *core]
    for path in files:
        target = root / "code_snapshot" / path.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, root / "theory_code_snapshot" / path.name)
    save_json(
        root / "registered_sources.json",
        {"code_sha256": {str(p.relative_to(PROJECT)): digest(p) for p in core}},
    )
    save_json(
        base / "registered_protocol.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "protocol_sha256": digest(base / "PROTOCOL.md"),
            "confirmation_seed": SEED,
            "core_sha256": {str(p.relative_to(PROJECT)): digest(p) for p in core},
            "allowed_gpus": list(range(6)),
            "plan_names": ["baseline", "variants", "target"],
            "primary": "paired target MSE at N=1024 by variant and source noise",
            "report_file": "G34_GENERAL_SYNTHETIC_REPORT_2026-11-08.md",
            "artifact_directory": "g34_general_2026-11-08",
            "target_answers_opened": False,
            "method_changed_after_previous_confirmation": False,
            "generator_sha256": digest(Path(__file__)),
        },
    )
    folds = {"fixed_train": 4096, "selection": 512, "meta": 1024, "test": 2048}
    protocol = {
        "created_utc": datetime.now(UTC).isoformat(),
        "stage": "general confirmation with previously frozen methods",
        "sources": 16,
        "conditions": [0, 1],
        "folds": folds,
        "cases": [],
        "sizes_allowlist": ["inputs", "hidden", "codes", "outputs"],
        "allowed_gpus": list(range(6)),
        "forbidden_gpus": [6, 7],
        "truth_policy": "Architecture dimensions and observed examples only.",
        "source_settings": {
            "adam_steps": 2500,
            "adam_starts": 12,
            "gn_steps": 32,
            "revival_rounds": 2,
        },
    }
    generation = {"seed": SEED, "noise": [0.0, 0.2], "cases": []}
    for number in range(6):
        variant, replicate = VARIANTS[number // 2], number % 2
        case, outputs = f"case_{number:02d}", 1 + replicate
        rng = np.random.default_rng(SEED + 100003 * number)
        index = general_index(rng, variant)
        code, bias, head, offset = general_numbers(rng, variant, replicate, 16, outputs)
        np.savez_compressed(
            root / "private" / f"{case}.npz",
            index=index,
            codes=code,
            bias=bias,
            readout=head,
            offset=offset,
        )
        sizes = {"inputs": 16, "hidden": 16, "codes": 4, "outputs": outputs}
        save_json(root / "public" / f"{case}.json", sizes)
        protocol["cases"].append({"name": case, **sizes})
        generation["cases"].append(
            {
                "name": case,
                "dataset": f"general_{variant}",
                "variant": variant,
                "replicate": replicate,
                "bias_style": "common" if replicate == 0 else "independent",
            }
        )
        panels = {name: {} for name in folds}
        clean_query = {}
        for task in range(16):
            for name, count in folds.items():
                x = rng.normal(size=(count, 16))
                if name in ("fixed_train", "selection"):
                    x *= np.resize([0.5, 1, 2, 4], count)[:, None]
                clean = outputs_for(
                    x, index, code[task], bias[task], head[task], offset[task]
                )
                noise = rng.normal(size=clean.shape)
                panels[name][f"x_{task}"] = x
                for condition, sigma in enumerate(generation["noise"]):
                    panels[name][f"y_{condition}_{task}"] = clean + sigma * noise
                if name == "test":
                    clean_query[f"y_{task}"] = clean
        np.savez_compressed(root / "private" / f"{case}_clean_test.npz", **clean_query)
        for condition in protocol["conditions"]:
            folder = root / "public" / f"{case}_c{condition}"
            folder.mkdir()
            for name, panel in panels.items():
                arrays = {}
                for task in range(16):
                    arrays[f"x_{task}"] = panel[f"x_{task}"]
                    arrays[f"y_{task}"] = panel[f"y_{condition}_{task}"]
                np.savez_compressed(folder / f"{name}.npz", **arrays)
        code, bias, head, offset = general_numbers(rng, variant, replicate, 8, outputs)
        target = {}
        for task in range(8):
            for name, count in (("support", 1024), ("selection", 256), ("query", 2048)):
                x = rng.normal(size=(count, 16))
                target[f"{name}_x_{task}"] = x
                target[f"{name}_y_{task}"] = outputs_for(
                    x, index, code[task], bias[task], head[task], offset[task]
                )
        np.savez_compressed(root / "private" / f"{case}_targets.npz", **target)
    save_json(root / "protocol.json", protocol)
    save_json(root / "private/generation.json", generation)
    save_json(
        root / "data_commitments.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "private_sha256": {
                p.name: digest(p) for p in sorted((root / "private").glob("*"))
            },
            "public_sha256": {
                str(p.relative_to(root)): digest(p)
                for p in sorted((root / "public").rglob("*"))
                if p.is_file()
            },
        },
    )
    jobs = []
    for case in protocol["cases"]:
        for condition in protocol["conditions"]:
            key = f"{case['name']}_c{condition}"
            jobs.append(
                {
                    "id": key,
                    "module": "experiments.g34_theory_v1.run",
                    "args": [
                        "--base",
                        str(root),
                        "--case",
                        case["name"],
                        "--condition",
                        str(condition),
                        "--method",
                        "baseline",
                    ],
                    "metric": str(root / "metrics" / f"{key}_baseline.json"),
                    "registration": str(root / "dense" / key / "fitted.npz"),
                    "log": str(root / "logs" / f"{key}_baseline.log"),
                }
            )
    save_json(base / "baseline_plan.json", {"jobs": jobs})
    plan_variants(base, False)
    print(json.dumps({"prepared": str(base), "cases": 6, "sources": 16}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    args = parser.parse_args()
    setup(args.base.resolve(), args.previous.resolve())


if __name__ == "__main__":
    main()
