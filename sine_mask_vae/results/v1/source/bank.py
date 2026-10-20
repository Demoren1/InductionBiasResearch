"""Freeze a protocol and collect uniformly random masks using neural training."""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

from core import fit, random_masks, save_json, task


DIRECTORY = Path(__file__).resolve().parent
RUN = DIRECTORY / "runs/v1"
OPTIONS = {"steps": 1024, "lr": 0.05, "scale": 2.0, "batch": 64}
SEEDS = [4213, 4214, 4215]


def families(split):
    remainder = {"train": (0, 1), "validation": (2,), "test": (3,)}[split]
    return [
        (a, b) for a in range(4) for b in range(4) if (b - a) % 4 in remainder
    ]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seal():
    RUN.mkdir(parents=True, exist_ok=True)
    plan = {
        "model": "f_j(x)=amplitude_j * sin(x @ W[:,j]); W shape 8x2",
        "teacher": "y_j=small_j*sin(low_j*x[row_j])+sin(high_j*x[row_j])",
        "data": "Fresh IID N(0,I_8) minibatches; normal MSE, autograd SGD",
        "mask": "Exactly two of 16 input edges; both output amplitudes trained",
        "optimizer": OPTIONS,
        "quality_threshold": 0.1,
        "bank_filter": "Measured MSE on 4096 fresh examples, never target supports",
        "source_functions": [0, 1],
        "source_splits": {
            split: families(split) for split in ("train", "validation", "test")
        },
        "validation_function": 100,
        "test_functions": [201, 202],
        "generator_seeds": SEEDS,
        "variants": ["vanilla", "partwise_training"],
        "vae": {
            "epochs": 600,
            "beta": 0.05,
            "warmup": 90,
            "patience": 120,
            "lr": 0.001,
            "batch_per_task": 16,
        },
        "budget": {
            "candidates": 64,
            "steps": 1024,
            "batch": 64,
            "examples_per_search": 4194304,
        },
        "validation_trials_per_family_seed": 16,
        "test_trials_per_family_function_seed": 64,
        "selection": "Only validation chooses dense settings; all decoder seeds retained",
        "bank_max_blocks": 8,
        "bank_fits_per_block": 8192,
        "source_card_roles": "80/20 assigned before filtering; minimum 10/2 elites",
        "scope": "Two labelled output roles known; sine activation, underparameterized student; no global-optimality claim",
    }
    path = RUN / "plan.json"
    if path.exists() and json.loads(path.read_text()) != json.loads(
        json.dumps(plan)
    ):
        raise RuntimeError("Refusing to change frozen plan")
    save_json(path, plan)
    source = RUN / "source"
    source.mkdir(exist_ok=True)
    for name in ("core.py", "bank.py", "geometry.py", "pilot.py"):
        shutil.copy2(DIRECTORY / name, source / name)
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal-only", action="store_true")
    args = parser.parse_args()
    geometry = json.loads((DIRECTORY / "geometry.json").read_text())
    pilot = json.loads((DIRECTORY / "pilot.json").read_text())
    if not geometry["gate_passed"] or any(
        fit["success"] for fit in pilot["fits"] if fit["kind"] == "dense"
    ):
        raise RuntimeError("Controls must pass before bank collection")
    plan = seal()
    if args.seal_only:
        return
    destination = RUN / "bank"
    destination.mkdir(exist_ok=True)
    manifest = []
    for a, b in families("train"):
        for function in plan["source_functions"]:
            name = f"a{a}_b{b}_f{function}"
            record_path = destination / f"{name}.json"
            if record_path.exists():
                manifest.append(json.loads(record_path.read_text()))
                continue
            started = time.perf_counter()
            spec = task(a, b, function)
            collected, good_counts = [], np.zeros(2, dtype=int)
            for block in range(plan["bank_max_blocks"]):
                seed = 92001 + (a * 8 + b * 2 + function) * 10007 + block * 117
                rng = np.random.default_rng(seed + 1)
                masks = random_masks(plan["bank_fits_per_block"], rng)
                roles = (rng.random(len(masks)) >= 0.8).astype(np.int8)
                result = fit(masks, spec, seed, **OPTIONS)
                good = result["empirical"] < plan["quality_threshold"]
                good_counts += np.bincount(roles[good], minlength=2)
                collected.append(
                    {
                        "masks": masks,
                        "roles": roles,
                        **{
                            key: result[key]
                            for key in (
                                "weights",
                                "amplitude",
                                "gradients",
                                "losses",
                                "empirical",
                            )
                        },
                    }
                )
                print(name, block + 1, good_counts.tolist(), flush=True)
                if np.all(good_counts >= [10, 2]):
                    break
            if not np.all(good_counts >= [10, 2]):
                raise RuntimeError(
                    f"Insufficient elites: {name}, {good_counts}"
                )
            arrays = {
                key: np.concatenate([item[key] for item in collected])
                for key in collected[0]
            }
            np.savez_compressed(destination / f"{name}_all.npz", **arrays)
            good = arrays["empirical"] < 0.1
            masks = arrays["masks"][good]
            features = np.stack(
                (arrays["weights"][good], arrays["gradients"][good] * masks),
                -1,
            )
            scale = np.abs(features).max((1, 2), keepdims=True).clip(1e-12)
            cards = destination / f"{name}_cards.npz"
            np.savez_compressed(
                cards,
                x=features / scale,
                masks=masks,
                roles=arrays["roles"][good],
            )
            record = {
                "id": name,
                "a": a,
                "b": b,
                "function": function,
                "cards": cards.name,
                "sha256": digest(cards),
                "fits": len(good),
                "elites": int(good.sum()),
                "roles": good_counts.tolist(),
                "seconds": time.perf_counter() - started,
                "task": spec,
            }
            save_json(record_path, record)
            manifest.append(record)
            save_json(destination / "manifest.json", manifest)
    save_json(
        RUN / "BANK_COMPLETE.json",
        {
            "tasks": len(manifest),
            "fits": sum(r["fits"] for r in manifest),
            "elites": sum(r["elites"] for r in manifest),
        },
    )


if __name__ == "__main__":
    main()
