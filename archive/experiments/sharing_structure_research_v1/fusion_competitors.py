"""WSR and OSCAR objectives with native exact equalities, no k-means readout.

Independent proximal fits are intersected across source functions to obtain one
common indexator. This multi-task readout is explicit, not an unchanged author
benchmark. WSR models equality; OSCAR additionally models relative signs.
"""

import argparse
import itertools
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from numba import njit

from experiments.darts_structured_tasks_v1.data import digest, save_json

from .competitor_models import signed_basis, validation_loss
from .competitor_pilot import regression_validation


@njit(cache=True)
def ordered_prox(values, fusion, lasso, signed):
    """Sorted isotonic proximal map, implemented by exact linear-time PAVA."""
    configurations, tasks, entries = values.shape
    result = np.empty_like(values)
    for configuration in range(configurations):
        for task in range(tasks):
            vector = values[configuration, task]
            source = np.abs(vector) if signed else vector
            order = np.argsort(source)
            levels = np.empty(entries)
            counts = np.empty(entries, dtype=np.int64)
            blocks = 0
            for position in range(entries):
                weight = (
                    position / (entries - 1)
                    if signed
                    else (2 * position - entries + 1) / (entries - 1)
                )
                levels[blocks] = (
                    source[order[position]] - fusion[configuration] * weight
                )
                if signed:
                    levels[blocks] -= lasso[configuration]
                counts[blocks] = 1
                blocks += 1
                while blocks > 1 and levels[blocks - 2] > levels[blocks - 1]:
                    count = counts[blocks - 2] + counts[blocks - 1]
                    levels[blocks - 2] = (
                        levels[blocks - 2] * counts[blocks - 2]
                        + levels[blocks - 1] * counts[blocks - 1]
                    ) / count
                    counts[blocks - 2] = count
                    blocks -= 1
            position = 0
            for block in range(blocks):
                value = (
                    max(0.0, levels[block])
                    if signed
                    else np.sign(levels[block])
                    * max(abs(levels[block]) - lasso[configuration], 0.0)
                )
                for _ in range(counts[block]):
                    original = order[position]
                    result[configuration, task, original] = (
                        value * np.sign(vector[original]) if signed else value
                    )
                    position += 1
    return result


def native_common_index(weights, signed, codes):
    tasks, outputs, inputs = weights.shape
    profiles = weights.reshape(tasks, -1).T.copy()
    labels, orientation, groups = [], [], {}
    for profile in profiles:
        nonzero = np.flatnonzero(profile)
        if not len(nonzero):
            labels.append(0)
            orientation.append(1)
            continue
        sign = int(np.sign(profile[nonzero[0]])) if signed else 1
        canonical = profile * sign
        canonical[canonical == 0] = 0.0
        key = canonical.tobytes()
        if key not in groups:
            groups[key] = len(groups) + 1
        labels.append(groups[key])
        orientation.append(sign)
    if len(groups) > codes:
        return None, len(groups)
    index = (np.array(labels) * np.array(orientation)).reshape(outputs, inputs)
    return index.astype(np.int16), len(groups)


def register(root):
    path = root / "fusion_registration.json"
    if path.exists():
        raise RuntimeError("Fusion controls already registered")
    if (root / "confirmation_models_frozen.json").exists():
        raise RuntimeError("Register extra competitors before opening target data")
    rows = [
        {"fusion": float(value), "lasso": float(value * ratio)}
        for value, ratio in itertools.product(
            np.geomspace(1e-5, 10, 7), (0.0, 0.1, 1.0)
        )
    ]
    plan = {
        "registered_utc": datetime.now(UTC).isoformat(),
        "methods": ["wsr_prox", "oscar_prox"],
        "rows": rows,
        "steps": 600,
        "checkpoint_every": 20,
        "pilot_source_jobs": [3, 9, 15, 21],
        "n": 131072,
        "smooth_objective": "Per-task half-SSE/n. Independent source functions, full sample curvature, FISTA with step 1/max_t lambda_max(G_t/n), initialized at dense OLS.",
        "penalties": {
            "wsr_prox": "lambdaR sum_{e<f}|w_e-w_f|/(E-1) + lambda1 sum_e|w_e|",
            "oscar_prox": "lambdaR sum_{e<f}max(|w_e|,|w_f|)/(E-1) + lambda1 sum_e|w_e|",
        },
        "native_readout": "Exact equality (WSR) or equality up to global sign of whole source weight profiles (OSCAR), fixed zero only if all source entries exactly zero. No tolerance, k-means, true mask or group sizes. Candidates with more than known K common groups are not eligible under this benchmark's codebook budget; all-zero I is eligible.",
        "capacity": "Unsigned WSR is compared on positive-index convolution geometries only. OSCAR supports the full signed dictionary. Same source-validation functions and HPO criterion; no teacher or target selection.",
        "selection": "One global setting by mean best observed source-validation MSE on pilot geometries; WSR conv-only, OSCAR all4. Confirmation checkpoint uses same source-validation criterion, settings frozen before new teacher audit.",
        "scope": "Transparent objective-level implementations with an explicit multi-task native-equality intersection, not unchanged WSR author scripts. OSCAR is a known signed fusion control; no novelty claimed.",
        "code_sha256": digest(Path(__file__)),
    }
    save_json(path, plan)
    (root / "fusion_pilot").mkdir()


def train(method, rows, stats, validation, plan, configuration):
    gram = stats["gram"] / configuration["n"]
    cross = stats["cross"] / configuration["n"]
    initial = np.linalg.solve(gram, cross.swapaxes(-1, -2)).swapaxes(-1, -2)
    count, shape = len(rows), initial.shape
    current = np.repeat(initial[None], count, axis=0)
    momentum, acceleration = current.copy(), 1.0
    step_size = 1 / np.linalg.eigvalsh(gram).max()
    fusion = np.array([row["fusion"] for row in rows]) * step_size
    lasso = np.array([row["lasso"] for row in rows]) * step_size
    signed = method == "oscar_prox"
    best, saved = np.full(count, np.inf), np.zeros((count, *shape[1:]), dtype=np.int16)
    steps, history = np.zeros(count, dtype=int), []
    started = time.perf_counter()
    for step in range(configuration["steps"] + 1):
        if (
            step % configuration["checkpoint_every"] == 0
            or step == configuration["steps"]
        ):
            groups = []
            for number in range(count):
                index, native_groups = native_common_index(
                    current[number], signed, plan["codes"]
                )
                groups.append(native_groups)
                if index is not None:
                    value = float(
                        validation_loss(
                            signed_basis(torch.tensor(index)[None], plan["codes"]),
                            validation,
                        )[0]
                    )
                    if value < best[number]:
                        best[number], saved[number], steps[number] = value, index, step
            history.append(
                {
                    "step": step,
                    "native_groups": groups,
                    "best_validation_mse": [
                        None if not np.isfinite(value) else float(value)
                        for value in best
                    ],
                }
            )
        if step == configuration["steps"]:
            break
        gradient = momentum @ gram[None] - cross[None]
        following = ordered_prox(
            (momentum - step_size * gradient).reshape(count, shape[0], -1),
            fusion,
            lasso,
            signed,
        ).reshape(current.shape)
        next_acceleration = (1 + np.sqrt(1 + 4 * acceleration**2)) / 2
        momentum = following + (acceleration - 1) / next_acceleration * (
            following - current
        )
        current, acceleration = following, next_acceleration
    return {
        "indices": saved,
        "selection_mse": best,
        "selected_step": steps,
        "history": history,
        "seconds": time.perf_counter() - started,
    }


def run_pilot(root, job_id):
    configuration = json.loads((root / "fusion_registration.json").read_text())
    plan = json.loads((root / "protocol.json").read_text())
    source = Path(plan["source_root"])
    source_plan = json.loads((source / "protocol.json").read_text())
    jobs = [
        (job, method)
        for job in configuration["pilot_source_jobs"]
        for method in configuration["methods"]
        if method == "oscar_prox"
        or source_plan["jobs"][job]["family"]["kind"] == "conv"
    ]
    source_job, method = jobs[job_id]
    path = root / "fusion_pilot" / f"job_{job_id:02d}.json"
    if path.exists():
        raise RuntimeError("Fusion pilot already complete")
    family = source_plan["jobs"][source_job]["family"]
    validation = regression_validation(family["id"], family["kind"], plan, "cpu")
    with np.load(
        source / "statistics" / f"job_{source_job:03d}_n{configuration['n']}.npz"
    ) as archive:
        stats = dict(archive)
    print("fusion pilot", job_id, method, family, flush=True)
    result = train(
        method, configuration["rows"], stats, validation, plan, configuration
    )
    arrays = {
        key: value for key, value in result.items() if isinstance(value, np.ndarray)
    }
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    save_json(
        path,
        {
            "method": method,
            "family": family,
            "seconds": result["seconds"],
            "history": result["history"],
        },
    )
    print(
        "fusion pilot finished",
        job_id,
        float(np.min(result["selection_mse"])),
        result["seconds"],
        flush=True,
    )


def freeze_hpo(root):
    configuration = json.loads((root / "fusion_registration.json").read_text())
    rows, hashes = {}, {}
    for method in configuration["methods"]:
        scores = []
        for job in range(6):
            path = root / "fusion_pilot" / f"job_{job:02d}.json"
            metadata = json.loads(path.read_text())
            hashes[str(path.relative_to(root))] = digest(path)
            if metadata["method"] == method:
                with np.load(path.with_suffix(".npz")) as archive:
                    scores.append(archive["selection_mse"])
        mean = np.mean(scores, axis=0)
        winner = int(mean.argmin())
        if not np.isfinite(mean[winner]):
            raise RuntimeError(f"No native budget-feasible pilot candidate: {method}")
        rows[method] = {
            "row": winner,
            "settings": configuration["rows"][winner],
            "mean_validation_mse": float(mean[winner]),
            "all_mean_validation_mse": [
                None if not np.isfinite(value) else float(value) for value in mean
            ],
        }
    destination = root / "fusion_hpo_frozen.json"
    if destination.exists():
        raise RuntimeError("Fusion HPO already frozen")
    save_json(
        destination,
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "choices": rows,
            "inputs_sha256": hashes,
            "registration_sha256": digest(root / "fusion_registration.json"),
        },
    )
    print(rows)


def confirm_case(root, job_id):
    configuration = json.loads((root / "fusion_registration.json").read_text())
    plan = json.loads((root / "protocol.json").read_text())
    registration = json.loads((root / "confirmation_registration.json").read_text())
    choices = json.loads((root / "fusion_hpo_frozen.json").read_text())["choices"]
    job = registration["jobs"][job_id]
    family = job["family"]
    folder = root / "confirmation" / f"case_{job_id:03d}"
    if not (folder / "complete.json").exists():
        print("fusion source case still running", job_id, flush=True)
        return
    with np.load(folder / f"source_n{configuration['n']}.npz") as archive:
        stats = dict(archive)
    validation = regression_validation(family["id"], family["kind"], plan, "cpu")
    for method in configuration["methods"]:
        if method == "wsr_prox" and family["kind"] != "conv":
            continue
        path = folder / f"fusion_{method}.json"
        if path.exists():
            continue
        result = train(
            method,
            [choices[method]["settings"]],
            stats,
            validation,
            plan,
            configuration,
        )
        np.savez_compressed(
            path.with_suffix(".npz"),
            **{
                key: value
                for key, value in result.items()
                if isinstance(value, np.ndarray)
            },
        )
        save_json(
            path,
            {
                "method": method,
                "family": family,
                "history": result["history"],
                "seconds": result["seconds"],
                "returned_index": bool(np.isfinite(result["selection_mse"][0])),
                "hpo_sha256": digest(root / "fusion_hpo_frozen.json"),
            },
        )
        print(
            "fusion confirmation",
            job_id,
            method,
            float(result["selection_mse"][0]),
            flush=True,
        )


def freeze_models(root):
    registration = json.loads((root / "confirmation_registration.json").read_text())
    hashes = {}
    for job_id, job in enumerate(registration["jobs"]):
        folder = root / "confirmation" / f"case_{job_id:03d}"
        for method in ("oscar_prox", "wsr_prox"):
            if method == "wsr_prox" and job["family"]["kind"] != "conv":
                continue
            for suffix in ("npz", "json"):
                path = folder / f"fusion_{method}.{suffix}"
                hashes[str(path.relative_to(root))] = digest(path)
    destination = root / "fusion_models_frozen.json"
    if destination.exists():
        raise RuntimeError("Fusion models already frozen")
    save_json(
        destination,
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "sha256": hashes,
            "hpo_sha256": digest(root / "fusion_hpo_frozen.json"),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("register", "pilot", "freeze", "confirm", "models_freeze")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "register":
        register(args.root)
    elif args.action == "freeze":
        freeze_hpo(args.root)
    elif args.action == "models_freeze":
        freeze_models(args.root)
    elif args.action == "confirm":
        configuration = json.loads(
            (args.root / "confirmation_registration.json").read_text()
        )
        for job in range(args.shard, len(configuration["jobs"]), args.shards):
            confirm_case(args.root, job)
    else:
        for job in range(args.shard, 6, args.shards):
            run_pilot(args.root, job)


if __name__ == "__main__":
    main()
