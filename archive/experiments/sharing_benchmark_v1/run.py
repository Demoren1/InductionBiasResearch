"""Registered hyperparameter selection followed by unseen-structure confirmation."""

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json

from . import hard, soft
from .data import BASE, prepare, prepare_family
from .linear import load_observations


def candidates(method):
    if method == "yeh_soft":
        return soft.configurations("yeh")
    return (
        soft.configurations(method)
        if method in ("yeh", "msr", "mtsr")
        else hard.configurations(method)
    )


def methods(plan):
    return [*plan["methods"], "yeh_soft", "posthoc_dp", "posthoc_lloyd"]


def execution_snapshot(root):
    hashes = {path.name: digest(path) for path in BASE.glob("*.py")}
    fingerprint = hashlib.sha256(
        json.dumps(hashes, sort_keys=True).encode()
    ).hexdigest()
    folder = root / "executed_code" / fingerprint[:16]
    if not folder.exists():
        folder.mkdir(parents=True)
        for name in hashes:
            shutil.copy2(BASE / name, folder / name)
        save_json(folder / "sha256.json", hashes)
    return str(folder.relative_to(root)), hashes


def run_one(root, profile, family, phase, method, seed, rows, device):
    folder = root / profile / f"family_{family}"
    prepared = json.loads((folder / "prepared.json").read_text())
    for name, expected in prepared["sha256"].items():
        if digest(folder / name) != expected:
            raise RuntimeError(f"Observation file changed: {folder / name}")
    plan = json.loads((root / "protocol.json").read_text())
    spec = json.loads((folder / "specification.json").read_text())
    destination = folder / phase / f"{method}_seed_{seed}"
    metadata_path = destination.with_suffix(".json")
    inputs = {
        "method": method,
        "seed": seed,
        "rows": rows,
        "observation_sha256": digest(folder / "observations.npz"),
        "specification_sha256": digest(folder / "specification.json"),
    }
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text())
        if (
            existing["inputs"] != inputs
            or digest(destination.with_suffix(".npz")) != existing["result_sha256"]
        ):
            raise RuntimeError(
                f"Existing run has different inputs/state: {destination}"
            )
        print("reuse", profile, family, phase, method, seed, flush=True)
        return
    snapshot, hashes = execution_snapshot(root)
    observed = load_observations(folder, device)
    if method == "yeh_soft":
        source = folder / phase / f"yeh_seed_{seed}"
        source_metadata = json.loads(source.with_suffix(".json").read_text())
        if source_metadata["inputs"]["rows"] != rows:
            # The soft and MAP deployments can select different hyperparameters.
            result = soft.run_yeh(rows, spec, plan, observed, seed)
            result = {
                "basis": result["soft_basis"],
                "selection_mse": result["soft_selection_mse"],
                "step": result["soft_step"],
                "history": result["soft_history"],
                "seconds": result["seconds"],
            }
        else:
            with np.load(source.with_suffix(".npz")) as archive:
                result = {
                    "basis": archive["soft_basis"],
                    "selection_mse": archive["soft_selection_mse"],
                    "step": archive["soft_step"],
                    "seconds": 0.0,
                    "history": source_metadata["soft_history"],
                }
    elif method == "yeh":
        result = soft.run_yeh(rows, spec, plan, observed, seed)
    elif method in ("msr", "mtsr"):
        result = soft.run_continuous(method, rows, spec, plan, observed, seed)
    else:
        result = hard.run(method, rows, spec, plan, observed, seed)
    arrays = {
        key: value for key, value in result.items() if isinstance(value, np.ndarray)
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination.with_suffix(".npz"), **arrays)
    save_json(
        metadata_path,
        {
            "inputs": inputs,
            "seconds": result["seconds"],
            "history": result["history"],
            "algorithm_history": result.get("algorithm_history", []),
            "soft_history": result.get("soft_history", []),
            "proposal_anchors": result.get("proposal_anchors"),
            "execution_snapshot": snapshot,
            "executed_sha256": hashes,
            "result_sha256": digest(destination.with_suffix(".npz")),
            "completed_utc": datetime.now(UTC).isoformat(),
        },
    )
    print(
        profile,
        family,
        phase,
        method,
        seed,
        "best_selection",
        float(result["selection_mse"].min()),
        "seconds",
        round(result["seconds"], 2),
        flush=True,
    )


def select_hyperparameters(root, profile, phase, families, seeds, name):
    plan = json.loads((root / "protocol.json").read_text())
    choices = {}
    for method in methods(plan):
        errors, hashes = [], {}
        for family in families:
            folder = root / profile / f"family_{family}" / phase
            for seed in seeds:
                path = folder / f"{method}_seed_{seed}.npz"
                with np.load(path) as archive:
                    errors.append(archive["selection_mse"])
                hashes[str(path.relative_to(root))] = digest(path)
        mean = np.array(errors).mean(0)
        row = int(mean.argmin())
        choices[method] = {
            "row": row,
            "settings": candidates(method)[row],
            "mean_selection_mse": float(mean[row]),
            "all_mean_selection_mse": mean.tolist(),
            "input_sha256": hashes,
        }
    save_json(
        root / name,
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "profile": profile,
            "choices": choices,
            "criterion": plan["selection"]["hpo"],
        },
    )
    print(
        "frozen",
        name,
        {key: value["settings"] for key, value in choices.items()},
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--stage",
        choices=("prepare", "pilot", "freeze", "confirmation", "native", "finalize"),
        required=True,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--method")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.stage == "prepare":
        prepare(args.root)
        plan = json.loads((args.root / "protocol.json").read_text())
        save_json(
            args.root / "candidates.json",
            {method: candidates(method) for method in methods(plan)},
        )
        return
    plan = json.loads((args.root / "protocol.json").read_text())
    device = torch.device(args.device)
    requested = methods(plan) if args.method is None else [args.method]
    if args.stage == "freeze":
        select_hyperparameters(
            args.root,
            "hidden",
            "pilot",
            plan["pilot_families"],
            plan["pilot_seeds"],
            "hpo_frozen.json",
        )
    elif args.stage == "pilot":
        for family in plan["pilot_families"]:
            for seed in plan["pilot_seeds"]:
                for method in requested:
                    run_one(
                        args.root,
                        "hidden",
                        family,
                        "pilot",
                        method,
                        seed,
                        candidates(method),
                        device,
                    )
    elif args.stage == "confirmation":
        frozen = json.loads((args.root / "hpo_frozen.json").read_text())
        for family in plan["confirmation_families"]:
            folder = args.root / "hidden" / f"family_{family}"
            if not folder.exists():
                prepare_family(args.root, plan, "hidden", family)
            for seed in plan["confirmation_seeds"]:
                for method in requested:
                    run_one(
                        args.root,
                        "hidden",
                        family,
                        "confirmation",
                        method,
                        seed,
                        [frozen["choices"][method]["settings"]],
                        device,
                    )
    elif args.stage == "native":
        for method in requested:
            run_one(
                args.root,
                "native",
                301,
                "calibration",
                method,
                1,
                candidates(method),
                device,
            )
        if args.method is not None:
            return
        select_hyperparameters(
            args.root, "native", "calibration", [301], [1], "native_hpo_frozen.json"
        )
        frozen = json.loads((args.root / "native_hpo_frozen.json").read_text())
        for seed in plan["confirmation_seeds"]:
            for method in requested:
                run_one(
                    args.root,
                    "native",
                    301,
                    "confirmation",
                    method,
                    seed,
                    [frozen["choices"][method]["settings"]],
                    device,
                )
    elif args.stage == "finalize":
        results = list(args.root.glob("*/family_*/confirmation/*.npz"))
        expected = (
            (len(plan["confirmation_families"]) + 1)
            * len(plan["confirmation_seeds"])
            * len(methods(plan))
        )
        if len(results) != expected:
            raise RuntimeError(
                f"Expected {expected} frozen results, got {len(results)}"
            )
        names = [
            *results,
            args.root / "hpo_frozen.json",
            args.root / "native_hpo_frozen.json",
            args.root / "candidates.json",
            args.root / "protocol.json",
        ]
        save_json(
            args.root / "all_models_frozen.json",
            {
                "frozen_utc": datetime.now(UTC).isoformat(),
                "sha256": {
                    str(path.relative_to(args.root)): digest(path) for path in names
                },
            },
        )


if __name__ == "__main__":
    main()
