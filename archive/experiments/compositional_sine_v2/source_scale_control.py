"""Check source-direction rescaling, using validation tasks only for tuning."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from run import fit_dense, population_mse, sample, save_json

RADII = [0.5, 1.0, 2.0, 4.0, 8.0]
ADAM_RATES = [0.001, 0.01, 0.03]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--adam", action="store_true")
    args = parser.parse_args()
    output = args.run / "source_scale_control"
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((args.run / "protocol.json").read_text())
    save_json(
        output / "seal.json",
        {
            "radii": RADII,
            "adam_rates": ADAM_RATES,
            "adam_enabled": args.adam,
            "device": args.device,
            "script_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "note": "Added after CPU pilot, before this control's evaluation. "
            "Radius and rate use validation tasks only.",
        },
    )
    (output / "run_snapshot.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(plan["threads"])
    torch.set_num_interop_threads(1)
    arrays = np.load(args.run / "target_tasks.npz")
    bank = np.load(args.run / "source_bank.npz")
    directions = bank["weights"]
    directions = directions / np.linalg.norm(directions, axis=0)
    tasks = [
        {"weights": w, "amplitudes": a}
        for w, a in zip(arrays["weights"], arrays["amplitudes"])
    ]
    split = plan["target"]["validation_tasks"]
    options = []
    for radius in RADII:
        losses = []
        for index, task in enumerate(tasks[:split]):
            rng = np.random.default_rng(plan["seed"] + 40000 + index)
            x, y = sample(task, plan["target"]["structure_examples"], rng)
            sx, sy = sample(task, plan["target"]["selection_examples"], rng)
            weights = directions * radius
            amplitudes = np.linalg.lstsq(np.sin(x @ weights), y, rcond=None)[0]
            losses.append(
                float(((np.sin(sx @ weights) @ amplitudes - sy) ** 2).mean())
            )
        options.append(
            {"radius": radius, "validation_mse": float(np.mean(losses))}
        )
    radius = min(options, key=lambda row: row["validation_mse"])["radius"]
    save_json(output / "radius_tuning.json", options)
    print("RADIUS " + json.dumps(options), flush=True)
    chosen = None
    if args.adam:
        options = []
        for rate in ADAM_RATES:
            settings = {
                "optimizer": "adam",
                "norm": radius,
                "hidden_units": directions.shape[1],
                "learning_rate": rate,
            }
            losses = []
            for index, task in enumerate(tasks[:split]):
                _, _, info = fit_dense(
                    plan,
                    task,
                    settings,
                    plan["target"]["tuning_restarts"],
                    plan["seed"] + 50000 + index,
                    directions * radius,
                    args.device,
                )
                losses.append(info["selection_mse"])
            row = {**settings, "validation_mse": float(np.mean(losses))}
            options.append(row)
            save_json(output / "adam_tuning.json", options)
            print("SCALE_TUNING " + json.dumps(row), flush=True)
        chosen = min(options, key=lambda row: row["validation_mse"])
    save_json(
        output / "selected_settings.json", {"radius": radius, "adam": chosen}
    )
    records, models = [], {}
    for index, task in enumerate(tasks[split:]):
        rng = np.random.default_rng(plan["seed"] + 20000 + index)
        x, y = sample(task, plan["target"]["structure_examples"], rng)
        started = time.perf_counter()
        weights = directions * radius
        amplitudes = np.linalg.lstsq(np.sin(x @ weights), y, rcond=None)[0]
        fitted = [
            (
                "rescaled_source_linear",
                weights,
                amplitudes,
                {"seconds": time.perf_counter() - started},
            )
        ]
        if args.adam:
            w, a, info = fit_dense(
                plan,
                task,
                chosen,
                plan["target"]["evaluation_restarts"],
                plan["seed"] + 60000 + index,
                directions * radius,
                args.device,
            )
            fitted.append(("rescaled_source_adam", w, a, info))
        for method, w, a, info in fitted:
            risk = population_mse(w, a, task)
            row = {
                "task": index,
                "method": method,
                "radius": radius,
                "population_mse": risk,
                "success": risk < plan["target"]["success_mse"],
                **info,
            }
            records.append(row)
            models[f"{index}_{method}_weights"] = w
            models[f"{index}_{method}_amplitudes"] = a
            save_json(output / "records.json", records)
            np.savez_compressed(output / "fitted_models.npz", **models)
            print("SCALE_RESULT " + json.dumps(row), flush=True)
    summary = {}
    for method in sorted({row["method"] for row in records}):
        rows = [row for row in records if row["method"] == method]
        summary[method] = {
            "successes": sum(row["success"] for row in rows),
            "tasks": len(rows),
            "mean_mse": float(
                np.mean([row["population_mse"] for row in rows])
            ),
            "mean_seconds": float(np.mean([row["seconds"] for row in rows])),
        }
    save_json(output / "summary.json", summary)
    print("SCALE_SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
