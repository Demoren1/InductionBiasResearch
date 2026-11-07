"""Measure unchanged fixed-I numerical objectives and batched execution speed."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json

from .shared_structure_data import load_source
from .shared_structure_numeric import fit_codes as serial_fit
from .shared_structure_numeric import fresh_codes
from .shared_structure_numeric_batch import fit_codes as batch_fit


def run(root):
    plan = json.loads((root / "protocol.json").read_text())
    path = root / "extract/job_0000.json"
    metadata = json.loads(path.read_text())
    job = metadata["job"]
    with np.load(path.with_suffix(".npz")) as arrays:
        keys = [metadata["records"][n]["key"] for n in metadata["selected_records"][:4]]
        indices = np.stack([arrays[key + "__canonical_I"] for key in keys])
    observed = [
        load_source(root, plan, job["family_id"], task, job["noise"], "cuda:0")
        for task in range(plan["sources"])
    ]
    x, vx = observed[0]["train_x"], observed[0]["fit_validation_x"]
    y = torch.stack([value[f"train_y_{job['noise']}"] for value in observed])
    vy = torch.stack([value[f"fit_validation_y_{job['noise']}"] for value in observed])
    initial = torch.stack(
        [
            fresh_codes(
                plan, plan["families"][job["family_id"]]["family"], task, "cuda:0"
            )
            for task in range(plan["sources"])
        ]
    )
    # Warm runtime/library initialization before the timed scientific workload.
    serial_fit(indices[0], x, y[0], vx, vy[0], initial[0], plan, max_steps=1)
    batch_fit(indices[:1], x, y[:1], vx, vy[:1], initial[:1], plan, max_steps=1)
    torch.cuda.synchronize()
    start = time.perf_counter()
    serial_codes, serial_histories = [], []
    for index in indices:
        codes, histories = [], []
        for task in range(plan["sources"]):
            code, history = serial_fit(
                index, x, y[task], vx, vy[task], initial[task], plan
            )
            codes.append(code)
            histories.append(history)
        serial_codes.append(torch.stack(codes))
        serial_histories.append(histories)
    serial_codes = torch.stack(serial_codes)
    torch.cuda.synchronize()
    serial_seconds = time.perf_counter() - start
    print(json.dumps({"serial_seconds": serial_seconds}), flush=True)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    batched_codes, batched_histories, _ = batch_fit(
        indices, x, y, vx, vy, initial, plan
    )
    torch.cuda.synchronize()
    batched_seconds = time.perf_counter() - start
    print(json.dumps({"batched_seconds": batched_seconds}), flush=True)
    score_errors, step_errors, start_errors = [], [], []
    trajectory_errors, length_differences = [], []
    for batch in range(len(indices)):
        for task in range(plan["sources"]):
            old, new = serial_histories[batch][task], batched_histories[batch][task]
            before, after = (
                np.array(old["best_validation_per_start"]),
                np.array(new["best_validation_per_start"]),
            )
            score_errors.append(
                float(np.max(np.abs(before - after) / (1 + np.abs(before))))
            )
            step_errors.append(
                int(
                    np.max(
                        np.abs(
                            np.array(old["best_step_per_start"])
                            - np.array(new["best_step_per_start"])
                        )
                    )
                )
            )
            start_errors.append(old["selected_start"] != new["selected_start"])
            length_differences.append(len(new["history"]) - len(old["history"]))
            for step in range(max(len(old["history"]), len(new["history"]))):
                first = np.array(
                    old["history"][min(step, len(old["history"]) - 1)]["train_mse"]
                )
                second = np.array(
                    new["history"][min(step, len(new["history"]) - 1)]["train_mse"]
                )
                trajectory_errors.append(
                    float(np.max(np.abs(first - second) / (1 + np.abs(first))))
                )
    result = {
        "kind": "fixed-I execution performance and numerical parity diagnosis, not structural audit",
        "source_protocol_sha256": digest(root / "protocol.json"),
        "source_job_sha256": digest(path),
        "source_arrays_sha256": digest(path.with_suffix(".npz")),
        "batched_code_sha256": digest(
            Path(__file__).with_name("shared_structure_numeric_batch.py")
        ),
        "diagnostic_code_sha256": digest(Path(__file__)),
        "gpu": 0,
        "measurement": "steady wall-clock under existing source-worker load; no profiler in timed sections",
        "indexators": len(indices),
        "tasks": plan["sources"],
        "signed_starts": len(initial[0]),
        "numeric_steps_limit": plan["numeric_code_steps"],
        "serial_seconds": serial_seconds,
        "batched_seconds": batched_seconds,
        "speedup": serial_seconds / batched_seconds,
        "max_relative_per_start_validation_difference": max(score_errors),
        "max_relative_training_trajectory_difference_with_frozen_tails": max(
            trajectory_errors
        ),
        "max_best_step_difference": max(step_errors),
        "history_length_differences": length_differences,
        "selected_restart_differences": sum(start_errors),
        "max_absolute_selected_C_difference": float(
            (serial_codes - batched_codes).abs().max()
        ),
        "batched_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "teacher_or_meta_Y_opened": False,
    }
    result["parity_passed"] = (
        result["max_relative_per_start_validation_difference"] < 1e-8
        and result["max_relative_training_trajectory_difference_with_frozen_tails"]
        < 1e-8
        and result["selected_restart_differences"] == 0
        and result["max_absolute_selected_C_difference"] < 1e-6
    )
    np.savez_compressed(
        root / "batched_numeric_diagnostic_arrays.npz",
        indexators=indices,
        serial_C=serial_codes.cpu().numpy(),
        batched_C=batched_codes.cpu().numpy(),
    )
    save_json(root / "batched_numeric_performance.json", result)
    print(json.dumps(result, indent=2), flush=True)
    if not result["parity_passed"]:
        raise ArithmeticError("Numerical differences need diagnosis before adoption")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError(
            "This short registered performance measurement uses GPU0 only"
        )
    torch.set_num_threads(1)
    run(args.root)


if __name__ == "__main__":
    main()
