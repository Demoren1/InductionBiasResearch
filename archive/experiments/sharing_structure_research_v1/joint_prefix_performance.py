"""Paired CPU/GPU whole-prefix timings without teacher or candidate selection."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json

from .count_exchanges import compositions
from .joint_prefix_benchmark import compute_candidate, load_search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"0", "1", "2", "3"}:
        raise RuntimeError("Choose a free GPU0--3 explicitly")
    if (root / "execution_measurement.json").exists():
        raise RuntimeError("Execution measurement already completed")
    torch.set_num_threads(1)
    plan = json.loads((root / "protocol.json").read_text())
    for name, expected in plan["code_sha256"].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise RuntimeError("Registered execution code changed")
    jobs = []
    for number, job in enumerate(plan["extract_jobs"]):
        if job["family_id"] != 4 or job["noise"] != 2.0:
            continue
        if job["method"] == "bootstrap" or (
            len(job["tasks"]) == 1 and job["tasks"] != [0]
        ):
            continue
        key = (
            "axis"
            if job["method"] == "axis_single_refit"
            else f"{len(job['tasks'])}/{int(job['refit'])}/{job['solver']}"
        )
        jobs.append((key, number, job))
    compositions(plan["hidden"], 2 * plan["codes"] + 1)
    warm = torch.eye(16, device="cuda:0", dtype=torch.float64)
    torch.linalg.eigh(warm @ warm)
    torch.cuda.synchronize()
    results = {}
    for key, number, job in jobs:
        data, initial_index, codes, axis = load_search(root, plan, job)
        started = time.perf_counter()
        cpu = compute_candidate(data, initial_index, codes, axis, job, plan, "cpu")
        cpu_seconds = time.perf_counter() - started
        torch.cuda.synchronize()
        started = time.perf_counter()
        gpu = compute_candidate(data, initial_index, codes, axis, job, plan, "cuda:0")
        torch.cuda.synchronize()
        gpu_seconds = time.perf_counter() - started
        same_indices = np.array_equal(cpu["indices"], gpu["indices"])
        code_difference = (
            float(np.max(np.abs(cpu["codebooks"] - gpu["codebooks"])))
            if cpu["codebooks"].shape == gpu["codebooks"].shape
            and cpu["codebooks"].size
            else 0.0
            if cpu["codebooks"].size == gpu["codebooks"].size == 0
            else None
        )
        parity = same_indices and code_difference is not None and code_difference < 1e-6
        results[key] = {
            "job": number,
            "cpu_seconds": cpu_seconds,
            "gpu_seconds": gpu_seconds,
            "cpu_over_gpu_time": cpu_seconds / gpu_seconds,
            "identical_prefix_indices": same_indices,
            "maximum_C_trace_difference": code_difference,
            "cpu_complete": cpu["complete"],
            "gpu_complete": gpu["complete"],
            "cpu_accepted_prefixes": len(cpu["indices"]),
            "gpu_accepted_prefixes": len(gpu["indices"]),
            "backend": "cuda"
            if parity
            and cpu_seconds / gpu_seconds >= plan["minimum_gpu_search_speedup"]
            else "cpu",
            "parity_passed": parity,
        }
        print(json.dumps({"key": key, **results[key]}), flush=True)
    save_json(
        root / "execution_measurement.json",
        {
            "cases": results,
            "source_protocol_sha256": digest(root / "protocol.json"),
            "scope": "Whole census including device transfers, QP, integer optimization and optional C-refit; excludes process startup/NPZ load and prewarmed composition cache/CUDA context. One paired case, not universal throughput or bitwise guarantee.",
            "teacher_meta_or_target_test_opened": False,
            "candidate_retained": False,
            "numeric_differences_use_cpu_reference": True,
        },
    )


if __name__ == "__main__":
    main()
