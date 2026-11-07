"""Benchmark-side data generator, never imported by structural discovery."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", default="experiments/random_structure_search_v1/protocol.json"
    )
    parser.add_argument(
        "--benchmark", default="experiments/random_structure_search_v1/benchmark.json"
    )
    args = parser.parse_args()
    plan = json.loads(Path(args.protocol).read_text())
    root = Path(plan["output"])
    frozen = json.loads((root / "frozen.json").read_text())
    for path, expected in frozen["sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Frozen input changed: {path}")
    benchmark_path = Path(args.benchmark)
    benchmark = json.loads(benchmark_path.read_text())
    output = root / "benchmark"
    output.mkdir(parents=True, exist_ok=False)
    blocks = benchmark["blocks"]
    width = benchmark["block_size"]
    tasks = benchmark["target_tasks"]
    permutation = np.random.default_rng(
        benchmark["source_permutation_seed"]
    ).permutation(blocks * width)
    rng = np.random.default_rng(benchmark["target_seed"])
    directions = rng.normal(size=(tasks, width)).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    data = {}
    for split in ["support", "query", "test"]:
        samples = benchmark[split + "_size"]
        labels = rng.integers(0, 2, size=(tasks, samples)).astype(np.float32)
        locations = rng.integers(blocks, size=(tasks, samples))
        values = (
            benchmark["noise_std"] * rng.normal(size=(tasks, samples, blocks, width))
        ).astype(np.float32)
        values[np.arange(tasks)[:, None], np.arange(samples)[None], locations] += (
            2 * labels[..., None] - 1
        ) * directions[:, None]
        flat = values.reshape(tasks, samples, -1)
        observed = np.empty_like(flat)
        observed[..., permutation] = flat
        data[split + "_x"] = observed
        data[split + "_y"] = labels
    np.savez_compressed(output / "target.npz", **data)
    np.savez_compressed(
        output / "teacher_audit_only.npz",
        permutation=permutation,
        directions=directions,
    )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "created_unix": time.time(),
                "learner_frozen_unix": frozen["frozen_unix"],
                "benchmark": benchmark,
                "target_sha256": hashlib.sha256(
                    (output / "target.npz").read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    print("Created independent target observations after learner freezing", flush=True)


if __name__ == "__main__":
    main()
