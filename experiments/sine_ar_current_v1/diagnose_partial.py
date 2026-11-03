"""Recover per-output risks for one original source task, for diagnosis only."""

import json
from pathlib import Path

import numpy as np
import torch

from experiments.sine_ar_current_v1.run import fit


def main():
    torch.set_num_threads(1)
    base = Path(__file__).resolve().parent
    source = base / "results/run_20261103"
    output = base / "results/diagnosis_20261103"
    plan = json.loads((source / "protocol.json").read_text())
    spec = json.loads((source / "source_tasks.json").read_text())[0]
    with np.load(source / "source/task_00.npz") as saved:
        masks, old_query = saved["masks"], saved["query"]
    result, _ = fit(
        plan,
        [spec],
        masks.reshape(1, 1, -1, 8, 2),
        [plan["source_sgd"]],
        plan["seed"],
        "cuda:0",
    )
    weights = result["weights"].reshape(-1, 8, 2).astype(np.float64)
    amplitude = result["amplitude"].reshape(-1, 2).astype(np.float64)
    low, high = np.zeros((8, 2)), np.zeros((8, 2))
    low[spec["rows"], [0, 1]] = spec["low"]
    high[spec["rows"], [0, 1]] = spec["high"]
    small = np.asarray(spec["small"])

    def kernel(left, right):
        return 0.5 * (
            np.exp(-((left - right) ** 2).sum(-2) / 2)
            - np.exp(-((left + right) ** 2).sum(-2) / 2)
        )

    target = (
        small**2 * kernel(low, low)
        + kernel(high, high)
        + 2 * small * kernel(low, high)
    )
    risks = (
        amplitude**2 * kernel(weights, weights)
        - 2
        * amplitude
        * (small * kernel(weights, low) + kernel(weights, high))
        + target
    )
    rows = []
    for threshold in [0.3, 0.325, 0.35]:
        admitted = (old_query >= 0.1) & (old_query < threshold)
        partial = (risks.min(1) < 0.1) & (risks.max(1) > 0.4)
        rows.append(
            {
                "threshold": threshold,
                "newly_accepted": int(admitted.sum()),
                "one_good_one_bad_output": int((admitted & partial).sum()),
                "fraction": float(partial[admitted].mean()),
            }
        )
    example = np.flatnonzero((old_query >= 0.1) & (old_query < 0.325))[:10]
    summary = {
        "source_task": 0,
        "purpose": "One-task numerical audit only; not added to the bank.",
        "additional_fits": len(masks),
        "seconds": float(result["seconds"]),
        "max_query_reproduction_error": float(
            np.max(np.abs(old_query - result["query"].ravel()))
        ),
        "thresholds": rows,
        "examples": [
            {
                "index": int(i),
                "active_edges": np.flatnonzero(masks[i]).tolist(),
                "source_query": float(old_query[i]),
                "population_per_output": risks[i].tolist(),
            }
            for i in example
        ],
    }
    np.savez_compressed(
        output / "source_task00_per_output.npz",
        masks=masks,
        query=old_query,
        per_output_population=risks,
        weights=weights,
        amplitude=amplitude,
    )
    (output / "partial_solution_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
