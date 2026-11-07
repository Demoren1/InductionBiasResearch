"""Source-only local refinement of WSR strength and learning-rate boundaries."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from experiments.dynamic_signal_v1.fit import fit
from experiments.dynamic_signal_v1.run import choose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-id", type=int, required=True)
    parser.add_argument("--size", type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(
        Path("experiments/dynamic_signal_v1/protocol_confirmation.json").read_text()
    )
    root = Path(plan["run"])
    parent = (
        root / "weight_sharing" / f"noise_{args.noise_id}" / f"calibrate_n{args.size}"
    )
    original = json.loads((parent / "selection.json").read_text())
    output = parent / "refinement"
    if (output / "complete.json").exists():
        print("Already refined", output, flush=True)
        return
    output.mkdir(exist_ok=False)
    best = original["config"]
    rates = sorted({best["lr"] / 3, best["lr"], best["lr"] * 3})
    sharing = (
        sorted(
            {
                best["sharing_penalty"] / 10,
                best["sharing_penalty"],
                best["sharing_penalty"] * 10,
            }
        )
        if best["sharing_penalty"]
        else [0, 1e-6]
    )
    lasso = (
        sorted({0, best["l1_penalty"], best["l1_penalty"] * 10})
        if best["l1_penalty"]
        else [0, 1e-5]
    )
    configs = [
        dict(best, lr=rate, sharing_penalty=value, l1_penalty=l1)
        for rate, value, l1 in itertools.product(rates, sharing, lasso)
    ]
    data = dict(np.load(root / "calibration" / f"noise_{args.noise_id}.npz"))
    torch.set_num_threads(2)
    arrays = fit(
        plan,
        data,
        "wsr_refinement",
        configs,
        args.size,
        plan["calibration_restarts"],
        torch.device("cuda"),
        output / "wsr",
    )
    chosen, grid = choose(
        arrays, configs, len(data["support_y"]), plan["calibration_restarts"]
    )
    original_score = min(item["mean_best_query_bce"] for item in original["grid"])
    improved = grid[chosen]["mean_best_query_bce"] < original_score
    result = {
        "config": configs[chosen] if improved else original["config"],
        "score": min(original_score, grid[chosen]["mean_best_query_bce"]),
        "improved": improved,
        "original_score": original_score,
        "grid": grid,
        "motivation": "Initial calibration selected learning-rate and/or regularization grid boundaries. One local refinement expands each around its best source-query value; no confirmation target exists yet.",
    }
    (output / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "complete.json").write_text(json.dumps({"calibration_only": True}) + "\n")
    print("REFINED", result["config"], result["score"], flush=True)


if __name__ == "__main__":
    main()
