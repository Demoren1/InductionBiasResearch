"""WSR subgradient control, following the authors' FC1 objective.

Primary source: https://proceedings.mlr.press/v238/shakerinava24a.html
Reference code revision b8597758a890bb4ff2424f11dbfa5b09cec90fe2,
get_R/R and train_epoch(ws_alg='subgradient'). The formula is implemented
independently with batched sorted weights. This is not the proximal solver.
"""

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
    parser.add_argument("--stage", choices=["calibrate", "target"], required=True)
    parser.add_argument(
        "--protocol", default="experiments/dynamic_signal_v1/protocol_confirmation.json"
    )
    args = parser.parse_args()
    plan = json.loads(Path(args.protocol).read_text())
    root = Path(plan["run"])
    output = (
        root
        / "weight_sharing"
        / f"noise_{args.noise_id}"
        / f"{args.stage}_n{args.size}"
    )
    output.mkdir(parents=True, exist_ok=False)
    role = "calibration" if args.stage == "calibrate" else "target"
    data = dict(np.load(root / role / f"noise_{args.noise_id}.npz"))
    torch.set_num_threads(2)
    device = torch.device("cuda")
    if args.stage == "calibrate":
        base = json.loads(
            (
                root
                / "calibration_fit"
                / f"noise_{args.noise_id}_n{args.size}"
                / "selection.json"
            ).read_text()
        )["adam"]["config"]
        # Include the independently tuned dense baseline and its neighbors.
        rates = sorted({base["lr"] / 3, base["lr"], base["lr"] * 3})
        configs = [
            dict(base, lr=rate, sharing_penalty=sharing, l1_penalty=lasso)
            for rate, sharing, lasso in itertools.product(
                rates, [0.0, 1e-6, 1e-5, 1e-4, 1e-3], [0.0, 1e-5, 1e-4]
            )
        ]
        restarts = plan["calibration_restarts"]
    else:
        selected = json.loads(
            (output.parent / f"calibrate_n{args.size}" / "selection.json").read_text()
        )
        refined = (
            output.parent / f"calibrate_n{args.size}" / "refinement/selection.json"
        )
        if refined.exists():
            selected = json.loads(refined.read_text())
        configs = [selected["config"]]
        restarts = plan["target_restarts"]
    arrays = fit(
        plan,
        data,
        "wsr_subgradient",
        configs,
        args.size,
        restarts,
        device,
        output / "wsr",
    )
    best, grid = choose(arrays, configs, len(data["support_y"]), restarts)
    (output / "selection.json").write_text(
        json.dumps({"config": configs[best], "grid": grid}, indent=2) + "\n"
    )
    (output / "complete.json").write_text(
        json.dumps(
            {
                "variant": "WSR subgradient objective on FC1 weights and biases; L1 optional; Adam; no oracle structure/source weights.",
                "reference_commit": "b8597758a890bb4ff2424f11dbfa5b09cec90fe2",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
