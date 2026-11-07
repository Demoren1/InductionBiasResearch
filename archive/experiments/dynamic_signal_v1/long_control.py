"""Optimization stress control: longer dense runs and learning-rate schedules."""

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
        "--protocol", default="experiments/dynamic_signal_v1/protocol.json"
    )
    args = parser.parse_args()
    plan = json.loads(Path(args.protocol).read_text())
    root = Path(plan["run"])
    output = (
        root / "long_control" / f"noise_{args.noise_id}" / f"{args.stage}_n{args.size}"
    )
    output.mkdir(parents=True, exist_ok=False)
    role = "calibration" if args.stage == "calibrate" else "target"
    data = dict(np.load(root / role / f"noise_{args.noise_id}.npz"))
    torch.set_num_threads(2)
    device = torch.device("cuda")
    selected = {}
    for method in ["adam", "sgd"]:
        if args.stage == "calibrate":
            base = json.loads(
                (
                    root
                    / "calibration_fit"
                    / f"noise_{args.noise_id}_n{args.size}"
                    / "selection.json"
                ).read_text()
            )[method]["config"]
            rates = sorted({base["lr"], base["lr"] / 3, base["lr"] * 3})
            decays = sorted({base["weight_decay"], 0.0})
            configs = [
                dict(base, lr=rate, weight_decay=decay, schedule=schedule)
                for rate, decay, schedule in itertools.product(
                    rates, decays, ["constant", "cosine"]
                )
            ]
            restarts = plan["calibration_restarts"]
        else:
            budget = args.size if args.size in plan["calibration_sizes"] else 1024
            settings = json.loads(
                (output.parent / f"calibrate_n{budget}" / "selection.json").read_text()
            )
            configs = [settings[method]["config"]]
            restarts = plan["target_restarts"]
        arrays = fit(
            plan,
            data,
            method,
            configs,
            args.size,
            restarts,
            device,
            output / method,
            steps=16384,
        )
        best, grid = choose(arrays, configs, len(data["support_y"]), restarts)
        selected[method] = {"config": configs[best], "grid": grid}
        (output / "selection.json").write_text(json.dumps(selected, indent=2) + "\n")
    (output / "complete.json").write_text(
        json.dumps(
            {
                "steps": 16384,
                "motivation": "At8192 labels some dense checkpoints reached the original4096-step limit; enlarge steps4x and search neighboring rates plus cosine decay. At1024 this also checks whether apparent plateau is a learning-rate effect. All decisions use calibration query only.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
