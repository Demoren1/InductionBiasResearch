import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.dynamic_signal_v1.fit import fit, model_indices


def configurations(plan, method):
    grid = plan["grids"]
    if method == "oracle":
        return [
            {
                "optimizer": "adam",
                "lr": rate,
                "weight_decay": decay,
                "momentum": 0.0,
                "initial_scale": scale,
            }
            for rate, scale, decay in itertools.product(
                grid["codebook_lr"],
                grid["codebook_scale"],
                grid.get("codebook_decay", [0.0]),
            )
        ]
    if method == "linear":
        return [
            {
                "optimizer": "adam",
                "lr": rate,
                "weight_decay": decay,
                "momentum": 0.0,
                "initial_scale": 1.0,
            }
            for rate, decay in itertools.product(grid["adam_lr"], grid["weight_decay"])
        ]
    if method == "adam":
        return [
            {
                "optimizer": "adam",
                "lr": rate,
                "weight_decay": decay,
                "momentum": 0.0,
                "initial_scale": scale,
            }
            for rate, decay, scale in itertools.product(
                grid["adam_lr"], grid["weight_decay"], grid["initial_scale"]
            )
        ]
    return [
        {
            "optimizer": "sgd",
            "lr": rate,
            "weight_decay": decay,
            "momentum": momentum,
            "initial_scale": scale,
        }
        for rate, decay, momentum, scale in itertools.product(
            grid["sgd_lr"],
            grid["weight_decay"],
            grid["momentum"],
            grid["initial_scale"],
        )
    ]


def choose(arrays, configs, tasks, restarts):
    loss = arrays["query_loss"].reshape(len(configs), tasks, restarts)
    ranked = loss.min(-1).mean(-1)
    selected = int(ranked.argmin())
    return selected, [
        {"config": config, "mean_best_query_bce": float(value)}
        for config, value in zip(configs, ranked)
    ]


def bayes(plan, role, noise_id, output):
    root = Path(plan["run"])
    observations = np.load(root / role / f"noise_{noise_id}.npz")
    audit = np.load(root / role / "teacher_audit.npz")
    x = observations["test_x"][..., audit["permutation"]]
    x = x.reshape(len(x), -1, plan["blocks"], plan["block_size"])
    projections = np.einsum("tngd,td->tng", x, audit["directions"])
    values = torch.tensor(projections / plan["noise"][noise_id] ** 2)
    logits = values.logsumexp(-1) - (-values).logsumexp(-1)
    predicted = (logits >= 0).numpy()
    accuracy = (predicted == observations["test_y"]).mean(1)
    output.write_text(
        json.dumps(
            {"bayes_accuracy": accuracy.tolist(), "mean": float(accuracy.mean())},
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["calibrate", "target"], required=True)
    parser.add_argument("--noise-id", type=int, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument(
        "--protocol", default="experiments/dynamic_signal_v1/protocol.json"
    )
    parser.add_argument("--variant", default="")
    args = parser.parse_args()
    plan = json.loads(Path(args.protocol).read_text())
    root = Path(plan["run"])
    torch.set_num_threads(2)
    device = torch.device("cuda")
    role = "calibration" if args.stage == "calibrate" else "target"
    output = (
        root
        / (
            ("calibration_fit" if args.stage == "calibrate" else "target_fit")
            + args.variant
        )
        / f"noise_{args.noise_id}_n{args.size}"
    )
    output.mkdir(parents=True, exist_ok=False)
    raw = dict(np.load(root / role / f"noise_{args.noise_id}.npz"))
    teacher = np.load(root / role / "teacher_audit.npz")
    oracle = model_indices(
        teacher["permutation"], plan["blocks"], plan["block_size"], device
    )
    started = time.time()
    choices = {}
    for method in ["oracle", "adam", "sgd", "linear"]:
        if args.stage == "calibrate":
            configs = configurations(plan, method)
            restarts = plan["calibration_restarts"]
        else:
            calibration_size = max(
                [size for size in plan["calibration_sizes"] if size <= args.size]
                or [min(plan["calibration_sizes"])]
            )
            selected = json.loads(
                (
                    root
                    / "calibration_fit"
                    / f"noise_{args.noise_id}_n{calibration_size}"
                    / "selection.json"
                ).read_text()
            )
            configs = [selected[method]["config"]]
            restarts = plan["target_restarts"]
        arrays = fit(
            plan,
            raw,
            method,
            configs,
            args.size,
            restarts,
            device,
            output / method,
            indices=oracle if method == "oracle" else None,
        )
        best, grid = choose(arrays, configs, len(raw["support_y"]), restarts)
        choices[method] = {"config": configs[best], "grid": grid}
        print("SELECTED", method, choices[method]["config"], flush=True)
        (output / "selection.json").write_text(json.dumps(choices, indent=2) + "\n")
    bayes(plan, role, args.noise_id, output / "bayes_audit.json")
    (output / "complete.json").write_text(
        json.dumps(
            {"seconds": time.time() - started, "gpu": torch.cuda.get_device_name()},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
