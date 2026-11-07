"""Source-only numerical/decoding calibration, freezing, then target fitting."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.real_image_ar.common import (
    BASE,
    fit_masks,
    load_plan,
    make_cases,
    numeric_configs,
)
from experiments.real_image_ar.prior import sample


def evaluate_method(plan, name, role, size, method, device):
    root = Path(plan["run"]) / name
    output = root / role / f"n{size}" / method
    if (output / "complete.json").exists():
        return
    output.mkdir(parents=True, exist_ok=True)
    data = dict(np.load(root / f"{role}_{size}.npz"))
    restarts = (
        plan["calibration_restarts"]
        if role == "calibration"
        else plan["target_restarts"]
    )
    long = method.endswith("_long")
    dense = method in [
        "adam",
        "sgd",
        "transfer",
        "frozen_transfer",
        "linear",
        "adam_long",
        "sgd_long",
    ]
    temperatures = (
        plan["temperatures"]
        if method in ["ar", "frequency"]
        else [1.0 if method in ["untrained_ar", "random", "bank"] else 0.0]
    )
    if role == "target":
        selected = json.loads(
            (root / "calibration" / f"n{size}" / method / "selection.json").read_text()
        )
        temperatures, configs = [selected["temperature"]], [selected["config"]]
    elif long:
        base = json.loads(
            (
                root
                / "calibration"
                / f"n{size}"
                / method.removesuffix("_long")
                / "selection.json"
            ).read_text()
        )["config"]
        configs = [
            dict(base, lr=rate, schedule=schedule)
            for rate in [base["lr"] / 3, base["lr"], base["lr"] * 3]
            for schedule in ["constant", "cosine"]
        ]
    else:
        configs = numeric_configs(plan, method)
    pretrained = None
    if method in ["transfer", "frozen_transfer"]:
        stored = np.load(root / "pretrained.npz")
        pretrained = [stored["weight"], stored["bias"]]
    candidates = []
    for temperature in temperatures:
        path = output / f"temperature_{temperature:g}".replace(".", "p")
        if path.with_suffix(".done.json").exists():
            arrays = dict(np.load(path.with_suffix(".npz")))
        else:
            masks = (
                None
                if dense
                else sample(
                    plan,
                    name,
                    method,
                    temperature,
                    len(data["support_y"]) * restarts,
                    device,
                )
            )
            arrays = fit_masks(
                plan,
                data,
                masks,
                method,
                configs,
                restarts,
                path,
                device,
                steps=4 * plan["steps"] if long else None,
                pretrained=pretrained,
            )
        loss = (
            arrays["query_loss"]
            .reshape(len(configs), len(data["support_y"]), restarts)
            .min(-1)
            .mean(-1)
        )
        candidates.extend(
            {
                "temperature": temperature,
                "config": config,
                "source_query_bce": float(score),
                "file": path.name + ".npz",
                "config_id": cfg,
            }
            for cfg, (config, score) in enumerate(zip(configs, loss))
        )
    chosen = min(candidates, key=lambda item: item["source_query_bce"])
    if role == "calibration":
        (output / "selection.json").write_text(json.dumps(chosen, indent=2) + "\n")
        (output / "grid.json").write_text(json.dumps(candidates, indent=2) + "\n")
    (output / "complete.json").write_text(
        json.dumps(
            {
                "role": role,
                "method": method,
                "support_per_class": size,
                "restarts": restarts,
                "steps": 4 * plan["steps"] if long else plan["steps"],
                "files": [
                    f"temperature_{temperature:g}".replace(".", "p") + ".npz"
                    for temperature in temperatures
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print("COMPLETE", name, role, size, method, chosen, flush=True)


def freeze(plan, name):
    root = Path(plan["run"]) / name
    if (root / "frozen.json").exists():
        return
    files = [
        root / name
        for name in [
            "bank.npz",
            "prior.pt",
            "untrained_ar.pt",
            "prior_inputs.npz",
            "prior.json",
            "pretrained.npz",
            "pretrain.json",
            "protocol.json",
            "prepared.json",
            "source.npz",
            "pretrain_data.npz",
            "original_bank.npz",
            "bank_audit/complete.json",
        ]
    ]
    files.extend(root.glob("calibration_*_ids.npz"))
    for size in plan["support_per_class"]:
        methods = plan["methods"] + (["adam_long", "sgd_long"] if size == 16 else [])
        for method in methods:
            directory = root / "calibration" / f"n{size}" / method
            if not (directory / "complete.json").exists():
                raise RuntimeError(f"Calibration not complete: {directory}")
            files.append(directory / "selection.json")
    primary = plan["primary"]
    rankings = {
        method: json.loads(
            (
                root
                / "calibration"
                / f"n{primary['support_per_class']}"
                / method
                / "selection.json"
            ).read_text()
        )["source_query_bce"]
        for method in primary["competitors"]
    }
    files.extend(
        [
            *[
                BASE / filename
                for filename in ["common.py", "run.py", "prior.py", "evaluate.py"]
            ],
            BASE / "protocol.json",
            Path("pattern/autoregressive/model.py"),
            Path("experiments/dynamic_signal_v1/fit.py"),
        ]
    )
    hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    }
    (root / "frozen.json").write_text(
        json.dumps(
            {
                "hashes": hashes,
                "primary_method": "ar",
                "primary_competitor": min(rankings, key=rankings.get),
                "source_query_rankings": rankings,
                "target_results_used": False,
            },
            indent=2,
        )
        + "\n"
    )
    make_cases(plan, name, "target")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["MNIST", "FashionMNIST"], required=True)
    parser.add_argument(
        "--stage", choices=["calibration", "freeze", "target"], required=True
    )
    parser.add_argument("--size", type=int, choices=[16, 64, 256], nargs="+")
    parser.add_argument(
        "--group", choices=["all", "structural", "dense"], default="all"
    )
    args = parser.parse_args()
    plan = load_plan()
    torch.set_num_threads(2)
    if args.stage == "freeze":
        freeze(plan, args.dataset)
        return
    if (
        args.stage == "target"
        and not (Path(plan["run"]) / args.dataset / "frozen.json").exists()
    ):
        raise RuntimeError("Target evaluation requires frozen source choices")
    for size in args.size or plan["support_per_class"]:
        methods = plan["methods"] + (["adam_long", "sgd_long"] if size == 16 else [])
        structural = {"ar", "frequency", "consensus", "bank", "random", "untrained_ar"}
        methods = [
            method
            for method in methods
            if args.group == "all"
            or (method in structural) == (args.group == "structural")
        ]
        for method in methods:
            evaluate_method(
                plan, args.dataset, args.stage, size, method, torch.device("cuda")
            )


if __name__ == "__main__":
    main()
