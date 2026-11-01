import argparse
import copy
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from pattern.autoregressive.model import IndexMADE
from pattern.autoregressive.run import fit_method, save_json
from pattern.datasets import partitions


def proposals(model, frequency, count, temperature, seed):
    torch.manual_seed(seed)
    if model is not None:
        return model.sample(count, temperature=temperature)
    if temperature == 0:
        return frequency.argmax(-1).expand(count, -1)
    distribution = (frequency.log() / temperature).softmax(-1)
    return torch.multinomial(distribution, count, replacement=True).T


def query_summary(path, patterns, restarts):
    data = np.load(path)
    loss = data["query_loss"].reshape(len(patterns), restarts)
    selected = np.arange(len(patterns)) * restarts + loss.argmin(1)
    return {
        "query_bce": float(data["query_loss"][selected].mean()),
        "selected_query_accuracy": float(data["query_accuracy"][selected].mean()),
    }


def pack_histories(output):
    manifest = []
    for path in output.rglob("*_history.json"):
        raw = path.read_bytes()
        packed = gzip.compress(raw, compresslevel=6, mtime=0)
        destination = path.with_suffix(".json.gz")
        destination.write_bytes(packed)
        manifest.append(
            {
                "file": str(destination.relative_to(output)),
                "json_sha256": hashlib.sha256(raw).hexdigest(),
                "json_bytes": len(raw),
                "gzip_bytes": len(packed),
            }
        )
        path.unlink()
    save_json(output / "history_storage.json", manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", default="pattern/autoregressive/debug_protocol.json"
    )
    args = parser.parse_args()
    plan = json.loads(Path(args.protocol).read_text())
    output = Path(plan["run"])
    output.mkdir(parents=True, exist_ok=False)
    calibration = output / "calibration"
    calibration.mkdir()
    started = time.time()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    prior = Path(plan["prior_run"])
    arrays = np.load(prior / "bank_inputs.npz")
    frequency = torch.tensor(arrays["frequency"], device=device)
    mapping = torch.tensor(arrays["mapping"], device=device)
    consensus = torch.tensor(arrays["consensus"].reshape(1, 88), device=device)
    torch.manual_seed(plan["seed"])
    model = IndexMADE().to(device)
    model.load_state_dict(torch.load(prior / "prior.pt", weights_only=True))
    model.eval().requires_grad_(False)
    dependencies = [
        Path(args.protocol),
        prior / "prior.pt",
        prior / "bank_inputs.npz",
        Path(plan["source_protocol"]),
        *Path("pattern/autoregressive").glob("*.py"),
    ]
    hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dependencies
    }
    save_json(output / "frozen_protocol.json", {**plan, "hashes": hashes})
    for path in Path("pattern/autoregressive").glob("*.py"):
        (output / ("snapshot_" + path.name)).write_bytes(path.read_bytes())
    split = {key: value.to(device) for key, value in partitions(plan["seed"]).items()}
    np.savez_compressed(
        output / "splits.npz", **{key: value.cpu() for key, value in split.items()}
    )
    calibration_plan = copy.deepcopy(plan)
    calibration_plan["run"] = str(calibration)
    calibration_plan["fit"]["steps"] = plan["calibration_steps"]
    calibration_plan["fit"]["restarts"] = plan["calibration_restarts"]
    source = json.loads(Path(plan["source_protocol"]).read_text())
    if not set(plan["calibration_patterns"]).issubset(source["source_patterns"]):
        raise ValueError("Calibration patterns must be source patterns")
    count = len(plan["calibration_patterns"]) * plan["calibration_restarts"]
    records = []
    choices = {}
    frozen_proposals = {}
    for family, generator in [("ar", model), ("frequency", None)]:
        for temperature in plan["temperatures"]:
            name = f"{family}_t{temperature:g}"
            indices = proposals(
                generator, frequency, count, temperature, plan["seed"] + 60
            )
            frozen_proposals[name] = indices.cpu().numpy()
            fit_method(
                calibration_plan,
                name,
                indices,
                None,
                torch.zeros(count, 0, device=device),
                mapping,
                plan["calibration_patterns"],
                split,
                device,
            )
            result = query_summary(
                calibration / (name + ".npz"),
                plan["calibration_patterns"],
                plan["calibration_restarts"],
            )
            records.append({"family": family, "temperature": temperature, **result})
        own = [row for row in records if row["family"] == family]
        best = min(row["query_bce"] for row in own)
        eligible = [
            row for row in own if row["query_bce"] <= best + plan["near_best_query_bce"]
        ]
        choices[family] = max(eligible, key=lambda row: row["temperature"])
    np.savez_compressed(calibration / "proposals.npz", **frozen_proposals)
    save_json(calibration / "summary.json", records)
    save_json(
        output / "selection_frozen.json",
        {
            "choices": choices,
            "selected_before_target_fit": True,
            "source_patterns": plan["calibration_patterns"],
            "selection_timestamp_ns": time.time_ns(),
        },
    )
    print("FROZEN TEMPERATURES", choices, flush=True)
    patterns = source["target_patterns"]
    count = len(patterns) * plan["fit"]["restarts"]
    target_proposals = {}
    for family, generator in [("ar", model), ("frequency", None)]:
        for suffix, temperature in [
            ("original", 1.0),
            ("calibrated", choices[family]["temperature"]),
            ("greedy", 0.0),
        ]:
            target_proposals[family + "_" + suffix] = proposals(
                generator, frequency, count, temperature, plan["seed"] + 70
            )
    target_proposals["inferred"] = consensus.expand(count, -1)
    np.savez_compressed(
        output / "proposals.npz",
        **{key: value.cpu() for key, value in target_proposals.items()},
    )
    for method in [*target_proposals, "dense_adam", "dense_sgd"]:
        indices = target_proposals.get(
            method, torch.zeros(count, 88, dtype=torch.long, device=device)
        )
        fit_method(
            plan,
            method,
            indices,
            None,
            torch.zeros(count, 0, device=device),
            mapping,
            patterns,
            split,
            device,
        )
    save_json(
        output / "complete.json",
        {"seconds": time.time() - started, "patterns": patterns},
    )
    pack_histories(output)


if __name__ == "__main__":
    main()
