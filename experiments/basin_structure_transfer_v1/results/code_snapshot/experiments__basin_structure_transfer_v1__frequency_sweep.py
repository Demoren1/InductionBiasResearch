"""Two predeclared harder regimes; retain and report the frequency-4 result."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.basin_structure_transfer_v1.core import fit, make_tasks
from experiments.basin_structure_transfer_v1.run import (
    ROOT,
    choose,
    draw,
    load_prior,
    save_fit,
)


def new_tasks(plan, extension, frequency, phase, device):
    """Same structural library; fresh, independent numerical target values."""
    spec = make_tasks(plan, phase, device)
    rng = np.random.default_rng(plan["seed"] + extension["task_seed_offset"]
                                + int(frequency) * 10000
                                + (1000 if phase == "evaluation" else 0))
    count = len(spec["families"])
    values = rng.normal(size=(count, plan["groups"]))
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    direction = np.take_along_axis(values, spec["indices"], axis=1)
    direction /= np.sqrt(plan["inputs"] / plan["groups"])
    width = extension["frequency_half_width"]
    arrays = {
        "direction": direction,
        "high": rng.uniform(frequency - width, frequency + width, count),
        "low": rng.uniform(*plan["low_frequency"], count),
        "alpha": rng.uniform(*plan["low_amplitude"], count),
    }
    spec.update({key: torch.tensor(value, device=device).float()
                 for key, value in arrays.items()})
    return spec


def run(plan, extension, frequency, device):
    source = Path(extension["source_run"])
    output = Path(extension["output"]) / f"frequency_{int(frequency)}"
    output.mkdir(parents=True, exist_ok=False)
    plan = {**plan, "radii": extension["radii"]}
    (output / "protocol.json").write_text(json.dumps(plan, indent=2))
    (output / "extension.json").write_text(json.dumps(extension, indent=2))
    paths = [ROOT / name for name in ("core.py", "run.py", "frequency_sweep.py",
                                     "frequency_sweep.json")]
    paths += [source / f"prior_{i}.pt" for i in range(len(plan["prior_seeds"]))]
    paths += [source / "structure_bank.npz", source / "numeric_baseline_bank.npz"]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in paths}
    (output / "frozen_before_calibration.json").write_text(json.dumps(hashes, indent=2))
    bank = torch.tensor(np.load(source / "structure_bank.npz")["train"], device=device)
    numeric = torch.tensor(np.load(source / "numeric_baseline_bank.npz")["weights"],
                           device=device)
    spec = new_tasks(plan, extension, frequency, "calibration", device)
    starts, count = plan["calibration_starts"], len(spec["families"])
    seed = plan["seed"] + int(frequency) * 10000 + extension["task_seed_offset"]
    proposals = draw(plan, bank, "retrieval", count, starts, seed + 10)
    chosen, code_choices = {}, []
    modes = [f"code_{opt}" for opt in plan["optimizers"]]
    modes += [f"dense_{opt}" for opt in plan["optimizers"]]
    modes += ["long_adam", "source_transfer_adam"]
    for mode in modes:
        optimizer = mode.split("_")[-1]
        configs = [{"optimizer": optimizer, "lr": lr, "radius": radius}
                   for radius in plan["radii"] for lr in plan["rates"]]
        long = mode.startswith("long_")
        result, history, timing = fit(
            plan, spec, configs, starts // 4 if long else starts, seed + 20,
            device, indices=proposals if mode.startswith("code_") else None,
            source_weights=numeric if mode == "source_transfer_adam" else None,
            steps=plan["steps"] * (4 if long else 1),
        )
        save_fit(output / "calibration" / mode, result, history, timing)
        chosen[mode] = choose(result, configs)
        if mode.startswith("code_"):
            code_choices.append(chosen[mode])
    chosen["code"] = min(code_choices, key=lambda item: item["calibration_query"])
    chosen["temperature"] = json.loads((source / "frozen_selection.json")
                                        .read_text())["temperature"]["temperature"]
    (output / "frozen_selection.json").write_text(json.dumps(chosen, indent=2))
    print(json.dumps({"frequency": frequency, "selection": chosen}), flush=True)
    hashes[str(output / "frozen_selection.json")] = hashlib.sha256(
        (output / "frozen_selection.json").read_bytes()).hexdigest()
    (output / "frozen_before_target.json").write_text(json.dumps(hashes, indent=2))
    spec = new_tasks(plan, extension, frequency, "evaluation", device)
    directory = output / "evaluation"
    directory.mkdir()
    np.savez_compressed(directory / "teacher_diagnostic.npz", **{
        key: value.cpu().numpy() if isinstance(value, torch.Tensor) else value
        for key, value in spec.items()
    })
    starts, count = plan["starts"], len(spec["families"])
    methods = [f"ar_{i}" for i in range(len(plan["prior_seeds"]))]
    methods += ["retrieval", "frequency", "random", "oracle", "dense_sgd",
                "dense_sgdm", "dense_adam", "long_adam", "source_transfer_adam"]
    for method in methods:
        indices = None
        if method.startswith("ar_"):
            index = int(method.split("_")[-1])
            model = load_prior(plan, source, index, device)
            indices = draw(plan, bank, "ar", count, starts, seed + 1010 + index,
                           model, chosen["temperature"])
        elif method in ("retrieval", "frequency", "random"):
            indices = draw(plan, bank, method, count, starts, seed + 1010)
        elif method == "oracle":
            indices = np.repeat(spec["indices"][:, None], starts, axis=1)
        config = chosen["code" if indices is not None else method]
        long = method.startswith("long_")
        result, history, timing = fit(
            plan, spec, [config], starts // 4 if long else starts, seed + 1020,
            device, indices=indices,
            source_weights=numeric if method == "source_transfer_adam" else None,
            steps=plan["steps"] * (4 if long else 1),
        )
        if indices is not None:
            result["indices"] = indices
        save_fit(directory / method, result, history, timing)
    unchanged = all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
                    for path, digest in hashes.items())
    (directory / "complete.json").write_text(json.dumps({
        "frozen_inputs_unchanged": unchanged,
        "frequency": frequency, "known_functions": 32, "foreign_functions": 8,
        "steps_per_search": plan["steps"] * starts,
        "temperature_fixed_from_frequency_4_calibration": chosen["temperature"],
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", required=True, type=float)
    args = parser.parse_args()
    plan = json.loads((ROOT / "protocol.json").read_text())
    extension = json.loads((ROOT / "frequency_sweep.json").read_text())
    if args.frequency not in extension["frequencies"]:
        parser.error("frequency is outside the predeclared sweep")
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA")
    torch.set_num_threads(4)
    run(plan, extension, args.frequency, torch.device("cuda:0"))


if __name__ == "__main__":
    main()
