"""New-cohort control with the same plain SGD in both representations."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.basin_release_v1.run import (
    REPOSITORY,
    ROOT,
    expanded_initial,
    snapshot_inputs,
)
from experiments.basin_routing_v1.run import canonical_proposals, make_targets
from experiments.basin_structure_transfer_v1.core import fit
from experiments.basin_structure_transfer_v1.run import load_prior, save_fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", required=True, type=int)
    args = parser.parse_args()
    protocol = json.loads((ROOT / "sgd_protocol.json").read_text())
    if args.frequency not in protocol["frequencies"]:
        parser.error("Frequency is outside the declared follow-up")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    output = Path(protocol["output"]) / f"frequency_{args.frequency}"
    output.mkdir(parents=True, exist_ok=False)
    hashes = snapshot_inputs(protocol, output)
    calibration_path = (
        Path(protocol["calibration_run"])
        / f"frequency_{args.frequency}/calibration/grid.json"
    )
    for path in (ROOT / "sgd_control.py", ROOT / "sgd_protocol.json",
                 calibration_path):
        path = path.resolve()
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix == ".py":
            name = str(path.relative_to(REPOSITORY)).replace("/", "__")
            (output / "code_snapshot" / name).write_bytes(path.read_bytes())
    grid = json.loads(calibration_path.read_text())
    paired = []
    for tied in grid["tied"]:
        if tied["optimizer"] != "sgd":
            continue
        released = next(
            row for row in grid["released"]
            if all(row[key] == tied[key]
                   for key in ("optimizer", "lr", "radius"))
        )
        paired.append({
            "optimizer": "sgd",
            "lr": tied["lr"],
            "radius": tied["radius"],
            "calibration_pair_mean_query": (
                tied["calibration_query"] + released["calibration_query"]
            ) / 2,
        })
    config = min(paired, key=lambda row: row["calibration_pair_mean_query"])
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    selection_path = output / "frozen_selection.json"
    selection_path.write_text(json.dumps(config, indent=2))
    hashes[str(selection_path.resolve())] = hashlib.sha256(
        selection_path.read_bytes()
    ).hexdigest()
    (output / "frozen_before_target.json").write_text(
        json.dumps(hashes, indent=2)
    )
    (output / "code_snapshot/manifest.json").write_text(
        json.dumps(hashes, indent=2)
    )
    source = Path(protocol["source_run"])
    base = json.loads((source / "protocol.json").read_text())
    plan = {**base, **protocol}
    spec = make_targets(plan, protocol, args.frequency, device)
    np.savez_compressed(
        output / "teacher_diagnostic.npz",
        **{
            key: value.cpu().numpy() if isinstance(value, torch.Tensor)
            else value for key, value in spec.items()
        },
    )
    count, starts = len(spec["families"]), protocol["starts"]
    known = spec["families"] < base["families"]
    audits, outcomes = {}, {}
    for index in protocol["prior_seeds"]:
        model = load_prior(plan, source, index, device)
        indices, _ = canonical_proposals(
            model, count, starts, protocol["proposal_seed"] + index
        )
        initial_pair = None
        for mode in ("tied", "released"):
            result, history, timing = fit(
                plan, spec, [config], starts, protocol["fit_seed"], device,
                indices=indices,
                release_initial=mode == "released",
                steps=protocol["steps"],
                query_seed=protocol["query_seed"],
            )
            result["indices"] = indices
            if mode == "tied":
                initial_pair = (
                    expanded_initial(result, indices),
                    result["initial_amplitude"].copy(),
                )
            else:
                error = float(np.max(np.abs(
                    initial_pair[0] - result["initial_values"][:, 0]
                )))
                amplitude_equal = np.array_equal(
                    initial_pair[1], result["initial_amplitude"]
                )
                if error > protocol["initial_weight_atol"]:
                    raise RuntimeError("Initial weights differ")
                if not amplitude_equal:
                    raise RuntimeError("Initial amplitudes differ")
                audits[str(index)] = {
                    "maximum_initial_weight_difference": error,
                    "initial_amplitudes_identical": amplitude_equal,
                }
            timing["model_updates_per_task"] = starts * protocol["steps"]
            name = f"{mode}_ar_{index}"
            directory = output / name
            save_fit(directory, result, history, timing)
            choice = result["query"][:, 0].argmin(1)
            rows = np.arange(count)
            saved = {
                key: result[key][rows, 0, choice]
                for key in ("query", "population", "weights", "amplitude")
            }
            saved["indices"] = indices[rows, choice]
            saved["candidate"] = choice
            np.savez_compressed(directory / "selected.npz", **saved)
            success = saved["population"] < protocol["success_mse"]
            outcomes[name] = {
                "known_solved": int(success[known].sum()),
                "foreign_solved": int(success[~known].sum()),
            }
            print(json.dumps({"method": name, **outcomes[name]}), flush=True)
    unchanged = all(
        hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
        for path, digest in hashes.items()
    )
    if not unchanged:
        raise RuntimeError("Frozen input changed")
    (output / "complete.json").write_text(json.dumps({
        "frozen_inputs_unchanged": unchanged,
        "frequency": args.frequency,
        "known_tasks": int(known.sum()),
        "foreign_tasks": int((~known).sum()),
        "config": config,
        "matching": audits,
        "outcomes": outcomes,
    }, indent=2))


if __name__ == "__main__":
    main()
