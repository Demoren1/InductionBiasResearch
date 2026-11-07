"""Compare retained weight sharing with released, matched initial weights."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.basin_routing_v1.run import (
    canonical_proposals,
    make_targets,
)
from experiments.basin_structure_transfer_v1.core import fit
from experiments.basin_structure_transfer_v1.run import load_prior, save_fit

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]


def snapshot_inputs(protocol, output):
    """Preserve exactly the source models and executable inputs used here."""
    source = Path(protocol["source_run"])
    paths = [
        ROOT / "protocol.json",
        ROOT / "run.py",
        ROOT.parent / "basin_structure_transfer_v1/core.py",
        ROOT.parent / "basin_structure_transfer_v1/run.py",
        ROOT.parent / "basin_routing_v1/run.py",
        ROOT.parent / "adaptive_tanh_v1/source.py",
        REPOSITORY / "pattern/autoregressive/model.py",
        source / "protocol.json",
        source / "structure_bank.npz",
    ]
    paths += [source / f"prior_{i}.pt" for i in protocol["prior_seeds"]]
    paths = [path.resolve() for path in paths]
    hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    snapshot = output / "code_snapshot"
    snapshot.mkdir()
    for path in paths:
        if path.suffix == ".py":
            name = str(path.relative_to(REPOSITORY)).replace("/", "__")
            (snapshot / name).write_bytes(path.read_bytes())
    (snapshot / "manifest.json").write_text(json.dumps(hashes, indent=2))
    return hashes


def calibrate(plan, protocol, output, frequency, bank, device):
    """Choose all numeric settings before generating evaluation tasks."""
    calibration_protocol = {
        **protocol,
        "known_tasks_per_family": protocol["calibration_tasks_per_family"],
        "foreign_tasks": 0,
        "target_seed": protocol["calibration_task_seed"],
    }
    spec = make_targets(plan, calibration_protocol, frequency, device)
    count = len(spec["families"])
    starts = protocol["calibration_starts"]
    rng = np.random.default_rng(protocol["calibration_proposal_seed"])
    bank_repeats = np.tile(np.arange(len(bank)), starts // len(bank))
    proposals = np.stack(
        [bank[rng.permutation(bank_repeats)] for _ in range(count)]
    )
    directory = output / "calibration"
    directory.mkdir()
    grid = {mode: [] for mode in ("tied", "released", "dense")}
    for mode in grid:
        for optimizer in protocol["optimizers"]:
            configs = [
                {"optimizer": optimizer, "lr": rate, "radius": radius}
                for radius in protocol["radii"]
                for rate in protocol["rates"]
            ]
            result, history, timing = fit(
                plan,
                spec,
                configs,
                starts,
                protocol["calibration_fit_seed"],
                device,
                indices=None if mode == "dense" else proposals,
                release_initial=mode == "released",
                steps=protocol["steps"],
                query_seed=protocol["calibration_query_seed"],
            )
            scores = result["query"].min(2).mean(0)
            grid[mode].extend(
                {**config, "calibration_query": float(score)}
                for config, score in zip(configs, scores)
            )
            save_fit(
                directory / f"{mode}_{optimizer}", result, history, timing
            )

    def criterion(item):
        return item["calibration_query"]

    tied = min(grid["tied"], key=criterion)
    selected = {
        "tied": tied,
        "released_matched": min(
            [
                item for item in grid["released"]
                if item["radius"] == tied["radius"]
            ],
            key=criterion,
        ),
        "released_tuned": min(grid["released"], key=criterion),
    }
    for optimizer in protocol["optimizers"]:
        selected[f"dense_{optimizer}"] = min(
            [
                item for item in grid["dense"]
                if item["optimizer"] == optimizer
            ],
            key=criterion,
        )
    selected["dense_preselected"] = min(
        [f"dense_{optimizer}" for optimizer in protocol["optimizers"]],
        key=lambda name: selected[name]["calibration_query"],
    )
    (directory / "grid.json").write_text(json.dumps(grid, indent=2))
    (output / "frozen_selection.json").write_text(
        json.dumps(selected, indent=2)
    )
    print(
        json.dumps({"frequency": frequency, "selection": selected}),
        flush=True,
    )
    return selected


def expanded_initial(result, indices):
    """Decode saved orthonormal group coordinates into effective weights."""
    codes = result["initial_values"][:, 0]
    counts = (indices[:, :, :, None] == np.arange(codes.shape[-1])).sum(2)
    divisor = np.sqrt(np.take_along_axis(counts, indices, axis=2))
    return np.take_along_axis(codes, indices, axis=2) / divisor


def run(protocol, frequency, device):
    source = Path(protocol["source_run"])
    base = json.loads((source / "protocol.json").read_text())
    plan = {
        **base,
        **{
            key: protocol[key]
            for key in ("steps", "batch", "query", "log_every")
        },
    }
    output = Path(protocol["output"]) / f"frequency_{frequency}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    hashes = snapshot_inputs(protocol, output)
    bank = np.unique(np.load(source / "structure_bank.npz")["train"], axis=0)
    if protocol["calibration_starts"] % len(bank):
        raise ValueError("Balanced calibration requires complete bank repeats")
    selected = calibrate(plan, protocol, output, frequency, bank, device)
    selection_path = output / "frozen_selection.json"
    hashes[str(selection_path.resolve())] = hashlib.sha256(
        selection_path.read_bytes()
    ).hexdigest()
    (output / "frozen_before_target.json").write_text(
        json.dumps(hashes, indent=2)
    )
    spec = make_targets(plan, protocol, frequency, device)
    np.savez_compressed(
        output / "teacher_diagnostic.npz",
        **{
            key: value.cpu().numpy() if isinstance(value, torch.Tensor)
            else value for key, value in spec.items()
        },
    )
    count, starts = len(spec["families"]), protocol["starts"]
    proposals, proposal_times = {}, {}
    for index in protocol["prior_seeds"]:
        model = load_prior(plan, source, index, device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        indices, generated = canonical_proposals(
            model, count, starts, protocol["proposal_seed"] + index
        )
        torch.cuda.synchronize(device)
        proposals[f"ar_{index}"] = indices
        proposal_times[f"ar_{index}"] = {
            "seconds": time.perf_counter() - started,
            "raw_samples": generated,
        }
    proposals["oracle"] = np.repeat(spec["indices"][:, None], starts, axis=1)
    np.savez_compressed(output / "proposals.npz", **proposals)
    (output / "proposal_timing.json").write_text(
        json.dumps(proposal_times, indent=2)
    )
    jobs = [
        (f"{mode}_ar_{index}", mode, f"ar_{index}")
        for index in protocol["prior_seeds"]
        for mode in ("tied", "released_matched", "released_tuned")
    ]
    jobs += [
        (f"{mode}_oracle", mode, "oracle")
        for mode in ("tied", "released_matched")
    ]
    jobs += [
        (f"dense_{opt}", f"dense_{opt}", None)
        for opt in protocol["optimizers"]
    ]
    initial_pairs, audits = {}, {}
    for name, mode, proposal_name in jobs:
        indices = None if proposal_name is None else proposals[proposal_name]
        result, history, timing = fit(
            plan,
            spec,
            [selected[mode]],
            starts,
            protocol["fit_seed"],
            device,
            indices=indices,
            release_initial=mode.startswith("released"),
            steps=protocol["steps"],
            query_seed=protocol["query_seed"],
        )
        if indices is not None:
            result["indices"] = indices
        if mode == "tied":
            initial_pairs[proposal_name] = (
                expanded_initial(result, indices),
                result["initial_amplitude"].copy(),
            )
        elif mode == "released_matched":
            weights, amplitude = initial_pairs[proposal_name]
            error = float(np.max(np.abs(
                weights - result["initial_values"][:, 0]
            )))
            amplitude_equal = np.array_equal(
                amplitude, result["initial_amplitude"]
            )
            audits[proposal_name] = {
                "maximum_initial_weight_difference": error,
                "initial_amplitudes_identical": amplitude_equal,
                "radius_identical": (
                    selected["tied"]["radius"] == selected[mode]["radius"]
                ),
                "fit_seed": protocol["fit_seed"],
                "training_seed": timing["training_seed"],
                "query_seed": timing["query_seed"],
                "proposals_identical": True,
            }
            if error > protocol["initial_weight_atol"] or not amplitude_equal:
                raise RuntimeError("Initial parameters do not match")
        timing["model_updates_per_task"] = timing["steps"] * timing["starts"]
        timing["training_model_examples_per_task"] = (
            timing["model_updates_per_task"] * plan["batch"]
        )
        timing["query_candidate_checkpoints"] = (
            (plan["steps"] // plan["log_every"] + 1) * starts
        )
        directory = output / name
        save_fit(directory, result, history, timing)
        choice = result["query"][:, 0].argmin(1)
        rows = np.arange(count)
        saved = {
            key: result[key][rows, 0, choice]
            for key in ("query", "population", "weights", "amplitude", "step")
        }
        saved["candidate"] = choice
        if indices is not None:
            saved["indices"] = indices[rows, choice]
        np.savez_compressed(directory / "selected.npz", **saved)
        known = spec["families"] < plan["families"]
        success = saved["population"] < protocol["success_mse"]
        print(
            json.dumps({
                "method": name,
                "frequency": frequency,
                "known_solved": int(success[known].sum()),
                "foreign_solved": int(success[~known].sum()),
            }),
            flush=True,
        )
    unchanged = all(
        hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
        for path, digest in hashes.items()
    )
    if not unchanged:
        raise RuntimeError("Frozen inputs changed while the experiment ran")
    (output / "complete.json").write_text(json.dumps({
        "frozen_inputs_unchanged": unchanged,
        "frequency": frequency,
        "known_tasks": int(known.sum()),
        "foreign_tasks": int((~known).sum()),
        "methods": [job[0] for job in jobs],
        "model_updates_per_method_task": starts * protocol["steps"],
        "matching": audits,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", type=int, required=True)
    args = parser.parse_args()
    protocol = json.loads((ROOT / "protocol.json").read_text())
    if args.frequency not in protocol["frequencies"]:
        parser.error("Frequency is outside the declared protocol")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training")
    torch.set_num_threads(4)
    run(protocol, args.frequency, torch.device("cuda:0"))


if __name__ == "__main__":
    main()
