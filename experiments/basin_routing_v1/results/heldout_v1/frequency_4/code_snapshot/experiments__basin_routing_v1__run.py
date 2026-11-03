"""Prospective equal-update evaluation of task-guided indexator routing."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.basin_structure_transfer_v1.core import (
    canonicalize,
    fit,
    make_tasks,
    target,
)
from experiments.basin_structure_transfer_v1.run import (
    draw,
    load_prior,
    save_fit,
)

ROOT = Path(__file__).resolve().parent


def make_targets(base, protocol, frequency, device):
    plan = {
        **base,
        "evaluation_tasks_per_family": protocol["known_tasks_per_family"],
        "foreign_tasks": protocol["foreign_tasks"],
    }
    spec = make_tasks(plan, "evaluation", device)
    rng = np.random.default_rng(protocol["target_seed"])
    count = len(spec["families"])
    codes = rng.normal(size=(count, plan["groups"]))
    codes /= np.linalg.norm(codes, axis=1, keepdims=True)
    direction = np.take_along_axis(codes, spec["indices"], axis=1)
    direction /= np.sqrt(plan["inputs"] / plan["groups"])
    arrays = {
        "direction": direction,
        "high": frequency + rng.uniform(-0.2, 0.2, count),
        "low": rng.uniform(*plan["low_frequency"], count),
        "alpha": rng.uniform(*plan["low_amplitude"], count),
    }
    spec.update(
        {
            key: torch.tensor(value, device=device).float()
            for key, value in arrays.items()
        }
    )
    return spec


@torch.no_grad()
def canonical_proposals(model, tasks, starts, seed):
    """Rejection sampling exactly conditions the first raw token on zero."""
    torch.manual_seed(seed)
    needed = tasks * starts
    accepted = []
    generated = 0
    while needed:
        batch = max(32, int(np.ceil(needed * 1.08)))
        samples = model.sample_cached(batch, temperature=1)
        generated += batch
        valid = samples[samples[:, 0] == 0][:needed]
        accepted.append(valid)
        needed -= len(valid)
    result = torch.cat(accepted).cpu().numpy().reshape(tasks, starts, -1)
    return result, generated


def digest_inputs(protocol, frequency):
    source = Path(protocol["source_run"])
    settings = (
        source
        if frequency == 4
        else (Path(protocol["frequency_settings"]) / f"frequency_{frequency}")
    )
    paths = [
        ROOT / "run.py",
        ROOT / "protocol.json",
        ROOT.parent / "basin_structure_transfer_v1/core.py",
        ROOT.parent / "basin_structure_transfer_v1/run.py",
        ROOT.parents[1] / "pattern/autoregressive/model.py",
        source / "protocol.json",
        source / "structure_bank.npz",
        source / "numeric_baseline_bank.npz",
        settings / "frozen_selection.json",
    ]
    paths += [source / f"prior_{i}.pt" for i in protocol["prior_seeds"]]
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


@torch.no_grad()
def score_query(weights, amplitude, spec, plan, seed, device):
    rng = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(
        len(weights),
        plan["query"],
        plan["inputs"],
        generator=rng,
        device=device,
    )
    weights = torch.as_tensor(weights, device=device)
    amplitude = torch.as_tensor(amplitude, device=device)
    prediction = torch.sin(torch.bmm(x, weights.transpose(1, 2)))
    prediction *= amplitude[:, None]
    return ((prediction - target(x, spec)[:, :, None]) ** 2).mean(1)


def store_selected(directory, result, spec, plan, device, extra=None):
    query = result["query"][:, 0]
    choice = query.argmin(1)
    tasks = np.arange(len(choice))
    selected = {
        "query": query[tasks, choice],
        "population": result["population"][:, 0][tasks, choice],
        "weights": result["weights"][:, 0][tasks, choice],
        "amplitude": result["amplitude"][:, 0][tasks, choice],
        "candidate": choice,
    }
    if "indices" in result:
        selected["indices"] = result["indices"][tasks, choice]
    if extra:
        selected.update(extra)
    np.savez_compressed(directory / "selected.npz", **selected)
    known = spec["families"] < plan["families"]
    print(
        json.dumps(
            {
                "method": directory.name,
                "known_solved": int(
                    (selected["population"][known] < 0.1).sum()
                ),
                "known_total": int(known.sum()),
                "foreign_solved": int(
                    (selected["population"][~known] < 0.1).sum()
                ),
            }
        ),
        flush=True,
    )


def run(protocol, frequency, device):
    source = Path(protocol["source_run"])
    base = json.loads((source / "protocol.json").read_text())
    settings = (
        source
        if frequency == 4
        else (Path(protocol["frequency_settings"]) / f"frequency_{frequency}")
    )
    chosen = json.loads((settings / "frozen_selection.json").read_text())
    plan = {**base, "batch": protocol["batch"], "query": protocol["query"]}
    output = Path(protocol["output"]) / f"frequency_{frequency}"
    output.mkdir(parents=True, exist_ok=False)
    hashes = digest_inputs(protocol, frequency)
    (output / "frozen_before_target.json").write_text(
        json.dumps(hashes, indent=2)
    )
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (output / "numeric_settings.json").write_text(json.dumps(chosen, indent=2))
    snapshot = output / "code_snapshot"
    snapshot.mkdir()
    for path in hashes:
        if path.endswith(".py"):
            name = str(Path(path).relative_to(ROOT.parents[1])).replace(
                "/", "__"
            )
            (snapshot / name).write_bytes(Path(path).read_bytes())
    (snapshot / "manifest.json").write_text(json.dumps(hashes, indent=2))
    spec = make_targets(base, protocol, frequency, device)
    np.savez_compressed(
        output / "teacher_diagnostic.npz",
        **{
            key: value.cpu().numpy()
            if isinstance(value, torch.Tensor)
            else value
            for key, value in spec.items()
        },
    )
    count = len(spec["families"])
    starts = protocol["static_starts"]
    bank = torch.tensor(
        np.load(source / "structure_bank.npz")["train"], device=device
    )
    unique_bank = np.unique(bank.cpu().numpy(), axis=0)
    numeric = torch.tensor(
        np.load(source / "numeric_baseline_bank.npz")["weights"], device=device
    )
    proposals, proposal_times, models = {}, {}, {}
    for index in protocol["prior_seeds"]:
        models[index] = load_prior(base, source, index, device)
        for mode in ("raw", "canonical"):
            name = f"{mode}_ar_{index}"
            torch.cuda.synchronize(device)
            begin = time.perf_counter()
            seed = protocol["proposal_seed"] + index
            if mode == "raw":
                indices = draw(
                    base, bank, "ar", count, starts, seed, models[index], 1
                )
                generated = count * starts
            else:
                indices, generated = canonical_proposals(
                    models[index], count, starts, seed
                )
            torch.cuda.synchronize(device)
            proposals[name] = indices
            proposal_times[name] = {
                "seconds": time.perf_counter() - begin,
                "raw_samples": generated,
            }
    proposals["retrieval"] = draw(
        base, bank, "retrieval", count, starts, protocol["proposal_seed"]
    )
    rng = np.random.default_rng(protocol["proposal_seed"])
    proposals["balanced_retrieval"] = np.stack(
        [
            unique_bank[rng.permutation(np.tile(np.arange(4), starts // 4))]
            for _ in range(count)
        ]
    )
    proposals["oracle"] = np.repeat(spec["indices"][:, None], starts, axis=1)
    screening = {}

    def screen(name):
        if name not in screening:
            result, history, timing = fit(
                plan,
                spec,
                [chosen["code"]],
                protocol["screen_starts"],
                protocol["initial_fit_seed"],
                device,
                indices=proposals[name],
                steps=protocol["screen_steps"],
                query_seed=protocol["gate_query_seed"],
            )
            result["indices"] = proposals[name]
            save_fit(output / "screening" / name, result, history, timing)
            selected = result["query"][:, 0].argmin(1)
            routed = proposals[name][np.arange(count), selected]
            final_query = (
                score_query(
                    result["weights"][:, 0],
                    result["amplitude"][:, 0],
                    spec,
                    plan,
                    protocol["final_query_seed"],
                    device,
                )
                .cpu()
                .numpy()
            )
            screening[name] = result, timing, routed, final_query
        return screening[name]

    for method in protocol["methods"]:
        routing = method.startswith(
            ("routed_", "resampled_", "random_route_", "oracle_route_")
        )
        if not routing:
            indices = proposals.get(method)
            long = method == "long_adam"
            result, history, timing = fit(
                plan,
                spec,
                [chosen["code" if indices is not None else method]],
                starts // 4 if long else starts,
                protocol["initial_fit_seed"],
                device,
                indices=indices,
                source_weights=numeric
                if method == "source_transfer_adam"
                else None,
                steps=protocol["static_steps"] * (4 if long else 1),
                query_seed=protocol["final_query_seed"],
            )
            if indices is not None:
                result["indices"] = indices
            timing["model_updates_per_task"] = 65536
            timing["query_sets"] = 1
            timing["query_candidate_checkpoints"] = timing["starts"] * (
                timing["steps"] // plan["log_every"] + 1
            )
            save_fit(output / method, result, history, timing)
            store_selected(output / method, result, spec, plan, device)
            continue

        index = 0 if method == "routed_retrieval" else int(method[-1])
        parent = (
            "retrieval"
            if method == "routed_retrieval"
            else f"canonical_ar_{index}"
        )
        previous, screen_timing, route, previous_query = screen(parent)
        route = route.copy()
        if method.startswith("resampled_"):
            indices, _ = canonical_proposals(
                models[index],
                count,
                protocol["fresh_starts"],
                protocol["proposal_seed"] + 1000 + index,
            )
        else:
            if method.startswith("random_route_"):
                random_choice = np.random.default_rng(
                    protocol["proposal_seed"] + 2000
                ).integers(starts, size=count)
                route = proposals[parent][np.arange(count), random_choice]
            elif method.startswith("oracle_route_"):
                route = spec["indices"].copy()
            indices = np.repeat(
                route[:, None], protocol["fresh_starts"], axis=1
            )
        result, history, timing = fit(
            plan,
            spec,
            [chosen["code"]],
            protocol["fresh_starts"],
            protocol["fresh_fit_seed"],
            device,
            indices=indices,
            steps=protocol["fresh_steps"],
            query_seed=protocol["final_query_seed"],
        )
        result["indices"] = indices
        timing.update(
            {
                "screening_method": parent,
                "screening_seconds": screen_timing["seconds"],
                "model_updates_per_task": (
                    protocol["screen_starts"] * protocol["screen_steps"]
                    + protocol["fresh_starts"] * protocol["fresh_steps"]
                ),
                "query_sets": 2,
                "query_candidate_checkpoints": (
                    protocol["screen_starts"] * 3
                    + protocol["fresh_starts"]
                    * (protocol["fresh_steps"] // plan["log_every"] + 1)
                ),
                "fresh_numerical_second_phase": True,
            }
        )
        save_fit(output / method, result, history, timing)
        combined = {
            key: np.concatenate([previous[key], result[key]], axis=2)
            for key in ("weights", "amplitude", "population")
        }
        combined["query"] = np.concatenate(
            [previous_query[:, None], result["query"]], axis=2
        )
        combined["indices"] = np.concatenate(
            [previous["indices"], result["indices"]], axis=1
        )
        extra = {
            "phase_two_solved": (result["population"][:, 0] < 0.1).any(1),
        }
        if not method.startswith("resampled_"):
            extra["route"] = route
            extra["route_exact"] = np.all(
                canonicalize(route) == spec["indices"], axis=1
            )
        store_selected(output / method, combined, spec, plan, device, extra)
    unchanged = hashes == digest_inputs(protocol, frequency)
    (output / "complete.json").write_text(
        json.dumps(
            {
                "frozen_inputs_unchanged": unchanged,
                "methods": protocol["methods"],
                "known_tasks": int((spec["families"] < 4).sum()),
                "foreign_tasks": int((spec["families"] == 4).sum()),
                "proposal_generation": proposal_times,
            },
            indent=2,
        )
    )
    if not unchanged:
        raise RuntimeError("Frozen inputs changed during the experiment")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frequency", type=int, choices=(4, 6, 8), required=True
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA")
    torch.set_num_threads(2)
    protocol = json.loads((ROOT / "protocol.json").read_text())
    run(protocol, args.frequency, torch.device("cuda:0"))


if __name__ == "__main__":
    main()
