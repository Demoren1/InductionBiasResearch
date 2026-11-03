"""Test a structural task feature and remove its signal as a control."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.basin_routing_v1.run import (
    ROOT,
    canonical_proposals,
    digest_inputs,
    make_targets,
    score_query,
    store_selected,
)
from experiments.basin_structure_transfer_v1.core import (
    canonicalize,
    fit,
    target,
)
from experiments.basin_structure_transfer_v1.run import load_prior, save_fit


@torch.no_grad()
def empirical_moment(spec, plan, seed, device):
    rng = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(
        len(spec["families"]),
        plan["query"],
        plan["inputs"],
        generator=rng,
        device=device,
    )
    y = target(x, spec)
    return (x.double() * y.double()[:, :, None]).mean(1).cpu().numpy()


def projection_scores(feature, indices, groups):
    projected = np.zeros(indices.shape, dtype=float)
    for group in range(groups):
        mask = indices == group
        count = mask.sum(2).clip(1)
        mean = (mask * feature[:, None]).sum(2) / count
        projected += mask * mean[:, :, None]
    return ((feature[:, None] - projected) ** 2).sum(2)


def two_best_structures(indices, query):
    canonical = canonicalize(indices)
    result = []
    for task in range(len(indices)):
        distinct = []
        for candidate in query[task].argsort():
            item = canonical[task, candidate]
            if not any(np.array_equal(item, old) for old in distinct):
                distinct.append(item)
            if len(distinct) == 2:
                break
        if len(distinct) == 1:
            distinct.append(distinct[0].copy())
        result.append(distinct)
    return np.array(result)


def run(protocol, frequency, cue, device):
    source = Path(protocol["source_run"])
    base = json.loads((source / "protocol.json").read_text())
    plan = {**base, "batch": protocol["batch"], "query": protocol["query"]}
    settings = Path(protocol["frequency_settings"]) / f"frequency_{frequency}"
    chosen = json.loads((settings / "frozen_selection.json").read_text())
    output = Path(protocol["output"]) / f"frequency_{frequency}_{cue}"
    output.mkdir(parents=True, exist_ok=False)
    hashes = digest_inputs(protocol, frequency)
    for name in ("cue_run.py", "cue_protocol.json"):
        path = ROOT / name
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
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
    if cue == "absent":
        spec["alpha"] = torch.zeros_like(spec["alpha"])
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
    models, proposals = {}, {}
    for index in protocol["prior_seeds"]:
        models[index] = load_prior(base, source, index, device)
        proposals[index], _ = canonical_proposals(
            models[index], count, starts, protocol["proposal_seed"] + index
        )
    bank = np.unique(np.load(source / "structure_bank.npz")["train"], axis=0)
    bank_proposals = np.repeat(bank[None], count, axis=0)
    numeric = torch.tensor(
        np.load(source / "numeric_baseline_bank.npz")["weights"], device=device
    )
    feature = empirical_moment(spec, plan, protocol["gate_query_seed"], device)
    feature_scores = {
        index: projection_scores(feature, proposals[index], 3)
        for index in protocol["prior_seeds"]
    }
    bank_scores = projection_scores(feature, bank_proposals, 3)
    np.savez_compressed(
        output / "feature_diagnostic.npz",
        empirical_moment=feature,
        bank_scores=bank_scores,
    )
    screening = {}

    def screen(index):
        if index not in screening:
            result, history, timing = fit(
                plan,
                spec,
                [chosen["code"]],
                protocol["screen_starts"],
                protocol["initial_fit_seed"],
                device,
                indices=proposals[index],
                steps=protocol["screen_steps"],
                query_seed=protocol["gate_query_seed"],
            )
            result["indices"] = proposals[index]
            save_fit(
                output / "screening" / f"ar_{index}", result, history, timing
            )
            query = score_query(
                result["weights"][:, 0],
                result["amplitude"][:, 0],
                spec,
                plan,
                protocol["final_query_seed"],
                device,
            )
            top_two = two_best_structures(
                proposals[index], result["query"][:, 0]
            )
            screening[index] = result, timing, query.cpu().numpy(), top_two
        return screening[index]

    for method in protocol["methods"]:
        two_phase = method.startswith(("routed_", "top_two_", "hedged_"))
        index = int(method[-1]) if method[-1].isdigit() else 0
        indices, route = None, None
        if method.startswith("canonical_ar_"):
            indices = proposals[index]
        elif method.startswith("moment_ar_"):
            choice = feature_scores[index].argmin(1)
            route = proposals[index][np.arange(count), choice]
            indices = np.repeat(route[:, None], starts, axis=1)
        elif method == "moment_retrieval":
            route = bank[bank_scores.argmin(1)]
            indices = np.repeat(route[:, None], starts, axis=1)
        elif method == "random_single_ar_0":
            route = proposals[0][:, 0]
            indices = np.repeat(route[:, None], starts, axis=1)
        elif method == "oracle":
            route = spec["indices"]
            indices = np.repeat(route[:, None], starts, axis=1)
        elif two_phase:
            previous, screen_timing, previous_query, best = screen(index)
            route = best[:, 0]
            if method.startswith("top_two_"):
                indices = np.repeat(best, 14, axis=1)
            elif method.startswith("hedged_"):
                fresh, _ = canonical_proposals(
                    models[index],
                    count,
                    14,
                    protocol["proposal_seed"] + 1000 + index,
                )
                indices = np.concatenate(
                    [np.repeat(route[:, None], 14, axis=1), fresh], axis=1
                )
            else:
                indices = np.repeat(
                    route[:, None], protocol["fresh_starts"], axis=1
                )
        fit_starts = protocol["fresh_starts"] if two_phase else starts
        fit_seed = (
            protocol["fresh_fit_seed"]
            if two_phase
            else protocol["initial_fit_seed"]
        )
        config = chosen["code" if indices is not None else method]
        result, history, timing = fit(
            plan,
            spec,
            [config],
            fit_starts,
            fit_seed,
            device,
            indices=indices,
            source_weights=numeric
            if method == "source_transfer_adam"
            else None,
            steps=protocol["fresh_steps"],
            query_seed=protocol["final_query_seed"],
        )
        if indices is not None:
            result["indices"] = indices
        timing["model_updates_per_task"] = 65536
        timing["uses_task_moment"] = method.startswith("moment_")
        timing["moment_is_used_for_initialization"] = False
        if two_phase:
            timing["screening_seconds"] = screen_timing["seconds"]
            timing["screening_method"] = f"ar_{index}"
        save_fit(output / method, result, history, timing)
        extra = {}
        if route is not None:
            extra["route"] = route
            extra["route_exact"] = (
                canonicalize(route) == spec["indices"]
            ).all(1)
        if two_phase:
            extra["phase_two_exact_attempts"] = (
                (canonicalize(indices) == spec["indices"][:, None])
                .all(2)
                .sum(1)
            )
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
            result = combined
        store_selected(output / method, result, spec, plan, device, extra)
    unchanged = all(
        hashlib.sha256(Path(path).read_bytes()).hexdigest() == value
        for path, value in hashes.items()
    )
    (output / "complete.json").write_text(
        json.dumps(
            {
                "frozen_inputs_unchanged": unchanged,
                "methods": protocol["methods"],
                "frequency": frequency,
                "cue": cue,
                "known_tasks": int((spec["families"] < 4).sum()),
                "foreign_tasks": int((spec["families"] == 4).sum()),
            },
            indent=2,
        )
    )
    if not unchanged:
        raise RuntimeError("Frozen inputs changed during evaluation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", type=int, choices=(6, 8), required=True)
    parser.add_argument("--cue", choices=("present", "absent"), required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA")
    torch.set_num_threads(2)
    parent = json.loads((ROOT / "protocol.json").read_text())
    followup = json.loads((ROOT / "cue_protocol.json").read_text())
    run(
        {**parent, **followup},
        args.frequency,
        args.cue,
        torch.device("cuda:0"),
    )


if __name__ == "__main__":
    main()
