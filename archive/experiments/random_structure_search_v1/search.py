"""Random structural search; the outer optimizer sees predictive scores only."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.mlp_structure_discovery_v1.fit import (
    fit_models,
    read_observations,
    untied_indices,
)


def canonicalize(tokens):
    values = list(dict.fromkeys(tokens.ravel().tolist()))
    values = [value for value in values if value]
    mapping = np.zeros(int(tokens.max()) + 1, dtype=np.int16)
    mapping[values] = np.arange(1, len(values) + 1)
    return mapping[tokens]


def decode(tokens, kind):
    return untied_indices(tokens > 0) if kind == "mask" else tokens.astype(np.int64)


def random_structure(shape, kind, plan, rng, fraction=None, codes=None):
    if fraction is None:
        fraction = float(rng.choice(plan["initial_fractions"]))
    retained = max(1, min(np.prod(shape), round(np.prod(shape) * fraction)))
    tokens = np.zeros(np.prod(shape), dtype=np.int16)
    active = rng.choice(len(tokens), retained, replace=False)
    if kind == "mask":
        tokens[active] = 1
    else:
        if codes is None:
            codes = int(rng.choice(plan["initial_code_counts"]))
        tokens[active] = rng.integers(1, codes + 1, size=retained)
    return canonicalize(tokens.reshape(shape))


def mutate(parent, other, kind, plan, rng):
    tokens = parent.copy()
    if rng.random() < 0.15:
        columns = rng.random(tokens.shape[1]) < 0.5
        tokens[:, columns] = other[:, columns]
    values = tokens.ravel()
    active = np.flatnonzero(values)
    inactive = np.flatnonzero(values == 0)
    amount = int(rng.choice(plan["mutation_sizes"]))
    if rng.random() < 0.15:
        if rng.random() < 0.5:
            removed = rng.choice(
                active, min(amount, max(0, len(active) - 1)), replace=False
            )
            values[removed] = 0
        else:
            added = rng.choice(inactive, min(amount, len(inactive)), replace=False)
            values[added] = (
                1
                if kind == "mask"
                else rng.integers(1, max(2, int(values.max()) + 1), len(added))
            )
    elif kind == "mask" or rng.random() < 0.35:
        changed = min(amount, len(active), len(inactive))
        removed = rng.choice(active, changed, replace=False)
        added = rng.choice(inactive, changed, replace=False)
        values[added] = values[removed]
        values[removed] = 0
    else:
        codes = np.unique(values[values > 0])
        operation = rng.integers(3)
        if operation == 0 and len(codes) >= 2:
            kept, removed = rng.choice(codes, 2, replace=False)
            values[values == removed] = kept
        elif operation == 1 and len(codes) < plan["maximum_codes"]:
            selected = np.flatnonzero(values == rng.choice(codes))
            chosen = selected[rng.random(len(selected)) < 0.5]
            values[chosen] = int(values.max()) + 1
        else:
            selected = rng.choice(active, min(amount, len(active)), replace=False)
            values[selected] = rng.choice(codes, len(selected))
    if not np.any(values):
        values[rng.integers(len(values))] = 1
    return canonicalize(tokens)


def fingerprint(kind, tokens):
    digest = hashlib.sha256(tokens.astype(np.int16).tobytes()).hexdigest()
    return kind + ":" + digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", default="experiments/random_structure_search_v1/protocol.json"
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    path = Path(args.protocol)
    plan = json.loads(path.read_text())
    root = Path(plan["output"])
    output = root / "search"
    output.mkdir(parents=True, exist_ok=False)
    (root / "protocol.json").write_text(path.read_text())
    paths = [Path(__file__), Path("experiments/mlp_structure_discovery_v1/fit.py")]
    (root / "started.json").write_text(
        json.dumps(
            {
                "started_unix": time.time(),
                "source_sha256": hashlib.sha256(
                    Path(plan["source_data"]).read_bytes()
                ).hexdigest(),
                "source_code_sha256": {
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
                },
                "torch": torch.__version__,
            },
            indent=2,
        )
        + "\n"
    )
    torch.set_num_threads(2)
    device = torch.device(args.device)
    data = read_observations(
        plan["source_data"], plan["discovery_tasks"], plan["transfer_support"]
    )
    shape = (data["support_x"].shape[-1], plan["hidden"])
    tasks = len(data["support_y"])
    rng = np.random.default_rng(plan["seed"])
    populations = {kind: [] for kind in ["mask", "coded"]}
    seen, archive, records, history = {}, [], [], []
    for generation in range(plan["generations"] + 1):
        for kind, population in populations.items():
            if generation:
                ranked = sorted(
                    population,
                    key=lambda item: records[seen[fingerprint(kind, item)]]["fitness"],
                )
                updated = [value.copy() for value in ranked[: plan["elites_per_kind"]]]
                used = {fingerprint(kind, value) for value in updated}
                while (
                    len(updated)
                    < plan["population_per_kind"] - plan["immigrants_per_kind"]
                ):
                    choices = rng.choice(len(ranked), 3, replace=False)
                    parent = ranked[int(choices.min())]
                    other = ranked[rng.integers(max(2, len(ranked) // 2))]
                    proposal = mutate(parent, other, kind, plan, rng)
                    key = fingerprint(kind, proposal)
                    if key not in used:
                        used.add(key)
                        updated.append(proposal)
                populations[kind] = updated
                population = updated
            used = {fingerprint(kind, value) for value in population}
            while len(population) < plan["population_per_kind"]:
                proposal = random_structure(shape, kind, plan, rng)
                key = fingerprint(kind, proposal)
                if key not in used:
                    used.add(key)
                    population.append(proposal)
        new = [
            (kind, value)
            for kind, population in populations.items()
            for value in population
            if fingerprint(kind, value) not in seen
        ]
        # Candidate construction has no access to fit_models' parameters. This
        # call returns only scalar predictive metrics when save_parameters=False.
        rows = [
            {
                "candidate": candidate,
                "task": task,
                "restart": 0,
                "lr": plan["search_learning_rate"],
                "weight_decay": plan["search_weight_decay"],
            }
            for candidate in range(len(new))
            for task in range(tasks)
        ]
        indices = np.stack([decode(value, kind) for kind, value in new])
        arrays = fit_models(
            data,
            np.repeat(indices, tasks, axis=0),
            rows,
            plan,
            output / f"generation_{generation:03d}",
            device,
            steps=plan["search_steps"],
            save_parameters=False,
        )
        losses = arrays["query_loss"].reshape(len(new), tasks)
        accuracies = arrays["query_accuracy"].reshape(len(new), tasks)
        for candidate, (kind, value) in enumerate(new):
            key = fingerprint(kind, value)
            seen[key] = len(archive)
            archive.append(value)
            records.append(
                {
                    "kind": kind,
                    "generation": generation,
                    "fingerprint": key,
                    "fitness": float(losses[candidate].mean()),
                    "accuracy": float(accuracies[candidate].mean()),
                    "task_losses": losses[candidate].tolist(),
                    "nonzeros": int(np.count_nonzero(value)),
                    "codes": int(value.max()),
                }
            )
        selected = {
            kind: sorted(
                [seen[fingerprint(kind, value)] for value in population],
                key=lambda i: records[i]["fitness"],
            )
            for kind, population in populations.items()
        }
        summary = {
            "generation": generation,
            "unique_evaluated": len(archive),
            "kinds": {
                kind: {
                    "best": records[ids[0]],
                    "median_fitness": float(
                        np.median([records[i]["fitness"] for i in ids])
                    ),
                }
                for kind, ids in selected.items()
            },
        }
        history.append(summary)
        np.savez_compressed(
            output / f"population_{generation:03d}.npz",
            tokens=np.stack([archive[i] for ids in selected.values() for i in ids]),
            archive_ids=np.array([i for ids in selected.values() for i in ids]),
        )
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(
            "GENERATION",
            generation,
            {
                kind: round(summary["kinds"][kind]["best"]["fitness"], 5)
                for kind in populations
            },
            "evaluated",
            len(archive),
            flush=True,
        )
    shortlist = []
    for kind in populations:
        eligible = [i for i, row in enumerate(records) if row["kind"] == kind]
        shortlist.extend(
            sorted(eligible, key=lambda i: records[i]["fitness"])[
                : plan["shortlist_per_kind"]
            ]
        )
        initial = [i for i in eligible if records[i]["generation"] == 0]
        shortlist.extend(sorted(initial, key=lambda i: records[i]["fitness"])[:16])
    shortlist = list(dict.fromkeys(shortlist))
    np.savez_compressed(
        output / "shortlist.npz",
        tokens=np.stack([archive[i] for i in shortlist]),
        archive_ids=np.array(shortlist),
    )
    (output / "shortlist.json").write_text(
        json.dumps([records[i] for i in shortlist], indent=2) + "\n"
    )
    (output / "all_scores.json").write_text(json.dumps(records, indent=2) + "\n")
    (output / "complete.json").write_text(
        json.dumps(
            {
                "unique_candidates": len(archive),
                "shortlist": len(shortlist),
                "generations": plan["generations"],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
