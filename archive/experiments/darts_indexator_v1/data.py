"""Observation partitions, provenance and label-independent index canonicalization."""

import hashlib
import itertools
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from pattern.datasets import input_table, labels_for

BASE = Path(__file__).resolve().parent


def save_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def configuration(protocol):
    plan = json.loads(Path(protocol).read_text())
    groups = [
        plan[key]
        for key in ("source_patterns", "selection_patterns", "target_patterns")
    ]
    flattened = [pattern for group in groups for pattern in group]
    if len(set(flattened)) != len(flattened) or len(flattened) != 16:
        raise ValueError(
            "Source, selection and target patterns must partition the domain"
        )
    if (plan["dimensions"], plan["hidden"], plan["codes"]) != (11, 8, 4):
        raise ValueError("This experiment implements the original Pattern 11-8-1 task")
    if not 0 < plan["support"] + plan["query"] + plan["selection"] < 2048:
        raise ValueError("Observation partitions must leave a disjoint test set")
    return plan


def observations(patterns, seed, plan):
    _, inputs = input_table()
    order = np.random.default_rng(seed).permutation(2048)
    boundaries = np.cumsum([0, plan["support"], plan["query"], plan["selection"], 512])
    targets = np.stack([labels_for(inputs, pattern).numpy() for pattern in patterns])
    result = {"patterns": np.array(patterns)}
    for split, left, right in zip(
        ("support", "query", "selection", "test"), boundaries[:-1], boundaries[1:]
    ):
        ids = order[left:right]
        result[split + "_ids"] = ids
        result[split + "_x"] = inputs[ids].numpy()
        result[split + "_y"] = targets[:, ids]
    return result


def prepare(protocol):
    plan = configuration(protocol)
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "protocol.json", plan)
    np.savez_compressed(
        output / "source.npz",
        **observations(plan["source_patterns"], plan["seed"], plan),
    )
    np.savez_compressed(
        output / "selection.npz",
        **observations(plan["selection_patterns"], plan["seed"] + 100001, plan),
    )
    snapshot = output / "code_snapshot"
    snapshot.mkdir()
    for path in BASE.iterdir():
        if path.suffix in (".py", ".json", ".md"):
            shutil.copy2(path, snapshot / path.name)
    dependency = Path("pattern/datasets.py")
    shutil.copy2(dependency, snapshot / "pattern_datasets.py")
    save_json(
        output / "started.json",
        {
            "started_utc": datetime.now(UTC).isoformat(),
            "protocol_sha256": digest(output / "protocol.json"),
            "source_sha256": digest(output / "source.npz"),
            "selection_sha256": digest(output / "selection.npz"),
            "target_labels_constructed": False,
            "software_tests_run": False,
        },
    )
    print("PREPARED", str(output), flush=True)


def canonical_indices(indices):
    best_key, best = None, None
    for permutation in itertools.permutations(range(1, 5)):
        mapping = np.array((0, *permutation), dtype=np.int16)
        renamed = mapping[indices]
        order = sorted(range(8), key=lambda column: tuple(renamed[:, column]))
        candidate = renamed[:, order]
        key = tuple(candidate.ravel())
        if best_key is None or key < best_key:
            best_key, best = key, candidate.copy()
    return best


def search_configurations(plan):
    search = plan["search"]
    return [
        {
            "id": number,
            "numeric_lr": numeric_lr,
            "architecture_lr": architecture_lr,
            "schedule": schedule,
            "sparsity_penalty": sparsity,
            "pair_penalty": pair,
            "variance_penalty": variance,
        }
        for number, (
            numeric_lr,
            architecture_lr,
            schedule,
            sparsity,
            pair,
            variance,
        ) in enumerate(
            itertools.product(
                search["numeric_learning_rates"],
                search["architecture_learning_rates"],
                search["schedules"],
                search.get("sparsity_penalties", [0.0]),
                search.get("pair_penalties", [0.0]),
                search.get("variance_penalties", [0.0]),
            )
        )
    ]
