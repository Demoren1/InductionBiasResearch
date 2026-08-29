"""Deterministic stratified task catalogue and meta-train/meta-test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shlex
import sys
from typing import Iterable

try:
    from motif_pair import config
    from motif_pair.data.generate import configure_compute_device, is_task_feasible, valid_pairs_for_gap
except ModuleNotFoundError:  # pragma: no cover - direct script invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import config  # type: ignore
    from data.generate import configure_compute_device, is_task_feasible, valid_pairs_for_gap  # type: ignore


def _ids_by_gap(tasks: Iterable[config.Task | str]) -> dict[int, list[str]]:
    grouped = {gap: [] for gap in config.GAPS}
    for task in tasks:
        parsed = config.parse_task(task)
        grouped[parsed.gap].append(parsed.id)
    return grouped


def make_catalog(seed: int, tasks_per_gap: int = config.TASKS_PER_GAP) -> list[str]:
    """Sample the same number of valid A/B pairs for every allowed gap."""
    candidates = {
        gap: [config.Task(a, b, gap).id for a, b in valid_pairs_for_gap(gap)]
        for gap in config.GAPS
    }
    if not 1 <= tasks_per_gap <= len(next(iter(candidates.values()))):
        raise ValueError("tasks_per_gap must be between 1 and the available pairs per gap")
    rng = random.Random(seed)
    catalog: list[str] = []
    for gap in config.GAPS:
        # ``sample`` preserves a random order; retaining it makes all derived
        # artifacts deterministic without pretending task ids are ordered.
        catalog.extend(rng.sample(candidates[gap], tasks_per_gap))
    validate_catalog(catalog, tasks_per_gap=tasks_per_gap)
    return catalog


def validate_catalog(catalog_tasks: Iterable[config.Task | str],
                     tasks_per_gap: int = config.TASKS_PER_GAP) -> None:
    task_ids = [config.parse_task(task).id for task in catalog_tasks]
    if len(task_ids) != tasks_per_gap * len(config.GAPS):
        raise ValueError("catalog has an unexpected total task count")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("catalog contains duplicate tasks")
    infeasible = [task_id for task_id in task_ids if not is_task_feasible(task_id)]
    if infeasible:
        raise ValueError(f"catalog contains infeasible tasks, e.g. {infeasible[0]}")
    grouped = _ids_by_gap(task_ids)
    for gap, ids in grouped.items():
        if len(ids) != tasks_per_gap:
            raise ValueError(f"gap {gap} has {len(ids)} catalog tasks, expected {tasks_per_gap}")


def make_split(seed: int, *, tasks_per_gap: int = config.TASKS_PER_GAP,
               train_per_gap: int = config.TRAIN_TASKS_PER_GAP) -> dict:
    """Create a globally motif-pair-disjoint 48/16 OOD split.

    Test ``(A, B)`` pairs never occur in meta-train at any gap.  Test pairs
    are also unique, so the 16 held-out tasks represent 16 genuinely unseen
    motif compositions rather than merely unseen ``(A, B, gap)`` triples.
    """
    if not 0 < train_per_gap < tasks_per_gap:
        raise ValueError("train_per_gap must be strictly between 0 and tasks_per_gap")
    # Rejection sampling is deterministic because every trial is derived from
    # ``seed``.  It prevents a random 6/2 allocation from making a motif role
    # itself OOD, which would confound the intended held-out *triple* test.
    for attempt in range(10_000):
        catalog_seed = seed + attempt * 1_000_003
        rng = random.Random(catalog_seed)
        test_per_gap = tasks_per_gap - train_per_gap
        heldout_pairs: set[tuple[str, str]] = set()
        test_by_gap: dict[int, list[tuple[str, str]]] = {}
        feasible = {gap: list(valid_pairs_for_gap(gap)) for gap in config.GAPS}
        failed = False
        for gap in config.GAPS:
            choices = [pair for pair in feasible[gap] if pair not in heldout_pairs]
            if len(choices) < test_per_gap:
                failed = True
                break
            chosen = rng.sample(choices, test_per_gap)
            test_by_gap[gap] = chosen
            heldout_pairs.update(chosen)
        if failed:
            continue

        train_tasks: list[str] = []
        test_tasks: list[str] = []
        catalog: list[str] = []
        for gap in config.GAPS:
            train_choices = [pair for pair in feasible[gap] if pair not in heldout_pairs]
            if len(train_choices) < train_per_gap:
                failed = True
                break
            train_pairs = rng.sample(train_choices, train_per_gap)
            gap_train = [config.Task(a, b, gap).id for a, b in train_pairs]
            gap_test = [config.Task(a, b, gap).id for a, b in test_by_gap[gap]]
            train_tasks.extend(gap_train)
            test_tasks.extend(gap_test)
            catalog.extend(gap_train + gap_test)
        if failed:
            continue
        split = {
            "split_seed": seed,
            "catalog_seed": catalog_seed,
            "tasks_per_gap": tasks_per_gap,
            "train_per_gap": train_per_gap,
            "test_per_gap": test_per_gap,
            "pair_disjoint": True,
            "catalog_tasks": catalog,
            "train_tasks": train_tasks,
            "test_tasks": test_tasks,
        }
        try:
            validate_split(split)
            return split
        except ValueError:
            continue
    raise RuntimeError("could not construct a role-covered split after 10,000 attempts")


def validate_split(split: dict) -> None:
    required = {"catalog_tasks", "train_tasks", "test_tasks"}
    missing = required - set(split)
    if missing:
        raise ValueError(f"split missing keys: {sorted(missing)}")
    tasks_per_gap = int(split.get("tasks_per_gap", config.TASKS_PER_GAP))
    train_per_gap = int(split.get("train_per_gap", config.TRAIN_TASKS_PER_GAP))
    test_per_gap = int(split.get("test_per_gap", tasks_per_gap - train_per_gap))
    if train_per_gap + test_per_gap != tasks_per_gap:
        raise ValueError("train/test per-gap counts do not add up")
    validate_catalog(split["catalog_tasks"], tasks_per_gap=tasks_per_gap)

    catalog = [config.parse_task(task).id for task in split["catalog_tasks"]]
    train = [config.parse_task(task).id for task in split["train_tasks"]]
    test = [config.parse_task(task).id for task in split["test_tasks"]]
    if len(train) != train_per_gap * len(config.GAPS):
        raise ValueError("unexpected total number of train tasks")
    if len(test) != test_per_gap * len(config.GAPS):
        raise ValueError("unexpected total number of test tasks")
    if len(set(train)) != len(train) or len(set(test)) != len(test):
        raise ValueError("train or test task list contains duplicates")
    if set(train) & set(test):
        raise ValueError("train and test tasks overlap")
    if set(train) | set(test) != set(catalog):
        raise ValueError("train/test tasks do not partition the catalog")
    for gap in config.GAPS:
        if len(_ids_by_gap(train)[gap]) != train_per_gap:
            raise ValueError(f"gap {gap} has wrong train coverage")
        if len(_ids_by_gap(test)[gap]) != test_per_gap:
            raise ValueError(f"gap {gap} has wrong test coverage")
    train_a = {config.parse_task(task).a for task in train}
    train_b = {config.parse_task(task).b for task in train}
    test_a = {config.parse_task(task).a for task in test}
    test_b = {config.parse_task(task).b for task in test}
    if train_a != set(config.MOTIFS) or train_b != set(config.MOTIFS):
        raise ValueError("meta-train must cover every motif in both A and B roles")
    if not test_a.issubset(train_a) or not test_b.issubset(train_b):
        raise ValueError("test motif roles are not covered by meta-train")
    train_pairs = {(config.parse_task(task).a, config.parse_task(task).b) for task in train}
    test_pairs = [(config.parse_task(task).a, config.parse_task(task).b) for task in test]
    overlap = train_pairs & set(test_pairs)
    if overlap:
        raise ValueError(f"train/test motif pairs overlap, e.g. {next(iter(overlap))}")
    if len(test_pairs) != len(set(test_pairs)):
        raise ValueError("held-out motif pairs must be unique across test tasks")


def write_split(split: dict, out: Path, shell_out: Path | None = None) -> None:
    """Persist JSON provenance and optional shell lists used by runner scripts."""
    validate_split(split)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split, indent=2) + "\n")
    if shell_out is not None:
        shell_out.parent.mkdir(parents=True, exist_ok=True)
        train_value = " ".join(split["train_tasks"])
        test_value = " ".join(split["test_tasks"])
        catalog_value = " ".join(split["catalog_tasks"])
        shell_out.write_text(
            f"CATALOG_TASKS={shlex.quote(catalog_value)}\n"
            f"TRAIN_TASKS={shlex.quote(train_value)}\n"
            f"TEST_TASKS={shlex.quote(test_value)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shell-out", type=Path, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    configure_compute_device(args.device)
    split = make_split(args.seed)
    write_split(split, args.out, args.shell_out)
    print("TRAIN_TASKS=" + " ".join(split["train_tasks"]))
    print("TEST_TASKS=" + " ".join(split["test_tasks"]))
    print(f"split -> {args.out}")


if __name__ == "__main__":
    main()
