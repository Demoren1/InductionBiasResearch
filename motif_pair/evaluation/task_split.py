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


def _canonical_heldout_gaps(heldout_gaps: Iterable[int]) -> tuple[int, ...]:
    """Validate and canonically order the two entirely held-out gap regimes."""
    requested = tuple(heldout_gaps)
    if len(requested) != 2:
        raise ValueError("gap-held-out split requires exactly two heldout gaps")
    if len(set(requested)) != len(requested):
        raise ValueError("heldout gaps must be distinct")
    invalid = set(requested) - set(config.GAPS)
    if invalid:
        raise ValueError(f"heldout gaps must be drawn from {config.GAPS}, got {sorted(invalid)}")
    return tuple(gap for gap in config.GAPS if gap in requested)


def _sample_role_covering_pairs(
    candidates: set[tuple[str, str]],
    *,
    count: int,
    rng: random.Random,
) -> list[tuple[str, str]]:
    """Sample pairs covering every motif exactly once in each role.

    With eight pairs for eight motifs this is a perfect matching in the
    feasible ordered-pair graph.  Backtracking is small (eight vertices), and
    random candidate order makes the chosen matching seed-dependent without
    relying on improbable rejection sampling of a role-covered set.
    """
    if count < len(config.MOTIFS):
        raise ValueError("tasks_per_gap must be at least the number of motifs for role coverage")
    by_a = {
        a: [pair for pair in sorted(candidates) if pair[0] == a]
        for a in config.MOTIFS
    }

    def search(remaining_a: tuple[str, ...], used_b: set[str]) -> list[tuple[str, str]] | None:
        if not remaining_a:
            return []
        # Most-constrained-first avoids exploring dead matchings; config order
        # keeps equal cases reproducible before the seeded candidate shuffle.
        a = min(
            remaining_a,
            key=lambda value: (sum(b not in used_b for _, b in by_a[value]), config.MOTIFS.index(value)),
        )
        choices = [(left, b) for left, b in by_a[a] if b not in used_b]
        rng.shuffle(choices)
        next_remaining = tuple(value for value in remaining_a if value != a)
        for pair in choices:
            remainder = search(next_remaining, used_b | {pair[1]})
            if remainder is not None:
                return [pair, *remainder]
        return None

    matching = search(config.MOTIFS, set())
    if matching is None:
        raise ValueError("no role-covering matching exists among the feasible pairs")
    if count == len(matching):
        return matching
    extras = sorted(candidates - set(matching))
    if len(extras) < count - len(matching):
        raise ValueError("too few pairs for the requested tasks_per_gap")
    return matching + rng.sample(extras, count - len(matching))


def make_gap_heldout_split(
    seed: int,
    heldout_gaps: Iterable[int],
    *,
    tasks_per_gap: int = config.TASKS_PER_GAP,
    pair_policy: str = "shared",
) -> dict:
    """Create a 48/16 split whose test gaps are absent from meta-training.

    This is the gap-OOD counterpart to :func:`make_split`: all sampled tasks
    at the two held-out structural regimes are meta-test tasks, while all
    sampled tasks at the other six regimes are meta-train tasks.  ``shared``
    (the primary gap-only OOD setting) reuses the same ordered motif pairs at
    every gap.  ``disjoint`` reserves test pairs globally for a stricter,
    joint gap-and-pair OOD setting.
    """
    heldout = _canonical_heldout_gaps(heldout_gaps)
    train_gaps = tuple(gap for gap in config.GAPS if gap not in heldout)
    feasible = {gap: list(valid_pairs_for_gap(gap)) for gap in config.GAPS}
    if any(len(feasible[gap]) < tasks_per_gap for gap in config.GAPS):
        raise ValueError("tasks_per_gap exceeds the available pairs for a gap")
    if pair_policy not in {"shared", "disjoint"}:
        raise ValueError("pair_policy must be either 'shared' or 'disjoint'")

    common_pairs = set.intersection(*(set(feasible[gap]) for gap in config.GAPS))
    if pair_policy == "shared" and len(common_pairs) < tasks_per_gap:
        raise ValueError("too few pairs are feasible at every gap for shared pair_policy")

    # Rejection sampling is deterministic and keeps every motif observable in
    # both roles during meta-training.  In the disjoint mode, a held-out test
    # pair is excluded from every training gap, rather than merely its own.
    for attempt in range(10_000):
        catalog_seed = seed + attempt * 1_000_003
        rng = random.Random(catalog_seed)
        if pair_policy == "shared":
            shared_pairs = _sample_role_covering_pairs(
                common_pairs, count=tasks_per_gap, rng=rng,
            )
            train_by_gap = {gap: shared_pairs for gap in train_gaps}
            test_by_gap = {gap: shared_pairs for gap in heldout}
        else:
            shared_pairs = None
            heldout_pairs: set[tuple[str, str]] = set()
            test_by_gap: dict[int, list[tuple[str, str]]] = {}
            failed = False
            for gap in heldout:
                choices = [pair for pair in feasible[gap] if pair not in heldout_pairs]
                if len(choices) < tasks_per_gap:
                    failed = True
                    break
                test_by_gap[gap] = rng.sample(choices, tasks_per_gap)
                heldout_pairs.update(test_by_gap[gap])
            if failed:
                continue

            train_by_gap = {}
            for gap in train_gaps:
                choices = [pair for pair in feasible[gap] if pair not in heldout_pairs]
                if len(choices) < tasks_per_gap:
                    failed = True
                    break
                train_by_gap[gap] = rng.sample(choices, tasks_per_gap)
            if failed:
                continue

        train_tasks = [
            config.Task(a, b, gap).id
            for gap in train_gaps
            for a, b in train_by_gap[gap]
        ]
        test_tasks = [
            config.Task(a, b, gap).id
            for gap in heldout
            for a, b in test_by_gap[gap]
        ]
        catalog = [
            config.Task(a, b, gap).id
            for gap in config.GAPS
            for a, b in (test_by_gap[gap] if gap in heldout else train_by_gap[gap])
        ]
        split = {
            "split_kind": "gap_heldout",
            "split_seed": seed,
            "catalog_seed": catalog_seed,
            "tasks_per_gap": tasks_per_gap,
            "heldout_gaps": list(heldout),
            "train_gaps": list(train_gaps),
            "pair_policy": pair_policy,
            "pair_disjoint": pair_policy == "disjoint",
            "catalog_tasks": catalog,
            "train_tasks": train_tasks,
            "test_tasks": test_tasks,
        }
        if shared_pairs is not None:
            split["shared_pairs"] = [list(pair) for pair in shared_pairs]
        try:
            validate_split(split)
            return split
        except ValueError:
            continue
    raise RuntimeError("could not construct a role-covered gap-held-out split after 10,000 attempts")


def _validate_gap_heldout_split(split: dict) -> None:
    """Validate a manifest produced by :func:`make_gap_heldout_split`."""
    required = {
        "split_kind", "heldout_gaps", "train_gaps", "catalog_tasks",
        "train_tasks", "test_tasks", "pair_policy", "pair_disjoint",
    }
    missing = required - set(split)
    if missing:
        raise ValueError(f"gap-held-out split missing keys: {sorted(missing)}")
    if split["split_kind"] != "gap_heldout":
        raise ValueError("unexpected gap-held-out split kind")
    heldout = _canonical_heldout_gaps(split["heldout_gaps"])
    train_gaps = tuple(gap for gap in config.GAPS if gap not in heldout)
    if tuple(split["train_gaps"]) != train_gaps:
        raise ValueError("train_gaps does not match heldout_gaps")
    pair_policy = split["pair_policy"]
    if pair_policy not in {"shared", "disjoint"}:
        raise ValueError("pair_policy must be either 'shared' or 'disjoint'")
    if bool(split["pair_disjoint"]) != (pair_policy == "disjoint"):
        raise ValueError("pair_disjoint is inconsistent with pair_policy")

    tasks_per_gap = int(split.get("tasks_per_gap", config.TASKS_PER_GAP))
    validate_catalog(split["catalog_tasks"], tasks_per_gap=tasks_per_gap)
    catalog = [config.parse_task(task).id for task in split["catalog_tasks"]]
    train = [config.parse_task(task).id for task in split["train_tasks"]]
    test = [config.parse_task(task).id for task in split["test_tasks"]]
    if len(train) != tasks_per_gap * len(train_gaps):
        raise ValueError("unexpected total number of gap-held-out train tasks")
    if len(test) != tasks_per_gap * len(heldout):
        raise ValueError("unexpected total number of gap-held-out test tasks")
    if len(set(train)) != len(train) or len(set(test)) != len(test):
        raise ValueError("train or test task list contains duplicates")
    if set(train) & set(test):
        raise ValueError("train and test tasks overlap")
    if set(train) | set(test) != set(catalog):
        raise ValueError("train/test tasks do not partition the catalog")

    train_by_gap = _ids_by_gap(train)
    test_by_gap = _ids_by_gap(test)
    for gap in heldout:
        if train_by_gap[gap]:
            raise ValueError(f"heldout gap {gap} appears in train")
    for gap in train_gaps:
        if test_by_gap[gap]:
            raise ValueError(f"training gap {gap} appears in test")
    for gap in train_gaps:
        if len(train_by_gap[gap]) != tasks_per_gap:
            raise ValueError(f"train gap {gap} has wrong coverage")
    for gap in heldout:
        if len(test_by_gap[gap]) != tasks_per_gap:
            raise ValueError(f"heldout gap {gap} has wrong test coverage")

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
    if pair_policy == "disjoint":
        overlap = train_pairs & set(test_pairs)
        if overlap:
            raise ValueError(f"train/test motif pairs overlap, e.g. {next(iter(overlap))}")
        if len(test_pairs) != len(set(test_pairs)):
            raise ValueError("held-out motif pairs must be unique across test tasks")
        if "shared_pairs" in split:
            raise ValueError("disjoint pair_policy must not declare shared_pairs")
        return

    if "shared_pairs" not in split:
        raise ValueError("shared pair_policy missing shared_pairs")
    try:
        shared_pairs = [tuple(pair) for pair in split["shared_pairs"]]
    except TypeError as error:
        raise ValueError("shared_pairs must be a list of ordered motif pairs") from error
    if len(shared_pairs) != tasks_per_gap or len(set(shared_pairs)) != tasks_per_gap:
        raise ValueError("shared_pairs must contain tasks_per_gap unique pairs")
    if any(a not in config.MOTIFS or b not in config.MOTIFS or a == b for a, b in shared_pairs):
        raise ValueError("shared_pairs contains an invalid ordered motif pair")
    expected_pairs = set(shared_pairs)
    if train_pairs != expected_pairs or set(test_pairs) != expected_pairs:
        raise ValueError("shared pair_policy must use shared_pairs at every gap")
    for gap in config.GAPS:
        catalog_pairs = {
            (config.parse_task(task).a, config.parse_task(task).b)
            for task in _ids_by_gap(catalog)[gap]
        }
        if catalog_pairs != expected_pairs:
            raise ValueError(f"gap {gap} does not use exactly the shared_pairs")


def validate_split(split: dict) -> None:
    if split.get("split_kind") == "gap_heldout":
        _validate_gap_heldout_split(split)
        return
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
    parser.add_argument(
        "--split-kind",
        choices=("pair_disjoint", "gap_heldout"),
        default="pair_disjoint",
        help="legacy pair-disjoint split or gap-held-out OOD split",
    )
    parser.add_argument(
        "--heldout-gaps",
        type=int,
        nargs=2,
        metavar=("GAP_1", "GAP_2"),
        help="the two test-only gaps; required for --split-kind gap_heldout",
    )
    parser.add_argument(
        "--pair-policy",
        choices=("shared", "disjoint"),
        default="shared",
        help="motif-pair policy for the gap-held-out split",
    )
    args = parser.parse_args()
    configure_compute_device(args.device)
    if args.split_kind == "gap_heldout":
        if args.heldout_gaps is None:
            parser.error("--heldout-gaps is required for --split-kind gap_heldout")
        split = make_gap_heldout_split(
            args.seed,
            args.heldout_gaps,
            pair_policy=args.pair_policy,
        )
    else:
        if args.heldout_gaps is not None:
            parser.error("--heldout-gaps is valid only for --split-kind gap_heldout")
        split = make_split(args.seed)
    write_split(split, args.out, args.shell_out)
    print("TRAIN_TASKS=" + " ".join(split["train_tasks"]))
    print("TEST_TASKS=" + " ".join(split["test_tasks"]))
    print(f"split -> {args.out}")


if __name__ == "__main__":
    main()
