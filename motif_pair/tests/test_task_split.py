import json

import pytest

from motif_pair import config
from motif_pair.data.generate import is_task_feasible
from motif_pair.evaluation.task_split import (
    make_gap_heldout_split,
    make_split,
    validate_split,
    write_split,
)


def test_split_is_reproducible_disjoint_and_stratified():
    first = make_split(42)
    second = make_split(42)
    assert first == second
    assert len(first["catalog_tasks"]) == 64
    assert len(first["train_tasks"]) == 48
    assert len(first["test_tasks"]) == 16
    assert set(first["train_tasks"]).isdisjoint(first["test_tasks"])
    train_pairs = {(config.parse_task(task).a, config.parse_task(task).b)
                   for task in first["train_tasks"]}
    test_pairs = [(config.parse_task(task).a, config.parse_task(task).b)
                  for task in first["test_tasks"]]
    assert train_pairs.isdisjoint(test_pairs)
    assert len(test_pairs) == len(set(test_pairs)) == 16
    assert {config.parse_task(task).a for task in first["train_tasks"]} == set(config.MOTIFS)
    assert {config.parse_task(task).b for task in first["train_tasks"]} == set(config.MOTIFS)
    for gap in config.GAPS:
        train = [task for task in first["train_tasks"] if config.parse_task(task).gap == gap]
        test = [task for task in first["test_tasks"] if config.parse_task(task).gap == gap]
        assert len(train) == 6
        assert len(test) == 2
    assert all(is_task_feasible(task) for task in first["catalog_tasks"])


def test_split_json_and_shell_artifacts_are_persisted(tmp_path):
    split = make_split(7)
    json_path = tmp_path / "split.json"
    shell_path = tmp_path / "split.sh"
    write_split(split, json_path, shell_path)
    loaded = json.loads(json_path.read_text())
    assert loaded == split
    shell = shell_path.read_text()
    assert "TRAIN_TASKS=" in shell and "TEST_TASKS=" in shell


def test_validate_split_rejects_overlap():
    split = make_split(42)
    split["test_tasks"][0] = split["train_tasks"][0]
    try:
        validate_split(split)
    except ValueError as error:
        assert "overlap" in str(error) or "partition" in str(error)
    else:
        raise AssertionError("overlapping split was accepted")


@pytest.mark.parametrize("heldout_gaps", ((5, 8), (3, 10)))
def test_gap_heldout_split_is_reproducible_and_structurally_ood(heldout_gaps):
    first = make_gap_heldout_split(42, heldout_gaps)
    second = make_gap_heldout_split(42, tuple(reversed(heldout_gaps)))
    assert first == second
    assert first["split_kind"] == "gap_heldout"
    assert first["heldout_gaps"] == sorted(heldout_gaps)
    assert first["train_gaps"] == [gap for gap in config.GAPS if gap not in heldout_gaps]
    assert first["pair_policy"] == "shared"
    assert first["pair_disjoint"] is False
    assert len(first["shared_pairs"]) == 8
    assert len(first["catalog_tasks"]) == 64
    assert len(first["train_tasks"]) == 48
    assert len(first["test_tasks"]) == 16
    assert all(is_task_feasible(task) for task in first["catalog_tasks"])
    assert {config.parse_task(task).gap for task in first["train_tasks"]}.isdisjoint(heldout_gaps)
    assert {config.parse_task(task).gap for task in first["test_tasks"]} == set(heldout_gaps)
    train_pairs = {(config.parse_task(task).a, config.parse_task(task).b)
                   for task in first["train_tasks"]}
    test_pairs = [(config.parse_task(task).a, config.parse_task(task).b)
                  for task in first["test_tasks"]]
    shared_pairs = {tuple(pair) for pair in first["shared_pairs"]}
    assert train_pairs == set(test_pairs) == shared_pairs
    for gap in config.GAPS:
        gap_pairs = {
            (config.parse_task(task).a, config.parse_task(task).b)
            for task in first["catalog_tasks"]
            if config.parse_task(task).gap == gap
        }
        assert gap_pairs == shared_pairs
    assert {config.parse_task(task).a for task in first["train_tasks"]} == set(config.MOTIFS)
    assert {config.parse_task(task).b for task in first["train_tasks"]} == set(config.MOTIFS)


def test_gap_heldout_split_rejects_invalid_heldout_configuration():
    with pytest.raises(ValueError, match="exactly two"):
        make_gap_heldout_split(42, (5,))
    with pytest.raises(ValueError, match="heldout gaps"):
        make_gap_heldout_split(42, (5, 11))


def test_validate_gap_heldout_split_rejects_pair_leakage():
    split = make_gap_heldout_split(42, (5, 8), pair_policy="disjoint")
    replacement = next(
        config.Task(parsed.a, parsed.b, split["heldout_gaps"][0]).id
        for task in split["train_tasks"]
        for parsed in (config.parse_task(task),)
        if is_task_feasible(config.Task(parsed.a, parsed.b, split["heldout_gaps"][0]))
    )
    old_test_task = split["test_tasks"][0]
    split["test_tasks"][0] = replacement
    split["catalog_tasks"][split["catalog_tasks"].index(old_test_task)] = replacement
    with pytest.raises(ValueError, match="motif pairs overlap"):
        validate_split(split)


def test_validate_gap_heldout_split_rejects_gap_leakage():
    split = make_gap_heldout_split(42, (5, 8))
    train_task = split["train_tasks"][0]
    test_task = split["test_tasks"][0]
    split["train_tasks"][0] = test_task
    split["test_tasks"][0] = train_task
    with pytest.raises(ValueError, match=r"appears in (train|test)"):
        validate_split(split)
