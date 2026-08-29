import json

from motif_pair import config
from motif_pair.data.generate import is_task_feasible
from motif_pair.evaluation.task_split import make_split, validate_split, write_split


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
