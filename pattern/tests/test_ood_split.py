import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from evaluation.ood_split import make_split, validate_split


def test_seeded_split_is_reproducible_and_exhaustive():
    first = make_split(seed=42)
    second = make_split(seed=42)

    assert first == second
    assert len(first["train_patterns"]) == 12
    assert len(first["test_patterns"]) == 4
    assert set(first["train_patterns"]).isdisjoint(first["test_patterns"])
    assert set(first["train_patterns"] + first["test_patterns"]) == set(config.PATTERNS)


def test_split_seed_42_expected_holdout():
    split = make_split(seed=42)
    assert split["test_patterns"] == ["0100", "1011", "0000", "0011"]


def test_validate_split_rejects_overlap():
    split = make_split(seed=42)
    split["test_patterns"][0] = split["train_patterns"][0]

    try:
        validate_split(split)
    except ValueError as error:
        assert "overlap" in str(error) or "partition" in str(error)
    else:
        raise AssertionError("overlapping split was accepted")
