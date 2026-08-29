from pathlib import Path

from motif_pair.data.audit import audit_split
from motif_pair.evaluation.task_split import make_split, write_split


def test_seed42_linear_raw_bit_shortcut_is_near_chance(tmp_path: Path):
    """Density/position features must not predict a held-out task label."""
    split_path = tmp_path / "split.json"
    write_split(make_split(42), split_path)
    result = audit_split(split_path, n_samples=2_048, seed=42, steps=300)
    assert result["mean_accuracy"] <= 0.54
    assert result["max_accuracy"] <= 0.58
