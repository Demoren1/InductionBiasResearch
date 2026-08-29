import hashlib

import torch

from motif_pair import config
from motif_pair.data.generate import (
    circular_windows,
    exhaustive_sequence_bank,
    generate_data,
    gold_mask,
    make_task_bank,
    sample_balanced,
)


TASK = config.Task("000", "111", 3)


def test_exhaustive_sequence_bank_has_every_16_bit_sequence_once():
    bank = exhaustive_sequence_bank()
    assert bank.shape == (2 ** config.SEQ_LEN, config.SEQ_LEN)
    values = (bank.to(torch.long) * (2 ** torch.arange(15, -1, -1))).sum(dim=1)
    assert torch.equal(values, torch.arange(2 ** config.SEQ_LEN))


def test_task_bank_labels_are_exact_relative_gaps_with_exact_occurrence_counts():
    bank = make_task_bank(TASK)
    x01 = (bank["x"] + 1.0) / 2.0
    windows = circular_windows(x01)
    a = torch.tensor([int(bit) for bit in TASK.a], dtype=torch.float32)
    b = torch.tensor([int(bit) for bit in TASK.b], dtype=torch.float32)
    a_matches = (windows == a.view(1, 1, -1)).all(dim=-1)
    b_matches = (windows == b.view(1, 1, -1)).all(dim=-1)
    assert torch.equal(a_matches.sum(dim=1), torch.ones(len(x01), dtype=torch.long))
    assert torch.equal(b_matches.sum(dim=1), torch.ones(len(x01), dtype=torch.long))
    delta = (b_matches.long().argmax(dim=1) - a_matches.long().argmax(dim=1)) % config.SEQ_LEN
    assert torch.equal(delta, bank["delta"])
    assert torch.equal(bank["y"], (delta == TASK.gap).float())
    assert set(delta.tolist()).issubset(set(config.GAPS))
    offsets = torch.arange(config.MOTIF_LEN)
    a_positions = (bank["a_start"][:, None] + offsets) % config.SEQ_LEN
    b_positions = (bank["b_start"][:, None] + offsets) % config.SEQ_LEN
    assert not (a_positions[:, :, None] == b_positions[:, None, :]).any()
    assert (bank["y"] == 0).any() and (bank["y"] == 1).any()


def test_balanced_sampler_is_deterministic_and_balanced():
    first = sample_balanced(TASK, n_samples=1_024, seed=19)
    second = sample_balanced(TASK, n_samples=1_024, seed=19)
    assert torch.equal(first["x"], second["x"])
    assert torch.equal(first["y"], second["y"])
    assert int(first["y"].sum()) == 512
    positive_counts = torch.bincount(first["ones_count"][first["y"] == 1], minlength=config.SEQ_LEN + 1)
    negative_counts = torch.bincount(first["ones_count"][first["y"] == 0], minlength=config.SEQ_LEN + 1)
    assert torch.equal(positive_counts, negative_counts)
    negatives = first["delta"][first["y"] == 0]
    assert set(negatives.tolist()).issubset(set(config.GAPS) - {TASK.gap})


def test_gold_mask_has_six_edges_per_column_and_changes_with_gap():
    gap_three = gold_mask(config.Task("000", "111", 3))
    gap_four = gold_mask(config.Task("000", "111", 4))
    assert gap_three.shape == (config.SEQ_LEN, config.H)
    assert int(gap_three.sum()) == config.K_ACTIVE == 96
    assert torch.equal(gap_three.sum(dim=0), torch.full((config.H,), 6, dtype=torch.long))
    assert not torch.equal(gap_three, gap_four)


def test_gap_only_condition_is_one_hot_and_ignores_motif_identity():
    first = config.task_to_condition(config.Task("000", "111", 6))
    second = config.task_to_condition(config.Task("010", "101", 6))
    assert first.shape == (config.COND_DIM,)
    assert torch.equal(first, second)
    assert int(first.sum()) == 1


def test_generate_data_writes_split_artifacts_to_explicit_data_dir(tmp_path):
    split = tmp_path / "split.json"
    task = TASK.id
    split.write_text('{"train_tasks": ["' + task + '"], "test_tasks": []}')
    data_dir = tmp_path / "isolated_data"

    generate_data(split, n_val=8, device="cpu", data_dir=data_dir)

    bank_path = data_dir / f"bank_{task}.pt"
    val_path = data_dir / f"val_{task}.pt"
    assert bank_path.is_file()
    assert val_path.is_file()
    val = torch.load(val_path, weights_only=True)
    assert val["task"] == task
    assert val["split_sha256"] == hashlib.sha256(split.read_bytes()).hexdigest()
    assert val["split_train_tasks"] == [task]
