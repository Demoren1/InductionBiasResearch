from __future__ import annotations

import sys
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.importance import importance_from
from models.cvae import canonicalize_hidden_columns
from models.mlp import BatchedMaskedMLP, generate_masks
from models.train import build_parser, own_mlp_slice
from selection.select_best import select_best
from selection.select_best import load_all_checkpoints
from models.cvae import read_split_provenance


def test_mask_generation_and_batched_forward_are_deterministic():
    masks_a = generate_masks(3, 16, 16, .375, seed=7)
    masks_b = generate_masks(3, 16, 16, .375, seed=7)
    assert torch.equal(masks_a, masks_b)
    assert masks_a.shape == (3, 16, 16)
    model = BatchedMaskedMLP(3, 16, 16)
    model.load_masks(masks_a)
    assert model(torch.randn(5, 16)).shape == (5, 3)


def test_worker_slices_partition_all_candidates():
    slices = [own_mlp_slice(512, worker, 3) for worker in range(3)]
    claimed = torch.cat([torch.arange(start, start + count) for start, count in slices])
    assert torch.equal(claimed, torch.arange(512))


def test_candidate_train_parser_accepts_split_specific_data_dir(tmp_path: Path):
    data_dir = tmp_path / "split_data"
    args = build_parser().parse_args([
        "--task", "A000_B001_G03", "--split_json", "split.json", "--data_dir", str(data_dir),
    ])
    assert args.data_dir == data_dir


def test_selection_uses_lowest_validation_bce_and_preserves_provenance():
    candidates = {
        "task": "A000_B001_G03",
        "global_idx": torch.tensor([0, 1, 2, 3]),
        "val_loss": torch.tensor([.5, .2, .4, .1]),
        "val_acc": torch.tensor([.5, .7, .6, .8]),
        "masks": torch.ones(4, 16, 16),
        "params": {"w1": torch.ones(4, 16, 16), "b1": torch.zeros(4, 16),
                   "w2": torch.zeros(4, 16, 1), "b2": torch.zeros(4, 1)},
        "n_mlps_per_task": 4,
        "mask_probability": .375,
        "split_path": "/tmp/split.json",
        "split_sha256": "hash",
        "split_train_tasks": ["A000_B001_G03"],
    }
    selected = select_best(candidates, .5)
    assert selected["global_idx"].tolist() == [3, 1]
    assert selected["split_sha256"] == "hash"


def test_candidate_shards_reject_stale_split_sha_despite_same_train_tasks(tmp_path):
    task = "A000_B001_G03"
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train_tasks": [task], "test_tasks": ["A001_B010_G04"]}))
    expected = read_split_provenance(split)
    shard = {
        "task": task, "global_idx": torch.tensor([0]),
        "params": {"w1": torch.ones(1, 16, 16), "b1": torch.zeros(1, 16),
                   "w2": torch.zeros(1, 16, 1), "b2": torch.zeros(1, 1)},
        "masks": torch.ones(1, 16, 16), "val_loss": torch.tensor([.2]),
        "val_acc": torch.tensor([.7]), "n_mlps_per_task": 1, "mask_probability": .375,
        **expected,
    }
    shard["split_sha256"] = "stale"
    directory = tmp_path / f"task_{task}"
    directory.mkdir()
    torch.save(shard, directory / "gpu0_round000.pt")
    try:
        load_all_checkpoints(task, ckpt_root=tmp_path, expected_provenance=expected)
    except ValueError as error:
        assert "SHA256 mismatch" in str(error)
    else:
        raise AssertionError("stale candidate shard must be rejected")


def test_importance_is_raw_masked_absolute_weight_normalized_per_model():
    w1 = torch.tensor([[[2.0, -4.0], [7.0, 1.0]], [[3.0, 0.0], [0.0, -6.0]]])
    masks = torch.tensor([[[1.0, 1.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])
    result = importance_from(w1, masks)
    expected = torch.tensor([[[.5, 1.0], [0.0, .25]], [[.5, 0.0], [0.0, 1.0]]])
    assert torch.allclose(result, expected)


def test_column_canonicalization_removes_hidden_permutations():
    maps = torch.zeros(2, 16, 16)
    for column in range(16):
        maps[:, column, column] = 1
        maps[:, (column + 1) % 16, column] = 1
        maps[:, (column + 2) % 16, column] = 1
    permutation = torch.tensor([7, 2, 13, 0, 15, 4, 9, 1, 10, 3, 12, 5, 14, 6, 11, 8])
    permuted = maps[:, :, permutation]
    assert torch.equal(canonicalize_hidden_columns(maps),
                       canonicalize_hidden_columns(permuted))


def test_column_canonicalization_is_permutation_invariant_with_ties():
    generator = torch.Generator().manual_seed(991)
    maps = (torch.rand(12, 16, 16, generator=generator) < .375).float()
    expected = canonicalize_hidden_columns(maps)
    for _ in range(10):
        permutation = torch.randperm(16, generator=generator)
        actual = canonicalize_hidden_columns(maps[:, :, permutation])
        assert torch.equal(actual, expected)
