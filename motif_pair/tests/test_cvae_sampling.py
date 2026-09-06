"""CPU tests for explicit and legacy CVAE condition encodings."""

from __future__ import annotations

import pytest
import torch

import config
from evaluation.eval_generated_masks import _bootstrap_strict_margin_ci, _masks
from models.cvae import CVAE, checkpoint_condition_encoding, task_condition


def test_legacy_one_hot_condition_is_unchanged_by_default():
    task = config.Task("000", "111", 6)
    expected = torch.zeros(config.COND_DIM)
    expected[config.GAPS.index(6)] = 1.0
    assert torch.equal(config.task_to_condition(task), expected)
    assert torch.equal(task_condition([task]), expected.unsqueeze(0))
    model = CVAE(mask_dim=12, latent_dim=3, hidden=7, cond_dim=config.COND_DIM)
    assert model.condition_encoding == config.CONDITION_ENCODING_ONE_HOT
    assert model.condition(task).shape == (1, config.COND_DIM)


def test_scalar_gap_condition_is_normalized_over_full_gap_universe():
    low = config.Task("000", "111", min(config.GAPS))
    middle = config.Task("000", "111", 6)
    high = config.Task("000", "111", max(config.GAPS))
    condition = task_condition([low, middle, high], condition_encoding="scalar")
    assert condition.shape == (3, 1)
    assert torch.allclose(condition[:, 0], torch.tensor([0.0, 3 / 7, 1.0]))

    model = CVAE(mask_dim=12, latent_dim=3, hidden=7, condition_encoding="scalar")
    assert model.cond_dim == 1
    samples = model.sample_topk([middle], n_per=4, k_active=5,
                                generator=torch.Generator().manual_seed(7))
    assert samples.shape == (4, 12)
    assert torch.equal(samples.sum(dim=1), torch.full((4,), 5.0))


def test_checkpoint_encoding_is_explicit_and_legacy_safe():
    assert checkpoint_condition_encoding({"cond_dim": config.COND_DIM}) == "one_hot"
    assert checkpoint_condition_encoding({"cond_dim": 0}) == "none"
    assert checkpoint_condition_encoding({"cond_dim": 1, "condition_encoding": "scalar"}) == "scalar"
    with pytest.raises(ValueError, match="non-legacy cond_dim"):
        checkpoint_condition_encoding({"cond_dim": 1})
    with pytest.raises(ValueError, match="condition mismatch"):
        checkpoint_condition_encoding({"cond_dim": 1, "condition_encoding": "one_hot"})


def test_cvae_rejects_dimension_encoding_mismatch():
    with pytest.raises(ValueError, match="requires cond_dim"):
        CVAE(mask_dim=12, latent_dim=3, hidden=7, cond_dim=config.COND_DIM,
             condition_encoding="scalar")


def test_wrong_gap_ties_reuse_identical_latent_samples_for_both_sides():
    torch.manual_seed(17)
    model = CVAE(mask_dim=config.MASK_DIM, latent_dim=4, hidden=12,
                 condition_encoding="scalar").eval()
    task = "A000_B001_G05"
    seed = 991
    actual = _masks(
        task, "cvae_wrong_gap", 3, cvae=model, vae=None, mean=None,
        generator=torch.Generator().manual_seed(seed), device=torch.device("cpu"),
        meta_train_gaps=[3, 4, 6, 7, 9, 10],
    )
    assert actual.shape == (6, config.MASK_DIM)
    for ordinal, wrong_gap in enumerate((4, 6)):
        condition = model.condition([f"A000_B001_G{wrong_gap:02d}"])
        expected = model.sample_topk(
            condition, 3, config.K_ACTIVE,
            generator=torch.Generator().manual_seed(seed),
        )
        assert torch.equal(actual[ordinal * 3:(ordinal + 1) * 3], expected)


def test_strict_margin_bootstrap_reselects_strongest_wrong_condition():
    # The first wrong condition dominates only when the second latent is
    # resampled; the second wrong condition wins for the observed mean and for
    # first-latent-only resamples.  A fixed-winner bootstrap would miss the
    # attainable -1.0 lower tail.
    interval = _bootstrap_strict_margin_ci(
        torch.tensor([1.0, 0.0]),
        [torch.tensor([0.0, 1.0]), torch.tensor([0.6, 0.6])],
        seed=17,
        n_resamples=10_000,
    )
    assert interval == pytest.approx([-1.0, 0.4])
