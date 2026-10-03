"""Regression coverage for packed same-task DeepSets child measurements."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from generator_evaluator.adapters import measure_mask
from generator_evaluator.data import InnerProtocol, RealReplay, TaskData, support_context, tensor_hash
from generator_evaluator.deepsets_batch import fit_deepsets_batch


def _task() -> TaskData:
    generator = torch.Generator().manual_seed(921)
    x_support = torch.randn(5, 5, 784, generator=generator)
    y_support = torch.randn(5, generator=generator)
    x_query = torch.randn(4, 5, 784, generator=generator)
    y_query = torch.randn(4, generator=generator)
    return TaskData("deepsets:train:fixture", "train", x_support, y_support, x_query, y_query,
                    support_context(x_support.mean(1), y_support),
                    torch.arange(25).reshape(5, 5), torch.arange(25, 45).reshape(4, 5),
                    {"family": "deepsets"})


def _masks() -> torch.Tensor:
    masks = torch.zeros(2, 784, 3)
    masks[0].flatten()[:90] = 1
    masks[1].flatten()[40:160] = 1
    return masks


def test_same_task_packed_fit_matches_independent_children() -> None:
    task, masks = _task(), _masks()
    protocol = InnerProtocol(steps=3, replicas=2, lr=.002, l2=.0001,
                             checkpoint_every=1, seed=111, metric="nmse")
    actual = fit_deepsets_batch(masks, task, protocol)
    expected = [measure_mask(mask, task, protocol) for mask in masks]
    assert len(actual) == len(expected) == 2
    for packed, scalar in zip(actual, expected):
        torch.testing.assert_close(torch.tensor(packed["replica_losses"]),
                                   torch.tensor(scalar["replica_losses"]), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(packed["effective_weights"], scalar["effective_weights"],
                                   atol=1e-5, rtol=1e-5)
        assert packed["actual_initialization_seed"] == protocol.seed
        assert packed["solver_protocol_seed"] == protocol.seed


def test_candidate_seeds_match_fresh_independent_fits_even_for_identical_dense_masks() -> None:
    task = _task()
    masks = torch.ones(2, 784, 3)
    protocol = InnerProtocol(steps=2, replicas=2, lr=.002, l2=.0001,
                             checkpoint_every=1, seed=111, metric="nmse")
    seeds = [521, 522]
    packed = fit_deepsets_batch(masks, task, protocol, initialization_seeds=seeds)
    for mask, seed, result in zip(masks, seeds, packed):
        scalar = measure_mask(mask, task, protocol, initialization_seed=seed)
        torch.testing.assert_close(result["effective_weights"], scalar["effective_weights"],
                                   atol=1e-5, rtol=1e-5)
        assert result["actual_initialization_seed"] == seed
        assert result["protocol_id"] == protocol.fingerprint
    assert not torch.equal(packed[0]["effective_weights"], packed[1]["effective_weights"])


def test_packed_results_keep_individual_state_history_and_adam_moments() -> None:
    task, masks = _task(), _masks()
    protocol = InnerProtocol(steps=2, replicas=2, lr=.002, l2=.0001,
                             checkpoint_every=1, seed=212, metric="nmse")
    results = fit_deepsets_batch(masks, task, protocol)
    for result in results:
        assert result["state_dict"]["weight"].shape == (1, 2, 784, 3)
        assert result["history"]["queryNMSE"].shape == (3, 1, 2)
        moments = result["optimizer_state"]["state"][0]["exp_avg"]
        assert moments.shape == (1, 2, 784, 3)
        assert result["optimizer_state"]["state"][0]["step"].ndim == 0


def test_packed_results_are_individually_replay_validated_artifacts() -> None:
    task, masks = _task(), _masks()
    protocol = InnerProtocol(steps=2, replicas=2, lr=.002, l2=.0001,
                             checkpoint_every=1, seed=313, metric="nmse")
    replay = RealReplay(protocol)
    with TemporaryDirectory() as temporary:
        for index, (mask, result) in enumerate(zip(masks, fit_deepsets_batch(masks, task, protocol))):
            artifact = Path(temporary) / f"child_{index}.pt"
            torch.save({"result": result, "mask_key": tensor_hash(mask),
                        "task_fingerprint": task.fingerprint}, artifact)
            replay.append(mask, task, result, origin="test", artifact_path=artifact)
        replay.validate()
