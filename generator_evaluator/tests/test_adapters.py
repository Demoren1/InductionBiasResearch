import dataclasses
import json

import torch

from generator_evaluator.adapters import (
    _pattern_table,
    build_pattern_fixture,
    make_pattern_test_tasks,
    measure_mask,
)
from generator_evaluator.data import InnerProtocol, TaskData


def _fixture():
    return build_pattern_fixture(seed=71, bank_steps=2, teacher_count=3,
                                 support_count=8, query_count=8, k=8)


def test_pattern_fixture_separates_bank_probe_and_evaluator_ids() -> None:
    bank, tasks, spec = _fixture()
    assert bank.tokens.shape[:3] == (1, 3, 8)
    assert bank.baseline_mask.sum().item() == 8
    assert bank.masks[0].sum().item() == 8
    assert bank.masks[-1].sum().item() == 88
    assert len(bank.provenance["teacher_density_counts"]) >= 3
    assert bank.quality is None
    json.dumps(bank.provenance)
    assert len(bank.states) == 3
    assert len([task for task in tasks if task.split == "train"]) == 10
    assert len([task for task in tasks if task.split == "validation"]) == 2
    bank_ids = set(bank.provenance["reserved_bank_ids"])
    probe_ids = set(bank.provenance["probe_ids"])
    assert probe_ids <= bank_ids
    for task in tasks:
        assert bank_ids.isdisjoint(task.support_ids.tolist())
        assert bank_ids.isdisjoint(task.query_ids.tolist())
        assert set(task.support_ids.tolist()).isdisjoint(task.query_ids.tolist())
    assert len(spec["patterns"]) == 4


def test_pattern_profiles_are_full_probe_contributions_and_baseline_is_aligned() -> None:
    bank, _, _ = _fixture()
    state = bank.states[0]["state_dict"]
    probe_ids = torch.tensor(bank.provenance["probe_ids"])
    _, x = _pattern_table()
    mask = bank.masks[0]
    pre = x[probe_ids] @ (state["w"] * mask) + state["b"]
    q = (x[probe_ids, :, None] * (state["w"] * mask)[None] * state["a"][None, None]
         * (pre > 0)[:, None])
    torch.testing.assert_close(bank.states[0]["q_signed"], q.mean(0))
    torch.testing.assert_close(bank.states[0]["q_abs"], q.abs().mean(0))
    torch.testing.assert_close(bank.states[0]["q_rms"], q.square().mean(0).sqrt())
    assert bank.diagnostics["aligned_q_abs"].shape == (3, 11, 8)
    assert bank.baseline_mask.sum().item() == 8


def test_pattern_measurement_uses_terminal_query_label_but_not_query_gradients() -> None:
    bank, tasks, _ = _fixture()
    task = next(task for task in tasks if task.split == "train")
    protocol = InnerProtocol(steps=3, replicas=2, lr=.01, checkpoint_every=1, seed=12)
    original = measure_mask(bank.baseline_mask, task, protocol)
    altered = dataclasses.replace(task, y_query=1.0 - task.y_query)
    changed = measure_mask(bank.baseline_mask, altered, protocol)
    assert original["label_source"] == "fresh_terminal_query"
    assert original["fixed_horizon"] is True
    assert original["protocol_id"] == protocol.fingerprint
    assert original["replica_losses"] != changed["replica_losses"]
    for before, after in zip(original["state_dict"], changed["state_dict"]):
        for name in before:
            torch.testing.assert_close(before[name], after[name])
    assert len(original["optimizer_state"]) == protocol.replicas
    assert len(original["history"]) == protocol.replicas

    fresh = measure_mask(bank.baseline_mask, task, protocol, initialization_seed=991)
    assert fresh["protocol_id"] == protocol.fingerprint
    assert fresh["actual_initialization_seed"] == 991
    assert fresh["seeds"] == [991, 10_998]


def test_pattern_test_materialization_is_task_held_out() -> None:
    _, tasks, spec = _fixture()
    tests = make_pattern_test_tasks(spec)
    observed = {task.task_id for task in tasks}
    assert tests and all(task.split == "test" and task.task_id not in observed for task in tests)
    assert all(torch.isin(task.support_ids, task.query_ids).sum() == 0 for task in tests)
    heldout_ids = set(spec["partitions"]["test"])
    assert all(set(task.query_ids.tolist()) <= heldout_ids for task in tests)
    assert all(task.provenance["query_sampling"] == "uniform_heldout" for task in tests)
