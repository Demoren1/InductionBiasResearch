"""Focused solver checks for the shared packed pattern fitting engine."""
from __future__ import annotations

import dataclasses
import unittest

import torch
from torch.nn import functional as F

from generator_evaluator.adapters import build_pattern_fixture
from generator_evaluator.data import InnerProtocol
from generator_evaluator.pattern_fit import PatternFitEngine
from pattern.task_quality.meta import child_logits, init_child


def _inputs():
    bank, tasks, _ = build_pattern_fixture(seed=117, bank_steps=1, teacher_count=3,
                                           support_count=8, query_count=8, k=8)
    return bank.masks[:2], [task for task in tasks if task.split == "train"][:2]


def _scalar(mask, task, protocol):
    """Literal pre-vectorisation reference for a single independent child."""
    states, losses = [], []
    for replica in range(protocol.replicas):
        params = init_child(protocol.seed + 10_007 * replica)
        optimizer = torch.optim.Adam(params.values(), lr=protocol.lr)
        rng = torch.Generator().manual_seed(protocol.seed + 73_003 * (replica + 1))
        for step in range(1, protocol.steps + 1):
            for group in optimizer.param_groups:
                group["lr"] = protocol.lr * max(protocol.lr_floor, .5 ** ((step - 1) // protocol.lr_decay_every))
            if protocol.batch_size is None or protocol.batch_size >= len(task.x_support):
                x, y = task.x_support, task.y_support
            else:
                rows = torch.randint(len(task.x_support), (protocol.batch_size,), generator=rng)
                x, y = task.x_support[rows], task.y_support[rows]
            penalty = .5 * protocol.l2 * sum(
                (value * mask if name == "w" else value).square().sum() for name, value in params.items())
            loss = F.binary_cross_entropy_with_logits(child_logits(x, mask, params), y) + penalty
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        states.append({name: value.detach().clone() for name, value in params.items()})
        losses.append(float(F.binary_cross_entropy_with_logits(child_logits(task.x_query, mask, params), task.y_query)))
    return states, losses


class PatternFitEngineTests(unittest.TestCase):
    def test_packed_full_batch_matches_scalar_reference(self) -> None:
        masks, tasks = _inputs()
        protocol = InnerProtocol(steps=3, replicas=2, lr=.01, l2=.03, checkpoint_every=1, seed=37)
        actual = PatternFitEngine(protocol).fit(masks, tasks)
        for mask, task, fitted in zip(masks, tasks, actual):
            expected_states, expected_losses = _scalar(mask, task, protocol)
            for expected, observed in zip(expected_states, fitted["state_dict"]):
                for key in expected:
                    torch.testing.assert_close(observed[key], expected[key], rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(torch.tensor(fitted["replica_losses"]), torch.tensor(expected_losses),
                                       rtol=2e-5, atol=2e-6)

    def test_packed_minibatches_match_scalar_and_keep_optimizer_rows_independent(self) -> None:
        masks, tasks = _inputs()
        protocol = InnerProtocol(steps=3, replicas=2, lr=.01, l2=.01, batch_size=4,
                                 checkpoint_every=1, seed=53)
        actual = PatternFitEngine(protocol).fit(masks, tasks)
        expected_states, _ = _scalar(masks[0], tasks[0], protocol)
        for expected, observed in zip(expected_states, actual[0]["state_dict"]):
            for key in expected:
                torch.testing.assert_close(observed[key], expected[key], rtol=2e-5, atol=2e-6)
        first, second = actual[0]["optimizer_state"]
        self.assertIsNot(first["state"], second["state"])
        self.assertEqual(len(first["state"]), 4)
        self.assertTrue(all(value.device.type == "cpu" for item in first["state"].values()
                            for value in item.values() if torch.is_tensor(value)))

    def test_query_labels_change_only_terminal_measurement(self) -> None:
        masks, tasks = _inputs()
        protocol = InnerProtocol(steps=2, replicas=2, batch_size=4, checkpoint_every=1, seed=71)
        changed = [tasks[0], dataclasses.replace(tasks[1], y_query=1 - tasks[1].y_query)]
        before, after = PatternFitEngine(protocol).fit(masks, tasks), PatternFitEngine(protocol).fit(masks, changed)
        for left, right in zip(before, after):
            for before_state, after_state in zip(left["state_dict"], right["state_dict"]):
                for key in before_state:
                    torch.testing.assert_close(before_state[key], after_state[key])
        self.assertNotEqual(before[1]["replica_losses"], after[1]["replica_losses"])


if __name__ == "__main__":
    unittest.main()
