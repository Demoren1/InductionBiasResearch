"""Unit tests for the motif oracle's constrained latent optimization."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.oracle_ideal import optimize_ideal


class ConditionedIdentity(torch.nn.Module):
    cond_dim = 1

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(16))
        self.conditions: list[torch.Tensor] = []

    def decode(self, z, condition):
        self.conditions.append(condition.detach().clone())
        # Keep c in the computational path, while the reachable support is
        # known exactly and remains independent of the condition value.
        return self.linear(z) + condition[:, :1] * 0


def test_oracle_finds_known_reachable_mask_without_changing_decoder():
    torch.manual_seed(19)
    model = ConditionedIdentity()
    target = torch.eye(4)
    initial_z = torch.randn(3, 16)
    before = model.linear.weight.detach().clone()
    result = optimize_ideal(model, initial_z, target, torch.tensor([[3 / 7]]),
                            steps=80, lr=.15, radius=8., temperature=.5)
    assert torch.equal(before, model.linear.weight)
    assert model.linear.weight.grad is None
    assert result["decoder_unchanged"]
    assert (result["best_hard"]["iou"] == 1).all()
    assert (result["best_soft"]["loss"] <= result["initial"]["loss"]).all()
    assert (result["best_hard"]["hard"].sum((1, 2)) == 4).all()
    assert (result["best_soft"]["z"].norm(dim=1) <= 8.00001).all()


def test_condition_is_repeated_for_every_start_and_step_zero_is_witness():
    model = ConditionedIdentity()
    target = torch.eye(4)
    # These logits already select a permuted version of target's columns.
    logits = ((target[:, [2, 0, 3, 1]] * 2 - 1) * .5).flatten()[None]
    condition = torch.tensor([[5 / 7]])
    result = optimize_ideal(model, logits, target, condition, steps=0, radius=8.)
    assert result["initial"]["iou"].item() == 1.
    assert torch.equal(result["initial"]["z"], result["best_hard"]["z"])
    assert result["best_hard_steps"].item() == 0
    assert model.conditions
    for seen in model.conditions:
        assert seen.shape == (1, 1)
        assert torch.equal(seen, condition)
