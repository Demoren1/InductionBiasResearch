"""Small direct checks for task-loss single-CVAE latent adaptation."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.single_z_mlp import optimize_task_z


class TinyScalarDecoder(torch.nn.Module):
    cond_dim = 1
    latent_dim = 32
    mask_dim = 256

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(33, 256, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.randn(256, 33, generator=torch.Generator().manual_seed(7)) * .1)

    def condition(self, tasks, *, device=None):
        return torch.full((len(tasks), 1), .5, device=device)

    def decode(self, z, c):
        return self.linear(torch.cat([z, c], dim=1))


def test_task_loss_updates_only_z_after_detached_mask_mlp_training():
    model = TinyScalarDecoder()
    initial = torch.randn(64, 32, generator=torch.Generator().manual_seed(8))
    before = model.linear.weight.detach().clone()
    result = optimize_task_z(model, initial, "A000_B110_G05", device="cpu",
                             outer_steps=1, inner_steps=1, z_lr=.05, radius=8.)
    assert result["decoder_unchanged"]
    assert torch.equal(before, model.linear.weight)
    assert model.linear.weight.grad is None
    assert not torch.equal(result["initial_z"], result["final_z"])
    assert (result["final_z"].norm(dim=1) <= 8.00001).all()
    assert (result["final_hard"].sum(1) == 96).all()
    assert result["history"][0]["z_grad_norm"]
