"""Direct tests for decoder-pair latent agreement; no GPU/run artifacts needed."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.decoder_pair_search import (_repeat_proposals_across_gaps, optimize_agreement,
                                            random_pair_search)


class TinyScalarDecoder(torch.nn.Module):
    cond_dim = 1
    latent_dim = 32

    def __init__(self, seed: int):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.linear = torch.nn.Linear(33, 256, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.randn(256, 33, generator=generator) * .1)

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.cat([z, c], dim=1))


def test_pair_optimizer_selects_soft_agreement_and_keeps_both_decoders_frozen():
    one, two = TinyScalarDecoder(1), TinyScalarDecoder(2)
    start1 = torch.randn(3, 32, generator=torch.Generator().manual_seed(3))
    start2 = torch.randn(3, 32, generator=torch.Generator().manual_seed(4))
    condition = torch.tensor([[0.], [.5], [1.]])
    before1, before2 = one.linear.weight.detach().clone(), two.linear.weight.detach().clone()
    result = optimize_agreement(one, two, start1, start2, condition,
                                 steps=4, lr=.03, radius=2., temperature=.5)
    assert result["decoder_unchanged"]
    assert torch.equal(one.linear.weight, before1)
    assert torch.equal(two.linear.weight, before2)
    assert one.linear.weight.grad is None and two.linear.weight.grad is None
    assert (result["best"]["soft_pair_loss"] <= result["prior"]["soft_pair_loss"] + 1e-7).all()
    assert (result["best"]["norm1"] <= 2.00001).all()
    assert (result["best"]["norm2"] <= 2.00001).all()
    assert (result["best"]["mask1"].sum(1) == 96).all()
    assert (result["best"]["mask2"].sum(1) == 96).all()


def test_random_pair_search_uses_per_row_best_independent_proposals():
    one, two = TinyScalarDecoder(11), TinyScalarDecoder(12)
    proposals1 = torch.randn(5, 2, 32, generator=torch.Generator().manual_seed(13))
    proposals2 = torch.randn(5, 2, 32, generator=torch.Generator().manual_seed(14))
    result = random_pair_search(one, two, proposals1, proposals2, torch.tensor([[.25], [.75]]),
                                radius=2., temperature=.5)
    assert result["proposal_index"].shape == (2,)
    assert ((result["proposal_index"] >= 0) & (result["proposal_index"] < 5)).all()
    assert (result["norm1"] <= 2.00001).all() and (result["norm2"] <= 2.00001).all()
    assert (result["mask1"].sum(1) == 96).all()


def test_random_proposals_are_identical_for_a_start_at_each_gap():
    proposals = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    repeated = _repeat_proposals_across_gaps(proposals, 5).reshape(2, 5, 3, 4)
    for gap in range(5):
        assert torch.equal(repeated[:, gap], proposals)
