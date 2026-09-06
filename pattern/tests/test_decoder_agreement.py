"""Tests for agreement-only decoder latent optimization."""

import unittest

import torch

from evaluation.decoder_agreement import (align_columns, hard_topk,
                                          optimize_agreement, soft_topk)
from models.cvae import CVAE


class DecoderAgreementTests(unittest.TestCase):
    def test_soft_topk_cardinality_and_implicit_gradient(self):
        x = torch.tensor([[1.1, -0.2, .7, -1.4, .3]], dtype=torch.double,
                         requires_grad=True)
        y = soft_topk(x, 2, .6)
        self.assertTrue(torch.allclose(y.sum(-1), torch.tensor([2.], dtype=torch.double), atol=1e-10))
        loss = (y * torch.tensor([[.2, -.3, 1.4, .8, -.6]], dtype=torch.double)).sum()
        loss.backward()
        analytic = x.grad.detach().clone()
        eps = 1e-5
        numeric = []
        for idx in range(x.numel()):
            plus, minus = x.detach().clone(), x.detach().clone()
            plus[0, idx] += eps
            minus[0, idx] -= eps
            weights = torch.tensor([[.2, -.3, 1.4, .8, -.6]], dtype=torch.double)
            numeric.append(((soft_topk(plus, 2, .6) * weights).sum() -
                            (soft_topk(minus, 2, .6) * weights).sum()) / (2 * eps))
        self.assertTrue(torch.allclose(analytic.flatten(), torch.stack(numeric), atol=2e-5, rtol=2e-4))

    def test_hungarian_alignment_is_permutation_invariant_but_not_vacuous(self):
        reference = torch.tensor([[[1., 0., 2.], [3., 4., 5.]]])
        other = reference[:, :, [2, 0, 1]].clone().requires_grad_()
        aligned = align_columns(reference, other)
        self.assertTrue(torch.equal(aligned, reference))
        aligned.square().sum().backward()
        self.assertGreater(other.grad.abs().sum().item(), 0.)
        mismatch = reference + 2.
        self.assertGreater((reference - align_columns(reference, mismatch)).square().mean().item(), 0.)

    def test_optimization_freezes_decoder_and_preserves_hard_cardinality(self):
        torch.manual_seed(19)
        model1, model2 = CVAE(mask_dim=64, latent_dim=4, hidden=12), CVAE(mask_dim=64, latent_dim=4, hidden=12)
        before1 = [p.detach().clone() for p in model1.parameters()]
        before2 = [p.detach().clone() for p in model2.parameters()]
        result = optimize_agreement(model1, model2, n_starts=3, steps=12,
                                    lr=.04, seed=23, temperature=.5,
                                    radius=3., device="cpu")
        self.assertTrue(torch.all(result["per_start_final_loss"] <= result["per_start_initial_loss"] + 1e-7))
        self.assertTrue(torch.equal(result["final_masks1"].sum((1, 2)), torch.full((3,), 32.)))
        self.assertTrue(torch.equal(result["final_masks2"].sum((1, 2)), torch.full((3,), 32.)))
        self.assertGreater((result["final_z1"] - result["initial_z1"]).abs().sum().item() +
                           (result["final_z2"] - result["initial_z2"]).abs().sum().item(), 0.)
        for before, after in zip(before1, model1.parameters()):
            self.assertTrue(torch.equal(before, after.detach()))
        for before, after in zip(before2, model2.parameters()):
            self.assertTrue(torch.equal(before, after.detach()))
        self.assertTrue(all(not p.requires_grad for p in model1.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model2.parameters()))
        # The decoder is frozen, but agreement still has a live path to both
        # independent latent vectors through the detached Hungarian gather.
        z1 = result["final_z1"].clone().requires_grad_()
        z2 = result["final_z2"].clone().requires_grad_()
        s1 = soft_topk(model1.decode(z1, z1.new_zeros(3, 0)), 32, .5).reshape(3, 8, 8)
        s2 = soft_topk(model2.decode(z2, z2.new_zeros(3, 0)), 32, .5).reshape(3, 8, 8)
        (s1 - align_columns(s1, s2)).square().mean().backward()
        self.assertGreater(z1.grad.abs().sum().item(), 0.)
        self.assertGreater(z2.grad.abs().sum().item(), 0.)
        self.assertTrue(all(p.grad is None for p in model1.parameters()))
        self.assertTrue(all(p.grad is None for p in model2.parameters()))

    def test_hard_topk_flat_masks(self):
        mask = hard_topk(torch.randn(4, 64), 32)
        self.assertTrue(torch.equal(mask.sum(-1), torch.full((4,), 32.)))


if __name__ == "__main__":
    unittest.main()
