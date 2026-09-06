"""Focused checks for the matched single-VAE task-loss latent baseline."""

import unittest

import torch

from evaluation.single_z_task import optimize_task_z
from models.cvae import CVAE


class SingleZTaskTests(unittest.TestCase):
    def test_search_freezes_decoder_and_returns_sparse_bounded_masks(self):
        torch.set_num_threads(2)
        torch.manual_seed(31)
        model = CVAE(mask_dim=64, latent_dim=4, hidden=12)
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        starts = torch.tensor([[.2, -.3, .1, .4], [-.4, .2, .5, -.1], [.3, .1, -.2, -.5]])

        result = optimize_task_z(
            model, starts, "0101", "cpu", outer_steps=2, warmup_steps=2,
            grad_steps=2, z_lr=.08, radius=.7, temperature=.5, seed=17,
        )

        self.assertEqual(result["initial_z"].shape, (3, 4))
        self.assertEqual(result["final_z"].shape, (3, 4))
        self.assertEqual(result["final_soft"].shape, (3, 8, 8))
        self.assertTrue(torch.equal(result["initial_masks"].sum((1, 2)), torch.full((3,), 32.)))
        self.assertTrue(torch.equal(result["final_masks"].sum((1, 2)), torch.full((3,), 32.)))
        self.assertTrue(torch.equal(result["best_val_masks"].sum((1, 2)), torch.full((3,), 32.)))
        self.assertTrue(torch.all(result["final_z"].norm(dim=1) <= .700001))
        self.assertGreater((result["final_z"] - result["initial_z"]).abs().sum().item(), 0.)
        self.assertTrue(result["decoder_unchanged"])
        self.assertEqual(len(result["history"]), 2)
        self.assertTrue(all(isinstance(row["pre_update_val_loss"], list) for row in result["history"]))
        self.assertTrue(torch.isfinite(result["best_val_loss"]).all())
        self.assertTrue(torch.isfinite(result["final_post_update_val_loss"]).all())
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
