"""Unit tests for amortized latent adapters."""

import unittest

import torch

from evaluation.latent_adapter import (
    ConstantAdapter, LinearAdapter, ResidualMLPAdapter, fixed_permutation,
    mask_metrics, project_ball, train_adapter,
)


class TinyDecoder(torch.nn.Module):
    latent_dim = 4
    mask_dim = 64
    cond_dim = 0

    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.layer = torch.nn.Linear(self.latent_dim, self.mask_dim)

    def decode(self, z, condition):
        return self.layer(z)


class LatentAdapterTests(unittest.TestCase):
    def test_all_adapters_start_as_identity_or_constant(self):
        z = torch.randn(5, 4)
        self.assertTrue(torch.equal(ConstantAdapter(4)(z), torch.zeros_like(z)))
        self.assertTrue(torch.equal(LinearAdapter(4)(z), z))
        self.assertTrue(torch.equal(ResidualMLPAdapter(4)(z), z))

    def test_projection_is_differentiable_and_bounded(self):
        z = torch.tensor([[3., 4.], [.1, .2]], requires_grad=True)
        projected = project_ball(z, 2.)
        self.assertTrue((projected.norm(dim=1) <= 2. + 1e-6).all())
        projected.square().sum().backward()
        self.assertIsNotNone(z.grad)

    def test_validation_global_permutation_generalizes(self):
        reference = torch.randn(7, 3, 4)
        permutation = torch.tensor([2, 0, 3, 1])
        other = reference[:, :, permutation]
        order = fixed_permutation(reference, other)
        self.assertTrue(torch.equal(other.index_select(-1, order), reference))

    def test_metrics_detect_exact_match_after_per_example_alignment(self):
        source = torch.tensor([[[1., 0.], [0., 1.]]])
        target = source[:, :, [1, 0]]
        metrics = mask_metrics(source, source, target, target)
        self.assertEqual(metrics["hard_exact"], 1.)
        self.assertEqual(metrics["hard_iou"], 1.)

    def test_training_changes_only_adapter(self):
        source = TinyDecoder(1).eval().requires_grad_(False)
        target = TinyDecoder(2).eval().requires_grad_(False)
        before_source = [value.clone() for value in source.state_dict().values()]
        before_target = [value.clone() for value in target.state_dict().values()]
        adapter = LinearAdapter(4)
        before_adapter = [value.clone() for value in adapter.state_dict().values()]
        z_train = torch.randn(8, 4)
        z_val = torch.randn(4, 4)
        trained, _ = train_adapter(
            adapter, source, target, z_train, z_val, steps=2, batch_size=4,
            eval_every=1, lr=1e-2, radius=3., k=32, temperature=.5, seed=3)
        self.assertTrue(any(not torch.equal(a, b) for a, b in
                            zip(before_adapter, trained.state_dict().values())))
        self.assertTrue(all(torch.equal(a, b) for a, b in
                            zip(before_source, source.state_dict().values())))
        self.assertTrue(all(torch.equal(a, b) for a, b in
                            zip(before_target, target.state_dict().values())))
        self.assertTrue(all(parameter.grad is None for parameter in source.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in target.parameters()))


if __name__ == "__main__":
    unittest.main()
