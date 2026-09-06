"""Focused tests for the full-U parameter-subspace implementation."""

from __future__ import annotations

import itertools
import unittest

import torch
import torch.nn.functional as F

from meta_pattern.models import (
    FullUGenerator,
    LearnedUTable,
    RandomU,
    adapt_v,
    forward_with_u,
    ideal_solution,
    init_v,
)


class FullUModelTests(unittest.TestCase):
    def test_analytic_detector_handles_boundaries_multiple_and_near_misses(self) -> None:
        pattern, sequence_length = "010", 8
        u, v = ideal_solution(pattern, seq_len=sequence_length, hidden=6, rank1=5, rank2=2)
        all_bits = torch.tensor(list(itertools.product((0.0, 1.0), repeat=sequence_length)))
        x = all_bits * 2 - 1
        logits = forward_with_u(x, u, v, seq_len=sequence_length, hidden=6)
        expected = torch.tensor([
            any(row[start:start + len(pattern)].tolist() == [0.0, 1.0, 0.0]
                for start in range(sequence_length - len(pattern) + 1))
            for row in all_bits
        ])
        self.assertTrue(torch.equal(logits.gt(0), expected))

        boundary_and_multiple = torch.tensor([
            [-1, 1, -1, -1, -1, -1, -1, -1],  # start = 0
            [-1, -1, -1, -1, -1, -1, 1, -1],  # final window
            [-1, 1, -1, 1, -1, 1, -1, 1],      # several matches
            [-1, 1, 1, -1, 1, 1, -1, 1],       # near misses only
        ], dtype=torch.float)
        expected_four = torch.tensor([True, True, True, False])
        self.assertTrue(torch.equal(
            forward_with_u(boundary_and_multiple, u, v, sequence_length, 6).gt(0), expected_four
        ))

    def test_shapes_normalization_and_seeded_controls(self) -> None:
        generator = FullUGenerator()
        u1, u2 = generator(5)
        self.assertEqual(u1.shape, (33 * 32, 16))
        self.assertEqual(u2.shape, (33, 4))
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(u1, dim=0), torch.ones(16)))
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(u2, dim=0), torch.ones(4)))
        self.assertTrue(torch.equal(u1, generator(torch.tensor(5))[0]))

        table = LearnedUTable(seq_len=8, hidden=6, rank1=4, rank2=3, length_min=3, length_max=4)
        self.assertEqual(table(3)[0].shape, (54, 4))
        random_u = RandomU(seq_len=8, hidden=6, rank1=4, rank2=3, length_min=3, length_max=4, seed=9)
        self.assertTrue(torch.equal(random_u(3)[0], random_u(3)[0]))
        self.assertFalse(torch.equal(random_u(3)[0], random_u(4)[0]))

        torch.manual_seed(123)
        first = torch.rand(1)
        random_u(3)
        second = torch.rand(1)
        torch.manual_seed(123)
        self.assertTrue(torch.equal(first, torch.rand(1)))
        self.assertTrue(torch.equal(second, torch.rand(1)))

    def test_generator_depth_capacity_and_unconditional_control(self) -> None:
        legacy = FullUGenerator(width=64, depth=2)
        # depth=2 is the original architecture and checkpoint layout.
        self.assertEqual(
            list(legacy.state_dict()),
            [
                "network.0.weight", "network.0.bias", "network.2.weight",
                "network.2.bias", "network.4.weight", "network.4.bias",
            ],
        )
        large = FullUGenerator(width=256, depth=3)
        self.assertGreater(
            sum(parameter.numel() for parameter in large.parameters()),
            sum(parameter.numel() for parameter in legacy.parameters()),
        )

        conditional = FullUGenerator(width=256, depth=3, condition_length=True)
        unconditional = FullUGenerator(width=256, depth=3, condition_length=False)
        self.assertEqual(
            sum(parameter.numel() for parameter in conditional.parameters()),
            sum(parameter.numel() for parameter in unconditional.parameters()),
        )
        self.assertEqual(str(conditional.network), str(unconditional.network))
        u_at_three = unconditional(3)
        u_at_five = unconditional(5)
        u_at_seven = unconditional(7)
        for first, second in zip(u_at_three, u_at_five):
            self.assertTrue(torch.equal(first, second))
        for first, second in zip(u_at_five, u_at_seven):
            self.assertTrue(torch.equal(first, second))

    def test_initial_v_and_minibatch_adaptation_are_seeded(self) -> None:
        u = RandomU(seq_len=5, hidden=3, rank1=2, rank2=2, length_min=2, length_max=2)(2)
        first = init_v(u, seed=44)
        _ = torch.rand(100)
        second = init_v(u, seed=44)
        self.assertTrue(torch.equal(first[0], second[0]))
        x = torch.tensor([[1., -1., 1., -1., 1.], [-1., 1., -1., 1., -1.],
                          [1., 1., -1., -1., 1.], [-1., -1., 1., 1., -1.]])
        y = torch.tensor([1., 0., 1., 0.])
        v_a = adapt_v(u, x, y, steps=3, lr=0.1, seed=7, create_graph=False, batch_size=2)
        _ = torch.rand(100)
        v_b = adapt_v(u, x, y, steps=3, lr=0.1, seed=7, create_graph=False, batch_size=2)
        self.assertTrue(torch.equal(v_a[0], v_b[0]))
        self.assertTrue(torch.equal(v_a[1], v_b[1]))
        self.assertTrue(all(torch.isfinite(value).all() for value in (*u, *v_a)))

    def test_functional_adam_matches_torch_adam(self) -> None:
        """The functional inner loop has ordinary Adam's forward update."""
        u = RandomU(seq_len=5, hidden=3, rank1=2, rank2=2, length_min=2, length_max=2)(2)
        x = torch.tensor([
            [1., -1., 1., -1., 1.], [-1., 1., -1., 1., -1.],
            [1., 1., -1., -1., 1.], [-1., -1., 1., 1., -1.],
        ], dtype=torch.float64)
        u = tuple(value.double() for value in u)
        y = torch.tensor([1., 0., 1., 0.], dtype=torch.float64)
        kwargs = dict(lr=0.03, seed=19, steps=5, optimizer="adam", init_scale=0.07,
                      adam_beta1=0.8, adam_beta2=0.92, adam_eps=1e-7)
        adapted = adapt_v(u, x, y, create_graph=False, **kwargs)

        reference = tuple(torch.nn.Parameter(value.detach().clone())
                          for value in init_v(u, seed=19, scale=0.07))
        torch_adam = torch.optim.Adam(reference, lr=0.03, betas=(0.8, 0.92), eps=1e-7)
        for _ in range(5):
            torch_adam.zero_grad()
            loss = F.binary_cross_entropy_with_logits(forward_with_u(x, u, reference, 5, 3), y)
            loss.backward()
            torch_adam.step()
        for actual, expected in zip(adapted, reference):
            self.assertTrue(torch.allclose(actual, expected, rtol=2e-12, atol=2e-12))

    def test_full_meta_gradient_matches_finite_difference_for_sgd_and_adam(self) -> None:
        x = torch.tensor([
            [1., -1., 1., -1., 1.], [-1., 1., -1., 1., -1.],
            [1., 1., -1., -1., 1.], [-1., -1., 1., 1., -1.],
        ], dtype=torch.float64)
        y = torch.tensor([1., 0., 1., 0.], dtype=torch.float64)
        for optimizer, steps, lr in (("sgd", 10, 0.8), ("adam", 5, 0.08)):
            torch.manual_seed(8)
            generator = FullUGenerator(
                seq_len=5, hidden=3, rank1=2, rank2=2, width=7, length_min=2, length_max=2
            ).double()

            def meta_loss(detach_adapted=False) -> torch.Tensor:
                u = generator(2)
                v = adapt_v(
                    u, x, y, steps=steps, lr=lr, seed=21, create_graph=True,
                    optimizer=optimizer,
                )
                if detach_adapted:
                    v = tuple(value.detach() for value in v)
                return F.binary_cross_entropy_with_logits(
                    forward_with_u(x.flip(0), u, v, 5, 3), y.flip(0)
                )

            loss = meta_loss()
            (gradient,) = torch.autograd.grad(loss, generator.network[-1].bias)
            (partial_gradient,) = torch.autograd.grad(meta_loss(True), generator.network[-1].bias)
            # Select a coordinate where differentiating through adaptation matters:
            # this test must fail if the adapted v is accidentally detached.
            index = int((gradient - partial_gradient).abs().argmax())
            self.assertGreater(abs(float(gradient[index] - partial_gradient[index])), 1e-4)
            epsilon = 1e-5
            analytic = gradient[index].item()
            with torch.no_grad():
                generator.network[-1].bias[index].add_(epsilon)
            plus = meta_loss().item()
            with torch.no_grad():
                generator.network[-1].bias[index].sub_(2 * epsilon)
            minus = meta_loss().item()
            with torch.no_grad():
                generator.network[-1].bias[index].add_(epsilon)
            numeric = (plus - minus) / (2 * epsilon)
            self.assertGreater(abs(analytic), 1e-8)
            self.assertAlmostEqual(analytic, numeric, delta=2e-7)


if __name__ == "__main__":
    unittest.main()
