import unittest

import torch
from torch import nn

from generator_evaluator.adapters import FunctionalBank
from generator_evaluator.cooperative_policy import DensityConditionedGenerator, align_elite_to_logits
from generator_evaluator.generator_objectives import (
    joint_mask_agreement,
    reconstruct_bank_masks,
)


class TableGenerator(nn.Module):
    """Small profile-indexed generator for deterministic objective tests."""

    def __init__(self) -> None:
        super().__init__()
        self.features, self.hidden, self.noise_dim, self.token_dim = 2, 2, 2, 1
        self.table = nn.Parameter(torch.zeros(2, self.features, self.hidden))
        self.last_indices = None
        self.last_density = None
        self.last_tokens = None

    def forward(self, tokens, noise, quality=None, *, density=None):
        indices = tokens[:, 0, 0, 0].round().long()
        self.last_indices = indices.detach().clone()
        self.last_density = density.detach().clone()
        self.last_tokens = tokens.detach().clone()
        return self.table.index_select(0, indices)


class FixedLogitGenerator(nn.Module):
    def __init__(self, values: torch.Tensor) -> None:
        super().__init__()
        self.features, self.hidden, self.noise_dim, self.token_dim = *values.shape, 2, 1
        self.logits = nn.Parameter(values.clone())
        self.last_noise = None

    def forward(self, tokens, noise, quality=None, *, density=None):
        self.last_noise = noise.detach().clone()
        return self.logits.unsqueeze(0).expand(noise.shape[0], -1, -1)


def make_bank(features: int, hidden: int, teachers: int = 2) -> FunctionalBank:
    tokens = torch.randn(1, teachers, hidden, 1, generator=torch.Generator().manual_seed(5))
    masks = torch.zeros(teachers, features, hidden)
    masks[:, 0, 0] = 1
    return FunctionalBank(tokens, None, masks, masks[0])


class GeneratorObjectiveTests(unittest.TestCase):
    def test_reconstruction_uses_per_row_density_reduces_loss_and_preserves_bank(self) -> None:
        tokens = torch.zeros(1, 2, 2, 1)
        tokens[0, 1, :, 0] = 1.0
        masks = torch.tensor([
            [[1.0, 0.0], [0.0, 0.0]],  # sparse: density 1/4
            [[1.0, 1.0], [1.0, 1.0]],  # dense: density 1
        ])
        bank = FunctionalBank(tokens, None, masks, masks[0])
        saved_tokens, saved_masks, saved_baseline = (
            bank.tokens.clone(), bank.masks.clone(), bank.baseline_mask.clone()
        )
        model = TableGenerator()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.08)

        before_params = model.table.detach().clone()
        first = reconstruct_bank_masks(
            model, bank, optimizer, rng=torch.Generator().manual_seed(17),
            batch_size=64, weight=0.0,
        )
        expected_density = bank.masks.index_select(0, model.last_indices).sum((1, 2)) / 4
        torch.testing.assert_close(model.last_density, expected_density)
        self.assertEqual(tuple(model.last_tokens.shape), (64, 1, 2, 1))
        self.assertEqual(set(model.last_indices.tolist()), {0, 1})
        self.assertTrue(torch.equal(before_params, model.table))
        self.assertGreaterEqual(first["reconstruction_overlap"], 0.0)
        self.assertLessEqual(first["reconstruction_overlap"], 1.0)

        training_rng = torch.Generator().manual_seed(21)
        for _ in range(24):
            reconstruct_bank_masks(model, bank, optimizer, rng=training_rng, batch_size=32)
        last = reconstruct_bank_masks(
            model, bank, optimizer, rng=torch.Generator().manual_seed(17),
            batch_size=64, weight=0.0,
        )
        self.assertLess(last["reconstruction_loss"], first["reconstruction_loss"])
        self.assertTrue(torch.equal(bank.tokens, saved_tokens))
        self.assertTrue(torch.equal(bank.masks, saved_masks))
        self.assertTrue(torch.equal(bank.baseline_mask, saved_baseline))

    def test_reconstruction_recovers_synthetic_teacher_masks_from_neuron_profiles(self) -> None:
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(19)
                features, hidden = 3, 4
                # Each teacher has the same feature-wise edge totals, but a
                # different multiset of neuron profiles.  There is no shared
                # Toeplitz pattern or target-side regularizer in this task.
                patterns = [
                    [[0, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1]],
                    [[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    [[1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 0, 0]],
                    [[1, 0, 1], [0, 1, 1], [1, 1, 1], [0, 0, 0]],
                ]
                masks = torch.tensor(patterns, dtype=torch.float32).transpose(1, 2).contiguous()
                profiles = torch.randn(
                    1, len(masks), hidden, 2,
                    generator=torch.Generator().manual_seed(33),
                )
                tokens = torch.cat((profiles, masks.transpose(1, 2).unsqueeze(0)), dim=-1)
                bank = FunctionalBank(tokens, None, masks, masks[0])
                model = DensityConditionedGenerator(
                    token_dim=5, features=features, hidden=hidden,
                    width=16, heads=4, layers=1, noise_dim=2,
                )
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                rng = torch.Generator().manual_seed(24)
                for _ in range(120):
                    logs = reconstruct_bank_masks(model, bank, optimizer, rng=rng, batch_size=8)

                overlaps = []
                with torch.no_grad():
                    for row, target in enumerate(masks):
                        logits = model(
                            tokens[:, row:row + 1], torch.zeros(1, model.noise_dim),
                            density=torch.full((1,), 0.5),
                        )[0]
                        aligned = align_elite_to_logits(target, logits)
                        predicted = logits.flatten().topk(int(aligned.sum())).indices
                        overlaps.append(float(aligned.flatten()[predicted].sum() / aligned.sum()))

                self.assertLess(logs["reconstruction_loss"], 0.01)
                self.assertGreaterEqual(min(overlaps), 0.95)
        finally:
            torch.set_num_threads(previous_threads)

    def test_bank_consensus_reconstruction_recovers_data_derived_baseline(self) -> None:
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(29)
                features, hidden = 3, 4
                supports = [
                    [0, 1, 4, 6, 8],
                    [0, 2, 4, 9, 10],
                    [1, 2, 5, 6, 11],
                    [0, 2, 5, 8, 11],
                ]
                masks = torch.zeros(len(supports), features * hidden)
                for row, active in enumerate(supports):
                    masks[row, active] = 1.0
                masks = masks.reshape(-1, features, hidden)
                # This synthetic baseline is a per-edge majority of the
                # supplied teacher masks, not a hand-authored pattern.
                baseline = (masks.sum(dim=0) >= 2).float()
                profiles = torch.randn(
                    1, len(masks), hidden, 2,
                    generator=torch.Generator().manual_seed(37),
                )
                tokens = torch.cat((profiles, masks.transpose(1, 2).unsqueeze(0)), dim=-1)
                bank = FunctionalBank(tokens, None, masks, baseline)
                model = DensityConditionedGenerator(
                    token_dim=5, features=features, hidden=hidden,
                    width=16, heads=4, layers=1, noise_dim=2,
                )
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                rng = torch.Generator().manual_seed(41)
                for _ in range(200):
                    logs = reconstruct_bank_masks(
                        model, bank, optimizer, rng=rng, batch_size=8,
                        bank_consensus=True,
                    )

                with torch.no_grad():
                    logits = model(
                        tokens, torch.zeros(1, model.noise_dim),
                        density=baseline.sum().reshape(1) / baseline.numel(),
                    )[0]
                    aligned = align_elite_to_logits(baseline, logits)
                    predicted = logits.flatten().topk(int(aligned.sum())).indices
                    overlap = float(aligned.flatten()[predicted].sum() / aligned.sum())

                self.assertEqual(logs["reconstruction_scope"], "bank_consensus")
                self.assertLess(logs["reconstruction_loss"], 0.01)
                self.assertGreaterEqual(overlap, 0.95)
        finally:
            torch.set_num_threads(previous_threads)

    def test_agreement_is_invariant_to_hidden_column_permutation(self) -> None:
        first_logits = torch.tensor([[5.0, -5.0], [5.0, -5.0], [-5.0, -5.0]])
        second_logits = first_logits.flip(1)
        models = [FixedLogitGenerator(first_logits), FixedLogitGenerator(second_logits)]
        optimizers = [torch.optim.SGD(model.parameters(), lr=0.1) for model in models]
        banks = [make_bank(3, 2), make_bank(3, 2)]

        logs = joint_mask_agreement(
            models, banks, optimizers, 2, rng=torch.Generator().manual_seed(31), sample_count=2,
        )
        self.assertAlmostEqual(logs["direct_agreement_loss"], 0.0, places=7)
        self.assertAlmostEqual(logs["direct_agreement_overlap"], 1.0, places=7)
        torch.testing.assert_close(models[0].last_noise, models[1].last_noise)

    def test_different_hard_masks_disagree_and_both_generators_get_gradients(self) -> None:
        first_logits = torch.tensor([[5.0, -5.0], [5.0, -5.0], [-5.0, -5.0]])
        second_logits = torch.tensor([[5.0, -5.0], [-5.0, -5.0], [-5.0, 5.0]])
        models = [FixedLogitGenerator(first_logits), FixedLogitGenerator(second_logits)]
        optimizers = [torch.optim.SGD(model.parameters(), lr=0.02) for model in models]
        banks = [make_bank(3, 2), make_bank(3, 2)]
        snapshots = [model.logits.detach().clone() for model in models]

        skipped = joint_mask_agreement(
            models, banks, optimizers, 2, rng=torch.Generator().manual_seed(32),
            weight=0.0, sample_count=1,
        )
        self.assertEqual(skipped["direct_agreement_loss"], 0.0)
        self.assertAlmostEqual(skipped["direct_agreement_overlap"], 0.5, places=7)
        self.assertTrue(all(torch.equal(old, model.logits) for old, model in zip(snapshots, models)))

        # Use fresh optimizers/models so the zero-weight pass cannot affect the
        # gradients or optimizer state in the training assertion.
        models = [FixedLogitGenerator(first_logits), FixedLogitGenerator(second_logits)]
        optimizers = [torch.optim.SGD(model.parameters(), lr=0.02) for model in models]
        trained = joint_mask_agreement(
            models, banks, optimizers, 2, rng=torch.Generator().manual_seed(32),
            weight=0.5, sample_count=1,
        )
        self.assertAlmostEqual(trained["direct_agreement_loss"], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(trained["direct_agreement_overlap"], 0.5, places=7)
        for model in models:
            self.assertIsNotNone(model.logits.grad)
            self.assertGreater(float(model.logits.grad.abs().sum()), 0.0)

    def test_three_generators_average_all_pairs_and_each_gets_gradients(self) -> None:
        first_logits = torch.tensor([[6.0, -6.0], [5.0, -5.0], [-6.0, -7.0]])
        second_logits = first_logits.flip(1)
        third_logits = torch.tensor([[6.0, -6.0], [-6.0, -7.0], [5.0, -5.0]])
        banks = [make_bank(3, 2) for _ in range(3)]
        models = [FixedLogitGenerator(values) for values in
                  (first_logits, second_logits, third_logits)]
        optimizers = [torch.optim.SGD(model.parameters(), lr=0.02) for model in models]

        measured = joint_mask_agreement(
            models, banks, optimizers, 2, rng=torch.Generator().manual_seed(41),
            weight=0.1, sample_count=2,
        )
        # A/B agree after swapping hidden columns; A/C and B/C each disagree
        # on two of six exact-K coordinates, so the three-pair mean is 2/9.
        self.assertAlmostEqual(measured["direct_agreement_loss"], 2.0 / 9.0, places=7)
        self.assertAlmostEqual(measured["direct_agreement_overlap"], 2.0 / 3.0, places=7)
        for model in models[1:]:
            torch.testing.assert_close(models[0].last_noise, model.last_noise)

        reordered_models = [FixedLogitGenerator(values) for values in
                            (third_logits, first_logits, second_logits)]
        reordered_optimizers = [torch.optim.SGD(model.parameters(), lr=0.02)
                                for model in reordered_models]
        reordered = joint_mask_agreement(
            reordered_models, [banks[2], banks[0], banks[1]], reordered_optimizers, 2,
            rng=torch.Generator().manual_seed(41), weight=0.1, sample_count=2,
        )
        self.assertAlmostEqual(reordered["direct_agreement_loss"],
                               measured["direct_agreement_loss"], places=7)
        self.assertAlmostEqual(reordered["direct_agreement_overlap"],
                               measured["direct_agreement_overlap"], places=7)

        trained_models = [FixedLogitGenerator(values) for values in
                          (first_logits, second_logits, third_logits)]
        trained_optimizers = [torch.optim.SGD(model.parameters(), lr=0.02)
                              for model in trained_models]
        joint_mask_agreement(
            trained_models, banks, trained_optimizers, 2,
            rng=torch.Generator().manual_seed(41), weight=0.5, sample_count=2,
        )
        for model in trained_models:
            self.assertIsNotNone(model.logits.grad)
            self.assertGreater(float(model.logits.grad.abs().sum()), 0.0)

    def test_direct_agreement_requires_at_least_two_aligned_inputs(self) -> None:
        model = FixedLogitGenerator(torch.zeros(3, 2))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        with self.assertRaises(ValueError):
            joint_mask_agreement([model], [make_bank(3, 2)], [optimizer], 2)
        with self.assertRaises(ValueError):
            joint_mask_agreement([model, model], [make_bank(3, 2)], [optimizer], 2)

    def test_invalid_budget_and_sample_count_are_rejected(self) -> None:
        models = [FixedLogitGenerator(torch.zeros(3, 2)) for _ in range(2)]
        banks = [make_bank(3, 2), make_bank(3, 2)]
        optimizers = [torch.optim.SGD(model.parameters(), lr=0.01) for model in models]
        with self.assertRaises(ValueError):
            joint_mask_agreement(models, banks, optimizers, 0)
        with self.assertRaises(ValueError):
            joint_mask_agreement(models, banks, optimizers, 2, sample_count=0)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
    def test_agreement_connects_generators_across_cuda_devices(self) -> None:
        first_values = torch.tensor([[5.0, -5.0], [5.0, -5.0], [-5.0, -5.0]])
        second_values = torch.tensor([[5.0, -5.0], [-5.0, -5.0], [-5.0, 5.0]])
        models = [FixedLogitGenerator(first_values).to("cuda:0"),
                  FixedLogitGenerator(second_values).to("cuda:1")]
        optimizers = [torch.optim.SGD(model.parameters(), lr=0.02) for model in models]
        shared_noise = torch.zeros(2, 2, device="cuda:0", requires_grad=True)
        logs = joint_mask_agreement(
            models, [make_bank(3, 2), make_bank(3, 2)], optimizers, 2,
            rng=torch.Generator().manual_seed(53), weight=0.5, sample_count=2,
            shared_noise=shared_noise,
        )
        self.assertGreater(logs["direct_agreement_loss"], 0.0)
        for model in models:
            self.assertIsNotNone(model.logits.grad)


if __name__ == "__main__":
    unittest.main()
