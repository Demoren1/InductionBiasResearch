from __future__ import annotations

import math
import unittest

import torch
import torch.nn.functional as F

from meta_pattern.data import PatternTask, sample_dataset
from pattern.length_interp.bank import random_exact_k_masks, select_top_indices, train_masked
from pattern.length_interp.mlp import BatchedMaskedMLP, fit_mlp, ideal_mask


class TestBankMlp(unittest.TestCase):
    def test_ideal_mask_has_exact_cardinality_and_all_hidden_active(self):
        for k in range(3, 9):
            mask = ideal_mask("0" * k)
            self.assertEqual(tuple(mask.shape), (32, 32))
            self.assertEqual(int(mask.sum()), 32 * k)
            self.assertTrue(torch.equal(mask.sum(0), torch.full((32,), float(k))))

    def test_batched_forward_matches_individual_formula_and_has_gradients(self):
        masks = torch.stack((ideal_mask("010"), ideal_mask("101")))
        model = BatchedMaskedMLP(masks, seed=7)
        x = torch.randn(5, 32)
        actual = model(x)
        expected = []
        for m in range(2):
            h = F.relu(x @ (model.w1[m] * masks[m]) + model.b1[m])
            expected.append(h @ model.w2[m] + model.b2[m])
        self.assertTrue(torch.allclose(actual, torch.stack(expected, dim=1)))
        actual.sum().backward()
        self.assertIsNotNone(model.w1.grad)
        self.assertGreater(float(model.w1.grad.abs().sum()), 0.0)

    def test_selection_is_lowest_bce_with_ceil_top_fraction(self):
        losses = torch.tensor([0.8, 0.2, 0.2, 0.4, 0.1])
        index = select_top_indices(losses, 0.4)
        self.assertEqual(index.tolist(), [4, 1])
        self.assertEqual(select_top_indices(losses, 0.01).tolist(), [4])

    def test_random_masks_are_seeded_exact_k_and_not_shared(self):
        first = random_exact_k_masks(4, seq_len=32, hidden=32, k_active=128, seed=18)
        second = random_exact_k_masks(4, seq_len=32, hidden=32, k_active=128, seed=18)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first.sum(dim=(1, 2)), torch.full((4,), 128.0)))
        self.assertFalse(torch.equal(first[0], first[1]))

    def test_random_mask_seed_changes_the_bank(self):
        first = random_exact_k_masks(3, seq_len=32, hidden=32, k_active=96, seed=1)
        second = random_exact_k_masks(3, seq_len=32, hidden=32, k_active=96, seed=2)
        self.assertFalse(torch.equal(first, second))

    def test_deterministic_fit_and_sum_objective(self):
        masks = torch.stack([ideal_mask("010")] * 3)
        x = torch.randn(24, 32)
        y = torch.randint(0, 2, (24,), dtype=torch.float32)
        first, second = BatchedMaskedMLP(masks, 21), BatchedMaskedMLP(masks, 21)
        fit_mlp(first, x, y, steps=3, batch_size=8, lr=1e-3, seed=31)
        fit_mlp(second, x, y, steps=3, batch_size=8, lr=1e-3, seed=31)
        for a, b in zip(first.parameters(), second.parameters()):
            self.assertTrue(torch.equal(a, b))

    def test_global_support_and_query_ids_are_disjoint(self):
        task = PatternTask("0110")
        support = sample_dataset(task, 256, seed=8, split="support")
        query = sample_dataset(task, 256, seed=9, split="query")
        self.assertFalse(set(support["ids"].tolist()) & set(query["ids"].tolist()))

    def test_train_masked_smoke_uses_query_for_metrics(self):
        masks = torch.stack([ideal_mask("0101")] * 2)
        query = sample_dataset(PatternTask("0101"), 64, seed=91, split="query")
        model, result = train_masked(masks, "0101", steps=2, batch_size=8, lr=1e-3,
                                     seed=3, device="cpu", support_pool_size=128,
                                     validation_data=query)
        self.assertEqual(tuple(result["validation_bce"].shape), (2,))
        self.assertEqual(result["query_size"], 64)
        self.assertEqual(int(model.masks[0].sum()), 128)


if __name__ == "__main__":
    unittest.main()
