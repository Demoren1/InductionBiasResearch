"""Tests for synthetic pattern data."""

import unittest

import torch

import config
from data.generate import contains_pattern, make_dataset


class DataGenerationTests(unittest.TestCase):
    def test_dataset_labels_match_sequences(self) -> None:
        dataset = make_dataset("0101", n_samples=257, seed=5)
        sequences = ((dataset["x"] + 1) / 2).float()
        expected = contains_pattern(sequences, config.pattern_to_bits("0101")).float()
        self.assertTrue(torch.equal(dataset["y"], expected))

    def test_invalid_positive_fraction_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_dataset("0101", n_samples=10, seed=5, pos_fraction=1.1)


if __name__ == "__main__":
    unittest.main()
