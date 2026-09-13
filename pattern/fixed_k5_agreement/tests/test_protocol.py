from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from pattern.fixed_k5_agreement.common import ideal_mask, task_split
from pattern.fixed_k5_agreement.config import Config
from pattern.fixed_k5_agreement.model import FixedVAE, vae_loss
from pattern.fixed_k5_agreement.search import _pair_metrics, _structure


class FixedK5ProtocolTests(unittest.TestCase):
    def test_full_split_is_24_8_and_orbit_safe(self) -> None:
        split = task_split(Config())
        self.assertEqual(len(split["train_patterns"]), 24)
        self.assertEqual(len(split["test_patterns"]), 8)
        self.assertEqual(len(set(split["train_patterns"] + split["test_patterns"])), 32)
        train = set(split["train_patterns"])
        for pattern in split["test_patterns"]:
            complement = "".join("1" if bit == "0" else "0" for bit in pattern)
            orbit = {pattern, pattern[::-1], complement, complement[::-1]}
            self.assertTrue(train.isdisjoint(orbit))

    def test_mask_contract(self) -> None:
        config = Config()
        target = ideal_mask(config)
        self.assertEqual(tuple(target.shape), (32, 32))
        self.assertEqual(int(target.sum()), 160)
        structure = _structure(target.unsqueeze(0), config)
        self.assertAlmostEqual(structure["gold_iou_mean"], 1.0)
        pair = _pair_metrics(target.unsqueeze(0), target.unsqueeze(0), config)
        self.assertEqual(pair["exact_count"], 1)
        self.assertAlmostEqual(pair["hamming_normalized_mean"], 0.0)

    def test_vae_shapes_and_loss(self) -> None:
        model = FixedVAE(mask_dim=1024, latent_dim=4, hidden=16)
        target = torch.rand(3, 1024)
        logits, mu, logvar = model(target)
        total, reconstruction, kl = vae_loss(logits, target, mu, logvar, 0.1)
        self.assertEqual(tuple(logits.shape), (3, 1024))
        self.assertEqual(tuple(model.decode(torch.randn(2, 4), torch.empty(2, 0)).shape), (2, 1024))
        self.assertTrue(torch.isfinite(torch.stack((total, reconstruction, kl))).all())

    def test_smoke_config_keeps_equal_random_budget(self) -> None:
        config = Config.smoke()
        self.assertEqual(config.random_proposals, config.agreement_steps + 1)
        self.assertEqual(len(config.pairs), 1)
        self.assertEqual(config.k_active, 160)


if __name__ == "__main__":
    unittest.main()
