"""Unit tests for the continuous-length CVAE without requiring experiment data."""

from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import torch
import torch.nn.functional as F

from pattern.length_interp.cvae import LengthCVAE, cvae_loss, kl_divergence, normalize_length
from pattern.length_interp.train_cvae import _load_bank, validate_cvae_patterns


class LengthCVAETests(unittest.TestCase):
    def test_continuous_scalar_condition_and_decoder_shapes(self) -> None:
        model = LengthCVAE(mask_dim=12, latent_dim=4, hidden=9)
        lengths = torch.tensor([3.0, 5.5, 8.0])
        expected = torch.tensor([[-1.0], [0.0], [1.0]])
        self.assertEqual(normalize_length(lengths).shape, (3, 1))
        self.assertTrue(torch.allclose(normalize_length(lengths), expected))
        z = torch.randn(3, 4)
        self.assertEqual(model.decode_lengths(z, lengths).shape, (3, 12))
        self.assertEqual(model.decode_lengths(z, 5.5).shape, (3, 12))
        x = torch.rand(3, 12)
        logits, mu, logvar = model(x, lengths)
        self.assertEqual(logits.shape, (3, 12))
        self.assertEqual(mu.shape, (3, 4))
        self.assertEqual(logvar.shape, (3, 4))

    def test_length_has_a_gradient_path_to_decoder(self) -> None:
        model = LengthCVAE(mask_dim=5, latent_dim=2, hidden=3)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.dec_fc1.weight[0, -1] = 1.0
            model.dec_fc2.weight[0, 0] = 1.0
            model.dec_out.weight[0, 0] = 1.0
        length = torch.tensor([6.0], requires_grad=True)
        out = model.decode_lengths(torch.zeros(1, 2), length)
        out[0, 0].backward()
        self.assertIsNotNone(length.grad)
        self.assertGreater(abs(float(length.grad[0])), 0.0)

    def test_loss_is_bce_sum_per_map_mean_plus_analytic_kl(self) -> None:
        logits = torch.tensor([[0.0, 1.0], [2.0, -1.0]])
        target = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        mu = torch.tensor([[1.0], [2.0]])
        logvar = torch.tensor([[0.0], [torch.log(torch.tensor(2.0))]])
        beta = 0.3
        total, recon, kl = cvae_loss(logits, target, mu, logvar, beta)
        expected_recon = F.binary_cross_entropy_with_logits(logits, target, reduction="none").sum(-1).mean()
        expected_kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1).mean()
        self.assertTrue(torch.allclose(recon, expected_recon))
        self.assertTrue(torch.allclose(kl, expected_kl))
        self.assertTrue(torch.allclose(kl, kl_divergence(mu, logvar)))
        self.assertTrue(torch.allclose(total, expected_recon + beta * expected_kl))

    def test_held_out_lengths_cannot_enter_cvae_data(self) -> None:
        valid = {"train": ["000", "1111", "010101", "00110011"],
                 "val": ["101", "0000", "110011", "11110000"], "test": ["00000"]}
        train, val = validate_cvae_patterns(valid)
        self.assertEqual(train, valid["train"])
        self.assertEqual(val, valid["val"])
        with self.assertRaisesRegex(ValueError, "interpolation-only"):
            validate_cvae_patterns({"train": ["000"], "val": ["00000"]})

    def test_loader_uses_the_bank_selected_importance_only(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "bank.pt"
            selected = torch.full((2, 32, 32), 0.25)
            torch.save({"top_fraction": 0.1,
                        "selected": {"importance": selected}}, path)
            result = _load_bank(path, "000")
        self.assertEqual(result.shape, (2, 1024))
        self.assertTrue(torch.equal(result, selected.reshape(2, 1024)))


if __name__ == "__main__":
    unittest.main()
