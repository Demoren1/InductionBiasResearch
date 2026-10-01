"""Focused CPU tests for raw functional teacher token encoding."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from deepsets_vaae.permutation_bank_encoder import (
    RawFunctionalBankEncoder,
    TrainOnlyFunctionalTeacherBank,
    permutation_consistency_loss,
    permute_teacher_hidden_columns,
)


class RawFunctionalBankEncoderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _write_tiny_context(self) -> Path:
        teachers, probes, features, hidden = 4, 2, 5, 3
        raw_psi = torch.arange(1, teachers * probes * hidden + 1, dtype=torch.float32).reshape(
            teachers, probes, hidden
        )
        raw_signed = torch.arange(1, teachers * features * hidden + 1, dtype=torch.float32).reshape(
            teachers, features, hidden
        )
        raw_abs = raw_signed.abs() + 2.0
        raw_rms = raw_signed.abs() + 3.0
        raw_masks = (torch.arange(teachers * features * hidden).reshape(teachers, features, hidden) % 2).float()
        orders = torch.tensor([
            [2, 0, 1],
            [1, 2, 0],
            [0, 2, 1],
            [2, 1, 0],
        ])

        def align(value: torch.Tensor) -> torch.Tensor:
            return value.gather(-1, orders[:, None, :].expand_as(value))

        bank_path = self.root / "bank.pt"
        torch.save({
            "state_dict": {"masks": raw_masks},
            "queryNMSE": torch.tensor([2.5, 999.0, 1.5, 888.0]),
            "audit_nmse": torch.tensor([-10000.0, -20000.0, 30000.0, 40000.0]),
        }, bank_path)
        context_path = self.root / "functional_context.pt"
        torch.save({
            "psi": align(raw_psi)[None],
            "q_signed_mean": align(raw_signed)[None],
            "q_abs_mean": align(raw_abs)[None],
            "q_rms": align(raw_rms)[None],
            "train_rows": torch.tensor([[0, 2]]),
            "heldout_rows": torch.tensor([[1, 3]]),
            "alignment_orders": orders[None],
            "bank_references": [{"path": str(bank_path)}],
        }, context_path)
        return context_path

    def test_loader_uses_only_train_rows_and_restores_raw_column_order(self) -> None:
        context_path = self._write_tiny_context()
        bank = TrainOnlyFunctionalTeacherBank(
            context_path,
            include_source_query_quality=True,
            include_training_masks=True,
        )
        refs = torch.tensor([[[0, 2], [0, 0]]])
        batch = bank.gather(refs)

        self.assertEqual(batch.tokens.shape, (1, 2, 3, 22))
        self.assertEqual(batch.references.tolist(), refs.tolist())
        self.assertEqual(bank.train_references.tolist(), [[0, 0], [0, 2]])
        self.assertEqual(bank.channel_slices["training_mask"], slice(17, 22))
        self.assertTrue(torch.isfinite(batch.tokens).all())
        self.assertTrue(torch.allclose(batch.source_query_quality, torch.tensor([[-1.0, 1.0]])))

        raw_psi = torch.arange(1, 4 * 2 * 3 + 1, dtype=torch.float32).reshape(4, 2, 3)
        raw_signed = torch.arange(1, 4 * 5 * 3 + 1, dtype=torch.float32).reshape(4, 5, 3)
        raw_abs = raw_signed.abs() + 2.0
        raw_rms = raw_signed.abs() + 3.0
        expected_psi = raw_psi[torch.tensor([2, 0])]
        expected_psi = expected_psi / expected_psi.square().mean(dim=(1, 2), keepdim=True).sqrt()
        expected_q_signed = raw_signed[torch.tensor([2, 0])]
        expected_q_abs = raw_abs[torch.tensor([2, 0])]
        expected_q_rms = raw_rms[torch.tensor([2, 0])]
        q_scale = expected_q_rms.flatten(1).amax(dim=1)[:, None, None]
        torch.testing.assert_close(batch.tokens[0, :, :, 0:2], expected_psi.transpose(1, 2))
        torch.testing.assert_close(batch.tokens[0, :, :, 2:7], (expected_q_signed / q_scale).transpose(1, 2))
        torch.testing.assert_close(batch.tokens[0, :, :, 7:12], (expected_q_abs / q_scale).transpose(1, 2))
        torch.testing.assert_close(batch.tokens[0, :, :, 12:17], (expected_q_rms / q_scale).transpose(1, 2))

        raw_mask = torch.load(
            self.root / "bank.pt", map_location="cpu", weights_only=False
        )["state_dict"]["masks"]
        expected_masks = raw_mask[torch.tensor([2, 0])].transpose(1, 2)
        torch.testing.assert_close(batch.tokens[0, :, :, 17:22], expected_masks)

        with self.assertRaisesRegex(ValueError, "not in saved train_rows"):
            bank.gather(torch.tensor([[[0, 1]]]))

        no_quality = TrainOnlyFunctionalTeacherBank(context_path)
        self.assertIsNone(no_quality.gather(refs).source_query_quality)

    def test_joint_permutation_moves_every_channel_together(self) -> None:
        tokens = torch.arange(2 * 2 * 3 * 4).reshape(2, 2, 3, 4).float()
        permutation = torch.tensor([
            [[2, 0, 1], [1, 2, 0]],
            [[1, 0, 2], [2, 1, 0]],
        ])
        shuffled, returned = permute_teacher_hidden_columns(tokens, permutation=permutation)
        self.assertTrue(torch.equal(returned, permutation))
        torch.testing.assert_close(shuffled, tokens.gather(2, permutation[..., None].expand_as(tokens)))

    def test_default_encoder_is_exactly_column_invariant(self) -> None:
        torch.manual_seed(4)
        model = RawFunctionalBankEncoder(
            token_dim=12, task_context_dim=7, features=9, hidden=4, width=16, attention_heads=4
        ).eval()
        tokens = torch.randn(2, 5, 4, 12)
        task = torch.randn(2, 7)
        quality = torch.randn(2, 5)
        shuffled, _ = permute_teacher_hidden_columns(tokens)
        embedding_a, logits_a = model(tokens, task, quality)
        embedding_b, logits_b = model(shuffled, task, quality)
        torch.testing.assert_close(embedding_a, embedding_b, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(logits_a, logits_b, atol=1e-6, rtol=1e-6)
        self.assertEqual(tuple(logits_a.shape), (2, 9, 4))
        self.assertLess(float(permutation_consistency_loss(logits_a, logits_b)), 1e-12)

    def test_teacher_order_is_invariant_but_cross_teacher_regrouping_changes_response(self) -> None:
        torch.manual_seed(31)
        model = RawFunctionalBankEncoder(
            token_dim=8, task_context_dim=3, features=6, hidden=4, width=16, attention_heads=4
        ).eval()
        tokens = torch.randn(2, 5, 4, 8)
        task = torch.randn(2, 3)
        quality = torch.randn(2, 5)

        teacher_order = torch.tensor([3, 0, 4, 1, 2])
        _, logits = model(tokens, task, quality)
        _, reordered_logits = model(tokens[:, teacher_order], task, quality[:, teacher_order])
        torch.testing.assert_close(logits, reordered_logits, atol=1e-6, rtol=1e-6)

        regrouped = tokens.clone()
        first_teacher_neuron = regrouped[:, 0, 0, :].clone()
        regrouped[:, 0, 0, :] = regrouped[:, 1, 0, :]
        regrouped[:, 1, 0, :] = first_teacher_neuron
        _, regrouped_logits = model(regrouped, task, quality)
        self.assertGreater(float((logits - regrouped_logits).abs().max()), 1e-5)

    def test_inconsistent_q_association_changes_output(self) -> None:
        torch.manual_seed(12)
        model = RawFunctionalBankEncoder(
            token_dim=8, task_context_dim=3, features=6, hidden=4, width=16, attention_heads=4
        ).eval()
        tokens = torch.randn(2, 3, 4, 8)
        task = torch.randn(2, 3)
        changed = tokens.clone()
        changed[:, :, :, 4:] = changed[:, :, torch.tensor([1, 0, 3, 2]), 4:]
        _, logits = model(tokens, task)
        _, mismatched_logits = model(changed, task)
        self.assertGreater(float((logits - mismatched_logits).abs().max()), 1e-5)

    def test_position_bias_makes_consistency_loss_nonvacuous_and_gradients_finite(self) -> None:
        torch.manual_seed(22)
        model = RawFunctionalBankEncoder(
            token_dim=10,
            task_context_dim=5,
            features=7,
            hidden=4,
            width=16,
            attention_heads=4,
            column_position_bias=True,
        )
        tokens = torch.randn(2, 3, 4, 10)
        task = torch.randn(2, 5, requires_grad=True)
        quality = torch.randn(2, 3, requires_grad=True)
        shuffled, _ = permute_teacher_hidden_columns(tokens)
        _, logits_a = model(tokens, task, quality)
        _, logits_b = model(shuffled, task, quality)
        loss = permutation_consistency_loss(logits_a, logits_b)
        self.assertGreater(float(loss.detach()), 1e-9)
        (loss + logits_a.square().mean()).backward()
        self.assertTrue(torch.isfinite(task.grad).all())
        self.assertTrue(torch.isfinite(quality.grad).all())
        self.assertTrue(all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))


if __name__ == "__main__":
    unittest.main()
