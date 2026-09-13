from __future__ import annotations

import unittest

import torch

from evaluation.single_z_task_hard_ste import hard_ste


class HardSteTests(unittest.TestCase):
    def test_forward_is_binary_and_backward_uses_soft(self):
        logits = torch.randn(3, 64, requires_grad=True)
        soft = torch.sigmoid(logits).reshape(3, 8, 8)
        flat = logits.detach().topk(32, dim=1).indices
        hard_flat = torch.zeros_like(logits).scatter(1, flat, 1.0)
        hard, mask = hard_ste(logits, soft)
        self.assertTrue(torch.equal(hard, hard_flat.reshape(3, 8, 8)))
        self.assertTrue(torch.equal(mask.detach(), hard))
        mask.square().sum().backward()
        self.assertIsNotNone(logits.grad)
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
