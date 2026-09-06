"""Check oracle reachability with a decoder whose reachable set is known."""
import unittest

import torch

from evaluation.oracle_ideal import optimize_ideal


class IdentityDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(16))

    def decode(self, z, condition):
        return self.linear(z)


class OracleIdealTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(19)
        self.model = IdentityDecoder()
        self.target = torch.eye(4)
        self.z = torch.randn(3, 16)

    def test_known_reachable_target_found_without_changing_decoder(self):
        before = self.model.linear.weight.detach().clone()
        result = optimize_ideal(self.model, self.z, self.target, steps=80,
                                lr=.15, radius=8., temperature=.5)
        self.assertTrue(torch.equal(self.z, result["initial"]["z"]))
        self.assertTrue(torch.equal(before, self.model.linear.weight))
        self.assertIsNone(self.model.linear.weight.grad)
        self.assertTrue((result["best_hard"]["iou"] == 1).all())
        self.assertTrue((result["best_soft"]["loss"] <= result["initial"]["loss"]).all())
        self.assertTrue((result["best_hard"]["hard"].sum((1, 2)) == 4).all())
        self.assertTrue((result["best_soft"]["z"].norm(dim=1) <= 8.00001).all())
        self.assertTrue((result["best_hard"]["aligned_hard"] == self.target).all())

    def test_zero_step_includes_initial_witness_and_column_permutations(self):
        logits = ((self.target[:, [2, 0, 3, 1]] * 2 - 1) * .5).flatten()[None]
        result = optimize_ideal(self.model, logits, self.target, steps=0, radius=8.)
        self.assertEqual(result["initial"]["iou"].item(), 1.)
        self.assertTrue(torch.equal(result["initial"]["z"], result["best_hard"]["z"]))
        self.assertEqual(result["best_hard_steps"].item(), 0)


if __name__ == "__main__":
    unittest.main()
