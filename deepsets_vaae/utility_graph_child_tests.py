"""CPU tests for fixed-horizon paired DeepSets child fits."""

from __future__ import annotations

import unittest

import torch

from deepsets_vaae.core import MaskedDeepSets
from deepsets_vaae.utility_graph_child import fit_children


def _fixture():
    generator = torch.Generator().manual_seed(712)
    masks = torch.zeros(3, 784, 2)
    for method in range(3):
        masks[method].view(-1)[method * 2:method * 2 + 80] = 1.0
    xs = torch.randn(2, 5, 5, 784, generator=generator)
    ys = torch.randn(2, 5, generator=generator)
    xq = torch.randn(2, 3, 5, 784, generator=generator)
    yq = torch.randn(2, 3, generator=generator)
    return masks, xs, ys, xq, yq


class UtilityGraphChildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls._old_threads)

    def test_fixed_horizon_query_invariance_and_numbered_reference_draws(self):
        masks, xs, ys, xq, yq = _fixture()
        captured = []
        kwargs = dict(
            masks=masks,
            x_support=xs,
            y_support=ys,
            x_query=xq,
            condition_seeds=[101, 202],
            replicas=[3, 3, 1],
            steps=4,
            lr=1e-3,
            l2=1e-5,
            device="cpu",
            reference_models=20,
            chunk_size=2,
            checkpoint_every=2,
        )
        result = fit_children(y_query=yq, checkpoint_callback=captured.append, **kwargs)
        flipped = fit_children(y_query=yq + 50.0, **kwargs)
        self.assertEqual(result["steps_run"], 4)
        self.assertTrue(result["fixed_horizon"])
        self.assertTrue(torch.equal(result["plateau_flags"], torch.zeros_like(result["plateau_flags"])))
        self.assertEqual(tuple(result["support_loss"].shape), (2, 3))
        self.assertEqual(tuple(result["query_loss"].shape), (2, 3))
        self.assertEqual(tuple(result["history"]["queryNMSE"].shape), (3, 2, 3))
        self.assertFalse(torch.allclose(result["query_loss"], flipped["query_loss"]))
        for name, value in result["state_dict"].items():
            torch.testing.assert_close(value, flipped["state_dict"][name], atol=0, rtol=0)

        # Replica IDs mean the exact numbered rows in the shared 20-draw bank,
        # even when replica 3 is first in a mask list with repeated IDs.
        first = captured[0]["state_dict"]
        reference = MaskedDeepSets(
            torch.ones(20, 784, 2), seed=101, initialization_reference_models=20
        )
        torch.testing.assert_close(first["weight"][0, 0], reference.weight[3])
        torch.testing.assert_close(first["weight"][0, 1], reference.weight[3])
        torch.testing.assert_close(first["weight"][0, 2], reference.weight[1])
        torch.testing.assert_close(first["readout"][0, 0], reference.readout[3])
        self.assertEqual(captured[0]["step"], 0)
        self.assertIn("queryNMSE", captured[-1])

    def test_support_minibatch_seed_and_learning_rate_schedule(self):
        masks, xs, ys, xq, yq = _fixture()
        kwargs = dict(
            masks=masks,
            x_support=xs,
            y_support=ys,
            x_query=xq,
            y_query=yq,
            condition_seeds=[11, 12],
            replicas=[0, 1, 2],
            steps=3,
            lr=2e-3,
            l2=1e-5,
            device="cpu",
            batch_size=2,
            seed=901,
            chunk_size=2,
            lr_decay_every=2,
            lr_floor=0.25,
            checkpoint_every=1,
        )
        first = fit_children(**kwargs)
        second = fit_children(**kwargs)
        for name, value in first["state_dict"].items():
            torch.testing.assert_close(value, second["state_dict"][name], atol=0, rtol=0)
        self.assertEqual(first["minibatch_seed"], 901)
        self.assertEqual(first["minibatch_condition_seeds"], [1910, 2919])
        torch.testing.assert_close(
            first["history"]["learning_rate"], torch.tensor([2e-3, 2e-3, 2e-3, 1e-3])
        )

    def test_fullbatch_gradient_is_chunk_size_invariant(self):
        masks, xs, ys, xq, yq = _fixture()
        kwargs = dict(
            masks=masks,
            x_support=xs,
            y_support=ys,
            x_query=xq,
            y_query=yq,
            condition_seeds=[77, 78],
            replicas=[0, 1, 2],
            steps=2,
            lr=1e-3,
            l2=1e-5,
            device="cpu",
            checkpoint_every=2,
        )
        small_chunks = fit_children(chunk_size=1, **kwargs)
        full_chunks = fit_children(chunk_size=5, **kwargs)
        for name, value in small_chunks["state_dict"].items():
            torch.testing.assert_close(value, full_chunks["state_dict"][name], atol=2e-6, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()

