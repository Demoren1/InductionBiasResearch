"""CPU tests for exact iid-five source-population DeepSets fitting."""

from __future__ import annotations

import itertools
import unittest

import torch

from deepsets_vaae.core import MaskedDeepSets
from deepsets_vaae.rebuilt_bank_population import (
    _packed_per_image_predictions,
    fit_population,
    iid5_population_nmse,
)


def _fixture():
    generator = torch.Generator().manual_seed(1553)
    masks = torch.zeros(2, 784, 3)
    for method in range(2):
        indices = torch.randperm(784 * 3, generator=generator)[:125]
        masks[method].view(-1)[indices] = 1.0
    images = torch.randn(7, 784, generator=generator)
    targets = torch.randn(7, generator=generator)
    qx = torch.randn(4, 5, 784, generator=generator)
    qy = torch.randn(4, generator=generator)
    return masks, images, targets, qx, qy


class RebuiltBankPopulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls._old_threads)

    def test_iid_five_formula_matches_explicit_set_enumeration(self):
        residual = torch.tensor([-0.4, 0.2, 0.7], dtype=torch.float64)
        exact = torch.stack([
            residual[list(draw)].sum().square() / 5.0
            for draw in itertools.product(range(len(residual)), repeat=5)
        ]).mean()
        torch.testing.assert_close(iid5_population_nmse(residual), exact)
        batch = torch.stack((residual, residual * -0.3))
        torch.testing.assert_close(
            iid5_population_nmse(batch),
            torch.stack([iid5_population_nmse(row) for row in batch]),
        )

    def test_packed_predictions_and_query_label_independence(self):
        masks, images, targets, qx, qy = _fixture()
        captured = []
        kwargs = dict(
            masks=masks,
            images=images,
            image_targets=targets,
            qx=qx,
            init_seed=412,
            replicas=[3, 7],
            lr=1e-3,
            l2=2e-5,
            cap=3,
            minimum=3,
            device="cpu",
            reference_models=20,
            checkpoint_every=1,
        )
        fitted = fit_population(qy=qy, checkpoint_callback=captured.append, **kwargs)
        changed_query = fit_population(qy=qy + 20.0, **kwargs)
        self.assertEqual(fitted["terminal_step"], 3)
        self.assertEqual(tuple(fitted["population_loss"].shape), (2,))
        self.assertEqual(tuple(fitted["query_loss"].shape), (2,))
        self.assertTrue(torch.isfinite(fitted["population_objective"]).all())
        self.assertFalse(torch.allclose(fitted["query_loss"], changed_query["query_loss"]))
        for name, value in fitted["state_dict"].items():
            torch.testing.assert_close(value, changed_query["state_dict"][name], atol=0, rtol=0)

        # Numbered replicas select those exact rows from the shared reference bank.
        initial_state = captured[0]["state_dict"]
        reference = MaskedDeepSets(
            torch.ones(20, 784, 3), seed=412, initialization_reference_models=20
        )
        torch.testing.assert_close(initial_state["weight"][0], reference.weight[3])
        torch.testing.assert_close(initial_state["weight"][1], reference.weight[7])
        torch.testing.assert_close(initial_state["readout"][0], reference.readout[3])
        torch.testing.assert_close(initial_state["readout"][1], reference.readout[7])

        initial_model = MaskedDeepSets(
            masks, seed=412, initialization_reference_models=20
        )
        initial_model.load_state_dict(initial_state)
        packed = _packed_per_image_predictions(initial_model, images)
        reference_predictions = initial_model(images[:, None])
        torch.testing.assert_close(packed, reference_predictions, atol=1e-6, rtol=1e-5)
        residual = packed - targets[None]
        torch.testing.assert_close(
            captured[0]["population_loss"], iid5_population_nmse(residual),
            atol=1e-7, rtol=1e-6,
        )
        self.assertIn("optimizer_state", fitted)
        self.assertEqual(captured[0]["step"], 0)

    def test_support_only_plateau_freezes_each_candidate_before_cap(self):
        masks = torch.ones(1, 784, 2)
        images = torch.zeros(3, 784)
        targets = torch.zeros(3)
        qx = torch.zeros(2, 5, 784)
        qy = torch.zeros(2)
        result = fit_population(
            masks, images, targets, qx, qy, init_seed=5, replicas=[0],
            lr=1e-3, l2=0.0, cap=300, minimum=100, device="cpu",
            reference_models=20, checkpoint_every=50, plateau_patience=3,
        )
        self.assertEqual(result["terminal_step"], 200)
        self.assertTrue(result["plateau_stopped"].item())
        self.assertFalse(result["capped"].item())
        self.assertEqual(result["stopping_steps"].item(), 200)
        self.assertTrue(torch.isfinite(result["state_dict"]["weight"]).all())


if __name__ == "__main__":
    unittest.main()
