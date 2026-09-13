from __future__ import annotations

import unittest

import torch

from evaluation.z_star_noise import ExperimentConfig, build_initials


class ZStarNoiseTests(unittest.TestCase):
    def test_controlled_noise_has_requested_l2_radius_without_projection(self):
        settings = ExperimentConfig.smoke()
        anchors = torch.zeros(1, 32)
        prior = torch.randn(2, 32)
        initials, setup = build_initials(anchors, prior, settings, seed=186)
        self.assertEqual(tuple(initials.shape), (4, 32))
        zero = setup["groups"]["noise_0"]
        half = setup["groups"]["noise_0.5"]
        self.assertEqual(zero["actual_source_distance"], [0.0])
        self.assertLess(abs(half["actual_source_distance"][0] - 0.5), 1e-6)


    def test_full_protocol_has_eight_models_and_nested_noise_grid(self):
        settings = ExperimentConfig()
        self.assertEqual(len(settings.model_seeds), 8)
        self.assertEqual(settings.anchors * settings.directions_per_anchor, 32)
        self.assertEqual(settings.noise_radii[0], 0.0)
        self.assertEqual(settings.noise_radii[-1], 8.0)


if __name__ == "__main__":
    unittest.main()
