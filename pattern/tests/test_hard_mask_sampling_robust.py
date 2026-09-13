from __future__ import annotations

import unittest

import torch

from evaluation.hard_mask_sampling_robust import robust_winners


class RobustSamplingTests(unittest.TestCase):
    def test_candidate_must_win_every_replicate_and_change_mask(self):
        masks = torch.zeros(2, 3, 8, 8)
        masks[:, :, :4] = 1
        masks[0, 1, 0, 0] = 0
        masks[0, 1, 4, 0] = 1
        masks[1, 1, 0, 0] = 0
        masks[1, 1, 4, 0] = 1
        scores = torch.tensor([
            [[1.0, .8, 1.2], [1.0, .8, 1.2]],
            [[1.0, .9, 1.2], [1.0, 1.1, 1.2]],
        ])
        # Parent 0 accepts candidate 1; parent 1 rejects because replicate 2 loses.
        winners, accepted = robust_winners(scores, masks, 1e-4)
        self.assertEqual(winners.tolist(), [1, 0])
        self.assertEqual(accepted.tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
