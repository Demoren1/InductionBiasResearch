import unittest

import torch

from evaluation.tune_agreement_hparams import hard_metrics


class AgreementHyperparameterTuningTests(unittest.TestCase):
    def test_hard_metrics_align_hidden_columns(self):
        first = torch.zeros(2, 8, 8)
        first[:, :4] = 1
        second = first[:, :, torch.tensor([3, 2, 1, 0, 7, 6, 5, 4])]
        metrics = hard_metrics(first, second)
        self.assertEqual(metrics["iou"], 1.0)
        self.assertEqual(metrics["exact_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
