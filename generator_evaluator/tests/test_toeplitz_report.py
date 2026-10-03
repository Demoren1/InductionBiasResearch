from pathlib import Path
import tempfile
import unittest

import torch

from generator_evaluator.mask_priors import SlidingWindowMaskPrior
from generator_evaluator.toeplitz_report import _proposal_scores, _write_proposal_plot


class StagedToeplitzReportTests(unittest.TestCase):
    def test_legacy_and_staged_artifacts_keep_the_epoch_order(self):
        prior = SlidingWindowMaskPrior()
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            (out / "proposals").mkdir()
            common = dict(masks=prior.mask()[None], sources=["generator:0001"])
            torch.save(common, out / "proposals/epoch_0001.pt")
            torch.save(dict(common, stage="quality", stage_epoch=2, epoch=2),
                       out / "proposals/quality_0002.pt")
            torch.save(dict(common, stage="cooperation", stage_epoch=1),
                       out / "proposals/cooperation_0001.pt")
            groups = _proposal_scores(out, prior)
            self.assertEqual(list(groups["0001"]), [1, 2, 3])
            self.assertEqual([values[0] for values in groups["0001"].values()], [1., 1., 1.])

    def test_proposal_grid_is_saved_for_multiple_generators(self):
        import matplotlib
        matplotlib.use("Agg")
        prior = SlidingWindowMaskPrior()
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            (out / "proposals").mkdir()
            (out / "figures").mkdir()
            torch.save(dict(stage="cooperation", stage_epoch=1, epoch=3,
                            masks=prior.mask()[None].expand(5, -1, -1),
                            sources=[f"generator:{index:04b}" for index in range(5)]),
                       out / "proposals/cooperation_0001.pt")
            _write_proposal_plot(out, prior, 1.)
            for extension in ("png", "pdf"):
                self.assertGreater((out / "figures" / f"toeplitz_generator_proposals.{extension}").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
