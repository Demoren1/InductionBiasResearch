# Omniglot WSR confirmation

18 November 2026. The previously frozen WSR, aSRR and Dense policies ran on the
same 4096 new Omniglot episodes across eight cohorts. The confirmation estimates
use these new episodes. The original 1024 episodes are reported alongside them
in the paper.

| Method | First-layer codes | Accuracy (95% CI) | CE |
| --- | ---: | ---: | ---: |
| WSR | 1,024 | 83.36% ± 0.28 pp | 0.5286 |
| aSRR | 1,024 | 83.25% ± 0.27 pp | 0.5338 |
| Dense | 12,544 | 81.12% ± 0.28 pp | 0.5872 |

The paired WSR minus aSRR difference was +0.11 pp with a 95% interval of
[−0.11, +0.32], down from [−0.20, +0.64] on the original episodes. The
confirmation gains over Dense were +2.24 pp [2.04, 2.44] for WSR and +2.13 pp
[1.96, 2.31] for aSRR. The intervals use episode means conditional on the
fixed source selection and held-out character pool.

The WSR grouping remains the one selected from source task 105. Both sharing
methods transfer only group assignments. Each five-way episode has 10 fitting,
five numerical validation and five query images per class. All three methods
use 24 fresh Adam starts of 500 steps and average the probabilities from four
validation-selected solutions. All 24 prediction records were checked before
opening the query labels.

[Registration](registration.json) · [Frozen choices](choices_frozen.json) ·
[Exact results by cohort](results.json) · [Prediction records](predictions_frozen.json)
