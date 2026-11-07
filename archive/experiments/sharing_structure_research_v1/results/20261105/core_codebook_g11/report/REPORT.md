# G11: independent codebooks from certified cores

Frozen prospective linear-Gaussian diagnostic: six independent geometries, three coefficient draws, three input-covariance regimes and two noise levels =108 dependent cases. All source outputs were frozen before true-I audits and fresh-C transfer. Exactly four signed nonzero groups and zero are known; locations, group sizes and numeric codes are hidden.

The new solver gives more certified structures than pairwise graphs. Strong zero-significance + weighted-DP controls remain more accurate at a fixed budget. No superiority over native competing methods or nonlinear recovery is established.

## Exact indexators (abstention counts as failure)

| Method | n8192 | n32768 | n131072 | n8388608 |
|---|---:|---:|---:|---:|
| Scalar DP | 47/108 | 65/108 | 80/108 | 106/108 |
| Marginal-information DP | 53/108 | 71/108 | 85/108 | 107/108 |
| Diagonal-Hessian DP | 53/108 | 71/108 | 85/108 | 107/108 |
| Zero significance + information DP | 59/108 | 81/108 | 89/108 | 107/108 |
| Point SLINK | 58/108 | 76/108 | 87/108 | 106/108 |
| Same-interval graph | 34/108 | 48/108 | 64/108 | 98/108 |
| Full pairwise F graph | 36/108 | 50/108 | 66/108 | 99/108 |
| Global interval certificate | 41/108 | 56/108 | 72/108 | 106/108 |
| One witness per code | 36/108 | 52/108 | 72/108 | 105/108 |
| Mean of certified cores | 37/108 | 55/108 | 73/108 | 104/108 |
| Nonzero-core GLS | 38/108 | 55/108 | 73/108 | 105/108 |
| Full GLS + certified zeros | 38/108 | 57/108 | 75/108 | 105/108 |

## Reliability and scope

No wrong confidence-issued I occurred at any of the eight looks. The guarantee is alpha=.05 per algorithm/source run across its eight looks, not a simultaneous guarantee for all108 benchmark cases. Marginal simultaneous coverage counts by look: {512: 108, 2048: 108, 8192: 106, 32768: 108, 131072: 106, 524288: 108, 2097152: 108, 8388608: 108}. These observations are compatible with, and do not prove, the finite Gaussian argument. Core methods split the same total n into two independent halves; alpha is divided across the A interval and B contrast stages. Full GLS exploits all forced zeros/nonzeros with the complete covariance.

A single source-function draw can contain nearly colliding code magnitudes. Some cases remain unresolved even at8.39m examples. More data cannot resolve an exactly colliding source codebook; multiple informative source functions remain necessary for general structural identification.

## Diagnosis and transfer

At n131072 full GLS issues75/108 exact I, versus72/108 global-interval and66/108 full-F. Mean-core and nonzero-GLS both issue73/108, witness-only72/108. Strong zero-cleanup+information-DP gives89/108. This is a small improvement in certified coverage, not superiority in unconditional exact recovery.

Fresh-C plots use100 new Gaussian codebooks per source geometry/draw, with I frozen and all C re-estimated using only new support. Source numeric weights are discarded. Confidence curves condition on issued structures; their issued counts are shown, so lower MSE on an easier subset is not an unconditional win. Source regimes/noise reuse the same target cohort and are dependent repeats.

## Nonlinear diagnostic is separate

G11 is a linear diagnostic only. Its exact Student argument does not certify nonlinear G9 or new DeepSets tasks. Independent native nonlinear competitors, new C transfer and robustness to misspecified noise remain outstanding.

## Artifacts

- [Exact recovery](exact_recovery.png)
- [Recall and observed projection loss](recall_and_projection_loss.png); closed-form sample budgets, not fictitious optimizer-step curves.
- [Fresh codebook transfer](fresh_codebook_transfer.png)
- [Observed W / CI width / chosen I / true I](selected_indexator_heatmap.png)
- [Machine-readable comparison](comparison.csv), [full summary](summary.json).

Weighted k-means/Hessian-weighted quantization, interval piercing, confidence clustering and post-selection sample splitting are known principles. The specific global signed-equality certificate is a candidate specialization; priority and advantage over strong native solvers remain open.


## Post-freeze mechanism audit

At n131072,86/108 cases have q=K on half A;22 abstain before codebook estimation. Full GLS reduces median code variance to0.15 of the arithmetic-core estimator, but improves exact issued I only73→75. Better numeric codebooks alone do not remove the remaining uncertain coordinate assignments or the first-stage bottleneck. No forced-I contradiction was found on any audited simultaneous-coverage event.

Paired effects are bootstrapped over six geometries, not108 dependent repetitions. They are descriptive and cannot establish strong general superiority. [Full diagnostic numbers](core_diagnosis.json), [core availability/variance](core_availability_and_codebook_variance.png), [cores and final I](certified_core_indexators.png).
