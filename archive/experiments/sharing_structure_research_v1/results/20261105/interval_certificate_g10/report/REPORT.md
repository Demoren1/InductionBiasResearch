# G10: global interval compatibility

Frozen prospective linear-Gaussian diagnostic: six independent geometries, three coefficient draws, three input-covariance regimes and two noise levels =108 dependent cases. All source outputs were frozen before true-I audits and fresh-C transfer. Exactly four signed nonzero groups and zero are known; locations, group sizes and numeric codes are hidden.

The new solver gives more certified structures than pairwise graphs. Strong zero-significance + weighted-DP controls remain more accurate at a fixed budget. No superiority over native competing methods or nonlinear recovery is established.

## Exact indexators (abstention counts as failure)

| Method | n8192 | n32768 | n131072 | n8388608 |
|---|---:|---:|---:|---:|
| Scalar DP | 48/108 | 69/108 | 84/108 | 101/108 |
| Marginal-information DP | 52/108 | 74/108 | 88/108 | 101/108 |
| Diagonal-Hessian DP | 52/108 | 73/108 | 88/108 | 101/108 |
| Zero significance + information DP | 65/108 | 80/108 | 92/108 | 102/108 |
| Point SLINK | 56/108 | 77/108 | 87/108 | 102/108 |
| Same-interval graph | 32/108 | 47/108 | 64/108 | 98/108 |
| Full pairwise F graph | 32/108 | 47/108 | 68/108 | 99/108 |
| Global interval certificate | 40/108 | 53/108 | 70/108 | 100/108 |

## Reliability and scope

No wrong confidence-issued I occurred at any of the eight looks. The guarantee is alpha=.05 per algorithm/source run across its eight looks, not a simultaneous guarantee for all108 benchmark cases. Marginal simultaneous coverage counts by look: {512: 108, 2048: 106, 8192: 108, 32768: 108, 131072: 102, 524288: 108, 2097152: 108, 8388608: 108}. The six q>K events at n131072 coincide with actual CI noncoverage; they cause abstention, not a false certificate. These observations are compatible with, and do not prove, the finite Gaussian argument.

A single source-function draw can contain nearly colliding code magnitudes. Some cases remain unresolved even at8.39m examples. More data cannot resolve an exactly colliding source codebook; multiple informative source functions remain necessary for general structural identification.

## Diagnosis and transfer

At n32768,33 cases have q=K but some ambiguous assignments,22 have q<K,53 issue. This motivates a separately registered next stage: use forced positions as cores, independently fit codebooks, and exploit full covariance to classify remaining positions. G10 alone does not establish that extension works.

Fresh-C plots use100 new Gaussian codebooks per source geometry/draw, with I frozen and all C re-estimated using only new support. Source numeric weights are discarded. Confidence curves condition on issued structures; their issued counts are shown, so lower MSE on an easier subset is not an unconditional win. Source regimes/noise reuse the same target cohort and are dependent repeats.

## Nonlinear diagnostic is separate

On already-opened G9 estimates, the global solver issues all16 noiseless views correctly, and only one of four noisy warm families per budget. Nonlinear CI coverage and stationarity are not certified. This is exploratory and does not inherit the exact linear Gaussian guarantee. Hidden-column permutation is accounted for in audit metrics; the initial wrong metric file is preserved beside its correction record.

## Artifacts

- [Exact recovery](exact_recovery.png)
- [Recall and observed projection loss](recall_and_projection_loss.png); closed-form sample budgets, not fictitious optimizer-step curves.
- [Fresh codebook transfer](fresh_codebook_transfer.png)
- [Observed W / CI width / chosen I / true I](selected_indexator_heatmap.png)
- [Machine-readable comparison](comparison.csv), [full summary](summary.json).

Weighted k-means/Hessian-weighted quantization, interval piercing, confidence clustering and post-selection sample splitting are known principles. The specific global signed-equality certificate is a candidate specialization; priority and advantage over strong native solvers remain open.
