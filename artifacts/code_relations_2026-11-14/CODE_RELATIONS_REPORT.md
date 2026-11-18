# Linear relations between recovered codes

This is a separate registered follow-up after earlier test results were seen. All existing source functions, nested source observations and validation tasks are reused. No teacher weights, filters, coefficients or mixing are read by source fitting. A new common test panel of 500 tasks per rank was generated only after all choices froze. Every target has one support observation and 19 queries.

The new method refits source outputs on the input coordinates retained by an existing index, averages signed weights within each code, and factors the uncentered empirical code covariance with SVD. The sqrt(group-size) metric preserves weight-space distances. Validation selects the existing index, map dimension and ridge. No iterative meta-training is added. All available source functions are used for the covariance estimate.

The primary comparison uses the same fresh least-squares/ridge target solver for all representations. Source numerical initialization and learned rates are discarded. MSR/MTSR rows here are representation diagnostics, not their native adaptation. Frozen native MSR/MTSR are also reevaluated on the same tasks and reported separately.

Dense-SVD is the full m=N=4760 control with no recovered index. LC-SVD is an architecture oracle supplied with width-3 locality; its equality with recovered-mask methods does not establish an advantage of nontrivial equality grouping. Confidence intervals are conditional on this environment and fixed recovered bank; this is not a multi-environment replication.

## Same target solver

| Rank | Source examples | G34 | Pair | G34 + SVD | Pair + SVD | MSR basis | MTSR basis | Dense-SVD | LC-SVD (supplied locality) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 1.97805e-15 | 1.97805e-15 | 1.97819e-15 | 1.97819e-15 | 0.000550164 | 5.55164e-06 | 0.0964464 | 1.98572e-15 |
| 1 | 80 | 1.97805e-15 | 1.97805e-15 | 1.97819e-15 | 1.97819e-15 | 0.000641255 | 8.98448e-07 | 2.16015e-15 | 1.98002e-15 |
| 1 | 200 | 1.97805e-15 | 1.97805e-15 | 1.97819e-15 | 1.97819e-15 | 0.000514485 | 2.87326e-07 | 1.99124e-15 | 1.97873e-15 |
| 2 | 20 | 0.697832 | 1.2482 | 1.40056e-15 | 1.40056e-15 | 0.0122128 | 1.05066e-05 | 0.0542231 | 1.40056e-15 |
| 2 | 80 | 0.531026 | 1.2482 | 1.39684e-15 | 1.39684e-15 | 0.0092501 | 1.30771e-06 | 1.50143e-15 | 1.39684e-15 |
| 2 | 200 | 0.697832 | 1.2482 | 1.3964e-15 | 1.3964e-15 | 0.0148704 | 9.69684e-07 | 1.40436e-15 | 1.3964e-15 |
| 5 | 20 | 0.610511 | 0.610511 | 8.40615e-16 | 8.40615e-16 | 0.052401 | 5.51776e-06 | 0.0650477 | 8.40615e-16 |
| 5 | 80 | 0.610511 | 0.610511 | 8.37466e-16 | 8.37466e-16 | 0.0331338 | 1.5769e-06 | 9.66134e-16 | 8.37466e-16 |
| 5 | 200 | 0.610511 | 0.610511 | 8.36962e-16 | 8.36962e-16 | 0.0412452 | 1.57044e-06 | 8.46263e-16 | 8.36962e-16 |

## Fixed index versus joint selection

The fixed-index rows preserve the previous fresh-selected index exactly. Joint selection may choose a less compressed index before learning code relations.

| Case | G34 fixed I + SVD | G34 joint | Selected m / d | Pair fixed I + SVD | Pair joint | Selected m / d |
|---|---:|---:|---|---:|---:|---|
| rank1_n20 | 1.97819e-15 | 1.97819e-15 | 3 / 3 | 1.97819e-15 | 1.97819e-15 | 3 / 3 |
| rank1_n80 | 1.97819e-15 | 1.97819e-15 | 3 / 3 | 1.97819e-15 | 1.97819e-15 | 3 / 3 |
| rank1_n200 | 1.97819e-15 | 1.97819e-15 | 3 / 3 | 1.97819e-15 | 1.97819e-15 | 3 / 3 |
| rank2_n20 | 0.163829 | 1.40056e-15 | 204 / 6 | 1.40056e-15 | 1.40056e-15 | 204 / 6 |
| rank2_n80 | 0.447001 | 1.39684e-15 | 204 / 6 | 1.39684e-15 | 1.39684e-15 | 204 / 6 |
| rank2_n200 | 0.163829 | 1.3964e-15 | 204 / 6 | 1.3964e-15 | 1.3964e-15 | 204 / 6 |
| rank5_n20 | 8.40615e-16 | 8.40615e-16 | 204 / 15 | 8.40615e-16 | 8.40615e-16 | 204 / 15 |
| rank5_n80 | 8.37466e-16 | 8.37466e-16 | 204 / 15 | 8.37466e-16 | 8.37466e-16 | 204 / 15 |
| rank5_n200 | 8.36962e-16 | 8.36962e-16 | 204 / 15 | 8.36962e-16 | 8.36962e-16 | 204 / 15 |

## Native competitor references on these same tasks

Three existing training seeds, original adaptation (MSR: nine learned SGD steps; MTSR: 500 Adam steps). These have a different target solver from the primary table.

| Case | MSR native | MTSR native |
|---|---:|---:|
| rank1_n20 | 0.000912333 | 0.0914441 |
| rank1_n80 | 0.000616327 | 0.0360295 |
| rank1_n200 | 0.000495409 | 0.0231127 |
| rank2_n20 | 0.0160167 | 0.0913094 |
| rank2_n80 | 0.026936 | 0.0336352 |
| rank2_n200 | 0.0149555 | 0.019061 |
| rank5_n20 | 0.054925 | 0.0760212 |
| rank5_n80 | 0.0501426 | 0.0262713 |
| rank5_n200 | 0.143576 | 0.0136422 |

The native MSR rank5/n200 mean has a CI95 half-width of 0.192935 and training-seed means 0.0538293, 0.0506101, and 0.326287. One target task contributes 68.43% of the total loss; the largest task/seed loss is 145.170. All 500 tasks and all seeds remain included. This instability does not occur in that model's common-solver representation row (MSE 0.0412452). No settings were changed after this observation.

## Interpretation

Both G34-SVD and Pair-SVD select 3/6/15 latent coefficients for ranks 1/2/5 at every source sample count. The selected equality-code counts are 3/204/204. Their test MSEs of roughly 1e-15 are at the rounding-error scale of this noiseless float32-generated linear data; the small differences among these errors are not meaningful method rankings.

At rank2 the old G34 index imposes false equalities. Learning code relations with that fixed index improves test MSE only to 0.163829 at n20/n200 or 0.447001 at n80. Selecting the index together with the linear map chooses 204 distinct local weights and removes this approximation error. At rank5 the old index already has the needed 204 distinct weights, so adding the map suffices.

Dense-SVD without any index reaches the same numerical floor at n80/n200. At n20 its MSEs are 0.0964464/0.0542231/0.0650477 for ranks1/2/5. The recovered mask enables a source refit with only three inputs per output, making 20 observations sufficient there. LC-SVD supplied with the true width-3 architecture matches the recovered-bank methods. The observed gain at n20 combines recovered sparsity with linear relations; at ranks2/5 it does not require nontrivial exact equality groups.

Learning these relations uses multiple source functions (400/800 here), not one dense layer. Generalization to noisy data, nonlinear networks, or image benchmarks has not been established by this follow-up. The native competitors retain their registered finite training budgets; exact source regression and SVD exploit the special linear setting directly.

## Artifacts and limits

`comparison_metrics.csv` includes task CI95, selected dimensions, and evaluation times. `cases/*/per_task.npz` preserves all test-task losses and native training seeds; `results.json` includes paired differences. `choices.json` preserves every validation candidate. `choices_frozen.json` predates test generation. Models contain the signed index and code map.

`banks/rank*_n*/g34.npz` and `pair.npz` export the 18 selected compact banks. Each contains an output-by-input signed index, the selected code map, and the validation-selected ridge. `banks/README.md` explains support-only recovery and prediction; `banks/manifest.json` records dimensions.

Incremental refit/SVD/validation times are recorded per case. Earlier bank construction and baseline meta-training are reused and excluded from those timings. Concurrent CPU evaluation times are not a controlled speed benchmark. No manuscript files are edited.
