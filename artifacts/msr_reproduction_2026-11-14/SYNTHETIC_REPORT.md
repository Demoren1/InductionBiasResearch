# MSR synthetic reproduction and comparison

These are new experiments following the paper's task counts, support sizes and optimization schedules. The public code differs from the paper; see the registered protocol. Native rows retain learned numeric initializations. G34/Pair + MAML are explicitly linear-layer adaptations with native-style meta-initialization. They are separate from the structure-only results below.

Each native result averages three training seeds on the same held-out tasks. The 95% interval is over task means, conditional on this environment; training-seed variation is saved separately. It is not a confidence interval over new mixing geometries.

Small source data has 2, 2 and 4 examples per source function for ranks 1, 2 and 5; large source data has 20 per function. There are 400 source functions for rank 1 and 800 for ranks 2 and 5. Discrete bank construction uses 32 fitted source functions, with masks estimated from all source functions. These are multi-task MSR comparisons, not evidence of single-source-task recovery.

Fixed translation MSR uses the paper's latent size 70 and outer learning rate 0.0005. For rotations, 0.0005 was inferred from the appendix's reference to the translation setup; the released synthetic default is 0.001. Rotation validation selects 0.001, and both settings are retained below. The rotation learning rate is therefore not an unambiguous author setting.

## Translation: small source dataset

| Method | Rank 1 | Rank 2 | Rank 5 |
|---|---:|---:|---:|
| MAML-FC | 3.116 ± 0.4 | 1.825 ± 0.17 | 0.9965 ± 0.07 |
| MAML-LC | 2.59 ± 0.35 | 1.545 ± 0.14 | 0.8559 ± 0.06 |
| MAML-Conv | 0.0009908 ± 0.00057 | 0.345 ± 0.038 | 0.4251 ± 0.026 |
| MTSR-FC | 2.966 ± 0.31 | 2.018 ± 0.13 | 1.358 ± 0.059 |
| MSR-Joint-FC | 0.1917 ± 0.13 | 0.2213 ± 0.023 | 0.3342 ± 0.034 |
| MSR (fixed reference settings) | 0.07484 ± 0.01 | 0.1641 ± 0.014 | 0.2518 ± 0.014 |
| MSR (validation-tuned m / learning rate) | 0.03854 ± 0.0052 | 0.07853 ± 0.0065 | 248.3 ± 4.4e+02 |
| G34 + MAML | 2.426 ± 0.33 | 1.428 ± 0.14 | 0.7871 ± 0.056 |
| Pair + MAML | 2.426 ± 0.33 | 1.428 ± 0.14 | 0.7871 ± 0.056 |

## Translation: large source dataset

| Method | Rank 1 | Rank 2 | Rank 5 |
|---|---:|---:|---:|
| MAML-FC | 2.973 ± 0.4 | 1.764 ± 0.17 | 0.9783 ± 0.07 |
| MAML-LC | 2.56 ± 0.35 | 1.526 ± 0.14 | 0.8525 ± 0.06 |
| MAML-Conv | 0.0009654 ± 0.00056 | 0.345 ± 0.038 | 0.4251 ± 0.026 |
| MTSR-FC | 0.08941 ± 0.019 | 0.1139 ± 0.014 | 0.1119 ± 0.0091 |
| MSR-Joint-FC | 0.003671 ± 0.0028 | 0.0347 ± 0.0066 | 0.137 ± 0.025 |
| MSR (fixed reference settings) | 0.001624 ± 0.00025 | 0.01242 ± 0.0011 | 0.09251 ± 0.0082 |
| MSR (validation-tuned m / learning rate) | 0.0009178 ± 0.00013 | 0.01477 ± 0.002 | 0.1269 ± 0.12 |
| G34 + MAML | 7.395e-05 ± 5.8e-05 | 0.8821 ± 0.1 | 0.7723 ± 0.055 |
| Pair + MAML | 7.497e-05 ± 6e-05 | 1.334 ± 0.13 | 0.7716 ± 0.055 |

## Rotation and reflection

| Method | 90° | 45° | 45° + flip |
|---|---:|---:|---:|
| MAML-Conv | 0.8187 ± 0.013 | 0.5749 ± 0.0087 | 0.5387 ± 0.0086 |
| MSR (fixed reference settings) | 0.1517 ± 0.0026 | 0.1116 ± 0.0022 | 0.121 ± 0.0019 |
| MSR (validation-tuned m / learning rate) | 0.09262 ± 0.0019 | 0.005626 ± 0.00017 | 0.006056 ± 0.00017 |
| G34 + MAML | 0.2417 ± 0.0042 | 0.3303 ± 0.0051 | 0.2876 ± 0.0046 |
| Pair + MAML | 0.226 ± 0.0038 | 0.1891 ± 0.003 | 0.1619 ± 0.0027 |

The 90° case reproduces the appendix's illustrative setting; the original numerical Table 2 reports 45° and 45° + flip only. Quadratic sufficient statistics preserve the mathematical MSE and gradient; stored float32 statistics limit interpretation of errors below approximately 1e-6.

## Chosen budgets

| Condition | MSR latent | G34 m | Pair m | G34 allowing full | Pair allowing full |
|---|---:|---:|---:|---:|---:|
| flip8 | 432 | 72 | 36 | 72 | 36 |
| rank1_large | 3 | 3 | 3 | 3 | 3 |
| rank1_small | 3 | 512 | 512 | 512 | 512 |
| rank2_large | 30 | 15 | 204 | 15 | 204 |
| rank2_small | 70 | 512 | 512 | 512 | 512 |
| rank5_large | 204 | 204 | 512 | 204 | 512 |
| rank5_small | 70 | 512 | 512 | 512 | 512 |
| rot4 | 108 | 36 | 36 | 36 | 36 |
| rot8 | 216 | 72 | 36 | 72 | 36 |

The validation-tuned rank-5 small-data MSR configuration is unstable: test MSE by training seed is 51.29, 671.14 and 22.36. It was selected before test evaluation and is retained without replacing it using test outcomes. The optional tuned MSR-Joint configuration in rank-2 large data produces nonfinite adaptation on 3 of 600 task/seed pairs; its aggregate is marked failed in the summary. The fixed MSR-Joint configuration in the table remains finite. Raw per-task arrays are preserved.

The full budget is the identity indexer and is exactly the unshared architectural control by definition. Equality there is not evidence of recovered useful sharing. Nominal and actually used codes are both preserved in machine-readable results.

## Fresh numerical parameters

All learned numerical initializations and inner learning rates are discarded. A fresh ridge fit on the single support observation uses a regularization setting selected on independent validation tasks. The source matrices remain continuous for MSR: there is no post hoc clustering. This is a separate structure-only diagnostic, not the original MSR adaptation algorithm.

| Condition | MSR | G34 constrained | Pair constrained | Dense architecture |
|---|---:|---:|---:|---:|
| flip8 | < 1e-6 (numerical floor) | < 1e-6 (numerical floor) | 0.01206 ± 0.00022 | < 1e-6 (numerical floor) |
| rank1_large | 0.0004254 ± 6.2e-05 | < 1e-12 | < 1e-12 | 2.924 ± 0.39 |
| rank1_small | 0.02353 ± 0.0031 | 2.162 ± 0.28 | 2.162 ± 0.28 | 2.924 ± 0.39 |
| rank2_large | 0.007359 ± 0.00063 | 0.8768 ± 0.1 | 1.14 ± 0.11 | 1.754 ± 0.17 |
| rank2_small | 0.1139 ± 0.0097 | 1.255 ± 0.12 | 1.255 ± 0.12 | 1.754 ± 0.17 |
| rank5_large | 0.05201 ± 0.0031 | 0.6501 ± 0.047 | 0.6501 ± 0.047 | 0.9727 ± 0.07 |
| rank5_small | 0.2299 ± 0.013 | 0.6899 ± 0.049 | 0.6899 ± 0.049 | 0.9727 ± 0.07 |
| rot4 | < 1e-6 (numerical floor) | < 1e-6 (numerical floor) | < 1e-6 (numerical floor) | < 1e-6 (numerical floor) |
| rot8 | < 1e-6 (numerical floor) | < 1e-6 (numerical floor) | 0.01414 ± 0.00027 | < 1e-6 (numerical floor) |

Fresh rows use one independently trained source representation selected on validation. Discrete banks are deterministic given their source data; their training-seed repetitions in native rows vary the meta-initialization, not the bank geometry. Applied image experiments remain a separate full 60000-step reconstruction; these synthetic results do not establish their outcomes.
