# Why source code covariance may fail to transfer

This is an exploratory diagnosis after the negative nonlinear G35 test. G34 and Pair are the primary baselines. New development episodes use seed35111601; no new nonlinear target data are generated or read. The same fixed class/person environments have been studied before. Numerical choices below are validation choices, not test results.

## Source parameter variability

The four retained numeric fits can differ in restart, learning rate and activity setting. They are not four new source functions. Weight vectors are normalized by global layer norm. Within-function variation is compared with variation of the per-function mean; this is a parameter diagnostic, not a decomposition of functional error.

| Dataset | Bank | Source functions | Source H CE, best / worst retained mean | Within fraction | Top2 energy | Mean top2 subspace overlap |
|---|---|---:|---:|---:|---:|---:|
| har | g34 | 4 | 0.04313 / 0.07023 | 0.767 | 0.580 | 0.134 |
| har | pair | 4 | 0.04405 / 0.07907 | 0.740 | 0.629 | 0.155 |
| fashion | g34 | 10 | 0.0792 / 0.08511 | 0.762 | 0.270 | 0.108 |
| fashion | pair | 10 | 0.08216 / 0.08874 | 0.748 | 0.359 | 0.274 |
| mnist | g34 | 10 | 0.01487 / 0.01749 | 0.749 | 0.378 | 0.341 |
| mnist | pair | 10 | 0.01765 / 0.02222 | 0.648 | 0.476 | 0.433 |

## Controlled nonlinear development ablation

Baseline m is retuned over8/16/32/64/128/512. Diagnostic maps use the same fixed m512 bank as the former selected nonlinear G35. Full-N Dense remains an explicit control. Code-coordinate shrinkage interpolates between the exact baseline and source second moments. Diagonal controls omit cross-code correlations. Group-isotropic controls change group energy without learning any correlation. Positive isotropic floors preserve the bank's function family. All use the same fresh Adam and matched coarse/refine selection.

| Dataset | Family | Candidate | Development CE | Updates |
|---|---|---|---:|---:|
| har | dense | dense | 0.331462 | 375 |
| har | g34 | g34_k512 | 0.319704 | 500 |
| har | g35_g34 | g35_g34_k512_d2_r1 | 0.380334 | 400 |
| har | g35_pair | g35_pair_k512_d2_r1 | 0.438900 | 325 |
| har | group_isotropic_g34 | group_isotropic_g34 | 0.352952 | 450 |
| har | group_isotropic_pair | group_isotropic_pair | 0.390999 | 350 |
| har | pair | pair_k512 | 0.355698 | 950 |
| har | shrink_g34_best_diagonal | shrink_g34_best_diagonal_l0.1 | 0.316820 | 475 |
| har | shrink_g34_best_full | shrink_g34_best_full_l0.01 | 0.314891 | 475 |
| har | shrink_g34_pooled_diagonal | shrink_g34_pooled_diagonal_l0.75 | 0.315716 | 475 |
| har | shrink_g34_pooled_full | shrink_g34_pooled_full_l0.05 | 0.308248 | 525 |
| har | shrink_pair_best_diagonal | shrink_pair_best_diagonal_l0.05 | 0.330064 | 300 |
| har | shrink_pair_best_full | shrink_pair_best_full_l0.01 | 0.358811 | 625 |
| har | shrink_pair_pooled_diagonal | shrink_pair_pooled_diagonal_l0.1 | 0.330917 | 275 |
| har | shrink_pair_pooled_full | shrink_pair_pooled_full_l0.01 | 0.327419 | 625 |
| fashion | dense | dense | 0.056692 | 75 |
| fashion | g34 | g34_k512 | 0.049144 | 525 |
| fashion | g35_g34 | g35_g34_k512_d2_r1 | 0.050920 | 500 |
| fashion | g35_pair | g35_pair_k512_d2_r1 | 0.077538 | 275 |
| fashion | group_isotropic_g34 | group_isotropic_g34 | 0.049296 | 500 |
| fashion | group_isotropic_pair | group_isotropic_pair | 0.061146 | 225 |
| fashion | pair | pair_k512 | 0.075472 | 500 |
| fashion | shrink_g34_best_diagonal | shrink_g34_best_diagonal_l1 | 0.048713 | 525 |
| fashion | shrink_g34_best_full | shrink_g34_best_full_l0.05 | 0.048580 | 525 |
| fashion | shrink_g34_pooled_diagonal | shrink_g34_pooled_diagonal_l1 | 0.048793 | 525 |
| fashion | shrink_g34_pooled_full | shrink_g34_pooled_full_l0.01 | 0.049253 | 525 |
| fashion | shrink_pair_best_diagonal | shrink_pair_best_diagonal_l1 | 0.072726 | 250 |
| fashion | shrink_pair_best_full | shrink_pair_best_full_l0.01 | 0.080855 | 200 |
| fashion | shrink_pair_pooled_diagonal | shrink_pair_pooled_diagonal_l1 | 0.071133 | 250 |
| fashion | shrink_pair_pooled_full | shrink_pair_pooled_full_l0.01 | 0.080227 | 650 |
| mnist | dense | dense | 0.166541 | 25 |
| mnist | g34 | g34_k512 | 0.182598 | 25 |
| mnist | g35_g34 | g35_g34_k512_d2_r1 | 0.198033 | 125 |
| mnist | g35_pair | g35_pair_k512_d2_r1 | 0.225372 | 50 |
| mnist | group_isotropic_g34 | group_isotropic_g34 | 0.181267 | 125 |
| mnist | group_isotropic_pair | group_isotropic_pair | 0.195143 | 125 |
| mnist | pair | pair_k512 | 0.188740 | 75 |
| mnist | shrink_g34_best_diagonal | shrink_g34_best_diagonal_l0.75 | 0.179556 | 25 |
| mnist | shrink_g34_best_full | shrink_g34_best_full_l0.01 | 0.181098 | 25 |
| mnist | shrink_g34_pooled_diagonal | shrink_g34_pooled_diagonal_l1 | 0.178827 | 150 |
| mnist | shrink_g34_pooled_full | shrink_g34_pooled_full_l0.01 | 0.182746 | 25 |
| mnist | shrink_pair_best_diagonal | shrink_pair_best_diagonal_l0.05 | 0.188287 | 75 |
| mnist | shrink_pair_best_full | shrink_pair_best_full_l0.01 | 0.199616 | 100 |
| mnist | shrink_pair_pooled_diagonal | shrink_pair_pooled_diagonal_l0.1 | 0.187068 | 75 |
| mnist | shrink_pair_pooled_full | shrink_pair_pooled_full_l0.01 | 0.193753 | 75 |

## Matching synthetic covariance sample counts

Only the number of source functions used by SVD changes. The recovered index and the full-data SVD dimension remain frozen from the previous experiment. Every source function supplies200 examples. At most min(source count, frozen dimension) coefficients are retained. Each partial count uses eight precommitted subsets; the full count uses one. The index itself still benefited from the original full source set. New synthetic audit tasks have one support observation and19queries; no choices are made from their errors.

| Rank | Bank | Source functions | Latent dimension | Audit MSE |
|---:|---|---:|---:|---:|
| 1 | g34 | 4 | 3 | 2.08137e-15 |
| 1 | g34 | 10 | 3 | 2.08137e-15 |
| 1 | g34 | 20 | 3 | 2.08137e-15 |
| 1 | g34 | 50 | 3 | 2.08137e-15 |
| 1 | g34 | 400 | 3 | 2.08137e-15 |
| 1 | pair | 4 | 3 | 2.08137e-15 |
| 1 | pair | 10 | 3 | 2.08137e-15 |
| 1 | pair | 20 | 3 | 2.08137e-15 |
| 1 | pair | 50 | 3 | 2.08137e-15 |
| 1 | pair | 400 | 3 | 2.08137e-15 |
| 2 | g34 | 4 | 4 | 0.448365 |
| 2 | g34 | 10 | 6 | 1.40394e-15 |
| 2 | g34 | 20 | 6 | 1.3513e-15 |
| 2 | g34 | 50 | 6 | 1.34046e-15 |
| 2 | g34 | 800 | 6 | 1.33484e-15 |
| 2 | pair | 4 | 4 | 0.448365 |
| 2 | pair | 10 | 6 | 1.40394e-15 |
| 2 | pair | 20 | 6 | 1.3513e-15 |
| 2 | pair | 50 | 6 | 1.34046e-15 |
| 2 | pair | 800 | 6 | 1.33484e-15 |
| 5 | g34 | 4 | 4 | 0.537095 |
| 5 | g34 | 10 | 10 | 0.210128 |
| 5 | g34 | 20 | 15 | 9.29934e-16 |
| 5 | g34 | 50 | 15 | 8.53506e-16 |
| 5 | g34 | 800 | 15 | 8.45795e-16 |
| 5 | pair | 4 | 4 | 0.537095 |
| 5 | pair | 10 | 10 | 0.210128 |
| 5 | pair | 20 | 15 | 9.29934e-16 |
| 5 | pair | 50 | 15 | 8.53506e-16 |
| 5 | pair | 800 | 15 | 8.45795e-16 |
