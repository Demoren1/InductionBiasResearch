# Exploratory G35 on nonlinear real tasks

A registered follow-up in previously studied class/person environments. New development episodes select every hyperparameter before new common target episodes are generated. MNIST/Fashion use class-disjoint source0–4, development5–6 and target7–9, with a784–16–2 ReLU MLP. HAR uses a561–16–6 ReLU MLP and the existing disjoint source/development/official-test people. Target support/query counts are32/200 for images and18/30 for HAR. There are192 image episodes and180 HAR episodes. Query images come from official test and support from official train.

G35 transfers a signed bank plus learned cross-code covariance. The strict variant has a small task latent vector; the residual variant retains all original code parameters and changes their shared linear map and initialization distribution. Hidden biases and output heads are always fresh. Neither source numeric initializations nor target query labels enter adaptation.

The source banks of MSR/LUT-Q/Yeh are reused, while a matched nonlinear MTSR control is newly fitted. All use one common fresh target Adam adapter and matched two-stage development search. These are FC/structure-transfer adaptations, not the unchanged published CNN or meta-initialization benchmarks. Continuous bases are never clustered into indexers.

Image CI95 describe episodes conditional on this fixed class environment, with possible reuse of images and class pairs. HAR intervals aggregate by nine target people. Neither interval measures independent repetitions of source bank recovery. Failed target fits remain failures.

## Selected constrained models

| Method | HAR accuracy | Fashion accuracy | MNIST accuracy |
|---|---:|---:|---:|
| Dense | 85.11% ± 2.61 | 96.30% ± 0.54 | 92.67% ± 0.50 |
| G34 | 84.35% ± 2.61 | 96.14% ± 0.57 | 92.08% ± 0.49 |
| Pair | 83.94% ± 3.55 | 95.50% ± 0.54 | 91.65% ± 0.56 |
| G35 (G34 bank) | 83.52% ± 2.66 | 95.95% ± 0.58 | 91.44% ± 0.54 |
| G35 (Pair bank) | 84.04% ± 3.82 | 95.86% ± 0.56 | 91.28% ± 0.54 |
| MSR-FC basis | 80.93% ± 2.59 | 95.99% ± 0.56 | 91.90% ± 0.50 |
| MTSR-FC basis | 87.46% ± 2.83 | 95.83% ± 0.56 | 92.18% ± 0.47 |
| LUT-Q | 85.00% ± 2.42 | 96.18% ± 0.55 | 91.33% ± 0.59 |
| Yeh MAP | 83.81% ± 3.04 | 95.90% ± 0.55 | 91.88% ± 0.56 |
| Yeh soft | 79.76% ± 3.88 | 94.43% ± 0.50 | 89.47% ± 0.69 |
| Full-index covariance | 85.83% ± 3.47 | 95.19% ± 0.59 | 91.23% ± 0.54 |

The G35 rows above always show the new transformation itself. The full-index covariance control has no nontrivial equality grouping. Validation-selected fallbacks are disclosed below; a fallback to G34 or Dense must not be credited to a new learned grouping.

## Selected G35 capacity and paired comparison

| Dataset | Bank | m | Map d | Residual | First-layer task parameters | All task parameters | Target updates | G35 − G34 accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| har | G35 (G34 bank) | 512 | 2 | 1.0 | 512 | 630 | 525 | -0.83 ± 1.09 pp |
| har | G35 (Pair bank) | 32 | 4 | 0.0 | 4 | 122 | 125 | -0.31 ± 1.94 pp |
| har | Full-index covariance | 8976 | 2 | 0.0 | 2 | 120 | 125 | +1.48 ± 2.52 pp |
| fashion | G35 (G34 bank) | 512 | 2 | 1.0 | 512 | 562 | 100 | -0.18 ± 0.16 pp |
| fashion | G35 (Pair bank) | 512 | 2 | 1.0 | 512 | 562 | 300 | -0.28 ± 0.20 pp |
| fashion | Full-index covariance | 12544 | 2 | 1.0 | 12544 | 12594 | 150 | -0.94 ± 0.29 pp |
| mnist | G35 (G34 bank) | 512 | 2 | 1.0 | 512 | 562 | 150 | -0.64 ± 0.29 pp |
| mnist | G35 (Pair bank) | 512 | 2 | 1.0 | 512 | 562 | 175 | -0.80 ± 0.42 pp |
| mnist | Full-index covariance | 12544 | 2 | 1.0 | 12544 | 12594 | 175 | -0.85 ± 0.44 pp |

## Validation-selected fallbacks

| Dataset | Selection rule | Selected candidate | Target accuracy |
|---|---|---|---:|
| har | g35_g34_with_base | g34_k512 | 84.35% |
| har | g35_g34_with_full | dense | 85.11% |
| har | g35_pair_with_base | pair_k512 | 83.94% |
| har | g35_pair_with_full | dense | 85.11% |
| har | msr_with_full | dense | 85.11% |
| har | mtsr_with_full | mtsr_k512 | 87.46% |
| fashion | g35_g34_with_base | g34_k512 | 96.14% |
| fashion | g35_g34_with_full | dense | 96.30% |
| fashion | g35_pair_with_base | pair_k512 | 95.50% |
| fashion | g35_pair_with_full | dense | 96.30% |
| fashion | msr_with_full | dense | 96.30% |
| fashion | mtsr_with_full | dense | 96.30% |
| mnist | g35_g34_with_base | g34_k512 | 92.08% |
| mnist | g35_g34_with_full | dense | 92.67% |
| mnist | g35_pair_with_base | pair_k512 | 91.65% |
| mnist | g35_pair_with_full | dense | 92.67% |
| mnist | msr_with_full | dense | 92.67% |
| mnist | mtsr_with_full | dense | 92.67% |

## Artifacts and limits

All per-task losses, paired differences and numerical choices are in each dataset's results.json/per_task.npz. comparison_metrics.csv contains CE, accuracy, intervals and capacity. All coarse/refinement trajectories and source configurations remain saved. New target predictions were completed and frozen before query-label scoring. Hyperparameters were not changed after test. Full m=N endpoints and shared-covariance controls remain explicit.

These exploratory results use fixed environments and finite optimizer/source budgets. They do not prove an identifiability result, a universal improvement, or novelty of the covariance/SVD construction. The provisional name G35 does not imply such a claim. The manuscript is not edited by this workflow.
