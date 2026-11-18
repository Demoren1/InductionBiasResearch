# G35 compared with G34 and Pair

Completed nonlinear test outcomes; each covariance extension is compared with its own baseline. These are fixed source environments, with192 image episodes per dataset and180 HAR episodes from9 target people. Intervals use episodes for images and target people for HAR; they do not cover independent source-bank recovery. Image episodes can reuse images.

| Method | HAR accuracy | Fashion-MNIST accuracy | MNIST accuracy |
|---|---:|---:|---:|
| G34 | 84.35% ± 2.61 | 96.14% ± 0.57 | 92.08% ± 0.49 |
| G35 on G34 bank | 83.52% ± 2.66 | 95.95% ± 0.58 | 91.44% ± 0.54 |
| Pair | 83.94% ± 3.55 | 95.50% ± 0.54 | 91.65% ± 0.56 |
| G35 on Pair bank | 84.04% ± 3.82 | 95.86% ± 0.56 | 91.28% ± 0.54 |

| Dataset | G35-G34 minus G34 | G35-Pair minus Pair |
|---|---:|---:|
| har | -0.83 ± 1.09 pp | +0.09 ± 2.06 pp |
| fashion | -0.18 ± 0.16 pp | +0.36 ± 0.23 pp |
| mnist | -0.64 ± 0.29 pp | -0.37 ± 0.44 pp |

G35 on the G34 bank selects m512, d2 and positive residual1 on all three datasets. It retains the same512 first-layer parameters and the same bank function family. Its loss relative to G34 therefore concerns numerical coordinates, initialization and their interaction with the fixed Adam budget. It does not demonstrate a worse recovered index: the index is exactly the same. The residual map is invertible on the active groups.

G35 on the Pair bank also retains512 first-layer parameters on the image datasets, but selects a strict4-parameter map on HAR from a32-code bank. The HAR Pair comparison therefore changes capacity as well as adaptation. Pair results are mixed, and should not be described as G35 uniformly losing against Pair.

The current evidence supports retaining G34 and Pair as practical baselines. It does not establish universal superiority of either method. The distinct diagnostic study in artifacts/g35_diagnosis_2026-11-14 investigates source-count limitations, variability of retained numerical fits, group scaling and off-diagonal covariance on new development episodes. No nonlinear test data are created by that diagnostic workflow. The manuscript remains untouched by this series.
