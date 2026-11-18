# Source sample ablation

New paired cohort: the source functions, nested observations and target tasks are common across source sample counts. Target adaptation always uses one support observation and 19 queries. All representation, m, ridge and optimizer choices were frozen on validation before opening the independent test panels (100 tasks for rank 1; 200 for ranks 2/5).

Native results average three numerical training seeds with a fixed recovered bank. G34/Pair + MAML transfer numerical initialization and learned rates. Fresh results discard both, use the source-seed-0 representation, and fit a validation-selected ridge on the target support. Fresh MSR/MTSR/Joint rows are structure diagnostics, not native adaptation.

All MSEs are lower-is-better. Task confidence intervals and training-seed SD are exported in comparison_metrics.csv and per-case results.json. Task intervals are conditional on the generated environment and the fixed source geometry. Failed adaptations remain failures rather than being removed.

Tuning scope: MSR searches m=3/6/15/30/70/204; MTSR here retains m=70 and selects its source outer learning rate. An additional MTSR dimension sweep is registered separately and will use new held-out target tasks.

## Native comparison

| Rank | Source examples | MSR fixed | MSR tuned | MTSR tuned | G34 + MAML | Pair + MAML |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 0.00257937 | 0.00103131 | 0.109157 | 0.000151484 | 0.000145588 |
| 1 | 80 | 0.00403069 | 0.000700702 | 0.0407044 | 0.00015427 | 0.000146953 |
| 1 | 200 | 0.00214278 | 0.000567124 | 0.0292372 | 0.000162849 | 0.000155568 |
| 2 | 20 | 0.0146781 | 0.0109953 | 0.0959631 | 0.963413 | 1.43141 |
| 2 | 80 | 0.0208636 | 0.0245612 | 0.0352163 | 0.569148 | 1.43143 |
| 2 | 200 | 0.0169476 | 0.0143098 | 0.0200865 | 0.783069 | 1.43113 |
| 5 | 20 | 0.101799 | 0.0556005 | 0.103219 | 0.745789 | 0.755298 |
| 5 | 80 | 0.101617 | 0.0518716 | 0.0292388 | 0.755304 | 0.755274 |
| 5 | 200 | 0.0969976 | 0.0443026 | 0.0151954 | 0.755192 | 0.755195 |

## Fresh structure comparison

| Rank | Source examples | MSR basis | MTSR basis | G34 | Pair |
|---|---:|---:|---:|---:|---:|
| 1 | 20 | 0.000615156 | 0.00350884 | 2.22872e-15 | 2.22872e-15 |
| 1 | 80 | 0.000757568 | 7.26913e-05 | 2.22872e-15 | 2.22872e-15 |
| 1 | 200 | 0.000596086 | 4.06437e-05 | 2.22872e-15 | 2.22872e-15 |
| 2 | 20 | 0.0123008 | 0.000810713 | 0.656581 | 1.2752 |
| 2 | 80 | 0.00968035 | 1.52254e-05 | 0.55181 | 1.2752 |
| 2 | 200 | 0.0153521 | 3.3646e-05 | 0.656581 | 1.2752 |
| 5 | 20 | 0.053739 | 0.00244537 | 0.638291 | 0.638291 |
| 5 | 80 | 0.0349133 | 7.37895e-06 | 0.638291 | 0.638291 |
| 5 | 200 | 0.0432208 | 1.06571e-05 | 0.638291 | 0.638291 |

## All native baselines: fixed settings

| Rank | Source examples | MAML-FC | MAML-LC | MAML-Conv | MTSR-FC | MSR-Joint-FC | MSR-FC |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 3.29214 | 2.83994 | 0.00110227 | 0.126246 | 0.00708729 | 0.00257937 |
| 1 | 80 | 3.27383 | 2.83782 | 0.00109079 | 0.0529497 | 0.0136954 | 0.00403069 |
| 1 | 200 | 3.26665 | 2.8385 | 0.00113327 | 0.0304411 | 0.00823312 | 0.00214278 |
| 2 | 20 | 1.87051 | 1.62483 | 0.369193 | 0.127122 | failed | 0.0146781 |
| 2 | 80 | 1.86554 | 1.62539 | 0.369204 | 0.0472442 | 3.50154e+07 | 0.0208636 |
| 2 | 200 | 1.86425 | 1.62453 | 0.369197 | 0.0263635 | 0.421604 | 0.0169476 |
| 5 | 20 | 0.945911 | 0.833454 | 0.359048 | 0.108013 | failed | 0.101799 |
| 5 | 80 | 0.94252 | 0.832799 | 0.359065 | 0.0292388 | failed | 0.101617 |
| 5 | 200 | 0.942169 | 0.832728 | 0.359065 | 0.0159269 | failed | 0.0969976 |

## All native baselines: validation-tuned settings

| Rank | Source examples | MAML-FC | MAML-LC | MAML-Conv | MTSR-FC | MSR-Joint-FC | MSR-FC |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 3.28827 | 2.63158 | 0.000132268 | 0.109157 | 0.0428739 | 0.00103131 |
| 1 | 80 | 3.27155 | 2.63121 | 0.000130133 | 0.0407044 | 0.0142749 | 0.000700702 |
| 1 | 200 | 3.26665 | 2.63069 | 0.000137397 | 0.0292372 | 0.00942261 | 0.000567124 |
| 2 | 20 | 1.86992 | 1.49976 | 0.368963 | 0.0959631 | failed | 0.0109953 |
| 2 | 80 | 1.86513 | 1.50101 | 0.368962 | 0.0352163 | 0.20161 | 0.0245612 |
| 2 | 200 | 1.86388 | 1.50045 | 0.368962 | 0.0200865 | 0.0311637 | 0.0143098 |
| 5 | 20 | 0.94547 | 0.761768 | 0.358555 | 0.103219 | 0.264837 | 0.0556005 |
| 5 | 80 | 0.942293 | 0.761563 | 0.358561 | 0.0292388 | 0.264412 | 0.0518716 |
| 5 | 200 | 0.942169 | 0.76152 | 0.358562 | 0.0151954 | failed | 0.0443026 |

## All fresh baselines

| Rank | Source examples | FC | LC | Conv | MTSR basis | Joint basis | MSR basis |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 3.25895 | 2.2135 | 2.22872e-15 | 0.00350884 | 0.00112382 | 0.000615156 |
| 1 | 80 | 3.25895 | 2.2135 | 2.22872e-15 | 7.26913e-05 | 0.00103523 | 0.000757568 |
| 1 | 200 | 3.25895 | 2.2135 | 2.22872e-15 | 4.06437e-05 | 0.000684951 | 0.000596086 |
| 2 | 20 | 1.8638 | 1.2752 | 0.368934 | 0.000810713 | 0.0173648 | 0.0123008 |
| 2 | 80 | 1.8638 | 1.2752 | 0.368934 | 1.52254e-05 | 0.0139063 | 0.00968035 |
| 2 | 200 | 1.8638 | 1.2752 | 0.368934 | 3.3646e-05 | 0.0391943 | 0.0153521 |
| 5 | 20 | 0.940026 | 0.638291 | 0.358393 | 0.00244537 | 0.12677 | 0.053739 |
| 5 | 80 | 0.940026 | 0.638291 | 0.358393 | 7.37895e-06 | 0.0771002 | 0.0349133 |
| 5 | 200 | 0.940026 | 0.638291 | 0.358393 | 1.06571e-05 | 0.0628937 | 0.0432208 |

## Selected dimensions

MSR dimensions count continuous latent coefficients; G34/Pair dimensions count nonzero hard equality groups. They are different representation classes. The fixed MSR row always uses m=70 and outer LR=0.0005. The following dimensions are chosen independently for native and fresh adaptation.

| Rank | Source examples | Native MSR m | Native G34 m | Native Pair m | Fresh MSR m | Fresh G34 m | Fresh Pair m |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 70 | 3 | 3 | 3 | 3 | 3 |
| 1 | 80 | 15 | 3 | 3 | 15 | 3 | 3 |
| 1 | 200 | 70 | 3 | 3 | 6 | 3 | 3 |
| 2 | 20 | 204 | 70 | 70 | 204 | 70 | 204 |
| 2 | 80 | 204 | 15 | 70 | 204 | 15 | 204 |
| 2 | 200 | 204 | 15 | 70 | 204 | 70 | 204 |
| 5 | 20 | 30 | 204 | 204 | 30 | 204 | 204 |
| 5 | 80 | 204 | 204 | 204 | 204 | 204 | 204 |
| 5 | 200 | 70 | 204 | 204 | 30 | 204 | 204 |

## Full m=N control

m=N=4760 gives one independent parameter per weight and is the MAML-FC identity endpoint. The with-full selections may choose that endpoint using validation; its performance is not evidence of recovered sharing.

| Rank | Source examples | Native full FC | Fresh full FC | Native G34 choice including full | Native Pair choice including full |
|---|---:|---:|---:|---|---|
| 1 | 20 | 3.28827 | 3.25895 | 3 | 3 |
| 1 | 80 | 3.27155 | 3.25895 | 3 | 3 |
| 1 | 200 | 3.26665 | 3.25895 | 3 | 3 |
| 2 | 20 | 1.86992 | 1.8638 | 70 | 70 |
| 2 | 80 | 1.86513 | 1.8638 | 15 | 70 |
| 2 | 200 | 1.86388 | 1.8638 | 15 | 70 |
| 5 | 20 | 0.94547 | 0.940026 | 204 | 204 |
| 5 | 80 | 0.942293 | 0.940026 | 204 | 204 |
| 5 | 200 | 0.942169 | 0.940026 | 204 | 204 |

Budget curves use source seed 0 for every plotted method, including MSR/MTSR horizontal references. The full endpoint is shown separately. Requested budgets can exceed the number of occupied codes; per-case results.json records both. Main sample-count curves use three-seed native means and one source-seed-0 fresh basis. Values below 1e-14 are plotted at 1e-14.
