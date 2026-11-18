# Source sample ablation

New paired cohort: the source functions, nested observations and target tasks are common across source sample counts. Target adaptation always uses one support observation and 19 queries. All representation, m, ridge and optimizer choices were frozen on validation before opening the independent test panels (100 tasks for rank 1; 200 for ranks 2/5).

Native results average three numerical training seeds with a fixed recovered bank. G34/Pair + MAML transfer numerical initialization and learned rates. Fresh results discard both, use the source-seed-0 representation, and fit a validation-selected ridge on the target support. Fresh MSR/MTSR/Joint rows are structure diagnostics, not native adaptation.

All MSEs are lower-is-better. Task confidence intervals and training-seed SD are exported in comparison_metrics.csv and per-case results.json. Task intervals are conditional on the generated environment and the fixed source geometry. Failed adaptations remain failures rather than being removed.

Tuning scope: MSR dimensions [3, 6, 15, 30, 70, 204]; MTSR dimensions [3, 6, 15, 30, 70, 204]. Both have validation-selected source outer rates. The original source-count study keeps MTSR at m=70; its expanded dimension search is a separate follow-up with new held-out tasks.

## Native comparison

| Rank | Source examples | MSR fixed | MSR tuned | MTSR tuned | G34 + MAML | Pair + MAML |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 0.00251588 | 0.000976952 | 0.0865025 | 6.27312e-05 | 6.32054e-05 |
| 1 | 80 | 0.00344294 | 0.000603102 | 0.0312681 | 6.28138e-05 | 6.30551e-05 |
| 1 | 200 | 0.001867 | 0.000480248 | 0.0218587 | 6.71305e-05 | 6.73012e-05 |
| 2 | 20 | 0.0147164 | 0.0128533 | 0.0952506 | 0.976983 | 1.468 |
| 2 | 80 | 0.0215886 | 0.0267765 | 0.0326133 | 0.579779 | 1.46805 |
| 2 | 200 | 0.0182262 | 0.014418 | 0.0187351 | 0.779152 | 1.46759 |
| 5 | 20 | 0.0957819 | 0.0639935 | 0.0669429 | 0.710174 | 0.718122 |
| 5 | 80 | 0.104411 | 0.0567386 | 0.0240352 | 0.71828 | 0.71825 |
| 5 | 200 | 0.0914569 | 0.0441501 | 0.0118513 | 0.718547 | 0.718552 |

## Fresh structure comparison

| Rank | Source examples | MSR basis | MTSR basis | G34 | Pair |
|---|---:|---:|---:|---:|---:|
| 1 | 20 | 0.000579688 | 5.66326e-06 | 1.99371e-15 | 1.99371e-15 |
| 1 | 80 | 0.00065846 | 9.50428e-07 | 1.99371e-15 | 1.99371e-15 |
| 1 | 200 | 0.000532156 | 3.01963e-07 | 1.99371e-15 | 1.99371e-15 |
| 2 | 20 | 0.0127405 | 1.09607e-05 | 0.744401 | 1.28817 |
| 2 | 80 | 0.00948301 | 1.32117e-06 | 0.549254 | 1.28817 |
| 2 | 200 | 0.014936 | 9.79555e-07 | 0.744401 | 1.28817 |
| 5 | 20 | 0.0503635 | 5.38432e-06 | 0.607331 | 0.607331 |
| 5 | 80 | 0.0326233 | 1.497e-06 | 0.607331 | 0.607331 |
| 5 | 200 | 0.0416775 | 1.57869e-06 | 0.607331 | 0.607331 |

## All native baselines: fixed settings

| Rank | Source examples | MAML-FC | MAML-LC | MAML-Conv | MTSR-FC | MSR-Joint-FC | MSR-FC |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 2.95977 | 2.53482 | 0.000889732 | 0.104962 | 0.00575705 | 0.00251588 |
| 1 | 80 | 2.93997 | 2.53178 | 0.000881229 | 0.0427698 | 0.00957057 | 0.00344294 |
| 1 | 200 | 2.93355 | 2.53312 | 0.000914202 | 0.0227892 | 0.00499306 | 0.001867 |
| 2 | 20 | 1.91165 | 1.64993 | 0.367713 | 0.126375 | failed | 0.0147164 |
| 2 | 80 | 1.90451 | 1.64923 | 0.367729 | 0.0451588 | 0.0537029 | 0.0215886 |
| 2 | 200 | 1.90242 | 1.64824 | 0.36772 | 0.0251618 | 0.048255 | 0.0182262 |
| 5 | 20 | 0.901563 | 0.796007 | 0.342761 | 0.0946573 | failed | 0.0957819 |
| 5 | 80 | 0.898985 | 0.795625 | 0.342767 | 0.0240352 | failed | 0.104411 |
| 5 | 200 | 0.898871 | 0.795669 | 0.342768 | 0.0123325 | failed | 0.0914569 |

## All native baselines: validation-tuned settings

| Rank | Source examples | MAML-FC | MAML-LC | MAML-Conv | MTSR-FC | MSR-Joint-FC | MSR-FC |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 2.95632 | 2.3511 | 9.82673e-05 | 0.0865025 | 0.0112546 | 0.000976952 |
| 1 | 80 | 2.93851 | 2.34937 | 9.63542e-05 | 0.0312681 | 0.00493666 | 0.000603102 |
| 1 | 200 | 2.93355 | 2.35042 | 0.00010167 | 0.0218587 | 0.00266839 | 0.000480248 |
| 2 | 20 | 1.91056 | 1.52157 | 0.36732 | 0.0952506 | failed | 0.0128533 |
| 2 | 80 | 1.90407 | 1.52119 | 0.367323 | 0.0326133 | 26170.8 | 0.0267765 |
| 2 | 200 | 1.90225 | 1.51994 | 0.367321 | 0.0187351 | 9504.68 | 0.014418 |
| 5 | 20 | 0.901353 | 0.724689 | 0.342623 | 0.0669429 | 0.244955 | 0.0639935 |
| 5 | 80 | 0.898954 | 0.724701 | 0.342623 | 0.0240352 | 0.244521 | 0.0567386 |
| 5 | 200 | 0.898871 | 0.724907 | 0.342624 | 0.0118513 | failed | 0.0441501 |

## All fresh baselines

| Rank | Source examples | FC | LC | Conv | MTSR basis | Joint basis | MSR basis |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 2.92308 | 1.99104 | 1.99371e-15 | 5.66326e-06 | 0.00100984 | 0.000579688 |
| 1 | 80 | 2.92308 | 1.99104 | 1.99371e-15 | 9.50428e-07 | 0.0009076 | 0.00065846 |
| 1 | 200 | 2.92308 | 1.99104 | 1.99371e-15 | 3.01963e-07 | 0.000597128 | 0.000532156 |
| 2 | 20 | 1.89878 | 1.28817 | 0.367125 | 1.09607e-05 | 0.0179128 | 0.0127405 |
| 2 | 80 | 1.89878 | 1.28817 | 0.367125 | 1.32117e-06 | 0.0139692 | 0.00948301 |
| 2 | 200 | 1.89878 | 1.28817 | 0.367125 | 9.79555e-07 | 0.0459584 | 0.014936 |
| 5 | 20 | 0.897836 | 0.607331 | 0.342294 | 5.38432e-06 | 0.116144 | 0.0503635 |
| 5 | 80 | 0.897836 | 0.607331 | 0.342294 | 1.497e-06 | 0.0720351 | 0.0326233 |
| 5 | 200 | 0.897836 | 0.607331 | 0.342294 | 1.57869e-06 | 0.06562 | 0.0416775 |

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
| 1 | 20 | 2.95632 | 2.92308 | 3 | 3 |
| 1 | 80 | 2.93851 | 2.92308 | 3 | 3 |
| 1 | 200 | 2.93355 | 2.92308 | 3 | 3 |
| 2 | 20 | 1.91056 | 1.89878 | 70 | 70 |
| 2 | 80 | 1.90407 | 1.89878 | 15 | 70 |
| 2 | 200 | 1.90225 | 1.89878 | 15 | 70 |
| 5 | 20 | 0.901353 | 0.897836 | 204 | 204 |
| 5 | 80 | 0.898954 | 0.897836 | 204 | 204 |
| 5 | 200 | 0.898871 | 0.897836 | 204 | 204 |

Budget curves use source seed 0 for every plotted method, including MSR/MTSR horizontal references. The full endpoint is shown separately. Requested budgets can exceed the number of occupied codes; per-case results.json records both. Main sample-count curves use three-seed native means and one source-seed-0 fresh basis. Values below 1e-14 are plotted at 1e-14.
