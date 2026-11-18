# Native synthetic appendix diagnostics

These diagnostics use completed fits, with no retraining or new test evaluation. They reproduce the roles of MSR Table 4 and Figures 6–7 rather than copying their numerical outcomes.

## Joint inner rates

The stored order is v then U. Error bars are sample SD across three training seeds. Signed rates are retained. A negative learned rate is a fitted update rule, not evidence of a theorem about U.

| Case | Outer LR choice | Outer LR | v rate, mean ± seed SD | U rate, mean ± seed SD |
|---|---|---:|---:|---:|
| rank1_small | fixed | 0.0005 | +0.23951 ± 0.00516 | -0.02491 ± 0.00267 |
| rank1_large | fixed | 0.0005 | +0.24090 ± 0.00347 | -0.01364 ± 0.00053 |
| rank2_small | fixed | 0.0005 | +0.28815 ± 0.02792 | -0.01125 ± 0.01260 |
| rank2_large | fixed | 0.0005 | +0.30260 ± 0.00499 | -0.02842 ± 0.00808 |
| rank5_small | fixed | 0.0005 | +0.40358 ± 0.00765 | -0.05153 ± 0.00652 |
| rank5_large | fixed | 0.0005 | +0.38945 ± 0.00728 | -0.04923 ± 0.01054 |
| rank1_small | tuned | 0.001 | +0.23033 ± 0.02507 | -0.00028 ± 0.02319 |
| rank1_large | tuned | 0.0005 | +0.24090 ± 0.00347 | -0.01364 ± 0.00053 |
| rank2_small | tuned | 0.0005 | +0.28815 ± 0.02792 | -0.01125 ± 0.01260 |
| rank2_large | tuned | 0.001 | +0.23721 ± 0.01014 | +0.01687 ± 0.02588 |
| rank5_small | tuned | 0.0005 | +0.40358 ± 0.00765 | -0.05153 ± 0.00652 |
| rank5_large | tuned | 0.0003 | +0.29802 ± 0.00292 | +0.11221 ± 0.00412 |

## Translation blocks

The Figure 6 style view uses the already-trained m=3 source configuration, outer LR 0.0005 and seed 0, showing the first 20 output blocks and first 22 input rows. This is an explicitly labelled diagnostic; the main paper-protocol performance row retains m=70. Both full raw m=3 and m=70 matrices are exported. Latent coordinates are not transformed, aligned, sorted or clustered.

## Rotation filters

Each displayed filter is produced by native MSR after one support update on validation task zero. All channels are shown in their original order with a common color scale for fixed and tuned models. These are fitted filters, not discrete indexers. The fixed rotation outer LR 0.0005 is inferred from an ambiguous appendix reference; the release default and validation winner use 0.001.

Full numerical arrays and model SHA256 provenance accompany the PNG/PDF figures.
