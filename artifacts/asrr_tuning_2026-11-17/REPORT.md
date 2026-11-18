# Broad aSRR tuning and independent confirmation

17 November 2026.

The former successful procedures were first replicated unchanged on
512 fresh episodes per dataset. A separate development search then
selected wider budgets and numerical rules. Two new cohorts evaluate
the frozen tuned policies, without retuning between cohorts.
The primary aSRR policy improves over Dense on both cohorts for
MNIST at both support sizes, HAR and Omniglot. On Fashion 100+50
the paired difference includes zero; on Fashion 16 the primary
aSRR policy loses a small amount of accuracy. The secondary CE
policies remain separately reported rather than replacing these choices.

## Fresh replication of the previously frozen aSRR procedures

512 new conditional episodes per dataset. No tuning on these query answers.

| Dataset | aSRR | Dense24 | Practical G34 | aSRR minus Dense (pp, 95% CI half-width) |
|---|---:|---:|---:|---:|
|mnist|96.27|96.02|96.09|0.254 ± 0.105|
|omniglot|81.75|81.16|79.95|0.594 ± 0.521|

This replication is separate from the separate broader search.
MNIST uses 100 fitting + 50 within-task validation + 100 query examples/class.
Omniglot uses 10/5/5 in a five-way task. Source/class environments were studied previously.


## Tuned primary policies

Primary selection maximizes development accuracy, breaking ties by CE
and fewer codes. The following pool uses unchanged policies in both
prespecified cohorts. Each image setting has 1024 episodes; HAR 180 episodes
use nine subject means. Environments/classes/subjects were studied before.

| Setting | Method | Accuracy, % (95% half CI) | CE | Codes | Inference ensemble | Starts trained | Max updates | Inference parameters |
|---|---|---:|---:|---:|---:|---:|---:|---:|
|MNIST 100+50|asrr|96.21 ± 0.12|0.1195|8192|4|24|1000|32968|
|MNIST 100+50|g34|96.21 ± 0.12|0.1175|8192|4|24|500|32968|
|MNIST 100+50|dense|96.05 ± 0.12|0.1217|12544|4|6|200|50376|
|MNIST 16|asrr|93.52 ± 0.18|0.1810|2048|16|16|50|33568|
|MNIST 16|g34|93.08 ± 0.20|0.1876|8192|16|16|50|131872|
|MNIST 16|dense|92.88 ± 0.21|0.1962|12544|4|16|50|50376|
|Fashion 100+50|asrr|97.66 ± 0.17|0.0702|2048|4|24|200|8392|
|Fashion 100+50|g34|97.65 ± 0.17|0.0834|8192|1|24|1000|8242|
|Fashion 100+50|dense|97.68 ± 0.17|0.0743|12544|4|24|500|50376|
|Fashion 16|asrr|96.21 ± 0.23|0.1223|1024|16|16|425|17184|
|Fashion 16|g34|96.33 ± 0.23|0.1224|4096|4|16|125|16584|
|Fashion 16|dense|96.31 ± 0.23|0.1236|12544|4|16|75|50376|
|HAR 3|asrr|91.63 ± 3.04|0.2692|128|16|16|250|3936|
|HAR 3|g34|90.87 ± 4.77|0.3213|64|16|16|1000|2912|
|HAR 3|dense|86.31 ± 3.55|0.3739|8976|16|16|1000|145504|
|Omniglot 10+5|asrr|83.07 ± 0.55|0.5371|1024|4|24|500|4500|
|Omniglot 10+5|g34|81.34 ± 0.58|0.5760|4096|4|24|1000|16788|
|Omniglot 10+5|dense|80.82 ± 0.58|0.5945|12544|4|24|500|50580|

## Separate confirmation cohorts

Accuracy-selected policies are identical in both cohorts.

| Setting | Cohort | aSRR accuracy, % | Dense accuracy, % | G34 accuracy, % |
|---|---|---:|---:|---:|
|MNIST 100+50|test_a|96.24 ± 0.16|96.11 ± 0.17|96.25 ± 0.16|
|MNIST 100+50|test_b|96.17 ± 0.17|96.00 ± 0.17|96.18 ± 0.17|
|MNIST 16|test_a|93.42 ± 0.27|92.77 ± 0.31|93.00 ± 0.29|
|MNIST 16|test_b|93.63 ± 0.25|92.99 ± 0.28|93.17 ± 0.27|
|Fashion 100+50|test_a|97.72 ± 0.24|97.73 ± 0.23|97.68 ± 0.24|
|Fashion 100+50|test_b|97.60 ± 0.24|97.62 ± 0.24|97.61 ± 0.24|
|Fashion 16|test_a|96.23 ± 0.32|96.35 ± 0.33|96.37 ± 0.33|
|Fashion 16|test_b|96.18 ± 0.32|96.28 ± 0.33|96.30 ± 0.33|
|HAR 3|test_a|91.81 ± 3.21|86.26 ± 3.42|91.11 ± 4.68|
|HAR 3|test_b|91.44 ± 3.14|86.37 ± 3.78|90.63 ± 4.90|
|Omniglot 10+5|test_a|83.41 ± 0.77|80.83 ± 0.83|81.26 ± 0.83|
|Omniglot 10+5|test_b|82.73 ± 0.79|80.80 ± 0.80|81.41 ± 0.80|

## Paired gains of tuned aSRR

| Setting | Cohort | aSRR minus | Accuracy difference, pp (95% half CI) |
|---|---|---|---:|
|MNIST 100+50|test_a|g34|-0.002 ± 0.047|
|MNIST 100+50|test_a|dense|0.133 ± 0.055|
|MNIST 100+50|test_b|g34|-0.005 ± 0.047|
|MNIST 100+50|test_b|dense|0.176 ± 0.056|
|MNIST 100+50|pooled_fixed|g34|-0.003 ± 0.033|
|MNIST 100+50|pooled_fixed|dense|0.154 ± 0.039|
|MNIST 16|test_a|g34|0.419 ± 0.096|
|MNIST 16|test_a|dense|0.644 ± 0.105|
|MNIST 16|test_b|g34|0.455 ± 0.088|
|MNIST 16|test_b|dense|0.640 ± 0.097|
|MNIST 16|pooled_fixed|g34|0.437 ± 0.065|
|MNIST 16|pooled_fixed|dense|0.642 ± 0.072|
|Fashion 100+50|test_a|g34|0.041 ± 0.061|
|Fashion 100+50|test_a|dense|-0.011 ± 0.049|
|Fashion 100+50|test_b|g34|-0.009 ± 0.054|
|Fashion 100+50|test_b|dense|-0.018 ± 0.044|
|Fashion 100+50|pooled_fixed|g34|0.016 ± 0.041|
|Fashion 100+50|pooled_fixed|dense|-0.014 ± 0.033|
|Fashion 16|test_a|g34|-0.140 ± 0.060|
|Fashion 16|test_a|dense|-0.126 ± 0.067|
|Fashion 16|test_b|g34|-0.116 ± 0.065|
|Fashion 16|test_b|dense|-0.093 ± 0.075|
|Fashion 16|pooled_fixed|g34|-0.128 ± 0.044|
|Fashion 16|pooled_fixed|dense|-0.109 ± 0.050|
|HAR 3|test_a|g34|0.704 ± 3.250|
|HAR 3|test_a|dense|5.556 ± 1.656|
|HAR 3|test_b|g34|0.815 ± 3.040|
|HAR 3|test_b|dense|5.074 ± 2.081|
|HAR 3|pooled_fixed|g34|0.759 ± 2.976|
|HAR 3|pooled_fixed|dense|5.315 ± 1.689|
|Omniglot 10+5|test_a|g34|2.156 ± 0.481|
|Omniglot 10+5|test_a|dense|2.586 ± 0.490|
|Omniglot 10+5|test_b|g34|1.312 ± 0.495|
|Omniglot 10+5|test_b|dense|1.922 ± 0.483|
|Omniglot 10+5|pooled_fixed|g34|1.734 ± 0.345|
|Omniglot 10+5|pooled_fixed|dense|2.254 ± 0.344|

## Separately frozen CE policies

| Setting | Method | Accuracy, % | CE | Codes |
|---|---|---:|---:|---:|
|MNIST 100+50|asrr_ce|96.25|0.1092|512|
|MNIST 100+50|g34_ce|96.16|0.1087|512|
|MNIST 100+50|dense_ce|96.17|0.1172|12544|
|MNIST 16|asrr_ce|93.63|0.1743|2048|
|MNIST 16|g34_ce|93.08|0.1876|8192|
|MNIST 16|dense_ce|92.91|0.1939|12544|
|Fashion 100+50|asrr_ce|97.66|0.0702|2048|
|Fashion 100+50|g34_ce|97.70|0.0743|8192|
|Fashion 100+50|dense_ce|97.68|0.0743|12544|
|Fashion 16|asrr_ce|96.34|0.1244|2048|
|Fashion 16|g34_ce|96.34|0.1290|2048|
|Fashion 16|dense_ce|96.32|0.1183|12544|
|HAR 3|asrr_ce|90.80|0.2728|256|
|HAR 3|g34_ce|90.87|0.3213|64|
|HAR 3|dense_ce|86.31|0.3739|8976|
|Omniglot 10+5|asrr_ce|82.80|0.5319|1024|
|Omniglot 10+5|g34_ce|81.06|0.5798|4096|
|Omniglot 10+5|dense_ce|80.82|0.5945|12544|

## Interpretation and scope

The searched code budgets extend through 8192; the result is the best
development-selected procedure in this declared search, not a global
optimum. Every selected bank transfers only its indexator; all numeric
states are fitted afresh. Ensembles increase inference parameters and
training starts are reported separately. No numerical source values transfer.
Practical G34 uses retained banks and source-only refinements rather
than a new exhaustive native structural search. All three methods
receive the same applicable numerical search per retained candidate;
their numbers of structural candidates differ. The previous LUT-Q/Yeh/MSR
narrow-budget tables are a separate matched comparison.
No software tests were created or run.

## Measured execution cost

| Stage | GPU jobs | Summed worker hours |
|---|---:|---:|
|d1|374|1.070|
|refine|36|0.200|
|d2|506|4.635|
|target|58|1.152|

Worker time includes process startup and data loading. These stages
reuse previously prepared Dense source fits; their original cost and
the separate faithful replication are excluded from this subtotal.
