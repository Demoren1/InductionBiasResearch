# Real-data follow-up of local SRR

15 November 2026. The synthetic development is complete. This follow-up
uses retained dense source models, never synthetic teacher parameters.
The existing class/subject environments have already been examined; this
is a conditional follow-up, not a new untouched benchmark.

## Recovery feasibility

Run the unchanged local SRR, pooled weight verification (score <= 9), and
the fixed radius-2 assignment-margin filter on Omniglot, MNIST,
Fashion-MNIST and UCI HAR. Use every retained source task: respectively
128, 10, 10 and 4. Source models have four independently fitted numerical
states. Normalize incoming directions; estimate their uncertainty by
Hungarian-aligned dispersion of these states, with the existing 0.0003
floor. This is a heuristic uncertainty, not a certified error bound.

Use multipliers 1.5, 2.5, 4 and 6 without new tuning. Allocated budgets
are 2/4/8/16/32/64/128/256/512/1024/2048/4096/8192 where below the
number of weights, plus the explicit full-budget endpoint. Budget 1 is
a separately labelled collapse control.

Do not force a result when SRR cannot identify a complete partition.
Record candidate construction, weight-verification acceptance and
margin-filter acceptance separately. A request for more groups than the
total number of local classes is impossible for this merging algorithm
and can be rejected before pair matching. Bound a pair-matching call by
one million enumerated subset pairs; report a computational limit
separately from statistical abstention. Do not call a quantizer or G34
as an implicit SRR fallback.

## Predictive evaluation

For Omniglot, retain the existing 784–16–5, five-way 10-train/5-numerical-
validation/5-query examples per class protocol. Source, development and
test characters are disjoint. Reuse the existing 48 development episodes
and source-only G34 bank with its documented nested capacity refinements.
This is the existing practical G34 pipeline, not a newly rerun exhaustive
G34 optimization. Include a fresh Dense fit and the explicit unshared
endpoint. Reuse already computed development results where applicable.

If SRR produces candidates, rank them using source weight verification
alone and retain at most one per budget. Choose a predictive budget only
on development query CE. Report strict-filtered SRR coverage separately
from any unfiltered diagnostic. Fix all choices before drawing 128 fresh
test episodes with seed 3511158601. All methods get the same six fresh
numerical starts (Adam 200 steps, learning rates .003/.01/.03, checkpoints
chosen on within-task numerical validation). Save predictions before
opening test query labels. No absent SRR prediction receives an accuracy.

The first four-dataset stage determines which additional predictive
comparisons are meaningful. Any second-dataset predictive protocol will
be saved in an explicit amendment before its development or test runs.
Existing published-in-project accuracies may be shown only with their
own protocol and cohort, never as if measured on these new episodes.

## Interpretation

There is no known ground-truth sharing indexator on these datasets.
Measure predictive performance, SRR coverage and weight diagnostics;
do not report exact recovery accuracy. Large numbers of codes or
uncertainty intervals spanning nearby code values can invalidate the
small-code recovery mechanism even if a compressed predictor works.
No manuscript edits, new software tests, source meta-training or old
MSR/CNN grid restarts are included.
