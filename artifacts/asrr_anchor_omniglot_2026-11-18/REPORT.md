# Omniglot aSRR with anchored source initialization

18 November 2026. On 1,024 held-out five-way episodes, the anchored bank reached
82.93% accuracy and the previously selected aSRR bank reached 83.10%. The paired
difference was −0.17 percentage points with a 95% interval of [−0.49, +0.15]. The
development advantage did not carry over to the held-out episodes.

| Episodes | Count | Anchored aSRR | Original aSRR | Paired difference, pp (95% CI) |
| --- | ---: | ---: | ---: | ---: |
| Development | 96 | 83.75% | 82.88% | |
| Test A | 512 | 82.91% | 82.95% | −0.04 [−0.50, +0.43] |
| Test B | 512 | 82.95% | 83.25% | −0.30 [−0.75, +0.14] |
| Pooled test | 1,024 | 82.93% | 83.10% | −0.17 [−0.49, +0.15] |

Pooled cross-entropy was 0.5403 for the anchored bank and 0.5399 for the original.
The paired cross-entropy difference was +0.0005 [−0.0059, +0.0069]. The intervals
use paired episode means and are conditional on the held-out character pool.

## Source banks and selection

All banks use the same 16 source tasks as the saved Omniglot aSRR benchmark.
Each new bank uses one unchanged Dense anchor and 15 Dense models initialized
with its first-layer weights and biases. Each descendant trains both layers on
its own source task with a fresh output head, using the existing source training
recipe and selecting one model on that source task's numerical validation set.
The 16 models retain their neuron order from initialization. Bank construction
applies per-neuron normalization and Ward clustering of input profiles, with
64 input groups and 16 codes per group. There is no neuron matching in the new
banks. The saved comparison bank used aligned, averaged source fits, so the
comparison covers the full source-bank procedures.

The old development winner fixed the family, 1,024-code budget and target fit
for every bank. Among the 16 anchors, mean development accuracy selected source
task 109 (zero-based task ID), with cross-entropy and anchor order as tie-breaks.
The indexators were frozen before either test cohort was drawn.

Each target episode uses 10 training, 5 numerical validation and 5 query images
per class. Both methods start with fresh numerical parameters, run the same
24 Adam starts for up to 500 steps, and average the predictions of the four
validation-selected solutions. Target fitting receives only the indexator and
episode panels. Query labels are scored after predictions are frozen.

The lineage check confirmed the exact anchor, initialization and 15 selected
descendants for all 16 banks. The experiment settings, development ranking,
frozen indexators and exact metrics are saved alongside this report.
[Registration](registration.json) · [Frozen choices](choices_frozen.json) ·
[Indexators](frozen_indexators) · [Exact scores](results.json) ·
[Prediction records](predictions_frozen.json)
