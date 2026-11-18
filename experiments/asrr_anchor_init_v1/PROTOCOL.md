# Starting aSRR source tasks from a trained first layer

MNIST uses the same ten source pairs from digits 0–4 and the existing
784–16–2 ReLU fits. For each pair in turn, retain its original fitted
network as the anchor. Copy its first-layer matrix and bias into all
12 starts for each of the other nine pairs. Their output layers and
optimizer states start afresh. Train every numerical parameter on its
own source data with the existing 2500-step Adam and four L-BFGS
continuations. Checkpoints are chosen on each source task's numerical
validation data. The independent Dense archive supplies the anchors and
the ordinary aSRR control.

Each bank uses exactly ten selected Dense models, one per source
task. For a warm cohort these are the anchor and nine descendants.
Normalize each fitted column within its own network. The warm branch
uses the inherited column numbers. The independent control uses the
usual Hungarian matching between the ten selected models. Both
branches omit averaging numerical restarts, joint scale calibration
and optional projection refinement.

Use the aSRR families, budgets and numerical rules already selected
by development accuracy in the 17 November study. The 16-example
setting uses input grouping with 2048 codes, 50 Adam updates at rate
0.003, activity quantile 0.9 and probability averaging over 16 starts.
The 100+50 setting uses global signed Ward with 8192 codes, 24 Adam
starts of up to 1000 updates and the four starts with best within-task
validation loss. Families, budgets and numerical rules are fixed.

Development uses the same 96 fresh episodes from classes 5 and 6 for
each initialization in a setting. Select the warm anchor by accuracy,
then cross-entropy, then source task number. As a secondary comparison,
evaluate the single anchor fixed by the original source validation
loss. The preliminary screen from before the protocol clarification
is excluded from selection.

Freeze each selected indexator before drawing
two 384-episode test cohorts per setting on pairs from 7–9, with 100
query images per class. The same episodes and numerical rules apply to
both source initializations. Every target model starts with fresh
numerical values. The scorer uses paired episode differences and 95%
intervals. These intervals are conditional on the source cohort and
class split.
