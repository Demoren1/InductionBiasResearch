# Dense-first adaptation on Omniglot

15 November 2026. Mechanism experiment, separate from the stopped CNN suite.
No manuscript changes. The architecture is the previous 784–16–5 ReLU MLP.

Use the cached Vinyals character split, inverted 28×28 pixels and a random
multiple-of-90-degree rotation per character. Five distinct base characters
per task; their twenty original drawings are partitioned into ten fitting,
five numerical-validation and five evaluation images. No artificial increase
in the number of independent drawings and no query augmentation. This is a
10-training-example setting with five additional labeled validation examples
per class, not a standard 5-shot benchmark.

Reuse the frozen source dense ensemble and retained G34 indexators. Additional
G34 proposals at 16 and 256 codes use the same source weights, positive-direction
Hungarian matching and the existing scale-aware projection (16 source tasks,
12 alternating iterations, full and retained connectivity masks). The two
retained 64-code structures are also candidates. The old source fit is shared
preparation, not a new timing measurement. Development chooses among these
structures globally; all 12544 independent weights are an explicit dense control.

Draw 48 development tasks on validation characters. Fit dense models using
only each task's training and numerical-validation drawings; retain their
normalized weight directions and dispersion across numerical solutions.
Numerical fits use Adam, learning rates .003/.01/.03, two fresh initializations,
200 steps, numerical-validation selection every 25 steps. Preserve train and
validation CE and balanced accuracy (= macro recall for balanced episodes).

First select the shared source indexator and budget by mean development
evaluation CE after fresh numerical fitting. On that same selected source
indexator, compare zero, one and four dense-weight projection steps, with
fixed-anchor categorical penalties 0/.25/1/4/16. Choose the update count for
ordinary adaptation and the penalty/update count for prior adaptation by
development evaluation CE. These are globally selected, not chosen using a
new target's evaluation answers. Numerical values remain free.

Projection includes zero and positive/negative code labels. It fits codes
and positive neuron scales with three weighted least-squares passes, fixing
one scale per connected component. Categorical costs are the previous Cauchy
discrepancy plus a penalty for differing from the fixed aligned anchor. Empty
codes are repaired using the smallest increase in that penalized cost.
The same projection and alignment apply to both penalty ablations. This is
a finite local procedure, not a guarantee of global minimization.

Controls: independently fitted dense, single-task signed Lloyd quantization
with development-selected 16/64/256 codes, and the retained 64-dimensional MSR
basis with fresh numerical fitting. MSR here means structure-only transfer
under the common MLP protocol, not the published native initialization-transfer
CNN experiment. The retained source basis and indexators used source training
characters only; development and test characters are disjoint from those.

After selecting and freezing all global choices, generate 128 new tasks from
test characters. Save support/validation panels separately from evaluation
answers. Learners receive evaluation pixels only. Freeze all predictions and
choices and verify their hashes before opening evaluation answers for scoring.
All methods use the same tasks and labeled observations. Report episode means,
paired confidence intervals conditional on this source ensemble, chosen budget,
assignment-change fraction, numerical fitting curves and elapsed stage times.
Do not compare those times with old single-run GPU measurements.
