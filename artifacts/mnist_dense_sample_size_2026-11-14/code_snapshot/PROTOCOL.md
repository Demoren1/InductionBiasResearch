# Dense sample-size control on MNIST

14 November 2026. The user asked whether25 Omniglot support examples are
insufficient and explicitly allowed changing the dataset. This is a new
source-only diagnostic, not another Omniglot target comparison.

## Fixed comparison

Use only the official MNIST training split, with its original28×28 pixels
divided by255. Never load official test images or labels. Three predefined
five-class problems:0/1/2/3/4,5/6/7/8/9 and0/2/4/6/8. These overlap and
are three diagnostic class sets, not independent dataset replications.

For each problem reserve500 original images per class for source H;
construct nested, class-balanced training sets with5,50,500 and4000
original images per class. Thus the training sizes are25/250/2500/20000.
No image augmentation. H originals never enter any training subset.

Keep the Omniglot Dense architecture784–16–5 and its12 scenarios:
Adam rates.001/.003/.01, weight decay0/1e−4, two restarts. Initialize
each scenario/problem identically across sample-size conditions. Use2500
updates, random minibatches of min(512,N), checkpoints every25 updates.
H cross-entropy selects checkpoints and scenarios. Report full training
and H losses/accuracies at selected and final states. H is a selection
panel, not an unbiased final test estimate; no target transfer is scored.

Save every scenario at steps0/25/100/500/1000/1500/2000/2250/2500,
along with all H-selected and all final parameter states. Compare
unit-normalized first-layer directions across the four H-best fits after
optimal neuron permutation, without sign flips. Measure within-run
direction drift from2250 to2500 and1500 to2500 in the H-best scenario.
Neither small drift nor low training loss proves parameter identification.

## Structural diagnostic

For each problem/sample size examine the H-selected state and the final
state of its H-best scenario. Use three masks:full; variable input pixels;
top50% weights per neuron among variable pixels. A variable pixel has
standard deviation at least.02 on that problem's fixed H input panel.
This common, label-free mask removes weights on almost constant pixels;
it is identical across the four sample-size conditions. Report omitted
weight energy. The masks and budgets are fixed before fitting.

Reuse the exact scalar saturation-interval implementation and the
unchanged constructive recoverer. Budgets8/16/32/64. A width rejection
excludes the entire saturation interval. If width permits, call the
recoverer at the minimum admissible epsilon, midpoint and upper interior
endpoint; failure at these three values alone does not exclude the
whole interval. Never force a candidate. Also record the independent
local-cover lower bound for16 codes. No unknown true indexator is used.

MNIST does not supply a known shared codebook. This can test whether
additional observations improve fitted functions, weight reproducibility
and criterion applicability. It cannot establish accuracy of recovering
an unknown ground-truth sharing or establish an Omniglot causal effect.
The network family and fixed update budget can themselves limit fitting.

Registration, data hashes and code hashes are frozen before training.
New model outputs are written to a separate artifact directory. Existing
results, G34, adapters and manuscript files are preserved. No software
tests are added or run.
