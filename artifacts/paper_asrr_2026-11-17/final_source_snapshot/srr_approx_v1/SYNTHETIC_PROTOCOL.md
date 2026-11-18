# Approximate SRR: controlled development

Theory is saved first in paper/theory/APPROXIMATE_SRR_THEORY_2026-11-15.md.
This is method development, not an independent final confirmation.

32 independent structures: balanced global sharing, unbalanced global
sharing, approximate global sharing with independent task deformation,
and shared input groups with independent codes in different neurons.
Four structures for each scenario and code-profile rank 2/5. Each has
8 numerical source tasks,24 inputs,4 hidden-column labels,96 weights.
Measurement-noise levels0/.01/.04 are paired on each structure, giving
96 conditions, not96 independent structures. Neurons are already aligned;
this component study does not establish dense-training or alignment accuracy.

Global scenarios use8 nonzero signed codes; the input-group scenario
uses6 input groups ×4 neurons=24 codes and positive index signs.
All weights are active. Source code values are linear combinations of
rank2/5 latent factors. Normalize each source column, then add supplied
Gaussian direction noise and normalize again. The approximate scenario
also adds independent coordinate deformation of relative norm.25 before
normalization, and the same distribution of deformation to new functions.

Every method receives the same observations and code grid
8/16/24/32/64/96. The full endpoint96 is a common positive unique index,
equivalent to Dense. aSRR uses global signed-profile Ward, positive-scale
synchronized Ward, neuron-local Ward and an input-group Ward family.
No position is thresholded to zero. No ground-truth index enters extraction.
The synchronized family uses the fixed auxiliary m24 seed.

G34 control is the existing practical scale-aware Algorithm1 projection,
full support,12 iterations, four initializations per budget: two quantile
anchors and two random assignments. This is explicitly a finite practical
G34 bank, not the older exhaustive small-code native pipeline. aSRR+A1
adds12-iteration projection from global and synchronized aSRR seeds,
retaining original seeds and the original G34 bank. All code labels/sign
gauges are canonicalized. Its fallback to G34 is reported as such.

New functions are linear regressions on96 independent Gaussian inputs.
Target neuron scales are1. Each structure has three development and
three different test coefficient functions, drawn from its same latent
code distribution. Every function has192 fit and96 numeric-validation
observations; development has256 structure-selection observations, test
has512 private query observations. Output-noise std=.2. Orthogonalized
lookup features give the same signed-code model class; fit ridge with
lambda0/.1/1/10/100, chosen per task on numeric validation.

Choose the shared candidate and m by mean development selection MSE,
then refit fresh numerical coefficients on the three test functions.
Freeze all predictions before auditing private query answers and truth I.
Report fresh noiseless-function MSE, pairwise sharing precision/recall,
selected m and exact source-projection bias; no forced exact-I criterion.
Dense is given the same ridge grid and fitting observations.

RNG seed3511158700. No new software tests; research experiments only.
