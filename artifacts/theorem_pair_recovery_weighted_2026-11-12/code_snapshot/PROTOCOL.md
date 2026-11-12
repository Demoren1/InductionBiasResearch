# Contrast-space pair signatures on retained dense models

12 November 2026. Retrospective experiment on the 24 source conditions
retained in `artifacts/anchor_start_2026-11-10/numerical_state.tar.gz`.
Historical results have already been disclosed. This is an exploratory
comparison on saved models, not a fresh confirmation experiment.

The learner reads only dense source descriptors, the initial source-fitted
bank, its numerical-validation losses, and public D/H/m/O dimensions.
Four existing anchors provide hidden-unit permutations. Supports use each
anchor and the original uncertainty-based 1/2/3-vote masks, deduplicated.
No true mask, groups, code values or neuron permutations enter proposals.

For each alignment and support, form the E-by-T matrix of log magnitudes,
subtract within-neuron means, and take left singular vectors. Try ranks
0 through m-1 because graph component count is unknown. Apply exactly
the two signature merge rules: local differences of norm at most t, and
ordered-pair differences at distance at most 2t, excluding zero signatures.
The fixed grid is 0 plus 32 logarithmically spaced values from 1e-10 to 0.2.
Do not forcibly complete partial partitions or split overmerged partitions.
Keep only exactly m nonempty groups with matching contrast dimension.

Rank unsigned proposals by a robust standardized log-factorization score
with fixed labels. Keep 32 unsigned proposals for sign reconstruction.
Signs use each observed task profile, with and without weighted rank-one
sign voting. Rank signed candidates by log score plus four times weighted
sign disagreement. Keep all constructed candidates for coverage analysis;
at most 32 receive subsequent numerical evaluation. Candidate generation
does not invoke k-means or the original alternating assignment algorithm.

These ranks, thresholds and noise scales are practical proposals, not
certified bounds on subspace error. Finite-source excitation, correct
support/alignment and repeated-pair connectivity must be diagnosed
separately. Signed recovery requires additional sign information.

Freeze every proposed bank before opening reference structures. Report
signed and unsigned bank coverage separately from actual source-only
selection; oracle coverage is not a selector. Compare with B0 and full G34
on identical stored source models. Any functional evaluation fits numerical
parameters on source support, chooses states on numerical validation,
and selects structure on the separate source meta sample. Target query
responses never select a candidate. No changes to the original G34 code,
published results or manuscript. No software tests.
