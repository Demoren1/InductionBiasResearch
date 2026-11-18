# Broad aSRR tuning and confirmation

17 November 2026. User authorizes adequate tuning and all available GPUs.
The objective is the best defensible aSRR performance within the declared
search, including faithful replication of the earlier successful setting.
Keep previous experiments immutable. Test responses never select models.

## Stage R: replicate the already frozen successful procedures

Use the original frozen aSRR, practical G34 and Dense24 policies from
`srr_approx_real_2026-11-15/choices_frozen.json`. Reuse their exact saved
indexators and numerical solvers/RNG settings. MNIST has 100 fitting,
50 within-task validation and 100 query examples per class. Omniglot has
10/5/5 per class in a five-way task. Draw 512 fresh episodes per dataset,
seeds 7611171101 (MNIST) and 7611171102 (Omniglot). Freeze predictions
before scoring. This independent conditional replication is separate
from the later tuned confirmation cohorts.

## Stage D: tune structure and numerical training on development

Datasets: MNIST, Fashion-MNIST, UCI HAR, and Omniglot when its original
official image cache is available or restored. Preserve source/class and
subject splits. Evaluate both the manuscript's small-support setting and
the earlier 100-example-per-class image setting; Omniglot uses its original
10/5/5 setting. HAR retains three support and five query windows/class.
Support-only tasks do not acquire extra labelled within-task validation.

Candidate budgets are 16/32/64/128/256/512/1024/2048/4096/8192 where
feasible, plus the common Dense endpoint. Keep all five original Ward
families for direct development selection, without source M pruning.
Reuse original banks wherever retained. Build missing families from the
same retained Dense source directions. Include original successful
indexators as fixed reference candidates. Positive-scale A1 refinement
may add candidates using source data only, preserving their seeds.

Stage D1 screens all source-only candidates on 48 development episodes
using the original 200-update, three-learning-rate, two-start solver.
For support-only tasks use development-selected fixed numeric rules.
Stage D2 advances the best three candidates in each structural family,
plus the original frozen winner and Dense, to 96 development episodes.
Compare rates .001/.003/.01/.03, activation quantiles .1/.5/.9, horizons
200/500/1000, and 6/24 numerical starts on within-task validation.
For support-only tasks select a global update count, rates/quantiles and
1/4/16-start probability ensemble on development; do not choose starts
or checkpoints using target query labels. Account for ensemble costs.
Practical G34 and Dense get the same applicable numerical search.

The primary selection maximizes mean development accuracy, breaking ties
by CE then fewer codes. Also retain a CE-selected policy separately.
No query-based swaps, family selection, budget selection or stopping.
This is a documented expansion after the earlier narrow-grid study.

## Stage T: freeze, then evaluate new episodes

After all choices are fixed, draw 512 fresh test episodes per image
setting, followed by another fixed 512-episode confirmation without
retuning. Use 90 fresh HAR episodes followed by 90 confirmation episodes;
intervals aggregate the same nine subject means. These remain conditional
on the already studied class/subject environment. Methods share episodes
and applicable numerical budgets; report Dense, G34 and aSRR together.
Keep old replication, new tuning, CE/accuracy policies, constrained banks
and Dense fallback distinct. Report every prespecified confirmation,
including negative results. Do not claim a guaranteed global optimum.

No software tests. Necessary checks cover frozen hashes, actual sample
counts, structure-only transfer, honest selection and paired uncertainty.

## Numerical implementation clarification before development fitting

Support-only screening compares three learning rates and three activity
quantiles with two starts each; all starts contribute to development
selection, while no target query response selects numeric states. In D2,
train a nested pool of 16 starts and compare probability ensembles using
the first 1/4/16 starts. Report that all 16 starts were fitted when the
chosen ensemble uses fewer members. This preserves compute accounting.

After the completed D1 screen, refine its best seed in each of the five
original aSRR families and practical G34, separately for each setting.
Use 12 source-only A1 steps through 1024 codes and 4 above, preserve all
seeds, and admit all 36 refinements to D2 alongside the screened finalists.
The tuned test and confirmation responses have not been generated.
