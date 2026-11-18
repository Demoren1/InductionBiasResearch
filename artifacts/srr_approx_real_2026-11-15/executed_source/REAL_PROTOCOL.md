# Approximate SRR after theoretical and controlled development

Before real-data tuning: theory APPROXIMATE_SRR_THEORY_2026-11-15.md and
96 paired controlled conditions (32 independent structures) completed.
All real class environments were examined previously; this is a conditional
research follow-up. New test episodes below remain unopened until all
new structures, code budgets and adaptation rules are fixed.

Primary baselines: Dense and retained practical G34 banks. Retained
MSR-FC and legacy SRR are secondary controls. No MSR source/CNN restart.
No manuscript changes. No new software tests.

Source information is the retained dense archive with four numerical
states per source: Omniglot16 H-ranked tasks (same16 as the G34 bank),
MNIST all10 tasks. Align each state's unit directions positively to the
selected source; average the four, then align sources to the first source.
This consensus estimates a stable component; it is not assumed to be a
certified unbiased estimate of one true ReLU parameterization.

Build global signed-profile Ward, positive-scale synchronized Ward
(fixed auxiliary m256), neuron-local signed Ward and shared-input Ward
from consensus directions. Add shared-input Ward from selected dense
directions as a consensus ablation. The shared-input variant groups input
coordinates jointly but keeps independent codes in different neurons;
no image coordinates, adjacency graph or oracle roles are provided.

Budgets16/32/64/128/256/512/1024/2048/4096/8192 and P=12544. No hard
zero threshold and no whole-index margin rejection. Canonicalize code
labels/sign gauges. The P endpoint is always the positive unique index
and the same Dense numerical fit, avoiding gauge-based extra restarts.
Keep all original G34 candidates; compare after the same canonicalization
and fresh numerical adaptation. This G34 baseline includes documented
source-only capacity refinements, not an exhaustive new search at every m.

Stage1: evaluate source-only candidates on retained development episodes
(Omniglot48; MNIST24 from the completed real SRR study). Numerical fitting
uses the same6start Adam200 solver as that study, rates.003/.01/.03 and
within-task numerical validation. MNIST has100 train+50 validation+100
query images/class; Omniglot10+5+5. All target numbers are fresh.

Stage2: for each new family, take its development-best compressed seed
and apply positive-scale Algorithm1 on source consensus weights. Use12
iterations if m<=1024,4 otherwise. Preserve initial and refined structures.
Apply the same refinement to the development-best original G34 candidate
as a stronger baseline, with the same source information. The bounded-memory
implementation evaluates the full Student categorical cost in blocks and
restores empty codes by minimum extra cost. It is a positive-scale
instantiation, not exhaustive signed-orientation G34.

Dense control: four independent6start pools, plus their per-task
numerical-validation selection (24 starts total). Select between the
6start and24start adaptation policies only on development query CE.
Compare any apparent advantage against this stronger selected Dense.

Choose final structures and m by development CE. Preserve the original
G34 bank in the final aSRR+Algorithm1 union; report whether the winner
is a new seed, refinement or G34 fallback. Add a fixed randomized partition
control with the selected new candidate's counts and signs. For an
input-family winner, permute input rows of absolute labels together;
otherwise permute absolute labels independently inside each neuron.
Keep signs at their original positions. Thus group counts, incidence
between groups and neurons, and signed-position templates are preserved.
This control is diagnostic, not another selected source seed.

Only then draw256 test episodes per dataset: Omniglot seed3511158801,
MNIST3511158802. Freeze all predictions before scoring any query labels.
Save per-episode paired differences and intervals, budgets, group counts,
source/projection diagnostics, curves and source hashes. No speed claim
from concurrent GPU jobs. Development after this test must be labelled
and use other fresh episodes for confirmation.
