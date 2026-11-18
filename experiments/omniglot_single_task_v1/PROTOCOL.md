# Direct single-task ratios on Aug-Omniglot

Registered on 14 November 2026 before source fitting or development access.
The user requested an Omniglot run and careful dataset-specific selection.

## Observations and comparators

Use the unchanged 784–16–5 ReLU model, 5-way 5-shot episodes, query-only
augmentation, Vinyals class split, 128 source and 128 development episodes.
Recreate source F/H/M panels with the historical generator and seeds. Fit
the unchanged dense ensemble: 12 settings/restarts, 2500 Adam updates,
H-selected checkpoints and four retained fits per source task.

The primary comparison reuses the 1000 episodes with seed 26111272013 and
the already frozen G34, Dense and weighted-SVD predictions from
`artifacts/omniglot_signature_2026-11-12`. Their answers were opened in the
previous experiment; this is a retrospective extension on a common cohort,
not an independent confirmation. New learners cannot open those answers.
First freeze the new structure, numerical rule and all predictions; only
then score on the retained answers. Verify reproduced observation hashes.

## Proposal construction

The two anchors are the source tasks with the lowest own H loss. Each
proposal uses only one anchor's normalized first-layer weights and its own
four-fit direction dispersion. Other tasks do not enter its pair features.
Use these masks: full; magnitude above 1, 2 or 3 times the own dispersion
(floor 0.0003, magnitude floor 0.002); retain the largest 25%, 50% or 75%
of magnitudes separately in each neuron. Deduplicate identical masks.

Use direct log-magnitude differences, without SVD. Merge within-neuron
differences of magnitude at most t; match nonzero ordered differences at
distance at most 2t and merge corresponding endpoints. A sorted scalar
forest represents all these matches exactly. This changes computation,
not the merge rule. Recompute each threshold independently.

Scan t=0 and 97 geometric thresholds from 1e-12 to 1. For each adjacent
scan interval whose endpoint group counts straddle a requested budget,
bisect at most 24 times, retaining only naturally complete partitions
with exactly 8, 16, 32 or 64 groups. No forced completion, k-means or
post-hoc merging is allowed. Save all scanned group counts. Do not assume
global monotonicity when searching: bisection is a bounded proposal
heuristic; any retained output exhausts the exact merge rule at its t.

Sign each index with its anchor's observed sign. Rank proposals with the
existing robust log-magnitude score and an omitted-edge penalty, using
only that anchor. Keep at most eight per budget, with at most two per
anchor/mask/budget and preference for distinct masks before extra slots.
The code budgets and masks are selected by source/development risk, not by
knowledge of an underlying real-data sharing partition.

## Selection and transfer

For every retained candidate use the unchanged source numerical adapter:
18 starts, 2000 Adam updates, one dense-derived initialization, H numerical
selection; independent source M selects one structure per budget. Only
then open development data. The unchanged development adapter selects
budget, LR in {0.003, 0.01, 0.03}, activity quantile in {0.1, 0.5, 0.9},
and update count through 2000, averaging each choice over two restarts.

Transfer only the selected index/basis and numerical rule. Use the
unchanged target adapter, fresh numerical parameters, support-only
training and fixed global update count. No source code values, biases,
readouts, task-specific scales or fitted numerical states transfer.

Report group sizes, cross-neuron sharing, support density, adaptation and
extraction times, source calibration and selection cost, all missing
budgets, development CE, episode-paired target accuracy/CE and intervals.
Distinguish exact scientific recovery from a real-data proposal heuristic:
Omniglot has no observed true sharing groups or verified separation bound.

If the direct rule fails or produces degenerate groups, preserve that
result. Any necessary modification must be recorded before final target
scoring, with its scientific reason and source/development-only evidence.
No software tests are added or run.
