# Wider budgets and native competitors

Registered 14 November 2026 after examining the original 96-episode results.
Those results and all their frozen artifacts remain intact. This extension
uses the same source F/H/M and development observations. No decisions use
the next cohort's query answers. The public class environments have already
been examined: this is a conditional follow-up, not an untouched benchmark.

Compare Dense, G34, direct Pair recovery, LUT-Q, Yeh continuous, Yeh MAP,
and MSR-FC (basis only). Discrete methods transfer their signed indices;
continuous competitors transfer their native real-valued basis. All target
coefficients, hidden biases and output parameters are fresh. MSR's learned
initialization and learning rates are not transferred in this matched test.
These are disclosed fully connected adaptations, not published benchmark
scores or unchanged author implementations.

Budgets: 8, 16, 32, 64, 128, 512, and the explicit no-sharing endpoint
12544, equal to the number of first-layer weights. The endpoint is a
positive unique label at every position, hence exactly the Dense model
class. It is reported as a common control, never as recovered sharing.
If development prefers 512 among constrained models, add 2048 before
freezing target rules. This conditional extension uses development only.
The full endpoint does not require allocating a 12544-square categorical
or continuous matrix. It is represented exactly by an index vector.

Reuse the already completed source Dense preparation and small-budget
G34/Pair calibration. Extract G34 and Pair proposals at the larger budgets
with their existing code, at most eight candidates per budget. Missing
complete Pair partitions remain missing.

Native competitor grids use the existing real_competitors implementation:
three outer learning rates, cold/Dense starts, two categorical regularizer
settings for Yeh, two inner rates for MSR; LUT-Q has two random seeds.
Budgets 8,16,32,64,128,512. LUT-Q:2500 updates; MSR:2000; Yeh:400 outer
updates with 250 initial and 20 subsequent inner Adam updates and a
20-term implicit inverse-Hessian approximation. Keep native checkpoints
by source H. Retain at most eight H-ranked native candidates for the
common calibration. No post-clustering of continuous representations.

Every new candidate receives the same 18-start/2000-update Adam source
calibration as the existing comparison, with one Dense-derived start.
H selects numerical checkpoints and independent M selects the structure.
Development chooses budget, rate, activation quantile and updates by
mean query CE over 16 episodes and both fresh restarts. Existing small
budget fits and curves are reused, not recomputed. An algebraically
equivalent weight-gather evaluation avoids enormous feature tensors at
large budgets; the numerical optimizer and selection rule are unchanged.

Only after all model choices are fixed, sample 96 new episodes from the
same target class pairs (16 support and 100 query images per class),
with RNG seed 26111498231. Run every selected method on that same cohort,
freeze all predictions, then open its query answers once. Save per-budget
development scores, source quality, native diagnostics, selection receipts,
models and exact predictions. Intervals are conditional episode variation.

The synthetic benchmark must also include native competitors. Use transfer
error, since exact index recovery is undefined for continuous bases. Keep
the existing two geometries and source functions, fix source-only choices,
then sample new target numerical functions. No oracle structure enters a
learner. Additional datasets are required if the applied comparisons fail
to distinguish the methods or expose only a degenerate no-sharing choice;
their protocol must be fixed before their query answers are opened.
