# Clustering inside aSRR: preregistered sequence

2026-11-15. No manuscript changes. Synthetic stage precedes real tuning.
Source and development class environments have been examined previously.
New synthetic structures and new real test episodes are used below.

Methods: Ward, single linkage, complete linkage, Euclidean Lloyd k-means,
signed Lloyd k-means, and official e-means (Bessa et al., arXiv:2611.09501v1).
Author source is pinned to Faiss v1.16.0, unmodified, stored with hashes.
Use native PyTorch FP32 on V100, not the H100-specific kernels. No claim
that these timings reproduce the paper's speedups.

For signed profile z use the upper triangle of zz^T/||z||, multiplying
off-diagonal coordinates by sqrt(2). Its Euclidean distance is exactly
the previous aSRR distance. Ward/single/complete and Euclidean Lloyd and
e-means share this representation. Signed Lloyd directly minimizes
sum min_s ||z-s*c||^2, alternating signs, assignments and means.
Input-group candidates keep the previous ordinary Euclidean input
representation and positive indices; signed Lloyd then aliases Lloyd.

Each stochastic fit gets four fixed random data-point initializations.
Lloyd has25 updates. e-means tries the official auto/5, quality/5,
quality/25 recipes, with the same four initial centroid choices; select
the fit by source SSE only, retaining each recipe's diagnostics. This
is a quality comparison, not equal-compute timing. Record last-epoch
labels as supplied by the author, without silently adding Lloyd steps.
Restore empty groups, for all centroid methods, by splitting a position
from a non-singleton group with largest residual. This postprocessing
ensures identical actual m; record the number of repairs.

## Synthetic

Use the existing controlled generator with NEW seed_base3511159100:
32 independent structures, ranks2/5, balanced/unbalanced/deformation/input,
paired measurement noise0/.01/.04,96conditions. D24,H4,T8,P96. Same
numeric fitting and coefficient generation as the previous aSRR series.
Shared budgets8/16/24/32/64/96. All methods receive the same observations.
Scopes: global, fixed synchronized profiles, local, shared input. The
synchronized input is calibrated ONCE with a fixed Ward auxiliary m24,
identical for every method. Local scope allocates m/H groups per neuron
for every method, separating allocation from clustering. Record the
previous adaptive-local Ward as a separate baseline.

For each method choose scope and m using only three development numeric
functions (fit192/H96/selection256); evaluate three separate functions
(query512) after freezing all predictions. Same ridge grid0/.1/1/10/100.
Include practical G34 on the same observations (four starts,12 projection
steps, full support), Dense and original adaptive-local Ward. Signed
pair precision/recall, query clean MSE, m, source objective and histories
are audited only after freezing. No synthetic oracle enters clustering.

## Real stage, only after synthetic results

Omniglot and MNIST retain exactly the previous source consensus,
development panels, target fitting, class split and budget grid
16/32/64/128/256/512/1024/2048/4096/8192/P12544. Compare local (m/H
per neuron) and shared-input scopes for every method; these were the
two predictive winners in the previous study. Retain original full-bank
Ward/G34/Dense controls, including the previously selected Ward models.
Do not expand real scopes after observing test outcomes.

For each method choose its best source-built candidate on development CE.
Run the same practical Algorithm1 from each selected compressed candidate,
12steps if m<=1024 else4; retain original seeds. Select within each method
by development CE. Also report the common paired one-SE selector,
minimum m within one paired standard error of the best CE, unchanged
from the earlier aSRR study, with the same rule for baseline banks.

Freeze every policy BEFORE generating512new episodes/dataset:
Omniglot3511159201, MNIST3511159202. Original numerical seed and six
starts; Dense has four pools,24starts, selected on within-task H.
Do not open query labels before all candidate predictions are frozen.
No tuning after scoring. Report paired CIs against Ward, G34 and Dense,
per-budget development curves, real train/H losses, synthetic recall,
actual code counts and source construction time (not concurrent fit time).

No software tests are created or run; research experiments and minimal Ruff.
