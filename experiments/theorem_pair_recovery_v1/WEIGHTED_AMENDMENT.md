# Uncertainty-weighted contrast space

This follow-up is registered after the uniform-SVD series was frozen and
its structural outcomes opened. It is explicitly a post-hoc exploratory
amendment on the same observations, not independent confirmation.

The first clean Pattern case has 13 almost-exact dense models and three
poor numerical solutions (source validation MSE about 0.175, 0.357, 0.099).
These three violate the exact shared log-space model and distort uniform
SVD. This diagnosis uses the source validation panel; the reference groups
were used only to measure the resulting subspace error after freeze.

Change only the spectral estimate: multiply each centered log-observation
column by inverse RMS direction uncertainty across its neurons. Use the
existing direction-uncertainty floor 0.0003, then normalize the maximum
weight to one. All weights are strictly positive, so this operation
preserves the exact noiseless column space. Its practical reliability still
depends on the uncertainty estimates; it is not a certified error bound.

Keep all 16 tasks, original anchors/masks, all ranks, threshold grid, sign
rules, ranking, limit 32, numerical fitting budget and source meta selection.
Reuse numerical fits of identical structures from the first series as well
as G34. Prior signatures not proposed by this weighted branch do not enter
its combined bank. Freeze all new source selections before its own audit.
Report both versions, including losses. No tuning of further weights,
thresholds or source subsets on the new results is planned.
