# Source-only follow-up of the remaining saturation interval

The first registered pass found one budget/mask combination whose contrast
components satisfied the necessary width inequalities: anchor107,
uncertainty multiplier3, 45 active entries and budget21. The public
recovery calls at the lower, midpoint and upper interior tolerance all
failed the required anchor-neighbour count.

The follow-up records a monotonicity argument to resolve the whole
remaining interval, rather than treating three samples as an exhaustive
search. Component membership and the deterministic representatives are
constant inside the saturation interval. Anchor-neighbour membership
only grows with epsilon. Evaluate those fixed representatives at the
closed upper endpoint; fewer than 2(m−2) neighbours excludes every
interior tolerance. This is a source-only diagnostic extension after
the first pass; no new transfer data, predictions or scoring enter it.
As with the original implementation, calculations are float64 rather
than formal interval certificates.

The same follow-up exports an explicit two-edge cycle witnessing the
previous selected indexator's minimax error on its own source anchor.
No new numerical network fit is needed.
