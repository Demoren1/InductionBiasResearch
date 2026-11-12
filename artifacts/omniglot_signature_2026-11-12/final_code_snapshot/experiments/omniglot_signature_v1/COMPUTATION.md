# Exact rejection rules for the large pair graph

The implementation changes the order of computation, while retaining the
zero-signature rule, equal-pair rule and calibration-rank check from the
synthetic experiment. It never completes a partition by arbitrary merging.

Let $E$ denote the set of active positions. A current partition $P$ assigns
each position to one component of the edges already processed. Write $K(P)$
for its number of groups. Its calibration graph has one vertex per group,
one vertex per active neuron, and an edge when the group appears in that
neuron. Write $g(P)$ for the graph's number of connected components. The
contrast-space dimension of this partition is

$$
d(P)=K(P)-g(P).
$$

Let $r$ be the SVD rank currently being examined and $m$ the prescribed final
number of groups. A retained partition must have $K(P)=m$ and $d(P)=r$.

## Partial partitions bound the remaining search

Every additional signature edge only merges groups. Thus $K(P)$ and $g(P)$
cannot increase. The dimension $d(P)$ cannot increase either: if $U_P$ is
the position-by-group indicator matrix and $P'$ coarsens $P$, a group-merging
matrix $A$ satisfies $U_{P'}=U_P A$. For the within-neuron centering operator
$\Pi$, it follows that

$$
\operatorname{range}(\Pi U_{P'})
\subseteq \operatorname{range}(\Pi U_P).
$$

Consequently the implementation can reject a partial scan as soon as
$K(P)<m$, $g(P)<m-r$, or $d(P)<r$. These are necessary conditions on any
possible completion, not a heuristic rank choice.

## Neurons with no nonzero pairs force separate components

At zero-rule threshold $t$, call a neuron inactive for the equal-pair rule
when every pair of its active positions has signature norm at most $t$.
Its positions merge by the zero rule, and none can participate in a nonzero
equal-pair match. Its group therefore stays separate from all other neurons.

If $z$ neurons have this property, the final calibration graph has at least
$z$ components, plus one more when any other active neurons remain. A code
budget requiring fewer components is impossible at this threshold.

## Explore whole components before counting isolated groups

A position is processed only after all its relevant radius matches have been
examined. Ordered pairs are present in both orientations, so this covers all
incident equal-pair edges, including occurrences as a second endpoint.

A current component is closed when all its positions have been processed.
No future edge can connect it to another component. If $q$ components are
closed, the final group count is at least $q$, plus one when unprocessed
positions remain. More than the largest eligible budget rejects this
threshold without scanning the rest of the cloud.

The scan prioritizes the smallest unfinished component. This avoids
processing the entire graph merely to establish that it has too many
components. Edges whose two endpoint identifications are already implied by
the current partition can be omitted because they cannot change any group.

## Floating-point screening

The GPU computes float64 squared-distance lower bounds in blocks. Their
roundoff allowance is `64 * eps * (rank + 4)` times the sum of the two squared
norms. The GPU returns every uncertain match; NumPy checks its coordinate
distance directly against the original radius. Large batches are split
instead of truncating their matches. No nearest-neighbour cap is used in
the final streaming backend.

Only completed rank jobs are preserved across performance amendments. The
distance-backend amendments record the code hashes. No target query answers
are used to choose these computational rules.
