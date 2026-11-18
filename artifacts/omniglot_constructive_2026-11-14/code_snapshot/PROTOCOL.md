# Constructive dense recovery on the saved Omniglot layer

14 November 2026. The user explicitly requested trying the methods in
`paper/theory/CONSTRUCTIVE_DENSE_RECOVERY_2026-11-14.md`.

## Input and isolation

Reuse the exact source dense ensemble and public descriptors from
`artifacts/omniglot_single_task_2026-11-14/`. No dense retraining. Use the
same two source-H-ranked anchors, 60 and 107, and their seven masks:
full, uncertainty multipliers 1/2/3, and per-neuron top25/50/75%.
The accepted-candidate comparison budgets remain 8/16/32/64. Additionally
check applicability for every integer budget 2 through 64 to allow the
new criterion's automatic budget proposal. One task per proposal.

This run uses source weight observations only. Neither development nor
target responses are used to generate or certify candidates. Previous
target results are known; any later transfer comparison is explicitly a
retrospective extension. Original code and prior outputs are preserved.

## Exhausting the scalar saturation condition

Let delta contain the sorted positive-oriented within-neuron log
differences, retaining multiplicities. Append zero at its beginning and
form adjacent gaps. The full ordered cloud is its reflection about zero.
At radius r=4epsilon, the full component count is twice the number of
positive-side gaps greater than r, plus one.

Consequently the condition C=m(m−1)+1 holds exactly when the number of
such gaps is m(m−1)/2. Its entire tolerance interval is obtained from
two adjacent ordered gap values; ties can make the interval empty.
This enumerates all possible saturation tolerances for each input mask
and budget, rather than relying on a grid. On that half-open interval
the component partition is constant. Its maximum diameter and zero
component radius are constant too. If their necessary tolerance is
already at least the interval's excluded upper endpoint, the complete
criterion cannot apply anywhere on that interval. This statement does
not exclude larger-tolerance explanations outside the saturated class.

For the four comparison budgets call the unchanged public
`recover_dense_structure` at the largest representable tolerance in
each nonempty saturation interval. Also call it for any other budget
that passes the width conditions, at the smallest admissible tolerance,
the midpoint and the upper interior endpoint. Keep all complete outputs;
never bypass arithmetic, matching, sign or minimax checks. Inconclusive
is distinct from infeasible at a particular tolerance. Float64 arithmetic
is not a formal interval certificate.

## Cycle/minimax validation and continuation

For all 32 previous direct-ratio candidates compute their optimal maximum
log-error on their own source anchor. Reuse the note's existing Chebyshev
LP. Also report the explicit two-edge-cycle obstruction for positions
given one code in one neuron. No new indexator is inferred by this LP.
For each mask/budget compute the independent local packing lower bound
on attainable uniform log-error, without common-code or cross-neuron
constraints. Record the norm of the omitted dense tail separately.

If complete nontrivial candidates are found, freeze their structures,
then use source H/M for numerical calibration/selection and development
for budget/adaptation selection, with the unchanged common adapters.
Only after freezing those choices may new target predictions be fitted
and scored. If none are found, report that result without assigning an
accuracy to a nonexistent model or fabricating a completed partition.

The general graphical-matroid recognizer is not implemented in the note's
package. Its full-rank premise for budget m requires at least m−1
independent observation profiles, so it cannot be applied to these
one-task inputs at the comparison budgets. Do not replace it by an
unregistered noisy SVD heuristic.

No software tests are added or run. Existing constructive examples and
the two synthetic descriptor replays are retained as prior controls and
are not retrained or recomputed for this Omniglot extension.
