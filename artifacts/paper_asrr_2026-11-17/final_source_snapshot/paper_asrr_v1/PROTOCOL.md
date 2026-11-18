# aSRR in the manuscript benchmark

Registered on 17 November 2026 before fitting the new aSRR banks.

This adds a method to the already evaluated 14 November benchmark. It reuses
the exact source/development observations and target episode draws, rather
than claiming a new independent test environment. Existing baseline results
and their immutable artifacts are retained. No aSRR choice uses target
responses; freeze all development choices and predictions before scoring.

## Candidate generator

Use the existing Ward aSRR implementation in `srr_approx_v1/core.py`.
Normalize each Dense first-layer column; match retained replicas to the
selected fit and average their directions. Align source neurons to the
best H-ranked source by Hungarian matching. Magnitudes are retained when
comparing profiles up to a fixed sign. Keep full support.

At each applied budget 8/16/32/64/128/512, propose at most seven indexators:
global signed Ward, scale-synchronized signed Ward, separate neuron Ward
trees, common input-coordinate Ward, the same input family from selected
Dense fits, and positive-scale A1 refinements of the first two seeds.
Duplicate indexators are removed. The local and input families only exist
at feasible budgets; input groups use an integer number of codes per neuron.
Synchronization uses a fixed auxiliary 256-code cut. A1 runs at most 12
iterations and retains its unmodified seed. This fixes the candidate count
before source M/development selection, instead of adapting refinements
after observing development performance as in the earlier aSRR study.

Source fitting and selection call the existing manuscript worker unchanged:
18 starts, 2000 Adam updates, numerical H checkpoints, independent source M
selection among candidates. Only the selected indexator transfers. The
development optimizer averages two starts and selects budget, learning
rate, activity quantile and update count. The full endpoint is the common
Dense model and its existing development choice. Target fitting uses one
fresh start with these frozen settings. All numeric source values are
discarded. Accuracy intervals use 96 image episodes or nine HAR subject
means. Also retain every constrained budget for the budget plot.

MNIST and Fashion use exactly the saved Dense archives (verify SHA256).
HAR source inputs and Dense fits were deleted on 16 November; reconstruct
the official observations with the original seeds and rerun the original
12-start Adam / four-continuation L-BFGS recipe. Compare the recovered
archive and retained descriptors with their original registrations. If
the archive differs, record that fact explicitly and do not describe it
as identical source weights.

## Two fixed synthetic geometries

Reuse the two original 16-by-16, four-code geometries and each one's single
fitted source task. Use its selected normalized Dense directions, without
the private teacher indexator. The same aSRR generator supplies global and
synchronized candidates and their A1 refinements; local/input families are
infeasible at four codes. Fix the auxiliary synchronization cut to 24, as
in the earlier controlled aSRR study. Use a constant 0.0003 uncertainty
floor when no replica archive is available. Source fitting is the common
18-start Adam / four-start Gauss--Newton routine, with independent source M
selection. Reconstruct the four target numerical functions using the
original seed, after freezing the source bank. Use the same 256/64/2048
fitting/selection/query sizes and original fresh numerical solver/RNG.
Open teacher structures only during target generation and final structural
auditing; extraction and source fitting cannot access private files.

No software tests are created or run. Necessary scientific checks cover
source hashes, the frozen selection, data access receipts, finite values,
budget sizes and the comparison with retained original observations.
