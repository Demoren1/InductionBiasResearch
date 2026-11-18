# Noisy dense extraction: budget selection and transfer

Registered follow-up to the raw96 and paired90 structural audits. Prior audit outcomes are already known. This study must not be described as a fresh confirmation of those audits or as the full native MSR benchmark.

The primary baselines remain G34 and Pair. Their saved candidates and the fixed-scale profile control are frozen by the preceding extraction. This study selects among those candidates; it cannot recover proposals discarded by their previous observed-weight selector. No new index is obtained from the generating structure.

## Fixed cases and information

- r1/2/5, n20/200, clean or observation noise0.1, both registered source-subset seeds:24 cases.
- Use paired masked-refit candidates, including every saved budget and identity-on-mask candidate. Full dense N4760 is an explicit control/fallback.
- Thirty-two separate source functions select budget and relative ridge from0/1e-8/1e-6/1e-4/1e-2/1. Their IDs are fixed by seed35111801+100r+subset_seed, excluding all32 extraction functions. Their labels are clean. This is additional source selection data, shared by all methods; it is not extraction from32 noisy descriptors alone.
- Selection is separate for1 and5 support examples, with the same19 query examples at positions5:24. The relative0.001 risk tie chooses fewer continuous coefficients, then fewer used codes. Neither true m nor rank-dependent latent dimension is supplied to selection.
- G34/Pair plus code-SVD are diagnostic extensions: refit the same32 noisy source functions on each candidate mask, average signed code weights, and perform multiplicity-weighted SVD. Search dimensions3/6/15/30 and the full available source rank. This tests whether previously useful inter-code relations survive noise; it is not a claim that exact equality sharing captures them. No numeric source initialization is transferred.
- All24 choices must freeze before generating new targets. Fresh target seed35111802+rank,256 functions per rank, common inputs/coefficients across methods/cases, clean support and query labels. These tasks share the previous fixed mixing environment; the two descriptor subsets are not independent environments.
- Teacher mixing is read only after the global freeze, to generate targets and post-selection oracle controls. The oracle exact-equality I and oracle continuous map are diagnostics, not selectable candidates.

## Evaluation and limits

Use float64 closed-form relative ridge throughout. Report task MSE and conditional mean confidence intervals, selected m, used coefficients, and candidate hashes. Dense and identity-on-mask controls use independent output regressions; all other bases use equivalent primal/dual solves with the same trace-per-observation ridge scale. Only support examples fit target numbers.

The1/5 support comparison separates index recovery from numerical identifiability of a new task. A recovered I204 has136 unobserved code directions with one68-dimensional output; this does not imply the true6/15-dimensional family is absent. Avoid calling the oracle-I MSE a lower bound for all adaptation rules.

No manuscript edits and no software tests. Preserve the original structural audits and all choices. A runtime error may be corrected with an explicit amendment; do not retune the registered grid using new target outcomes.
