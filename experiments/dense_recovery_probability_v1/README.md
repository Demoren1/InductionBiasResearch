# Recovery from fitted dense layers and success probabilities

Mathematical statements:

- [Recovery theorem and probabilities](../../paper/theory/DENSE_RECOVERY_SUCCESS_2026-11-15.md).
- [Proofs](../../paper/theory/DENSE_RECOVERY_SUCCESS_PROOFS_2026-11-15.md).
- [Frozen probability study](../../artifacts/dense_recovery_probability_2026-11-15/REPORT.md).

The public function recover_from_dense accepts an aligned array of fitted
dense weights with shape tasks×inputs×neurons, a simultaneous absolute
weight-error bound, a deterministic lower bound on active magnitudes,
and the number of used code groups. The active lower bound must exceed
twice the error bound.

~~~python
from experiments.dense_recovery_probability_v1.recover import (
    recover_from_dense,
)

result = recover_from_dense(
    weights,
    absolute_error=eta,
    minimum_active=a,
    codes=4,
)
~~~

The function thresholds support, tries normalized sign profiles,
then repeated-pair unions, then the previous arithmetic-root constructor.
A returned index is a signed, one-based inputs×neurons array with zero
for inactive weights. Under the stated error and separation conditions,
it equals the generating grouping up to code renaming and code sign gauges.
The result alone is not an unconditional certificate for arbitrary
neural-network fits; the absolute error bound needs justification.

The new public recover_repeated_pair_partition function lives with its
shared contrast construction in
experiments/constructive_dense_recovery_v1/recovery.py.
The existing saturated-pair constructor was preserved.
bounds.py implements the explicitly defined probabilistic models:
primitive short relations, anti-concentration, random overlap,
sign-profile collisions and Gaussian least-squares errors.

To reproduce the CPU research experiment in a new directory:

~~~bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 -m experiments.dense_recovery_probability_v1.study \
    --output /tmp/dense-probability-new

MPLCONFIGDIR=/tmp/dense-probability-mpl \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 -m experiments.dense_recovery_probability_v1.report \
    --root /tmp/dense-probability-new
~~~

The output directory must not exist, preserving previous frozen results.
The default study uses 120 new random structures for magnitude recovery
and 1000 sign-profile replicates. It fits full dense linear layers from
generated noisy labels by analytic OLS. Tolerances follow the theorem;
true weights do not choose thresholds. Candidates are frozen before
the structural audit. There is no iterative optimization trace.

Settings with different task counts, sample sizes and noise levels are
paired; per-cell replicates are independent. Confidence intervals are
per-cell exact binomial intervals. No software tests were created or run.
The study is a linear-model validation, not a new ReLU or transfer result.
