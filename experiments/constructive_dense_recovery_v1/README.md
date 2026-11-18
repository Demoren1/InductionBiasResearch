# Constructive recovery from dense layers

The mathematical statements and proofs are in:

- [Dense recovery](../../paper/theory/CONSTRUCTIVE_DENSE_RECOVERY_2026-11-14.md).
- [Proofs](../../paper/theory/CONSTRUCTIVE_DENSE_RECOVERY_PROOFS_2026-11-14.md).
- [Diagnostics and limitations](../../paper/theory/CONSTRUCTIVE_DENSE_DIAGNOSTICS_2026-11-14.md).

The public function in recovery.py takes log-weight profiles, receiving
neuron indices, an absolute log-error tolerance, an optional code budget
and optional observed signs. It returns complete compact families,
an exclusion at the given budget, or an inconclusive result.
The full criterion requires saturated contrast counts, the two-clique
anchor graph and unique root matchings. It uses float64 and HiGHS;
its output is not an interval-certified mathematical proof.

One-task usage:

~~~python
import numpy as np

from experiments.constructive_dense_recovery_v1.recovery import (
    recover_dense_structure,
)

rows, neurons = np.where(mask)
result = recover_dense_structure(
    np.log(np.abs(weights[rows, neurons])),
    neurons,
    epsilon=1e-6,
    max_codes=4,
    signs=np.sign(weights[rows, neurons]),
)
~~~

A family's indexator is zero based and applies to the supplied positions,
in their input order. neuron_names maps internal neuron indices back
to input names. Code logs and neuron log-scales have shape tasks×codes
and tasks×neurons. The predicted log weight is the code log minus
the neuron log-scale. fixed_signs are position-wise; code_sign_profiles
supply the task-wise signs. free_neurons compactly lists independent
alternative code assignments. Their counts refer to partitions
up to a global renaming of codes.

Reproduce the CPU research illustrations and the saved-descriptor replay:

~~~bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 -m experiments.constructive_dense_recovery_v1.illustrate

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 -m experiments.constructive_dense_recovery_v1.audit
~~~

The first command freezes algorithm outputs and selects the first complete
explanation in a fixed ascending tolerance grid. The second command opens
previously studied reference indexators only after that freeze.
The default output is artifacts/constructive_dense_recovery_2026-11-14.
The descriptor file comes from the previous single-task experiment.
No new training or transfer occurs, and the replay is not a new blind study.
These computations are analytical illustrations and a research replay,
not a software test suite.

The broader graphical-matroid construction and the cycle-mean fitting
criterion are proved in the notes. This implementation covers the
saturated-pair criterion and uses an equivalent linear program for fitting;
a general graphical-matroid recognizer is not implemented.
