# Recovery from contrast-space pair signatures

This experiment applies the constructive rule of manuscript Theorem 1 to
retained dense source models. It is separate from the registered G34 core.
The [protocol](PROTOCOL.md) fixes support proposals, rank/threshold grids,
sign reconstruction, candidate limits and the source evaluation budget.

The method centers log magnitudes inside each neuron, estimates their
shared space by SVD, and merges positions with zero or equal ordered-pair
signatures. It retains complete partitions with the prescribed number of
codes. It does not use k-means or force partial groups into complete ones.
Numerical least squares scores a partition after its groups are fixed.

The retained 10 November source archive is required. This is a retrospective
experiment on 24 conditions with historical outcomes already known; no new
independent confirmation claim is made. Learner functions never open
reference structures or target responses. All candidate banks and all
source-based selections are frozen before the recovery audit.

From the repository root, using the original Python/CUDA environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/udeneev-av/myvenv/bin/python \
  -m experiments.theorem_pair_recovery_v1.run \
  --output artifacts/theorem_pair_recovery_2026-11-12
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/udeneev-av/myvenv/bin/python \
  -m experiments.theorem_pair_recovery_v1.observations \
  --output /tmp/ibr-theorem-pair-20261112
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  /home/udeneev-av/myvenv/bin/python \
  -m experiments.theorem_pair_recovery_v1.execute \
  --work /tmp/ibr-theorem-pair-20261112 \
  --output artifacts/theorem_pair_recovery_2026-11-12
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/udeneev-av/myvenv/bin/python \
  -m experiments.theorem_pair_recovery_v1.score \
  --work /tmp/ibr-theorem-pair-20261112 \
  --output artifacts/theorem_pair_recovery_2026-11-12
MPLCONFIGDIR=/tmp/ibr-mpl-cache \
  /home/udeneev-av/myvenv/bin/python \
  -m experiments.theorem_pair_recovery_v1.plots \
  --output artifacts/theorem_pair_recovery_2026-11-12
```

Extraction and observation preparation require new output directories.
Fitting uses GPUs 0–5, one independent process per GPU. Existing identical
baseline numerical fits are reused; novel structures receive four starts
and 48 Gauss–Newton steps on each of 16 source tasks. Numerical validation
selects states, and a separate source meta sample selects structures.
The combined bank has an additional numerical budget; it is not an equal
compute comparison. No target adaptation or software tests are run.

The numerical state files remain local and ignored by Git. Compact banks,
selections, masks, scores, source-data hash comparison, provenance and
post-freeze diagnostics are retained with the report. Published G34
artifacts and the manuscript are not overwritten.
