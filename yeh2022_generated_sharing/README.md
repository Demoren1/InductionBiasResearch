# Yeh et al. (2022) with generated sharing matrices

Isolated experiments based on **Equivariance Discovery by Learned
Parameter-Sharing** (AISTATS 2022).  The original method directly optimizes a
relaxed row-stochastic assignment matrix `A`.  Here the central comparison is

```text
direct:     A = softmax(L)
generated:  A = softmax(G_psi(z))
model:      theta = A v
```

The task-specific numerical parameters `v` are fitted in the lower problem;
the structure is selected on validation data and hardened before the final
train+validation refit.

Sources:

- paper: <https://proceedings.mlr.press/v151/yeh22b/yeh22b.pdf>
- official release: <https://github.com/raymondyeh07/equivariance_discovery>

## Benchmarks

| Benchmark | Paper location | Implementation |
|---|---|---|
| Gaussian shared means | Sec. 5.1, Figs. 2–6 | batched analytic lower solve |
| standard/alternating sum | Sec. 5.2, Fig. 7 | paper MLP and generated/direct position sharing |
| 1D cross-correlation | Sec. 5.3, Fig. 8 | analytic lower solve, explicit valid correlation |
| unit-step denoising | Appendix D.1 | analytic lower solve, Toeplitz comparison |

Every experiment reports `no_sharing`, `oracle`, `direct`, and `generated`.
Partition distance is invariant to renaming the sharing groups.

The main generated-structure experiment uses one generator over five data
tasks and compares two latent modes:

```text
per-task: U_tau = G_psi(z_tau)
global:   U_tau = G_psi(z)
```

For the linear benchmarks, the selected setup uses the exact task loss,
binary masks in the forward pass, and a straight-through softmax gradient.
Hyperparameters were selected on the smallest cross-correlation validation
problem and separately on denoising validation data; test data were not used
for selection.

The current Gaussian `generated` result is a **transductive multi-task**
experiment: one generator is learned jointly on the same Monte Carlo tasks
that are evaluated, while every task has its own latent.  It tests whether a
generator can compactly represent the discovered family, but is not yet a
held-out transfer result.  The final report labels this explicitly.

## Reproducibility boundaries

The official release contains Gaussian and sum-of-numbers code only.  Its
referenced `projects/ConvSharing` directory is absent, so cross-correlation and
denoising cannot be byte-for-byte reproduced.  This package therefore states
all missing choices explicitly:

- cross-correlation uses no-padding/valid boundaries;
- `N(0, 0.1)` is interpreted as standard deviation `0.1`;
- sum inputs follow the paper's inclusive set `{1, ..., 10}` (the released
  code instead samples `{1, ..., 9}`);
- noisy sum labels are used only for train/validation; test labels are clean;
- the paper states Adam for Gaussian, while its released main loop uses
  RMSprop; both are selectable and the release-compatible default is RMSprop.
- the released Neumann routine omits the leading step-size factor in the
  inverse-Hessian series; this implementation includes it and records that
  mathematical correction as a protocol deviation for sum-of-numbers.

## Quick checks

```bash
source /home/udeneev-av/miniconda3/etc/profile.d/conda.sh
conda activate ras
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m unittest discover -s yeh2022_generated_sharing/tests -v

python -m yeh2022_generated_sharing.run_gaussian \
  --quick --device cpu --output-dir yeh2022_generated_sharing/outputs/smoke_gaussian

python -m yeh2022_generated_sharing.run_linear \
  --benchmark cross_correlation --device cpu --outer-steps 10 \
  --input-length 5 --kernel-length 3 --train-size 12 --validation-size 12 \
  --test-size 32 --output yeh2022_generated_sharing/outputs/smoke_crosscorr
```

`scripts/run_main_generated.sh` reproduces the principal generator results;
`scripts/run_paper_suite.sh` contains the wider paper sweep. Both assign one
CPU core to each GPU process. Generated outputs live under `outputs/` and are
ignored by git.
