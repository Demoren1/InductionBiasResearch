# Scalar-conditioned gap-OOD experiment

## Question

This protocol tests whether the generator learns a gap-dependent structural
inductive bias, rather than only recalling a category that was present during
meta-training.  Every meta-test gap is absent from the CVAE training tasks.
The ordered motif pairs are deliberately shared across gaps, so motif identity
and pair composition are controlled while the structural variable changes.

The legacy one-hot, pair-disjoint experiment remains available as a transfer
control.  It should not be described as unseen-gap generalization.

## Primary design

- Gap condition: one normalized scalar, `c(g) = (g - 3) / 7`.
- Eight valid gaps: `3, ..., 10`.
- Six meta-train gaps (48 tasks) and two meta-test gaps (16 tasks).
- Eight identical feasible ordered `(A, B)` pairs at every gap
  (`pair_policy=shared`).
- The CVAE, beta selection, conditional-mean baseline, and all checkpoint
  selection use meta-train artifacts only.
- Every generated mask has exactly 96 active connections before its downstream
  MLP weights are trained from scratch.

Two profiles are provided:

| Profile | Held-out gaps | Interpretation |
|---|---|---|
| interpolation | `5, 8` | both scalar values lie between observed train gaps |
| extrapolation | `3, 4` | both scalar values lie below the observed train range |

The tempting endpoint split `{3, 10}` is intentionally not used.  After the
exchangeable hidden columns are aligned, the ideal support for gap 10 is
exactly permutation-equivalent to the support for gap 6.  Thus `{3, 10}` would
not make both test structures unseen.  In contrast, with held-out `{3, 4}` the
canonical test separations are absent from train as well as outside the scalar
training range.

## Baselines and measurements

The downstream evaluation compares CVAE, unconditional VAE, uniform exact-96
random masks, Bernoulli random masks, a train-only gap-mean baseline, a
wrong-gap CVAE ablation, and the ideal support.  The train-only mean linearly
interpolates between nearest observed gaps and clamps to the nearest train
boundary outside their range.  The wrong-gap ablation uses the same latent
sample and changes only the gap condition.  If two observed gaps are equally
near an interpolation target, both conditions are evaluated and averaged;
choosing only the lower neighbour would make the result depend on an arbitrary
tie-break.  A stricter condition-fidelity diagnostic also reports correct-gap
IoU minus the better of the two wrong-gap IoUs, with a paired latent bootstrap
interval.

Reported quantities are downstream validation BCE and accuracy after fresh
MLP training, plus best-hidden-permutation IoU against the ideal Toeplitz
support.  Held-out labels, losses, and ideal masks are never used to select a
CVAE checkpoint or generated sample.

## Commands

Run from `motif_pair/` in the `ras` conda environment:

```bash
# gaps 5 and 8 held out
bash scripts/10_gap_ood.sh

# gaps 3 and 4 held out
GAP_OOD_PROFILE=extrapolation bash scripts/10_gap_ood.sh
```

The full numerical pipeline is CUDA-only and fails rather than silently
falling back to CPU.  GPU allocation and experiment scale can be overridden,
for example:

```bash
GPU_IDS="0 1 2 3" BETA_GPU_IDS="0 1 2 3" \
  N_MLPS_PER_TASK=512 EVAL_N_MASKS=64 bash scripts/10_gap_ood.sh
```

## Artifacts

The default run roots are:

- `outputs/ood/gap_interp_g05_g08_seed_42/`
- `outputs/ood/gap_extrap_g03_g04_seed_42/`

Each run stores `split.json`, `experiment.json`, split-specific data and
checkpoints, generator selections, raw evaluation records, a summary, and a
plot manifest.  Figures are saved as PNG and PDF.  In particular:

- `plots/00_gap_ood_design.{png,pdf}` visualizes the protocol and ideal
  supports without claiming a numerical result;
- `plots/11_gap_ood_generalization.{png,pdf}` is the publication-oriented
  held-out accuracy/IoU comparison generated after evaluation.

Checkpoint and evaluation provenance includes the raw split SHA-256, ordered
meta-train task list, condition encoding and scalar normalization, baseline
source gaps, and the wrong-gap policy.

## Current interpolation diagnostic (seed 42)

With both equally-near wrong conditions retained, the correct-gap CVAE reaches
accuracy `0.7355` and IoU `0.6082`, versus `0.7348` and `0.6050` for the
nearest-wrong-gap aggregate.  The paired accuracy difference is only
`+0.0008` (95% t interval `[-0.0006, +0.0021]`), whereas the IoU difference is
`+0.0033` (95% t interval `[+0.0015, +0.0050]`).  This removes the previous
lower-neighbour tie artifact, but it does not establish condition fidelity:
the strict margin against the stronger adjacent condition is `-0.0167` on
average and is positive for `0/16` held-out tasks.  The auditable outputs are
the canonical `outputs/ood/gap_interp_g05_g08_seed_42/eval/` artifacts.
`eval_all_nearest/` retains the narrow two-method rerun and its merged copy as
an additional audit trail; it is not the pipeline's canonical output directory.
