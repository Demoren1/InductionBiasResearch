# Circular Motif-Pair OOD Experiment

This experiment tests whether structural information learned from successful
networks transfers when the useful support is task-dependent.

## Model choice

The primary generator is a gap-conditioned CVAE. An otherwise identical
unconditional VAE is trained as an ablation. VQ-VAE is intentionally deferred:
fixed-cardinality top-K decoding already produces discrete masks, while a
codebook would add a second collapse mode without testing the conditioning
hypothesis.

For this first version, the condition is only the one-hot gap. Motif identities
are not supplied because they determine learned weight signs, not the binary
support. Train and test are globally disjoint in the ordered motif pair
`(A,B)`, while every motif role and every gap remains covered in meta-train.
Thus the claim under test is compositional OOD for unseen motif pairs at seen
structural regimes, not extrapolation to unseen gaps.

## Task

- Circular input sequence: 16 values in `{−1,+1}`.
- Task: `tau=(A,B,g)`, with distinct 3-bit motifs and `g in {3,...,10}`.
- Positive: the unique circular occurrence of `B` starts exactly `g` positions
  after the unique occurrence of `A`.
- Negatives contain the same unique motifs at another allowed gap.
- The exhaustive `2^16` sequence universe is filtered before sampling; positive
  injection is not used.
- Gold support: hidden unit `h` sees the three positions at `h` and the three
  positions at `h+g` modulo 16. Shape `16×16`, exactly 96 active edges.

The default deterministic catalog has eight feasible tasks per gap. Six per
gap are meta-train and two are held out: 48 train / 16 test tasks. All 16 test
motif pairs are unique and none appears in train at another gap.

The generator target is the continuous normalized `|W1 * mask|` importance
map of each task's top 10% candidate MLPs (ranked by validation BCE).  It is
not the candidate's binary connectivity mask.  Hidden columns are
canonicalized before VAE/CVAE training because they are exchangeable.
Canonicalization only permutes columns: it preserves every continuous map
value and uses neither the task gap, labels, nor held-out tasks.

## Compared priors

- `random_exact96` (primary null);
- Bernoulli random with density `0.375`;
- train-only conditional mean importance;
- unconditional VAE;
- gap-conditioned CVAE;
- CVAE decoded with an intentionally wrong gap;
- ideal support (diagnostic ceiling).

All generated masks are top-96. Evaluation trains fresh masked MLP weights on
held-out tasks. Target-task architecture selection and latent optimization are
not used. Structural similarity is Hungarian best-permutation IoU.

## Pipeline

The recommended entry point runs every CUDA stage, performs the train-only
beta sweep, evaluates held-out tasks, and writes stage-wise PNG/PDF plots plus
`plots/manifest.json`:

```bash
GPU_IDS="0 1 2 3" BETA_GPU_IDS="3 6 7" SPLIT_SEED=42 ./run_all.sh
```

`BETAS` overrides the default log-scale grid
`1 0.3 0.1 0.03 0.01 0.003 0.001 0.0003`.  The largest beta with a stable,
non-collapsed posterior on the internal validation split is promoted.  The
held-out OOD tasks are never consulted during this choice; every candidate,
threshold, and failure reason is recorded under `generative/selection.json`.

Every split-specific run keeps its datasets under `$RUN_ROOT/data`.  SHA256 of
the exact split manifest and the ordered train-task list are checked at every
artifact boundary (data, candidate shards, importance, generator, evaluation),
so stale files from another split fail fast.  A full run also clears only old
generated candidate shards before retraining; set `CLEAN_CANDIDATES=0` solely
for an intentional manual resume.

The individual stages remain available for debugging:

```bash
conda activate ras
DATA_GPU=3 SPLIT_SEED=42 SPLIT_JSON=outputs/split.json bash scripts/01_generate_data.sh
GPU_IDS="0 1 2 3" SPLIT_JSON=outputs/split.json bash scripts/02_train.sh
POSTPROC_GPU=3 SPLIT_JSON=outputs/split.json bash scripts/03_select.sh
GEN_GPU=3 SPLIT_JSON=outputs/split.json bash scripts/04_train_generative.sh
EVAL_GPU=3 SPLIT_JSON=outputs/split.json bash scripts/05_eval.sh
```

The older fixed-beta split-specific wrapper is also retained:

```bash
GPU_ID=3 SPLIT_SEED=42 GPU_IDS="0 1 2 3" bash scripts/09_ood.sh
```

Defaults: 512 candidate MLPs per meta-train task, 1,000 Adam steps,
continuous importance maps from the top 10% candidates, 80 VAE epochs,
64 downstream masks per method, and
1,000 downstream training steps.

All production numerical CLIs require CUDA and fail instead of silently
falling back to CPU. CPU is used only for JSON/files and the small structural
Hungarian diagnostic.

## Current result

With top-10% continuous importance targets and train-only beta tuning, seed 42
gives CVAE accuracy `0.7480` versus `0.7274` for exact-96 random (`+0.0206`,
paired 95% CI `[+0.0109, +0.0304]`) and `0.7381` for the unconditional VAE.
The correct-gap CVAE is structurally better than wrong-gap, although their
accuracy interval still crosses zero.  The train-only conditional mean reaches
`0.7497`, so the learned generator has not beaten that simple baseline.  This
is positive single-split evidence, not yet a multi-seed confirmation. See
[`OOD_RESULTS.md`](OOD_RESULTS.md) for the full interpretation.

## Interpretation boundary

The support depends on gap but not motif identity. A positive result shows
that a conditional generator can select a shared gap-specific structural prior
for unseen motif-pair tasks. A separate harder experiment must hold out entire
gap values and use a condition encoding suitable for interpolation.
