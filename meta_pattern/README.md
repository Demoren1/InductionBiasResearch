# Meta-learning full U on pattern-32

Current results, diagnostics, and limits are summarized in
[the 8 September 2026 final report](../mds/FINAL_REPORT_2026-09-08.md).

Independent continuation of `pattern/`: the same binary substring-detection
problem and two-layer ReLU MLP, extended to length-32 inputs and pattern
lengths **3, 4, 5, 6, 7, 8**. The implementation reuses the mathematical task
and matched-filter diagnostic, but replaces the old injected-positive sampler,
importance-map/VAE objective, and masked-MLP training loop. Existing experiments
and their checkpoints are not changed or loaded.

## What is learned

For each length k, a small deterministic MLP generates two **full,
unfactorized** parameter bases:

```
U1(k): [1056, 16]     # (32 input coordinates + bias) × 32 hidden units
U2(k): [33, 4]        # 32 hidden coordinates + output bias
vec(W1_augmented) = U1(k) @ v1_task
W2_augmented      = U2(k) @ v2_task
```

The first layer is reshaped to `[33, 32]`; the final input row holds its biases.
The network is `ReLU([x, 1] @ W1_augmented)`, followed by the second affine map.
Each task adapts only its own 20 coefficients `(v1, v2)`, including the degrees
of freedom for biases and readout. The generator sees **only the scalar pattern
length**, normalized to [-1, 1]. It never sees the pattern bits or examples.
Its SiLU hypernetwork has configurable width and depth; the default is
`1 → 64 → 64 → 17028`. `--unconditional` replaces the scalar input with zero
while preserving the same hypernetwork capacity, providing a direct control
for length conditioning.

The matrices contain 17,028 float32 values total (about 66.5 KiB per length),
but the generator parameters, gradients, optimizer state and unrolled training
graphs cost additional memory. There is no Kronecker factorization. Small
`rank1` and `rank2` still restrict the task weight spaces; full U does **not**
mean unrestricted task weights or guaranteed convolutional structure.

Nonzero U columns are normalized to unit L2 norm to control scale. They are not
forced to be orthogonal. Rank/singular values are saved in evaluation results.
No ideal-mask/U reconstruction loss, KL loss, or latent z is used.

## Data and task separation

- There are 504 binary patterns in the full length-3..8 catalogue.
- Labels are exact substring existence on a **linear**, non-circular sequence.
  Zero, one and multiple occurrences are all allowed.
- A sample is drawn uniformly as a 32-bit integer. Balanced datasets use label
  rejection sampling, with half positive and half negative examples. Odd-sized
  datasets have one extra negative. No pattern is injected into a sequence.
- A fixed hash of the input ID partitions the entire input space into support,
  query and test (approximately 60/20/20), independently of the task and sampling
  seed. Input identity can never cross these partitions, even between tasks.
  Calls to the **same** partition may overlap; they are fresh episodes, not
  guaranteed mutually disjoint episode datasets.
- Task train/validation/test splits keep a pattern, its reversal, its bit
  complement and the reversed complement in the same split. Ratios approximate
  60/20/20 within every length. For length 3 there are only three equivalence
  groups, forcing a 4/2/2 pattern split. This is a small and restrictive family;
  accuracy there should be interpreted separately by length.
- Training cycles lengths with equal exposure, then samples a training pattern
  within that length. Validation and reporting weight lengths equally.
- Final evaluation includes both balanced and naturally sampled test inputs;
  it reports BCE, accuracy, class frequency and per-class accuracy. Natural
  accuracy alone is misleading for short, frequent patterns.

The sampler is CPU-based NumPy bit operations and supports only input length
32 in this protocol. It has a bounded rejection budget and fails explicitly
if an impractical sample request cannot be filled.

## Inner and outer optimization

1. Generate U(k) once for an episode, shared across that task's examples.
2. Initialize fresh v from a private seeded Gaussian distribution.
3. Adapt v on support using functional minibatch SGD or Adam, holding U's
   values fixed.
4. Compute query BCE and differentiate **through all inner updates** into the
   generator. U and adapted v are not detached during meta-training.
5. Average task gradients and update the generator using Adam.

The base configuration uses 20 inner SGD steps at learning rate 0.1, 256
support/query examples, minibatches of 128, six tasks per outer step, and outer
Adam at learning rate 0.001. `--inner-optimizer {sgd,adam}` and
`--init-scale` select the task-vector optimizer and initialization explicitly.
These are experimental settings, **not calibrated optima**; different methods
use matched data, v initialization, and minibatch seeds.

Checkpoint selection uses query BCE on held-out **validation patterns at
training lengths**. It never uses final test patterns or held-out lengths.
The initial checkpoint is eligible, so `best.pt` can remain step 0 if training
does not improve validation. `latest.pt` records the final training state.

At final evaluation, freeze the selected U generator/table and fit fresh v for
each held-out pattern. Report learning curves at 20, 100 and 500 adaptation
steps by default. Evaluate every held-out pattern in the full preset.

## Controls

| Method | Shared structure |
|---|---|
| `generator` | Full U generated from length by a learned MLP |
| `table` | Independent directly learned full U for each training length |
| `random` | Fixed seeded random full U for each length |
| `ideal` | Analytical sliding-filter U, followed by ordinary fresh-v training |

The `ideal` result measures optimization with known structure. It is **not**
the analytical perfect solution: that separate solution is constructed by
`ideal_solution()` and checked independently in tests. The ideal uses only its
necessary basis columns; unused columns are zero. This can make its effective
dimension lower than the dense controls, despite matching tensor dimensions.

The table has no learned prediction for a missing length; evaluation skips
such entries. Generator results are separated into known lengths,
interpolation and extrapolation. Do not average these regimes together.
This implementation tests interpolation within 3..8; expanding beyond that
range requires an explicit protocol/model change.

## Running

From the repository root, using the existing `ras` environment:

```bash
source /home/udeneev-av/miniconda3/etc/profile.d/conda.sh
conda activate ras
python -m unittest discover -s meta_pattern/tests -v

# Four methods × two seeds, one independent process per GPU.
python meta_pattern/scripts/run_suite.py --preset smoke --out meta_pattern/outputs/smoke_example
python meta_pattern/scripts/run_suite.py --preset pilot --out meta_pattern/outputs/pilot_example
python meta_pattern/scripts/run_suite.py --preset full --out meta_pattern/outputs/full_example

# Train one generator, then evaluate its selected checkpoint.
python -m meta_pattern.train --method generator --out meta_pattern/outputs/single
python -m meta_pattern.evaluate --checkpoint meta_pattern/outputs/single/best.pt --out meta_pattern/outputs/single/evaluation.json

# Separate unseen-length experiment: hold out 5 and 7 from meta-training and selection.
python meta_pattern/scripts/run_suite.py --preset pilot --train-lengths 3 4 6 8 --out meta_pattern/outputs/interpolation_example
```

Suite GPU IDs and seeds can be changed with `--gpus` and `--seeds`. The launcher
queues jobs when fewer than eight GPUs are supplied. It checks worker return
codes, writes separate logs, and runs the report only after every job succeeds.
Output paths are protected against accidental overwrite.

`meta_pattern.train` can resume a stopped run from `latest.pt`, writing a
self-contained continuation into a **new** output directory and preserving the
historical `best.pt` used for validation selection. Only `outer_steps` and
`validate_every` may differ from the checkpoint configuration; every model,
data, and optimization setting must otherwise match. For example:

```bash
python -m meta_pattern.train \
  --out meta_pattern/outputs/continuation \
  --resume meta_pattern/outputs/phase1/latest.pt \
  --outer-steps 3000 --validate-every 100
```

For the current held-length protocol, train only on lengths 3, 4, 6, and 8,
select by their validation patterns, then run the dedicated exhaustive evaluator:

```bash
python -m meta_pattern.train \
  --out meta_pattern/outputs/interpolation_example \
  --method generator --seed 42 --train-lengths 3 4 6 8 \
  --all-unseen-patterns --data-seed 20260906 \
  --inner-optimizer adam --init-scale 0.1 --inner-lr 0.1 \
  --inner-steps 50 --tasks-per-step 4 --support-size 1024 --query-size 1024 \
  --batch-size 128 --val-tasks-per-length 0 --outer-steps 1000 \
  --validate-every 100 --device cuda
python -m meta_pattern.evaluate_interpolation \
  --checkpoint meta_pattern/outputs/interpolation_example/best.pt \
  --out meta_pattern/outputs/interpolation_example/evaluation.json \
  --steps 50 500 2000 --repeats 2 --support-size 8192 --test-size 2048 \
  --batch-size 128 --device cuda
```

The analytic-U calibration is sharded by optimizer setting and never reads
lengths 5/7 or their test partition:

```bash
python -m meta_pattern.calibrate \
  --out meta_pattern/outputs/calibration_example --shard 0 --shards 5 \
  --seed 42 --device cuda
```

`smoke` uses three outer steps and tiny data to check execution only. `pilot`
uses 100 outer steps and two test tasks per length; it is a limited diagnostic.
The pilot's final test subset is the first two held-out patterns in lexical
order, so it is not a representative random sample of all patterns. In
contrast, a capped validation subset is selected by a deterministic uniform
sample within each length; `--val-tasks-per-length 0` evaluates every validation
pattern. Do not generalize the pilot averages to all 504 tasks.
`full` uses 1,000 outer steps and all held-out patterns. None of these settings
guarantees convergence; first inspect whether learned-v ideal controls improve.

Every training run saves its configuration, exact task split, source hashes,
train/validation histories, best/latest checkpoint and completion status.
Evaluation records both training and current source hashes, per-task results,
and per-length aggregates. The suite produces `summary.json` and `RESULTS.md`.

## Capacity study and 7 September diagnostics

The first capacity comparison finished five conditional/control models for one
generator seed (42), each with 1,000 outer updates and 50-step Adam adaptation.
For frozen \(U\), fresh \(v\) was then trained for 2,000 steps on every unseen
length-5 and length-7 pattern. The small conditional generator (1.1M
parameters) reached 69.63% / 66.94% balanced accuracy; the 4.5M and 9.3M
conditional models reached 69.54% / 66.78% and 69.39% / 66.87%, respectively.
Increasing generator capacity therefore did not help in this single-seed run.
Seed 43 and the planned 3,000-step continuation were stopped by the user, so
this is not a seed-robust capacity conclusion. The full table, including random,
ideal, unconditional, and wider-\(v\) controls, is in
[`seed42_accuracy/ACCURACY.md`](outputs/interpolation_capacity_20260906/adam50/seed42_accuracy/ACCURACY.md).

The 7 September diagnostic used only 12 known-length validation patterns
(lengths 3, 4, 6, 8), never lengths 5/7 or the test partition. At 2,000
task-vector updates, support-selected optimizer/restart tuning raised frozen
learned-\(U\) query accuracy only to 76.35%, while the analytic \(U\) reached
94.00% under the same policy. Continuing outer training for 50 updates with a
500-step inner horizon improved 2,000-step accuracy by only +0.06 p.p. versus
the matched 50-step continuation (95% descriptive interval [-0.11, +0.24]).
These controls point to the quality of the learned subspace rather than a
simple task-vector optimizer setting or short inner horizon, but they do not
prove convergence. See
[`u_diagnosis_20260907/RESULTS.md`](outputs/u_diagnosis_20260907/RESULTS.md).

During the capacity sweep, the ordinary float32 global gradient-norm reduction
could overflow even when every gradient element was finite. Training now
reduces the norm in float64 before clipping and still rejects genuinely
non-finite gradients.

## What the tests establish

Independent string-based labels agree with the vectorized sampler for all six
lengths, support/query/test partitions are disjoint, and task equivalence groups
do not leak. The analytical detector passes boundary/multiple/near-match cases
and exhaustive small-input tests. A float64 finite-difference check verifies
the full meta-gradient through adaptation. An end-to-end CPU test checks
nonzero training gradients, checkpoint behavior and deterministic evaluation.

Useful next measurements are support-size learning curves, several inner
budgets/ranks, and direct tests of weight sharing in generated spaces. A higher
accuracy alone does not establish that the generator has recovered convolution.

Basis: Zhou, Knowles and Finn, [Meta-Learning Symmetries by Reparameterization](https://arxiv.org/abs/2007.02933),
especially sections 4.1/4.3 and Appendix A. Our length-conditioned generator
and unfactorized small-model protocol are experimental choices in this project.
