# Same VAE: task-loss z optimization versus decoder agreement

This comparison uses **VAE seed 42 from the completed two-decoder experiment**.
Only that decoder's masks are used on both sides; results are not averaged over
VAE42 and VAE43. The single-z baseline is task-adapted, while agreement with the
independently trained VAE43 uses no target labels.

The completed run does not show an advantage for agreement over this single-z
procedure. Single-z final: gold IoU 0.7249, accuracy 0.9225. Agreement after 1000
updates: IoU 0.7046, accuracy 0.9205; after 30 updates: 0.6888 and 0.9194.
The single-z checkpoint selected by search-validation gives 0.7156 and 0.9206.
The final-iterate accuracy difference versus 1000-step agreement is only 0.20
percentage points, with different signs across tasks. These are conditional
results for one checkpoint/split and different label/compute budgets.

The starting points are exactly the 64 stored `initial_z1` vectors. Both methods
use fixed-cardinality soft top-32, temperature 0.5, and a radius-8 latent bound.
The previous plain sigmoid mask mapping is replaced by the same soft top-32
mapping as agreement, so sparsity of the search relaxation is controlled.

The single-z update reproduces the previous direct-gradient recipe: for each
of 30 outer iterations, train fresh masked MLPs for 300 detached-mask warm-up
steps, then 100 live-mask steps accumulating direct gradients into z, and add
the search-validation BCE gradient. The inner Adam trajectory is not
differentiated through. This is not a full bilevel hypergradient.

Every start has its own MLP and z; clipping and projection operate independently
per start. The final z is evaluated alongside a separate best-search-validation
checkpoint drawn from the pre-update states (including initialization). These
are declared variants, not a choice based on final task or gold results.

Controls include the original 1000-update agreement masks and a newly repeated
30-update agreement run with the same starts. The latter controls the number
of outer updates, but not compute: single-z trains 12000 inner MLP steps per
task and has labels; agreement only evaluates decoders. Learning rates also
retain their respective recipes (single-z 0.05, agreement 0.03).

## Data and final evaluation

Four held-out tasks: `0100`, `1011`, `0000`, `0011`.

- Search-validation: 1024 examples, seed `80000 + int(pattern,2)`.
- Search training: a separate on-the-fly seeded stream.
- Final evaluation: existing seed `1000 + int(pattern,2)`, never used in search.
- All fixed masks evaluated using fresh MLPs for 2000 steps, with identical
  initial weights and training batches across methods for each start/task.
- Exact-32 random masks, prior samples, random-pair search and ideal masks are
  included as controls. Gold is only a final structural diagnostic.

Repeated bit strings can occur because this is a finite synthetic family.
The separation concerns independently generated data streams and task-level
meta-training, not a partition of the 256 possible input sequences.

## Reproduce

Use the `ras` environment with GPU access outside the sandbox, setting
`CUDA_VISIBLE_DEVICES` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

```bash
python pattern/evaluation/compare_single_z.py --stage agreement_control
python pattern/evaluation/compare_single_z.py --pattern 0100 --stage all
# Repeat the second command for 1011, 0000 and 0011.
python pattern/evaluation/compare_single_z.py --stage report
```

Outputs are kept separately under
`outputs/decoder_agreement/seed_20260906/single_z_comparison/`. Existing artifacts
are not overwritten. See [the results](outputs/decoder_agreement/seed_20260906/single_z_comparison/RESULTS.md).

This is one VAE checkpoint and one task split. Any observed ordering describes
these specified search procedures; it does not establish superiority over all
possible single-latent optimization methods.
