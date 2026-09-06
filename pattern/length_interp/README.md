# Pattern-32 CVAE interpolation

Run from any directory:

```bash
bash /home/udeneev-av/ResearchProject/pattern/scripts/11_length_interp_hour.sh
```

The script activates `ras`, uses all eight GPUs, records console/worker logs,
and limits execution to 60 minutes plus at most 30 seconds for forced cleanup.
It exits earlier if all stages finish. The default output directory is
`pattern/outputs/length_interp/hour_YYYYMMDD_HHMMSS`.

## Hour script settings

- x: length 32; hidden layer: 32 ReLU units.
- CVAE training and validation lengths: 3, 4, 6, 8.
- Held-out interpolation lengths: 5 and 7, all 160 possible patterns.
- 1,000 MLPs per bank task, 2,000 Adam steps, batch 128, learning rate 0.001.
- Independent random exact-K masks, K=32×pattern_length. No ideal masks enter
  the learned bank. Support pool 8,192; separate query set 2,048.
- Select exactly the best 100 networks per task by query BCE. Save unsigned
  `abs(W1*mask)` importance normalized by each map's maximum.
- CVAE latent 32, hidden 256, BCE sum per map + 0.1 KL, 80 epochs, seeds 42/43.
  Its only condition is the scalar `2*(length-3)/5-1`, not a one-hot vector.
- CVAE train/validation patterns are separated by reversal/complement orbit
  at known lengths. Training balances lengths rather than allowing length 8
  to dominate. No banks at length 5 or 7 are built for CVAE.
- Final evaluation freezes CVAE, samples 32 masks per condition, then trains
  fresh MLPs for 2,000 steps and two initialization repeats on each held-out
  pattern. Support/query/test input IDs use globally disjoint hash partitions.

Controls: sampled CVAE, wrong lower/upper conditions with the same z and
target-length K, z=0 CVAE, interpolation of train-only mean importance maps,
random exact-K masks, and ideal sliding-window masks. Every method uses the
same initial weights and minibatch sequence per mask ordinal. Ideal masks
repeat valid linear windows to fill all 32 hidden columns, matching the old
pattern convention; sequences are not circular.

The report separates lengths 5 and 7, includes paired task bootstrap intervals,
and saves natural-distribution results and column-matched structural IoU.
Bootstrap intervals are conditional on the two trained CVAE seeds; they do not
establish uncertainty across arbitrary independent CVAE training runs.

This directory is an isolated extension within `pattern`; the old length-8
configuration and outputs are not overwritten. It reuses the audited sampler
from `meta_pattern.data`, but trains masks/importance CVAE, not U.

## Overrides and runtime

```bash
TIME_LIMIT=90m BANK_MLPS=2000 bash pattern/scripts/11_length_interp_hour.sh
OUT_DIR=/absolute/path/to/new/output GPU_IDS="0 1 2 3 4 5 6 7" bash pattern/scripts/11_length_interp_hour.sh
```

Runtime depends on other GPU workloads. An hour is a cap, not a guarantee of
completion. On timeout all owned workers are stopped, completed bank files and
CVAE checkpoints remain, and the script exits nonzero. It does not fabricate a
final report for incomplete stages. Automatic resumption is not implemented;
use a new output directory for a new full invocation.

The lower-level Python launcher defaults to 2,000 bank MLPs; the hour wrapper
explicitly chooses 1,000 to leave more time for CVAE and downstream evaluation.

```bash
python -m unittest discover -s pattern/length_interp/tests -v
python -m pattern.length_interp.run --smoke --out pattern/outputs/length_interp/new_smoke
```

A complete eight-GPU smoke run and 13 targeted tests passed before delivery.
The smoke is an execution check, not a quality experiment.
