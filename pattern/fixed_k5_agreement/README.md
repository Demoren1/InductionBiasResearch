# Fixed-k5 agreement on length-32 sequences

This is an isolated scale-up of the fixed-length pattern agreement experiment:
32 inputs, 32 hidden units, pattern length 5, and exact-160 masks. It does not
modify or reuse the held-out-k5 arm of `pattern.length_interp`.

The full protocol uses an orbit-safe 24/8 task split, 4,000 masked MLPs per
meta-train task, the best 10% importance maps, and 64 independent VAE pairs
(128 unconditional VAEs). Each pair receives the same 64 latent starts.

Methods:

- label-free two-decoder agreement;
- equal-state-count random latent-pair search selected by agreement loss;
- target-aware individual single-z adaptation on VAE 1 for every held-out task;
- prior, uniform exact-160, and analytic ideal controls.

Individual single-z has access to held-out support/query labels and a much
larger compute budget. It is a supervised comparison, not a matched label-free
baseline. Ideal support and final test examples are excluded from every search.

## Overnight run

```bash
bash pattern/scripts/13_seq32_k5_agreement_night.sh
```

The launcher activates the `ras` conda environment and uses GPUs 0--7. It
prints its timestamped output directory at startup. To choose a stable path:

```bash
OUT_DIR=pattern/outputs/fixed_k5_agreement/my_run \
  bash pattern/scripts/13_seq32_k5_agreement_night.sh
```

Stages are resumable by using the same `OUT_DIR`:

```bash
OUT_DIR=pattern/outputs/fixed_k5_agreement/my_run STAGE=agreement \
  bash pattern/scripts/13_seq32_k5_agreement_night.sh
```

Valid stages are `prepare`, `bank`, `vae`, `agreement`, `individual`,
`evaluate`, `aggregate`, and `all`. Existing artifacts are verified or reused;
partial or protocol-incompatible artifacts fail closed instead of being
overwritten. Logs are stored below `OUT_DIR/logs`.

The final stage creates `summary.json`, `comparison.png`, `mask_examples.png`,
and a preliminary `RESULTS.md`. The latter is intended to be audited and then
turned into the final research note after the night run completes.

## Small execution check

The smoke uses one VAE pair, two starts, two held-out tasks, and tiny budgets:

```bash
source /home/udeneev-av/miniconda3/etc/profile.d/conda.sh
conda activate ras
python -m pattern.fixed_k5_agreement.run \
  --out /tmp/pattern_k5_smoke --smoke --stage all --device cpu --gpus 0
```
