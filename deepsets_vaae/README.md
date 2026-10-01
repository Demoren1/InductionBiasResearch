# DeepSets + VAAE: mask-transfer pilot

This experiment learns connectivity inside a shared, raw-pixel item encoder.
Set summation and sharing the encoder across set elements are provided by the
architecture; discovering permutation symmetry is outside this experiment.

Each task assigns ten centered, unit-variance costs to MNIST digits. A label is
the sum of the costs of five images. Only complete-set labels train the models.
Four source tasks produce banks of successful sparse networks. One VAE per
source task models continuous normalized importance maps after common
Hungarian column alignment. Frozen-decoder agreement generates a fixed mask
without accessing target task labels. Eight new cost vectors evaluate transfer.

The task vectors are fixed before the run. Independent seeds vary image pools,
banks, VAEs, latent starts, and downstream initialization. Intervals describe
independent training repeats conditional on this fixed task split.

All sparse methods have exactly 5,018 of 25,088 first-layer edges. The dense
reference has all edges. Methods are agreement, aligned mean, one VAE using
the same initial codes as agreement, random, and dense. All numerical weights
are trained afresh and paired across methods. Target budgets include training
and validation labels: about 80% train and 20% checkpoint validation, using
different image pools. The test is used only after checkpoint selection.

Source training, source validation, target training, target validation, and
target test have mutually disjoint row IDs and exact pixel arrays. Original
handwritten-image identities behind MNIST8m augmentations are unavailable, so
independence by underlying handwritten identity is not asserted.

## Run

Use the `ras` environment. From the repository root:

```bash
python -m deepsets_vaae.run --out outputs/deepsets_vaae/smoke --seed 4199 --smoke
python -m deepsets_vaae.launch --out outputs/deepsets_vaae/20261001_pilot
```

## Export trained target weights

The pilot's target evaluation records do not include numerical weights.  To
recreate them from a completed seed without changing that seed's artifacts,
run:

```bash
python -m deepsets_vaae.export_weights \
  --original-out outputs/deepsets_vaae/20261001_pilot/seed_4100 \
  --seed 4100 --out outputs/deepsets_vaae/weighted_exports/seed_4100 \
  --all-conditions --device cuda:0
```

Without `--all-conditions`, the command exports the representative task 0 at
budget 256.  With it, it exports all eight target tasks and all four budgets.
Every `target_task{task}_budget{budget}.pt` contains one restored checkpoint
batch for all five methods and four paired initializations: CPU
`state_dict` (`weight`, `masks`, `bias`, `readout`, `per_image_offset`),
`effective_weight = weight * masks`, `method_names`, `replica_indices`, and
the matching 20 result records.  `records.json` copies the original records
alongside replayed records; `export_provenance.json` records source hashes and
requires every replayed MSE to differ by at most `1e-4`.

The launcher snapshots GPU computation utilization and starts one repeat on
every GPU at exactly zero utilization, using its UUID. Memory occupancy does
not affect selection. It does not stop or alter existing processes.

Each repeat saves its protocol and source hashes, data provenance, source
banks, VAE checkpoints, masks, diagnostics, progress, and per-task results.
The launcher waits for every repeat and writes `REPORT.md`, `summary.json`,
and learning curves. Failed workers remain visible in `allocation.json` and
their individual logs; incomplete repeats are not silently reported as done.

Save results with plots and explanations for every plot. Reports should explain
axes, aggregation and uncertainty, the supported conclusion, and limitations.
Plots are provided as PNG and PDF, with plotted source values saved in JSON.

This first pilot keeps 32 of 128 bank candidates and trains each VAE on a
small map collection. It is an initial comparison, not evidence that the
bank captures the entire family of near-optimal structures. Training and
validation diagnostics should inform any separate follow-up run.
