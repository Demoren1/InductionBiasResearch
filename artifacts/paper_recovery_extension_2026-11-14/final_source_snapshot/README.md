# Expanded recovery comparison

The current manuscript comparison: one synthetic transfer benchmark and
MNIST, Fashion-MNIST and UCI HAR. Methods: Dense, G34, Pair recovery,
LUT-Q, Yeh soft/MAP, MSR-FC (basis only). See [the protocol](PROTOCOL.md)
and [results](../../paper/theory/PAPER_RECOVERY_EXTENDED_COMPARISON_2026-11-14.md).

The common constrained grid is 8/16/32/64/128/512. A positive unique index
at every first-layer weight explicitly executes the full budget; its
predictions match Dense bit for bit. It is a common control without
sharing. Larger code budgets can retain a dominant group; group counts,
zeros and unused codes are saved in `structure_diagnostics.json`.

The extension reuses the completed source/development data and small-bank
fits from `paper_recovery_comparison_v1`; that cohort remains immutable.
`prepare.py` creates a separate runtime with hard-linked immutable inputs.
`har.py` reads the official UCI archive and fixes source/development people.
`source.py` prepares HAR Dense using the same Adam/L-BFGS recipe.
The existing native competitor/projection/extraction modules are unchanged.
`workers.py` calibrates every new bank using the common 18-start/2000-update
solver and selects by separate source M; development averages both restarts.
`adapter.py` implements the same optimizer with memory-bounded evaluation
for large signed indexators. Continuous representations are never clustered.

`queue.py` runs one subprocess per GPU; completed job receipts support a
resume after the previous parent has exited. The executed run used all 8
V100 cards alongside existing Yambda processes, without stopping them.
`pipeline.py` waits for the native jobs and the initial ready-only refit
queue, then finishes refit/development. Inspect `jobs/*.json` before any
resume; a failed partial directory needs inspection and archival first.
Do not start duplicate parents for the same output directories.

After all development choices are complete, `evaluate.py --phase freeze`
draws the new common cohort. `queue.py --phase target` fits every frozen
budget support-only. `evaluate.py --phase score` freezes every prediction
before opening query answers. HAR intervals use 9 subject means, not 90
independent people. This is a conditional follow-up on already explored
public image environments; it does not create a new class environment.

The synthetic sequence is `synthetic.py --phase prepare`, `--phase native`,
`--phase source --root ... --device ...` for each geometry, then
`synthetic_targets.py`. It freezes all native/discrete bases before drawing
new numerical functions. The common fit is18 Adam starts/2000 updates plus
four Gauss--Newton starts/48 updates, with 256 fitting and 64 selection
examples; no numerical source parameters transfer.

`audit.py` is a one-time scientific provenance/selection audit of the actual
completed run. It verifies the HPO minima, frozen hashes, structure-only
transfer, source separation and official HAR rows. It is not a software
test suite. Freeze, score and audit are one-time stages; do not rerun them
over completed directories. Existing results should be read directly.

Tables, plots and the research report can be regenerated without GPUs,
private files, runtime checkpoints or retraining:

```bash
MPLCONFIGDIR=/tmp/paper-pair-mpl python -m experiments.paper_recovery_extension_v1.report
make -C paper
```

Runtime: `artifacts/local/paper_recovery_extension_2026-11-14`.
Compact: `artifacts/paper_recovery_extension_2026-11-14`.
The compact bundle includes development models/curves, exact scoring
arrays, Dense curves/quality, freezes, source group diagnostics and audit.
Raw native states and source refits remain in runtime; their registrations
and file hashes are retained. `execution_snapshot` and per-native executed
code retain the versions used for the experiments. `budget_grid_frozen.json`
records the grid amendment before extension development scores existed.
`synthetic_provenance_fix.json` documents an initial provenance-path error
that stopped before training; fixed target observations were not redrawn.
No software tests were created or run. Concurrent timings support no speed
comparison between methods.
