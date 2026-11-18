# Single-task ratios on Omniglot

This package applies the direct ratio rule from `single_task_recovery_v1`
to the fixed real-data comparison. See [PROTOCOL.md](PROTOCOL.md).

The original dense learner, G34 implementation and common numerical
adapters remain unchanged. Each proposal uses one source episode; the
bank selection can use the original source cohort.

Runtime observations and dense states belong under
`artifacts/local/omniglot_single_task_2026-11-14/`. Compact evidence belongs
under `artifacts/omniglot_single_task_2026-11-14/`; the research report goes
under `paper/theory/`.

Use `/home/udeneev-av/myvenv/bin/python` with `OPENBLAS_NUM_THREADS=1`
and `OMP_NUM_THREADS=2`. Every command takes `--root` for the study and
`--output` for compact evidence, unless it is an unchanged common adapter.

| Entry point | Action |
|---|---|
| `prepare --raw PATH` | Register the protocol and reproduce the prior cohort |
| Common `fitting.real_dense --steps 2500` | Recreate source weight observations |
| `extract` | Exact scalar forests and source-only complete proposal bank |
| `bank_fit --codes K --device cuda:N` | Calibrate and select each available K |
| `finish --stage freeze-source` | Freeze source choices before development |
| Common `fitting.real_validation --method ratio` | Select code budget and numerical rule |
| `finish --stage export-targets` | Verify historical hashes; export support/query inputs |
| Common `fitting.real_target --method ratio` | Fit fresh target numbers on support only |
| `finish --stage freeze-predictions` | Freeze new predictions and comparator hashes |
| `finish --stage score` | Open answers and compute paired comparisons |
| `report` | Render the saved results without training |

The common adapters belong to `experiments.paper_structure_benchmark_v1`;
the other entry points belong to this package. Prepare expects the standard
torchvision Omniglot directory, downloaded for both `background=True` and
`background=False`. GPU calls require the host execution context rather
than the CPU-only filesystem sandbox. The original run used GPU0–3 for
source calibration and GPU0 for dense/development/target adaptation.

Do not rerun fitting to inspect completed evidence. All candidates, source
choices, development curves, target numerical states and exact predictions
are preserved. There is no real-data ground-truth indexator; the report
distinguishes functional transfer from exact structural recovery.
