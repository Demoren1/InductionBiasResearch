# aSRR under the existing manuscript protocol

17 November 2026. Added-method evaluation on the original source, development and target observations. Baselines remain immutable in `artifacts/paper_recovery_extension_2026-11-14`.

Selected banks/curves and all query predictions/answers are retained; `scientific_audit.json` checks the 18 applied budget choices. `source_manifest.json` indexes the final source snapshot. Registration precedes candidate fitting; choices and predictions were frozen before aSRR scoring. HAR Dense was regenerated and matches the original archive SHA256 exactly.

Report: `paper/theory/ASRR_MATCHED_RESULTS_2026-11-17.md`.

Regenerate manuscript tables and the budget plot without GPUs:

```bash
MPLCONFIGDIR=/tmp/paper-asrr-mpl python -m experiments.paper_asrr_v1.report
make -C paper
```

Do not repeat freeze, scoring or finalization over completed folders. No software tests were created or run.
