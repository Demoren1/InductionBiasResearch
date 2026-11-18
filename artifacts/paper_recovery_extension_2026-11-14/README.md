# Expanded recovery comparison, 14 November 2026

[Results](../../paper/theory/PAPER_RECOVERY_EXTENDED_COMPARISON_2026-11-14.md),
[protocol and reproduction](../../experiments/paper_recovery_extension_v1/README.md).

One synthetic transfer benchmark and three applied tasks; Dense, G34,
Pair recovery, LUT-Q, Yeh soft/MAP and MSR-FC. Every applied method uses
the same source observations and downstream numerical solver. Each
dataset has 36 selected source banks, 37 development rules, and38 frozen
target models including Dense and the full-code control. All 114 target
evaluations completed before query scoring. The full-code predictions
match Dense exactly. `audit.json` passed with no access violations.

`mnist/`, `fashion/`, `har/`: registration, Dense training curves/quality,
development choices/models/curves, model and prediction freezes,
`scoring.npz` (truth, logits, per-episode scores for every budget),
`summary.json`, and pre-scoring endpoint checks. HAR intervals use 9 subject
means. The image environments were previously explored; these are new
episodes with frozen source/development choices, not new environments.

`synthetic_results.json`, `synthetic_scoring.npz`: new numerical functions,
all seven methods, common 18-start Adam plus 4-start GN solver.
`synthetic_structure_audit.json`: G34 and Pair recover both signed
structures exactly. Continuous competitors retain their native bases.

`jobs/` and the pipeline logs record completed GPU subprocesses; they
are provenance, not live jobs. `structure_diagnostics.json` distinguishes
zero connections, dominant groups and unused allocated codes.
`budget_grid_frozen.json` records the common finite grid amendment before
development choices. `synthetic_jobs/provenance_path_failure/` and
`synthetic_provenance_fix.json` preserve the initial error and its repair.
It occurred before fitting; the target data and source choices were kept.

The compact bundle reproduces plots/tables without GPUs using `report.py`.
Full source candidates, native states and target numerical parameters
remain under `artifacts/local/paper_recovery_extension_2026-11-14`.
Original three-method results remain in
`../paper_recovery_comparison_2026-11-14/`; they are not the current tables.
