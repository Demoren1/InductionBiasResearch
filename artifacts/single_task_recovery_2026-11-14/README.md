# Single-task ratio recovery

[Result](REPORT.md), [protocol](../../experiments/single_task_recovery_v1/PROTOCOL.md), [code](../../experiments/single_task_recovery_v1/README.md).

- recovery.csv / structural_audit.json: frozen-bank recovery and cost.
- transfer_per_task.csv / transfer_results.json: every target MSE.
- recovery_structures.npz: reference, banks, selected indexators.
- single_source_descriptors.npz: fitted descriptors for CPU replay.
- transfer_predictions.npz: predictions and post-freeze audit answers.
- diagnostics/: oracle scale/noise controls.
- receipts/, targets/*/*/access_receipt.json: data-access records.
- code_snapshot/executed/: source before presentation-only formatting.

All raw inputs and numerical states remain local. Compact NPZ exports are explicitly retained. Rebuild this report without training using code_snapshot/export_report.py. No source or target outcomes may be used to tune a purported new blind confirmation.

[Графики PNG/PDF](figures/GALLERY.md).
