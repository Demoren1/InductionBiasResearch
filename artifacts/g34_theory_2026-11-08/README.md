# G34 theory experiment — 8 November 2026

Compact evidence for
[the report](../../archive/reports/2026-11-08/G34_THEORY_EXPERIMENT_REPORT_2026-11-08.md).
The experiment compares immutable G34, diversity within its original pool,
and proposals derived from log ratios, positive scales and sign profiles.

The primary result on noisy synthetic DeepSets is a 75.52% target-MSE
reduction. Clean DeepSets and one clean Pattern case regress, so the new
branch is not adopted as a general replacement. See interpretation.md and
the full report for independent-family results and limitations.

- transfer.csv: all 2304 per-task/stage/support measurements.
- recovery.csv, costs.csv: 36 source-condition/method records each.
- index_audit.json: selected indexators aligned for display, true I and
  actual first-source dense weights.
- PNG files: quality, source/target loss, mask precision/recall and heatmaps.
- protocol.md, method_registration.json, frozen_results.json: protocol,
  method hashes and the 1472-file prediction freeze.

Audit truth was opened after all 42 prediction files were frozen.
These audit answers are excluded from future learner/model selection inputs.
The raw results directory was removed on 8 November. Exact predictions,
curves, selected structures and previously opened audit answers are
retained in `audit_arrays.tar.gz`; registrations, receipts, executed code
and numerical/scoring records are in `audit_records.tar.gz`.
[The cleanup record](../../archive/2026-11-08/README.md) describes their
contents and reconstruction limits. Source code is preserved in
[the experiment archive](https://github.com/Demoren1/InductionBiasResearch/tree/ba7384914dd7aaabe0ed627e812cbb2b11731a6f/archive/experiments/g34_theory_v1).

Архивы `audit_arrays.tar.gz` и `audit_records.tar.gz` сохранены локально
на рабочем сервере и исключены из Git; в clone доступны сводки, рисунки
и манифесты. Правила хранения — в [README артефактов](../README.md).

## Уборка 12 ноября

Тяжёлые локальные audit_arrays.tar.gz и audit_records.tar.gz удалены по запросу пользователя. Сводки, выбранные структуры, рисунки и исходные фиксации сохранены. Полный повтор требует заново получить исходные данные. Исследовательский код прежней ветки доступен в Git-ревизии ba7384914dd7aaabe0ed627e812cbb2b11731a6f. [Журнал уборки](../../archive/2026-11-12/README.md).
