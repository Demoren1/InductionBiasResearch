# G34: general synthetic, 8 ноября 2026

[Итоговый отчёт](../../archive/reports/2026-11-08/G34_GENERAL_SYNTHETIC_REPORT_2026-11-08.md).

Проверка неизменённой theory-ветви на local/quartet_mix/random:
16 входов, 16 ReLU-нейронов, 4 кода, 1/2 выхода, свободные выходные
коэффициенты и два bias-страта. Source noise 0/.2, 16 source и
8 новых target-функций на семейство; размеры выборок — в protocol.md.

- transfer.csv, recovery.csv, costs.csv: качество переноса,
  восстановление структуры и стоимость с полной точностью.
- index_audit.json, PNG: true/selected I и реальные dense-веса,
  качество и source/target-кривые.
- theory_conditions.csv, theory_diagnostic.json: диагностика
  условий при истинной маске и teacher-выравнивании после freeze.
- Регистрации, commitments и frozen_results.json: протокол и хеши
  исходных замороженных результатов.
- audit_arrays.tar.gz: точные предсказания всех сравниваемых методов,
  кривые, выбранные структуры и ранее открытые ответы для аудита.
- audit_records.tar.gz: receipts, снимки кода, записи численного
  fitting и scoring с исходными путями.

Сырые results/20261108_general/ удалены при очистке 8 ноября.
[Исходники](https://github.com/Demoren1/InductionBiasResearch/tree/ba7384914dd7aaabe0ed627e812cbb2b11731a6f/archive/experiments/g34_theory_v1)
перенесены в архив. [Состав и границы восстановления](../../archive/2026-11-08/README.md)
отличают сохранённый аудит от полного повторного обучения/scoring.
Аудитная истина не является входом обучения или отбора структур.

Архивы `audit_arrays.tar.gz` и `audit_records.tar.gz` сохранены локально
на рабочем сервере и исключены из Git; в clone доступны сводки, рисунки
и манифесты. Правила хранения — в [README артефактов](../README.md).

## Уборка 12 ноября

Тяжёлые локальные audit_arrays.tar.gz и audit_records.tar.gz удалены по запросу пользователя. Сводки, выбранные структуры, рисунки и исходные фиксации сохранены. Полный повтор требует заново получить исходные данные. Исследовательский код прежней ветки доступен в Git-ревизии ba7384914dd7aaabe0ed627e812cbb2b11731a6f. [Журнал уборки](../../archive/2026-11-12/README.md).
