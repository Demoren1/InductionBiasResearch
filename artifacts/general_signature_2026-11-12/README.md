# Сигнатуры на общей синтетике

[Результат и интерпретация](REVIEW.md), [все численные результаты](REPORT.md).
Первый геометрический кандидат взвешенной SVD проверен на 48 новых
численных функциях при прежних шести исходных структурах.

- `predictions/`, `scoring_answers.npz`, `transfer_per_task.csv` — точные
  предсказания, ответы и ошибки всех задач.
- `structures/`, `recovery_structures.npz`, `recovery.csv` — выбранные
  индексаторы и поздний аудит восстановления.
- `states/` — выбранные численные параметры всех 30 уникальных подгонок.
- `curves/` — кривые подгонки для N=1024; `target_generation.npz` —
  параметры генерации новых target-функций.
- `compact_manifest.json` — состав после уборки; прежний
  `bundle_manifest.json` и его копия в `provenance/` описывают исходный экспорт.

Тяжёлый рабочий каталог из `plan.json` удалён по запросу пользователя
12 ноября. Исходные фиксации сохранены как происхождение, а не обещание
наличия всех промежуточных файлов. [Журнал](../../archive/2026-11-12/README.md).

Аудит без обучения:

```bash
python -m experiments.paper_structure_benchmark_v1.reporting.signature_audit \
  --base artifacts/general_signature_2026-11-12 --dataset general \
  --output /tmp/ibr-general-audit.json
```

Код: [general_signature_v1](../../experiments/general_signature_v1/README.md).
