# Omniglot: первый геометрический кандидат взвешенных сигнатур

Эксперимент завершён. Основной вариант дал 34,62% на 1000 новых эпизодах,
G34 в этом повторном прогоне — 41,02%, Dense — 37,07%.

- [Результат, структуры и независимая оценка](REVIEW.md) объясняют выбор
  ранга 0, дополнительную проверку восьми групп и ограничения гарантии.
- [Основной отчёт](REPORT.md), [метрики](summary.json) и `per_episode.npz`
  содержат первичное сравнение и парные интервалы.
- [Дополнительная диагностика восьми групп](k8_diagnostic.json) оценивает
  заранее замороженные структуру и численный адаптер после основной проверки.
- `registration.json`, `distance_backend_amendments/`, `ranks/`,
  `partitions_frozen.json` сохраняют полный перебор 512 комбинаций.
- `geometric_choices_frozen.json`, `development_choices_frozen.json`,
  `predictions_frozen.json` фиксируют порядок выбора и открытия ответов.
- `models/` содержит выбранные структуры, настройки и предсказания.
- `final_code_snapshot/` и `execution_code_freeze.json` сохраняют код.
- [Контроль эквивалентности правил](merge_rule_equivalence.json) и
  [вычислительные следствия теоремы](../../experiments/omniglot_signature_v1/COMPUTATION.md)
  описывают точный потоковый поиск.

При [уборке 12 ноября](../../archive/2026-11-12/README.md) удалены полные
исходные ансамбли, observations, development-кэши и изображения завершённого
запуска. В `scoring_answers.npz` сохранены ответы всех 1000 эпизодов;
`models/` содержит все предсказания и выбранные состояния структурных моделей,
а для Dense — состояние первой задачи. `compact_manifest.json` фиксирует
сохранённый комплект. Аудит точности и CE без обучения:

```bash
python -m experiments.paper_structure_benchmark_v1.reporting.signature_audit \
  --base artifacts/omniglot_signature_2026-11-12 --dataset omniglot \
  --output /tmp/ibr-omniglot-audit.json
```

Исходные панели воспроизведены по прежним хешам; численные модели обучены
заново. Исторические предсказания benchmark 7 ноября не перезаписывались.
