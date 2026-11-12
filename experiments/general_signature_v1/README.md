# Первый геометрический кандидат: перенос на общей синтетике

Отдельное продолжение проверки взвешенной SVD. Сохранённые source-only
структуры local/quartet_mix/random из серии 10–12 ноября проверяются
на новых численных функциях с одинаковой адаптацией. Сравнение включает
G34, первый геометрический кандидат, его перемешивание и Dense.
В одном из 12 условий полного сигнатурного кандидата нет; оно отмечается
как отказ, без подстановки другой структуры. Это продолжение на прежних
структурах, не независимая проверка по новым source-семействам.

Подробности и зафиксированные настройки — [PROTOCOL.md](PROTOCOL.md).
Из корня репозитория:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  /home/udeneev-av/myvenv/bin/python -m experiments.general_signature_v1.prepare \
  --root artifacts/local/general_signature_2026-11-12/study \
  --output artifacts/general_signature_2026-11-12
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  /home/udeneev-av/myvenv/bin/python -m experiments.general_signature_v1.execute \
  --output artifacts/general_signature_2026-11-12 --gpus 0 1 2 3 4 5 6 7
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  /home/udeneev-av/myvenv/bin/python -m experiments.general_signature_v1.score \
  --output artifacts/general_signature_2026-11-12
MPLCONFIGDIR=/tmp/ibr-mpl-cache \
  /home/udeneev-av/myvenv/bin/python -m experiments.general_signature_v1.plots \
  --output artifacts/general_signature_2026-11-12
```

Подгонки используют исходный `fitting.target_fit`: 18 стартов Adam,
2000 обновлений, четыре GN-старта и 48 GN-обновлений. Сначала фиксируются
все предсказания; отдельный scorer затем открывает query. Одинаковые
канонические структуры внутри одного семейства переиспользуют одну
подгонку. Target-параметры свежие, все свободные bias и readout обучаются.
Результаты всех задач, ответы для аудита, структуры, receipts и снимок
кода сохраняются. После уборки завершённой серии от 12 ноября выбранные
состояния находятся в `states/`, кривые N=1024 — в `curves/`; тяжёлый
рабочий каталог удалён. `plots` умеет читать этот компактный комплект.
Для нового запуска используйте новые каталоги `--root` и `--output`.
Software tests не создаются и не запускаются.

Аудит сохранённых предсказаний без обучения:

```bash
python -m experiments.paper_structure_benchmark_v1.reporting.signature_audit \
  --base artifacts/general_signature_2026-11-12 --dataset general \
  --output /tmp/ibr-general-audit.json
```
