# Широкий подбор aSRR

Серия от 17 ноября 2026 года: повтор прежнего успешного опыта,
расширенный development-подбор и две новые когорты после фиксации выбора.
Протокол находится в [PROTOCOL.md](PROTOCOL.md). Это отдельная серия
от ограниченного сравнения `paper_asrr_v1`.

Проверяются MNIST и Fashion-MNIST с 16 support-примерами на класс
и с 100 fitting + 50 дополнительными numerical-selection примерами;
HAR с тремя support-окнами на активность; Omniglot 5-way с 10/5/5
примерами на класс. Структуры переносят только индексатор, все численные
параметры на новых задачах обучаются заново.

Пути результатов: `artifacts/asrr_tuning_2026-11-17`; рабочие данные:
`artifacts/local/asrr_tuning_2026-11-17`. Завершённые стадии отказываются
перезаписывать результаты. Последовательность исходного запуска:

```bash
python -m experiments.asrr_tuning_v1.replicate prepare --datasets mnist omniglot
python -m experiments.asrr_tuning_v1.replicate queue --datasets mnist omniglot
python -m experiments.asrr_tuning_v1.replicate score --datasets mnist omniglot
```

Затем `banks --dataset DATASET` для mnist/fashion/har/omniglot и стадии
подбора для всех шести постановок:

```bash
asrr_settings=(mnist100 mnist16 fashion100 fashion16 har3 omniglot10)
python -m experiments.asrr_tuning_v1.study prepare --phase d1 --settings "${asrr_settings[@]}"
python -m experiments.asrr_tuning_v1.study run --phase d1 --gpus 0 1 2 3 4 5 6 7
python -m experiments.asrr_tuning_v1.refine prepare --settings "${asrr_settings[@]}"
python -m experiments.asrr_tuning_v1.study run --phase refine --gpus 0 1 2 3 4 5 6 7
python -m experiments.asrr_tuning_v1.refine collect
python -m experiments.asrr_tuning_v1.study prepare --phase d2 --settings "${asrr_settings[@]}"
python -m experiments.asrr_tuning_v1.study run --phase d2 --gpus 0 1 2 3 4 5 6 7
python -m experiments.asrr_tuning_v1.evaluate freeze --settings "${asrr_settings[@]}"
python -m experiments.asrr_tuning_v1.evaluate prepare-targets
python -m experiments.asrr_tuning_v1.study run --phase target --gpus 0 1 2 3 4 5 6 7
python -m experiments.asrr_tuning_v1.evaluate score
MPLCONFIGDIR=/tmp/asrr-tuning-mpl python -m experiments.asrr_tuning_v1.report
```

`study run` возобновляет только очередь без неразобранных ошибок. Нельзя
одновременно запускать два родительских процесса одной очереди.
После обновления и сборки статьи `python -m experiments.asrr_tuning_v1.finalize`
сохраняет научный аудит выбора, снимок кода и комплект воспроизводимости.
Программные тесты не создаются и не запускаются.
