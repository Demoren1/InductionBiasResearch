# Выбор изученного индексатора по данным новой задачи

Продолжение `basin_structure_transfer_v1`: те же замороженные AR-модели,
новые численные задачи и полностью новые кодбуки в каждой попытке.

Первая серия проверяет исправление первого индекса и ранний выбор I
по качеству короткого численного обучения. На задачу выделено 65536
обновлений модели. Gate и финальная query независимы. Истинные структуры
и точный population-риск используются только oracle и диагностикой.

```bash
CUDA_VISIBLE_DEVICES=5 python3 -m experiments.basin_routing_v1.run --frequency 4
CUDA_VISIBLE_DEVICES=6 python3 -m experiments.basin_routing_v1.run --frequency 6
CUDA_VISIBLE_DEVICES=7 python3 -m experiments.basin_routing_v1.run --frequency 8
python3 -m experiments.basin_routing_v1.report
```

[Протокол](protocol.json), [результаты](../../reports/worknotes/BASIN_ROUTING_RESULTS_2026-11-02.md).
Готовые каталоги запусков не перезаписываются. Снимки кода и хеши входов
создаются до генерации итоговых функций; после запуска хеши сверяются.

После первой серии объявлен [отдельный протокол](cue_protocol.json) на
ещё одной новой выборке: резерв попыток, два лучших I и выбор I по
эмпирическому среднему x*y. Признак используется только для выбора
структуры; численные параметры не получают его как инициализацию.

```bash
CUDA_VISIBLE_DEVICES=5 python3 -m experiments.basin_routing_v1.cue_run --frequency 6 --cue present
CUDA_VISIBLE_DEVICES=6 python3 -m experiments.basin_routing_v1.cue_run --frequency 8 --cue present
CUDA_VISIBLE_DEVICES=7 python3 -m experiments.basin_routing_v1.cue_run --frequency 6 --cue absent
CUDA_VISIBLE_DEVICES=5 python3 -m experiments.basin_routing_v1.cue_run --frequency 8 --cue absent
```

Отрицательный контроль удаляет слабую низкочастотную компоненту.
Настройки оптимизаторов остаются прежними: это проверка механизма,
а не новый независимо настроенный набор оптимизаторов для другой цели.
[Теория](../../reports/worknotes/theory/INDEXATOR_TASK_ROUTING_THEORY_2026-11-02.md).

Дополнительные сильные контроли без изученной библиотеки используют
тот же признак для выделения групп или плотной инициализации.
Для каждой из четырёх комбинаций frequency/cue:

```bash
CUDA_VISIBLE_DEVICES=5 python3 -m experiments.basin_routing_v1.cue_controls --frequency 6 --cue present
python3 -m experiments.basin_routing_v1.cue_report
```

Отчёт строится после завершения всех четырёх условий и контролей.
[Полные результаты](../../reports/worknotes/INDEXATOR_TASK_CUE_RESULTS_2026-11-02.md),
[проверка завершения и бюджетов](results/completion_audit.json).

Все обучения выполняются на свободных GPU. Модульные тесты не добавляются.
