# Общий старт исходных моделей для aSRR

Эксперимент MNIST от 18 ноября. Описание выбора и численных правил
находится в [протоколе](PROTOCOL.md). Исходные данные и независимо
обученные модели берутся из существующей серии статьи.

Команды запуска новой серии из корня репозитория.

```bash
python -m experiments.asrr_anchor_init_v1.source prepare
python -m experiments.asrr_anchor_init_v1.source run --gpus 2 3 6 7
python -m experiments.asrr_anchor_init_v1.bank --workers 4
python -m experiments.asrr_anchor_init_v1.transfer prepare development
python -m experiments.asrr_anchor_init_v1.transfer run development --gpus 0 1 2 3 4 5 6 7
python -m experiments.asrr_anchor_init_v1.transfer freeze
python -m experiments.asrr_anchor_init_v1.transfer prepare target
python -m experiments.asrr_anchor_init_v1.transfer run target --gpus 0 1 2 3 4 5 6 7
python -m experiments.asrr_anchor_init_v1.report
```

Итоговые метрики, индексаторы и записи обучения сохраняются в
`artifacts/asrr_anchor_init_2026-11-18`. Рабочие выборки и
промежуточные состояния находятся в `artifacts/local`.
