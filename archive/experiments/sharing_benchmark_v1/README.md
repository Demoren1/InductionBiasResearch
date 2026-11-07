# Sharing benchmark v1

Сравнение поиска общего дискретного шэринга с LC, LUT-Q, Yeh и MSR/MTSR. Основная задача — семейство линейных свёрток с независимо меняющимися коэффициентами. Перестановки входных и выходных координат скрывают геометрию; обучение не получает истинную маску или группы. Размер кодбука известен всем методам.

Результаты запуска 5 ноября 2026 года: [отчёт](results/20261105/convolution_competitors/report/REPORT.md), [таблица CSV](results/20261105/convolution_competitors/report/comparison.csv). Все строки представляют наши реализации и явно обозначенные адаптации, а не заимствованные числа из статей.

По уточнению пользователя отдельно сохранено [сравнение без кластеризации U](results/20261105/convolution_competitors/report/direct/REPORT.md): исходные непрерывные MSR/MTSR, soft/MAP Yeh, графики переноса, матрицы весов и необработанные столбцы U. Это срез тех же замороженных моделей; не новая репродукция авторских скриптов.

## Протокол

- `protocol.json` описывает cohort, число наблюдений, seed и правила выбора. `data.py` сохраняет observations отдельно от teacher. Target-задачи появляются только после полной фиксации моделей.
- Hidden: матрица 14×16, три числа плюс фиксированный ноль и отрицательные двойники, 64 source-задачи, шум 0.1. Два pilot-семейства для подбора; шесть новых семейств и три seed для проверки. Три повтора детерминированной линейной DP-версии не являются независимыми геометриями.
- Native: распределение rank1 из авторского MSR-генератора, размеры 70→68, 400 задач по 20 наблюдений без шума. Это воспроизведение распределения и размеров, а не побитное воспроизведение исходного эксперимента.
- `hard.py` содержит восемь source-предложений для каждого алгоритма банка с последующим выбором одного общего I по всем source-задачам. `soft.py` содержит MSR/MTSR и Yeh с таким же банком. `yeh_shared.py` отдельно обучает один общий категориальный P на всех задачах.
- Все checkpoint/hyperparameters выбираются на независимых coefficient-validation задачах. MSR/MTSR сохраняют вещественный U и оцениваются в своём пространстве. `decode.py` — дополнительный конкурент: кластеризация строк найденного U с отдельным validation-подбором.
- В основном сравнении все численные source-параметры отбрасываются. Новые числа fit-ятся точным МНК; так неправильная структура отделяется от проблем численного оптимизатора. `evaluate.py` проверяет точное совпадение I с допустимым переименованием/знаками кодов, маску, непрерывное подпространство, перенос и отрицательные контроли.
- Tiny-контроль полностью перебирает все канонические signed-I матрицы 2×4 с не более чем двумя группами. Его сертификат глобального оптимума не распространяется на большой benchmark.

## Воспроизведение

Запускать из корня репозитория, используя свободный GPU. Для нового запуска указать новый каталог результатов: `prepare` не перезаписывает существующий. Python использован из `/home/udeneev-av/myvenv/bin/python`.

```bash
python -m experiments.sharing_benchmark_v1.run RESULTS --stage prepare
python -m experiments.sharing_benchmark_v1.run RESULTS --stage pilot
python -m experiments.sharing_benchmark_v1.run RESULTS --stage freeze
python -m experiments.sharing_benchmark_v1.run RESULTS --stage confirmation
python -m experiments.sharing_benchmark_v1.run RESULTS --stage native
python -m experiments.sharing_benchmark_v1.yeh_shared RESULTS --stage pilot
python -m experiments.sharing_benchmark_v1.yeh_shared RESULTS --stage native
python -m experiments.sharing_benchmark_v1.yeh_shared RESULTS --stage confirmation
python -m experiments.sharing_benchmark_v1.decode RESULTS --stage pilot
python -m experiments.sharing_benchmark_v1.decode RESULTS --stage confirmation
python -m experiments.sharing_benchmark_v1.run RESULTS --stage finalize
python -m experiments.sharing_benchmark_v1.evaluate RESULTS
python -m experiments.sharing_benchmark_v1.evaluate RESULTS --tiny
python -m experiments.sharing_benchmark_v1.dense_adam RESULTS
python -m experiments.sharing_benchmark_v1.report RESULTS
python -m experiments.sharing_benchmark_v1.direct_report RESULTS
```

Подбор содержит 12 настроек каждого основного метода. Для общего Yeh после слабого pilot-результата до открытия teacher/test проверена отдельная расширенная сетка из 36 настроек, включая uniform initialization и охлаждение. Начальные попытки сохранены, причина изменения зарегистрирована в `shared_validation_expansion.json`. Параметры кластеризации U — шесть штрафов и восемь стартов; старт выбирается по source, штраф по pilot-validation. Подбор не продолжался по результатам test.

`all_models_frozen.json`, `*_shared_models_frozen.json` и `decoded_models_frozen.json` фиксируют состояния до аудита. `benchmark_frozen.json` объединяет их хеши. Для каждого фактического запуска сохраняется копия исполнявшихся модулей и их SHA256 в `executed_code/`. Тяжёлые trajectory/checkpoint-файлы не сохраняются. Полный аудит содержит raw I, допустимо выровненный I, basis, новые кодбуки и матрицы.

## Первоисточники

- [Yeh et al., 2022](https://proceedings.mlr.press/v151/yeh22b/yeh22b.pdf), [авторский код](https://github.com/raymondyeh07/equivariance_discovery).
- [MSR/MTSR, 2021](https://arxiv.org/abs/2007.02933), [авторский код](https://github.com/AllanYangZhou/metalearning-symmetries).
- [Learning-Compression, 2017](https://arxiv.org/abs/1707.04319).
- [LUT-Q](https://arxiv.org/abs/1911.04951).

В `references/` лежат неизменённые авторские файлы, использованные для сверки алгоритмов и генератора данных. Для этого эксперимента не создавались и не запускались software-тесты.
