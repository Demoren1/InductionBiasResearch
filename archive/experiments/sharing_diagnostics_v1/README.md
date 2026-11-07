# Диагностика поиска общего шэринга

Ретроспективная диагностика прежних ошибок и проверка исправлений на новых семействах скрытой линейной свёртки.

Результаты: [полный отчёт](results/20261105/diagnosis_and_fixes/report/REPORT.md), [таблица CSV](results/20261105/diagnosis_and_fixes/report/comparison.csv), [индексаторы](results/20261105/diagnosis_and_fixes/report/corrected_indexators.png), [кривые правок I](results/20261105/diagnosis_and_fixes/report/query_edit_curves.png).

Главный результат: правки дискретного I по средней source-query ошибке после нового fit отдельных C_t исправили прежний банк во всех 18 запусках шести новых геометрий. Ни маска, ни группы не фиксируются. Совместный векторный projector дал 17/18; на 64 функциях одноразовая проекция дала столько же. Плавное включение штрафа защищает oracle-инициализацию, но не решает слепой поиск само по себе.

## Разделение данных

- Старые семейства 201–206 используются только в `--stage diagnose`: все предложения на всех итерациях, проверка source/query отбора, oracle-инициализация и объектив проекции. Эти teacher-данные никогда не передаются новому поиску.
- Новый random seed, pilot-семейства 1001/1002 и независимые coefficient-validation задачи выбирают одну из 12 настроек для каждого основного варианта.
- Дополнительные причинные контроли регистрируются до confirmation и наследуют выбранную настройку общего поиска без нового HPO.
- Шесть новых семейств 1101–1106, по три seed; все 180 состояний фиксируются до открытия новых teacher/test. Затем по 100 новых задач с независимо меняющимися C, по 1/4/16 support и 256 query.
- Истины доступны только генератору, oracle-контролям и итоговому аудиту. Обучению известны размеры и три кода с фиксированным нулём/отрицательными двойниками; нет истинных позиций, числа активных связей, групп или перестановок.

## Воспроизведение

Из корня репозитория, на свободном GPU. Для нового запуска использовать новый каталог RESULTS.

```bash
python -m experiments.sharing_diagnostics_v1.run RESULTS --stage register
python -m experiments.sharing_diagnostics_v1.run RESULTS --stage diagnose
python -m experiments.sharing_diagnostics_v1.run RESULTS --stage pilot
python -m experiments.sharing_diagnostics_v1.run RESULTS --stage controls
python -m experiments.sharing_diagnostics_v1.run RESULTS --stage confirmation
python -m experiments.sharing_diagnostics_v1.run RESULTS --stage evaluate
python -m experiments.sharing_diagnostics_v1.report RESULTS
```

Исходный benchmark задаётся `--previous`, по умолчанию `experiments/sharing_benchmark_v1/results/20261105/convolution_competitors`. Интерпретатор фактического запуска — `/home/udeneev-av/myvenv/bin/python`. `diagnose` и `pilot` можно выполнять независимо на разных GPU.

`registered_code` хранит код фактической диагностики и pilot. После pilot добавлены три контроли в `controls_frozen.json`; фактические исходники confirmation/evaluation сохранены в `confirmation_code`. Дальнейшая добавленная CLI-команда `controls` воспроизводит ту же политику без изменения уже выбранных настроек. SHA256 результатов и кода записаны в метаданных, `models_frozen.json` создан до нового test.

## Ограничения

Прежний DP остаётся точной скалярной проекцией, без гарантии глобального решения задачи обучения. Общий векторный projector локален. Перебор однопозиционных правок использует быстрый точный inner-fit линейной модели; перенос такого поиска на нелинейный DeepSets требует отдельной проверки вычислительной стоимости и качества приближённого fit C. Результаты не объявляются новой алгоритмической идеей или глобальной гарантией. Детерминированные повторы bank/coordinate не являются новыми независимыми геометриями.

Прежние файлы benchmark и frozen-метод DeepSets не изменялись. Software-тесты не создавались и не запускались; выполнены необходимые научные проверки и Ruff для нового кода. Коммитов и публикации нет.
