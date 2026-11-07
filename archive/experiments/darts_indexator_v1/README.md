# DARTS-style поиск индексатора Pattern

## Фиксированный Pattern 0110: три индекса

Итог после абляции и подбора LR: **совместное обучение нашло точный I**.
Подтверждённый search seed9: 0/88 ошибок, 32 нужные связи, 16 назначений
каждому ненулевому коду, test и полный домен2048 — 100% BA. Пилотный
source-кандидат seed7 независимо дал тот же I. Из трёх новых search seed
точное восстановление получилось у одного; средняя BA32 свежих C79.50%,
query-выбранный C100%. [Краткое итоговое сравнение](results/single_0110_20261104/FINAL_COMPARISON.md).

Актуальная проверка по запросу пользователя — только `0110`, без переноса на
другие паттерны. Кодбук содержит фиксированный `C0=0` и два свободно обучаемых
значения `C1,C2`. Общий hidden bias, общий readout и выходной offset дают
вместе с кодбуком пять численных параметров. На каждой позиции матрицы 11×8
обучается softmax по трём индексам. После argmax фиксируется индексатор и
дообучаются уже полученные численные параметры.

Протокол — `single_protocol.json`; результаты и фигуры —
[single_0110_20261104/REPORT.md](results/single_0110_20261104/REPORT.md).
48 пилотных запусков сравнивают шаги обучения, обычный/охлаждаемый softmax
и отсутствие/наличие общего штрафа активных связей. 12 подтверждающих запусков
сравнивают first-order и один дифференцируемый виртуальный шаг на новых seed.
Во всех случаях поддерживаются четыре текущих кодбука; позиции окон и число
активных связей поиску не передаются. Имена двух ненулевых кодов изначально
согласованы случайной инициализацией разных знаков; ограничения знаков при
обучении нет.

Разбиение наблюдений: 768 support, 256 architecture-query, 512 selection,
512 test. Отбор по средней selection BCE после дискретизации и дообучения
четырёх кодбуков. Точное совпадение с истинным I проверяется только после
заморозки, с учётом глобальной перестановки двух кодов и скрытых нейронов.
Классификация и восстановление структуры оцениваются раздельно.

Исходный режим чередования 3:1: точного восстановления нет ни в одном из 300 сохранённых
снимков. Выбранный I: 12/88 категориальных ошибок, F1 маски 81.25%, test
balanced accuracy 84.67% после дообучения исходного C и при query-выборе
среди 32 свежих стартов. Истинный I при таком же обучении свежих чисел
достигает 100% (среднее стартов 80.14%). Найденная структура полностью
отключает второй вход, поэтому точное решение невозможно при любом кодбуке;
конкретная противоречащая пара и верхняя граница качества сохранены в отчёте.

По следующему запросу пользователя добавлены два плотных Adam-контроля:
91 параметр с теми же общими bias/readout и обычный MLP со 105 параметрами.
В них все численные веса обучаются совместно. В DARTS-поиске обновления
поочерёдные: три шага кодбука на support и один шаг логитов на query.
Контроли сравнивают восемь режимов LR/масштаба на восьми стартах, затем
проверяют по 32 свежих старта. Разбиение данных и бюджет дообучения те же.
Дополнение зарегистрировано после основного test; его настройки выбирает
только source selection. Хитмапы показывают реальные веса без квантования,
на общей шкале. [Adam-сравнение](results/single_0110_20261104/adam_baseline/REPORT.md).

Затем `update_ablation.py` сравнивает 3:1, 1:1, синхронные частные градиенты
с двумя loss и совместный support-loss. У всех парные старты и minibatch,
подбор по независимой selection после argmax и дообучения C. Настоящие
validation-кривые измерены до ablation test; query split-режимов не считается
независимой validation. [Парная абляция](results/single_0110_20261104/update_ablation/REPORT.md).

`joint_tuning.py` дополнительно сравнивает alpha LR0.003/0.01/0.03 при
numeric LR0.03 и sparsity0/.01. Высокий LR переиспользует уже обученные
состояния. Два source-selected режима подтверждаются на новых seed8/9/10.
Выбран alpha LR0.01, sparsity0.01, checkpoint4500 из6000; P и C обучаются
совместно только на support. После argmax меняются только C и общая голова.
[Подробности подбора](results/single_0110_20261104/joint_tuning/REPORT.md).

```bash
python -m experiments.darts_indexator_v1.single prepare
python -m experiments.darts_indexator_v1.search --output experiments/darts_indexator_v1/results/single_0110_20261104 --phase pilot --device cuda:0
python -m experiments.darts_indexator_v1.single collect --phase pilot
python -m experiments.darts_indexator_v1.single fine_tune --phase pilot --device cuda:0
python -m experiments.darts_indexator_v1.single select --phase pilot
python -m experiments.darts_indexator_v1.search --output experiments/darts_indexator_v1/results/single_0110_20261104 --phase confirmation --device cuda:0
python -m experiments.darts_indexator_v1.single collect --phase confirmation
python -m experiments.darts_indexator_v1.single fine_tune --phase confirmation --device cuda:0
python -m experiments.darts_indexator_v1.single select --phase confirmation
python -m experiments.darts_indexator_v1.single freeze
python -m experiments.darts_indexator_v1.single fresh_calibrate --device cuda:0
python -m experiments.darts_indexator_v1.single evaluate --device cuda:0
MPLCONFIGDIR=/tmp/ib-darts-mpl python -m experiments.darts_indexator_v1.single_report
python -m experiments.darts_indexator_v1.single_adam calibrate --device cuda:0
python -m experiments.darts_indexator_v1.single_adam evaluate --device cuda:0
MPLCONFIGDIR=/tmp/ib-darts-mpl python -m experiments.darts_indexator_v1.single_adam report
python -m experiments.darts_indexator_v1.update_ablation prepare
python -m experiments.darts_indexator_v1.update_ablation run --mode alternate3 --device cuda:0
python -m experiments.darts_indexator_v1.update_ablation run --mode alternate1 --device cuda:0
python -m experiments.darts_indexator_v1.update_ablation run --mode synchronous_split --device cuda:0
python -m experiments.darts_indexator_v1.update_ablation run --mode joint_support --device cuda:0
MPLCONFIGDIR=/tmp/ib-darts-mpl python -m experiments.darts_indexator_v1.update_ablation validate
python -m experiments.darts_indexator_v1.update_ablation freeze
python -m experiments.darts_indexator_v1.update_ablation evaluate --device cuda:0
MPLCONFIGDIR=/tmp/ib-darts-mpl python -m experiments.darts_indexator_v1.update_ablation report
python -m experiments.darts_indexator_v1.joint_tuning prepare
python -m experiments.darts_indexator_v1.joint_tuning pilot --device cuda:0
python -m experiments.darts_indexator_v1.joint_tuning confirmation --device cuda:0
python -m experiments.darts_indexator_v1.joint_tuning finalize --device cuda:0
MPLCONFIGDIR=/tmp/ib-darts-mpl python -m experiments.darts_indexator_v1.joint_tuning report
```

Search и fine_tune допускают разбиение по GPU `--shard`/`--shards`.
Существующий каталог запуска не перезаписывается; для повторного запуска
нужно указать новый output в копии протокола и передать его стадиям через
`--output`. Контроль с истинным I и аналитическими числами выполняется после
заморозки; test не выбирает настройки или индексаторы.

## Предыдущая многозадачная проверка: пять индексов

Каждый входной вес является смесью четырёх обучаемых значений кодбука и
фиксированного нуля. Один набор логитов 11×8×5 общий для десяти Pattern-задач;
каждая задача имеет четыре независимых численных старта. Численные параметры
продолжают обучаться на support между обновлениями логитов по query. В конце
назначения дискретизируются argmax и фиксируются для свежего обучения чисел.

Сохраняется исходная постановка `pattern/codebook`: 11 входов, 8 ReLU,
четыре кода, общее скрытое смещение, общий readout и выходное смещение —
7 численных параметров на задачу. Никакие истинные связи, окна, плотность,
степени узлов или ёмкости групп поиску не задаются. Разрешено только число
категорий. Истинный индексатор появляется после заморозки как отдельный oracle.

Основа — [DARTS, Liu et al., ICLR 2019](https://arxiv.org/pdf/1806.09055),
§2–3, и [авторский алгоритм обновления архитектуры](https://github.com/quark0/darts/blob/master/cnn/architect.py).
Это адаптация к скалярным группам равных весов, а не воспроизведение CNN NAS.
First-order держит числа фиксированными на архитектурном шаге. Дополнительный
контроль дифференцирует через один виртуальный Adam-шаг с текущими моментами;
это приближение конечной адаптации, не точный градиент полностью решённой задачи.

## Отбор без целевых меток

Протокол до запуска фиксирует 12 настроек: два шага обучения чисел, три шага
логитов и два режима softmax, по два search seed. Обычный softmax сравнивается
с постепенным снижением температуры до 0.1 и поздним штрафом энтропии 0.02.
Числа сначала проходят 512 warmup-шагов, затем три обновления на шаг логитов.

Из заранее заданных checkpoint получаются жёсткие индексаторы. Канонизация
перебирает переименования четырёх кодов и перестановки скрытых колонок без
учителя. На двух отдельных исходных функциях 1110/1111 каждый индексатор
обучается со свежими кодбуками. В пилоте проверяются шесть численных режимов
с четырьмя стартами; каждый старт выбирает checkpoint по query BCE. Средний
BCE на третьей, disjoint selection-выборке по всем стартам и обеим функциям
выбирает настройку и две поисковые конфигурации. Soft-loss и близость к oracle
не участвуют в отборе.

Две конфигурации подтверждаются на трёх новых search seed, с first-order и
дифференцируемым виртуальным шагом. Численная настройка уже фиксирована,
на индексатор восемь новых стартов. Конфигурация/order выбираются по среднему
качеству трёх seed после выбора checkpoint каждого seed; primary-индексатор
выбирается внутри этой группы по selection BCE. Все три сохраняются.

Функции 0010/0100/1011/1101 открываются после заморозки. Они исторически уже
изучались: новые разбиения и численные seed являются повторной проверкой,
не новым benchmark-семейством. Финальные методы получают по 32 свежих старта
и 4096 Adam-шагов; query выбирает checkpoint и старт, test не участвует в выборе.
Сравниваются shared и untied варианты, случайные контроли, плотные сети и oracle.

## Запуск

Из корня репозитория, `/home/udeneev-av/myvenv/bin/python`, PyTorch 2.6:

```bash
python -m experiments.darts_indexator_v1.run prepare
python -m experiments.darts_indexator_v1.search --phase pilot --device cuda:0
python -m experiments.darts_indexator_v1.run collect --phase pilot
python -m experiments.darts_indexator_v1.run calibrate --phase pilot --device cuda:0
python -m experiments.darts_indexator_v1.run select --phase pilot
python -m experiments.darts_indexator_v1.search --phase confirmation --device cuda:0
python -m experiments.darts_indexator_v1.run collect --phase confirmation
python -m experiments.darts_indexator_v1.run calibrate --phase confirmation --device cuda:0
python -m experiments.darts_indexator_v1.run select --phase confirmation
python -m experiments.darts_indexator_v1.run freeze
python -m experiments.darts_indexator_v1.evaluate controls_prepare
python -m experiments.darts_indexator_v1.evaluate control_fit --device cuda:0
python -m experiments.darts_indexator_v1.evaluate fine_tune --device cuda:0
python -m experiments.darts_indexator_v1.evaluate target_prepare
python -m experiments.darts_indexator_v1.evaluate target_fit --device cuda:0
python -m experiments.darts_indexator_v1.evaluate score
MPLCONFIGDIR=/tmp/ib-darts-mpl python -m experiments.darts_indexator_v1.report
```

Search и calibration допускают разбиение по GPU через `--shard`/`--shards`.
Сохраняются фактические loss/recall по шагам, мягкие и жёсткие состояния,
индивидуальные численные fits, протокол и происхождение выбора. Программные
тесты не создаются и не запускаются. Удалённые прежние опыты не восстанавливаются.

После уточнения пользователя явно добавлен этап `fine_tune`: сохраняются
кодбуки выбранного checkpoint и его raw argmax-индексатор, затем числа обучаются
ещё 4096 шагов с исходным search learning rate и сброшенными моментами Adam.
Его фактические кривые отделены от переноса с новыми кодбуками.

Из-за вариативности независимых search seed отдельно сохранён и оценивается
победитель пилота, выбранный source selection до подтверждающей серии.
`pilot_reference_frozen.json` регистрируется до создания target labels.
Основной confirmation-based выбор при этом не меняется. Он не подменяется
пилотным результатом после просмотра test-качества.

Итоги: [REPORT.md](results/pattern_20261104/REPORT.md), все численные сравнения —
`comparison.csv`, состояния выбранного I — `selected_indexator.json` и
`frozen_indexators.npz`. `code_snapshot` сохраняет исходный pilot core,
`analysis_snapshot` — окончательную реализацию и документы; дополнение этапов
оценки описано в `evaluation_protocol.json`.

## Численная оптимизация фиксированного Pattern 0110

После single-поиска и подбора joint LR отдельно исследовано обучение свежих
чисел при фиксированном истинном I. Скрытый сдвиг b уже обучается и общий
для восьми ReLU; выход имеет обучаемые коэффициент a и сдвиг d. Вместе
с C1/C2 это пять параметров. Диагностика сравнивает исходный BatchedAdam,
стандартный torch.optim.Adam, полный batch, выбор начального кодбука
C или -C и инициализацию имеющихся b/d. Выбор настроек идёт по selection,
checkpoint и ветки инициализации — по query. Новые seed проверяют
численную сходимость на известном паттерне.

`single.fit_fixed` принимает необязательный
`paired_initialization_steps=64/256` внутри specification. Обе ветки проходят
этот warmup на support, query BCE выбирает одну; её Adam moments сохраняются.
Суммарный бюджет включает отброшенную ветку: при 4096 и warmup64 выбранная
ветка получает 4032 обновления, отброшенная — 64. Значение 0 сохраняет
прежнее поведение. Исторический поиск P и дообучение исходных C не менялись.

Итоги, реальная динамика потерь и состояния:
[численная диагностика](results/single_0110_20261104/numeric_optimization_20261104/REPORT.md).
Для воспроизведения на сохранённом single/joint запуске:

```bash
python -m experiments.darts_indexator_v1.numeric_optimization prepare
python -m experiments.darts_indexator_v1.numeric_optimization reproduce --device cuda:0
python -m experiments.darts_indexator_v1.numeric_optimization calibrate --device cuda:0
python -m experiments.darts_indexator_v1.numeric_optimization confirm --device cuda:0
python -m experiments.darts_indexator_v1.numeric_optimization bias_calibrate --device cuda:0
python -m experiments.darts_indexator_v1.numeric_optimization bias_confirm --device cuda:0
MPLCONFIGDIR=/tmp/ib-darts-mpl python -m experiments.darts_indexator_v1.numeric_optimization report
```

`prepare` создаёт новую папку и не перезаписывает существующие результаты;
для отдельного повтора нужен другой `--output` с копией исходных single-данных
и `joint_tuning/evaluation_fits.npz`.

## Удвоение классов индексатора: C и -C

`signed_codebook.py` сравнивает ровно два варианта на том же Pattern 0110:
исходный `[0,c1,c2]` и предложенный пользователем `[0,c1,c2,-c1,-c2]`.
Оба имеют два обучаемых числа кодбука и три параметра головы. Отрицательные
значения связаны с исходными числами, не обучаются независимо. Во время поиска
эффективный вес равен `(P1-P3)*c1 + (P2-P4)*c2`; после argmax сохраняются
целочисленные I со значениями `0,1,2,-1,-2`. Знак I выбирает C или -C,
а физический знак веса зависит также от обученного C. Числа свободно меняют знак.

Используется совместное обучение на support с прежней сеткой alpha LR
0.003/0.01/0.03 и sparsity0/0.01. По три новых pilot и confirmation seed,
выбор по независимой selection после argmax и продолжения четырёх исходных C.
Дополнительно сохраняются свежие C и отдельный лучший индивидуальный кандидат,
выбранный по source до test. Сопоставление I с учителем выполняется только
после freeze и учитывает эквивалентные способы использовать знаковые пары.

```bash
python -m experiments.darts_indexator_v1.signed_codebook prepare
python -m experiments.darts_indexator_v1.signed_codebook oracle --device cuda:0
python -m experiments.darts_indexator_v1.signed_codebook pilot --device cuda:0
python -m experiments.darts_indexator_v1.signed_codebook confirmation --device cuda:0
python -m experiments.darts_indexator_v1.signed_codebook freeze
python -m experiments.darts_indexator_v1.signed_codebook evaluate --device cuda:0
MPLCONFIGDIR=/tmp/ib-darts-mpl python -m experiments.darts_indexator_v1.signed_codebook report
```

Результаты и хитмапы:
[знаковые пары](results/single_0110_20261104/signed_codebook_20261104/REPORT.md).
`prepare` требует новую папку результатов; существующие файлы не перезаписывает.

Следующий этап `signed_regularization.py` сравнивает тот же пятизначный кодбук
без нового штрафа, с `sum P(+c)*P(-c)` и с нормированной дисперсией численных
значений категориальной смеси. Второй штраф подавляет компенсацию и между
разными кодами; одинаковые численные значения не штрафуются. Его градиент
направлен только в P, масштаб нормирован на средний квадрат двух базовых C.
Числа учатся по прежнему support-loss. Это мягкие ограничения.

```bash
python -m experiments.darts_indexator_v1.signed_regularization prepare
python -m experiments.darts_indexator_v1.signed_regularization pilot --device cuda:0
python -m experiments.darts_indexator_v1.signed_regularization confirmation --device cuda:0
python -m experiments.darts_indexator_v1.signed_regularization freeze
python -m experiments.darts_indexator_v1.signed_regularization evaluate --device cuda:0
MPLCONFIGDIR=/tmp/ib-darts-mpl python -m experiments.darts_indexator_v1.signed_regularization report
```

Три коэффициента 0.01/0.1/1 выбираются по независимой selection на новых
pilot seed17/18/19, затем подтверждаются на20/21/22. Подробности и хитмапы —
[штрафы компенсации](results/single_0110_20261104/signed_codebook_20261104/regularization/REPORT.md).

Для строгой несовместимости внутри каждой пары реализована
`model.canonical_signed_probabilities`: из P(+c) и P(-c) вычитается их минимум,
а удвоенный минимум добавляется P0. Сумма вероятностей и мягкая W сохраняются
для любого C. Затем берётся argmax преобразованного P. Это отдельная процедура
дискретизации, исходные состояния поиска и исторические I не перезаписываются.

```bash
python -m experiments.darts_indexator_v1.signed_regularization project_original --device cuda:0
python -m experiments.darts_indexator_v1.signed_regularization project --device cuda:0
```

Первая команда проверяет исходный удвоенный кодбук, вторая — завершённую абляцию
штрафов. Выбранные configuration/seed/snapshot сохраняются; меняется только
дискретизация. [Исходный случай и хитмапа](results/single_0110_20261104/signed_codebook_20261104/regularization/projection_original/REPORT.md).
