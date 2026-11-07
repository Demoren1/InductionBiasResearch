# DARTS-style поиск индексатора Pattern

## Фиксированный Pattern 0110: три индекса

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

Итог этого запуска: точного восстановления нет ни в одном из 300 сохранённых
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
