# Signed DARTS: DeepSets и две синусоиды

Продолжение `darts_indexator_v1`, исходные результаты которого сохранены.
Основная серия — `results/20261104/`; итоговые таблицы и PNG/SVG строятся
в `results/20261104/REPORT.md` модулем `report`.

Каждый словарь содержит фиксированный ноль, m свободных чисел C и связанные
с ними −C. Перед дискретизацией взаимно гасящаяся масса каждой пары
переносится в нулевой класс; мягкие матрицы сохраняются. Затем индексаторы
фиксируются и численные параметры обучаются отдельно. Истинная структура
недоступна поиску, численные размерности и количество кодов заданы.

## Постановки

- Синтетический DeepSets: восемь двухмерных точек, два ReLU-фильтра на точку,
  скрытая перестановка входных координат. Поиск по полной матрице 16×16,
  четыре независимых кода и общие численные b/a/d. Истинный I используется
  только отдельным контролем и диагностикой после выбора.
- MNIST/Fashion-MNIST: изображения представлены 16 перемешанными точками
  с координатами и интенсивностью. Компактный эталон phi 3→8 ReLU,
  mean pooling, rho 8→32 ReLU→10. Поиск по общему MLP 48→128→8→32→10
  обучает связи и sharing первого слоя, bias и pooling; rho обучается
  численно. Это другой размер модели относительно старого 196-точечного
  пилота. Для реальных меток точного истинного индексатора нет.
- Синусоиды: 24 Gaussian-входа, неизвестное трёхгрупповое направление;
  слабая низкая и сильная высокая компоненты, частоты 4/6/8. Один sin-нейрон
  ученика имеет ненулевой floor. Случайные старты dense Adam/SGD/SGDM,
  длинный контроль и моментная инициализация рассматриваются отдельно.

## Запуск

Из корня репозитория, существующий `/home/udeneev-av/myvenv/bin/python`,
свободная GPU через `CUDA_VISIBLE_DEVICES`. Не занимать чужие GPU.
Выходные каталоги не перезаписываются. Программные тесты не создаются
и не запускаются; команды ниже выполняют исследовательские обучения.

```bash
python -m experiments.darts_structured_tasks_v1.run --task deepsets --stage prepare
python -m experiments.darts_structured_tasks_v1.run --task deepsets --stage controls
python -m experiments.darts_structured_tasks_v1.run --task deepsets --stage pilot
python -m experiments.darts_structured_tasks_v1.run --task deepsets --stage confirmation
python -m experiments.darts_structured_tasks_v1.run --task deepsets --stage fresh
python -m experiments.darts_structured_tasks_v1.run --task deepsets --stage freeze
python -m experiments.darts_structured_tasks_v1.evaluate experiments/darts_structured_tasks_v1/results/20261104/deepsets
```

`refine` регистрирует отдельное уточнение по результатам исходной валидации.
`mask_sharing` — отдельный exploratory опыт: заморозка найденной бинарной
маски, обучение независимых весов внутри неё, signed k-means и fixed-I refit.
Это дополнительный шаг извлечения групп, не результат одного argmax P.

```bash
python -m experiments.darts_structured_tasks_v1.refine experiments/darts_structured_tasks_v1/results/20261104/deepsets
python -m experiments.darts_structured_tasks_v1.mask_sharing experiments/darts_structured_tasks_v1/results/20261104/deepsets_refinement
```

Изображения: модуль `images`, `--dataset MNIST` или `FashionMNIST`, стадии
`prepare`, `controls`, `pilot`, `confirmation`, `freeze`; затем модуль
`image_evaluate --dataset DATASET`. Official test открывается после freeze.

Для синусоид те же стадии `run`, что для синтетики, с `--task sine`
и `--frequency 4`, `6` или `8`. При отборе каждого I используются новые C
с четырёх радиусов; query выбирает численный старт, selection — структуру.
Модуль `quantize_sine --register` отдельно регистрирует контроль кластеризации
мягких матриц; после получения исходной серии он запускается с путём
к `sine_4`, `sine_6` или `sine_8`.

`moment_codebook` отдельно проверяет численный оптимизатор на уже замороженных
синусоидальных I с инициализацией C по support-моменту. `sine_capacity` после
freeze вычисляет минимальный Gaussian-риск в подпространстве каждого I
и диагностирует представимость функций при смене истинных численных кодов.
Эти результаты следует отделять от эмпирического обучения переносу.

```bash
python -m experiments.darts_structured_tasks_v1.report
```

## Воспроизводимость и границы

Сохранены реальные исходники исполненных стадий, протоколы и критерии выбора,
кривые train/query, source-selected состояния и индексаторы, P до/после
переноса массы, независимые отложенные метрики и манифесты. Проверка C
со свежих стартов переинициализирует все свободные численные параметры,
включая rho у изображений. Подбор не использует итоговую оценку.

Стоимость поиска P, многократных refit, гиперпараметров и кластеризации
дополнительна к обучению окончательного C. Она не равна стоимости одного
плотного обучения. Значения, полученные на одной скрытой структуре и одной
численной функции, не являются статистикой переноса на семейство задач.

## Абляции поиска шэринга

`sharing_ablation` сравнивает постоянный и плавно вводимый нормированный
штраф дисперсии значений кодбука, sparsity, entropy и охлаждение. Поиск
маски и групп идёт с нуля; начальные состояния и minibatch парные между
вариантами. Стадии `prepare`, `pilot`, `confirmation`, `fresh`; затем
`sharing_report` замораживает численные состояния, создаёт новый held-out
и сохраняет таблицы и хитмапы. Teacher используется после выбора.

```bash
python -m experiments.darts_structured_tasks_v1.sharing_ablation --parent experiments/darts_structured_tasks_v1/results/20261104/deepsets_refinement --output experiments/darts_structured_tasks_v1/results/20261104/deepsets_sharing_ablation --stage prepare
```

`sharing_geometry` — отдельное сравнение на общей маске, ранее найденной
самим DARTS: свободные логиты, softmax расстояний до обучаемых кодов,
ближайший дискретный код с прямым переносом градиента на латентный вес.
Последний градиент является приближением. Проверяется также численный
warmup0/1000/3000, включённый в общий бюджет12000 шагов. Коды и назначения
переопределяются во время обучения; k-means не применяется. Это условный
поиск шэринга на выученной маске, его нельзя называть новым полным поиском
подключений с нуля. Стадии и аргументы те же, что у `sharing_ablation`.

```bash
python -m experiments.darts_structured_tasks_v1.sharing_report experiments/darts_structured_tasks_v1/results/20261104/deepsets_sharing_ablation
```

Для повторного построения рисунков без новой оценки используется
`sharing_report --report-only OUTPUT`. Обе серии сохраняются отдельно
от завершённых исходных экспериментов.

`sharing_full` отдельно проверяет ближайший код с STE на полной матрице
без готовой маски, с несколькими коэффициентами L1. Итог трёх серий:
`results/20261104/SHARING_ABLATIONS.md`. Модуль `sharing_summary` собирает
этот отчёт из завершённых результатов без повторных обучений.

## Жёсткое сэмплирование индексаторов

`hard_sampling` сравнивает мягкую смесь, hard argmax со straight-through,
hard Gumbel-Softmax с одним и четырьмя сэмплами, а также отдельный вариант
с переносом массы перед sampling. В hard forward на позиции выбирается
ровно один код; индексатор общий для всех примеров minibatch. Несколько
сэмплов используют один кодбук, а backward в логиты является смещённой
Gumbel-Softmax оценкой. Температура surrogate не меняет закон hard sampling.

Сравнение отдельно запускается на прежней найденной маске и на полной
матрице. Настройки выбираются по дискретному MAP индексатору после C-fit,
а не по мягкой или усреднённой stochastic ошибке. Перенос массы здесь
меняет распределение структур и исследуется как отдельная абляция.

```bash
python -m experiments.darts_structured_tasks_v1.hard_sampling --root experiments/darts_structured_tasks_v1/results/20261104/deepsets_hard_sampling --parent experiments/darts_structured_tasks_v1/results/20261104/deepsets_refinement --stage prepare
python -m experiments.darts_structured_tasks_v1.hard_sampling --root experiments/darts_structured_tasks_v1/results/20261104/deepsets_hard_sampling --stage pilot
python -m experiments.darts_structured_tasks_v1.hard_sampling --root experiments/darts_structured_tasks_v1/results/20261104/deepsets_hard_sampling --stage confirmation
python -m experiments.darts_structured_tasks_v1.hard_sampling --root experiments/darts_structured_tasks_v1/results/20261104/deepsets_hard_sampling --stage fresh
python -m experiments.darts_structured_tasks_v1.sampling_report experiments/darts_structured_tasks_v1/results/20261104/deepsets_hard_sampling --reference experiments/darts_structured_tasks_v1/results/20261104/deepsets_refinement
```

Регистрации, исходники выполненных запусков, вероятности, логиты, численные
кодбуки и метрики сохраняются внутри каждой области поиска. Отчёт
`sampling_report --report-only` перестраивает только таблицы и рисунки.

Результаты завершённого запуска: на общей найденной маске hard Gumbel с
четырьмя сэмплами восстановил I в2/3 новых seed, с одним — в1/3; hard
argmax и мягкие контроли — в0/3. Полный поиск не восстановил I ни в одной
ветке. Для выбранного точного I30/32 свежих C сошлись до NMSE<1e-6;
два старта остались в плохом решении. Подробные таблицы, выбранные I/C/W
и хитмапы: `results/20261104/deepsets_hard_sampling/REPORT.md`.

`sampling_reproduction` отдельно воспроизводит прежний успешный протокол:
четыре одновременно обучаемых кодбука,512 шагов численного warmup,
alpha lr0.1 и температуру исходных вероятностей1→0.03. Мягкий контроль
вызывает исходную `train.search`; hard argmax и Gumbel1/4 используют те
же старты и параметры. Это контроль на исходных seed43/44/45, а не новая
независимая статистика. В hard-ветках выбирается один код после warmup;
масса перед sampling не переносится. Стадии: `prepare`, `run`, `report`;
аргументы `--root` и `--parent` указывают новый каталог результатов и
исходный `deepsets_refinement` соответственно.

В контроле прежних настроек soft получил точную маску1/3, hard argmax
и Gumbel1/4 —0/3; точного полного I не было. Начальные P/C новых веток
совпадают точно. По отдельной диагностике сохранённой плохой маски22 из23
удалений, улучшающих query, ухудшают support. Отчёт и границы интерпретации:
`results/20261104/deepsets_sampling_reproduction/REPORT.md`.

## Разделение обновлений C и P при hard sampling

`sampling_alternating` сравнивает три режима с одинаковыми стартами и
бюджетом: совместные support-градиенты при прежнем C, поочерёдные
support/support и support/query при обновлённом C. Четыре hard Gumbel
структуры используют один кодбук. На численной стадии P отсоединяется,
на архитектурной стадии C и голова отсоединяются. Это first-order
alternating, без градиента через численную адаптацию.

На семью шесть пар скоростей C/P; pilot221/222 выбирает настройки,
confirmation223/224/225 проверяет их. Query в support/query участвует
в обучении P, поэтому независимым source-контролем служит selection.
Полная матрица и условный поиск внутри прежней маски сохраняются отдельно.
Действующие правила этого эксперимента записаны в `registration.json`;
общие правила в скопированном исходном `protocol.json` относятся к
родительскому протоколу и не переопределяют регистрацию новой абляции.

```bash
python -m experiments.darts_structured_tasks_v1.sampling_alternating --root experiments/darts_structured_tasks_v1/results/20261104/deepsets_sampling_alternating --parent experiments/darts_structured_tasks_v1/results/20261104/deepsets_refinement --stage prepare
python -m experiments.darts_structured_tasks_v1.sampling_alternating --root experiments/darts_structured_tasks_v1/results/20261104/deepsets_sampling_alternating --stage pilot
python -m experiments.darts_structured_tasks_v1.sampling_alternating --root experiments/darts_structured_tasks_v1/results/20261104/deepsets_sampling_alternating --stage confirmation
python -m experiments.darts_structured_tasks_v1.sampling_alternating --root experiments/darts_structured_tasks_v1/results/20261104/deepsets_sampling_alternating --stage fresh
python -m experiments.darts_structured_tasks_v1.alternating_report experiments/darts_structured_tasks_v1/results/20261104/deepsets_sampling_alternating --reference experiments/darts_structured_tasks_v1/results/20261104/deepsets_refinement
```

В завершённой серии средняя selection NMSE полного поиска составила
0.0947/0.0918/0.1173 для joint/support-support/support-query;
точных масок и I не было. На готовой маске —0.0272/0.0386/0.0598,
точного I0/3 во всех семьях. Совместный пилот дал точный I2/2,
но confirmation этот успех не повторил. У выбранного совместным
обучением I правильная маска и одна ошибка шэринга. Таблица,
свежие C и хитмапы с исходным Adam сохранены в
`results/20261104/deepsets_sampling_alternating/REPORT.md`.

## Повторяющиеся циклы P → argmax → C

`cyclic_search` проверяет полный поиск без готовой маски. На каждом
из200 циклов P получает4 support-обновления при фиксированном C,
затем C получает64 support-обновления при фиксированном P. На hard-C
ветке перед численной фазой заново вычисляется текущий argmax;
в следующем цикле P снова обучается. Логиты и моменты Adam сохраняются.

Сравниваются soft-P/soft-C, предложенный soft-P/hard-C и hard-P/hard-C
с softmax STE. Четыре численные реплики; pilot241/242 выбирает одну
из шести пар скоростей на семью, confirmation243/244/245 проверяет её.
Query/selection выбирают численные checkpoints и дискретную структуру
после C-fit. Учитель используется только после source/fresh-freeze.

```bash
python -m experiments.darts_structured_tasks_v1.cyclic_search --root experiments/darts_structured_tasks_v1/results/20261104/deepsets_cyclic_full --parent experiments/darts_structured_tasks_v1/results/20261104/deepsets_refinement --stage prepare
```

После `prepare` последовательно выполняются `--stage pilot`,
`--stage confirmation` и `--stage fresh` с тем же `--root`.
Модуль `cyclic_report ROOT --reference PARENT` создаёт итоговую
оценку, таблицу, выбранные I/C/W и хитмапы с исходным Adam.

В завершённом циклическом сравнении confirmation selection NMSE:
soft/soft0.2676, soft-P/hard-C0.1257, hard/hard0.1095;
точных масок и I0/3 во всех режимах. Предложенная схема уменьшила
разрыв между мягкой и argmax моделью, однако восстановление полного
индексатора с нуля этим расписанием не решено. Итог:
`results/20261104/deepsets_cyclic_full/REPORT.md`.

## Полный индексатор через проекционное согласование

Новая серия исследует независимую непрерывную матрицу Z и дискретную
Q=C[I]. Варианты с hard nearest + STE сохранены как отрицательный контроль
в `deepsets_projection_full`: точного I ни в одном из трёх запусков нет.

`consensus_search` обучает Z по задаче и штрафу расхождения с Q. Каждые
50 шагов заново вычисляются I/C для всех позиций. Сравниваются обычный
квадратичный штраф и ADMM-подобные dual-обновления. Вариант `--projection
dynamic` использует точную одномерную проекцию: одновременно выбирает
нулевые позиции и до четырёх групп. Маска и число активных связей не заданы.

Локальный Lloyd оставлял пустой код: точных I — 0/3.
С точной проекцией оба ADMM-варианта восстановили I в 3/3,
а квадратичный штраф без dual — в 2/3. Для выбранных точных I все 32 свежих
C на исходной функции сошлись почти до нуля. Итоговый алгоритм использует
проекционное согласование; общие механизмы ADMM и одномерного DP известны.

Для повторения нужен новый каталог результатов. `consensus_search`
имеет стадии `prepare`, `pilot`, `confirmation`, `fresh`; при `prepare`
указываются `--parent` и `--projection dynamic`. `consensus_report ROOT
--reference PARENT` сохраняет оценку, I/C/W, хитмапы и кривые.

`structure_transfer` отдельно проверяет фиксированные I на новых истинных
C: стадии `register`, `prepare`, `run`. Во всех target-попытках численные
параметры инициализируются заново. Когорты `transfer` и
`transfer_confirmation` разделены: первая даёт парное сравнение на прежних
шести функциях, вторая использует ещё шесть ранее не оценённых функций
с зарегистрированным `--seed-offset 11100000`. Исходные индексаторы
фиксируются до генерации новых функций. Идентичный dense-контроль первой когорты переиспользуется только
после проверки данных, протокола и хеша численных состояний.

`transfer_report ROOT --cohort transfer_confirmation` строит графики
и таблицы без повторного обучения. `projection_summary` собирает всю
серию и сравнение одинаковых настроек из сохранённых данных.

Итог с полными таблицами, переносом, хитмапами и train/query кривыми:
[projection_research/REPORT.md](results/20261104/projection_research/REPORT.md).
Формулы и границы гарантий: [PROJECTION_METHOD.md](PROJECTION_METHOD.md).
Результат пока подтверждён на синтетическом DeepSets; новые запуски
успешного варианта на Pattern, синусах и изображениях не выполнялись.

## Циклическое дообучение C и сброс Z

Успешный вариант без перезаписи Z сохранён отдельно в
[frozen_methods/deepsets_consensus_exact_v1](frozen_methods/deepsets_consensus_exact_v1/README.md):
фактически исполненные исходники, протокол, выбранные I/C/W и SHA256
исходных результатов и архивной копии. Новый поиск не загружает эти
структуры или численные параметры для инициализации.

`reset_search` проверяет предложенный цикл: обучение Z → точная проекция
в I/C → task-обучение C при фиксированном I → присваивание Z=C[I].
Контроли сохраняют Z, исключают task-C-fit или отключают штраф
согласования. При копировании проверяются сохранение и очистка моментов
Adam у Z. Гиперпараметры выбираются по source-валидации.

```bash
python -m experiments.darts_structured_tasks_v1.reset_search --root NEW_ROOT --parent experiments/darts_structured_tasks_v1/results/20261104/deepsets_consensus_exact --frozen-method experiments/darts_structured_tasks_v1/frozen_methods/deepsets_consensus_exact_v1 --stage prepare
```

При подготовке `--long-cycles` включает блоки Z/C по 800/200, 500/500
или 200/800 шагов вместо 40/10, 25/25 или 10/40. В обоих случаях общий
бюджет — 2000 warmup и 10000 дальнейших градиентных проходов;
контроль исходного метода сохраняет проекцию каждые 50 шагов.
Затем выполняются стадии `pilot`, `confirmation`, `fresh` и
`reset_report ROOT --reference results/20261104/deepsets_refinement`.

`reset_summary --stage freeze` фиксирует выбор между обеими длинами
блока по source-selection до генерации новых transfer-функций.
После этого `structure_transfer` и `transfer_report` оценивают
фиксированные I с новыми кодбуками; `reset_summary --stage report`
собирает итог из сохранённых результатов.

Общий отчёт, таблицы, индексаторы и сравнение с плотным Adam:
[reset_research/REPORT.md](results/20261104/reset_research/REPORT.md).

## Проекционное согласование на синусоидах

`sine_consensus` переносит тот же поиск с сохранением Z и точной DP-проекцией
на исходные задачи f=4/6/8. Цель — сумма двух sin, ученик содержит одну sin
с обучаемой амплитудой; лучший population MSE около 0.05, плохой локальный
минимум около 0.5. Все 24 позиции доступны, три свободных кода и связанные
отрицательные двойники. Общие `consensus_search`, DP, модель и обучение
кодбуков не изменяются.

```bash
python -m experiments.darts_structured_tasks_v1.sine_consensus NEW_ROOT --stage prepare --parent experiments/darts_structured_tasks_v1/results/20261104
python -m experiments.darts_structured_tasks_v1.sine_consensus_transfer NEW_ROOT --stage register
```

Затем последовательно выполняются стадии `pilot`, `confirmation`, `numerics`
модуля `sine_consensus`. Пилот выбирает из 32 настроек, подтверждение проверяет
его победителя и исходные параметры DeepSets на трёх новых seed. Каждый
снимок I получает одинаковый refit кодбука; затем идут 32 свежих старта.
Архивные calibration состояния контролей на тех же данных переиспользуются
после проверки SHA256, новые 32-start сравнения обучаются заново.

После фиксации всех source-I модуль `sine_consensus_transfer` со стадиями
`prepare`, `run` проверяет четыре новые коэффициентные функции с f=6.
Все source-I сравниваются на одинаковых задачах с полностью свежими числами;
есть oracle, shuffled sharing, dense random/moment контроли. Аналитические
capacities по 32 направлениям показаны отдельно от обученного переноса.

`sine_consensus_report ROOT --stage source` открывает новый source-test и
точный Gaussian-риск после numerical freeze. После переноса выполняются
стадии `transfer`, `report`. Сохраняются все исходные/выровненные I/C/W,
train/query, ландшафт риска и хитмапы с плотным Adam.

Итог: [sine_consensus/REPORT.md](results/20261105/sine_consensus/REPORT.md).

### Перенос частоты при фиксированном I с f=6

`sine_frequency_transfer` сохраняет направление исходной функции и меняет
только частоту на f=4/6/8. Единственный I берётся из source-validation при
f=6; кодбуки и амплитуды на каждой частоте обучаются с нуля. f=6 служит
контролем повторной численной подгонки. Это другой перенос, чем изменение
трёх коэффициентов направления в `sine_consensus_transfer`.

```bash
python -m experiments.darts_structured_tasks_v1.sine_frequency_transfer NEW_ROOT --parent experiments/darts_structured_tasks_v1/results/20261105/sine_consensus --stage prepare
python -m experiments.darts_structured_tasks_v1.sine_frequency_transfer NEW_ROOT --stage run
python -m experiments.darts_structured_tasks_v1.sine_frequency_transfer NEW_ROOT --stage report
```

12 настроек радиуса/Adam lr получают по 32 calibration-старта; query выбирает
чекпойнты и один старт на настройку, selection — настройку. Затем обучаются
32 новых старта. I, протокол и численный выбор сохраняются с SHA256 до
teacher-аудита и нового test. Среднее всех стартов и число близких к
глобальному минимуму результатов показываются вместе с выбранной моделью.

Отчёт и хитмапы:
[sine_frequency_transfer/REPORT.md](results/20261105/sine_frequency_transfer/REPORT.md).
