# Motif Pair: перенос структуры на новые задачи

## Краткий итог

Эксперимент проверяет, можно ли по успешным MLP на известных задачах выучить
распределение полезных масок связности и перенести его на задачи с ранее не
встречавшимися парами мотивов.

На текущем разбиении `pair_disjoint_seed_42` получен положительный результат:

- gap-conditioned CVAE достигает accuracy `0.7480` против `0.7274` у случайной
  маски той же мощности, то есть даёт `+0.0206`;
- unconditional VAE достигает `0.7381`, поэтому даже без знания gap выученный
  структурный prior полезнее случайной связности;
- CVAE лучше VAE на `+0.0100` accuracy и `+0.058` structural IoU;
- правильный gap даёт более похожие на gold маски, чем намеренно неправильный;
- posterior collapse устранён подбором `beta`: обе модели выбрали `beta=0.1`,
  KL остаётся ненулевой, latent dimensions используются;
- CVAE пока не превосходит простой train-only conditional mean (`0.7497`);
- результат получен на одном split seed, поэтому это сильное предварительное
  свидетельство, но ещё не окончательная multi-seed проверка.

Основной вывод: VAE выучила общий структурный prior на круговые диагональные
полосы, а CVAE дополнительно выучила часть зависимости структуры от gap. Это
learned prior, а не встроенная в архитектуру свёрточная или equivariant
структура.

## 1. Исследовательский вопрос

Нас интересует не способность одной сети решить одну задачу. Мы проверяем
следующую цепочку:

1. Обучить много MLP с разными случайными масками на наборе meta-train задач.
2. Выделить MLP, которые лучше остальных решают свои задачи.
3. По их обученным весам построить continuous importance maps.
4. Обучить VAE/CVAE моделировать распределение этих карт.
5. Сгенерировать маски для новых задач, не используя их labels, validation
   loss или обученные на них importance maps.
6. Обучить новые MLP с этими масками и проверить downstream качество.

Положительный результат означает, что в successful networks присутствует
переносимая структура и генератор способен воспроизвести её для новой задачи.

## 2. Задача motif pair

Один объект — круговая последовательность длины 16 со значениями `-1/+1`.
Задача задаётся тройкой

```text
tau = (A, B, gap),
```

где:

- `A` и `B` — разные трёхбитовые мотивы;
- `gap` принимает значения от 3 до 10;
- positive example содержит уникальные вхождения `A` и `B`, причём начало
  `B` находится ровно через `gap` позиций после начала `A`;
- negative example содержит те же уникальные мотивы, но с другим допустимым
  расстоянием между ними.

Данные строятся фильтрацией полного пространства из `2^16` последовательностей.
Положительные примеры не создаются искусственной вставкой мотивов. Отдельный
shortcut audit показал accuracy `0.5033` для линейного классификатора по raw
bits и простым count features; максимум по test-задачам равен `0.5254`.
Следовательно, заметного простого линейного shortcut в данных не обнаружено.

### Идеальная маска

MLP имеет 16 входов и 16 hidden units. Первый слой задаётся бинарной матрицей
`16 x 16`.

Для hidden unit `h` идеальная маска открывает:

- три позиции `h, h+1, h+2`;
- три позиции `h+gap, h+gap+1, h+gap+2`;
- индексы считаются по модулю 16.

Каждый hidden unit получает шесть входов, поэтому gold mask содержит ровно
`16 * 6 = 96` активных рёбер из 256. В матричном виде это две круговые
диагональные полосы ширины 3, расстояние между которыми определяется gap.

Binary support зависит от gap, но не от значений мотивов `A` и `B`. Мотивы
влияют на знаки и величины обученных весов. Поэтому эксперимент проверяет
перенос общей gap-dependent структуры на новые комбинации мотивов.

## 3. OOD-разбиение

Используется pair-disjoint split с seed 42:

- 8 допустимых задач на каждый gap;
- 6 задач на gap попадают в meta-train;
- 2 задачи на gap остаются для test;
- всего 48 meta-train и 16 held-out задач;
- ни одна ordered pair `(A,B)` из test не встречается в meta-train ни при
  каком gap;
- все восемь gaps и все мотивы в обеих ролях представлены в meta-train.

Это compositional OOD по новым парам мотивов при уже известных структурных
режимах. Это не экстраполяция на невиданные значения gap.

Split и все зависящие от него артефакты связаны SHA256
`33e3a2c40aeee04b0573855782930be5485e718774eac594337352e5568cc7af`.
Путь к split, его hash и точный список train tasks проверяются на границах
численных артефактов: datasets, candidate shards, importance maps,
генеративные checkpoints и downstream evaluation. Артефакт от другого
разбиения на этих стадиях должен завершить pipeline ошибкой, а не быть молча
использован. Plotting проверяет наличие ожидаемых файлов текущего run root, но
не выполняет отдельную полную provenance-проверку их содержимого.

## 4. Как получены обучающие данные для VAE/CVAE

### 4.1. Candidate MLP

Для каждой из 48 meta-train задач обучается 512 независимых MLP:

- один hidden layer из 16 units;
- случайная Bernoulli mask на первом слое;
- вероятность активного ребра `96/256 = 0.375`;
- 1000 шагов Adam;
- batch size 128;
- learning rate `1e-3`.

Модели ранжируются по BCE на validation-наборе своей meta-train задачи.
Сохраняются лучшие 10%, то есть примерно 51 candidate на задачу.

### 4.2. Continuous importance maps

VAE и CVAE обучаются не на исходных бинарных масках candidate MLP. Для каждого
candidate вычисляется

```text
importance = abs(W1 * binary_mask).
```

Карта нормируется на собственный максимум и принимает значения в `[0,1]`.
Таким образом, target содержит информацию не только о наличии ребра, но и о
том, насколько сильно обученная сеть его использовала.

В генеративный dataset входят importance maps только от top-10% candidate MLP
с минимальным validation BCE. Test-задачи при построении этого dataset не
используются.

Hidden units перестановочно эквивалентны. Перед обучением генератора столбцы
importance maps канонизируются без использования gap, labels или held-out
tasks. Это удаляет случайный шум от нумерации hidden units, сохраняя значения
карты и её структуру.

## 5. Генеративные модели

Используются две модели с одинаковыми encoder/decoder backbone:

- размер входа: 256;
- latent dimension: 32;
- hidden dimension encoder/decoder: 256;
- reconstruction loss: Bernoulli BCE по continuous importance target;
- 80 эпох обучения;
- batch size 256;
- Adam с learning rate `1e-3`.

### Unconditional VAE

VAE не получает описание задачи. Она должна выучить marginal distribution по
всем meta-train importance maps и всем gaps. Поэтому от неё ожидается общий
prior на правдоподобную геометрию, но не точный выбор gap для новой задачи.

### Gap-conditioned CVAE

CVAE дополнительно получает one-hot condition длины 8, кодирующий gap. Мотивы
`A` и `B` в condition не входят, поскольку gold support от них не зависит.

CVAE проверяет, может ли генератор выучить семейство структур и выбирать его
член по известному gap.

VQ-VAE сейчас не используется. Top-96 уже превращает decoder scores в
дискретную маску фиксированной мощности, а codebook добавил бы новый механизм
collapse, не устраняя основную проблему conditioning.

## 6. Подбор beta и проверка posterior collapse

Проверялась сетка

```text
1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003.
```

Beta выбирается только по внутреннему разбиению meta-train importance maps.
OOD-задачи и downstream OOD-результаты при выборе beta не используются.

Правило выбора: взять наибольшую beta, для которой все эпохи в последних 10%
обучения проходят anti-collapse guard. Проверяются:

- validation KL не ниже 1;
- минимум две активные latent dimensions;
- конечность всех диагностик;
- reconstruction posterior не хуже режима `z=0`.

Обе модели выбрали `beta=0.1`.

| Модель | Validation KL | Active latent dimensions | Posterior vs `z=0` recon gap |
|---|---:|---:|---:|
| VAE | 2.232 | 13 | 2.162 |
| CVAE | 2.152 | 24 | 1.942 |

Для CVAE KL на последних восьми эпохах находится в диапазоне
`1.992..2.190`; все восемь эпох проходят guard. `beta=1` приводит к collapse,
а `beta=0.3` не проходит требование стабильности финального хвоста.

Следовательно, latent используется обеими выбранными моделями. Текущий
результат нельзя объяснить детерминированным decoder с вырожденным KL.

## 7. Протокол OOD-оценки

Для каждой из 16 held-out задач и каждого stochastic метода генерируется 64
маски. Decoder выдаёт 256 scores, после чего выбираются top-96 связей.

Для каждой маски создаётся новый MLP, его веса обучаются с нуля на данных
held-out задачи в течение 1000 шагов. Инициализации MLP и train batches
согласованы между методами, поэтому сравнения являются paired и отличаются
прежде всего маской.

В этом эксперименте отсутствуют:

- выбор маски по target validation loss;
- обучение на target importance maps;
- оптимизация latent `z` под held-out задачу;
- использование ideal mask при генерации или выборе checkpoint.

Сравниваются следующие методы:

- `random`: Bernoulli mask с ожидаемой плотностью 0.375;
- `random_exact96`: случайная маска ровно из 96 рёбер, основной null baseline;
- `vae`: top-96 sample unconditional VAE;
- `cvae`: top-96 sample CVAE с правильным gap;
- `cvae_wrong_gap`: тот же latent sample, но condition циклически сдвинут на
  неправильный gap;
- `conditional_mean`: top-96 из средней train importance map данного gap;
- `ideal`: gold support, диагностический ceiling.

### Structural IoU

Нумерация hidden units произвольна. Поэтому обычный IoU двух матриц вводил бы
в заблуждение. Используется Hungarian best-permutation IoU: столбцы
сгенерированной маски оптимально сопоставляются со столбцами gold mask, после
чего считается intersection over union.

Метрика удаляет только перестановку hidden units. Она не переставляет входные
позиции и не использует downstream labels.

## 8. Основной результат

| Метод | Accuracy | BCE | Hungarian IoU |
|---|---:|---:|---:|
| random | 0.7268 | 0.5641 | 0.436 |
| random exact-96 | 0.7274 | 0.5630 | 0.438 |
| unconditional VAE | 0.7381 | 0.5549 | 0.574 |
| CVAE, wrong gap | 0.7403 | 0.5537 | 0.580 |
| gap-conditioned CVAE | 0.7480 | 0.5434 | 0.632 |
| train-only conditional mean | **0.7497** | **0.5426** | 0.660 |
| ideal support | 0.7509 | 0.5452 | **1.000** |

### Paired сравнения по 16 held-out задачам

| Сравнение | Метрика | Средняя разница | 95% t-CI | p-value | Побед |
|---|---|---:|---:|---:|---:|
| CVAE - random exact-96 | accuracy | +0.02063 | `[+0.01089, +0.03036]` | 0.00041 | 15/16 |
| CVAE - random exact-96 | IoU | +0.19342 | `[+0.10555, +0.28129]` | 0.00029 | 14/16 |
| CVAE - VAE | accuracy | +0.00998 | `[+0.00145, +0.01850]` | 0.0248 | 9/16 |
| CVAE - VAE | IoU | +0.05813 | `[+0.01415, +0.10212]` | 0.0130 | 10/16 |
| CVAE - wrong gap | accuracy | +0.00772 | `[-0.00158, +0.01703]` | 0.0972 | 11/16 |
| CVAE - wrong gap | IoU | +0.05147 | `[+0.00489, +0.09805]` | 0.0326 | 11/16 |

Главный OOD gain равен `+0.0206` accuracy относительно случайной маски той же
мощности. Интервал целиком выше нуля, а эффект положителен на 15 из 16 задач.

CVAE также значимо лучше unconditional VAE. Следовательно, improvement нельзя
объяснить только тем, что обе модели выучили общий marginal prior: gap condition
даёт дополнительную информацию.

Correct-gap CVAE структурно лучше wrong-gap. Для accuracy направление то же,
но 95% интервал пока пересекает ноль. На одном split нельзя уверенно утверждать,
что именно correct condition стабильно улучшает downstream accuracy.

## 9. Насколько маски похожи на идеальные

У VAE, CVAE, conditional mean и ideal все маски содержат ровно 96 рёбер.
Поэтому структурная разница не связана с разной sparsity.

После оптимальной перестановки hidden columns среднее количество рёбер,
совпавших с ideal support, равно:

| Метод | Совпавшие ideal edges из 96 | Mean IoU |
|---|---:|---:|
| random exact-96 | 58.48 | 0.438 |
| VAE | 68.74 | 0.574 |
| CVAE, wrong gap | 69.69 | 0.580 |
| CVAE | 73.25 | 0.632 |
| conditional mean | 75.25 | 0.660 |
| ideal | 96.00 | 1.000 |

VAE восстанавливает примерно на десять ideal edges больше случайной маски.
Её преимущество над random exact-96 по IoU равно `+0.135`, 95% CI
`[+0.060, +0.210]`, `p=0.0016`; VAE лучше random на 12 из 16 задач.

Это убедительное свидетельство того, что VAE выучила нетривиальную геометрию,
а не только среднюю плотность матриц. Распределение численной IoU показывает,
что качество воспроизведения этой геометрии сильно зависит от gap.

### Структурный IoU по gap

| Gap | Random exact-96 | VAE | CVAE | Wrong-gap CVAE | Conditional mean |
|---:|---:|---:|---:|---:|---:|
| 3 | 0.430 | 0.411 | 0.583 | 0.447 | 0.613 |
| 4 | 0.444 | 0.394 | 0.507 | 0.414 | 0.512 |
| 5 | 0.447 | 0.473 | 0.430 | 0.470 | 0.422 |
| 6 | 0.448 | 0.575 | 0.518 | 0.607 | 0.587 |
| 7 | 0.440 | 0.755 | 0.844 | 0.704 | 0.864 |
| 8 | 0.410 | 0.648 | 0.734 | 0.738 | 0.745 |
| 9 | 0.437 | 0.755 | 0.871 | 0.726 | 0.939 |
| 10 | 0.451 | 0.581 | 0.567 | 0.538 | 0.600 |

Unconditional VAE особенно хорошо совпадает с gold на gaps 7 и 9 и плохо на
gaps 3 и 4. Она не получает gap, поэтому генерирует одну и ту же смесь
структур для любой новой задачи. Разница по gap означает, что эта смесь лучше
покрывает одни structural modes и хуже другие. Это общий prior, а не полное
восстановление правила `mask(gap)`.

CVAE улучшает средний IoU и заметно выигрывает на gaps 3, 4, 7, 8 и 9, но
conditioning остаётся несовершенным. На отдельных gaps VAE или wrong-gap
вариант оказываются не хуже.

### Симметрии gold mask

После разрешённой перестановки hidden columns некоторые gaps структурно
неразличимы:

- gold supports для gaps 6 и 10 эквивалентны;
- gold supports для gaps 7 и 9 эквивалентны.

Это следует из круговой геометрии и симметрии двух receptive fields. Поэтому
высокий IoU на gaps 7 и 9 не доказывает наличие двух независимо выученных
latent modes. При анализе gap recovery эти пары следует считать классами
эквивалентности.

### Почему IoU не обязан строго повторять accuracy

Ideal mask имеет IoU `1.0`, но её accuracy лишь на `0.0029` выше CVAE. Более
того, её BCE немного хуже CVAE и conditional mean. Это не ошибка: gold support
описывает известную достаточную структуру, но при конечном числе шагов обучения
не обязан быть единственным или оптимальным support для конкретной
оптимизации MLP.

Across-task корреляция также смешивает структурное качество с разной
сложностью gaps. В описательном post-hoc анализе 1024 VAE-масок, после
центрирования IoU и accuracy внутри каждой задачи, pooled Pearson correlation
равна `r = 0.323`; отдельная корреляция положительна на 13 из 16 задач. Для
этой величины не считался отдельный confidence interval, поэтому она не
является независимым статистическим подтверждением. Structural IoU полезен
как диагностика prior, но не заменяет downstream evaluation.

## 10. Какой inductive bias выучен

### Что подтверждено

Unconditional VAE предпочитает:

- локальные связи вместо равномерно случайных;
- круговые диагональные полосы;
- конфигурации, похожие на пару receptive fields;
- motif-invariant структуру, общую для meta-train задач.

CVAE дополнительно использует gap и частично выбирает подходящее расстояние
между полосами.

Downstream результат подтверждает функциональную полезность этого prior:
VAE лучше random, а CVAE лучше VAE и random на новых motif pairs.

### Что не подтверждено

Нельзя утверждать, что:

- VAE точно восстанавливает ideal mask;
- CVAE безошибочно реализует правило для каждого gap;
- генеративная модель лучше любого простого gap-conditioned baseline;
- результат переносится на unseen gaps;
- эффект устойчив по разным pair-disjoint split seeds;
- bias встроен в архитектуру модели.

VAE/CVAE состоят из fully connected layers и не имеют convolution, circular
equivariance или Toeplitz constraints. Поэтому корректная формулировка —
`learned structural prior`, а не `architectural inductive bias`.

## 11. Что удалось сделать

1. Построена задача, где полезная support-структура меняется между задачами.
2. Устранён pair leakage: test motif pairs глобально отсутствуют в meta-train.
3. Добавлена provenance-проверка split-specific численных артефактов от
   datasets до downstream evaluation.
4. Показано, что binary candidate masks являются слишком слабым target, и
   генератор переведён на continuous top-10% importance maps.
5. Hidden-column permutation noise устранён train-only canonicalization.
6. Posterior collapse обнаруживается и предотвращается train-only beta sweep.
7. VAE выучила полезный общий structural prior.
8. CVAE показала дополнительный gap-conditioned transfer.
9. Все тяжёлые нейросетевые стадии выполняются на GPU. CPU используется для
   файлов, JSON, статистики, Hungarian-диагностики и построения графиков.
10. Полный эксперимент воспроизводится одной командой и сохраняет графики всех
    стадий.

## 12. Ограничения текущего результата

### Один split seed

Paired интервалы используют 16 задач внутри одного split. Они учитывают
неоднородность test-задач, но не вариативность самого pair-disjoint разбиения.
Для paper-level вывода нужен запуск нескольких заранее выбранных split seeds с
неизменным протоколом.

### Conditional mean остаётся сильнее

Train-only conditional mean достигает `0.7497` accuracy и `0.660` IoU против
`0.7480` и `0.632` у CVAE. Разница accuracy мала и статистически незначима:
CVAE - conditional mean равно `-0.00170`, 95% CI
`[-0.00407, +0.00068]`, `p=0.148`.

Поэтому текущий результат подтверждает перенос выученного prior, но не
необходимость stochastic генератора по сравнению с простым prototype.

### Визуализация структуры

`plots/07_generator_masks.png` показывает для каждого gap один paired latent
sample VAE и CVAE, их версии после Hungarian alignment и ideal support в
крайнем правом столбце. На самих панелях приведены circular-Toeplitz score и
IoU, поэтому роль перестановки hidden units видна непосредственно.

`plots/07_generator_toeplitzness.png` отдельно показывает circular-Toeplitz
`R²` по 64 samples на gap относительно random exact-96, conditional mean и
ideal, а также среднюю occupancy каждой круговой диагонали. Эта диагностика
использует только уже обученные checkpoints и не участвует в выборе моделей.

`plots/02_data_examples.png` содержит реальные positive и hard-negative
примеры held-out задачи с отмеченными позициями мотивов и фактическим
расстоянием между ними.

## 13. Следующий эксперимент: оптимизация latent z

Следующий содержательный шаг — проверить, является ли frozen VAE decoder
полезным пространством поиска архитектуры под новую задачу.

Для каждой held-out задачи предлагается:

1. Заморозить unconditional VAE целиком.
2. Создать отдельные target-validation и target-test sets. Текущий
   `val_*.pt` нельзя одновременно использовать для оптимизации `z` и для
   финальной оценки.
3. Создать обучаемый latent `z` размерности 32.
4. Получить continuous mask
   `p(z) = sigmoid(frozen_decoder(z))`.
5. Обучать fresh MLP на target-train с первым слоем `W1 * p(z)`.
6. Вычислять BCE на target-validation split и обновлять только `z`.
7. После оптимизации взять top-96 значений `p(z)`.
8. С этой бинарной маской заново обучить несколько fresh MLP.
9. Один раз оценить их на untouched target-test split.

Hard top-96 нельзя применять внутри обычного gradient path: `topk/scatter`
недифференцируемы. Во время поиска используется soft mask, бинаризация делается
только перед финальным retraining и test.

Первый запуск должен повторять вариант из `pattern`: warm-up MLP с
`p(z).detach()`, затем несколько шагов с live soft mask и прямой gradient от
train/validation BCE в `z`. Обновления MLP через обычный optimizer не входят в
autograd graph. Поэтому это truncated/direct-path bilevel adaptation, а не
полный hypergradient через всю траекторию обучения MLP. Если такой вариант
даст устойчивый эффект, отдельной абляцией можно сделать functional unroll
последних inner steps.

Обязательные baselines:

- `vae_z0`: frozen VAE при `z=0`;
- `vae_best_of_K`: лучший из K prior samples по тому же target-validation;
- `vae_z`: обучаемый z, основной вариант;
- `cvae_z0` и `cvae_z` с фиксированным правильным gap;
- `free`: напрямую оптимизируемые 256 mask logits без decoder;
- `conditional_mean`, random exact-96 и ideal.

Основной научный вопрос: сможет ли `vae_z` выбрать подходящий structural mode и
догнать CVAE, не получая gap явно. Сравнение с `free` покажет, служит ли VAE
manifold полезным регуляризатором поиска.

Этот эксперимент использует labelled validation data новой задачи. Поэтому он
будет проверять target-time adaptation, а не текущий zero-shot OOD transfer, и
его результаты должны публиковаться отдельным блоком.

## 14. Воспроизведение

Из директории `motif_pair`:

```bash
GPU_IDS="0 1 2 3 4 5 6 7" SPLIT_SEED=42 ./run_all.sh
```

Основные параметры можно менять через environment variables:

- `N_MLPS_PER_TASK`, default 512;
- `TRAIN_STEPS`, default 1000;
- `TOP_FRAC`, default 0.1;
- `CVAE_EPOCHS`, default 80;
- `BETAS`, default полная beta grid;
- `EVAL_N_MASKS`, default 64;
- `EVAL_STEPS`, default 1000;
- `GPU_IDS` и `BETA_GPU_IDS`.

Тяжёлые production-стадии обучения и inference требуют CUDA и не переходят
молча на CPU. Лёгкая постобработка, статистика и часть plotting выполняются на
CPU.

## 15. Артефакты текущего запуска

- Split: `outputs/ood/pair_disjoint_seed_42/split.json`.
- Candidate MLP и importance maps:
  `outputs/ood/pair_disjoint_seed_42/checkpoints/`.
- Beta sweep и выбранные checkpoints:
  `outputs/ood/pair_disjoint_seed_42/generative/selection.json`.
- Полные результаты по каждой маске:
  `outputs/ood/pair_disjoint_seed_42/eval/eval_results.json`.
- Итоговые таблицы и paired statistics:
  `outputs/ood/pair_disjoint_seed_42/eval/summary.json`.
- Shortcut audit:
  `outputs/ood/pair_disjoint_seed_42/shortcut_audit.json`.
- PNG/PDF всех стадий:
  `outputs/ood/pair_disjoint_seed_42/plots/`.
- Manifest графиков:
  `outputs/ood/pair_disjoint_seed_42/plots/manifest.json`.

Manifest отмечает все восемь стадий визуализации как `generated`.

## 16. История исправлений

Первая версия держала out отдельные task triples, но большинство test pairs
встречалось в train с другим gap. Она не являлась корректным unseen-pair OOD.

Вторая версия использовала binary masks как generative target. Она немного
улучшала structural IoU, но почти не давала downstream gain: значения
активных весов, необходимые для различения полезных рёбер, были потеряны.

Текущая версия исправляет обе проблемы:

- split глобально pair-disjoint;
- target — continuous normalized importance maps лучших 10% candidate MLP;
- beta выбирается без обращения к OOD evaluation;
- KL collapse контролируется по стабильности финального участка обучения;
- evaluation не выполняет target selection или latent optimization.

Именно текущий запуск `outputs/ood/pair_disjoint_seed_42/` следует использовать
как основной результат эксперимента.
