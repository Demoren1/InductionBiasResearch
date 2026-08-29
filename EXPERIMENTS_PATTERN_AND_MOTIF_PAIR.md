# Эксперименты `pattern` и `motif_pair`

Дата среза: 2026-08-29.

Этот документ — самодостаточное техническое описание двух последовательных
экспериментов по извлечению структурного inductive bias из успешно обученных
нейросетей. Он предназначен в том числе для передачи языковой модели как
контекст для дальнейшего обсуждения проекта.

Главная идея проекта: обучить много обычных MLP с разными случайными масками
связности, выбрать сети, лучше решившие исходные задачи, извлечь из их весов
структурный сигнал и обучить VAE/CVAE генерировать новые маски. Полезность
выученного prior проверяется не качеством реконструкции, а качеством **новых
MLP, обученных с нуля с этими масками на held-out задачах**.

Важно: корневой [`README.md`](README.md) описывает более ранний соседний
эксперимент MA(k,s). Он концептуально связан с этой программой исследований,
но не является реализацией `pattern` или `motif_pair`. Общая математическая
формализация находится в [`paper/preliminary.tex`](paper/preliminary.tex).

## 1. Исследовательский вопрос

Пусть задача `tau` задаёт распределение данных и функцию потерь. Архитектура
базовой модели имеет обучаемые веса `W` и дискретную структуру `M`, в данных
экспериментах — бинарную маску первого слоя. Для каждой meta-train задачи
семплируется банк кандидатов:

```text
M_i ~ p0(M),
W_i* = train(M_i, D_train_tau),
score_i = loss(M_i, W_i*, D_val_tau).
```

Из кандидатов с наименьшим validation loss строится dataset представлений
успешных решений. Представление — не сама бинарная маска, а нормированная
карта величин использованных весов:

```text
I_i = abs(W1_i * M_i) / max(abs(W1_i * M_i)).
```

На картах `I_i` обучается генеративная модель. Decoder выдаёт непрерывные
scores, а оператор `top-K` превращает их в бинарную маску фиксированной
мощности:

```text
z ~ N(0, I),
scores = decoder(z, condition),
M_generated = top_K(sigmoid(scores)).
```

После этого для каждой сгенерированной маски создаётся новый MLP. Его веса
обучаются на данных целевой задачи с нуля. Если такие MLP систематически лучше
MLP с масками из исходного случайного распределения `p0`, генератор выучил
полезный structural prior.

### 1.1. Что здесь называется structural prior

Structural prior — распределение по допустимым связям первого слоя. Он задаёт,
какие входные координаты могут поступать в каждый hidden unit, но не переносит
обученные веса базовых MLP.

В обоих экспериментах downstream-модель остаётся fully connected MLP без
свёртки, weight sharing, Toeplitz constraint или circular equivariance.
Поэтому корректно говорить о **выученном структурном prior**, а не о bias,
жёстко встроенном в архитектуру.

### 1.2. Четыре объекта, которые нельзя смешивать

1. `M_candidate` — случайная бинарная маска, с которой обучался candidate MLP.
2. `I = |W1 * M_candidate| / max(...)` — непрерывная importance-карта
   успешно обученного candidate.
3. `decoder(z, c)` — непрерывные logits/scores генератора.
4. `M_generated = top-K(scores)` — бинарная маска, с которой обучается новый
   downstream MLP.

VAE/CVAE обучается на объекте 2, а функциональное качество измеряется на
объекте 4. Хорошая reconstruction loss сама по себе не является конечным
результатом эксперимента.

## 2. Общий экспериментальный конвейер

Оба эксперимента используют одну методологическую схему.

### Шаг 1. Построение синтетических задач

Для каждой task identity определён on-the-fly train stream и фиксированные или
seeded evaluation samples. Основной pipeline не предполагает универсальный
персистентный three-way train/validation/test split: candidate selection и
финальная downstream evaluation используют раздельно сгенерированные samples.
Отдельное разделение target validation/test применяется только в явно
описанном z-optimization protocol. Синтетическая среда имеет известную
достаточную gold support, поэтому можно отдельно сравнивать функциональное
качество и структурное сходство.

### Шаг 2. Обучение банка masked MLP

Для каждой meta-train задачи обучается много MLP. Маска первого слоя
семплируется один раз и остаётся фиксированной; веса внутри разрешённых связей
обучаются Adam по BCE. Кандидаты ранжируются по validation BCE.

### Шаг 3. Отбор успешных решений и importance maps

Берутся top-10% кандидатов с наименьшим validation BCE. Их masked first-layer
weights превращаются в нормированные unsigned importance maps. В отличие от
binary connectivity, такая карта сохраняет информацию о том, какие из
разрешённых связей реально получили большие веса.

### Шаг 4. Обучение генератора

VAE или CVAE моделирует распределение continuous importance maps. В
`pattern` генератор безусловный, потому что gold support одинакова для всех
patterns. В `motif_pair` CVAE получает gap, потому что gold support зависит от
gap; unconditional VAE служит ablation.

### Шаг 5. Zero-shot downstream evaluation

Генератор строит маски для meta-test задач, не используя их importance maps,
validation loss или labels для выбора маски. Для каждой маски обучается свежий
MLP. Основное сравнение — со случайной маской **той же точной мощности**.

Это отделяет перенос структуры от переноса весов и от target-time architecture
search.

## 3. Эксперимент `pattern`

Основные источники: [`pattern/README.md`](pattern/README.md),
[`pattern/RESULTS.md`](pattern/RESULTS.md),
[`pattern/OOD_RESULTS.md`](pattern/OOD_RESULTS.md) и
[`pattern/THEORETICAL_STATE.md`](pattern/THEORETICAL_STATE.md).

### 3.1. Цель

Проверить, можно ли из множества успешно обученных sparse MLP извлечь общий
prior на локальные окна и перенести его между задачами распознавания разных
4-битных patterns.

Ключевая особенность среды: знаки полезных matched-filter weights зависят от
pattern, но геометрия first-layer support одинакова для всех 16 задач. Поэтому
генератору не требуется знать identity pattern.

### 3.2. Задача и данные

- Вход: `x in {-1, +1}^8`.
- Task identity: одна из 16 строк `0000` ... `1111` длины 4.
- Label: `y=1`, если pattern встречается в `x` как непрерывная
  подпоследовательность хотя бы в одном месте.
- В последовательности длины 8 есть пять окон длины 4: starts `0 ... 4`.
- Validation set содержит 2048 примеров на задачу, целевая доля positive — 0.5.

Генератор данных сначала семплирует случайные последовательности, а затем для
части отрицательных примеров вставляет pattern в случайное окно, чтобы получить
нужную долю positive. Поэтому данные нельзя описывать как безусловно равномерную
выборку из всех `2^8` последовательностей. Train batches создаются заново во
время обучения; validation data фиксированы seed.

Реализация: [`pattern/data/generate.py`](pattern/data/generate.py), параметры:
[`pattern/config.py`](pattern/config.py).

### 3.3. Gold solution

Для каждого возможного start `w` достаточно matched filter длины 4. Его taps
равны pattern в `-1/+1` кодировке, а bias отделяет полное совпадение от
несовпадения.

Gold binary support имеет форму `8 x 8`. Для hidden column `h`:

```text
w = h mod 5
M_gold[w : w + 4, h] = 1.
```

Hidden width равна 8, поэтому окна 0, 1 и 2 представлены дважды, а окна 3 и 4
по одному разу. Каждая колонка содержит четыре связи, всего `8 * 4 = 32`
активных рёбер из 64.

Эта support pattern-independent. Конкретный pattern определяет значения и
знаки weights, но не то, какие input positions должны быть доступны hidden
units.

### 3.4. Candidate MLP

Базовая модель — однослойный masked MLP:

```text
logits(x) = W2^T ReLU((W1 * M)^T x + b1) + b2.
```

Маска умножается только на `W1` и фиксирована во время обучения. Default
configuration на одну pattern-задачу:

| Параметр | Значение |
|---|---:|
| input dimension | 8 |
| hidden units | 8 |
| candidate MLP | 2000 |
| proposal mask | Bernoulli(0.5) |
| активных рёбер в среднем | 32 из 64 |
| Adam steps | 2000 |
| batch size | 128 |
| learning rate | `1e-3` |
| selection | top-10% по validation BCE |

Выбранные task-specific masks сохраняются как `best10pct.pt`. Они полезны как
oracle/reference в in-distribution анализе, но запрещены в held-out task OOD:
иначе маска была бы выбрана с использованием качества на целевой задаче.

### 3.5. Importance maps

Для candidate строится:

```text
importance = abs(W1 * M)
importance /= importance.max()
```

Карта unsigned и лежит в `[0,1]`. В актуальном default pipeline генератор
читает `importance.pt` и оставляет top-10% карт по сохранённому validation BCE.
Используются raw, то есть не выровненные по перестановкам hidden units, maps.

### 3.6. Генератор

Класс в коде называется `CVAE`, но `CVAE_COND_DIM=0`; фактически это
unconditional VAE.

| Параметр | Значение |
|---|---:|
| input / mask dimension | 64 |
| latent dimension | 32 |
| encoder/decoder hidden dimension | 256 |
| epochs | 80 |
| batch size | 128 |
| reconstruction | BCE-with-logits, sum по 64 dimensions |
| KL coefficient | `beta=0.1` |
| targets | raw continuous top-10% importance maps |

При генерации для каждого sample независимо берётся `z ~ N(0,I)`, decoder
возвращает 64 scores, после чего global top-32 создаёт маску ровно с 32
связями.

Независимый `z` для каждого sample — существенная часть текущего протокола.
Старые результаты, полученные до исправления повторного использования одного
latent для нескольких masks, нельзя напрямую смешивать с текущими.

### 3.7. Baselines и метрики

Используются:

- `random`: Bernoulli(0.5) mask;
- `random_exact32`: равномерная случайная маска ровно с 32 связями;
- `mean_imp`: top-32 среднего meta-train importance;
- `det_reg`: deterministic regressor importance map;
- `cvae`: top-32 sample unconditional VAE;
- `top10%`: реальная mask успешного candidate — task oracle;
- `ideal`: exact Toeplitz support — synthetic oracle.

Основная functional metric — accuracy и BCE свежих MLP после одинакового
training budget. Structural diagnostic — IoU с gold после Hungarian matching
hidden columns. Перестановка используется только для диагностики, а не при
downstream training.

### 3.8. Основной in-distribution результат

Текущий согласованный main run: raw top-10% importance, BCE-sum, `beta=0.1`,
64 независимо сгенерированные top-32 masks на задачу, evaluation seed 42.

| Источник маски | Mean accuracy | SD по 16 task means | Mean BCE |
|---|---:|---:|---:|
| mean importance | 0.7876 | 0.0314 | 0.3991 |
| deterministic regressor | 0.7875 | 0.0312 | 0.3994 |
| random Bernoulli(0.5) | 0.8877 | 0.0504 | 0.2848 |
| **VAE, raw top-10%** | **0.9181** | 0.0402 | 0.2351 |
| selected top-10% candidate masks | 0.9246 | 0.0390 | 0.2219 |
| exact ideal Toeplitz | 0.9302 | 0.0310 | 0.2090 |

VAE выигрывает у Bernoulli random около 3.04 percentage points, отстаёт от
task-specific selected oracle на 0.65 п.п. и от exact ideal на 1.21 п.п.

Все 64 VAE masks в этом запуске различны. Mean best-permutation IoU с gold
равен 0.6442, range 0.5238–0.7778. Следовательно, VAE выучила полезный, но не
точный Toeplitz prior.

Этот result — single generator/evaluation configuration. Он не даёт полной
оценки variability по candidate training seeds, VAE seeds и downstream seeds.

### 3.9. Held-out task OOD

Более строгий protocol разделяет 16 patterns на:

- 12 meta-train tasks;
- 4 held-out tasks.

VAE, `mean_imp` и `det_reg` видят importance maps только 12 train patterns.
Затем на четырёх held-out patterns обучаются fresh downstream MLP. Target
top-10% oracle и target-time latent optimization исключены. Primary null —
`random_exact32`, совпадающий с VAE по cardinality.

Для split seeds 42–46 получены пять split-level macro means:

| Split seed | VAE | Random exact-32 | Gain |
|---:|---:|---:|---:|
| 42 | 0.9155 | 0.8912 | +0.0243 |
| 43 | 0.9078 | 0.8814 | +0.0264 |
| 44 | 0.9159 | 0.8955 | +0.0203 |
| 45 | 0.9121 | 0.8848 | +0.0273 |
| 46 | 0.8919 | 0.8575 | +0.0344 |

Агрегат:

| Метод | Mean accuracy | Population SD по splits | Mean BCE |
|---|---:|---:|---:|
| random exact-32 | 0.8821 | 0.0132 | 0.2977 |
| **VAE, train-only** | **0.9086** | 0.0088 | **0.2539** |
| ideal Toeplitz | 0.9240 | 0.0098 | 0.2226 |

Mean gain VAE над exact-32 random равен `+0.0266`; SD gain по пяти splits —
`0.0046`; направление положительно во всех пяти splits.

Знак `±` здесь — описательный population SD по пяти split-level means, а не
confidence interval для внешнего распределения задач. Пять splits используют
разные разбиения одних и тех же 16 возможных tasks.

Корректный вывод: VAE переносит общий pattern-independent local-window prior
на held-out identities **внутри специально построенного синтетического
семейства**. Это не перенос на новую геометрию или другой task family.

### 3.10. Alignment experiments

Hidden units перестановочно эквивалентны, поэтому raw importance maps содержат
permutation nuisance. В проекте есть отдельные методы alignment:

- `gold`: Hungarian alignment к известной ideal mask; oracle diagnostic;
- `window`: alignment к известному семейству пяти local windows;
- `refmatch`: alignment к выбранной реальной reference map;
- `selfalign`: data-only iterative matching успешных maps.

Они **не входят** в default pipeline. Исторически некоторые aligned variants
давали accuracy около 0.93, однако часть этих чисел получена до исправления
independent latent sampling. Их нужно воспроизвести заново, прежде чем
сравнивать с актуальным raw result 0.9181.

Кроме того, `window` использует сильное знание о допустимом семействе окон, а
`gold` прямо использует ответ. Это не равноправные structure-agnostic baselines.

### 3.11. Bilevel latent optimization

В `pattern` есть отдельная target-adaptive ветка: decoder замораживается, а
latent `z` оптимизируется по labelled validation data конкретной задачи.
Сравниваются:

- `z`: оптимизация 32-dimensional latent через frozen decoder;
- `z0`: decoder prior mode при `z=0`;
- `free`: прямая оптимизация 64 mask logits без decoder manifold.

После оптимизации создаётся одна top-32 mask на задачу и оценивается несколькими
новыми MLP. Это one-mask protocol с target labels; его нельзя объединять с
zero-shot 64-mask таблицами.

В отдельном более позднем zopt-анализе, отражённом в
`pattern/THEORETICAL_STATE.md` и артефактах
`outputs/eval/zopt_results.json`/`effective_mask_analysis.json`, режим `z`
лучше `z0` и `free` во внутренних сравнениях и имеет более Toeplitz-like
support. Это не тот же run, что raw main result 0.9181; в `pattern/RESULTS.md`
зафиксирован более ранний fresh z-only запуск без нового `z0/free`. Конкретный
random reference в высоковариативном one-mask protocol может быть выше `z`.
Корректная интерпретация — decoder manifold может регуляризовать target-time
search, а не доказательство zero-shot superiority.

### 3.12. Effective structure и роль второго слоя

Для сохранённых `z`, `z0`, `free` masks дополнительно анализировалась
effective importance:

```text
E = mean_repeats(abs(W1 * M) * abs(W2)).
```

После best-column alignment доля массы на Toeplitz support увеличивается:

| Режим | Toeplitz mass binary support | Toeplitz mass effective map |
|---|---:|---:|
| z | 0.8145 | 0.8529 |
| z0 | 0.7812 | 0.8267 |
| free | 0.7344 | 0.7948 |

Второй слой действительно подавляет часть неструктурных связей, но порядок
`z > z0 > free` сохраняется. Анализ post-hoc и корреляционный: он не доказывает
каузальную необходимость Toeplitz geometry.

### 3.13. Ограничения `pattern`

1. Все tasks имеют одну и ту же gold support; OOD меняет pattern identity, но
   не структурный regime.
2. Headline in-distribution run не даёт uncertainty по всем источникам seeds.
3. Пять OOD splits переиспользуют конечный набор из 16 tasks.
4. `|W1 * M|` — proxy полезности связи; signs отброшены, `W2` не входит.
5. Gold IoU и gold alignment — diagnostics, а не независимые evidence.
6. Alignment history требует воспроизведения после исправления latent sampling.
7. Хорошее downstream качество не означает точного восстановления Toeplitz
   rule: observed mean IoU заметно ниже 1.

## 4. Эксперимент `motif_pair`

Основные источники: [`motif_pair/README.md`](motif_pair/README.md) и
[`motif_pair/OOD_RESULTS.md`](motif_pair/OOD_RESULTS.md).

### 4.1. Зачем понадобился второй эксперимент

В `pattern` structural support одинакова для всех задач. Поэтому unconditional
VAE достаточно выучить один общий local-window prior.

`motif_pair` усложняет вопрос: полезная support зависит от task variable
`gap`. Генератор должен не только выучить общий circular/local prior, но и
выбрать подходящий structural mode по condition. При этом motif pair `(A,B)`
на meta-test ранее не встречалась.

### 4.2. Задача

Task задаётся тройкой:

```text
tau = (A, B, gap),
```

где:

- `A` и `B` — разные ordered 3-bit motifs;
- вход — circular sequence `x in {-1,+1}^16`;
- `gap in {3,4,5,6,7,8,9,10}`;
- positive: unique occurrence `B` начинается ровно через `gap` позиций после
  unique occurrence `A`, по модулю 16;
- negative: те же unique motifs находятся на другом допустимом gap.

Полное пространство `2^16` sequences перебирается и фильтруется. Positive
examples не создаются artificial injection. Negative examples являются hard
negatives: оба мотива присутствуют ровно по одному разу, но их relative gap
неверен.

При balanced sampling positive и negative дополнительно matched по total
number of ones. Это уменьшает простой density shortcut.

### 4.3. Shortcut audit

До neural pipeline обучается linear probe по raw 16 bits и simple count
features. Для split seed 42:

```text
mean accuracy = 0.5033
max accuracy  = 0.5254
```

Это не обнаруживает простой линейный shortcut, но не исключает все возможные
нелинейные shortcuts.

### 4.4. Gold support

Downstream MLP имеет 16 inputs и 16 hidden units; mask `16 x 16`.
Для hidden unit `h`:

```text
M_gold[h : h+3 modulo 16, h] = 1
M_gold[h+gap : h+gap+3 modulo 16, h] = 1.
```

Каждая hidden column видит два локальных трёхпозиционных receptive fields.
Всего `16 * 6 = 96` активных рёбер из 256. Геометрически это две circular
diagonal bands ширины 3, separation которых задаётся gap.

Binary support зависит от gap, но не от motif identities. `A` и `B` влияют на
нужные signs/values weights. Поэтому condition CVAE — только gap one-hot.

### 4.5. Pair-disjoint OOD split

Для каждого gap строится каталог из восьми допустимых tasks:

- 6 tasks на gap идут в meta-train;
- 2 tasks на gap идут в held-out test;
- итого 48 train и 16 test tasks.

Split validator обеспечивает:

- test ordered pair `(A,B)` не встречается ни в одной train task, даже при
  другом gap;
- held-out pairs уникальны;
- каждый motif встречается в meta-train и в роли `A`, и в роли `B`;
- все восемь gaps представлены в meta-train.

Это **compositional OOD по unseen ordered motif pairs при seen gaps**. Это не
экстраполяция на невиданные gaps.

Split, ordered train task list и SHA256 provenance переносятся через datasets,
candidate shards, importance maps, generator checkpoints и evaluation. Stale
artifact от другого split должен приводить к явной ошибке.

Текущий сохранённый run:

```text
pair_disjoint_seed_42
split SHA256 = 33e3a2c40aeee04b0573855782930be5485e718774eac594337352e5568cc7af
```

### 4.6. Candidate bank

На каждой из 48 meta-train tasks обучается:

| Параметр | Значение |
|---|---:|
| candidate MLP | 512 |
| input / hidden | 16 / 16 |
| proposal mask | Bernoulli(0.375) |
| expected active edges | 96 из 256 |
| Adam steps | 1000 |
| batch size | 128 |
| learning rate | `1e-3` |
| selection | top-10% по validation BCE |

Targets генератора — continuous normalized `abs(W1*M)` maps выбранных
candidates, а не их binary masks.

### 4.7. Canonicalization hidden units

Hidden columns exchangeable: одна и та же функция может быть представлена
разными перестановками hidden units. Перед VAE/CVAE maps canonicalize generic
local three-position anchor signatures.

Canonicalization:

- только переставляет columns;
- сохраняет все значения continuous map;
- не использует task gap;
- не использует labels;
- не использует held-out tasks.

Её цель — убрать permutation nuisance, не подсказывая генератору gold gap.

### 4.8. VAE и gap-conditioned CVAE

Модели имеют одинаковый encoder/decoder backbone:

| Параметр | Значение |
|---|---:|
| input dimension | 256 |
| latent dimension | 32 |
| hidden dimension | 256 |
| epochs | 80 |
| batch size | 256 |
| learning rate | `1e-3` |
| reconstruction | BCE по continuous importance targets |
| decoding | global top-96 |

Различия:

- unconditional VAE получает condition dimension 0 и моделирует marginal по
  всем gaps;
- CVAE получает one-hot dimension 8, кодирующий gap;
- motif identities не передаются ни одной модели.

VQ-VAE намеренно отложен: top-96 уже даёт discrete fixed-cardinality output, а
codebook добавил бы отдельный collapse mechanism без прямой проверки главной
conditioning hypothesis.

### 4.9. Train-only beta selection и posterior collapse

Проверяется grid:

```text
1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003.
```

Beta выбирается только по internal validation split meta-train importance
maps. Held-out tasks и их downstream results не используются.

Выбирается наибольшая beta, стабильно проходящая anti-collapse guard на всём
финальном training tail:

- validation KL не ниже 1;
- минимум две dimensions, active по variance posterior mean
  (`Var_x(mu) > 1e-2`);
- diagnostics конечны;
- posterior reconstruction лучше reconstruction при `z=0`.

Обе модели выбрали `beta=0.1`:

| Модель | Validation KL | Active dimensions | Posterior vs `z=0` recon gap |
|---|---:|---:|---:|
| VAE | 2.232 | 13 | 2.162 |
| CVAE | 2.152 | 24 | 1.942 |

Таким образом, выбранные checkpoints не являются полностью collapsed
deterministic decoders.

### 4.10. Held-out evaluation

Для каждой из 16 test tasks и каждого stochastic method генерируется 64 masks.
VAE/CVAE scores бинаризуются top-96. Для каждой mask обучается новый MLP с
нуля 1000 steps.

Инициализации weights и train batches paired между методами по ordinal sample.
Это уменьшает шум сравнения масок.

В protocol отсутствуют:

- selection по target validation loss;
- обучение на target importance maps;
- target-time latent optimization;
- использование ideal mask для выбора generator checkpoint.

### 4.11. Baselines

- `random`: Bernoulli(0.375);
- `random_exact96`: равномерная mask ровно с 96 edges, primary null;
- `conditional_mean`: top-96 среднего train importance для данного gap;
- `vae`: unconditional VAE sample;
- `cvae`: CVAE sample с правильным gap;
- `cvae_wrong_gap`: тот же latent sample с намеренно неверным gap condition;
- `ideal`: gold support, diagnostic ceiling.

`conditional_mean` особенно важна: она проверяет, нужен ли stochastic latent
generator, или достаточно gap-conditioned prototype, посчитанного только по
meta-train tasks.

### 4.12. Метрики

Functional:

- downstream accuracy;
- downstream BCE.

Structural:

- Hungarian best-permutation IoU между generated и gold mask.

Hungarian matching переставляет только hidden columns. Input positions не
переставляются, labels не используются. IoU — diagnostic и не заменяет
downstream evaluation.

Paired t-intervals текущего run строятся по 16 held-out task-level differences.
Они отражают variability tasks внутри одного split, но не variability между
pair-disjoint splits.

### 4.13. Результат split seed 42

| Метод | Accuracy | BCE | Hungarian IoU |
|---|---:|---:|---:|
| random Bernoulli | 0.7268 | 0.5641 | 0.436 |
| random exact-96 | 0.7274 | 0.5630 | 0.438 |
| unconditional VAE | 0.7381 | 0.5549 | 0.574 |
| CVAE, wrong gap | 0.7403 | 0.5537 | 0.580 |
| **CVAE, correct gap** | **0.7480** | **0.5434** | **0.632** |
| train-only conditional mean | **0.7497** | **0.5426** | 0.660 |
| ideal support | 0.7509 | 0.5452 | **1.000** |

Основные paired comparisons:

| Сравнение | Metric | Mean difference | 95% task-level t-CI | p-value | Wins |
|---|---|---:|---:|---:|---:|
| CVAE - random exact-96 | accuracy | +0.02063 | `[+0.01089,+0.03036]` | 0.00041 | 15/16 |
| CVAE - random exact-96 | IoU | +0.19342 | `[+0.10555,+0.28129]` | 0.00029 | 14/16 |
| CVAE - VAE | accuracy | +0.00998 | `[+0.00145,+0.01850]` | 0.0248 | 9/16 |
| CVAE - VAE | IoU | +0.05813 | `[+0.01415,+0.10212]` | 0.0130 | 10/16 |
| CVAE - wrong gap | accuracy | +0.00772 | `[-0.00158,+0.01703]` | 0.0972 | 11/16 |
| CVAE - wrong gap | IoU | +0.05147 | `[+0.00489,+0.09805]` | 0.0326 | 11/16 |

CVAE превосходит matched-cardinality random и unconditional VAE. Это evidence,
что generator выучил общий structural prior и что gap condition добавляет
полезную информацию.

Correct-gap CVAE структурно лучше wrong-gap CVAE. Accuracy direction также
положительное, но confidence interval пересекает zero; устойчивое functional
преимущество correct condition над wrong condition пока не подтверждено.

### 4.14. Почему conditional mean меняет интерпретацию

`conditional_mean` имеет accuracy 0.7497 против 0.7480 у CVAE и IoU 0.660
против 0.632. Difference CVAE minus conditional mean:

```text
accuracy = -0.00170
95% CI   = [-0.00407, +0.00068]
p        = 0.148
```

Следовательно, текущий результат не показывает превосходство stochastic CVAE
над простым train-only gap-conditioned prototype. Он показывает, что из
successful networks извлекается переносимая gap-dependent структура, а CVAE —
один работающий механизм её представления.

### 4.15. Геометрические симметрии

Некоторые gold masks эквивалентны после разрешённой permutation hidden
columns: gaps 6 и 10 образуют одну equivalence, gaps 7 и 9 — другую. Поэтому
Hungarian IoU не всегда различает все gap identities как независимые modes.

Это не invalidates functional comparison, но ограничивает интерпретацию
gap-wise structural plots.

### 4.16. Почему ideal не всегда лучший по BCE

Ideal support — известная достаточная конструкция, а не доказанный global
optimum finite-budget training. При 1000 Adam steps CVAE/conditional mean могут
иметь немного лучший BCE, хотя ideal даёт highest accuracy и IoU 1.

Поэтому нельзя оценивать правильность gold geometry только тем, заняла ли она
первое место по каждой finite-training metric.

### 4.17. Ограничения `motif_pair`

1. Headline получен только на одном pair-disjoint split seed 42.
2. Task-level paired CI не учитывает variability выбора task catalog/split.
3. CVAE не превосходит conditional mean.
4. Correct-gap accuracy против wrong-gap пока не significant на уровне 0.05.
5. Все gaps видимы в meta-train; unseen-gap extrapolation не проверяется.
6. Linear shortcut audit не исключает сложные nonlinear shortcuts.
7. Gold support не доказана оптимальной для finite-width/finite-step MLP.
8. IoU — post-hoc diagnostic; он не равен functional performance.
9. Learned masks приближают, но не точно восстанавливают gold rule.

## 5. Сопоставление экспериментов

| Свойство | `pattern` | `motif_pair` |
|---|---|---|
| Input | length 8, linear | length 16, circular |
| Task identity | один 4-bit pattern | ordered `(A,B,gap)` |
| Gold support | одно shared Toeplitz family | gap-dependent circular two-band family |
| Mask size | `8 x 8`, top-32 | `16 x 16`, top-96 |
| Candidate/task | 2000 | 512 |
| Candidate steps | 2000 | 1000 |
| Selected targets | top-10% continuous `|W1*M|` | top-10% continuous `|W1*M|` |
| Hidden canonicalization | raw default; alignment optional | canonicalization default |
| Generator | unconditional VAE | VAE ablation + gap-conditioned CVAE |
| OOD axis | unseen pattern identities | unseen ordered motif pairs |
| Structural regime at test | тот же shared support | seen gaps, unseen pair combinations |
| Primary null | random exact-32 | random exact-96 |
| Headline gain | +0.0266 mean OOD gain, 5 splits | +0.0206, one split |
| Strong simple baseline | mean/regressor заметно хуже VAE | conditional mean немного лучше CVAE |

Логика развития проекта:

1. В пяти re-splits одних и тех же 16 synthetic `pattern` tasks raw importance
   maps успешных MLP содержат общий сигнал о local-window connectivity, а
   unconditional VAE переносит его на held-out pattern identities лучше
   matched-cardinality random.
2. В одном pair-disjoint synthetic `motif_pair` split unconditional VAE
   извлекает общий circular/local prior, а gap-conditioned CVAE улучшает его
   для unseen motif pairs относительно VAE и matched-cardinality random, но не
   превосходит conditional mean.
3. Сильный `conditional_mean` показывает, что следующий вопрос должен быть не
   только «есть ли transfer», но и «когда stochastic latent distribution лучше
   детерминированного conditioned prototype».

## 6. Что совокупно подтверждено

В рамках текущих synthetic environments:

1. Validation-selected successful MLP содержат в `|W1*M|` переносимый
   structural signal.
2. Generator может превратить этот signal в binary masks, функционально лучшие
   matched-cardinality random masks.
3. Улучшение наблюдается после полного retraining downstream weights; это не
   перенос pretrained weights.
4. В `pattern` общий local prior переносится между held-out task identities.
5. В `motif_pair` unconditional VAE учит общий circular/local prior, а gap
   condition даёт дополнительный signal для новых motif pairs.
6. Structural similarity и downstream accuracy в целом движутся согласованно,
   хотя не эквивалентны.

## 7. Что пока не подтверждено

Текущие результаты не доказывают:

- перенос на естественные данные или реальные architectures;
- универсальный способ discovery inductive bias;
- точное восстановление Toeplitz/circular gold rule;
- causal necessity конкретной gold support;
- превосходство VAE/CVAE над любым простым estimator prior;
- multi-seed robustness `motif_pair`;
- extrapolation `motif_pair` на unseen gaps;
- superiority target-time latent optimization над random в общем случае;
- что высокая Hungarian IoU гарантирует лучшее downstream качество;
- что importance `|W1*M|` — оптимальное представление успешного решения.

## 8. Воспроизведение

Окружение проекта:

```bash
conda activate ras
```

### 8.1. `pattern`

Полный pipeline:

```bash
cd pattern
GPU_IDS="0 1 2 3" bash scripts/run_all.sh
```

`run_all.sh` выполняет data generation, candidate training, selection и
importance extraction, VAE/regressor training, downstream evaluation. По
default он также запускает отдельную z-optimization ветку. Для чистого
zero-shot pipeline команда имеет вид:

```bash
RUN_Z_OPT=0 GPU_IDS="0 1 2 3" bash scripts/run_all.sh
```

Отдельные стадии:

```bash
bash scripts/01_generate_data.sh
GPU_IDS="0 1 2 3" bash scripts/02_train.sh
bash scripts/03_select.sh
GPU_IDS="0" bash scripts/04_train_cvae.sh
GPU_IDS="0 1 2 3" bash scripts/05_eval.sh
```

Held-out split:

```bash
SPLIT_SEED=42 GPU_ID=3 bash scripts/09_ood.sh
```

Агрегация seeds 42–46:

```bash
python evaluation/aggregate_ood.py \
  --summaries outputs/ood/split_seed_{42,43,44,45,46}/summary.json \
  --out outputs/ood/aggregate_seeds_42_46.json
```

### 8.2. `motif_pair`

Recommended full pipeline:

```bash
cd motif_pair
GPU_IDS="0 1 2 3" BETA_GPU_IDS="3 6 7" SPLIT_SEED=42 ./run_all.sh
```

Он выполняет:

1. split, datasets и shortcut audit;
2. candidate banks;
3. top-10% selection и importance extraction;
4. train-only beta sweep VAE/CVAE;
5. held-out evaluation;
6. PNG/PDF plots и `plots/manifest.json`.

Нижележащие scripts полезны для debugging fixed-beta run, но **не
воспроизводят reported train-only beta sweep и promotion**. Для headline
результата следует использовать только согласованный `run_all.sh` выше.
Debug-only fixed-beta stages:

```bash
DATA_GPU=3 SPLIT_SEED=42 SPLIT_JSON=outputs/split.json bash scripts/01_generate_data.sh
GPU_IDS="0 1 2 3" SPLIT_JSON=outputs/split.json bash scripts/02_train.sh
POSTPROC_GPU=3 SPLIT_JSON=outputs/split.json bash scripts/03_select.sh
GEN_GPU=3 SPLIT_JSON=outputs/split.json bash scripts/04_train_generative.sh
EVAL_GPU=3 SPLIT_JSON=outputs/split.json bash scripts/05_eval.sh
```

`scripts/06_plot.sh` нужно запускать только с `RUN_ROOT` и остальными paths,
явно указывающими на тот же debug run; его bare defaults относятся к другому
историческому output root.

Heavy numerical stages требуют CUDA и намеренно не делают silent CPU fallback.

## 9. Карта кода и артефактов

### `pattern`

```text
pattern/
  config.py                    размеры и default hyperparameters
  data/generate.py             synthetic pattern data и ideal mask
  models/mlp.py                candidate/downstream masked MLP
  models/cvae.py               VAE и top-K sampling
  models/train_cvae.py         generator training
  selection/select_best.py     top-10% candidates
  evaluation/importance.py     continuous |W1*M| maps
  evaluation/eval_generated_masks.py
                               downstream comparison
  evaluation/ood_split.py      12/4 pattern split
  evaluation/aggregate_ood.py  aggregation across split seeds
  scripts/09_ood.sh            leakage-safe held-out pipeline
  RESULTS.md                   current main run
  OOD_RESULTS.md               five-split held-out result
  THEORETICAL_STATE.md         interpretation and open questions
```

Regenerable outputs:

```text
pattern/outputs/
  data/
  checkpoints/pattern_<pattern>/
  cvae/
  eval/
  ood/split_seed_<seed>/
  plots/
```

### `motif_pair`

```text
motif_pair/
  config.py                       Task и hyperparameters
  data/generate.py                exhaustive circular data bank
  data/audit.py                   shortcut probe
  models/mlp.py                   masked MLP
  models/cvae.py                  VAE/CVAE, canonicalization, provenance
  models/train_cvae.py            training и collapse diagnostics
  models/sweep_beta.py            train-only beta selection
  selection/select_best.py        top-10% candidates
  evaluation/task_split.py        pair-disjoint split
  evaluation/importance.py        continuous importance
  evaluation/eval_generated_masks.py
                                  held-out downstream evaluation
  evaluation/structural.py        Hungarian IoU
  evaluation/summarize.py         paired statistics
  evaluation/plot_pipeline.py     stage-wise diagnostics/manifest
  run_all.sh                      production entry point
  OOD_RESULTS.md                  full result and interpretation
```

Regenerable outputs:

```text
motif_pair/outputs/ood/pair_disjoint_seed_42/
  split.json
  data/
  shortcut_audit.json
  checkpoints/
  generative/selection.json
  eval/eval_results.json
  eval/summary.json
  plots/manifest.json
```

Binary checkpoints, datasets and plots under `outputs/` исключены из Git и
должны воспроизводиться pipeline, а не храниться в репозитории.

## 10. Приоритетные следующие эксперименты

### 10.1. Multi-seed `motif_pair`

Зафиксировать несколько pair-disjoint split seeds до запуска, повторить весь
train-only beta selection и downstream pipeline, затем считать uncertainty по
split-level means. Это главный шаг для проверки headline result.

### 10.2. Unseen-gap transfer

Hold out целые gap values. Текущая one-hot condition не умеет интерполировать
на unseen category, поэтому потребуется condition representation с круговой
геометрией или числовым relative-position encoding. Это будет более сильная
задача, чем текущий unseen-pair/seen-gap OOD.

### 10.3. Понять преимущество stochastic generator

Сравнить CVAE с:

- conditional mean;
- mixture/prototype baselines;
- nearest-neighbour prior;
- deterministic conditioned decoder;
- одинаковым diversity/cardinality budget.

Нужно определить task families, где distribution over structures полезнее
одного усреднённого prototype.

### 10.4. Повторить alignment `pattern`

Перезапустить raw, selfalign, refmatch, window и gold variants с исправленным
independent-z sampling и одинаковыми seeds/downstream budgets. Отдельно
маркировать data-only methods и методы, использующие знание gold family.

### 10.5. Улучшить representation успешного решения

Сравнить:

- binary candidate mask;
- `|W1*M|`;
- signed `W1*M`;
- effective `|W1*M|*|W2|`;
- sensitivity/gradient/Fisher-style importance;
- representations с явной permutation invariance.

### 10.6. Target-time adaptation как отдельный protocol

Для frozen-decoder latent optimization нужен отдельный target validation set,
не совпадающий с final test. Следует сравнить `z`, `z0`, free logits, random
search и simple conditioned prototypes при одинаковом target-data budget.
Результаты нельзя объединять с zero-shot evaluation.

### 10.7. Causal interventions

Для различения корреляции и причинности можно:

- сохранять cardinality, но разрушать локальность;
- сохранять diagonal profile, но менять gap;
- переставлять input positions при фиксированной hidden permutation;
- удалять связи с высокой/низкой effective importance;
- проверять, сохраняется ли downstream gain после контролируемых perturbations.

## 11. Короткая версия для обсуждения

Проект изучает, можно ли извлечь переносимый structural prior из множества
успешно обученных masked MLP. Для каждой meta-train задачи обучается банк MLP
со случайной фиксированной маской первого слоя. Лучшие 10% по validation BCE
дают continuous normalized maps `|W1*M|`, на которых обучается VAE/CVAE.
Decoder samples превращаются через top-K в новые fixed-cardinality masks, после
чего новые MLP обучаются с нуля на held-out задачах.

В `pattern` все 16 задач распознавания 4-bit substring в length-8 sequence
имеют одну pattern-independent Toeplitz support. Unconditional VAE переносит
этот prior: на пяти 12/4 task splits accuracy 0.9086 против 0.8821 у exact-32
random, mean gain +0.0266, положительный во всех splits. Это transfer внутри
семейства с одной общей geometry, а не новая structural regime.

В `motif_pair` задача распознаёт ordered pair 3-bit motifs в circular length-16
sequence; useful support зависит от gap 3–10 и имеет 96 edges. Pair-disjoint
split содержит 48 meta-train и 16 held-out tasks с unseen `(A,B)` pairs, но
seen gaps. На split seed 42 gap-conditioned CVAE даёт 0.7480 против 0.7274 у
exact-96 random и 0.7381 у unconditional VAE. Однако conditional mean даёт
0.7497, correct-vs-wrong-gap accuracy CI пересекает zero, и multi-split
robustness ещё не проверена.

Сильнейший общий вывод в зафиксированных synthetic protocols: successful
sparse networks содержат переносимый signal о полезной connectivity, а
генеративная модель способна превратить его в masks, функционально лучшие
matched-cardinality random. Для `pattern` evidence получено на пяти re-splits
одних 16 task identities, для `motif_pair` — на одном pair-disjoint split; в
последнем conditional mean немного лучше CVAE. Открытый вопрос — когда нужен
именно stochastic latent generator, насколько prior переносится на новые
structural regimes и сохраняется ли эффект вне синтетических сред.
