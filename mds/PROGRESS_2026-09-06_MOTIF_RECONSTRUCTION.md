# Motif-pair: исходные importance-карты и реконструкции CVAE

Дата: 2026-09-06. Продолжение [oracle-диагностики](PROGRESS_2026-09-06_MOTIF_ORACLE_IDEAL.md).

## Результат

**Основной недостаток бинарного ideal-support присутствует уже в отдельных
входных importance-картах. CVAE в среднем улучшает их top-96 структуру,
а не разрушает готовую ideal.** Но это не означает отсутствия структурной
информации в непрерывных картах: усреднение карт одного gap даёт гораздо более
сильную структуру, чем top-96 отдельной карты.

На внутренней валидации CVAE, только на известных gaps:

| Профиль модели | Исходная карта, top-96 | Реконструкция через mu, top-96 | Decoder при z=0 | Средняя карта gap по internal train |
|---|---:|---:|---:|---:|
| interpolation | 0.4490 | 0.6054 | 0.6454 | 0.6528 |
| extrapolation | 0.4489 | 0.6717 | 0.7043 | 0.6826 |

В таблице средний IoU с ideal после оптимальной перестановки скрытых колонок.
Средние взвешены по картам; internal-val содержит разное число карт для разных gaps.
Названия профилей обозначают **ранее обученные модели**, а не режим оценки
этой диагностики: unseen gaps здесь не использовались.

Ни у одной из 2448 входных карт каждого профиля top-96 не равен ideal.
Максимальный исходный IoU — 0.5238 у interpolation и 0.5360 у extrapolation.
Точных posterior-mean реконструкций ideal тоже нет.

## Единица сравнения и происхождение данных

Единица сравнения — одна и та же отобранная карта конкретного кандидата
MLP при фиксированных task и gap: её канонизированный вход x и
`sigmoid(D(mu(x,c),c))` того же замороженного CVAE.
Вход служит эталоном **точности реконструкции**; отдельный ideal(gap) служит
эталоном **структуры**. Эти эталоны не совпадают, и их качества нельзя смешивать.

| Этап / фактор | Статус | Что проверено |
|---|---|---|
| Профили/checkpoint | зафиксированы; между профилями различаются | Те же seed-42 checkpoint, beta 0.1 / 0.3, epochs 79 / 78; SHA из oracle-протокола |
| Источники карт | совпадают с путями и provenance checkpoint | 48 train-задач × 512 кандидатов; текущие hashes файлов сохранены |
| Отбор | совпадает с production loader | По 51 карте с минимальным source validation BCE; никакого отбора по gold |
| Preprocessing | совпадает | Те же нормированные continuous importance и канонизация колонок; тензоры совпали побитово |
| Internal split | воспроизведён | Seed 42, глобальный randperm, 2081 train / 367 val; совпали loader-тензоры и checkpoint counts |
| Condition | фиксирован | Scalar `(gap-3)/7`; только gaps из train split |
| Реконструкция | намеренное отличие от stochastic forward | Использован posterior mean mu, без сэмплирования z |
| Модель | зафиксирована | Нет обучения, оптимизации z или изменения весов |
| Runtime | зафиксирован в диагностике | ras, PyTorch 2.3.1+cu121, A100, deterministic CUDA; повторный decode совпал |
| Исторический runtime/байты входов | не полностью удостоверены | Старые checkpoint не содержат исторических hashes карт/runtime; нынешние источники зафиксированы, deterministic val KL совпал со старым |
| Сериализация/индексы | проверены | Сохранены task, global candidate index, gap, принадлежность internal-val; нет CSV/межъязыкового преобразования |

Карты — это `abs(W1 * binary_mask) / max(abs(W1 * binary_mask))` для каждого
кандидата. CVAE обучался на **непрерывных** значениях, а не на исходной бинарной
связности MLP и не на ideal. Колонки канонизируются до encoder; повторная
канонизация выхода decoder не выполняется.

Здесь 2448 карт на профиль, из них 367 internal-val. Это валидация на картах
из meta-train задач, а не на held-out задачах или gaps.

## Что происходит до CVAE

В отобранной карте в среднем 103.34 / 103.43 ненулевых элемента для
interpolation / extrapolation. Top-96 сохраняет в среднем **92.78% / 92.73%**
её ненулевой связности. Исходные маски кандидатов генерировались Bernoulli:
такой top-96 лишь немного прореживает случайный support, даже если величины
обученных весов уже несут полезную информацию.

Следовательно, низкий IoU бинаризации отдельной карты нельзя трактовать как
отсутствие сигнала в весах. Сильное улучшение после усреднения по gap
показывает наличие общей структуры в совокупности карт.

Канонизация переставляет колонки, не меняя значения карты. Проверено:

- отсортированные значения raw и canonical совпадают побитово;
- IoU ненулевого support с ideal инвариантен;
- доля непрерывной массы на ideal после собственного Hungarian alignment
  инвариантна с точностью 1e-6;
- средние raw/canonical top-96 IoU отличаются менее чем на 0.0002.

Top-96 на raw/canonical может различаться на отдельных примерах из-за ties:
у 286/2448 карт interpolation и 277/2448 extrapolation 96-й элемент равен нулю.
Среднее число добавленных нулевых связей — 0.456 / 0.429. Максимальная
разница raw/canonical top-96 IoU на одном примере — 0.03184. Это эффект
разрешения равенств при бинаризации, а не изменения continuous-карты.

## Поведение CVAE

На всех выбранных картах (train и internal-val вместе) реконструкция улучшает
top-96 IoU относительно соответствующего входа в 84.60% случаев у
interpolation и 94.36% у extrapolation. Ухудшение — в 13.81% и 3.92% случаев,
остальные не меняются.

| Профиль | Часть | Input IoU | Reconstruction IoU | z=0 IoU | Gap-mean IoU |
|---|---|---:|---:|---:|---:|
| interpolation | internal train | 0.4485 | 0.6092 | 0.6564 | 0.6569 |
| interpolation | internal val | 0.4490 | 0.6054 | 0.6454 | 0.6528 |
| extrapolation | internal train | 0.4470 | 0.6874 | 0.7201 | 0.6957 |
| extrapolation | internal val | 0.4489 | 0.6717 | 0.7043 | 0.6826 |

Разрыв train/val невелик на уровне этих метрик; он не объясняет главный
эффект. Из этого не следует доказательство отсутствия переобучения вообще.

Качество реконструкции конкретного входа на internal-val:

| Профиль | BCE(input, reconstruction) | BCE(input, z=0) | MSE(input, reconstruction) | MSE(input, z=0) | IoU reconstruction/input, исходные колонки | IoU reconstruction/input, после перестановки |
|---|---:|---:|---:|---:|---:|---:|
| interpolation | 98.7238 | 99.7812 | 0.04632 | 0.04752 | 0.3271 | 0.4513 |
| extrapolation | 99.6463 | 100.6306 | 0.04833 | 0.04950 | 0.3290 | 0.4485 |

BCE суммируется по 256 значениям карты и усредняется по примерам; MSE
усредняется и по значениям, и по примерам. BCE считался по logits.
Это deterministic posterior-mean reconstruction: он не обязан совпадать со
старым validation reconstruction loss, где latent сэмплировался из posterior.

**mu помогает точнее реконструировать конкретную importance-карту, но в среднем
отдаляет top-96 от ideal относительно z=0.** Значит, увеличение точности
исходной reconstruction-цели само по себе не гарантирует лучшей ideal-структуры.
Это свидетельство использования latent для особенностей примеров, а не
доказательство полной posterior collapse.

И улучшение top-96 не означает улучшения любой структурной метрики:
доля непрерывной массы на ideal после Hungarian alignment падает с
0.7059 до 0.5626 у interpolation и с 0.7129 до 0.5804 у extrapolation
(все карты). Reconstruction сглаживает разреженные входы; эта массовая
метрика зависит от распределения значений и не равна бинарному IoU.

## Различия между gaps

Ниже только internal-val. Gap-mean вычислен по internal-train картам этого
gap, исключая internal-val, и потому отличается от ранее опубликованного
conditional mean по всем meta-train картам.

| Профиль | Seen gap | Input IoU | Reconstruction IoU | z=0 IoU | Gap-mean IoU |
|---|---:|---:|---:|---:|---:|
| interpolation | 3 | 0.4367 | 0.4963 | 0.5238 | 0.6271 |
| interpolation | 4 | 0.4488 | 0.4283 | 0.4222 | 0.5238 |
| interpolation | 6 | 0.4531 | 0.6024 | 0.6271 | 0.4884 |
| interpolation | 7 | 0.4506 | 0.7892 | 0.8641 | 0.7455 |
| interpolation | 9 | 0.4574 | 0.7677 | 0.8641 | 0.9394 |
| interpolation | 10 | 0.4499 | 0.5948 | 0.6271 | 0.6134 |
| extrapolation | 5 | 0.4558 | 0.4759 | 0.4436 | 0.5610 |
| extrapolation | 6 | 0.4511 | 0.5888 | 0.6410 | 0.4884 |
| extrapolation | 7 | 0.4516 | 0.8318 | 0.9010 | 0.7143 |
| extrapolation | 8 | 0.4271 | 0.7220 | 0.7455 | 0.8462 |
| extrapolation | 9 | 0.4574 | 0.8679 | 0.9200 | 0.9394 |
| extrapolation | 10 | 0.4499 | 0.6149 | 0.6552 | 0.6134 |

Важное исключение из среднего улучшения — interpolation gap 4:
0.4488 → 0.4283, при gap-mean 0.5238. Следовательно, CVAE не одинаково хорошо
извлекает общую структуру для всех известных gaps.

![Реконструкция на известных gaps](../motif_pair/outputs/reconstruction_diagnostic/20260906/seen_gap_reconstruction.png)

## Минимальный парный пример

Для interpolation взята **первая internal-val карта gap 9** в сохранённом
порядке: row 1654, task `A000_B110_G09`, global candidate 154. Выбор не
оптимизирован по IoU; на иллюстрациях одинаковое правило для всех gaps.

- Input top-96 IoU с ideal: 0.4769.
- Reconstruction через mu: 0.6696; BCE к input: 80.8465.
- Decoder при z=0: 0.8641; BCE к тому же input: 88.0840.

Один конкретный пример воспроизводит расхождение целей: mu лучше объясняет
свою входную карту, а z=0 даёт более близкий к ideal support. Полные данные
этого примера находятся в `interp/diagnostic.pt` с указанным row.

## Гипотезы и границы вывода

| Гипотеза | Решающий контроль | Статус |
|---|---|---|
| H1: отдельные входы уже имеют слабый ideal-support | Top-96 входов до decoder | Поддержана для бинарного support: mean ≈0.449, точных ideal нет |
| H2: CVAE главным образом уничтожает готовую ideal при реконструкции | Парные input/reconstruction IoU на internal-val | Не поддержана как общее объяснение: средний IoU растёт; локальное ухудшение есть для interpolation gap 4 |
| H3: preprocessing портит структуру | Raw/canonical invariance по continuous mass и положительному support | Опровергнута для этих преобразований; малые top-96 различия объясняются ties |
| H4: результат вызван неверным split/checkpoint или stochastic forward | Provenance, loader equality, deterministic KL, повторный decode | Не подтверждена; проверки совпали |
| H5: точная ideal недоступна в decoder | Полное исследование образа decoder | Открыта: ни нынешняя реконструкция, ни конечный прошлый oracle-поиск этого не доказывают |

**Наблюдаемое:** нет готовой exact-96 ideal у отдельных входов; в среднем
reconstruction улучшает top-96; простые gap-mean/z=0 сильнее среднего posterior
reconstruction по gold IoU; mu лучше z=0 по реконструкции input.

**Интерпретация:** CVAE извлекает часть общей структуры и сглаживает вариации
кандидатов, но reconstruction-задача заставляет его учитывать и особенности
случайных support. Она не требует, чтобы decoder содержал точную ideal.
Это объясняет, почему хорошая reconstruction сама по себе не обосновывает
ожидание ideal-пересечения двух decoder. Полная причина неудачи oracle остаётся
не разделённой между представлением модели и трудностью поиска.

**Следующая отдельная проверка:** controlled CVAE на ideal-масках только seen
gaps с последующей проверкой reconstruction/oracle отдельно на seen/unseen.
Она отделит способность архитектуры и алгоритма воспроизвести заданную
геометрию от свойств текущих importance-целей. Это диагностическая смена
обучающего распределения, не исправление текущего pipeline и не zero-shot
результат прежней модели. Новое обучение в этой работе не запускалось.

## Журнал проверок и артефакты

1. До расчётов прочитаны production loader, importance extraction и trainer;
   независимо проверены provenance и семантика внутренних split.
2. Диагностика воспроизвела source selection/canonicalization побитово.
3. Val KL: interpolation 1.97445738 против checkpoint 1.97445743;
   extrapolation 1.25506079 против 1.25506086. Это поддерживает совпадение
   модели, preprocessing и состава val на детерминированной величине.
4. Независимый `best_permutation_iou` проверил все 24 480 бинарных масок:
   2 профиля × 2448 карт × 5 этапов. Совпадение в пределах 1e-7.
5. Состояние модели и hashes checkpoint не изменились. Исходный pipeline
   не редактировался; добавлена только отдельная диагностика и отчёт.
   Семантика расчёта и итоговые числа прошли независимое ревью.
6. В служебном расчёте среднего обнаружена ошибка broadcast у PyTorch 2.3.1
   с deterministic CUDA boolean assignment. Она воспроизведена на отдельном
   тензоре без модели. В диагностике использовано явное contiguous expansion
   с той же математикой; deterministic mode сохранён. До исправления новые
   результаты не сохранялись, исходные данные не затрагивались.

Файлы:

- [Сводные числа и threshold-диагностика](../motif_pair/outputs/reconstruction_diagnostic/20260906/summary.json)
- [Interpolation: summary](../motif_pair/outputs/reconstruction_diagnostic/20260906/interp/summary.json)
- [Extrapolation: summary](../motif_pair/outputs/reconstruction_diagnostic/20260906/extrap/summary.json)
- [Примеры interpolation](../motif_pair/outputs/reconstruction_diagnostic/20260906/interp/examples.png)
- [Примеры extrapolation](../motif_pair/outputs/reconstruction_diagnostic/20260906/extrap/examples.png)
- [Диагностический код](../motif_pair/evaluation/diagnose_reconstruction.py)
- [Построение отчётных графиков](../motif_pair/evaluation/report_reconstruction.py)

В `interp/diagnostic.pt` и `extrap/diagnostic.pt` сохранены исходные и
канонизированные карты, реконструкции, z=0, gap-mean, их top-96 маски,
mu/logvar, conditions, targets, индексы кандидатов и internal split.

Запуск из `motif_pair` в `ras` с доступом к CUDA:

```bash
CUDA_VISIBLE_DEVICES=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 python evaluation/diagnose_reconstruction.py \
  --profile interp --out outputs/reconstruction_diagnostic/20260906/interp
CUDA_VISIBLE_DEVICES=3 CUBLAS_WORKSPACE_CONFIG=:4096:8 python evaluation/diagnose_reconstruction.py \
  --profile extrap --out outputs/reconstruction_diagnostic/20260906/extrap
python evaluation/report_reconstruction.py --root outputs/reconstruction_diagnostic/20260906
```

Повторный диагностический запуск требует нового `--out`: перезапись
существующих результатов запрещена.

Продолжение: [обучение CVAE непосредственно на ideal](PROGRESS_2026-09-06_MOTIF_IDEAL_CVAE.md)
и [согласие двух замороженных CVAE против task-loss поиска одного z](PROGRESS_2026-09-06_MOTIF_PAIR_AND_SINGLE_Z.md).
