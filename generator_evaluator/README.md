# Cooperative generator/evaluator

Основная схема — совместный цикл `--training-mode joint`: после начальной
реконструкции каждый update сочетает качество своей задачи, measured-target
feedback, реконструкцию и agreement на общем случайном латенте. Оценщик
обновляется по реальным измерениям во время поиска. Отдельный этап cooperation
и обучаемый общий латент используются только в экспериментальном режиме
`--training-mode staged`.

Строгие определения и псевдокод основного алгоритма:
[mds/best_algo.md](../mds/best_algo.md). Запуски обоих доменов:
`sh_scripts/pattern.sh` и `sh_scripts/deepsets.sh`.

Банки в `inputs.pt`, checkpoints и frozen-артефактах сохраняются как ссылки:
карты и их данные записываются один раз в общий `bank_assets/` каталога запуска.
Bootstrap и search переиспользуют неизменившиеся карты. Этот каталог нужен
для resume и warm-start; ссылки пока привязаны к абсолютному пути запуска.
Неудачные записи удаляют собственные временные файлы и сохраняют предыдущий checkpoint.

Реализация следует [кооперативному плану](../mds/PLAN_GENERATOR_EVALUATOR.md):
отдельный Transformer-генератор на каждый train-паттерн, индивидуальный
functional bank для него и один глобальный context-conditioned ансамбль
оценщиков. Генераторы предлагают маски, а общий поиск кросс-оценивает каждую
маску на каждом train-паттерне; это не один общий генератор.

## Cooperative pattern-run

Из корня проекта, в окружении `ras`, в отдельном терминале на выбранной карте:

```bash
source /home/udeneev-av/miniconda3/etc/profile.d/conda.sh
conda activate ras
CUDA_VISIBLE_DEVICES=GPU-6784dc4e-6ec9-2266-5d23-85bf1b1c2af3 \
python -u -m generator_evaluator.cooperative_run \
  --train-patterns 0001 0011 --test-pattern 0101 \
  --preset pattern-small --seed 4100 --device cuda:0 --progress \
  --out outputs/generator_evaluator/20261002_pattern_small_seed4100
```

Вторая независимая реплика на другой карте:

```bash
CUDA_VISIBLE_DEVICES=GPU-f8501f2d-53bc-9087-6041-64ee69876325 \
python -u -m generator_evaluator.cooperative_run \
  --train-patterns 0001 0011 --test-pattern 0101 \
  --preset pattern-small --seed 4101 --device cuda:0 --progress \
  --out outputs/generator_evaluator/20261002_pattern_small_seed4101
```

Каждый процесс видит назначенную карту как `cuda:0`. `--progress` показывает
построение банков по плотностям, terminal child fits, предложения генераторов,
кросс-оценку, обновления глобального evaluator, feedback functional maps,
distillation, выбор на validation-pool и sealed test. `pattern-small` — компактный
протокол для этой простой задачи: 500 support-only шагов реальной метки и две
инициализации, $K=32$ для $11\times8$, 5 teacher-решений на паттерн (200
support-only шагов, одна инициализация), probe из 32 точек,
Transformer ширины 16 (2 heads, 1 слой, noise 4), ансамбль из 2 оценщиков,
2 generator-эпохи по 4 updates и 10 evaluator-эпох в каждом refresh. В раунде
по 8 предложений каждого генератора и измеряются 2 кандидата. Дополнительного
auxiliary-acquisition нет.

Входной банк всё же сохраняет пять плотностей $K\in\{9,32,44,62,88\}$, включая
dense, и чередует mixed/single-density view. Генератор предлагает только
целевой $K=32$: так сохраняется защита от sparse functional-map mismatch без
дорогого отдельного этапа измерений для вспомогательных бюджетов. Независимые
child fits с одинаковыми размерами объединяются в батчи; это меняет только
скорость, не правило получения метки.

`0001` и `0011` — train, `0101` — test. Параметры можно изменить, лишь если
train и test принадлежат разным reversal/complement-орбитам; runner проверяет это
до вычислений. Validation — отдельные наблюдения тех же train-паттернов; третьей
validation-задачи нет. Test остаётся закрытым до frozen checkpoint.

Для `pattern-small` выбираются 64 support и 64 query наблюдения из независимых
reserved pools по 256 наблюдений. Query
равномерно выбирается из reserved query-pool и не зависит от class-balanced
support-count, поэтому его class prior не наследует support bias.
До teacher fits opaque preflight читает только class counts будущего test-support,
чтобы отклонить невозможный balanced size; test-query и финальные `TaskData`
остаются закрытыми до frozen checkpoint, без сохранения их labels или features.

Для продолжения прерванного запуска повтори те же аргументы с `--resume`.
Изменение паттернов, банка, solver или бюджета требует нового `--out`.

## Что является настоящими данными

Каждая train-label evaluator — реальное query-измерение после нового support-only
обучения весов с фиксированной маской. Forecast evaluator не используется как
метка. Acquisition получает pooled generator proposals, random exploration и
edge-swap мутации qualified elites; каждая выбранная маска проходит terminal
fits на обоих train-паттернах. В банк правильного генератора возвращаются только
одна карта каждой плотности для сохранения anchors и один дополнительный
лучший joint candidate с $K=32$; held-out карты туда не попадают. Возвращаемые
functional maps и states используют тот же token/probe формат, что и исходные
решения. Банк строится по нескольким sparse-плотностям и dense, а обучение
генератора чередует mixed-density и single-density views, чтобы sparse feedback
не был OOD.

Локальный policy-step каждого генератора оптимизирует predicted delta только его
собственного паттерна. Квалифицированная общая элита требует настоящего качества
не хуже dense на **обоих** паттернах с margin $0$; затем отдельный второй
optimizer step distill-ит её в каждый генератор через column-Hungarian-aligned
BCE с весом $0.1$. Held-out topology labels
сохраняются в replay для диагностики, но не попадают в optimizer/evaluator train,
feedback bank или distillation; selection data устроены так же. Полный run
сравнивает agreement mask с dense, random и двумя functional baselines, по одному
от каждого train-паттерна. Cooperative CLI не включает прямой REINFORCE-контроль:
он доступен только в legacy shared-generator runner. Эксперимент не гарантирует
улучшение accuracy: у двух паттернов может не быть общей качественной маски.

Implementation и CPU smoke полного пайплайна проверены. `pattern-small` —
быстрый проверочный эксперимент, а не подтверждение научной гипотезы. Полный
вариант сохранён как явный `--preset full`; пользовательский запуск с 2000
шагами уже начат, его результаты здесь не интерпретированы. В микрозамере восемь
пар `(маска, паттерн)` с двумя инициализациями и 500 шагами заняли 0.56 s на CPU
и 0.92 s на GPU 1; это не оценка времени всего пайплайна. В артефактах
сохраняется raw/canonical overlap предложений как диагностика agreement.

## Legacy shared-generator runner

`python -m generator_evaluator.run` — сохранённая предыдущая реализация с одним
общим генератором. Она не реализует cooperative agreement и не должна
использоваться для нового эксперимента. Её historical smoke-results остаются в
`outputs/generator_evaluator/`.
