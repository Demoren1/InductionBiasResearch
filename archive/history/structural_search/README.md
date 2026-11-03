# Structural Search for Neural Optimization

## Current manuscript: integrated draft for author review

The [English paper](main.pdf) is organized around three connected results:
conditions for finite-budget structural search, transfer of sharing relations
across tasks that require new numerical weights, and approximate structure
followed by unrestricted refinement. The introduction states these contributions
and connects each to its theoretical and experimental evidence. The main theory
includes proposal coverage, candidate selection, source identification, and
the exact subspace error floor; complete proofs remain in the appendices.

The experimental-design section contains a common protocol table that
distinguishes initialization reuse, success thresholds, and evaluation samples
across studies. Discussion and conclusions follow the same three claims and
retain negative results and scope limitations. This editorial revision uses
the completed frozen experiments; it adds no measurements or training runs.
The next stage is author review and revisions to this integrated draft.

Editable files are [main.tex](main.tex) and the sections in `latex/`.
The [source archive](latex_source.zip) contains the current manuscript,
bibliography, tables, and figures for Overleaf or local compilation.

## Robustness and two-layer studies completed: 26 October 2026

The [English paper](main.pdf) now includes both follow-ups, an exact
irreducible-error proposition for approximate tying, and its proof.
See the [protocols](ROBUSTNESS_AND_TWOLAYER_PROTOCOL.md),
[complete results](results/ROBUSTNESS_TWOLAYER_RESULTS.md), and
[JSON summary](results/robustness_twolayer_summary.json).

The studies produced 10,848 returned models across 168 task/source-bank
contexts. Each condition contains 96 searches per method, using three
source banks. Main findings:

- Approximate target structure: at residual norm one, inferred components
  followed by unrestricted fine-tuning succeed in 46/96 searches (47.9%);
  hard tying and dense Adam succeed in none. The derived error floor
  explains the hard-tying limitation. At residual norm two, performance
  deteriorates even with the oracle base assignment.
- Source corruption: at noise level 0.5, VAE plus release succeeds in
  44/96 searches (45.8%), versus 16/96 (16.7%) for inferred components.
  The dependency extractor fragments its support from 16 to 4,096--131,072
  combinations. The VAE has greater exact-assignment proposal coverage.
  This comparison does not establish superiority over all noise-aware priors.
- A realizable two-layer tanh network is a negative control: dense Adam
  and numerical source transfer succeed in 96/96 searches, VAE in 72/96,
  inferred components in 78/96, and components plus release in 80/96.
  Structural search is beneficial only under appropriate optimization and
  coverage conditions; an additional layer does not ensure an advantage.

All saved candidate selections and reported success counts were audited.
Compact raw results, source snapshots, and prior checkpoints are exported to
`runs/robustness_v1/` and `runs/twolayer_v1/`. The excluded aborted robustness
run had a JSON filename error; the numerical protocol was unchanged.
Three source-bank shards avoid counting duplicate work. The coverage
diagnostic is post-run analysis, stored separately in
`results/robustness_coverage_diagnostic.json`.

```bash
python3 -m paper_vae_solver.robustness_experiments --device cuda:6 --output /dev/shm/vae-robustness-new
python3 -m paper_vae_solver.twolayer_experiments --device cuda:7 --output /dev/shm/vae-twolayer-new
MPLCONFIGDIR=/tmp/paper_vae_solver_matplotlib python3 -m paper_vae_solver.robustness_twolayer_report
```

## Follow-up completed: 26 October 2026

The [updated English paper](main.pdf) includes a hidden-grouping study and a
source-identifiability proposition with proof. See the
[full follow-up results](results/HIDDEN_STRUCTURE_RESULTS.md) and
[frozen protocol](HIDDEN_STRUCTURE_PROTOCOL.md).

The study completed 132 task contexts and 20,988 returned solutions across
16 methods, using three source-bank replications. At 32 weights, inferred
components achieve 152/288 successes (52.8%), the full-vector VAE 111/288
(38.5%), and tuned dense Adam 3/288 (1.0%). A specialized non-gradient sine
solver over inferred support succeeds on all 24 evaluated contexts at both
32 and 48 weights. Its additional model assumptions and different resource
accounting are explicit. The two-block source family is non-identifying;
breaking source-target coordinate correspondence eliminates observed transfer
success. VAE-specific superiority and general structure-discovery scalability
remain unestablished.

Compact raw artifacts, checkpoints, source snapshots, and immutable plans are
in `runs/hidden_structure_v1/`; the JSON summary is
`results/hidden_structure_summary.json`. The results were audited once against
the saved candidate selection indices and population risks. Original banks
were reused, and subset sizes do not imply reduced source acquisition cost.

```bash
python3 -m paper_vae_solver.hidden_structure_experiments --blocks 2 4 --device cuda:6 --output /dev/shm/udeneev-av-hidden-structure-v1-j2j4
python3 -m paper_vae_solver.hidden_structure_experiments --blocks 6 --device cuda:7 --output /dev/shm/udeneev-av-hidden-structure-v1-j6
MPLCONFIGDIR=/tmp/paper_vae_solver_matplotlib python3 -m paper_vae_solver.hidden_structure_report
```

Исследовательский макет, 25 октября 2026. Основной объект — алгоритм
«предложить структуру → обучить её непрерывные параметры → выбрать → повторно
обучить». VAE служит одной из заменяемых реализаций генератора. Такое
позиционирование согласовано с пользователем; превосходство VAE над простыми
prior не является необходимой гипотезой работы.

## Читать

- [Статья на научном английском — main.pdf](main.pdf): основной текст, результаты и полные доказательства в приложениях.
- [LaTeX-исходник](main.tex) и [инструкция сборки](BUILD.md); [архив для Overleaf](latex_source.zip).
- [Макет статьи](paper.md): постановка, алгоритм, основные результаты и границы выводов.
- [Теория с доказательствами](theory.md): 12 утверждений, включая конкретное разделение с dense population GD.
- [Расширение на minibatch SGD](theory_sgd.md): четыре утверждения с явными достаточными условиями на шум и размер batch.
- [Переход от масок к codebook × indices](codebook.md): четыре утверждения, включая общее вероятностное разделение для масок и codebook, роль нормировки и ограничения представимости.
- [Результаты масок](results/RESULTS.md): семь блоков завершённых экспериментов, таблицы и графики.
- [Результаты codebook](results/CODEBOOK_RESULTS.md): 14 методов, три размерности и девять независимо обученных source-банков.
- [Протокол и воспроизводимость](PROTOCOL.md): бюджеты, данные, выбор настроек, исходники и ограничения измерений.
- [Связанные работы](references.md): проверенные первичные источники и отношение к нашей постановке.

## Состояние

Завершены: сопоставление 11 методов, масштабирование до 64 входов,
распределение бюджета, три независимых банка и новых VAE, population-GD опыт
для теории, контроль зависимых частей, отрицательный tanh-контроль.

Основное сравнение: VAE 259/384, product 251/384, uniform по известным ролям
269/384, перебор структур 311/384, dense Adam 103/384, dense SGD 6/384.
Независимые банки воспроизводят преимущество структурного поиска; VAE
не превосходит простые генераторы. На tanh direct SGD сильнее.

Завершён и codebook-опыт: source-решения обучаются dense SGD на простых
задачах, из них восстанавливаются назначения; target требует новых сочетаний
блоков и более высоких частот. При 32 весах VAE даёт 377/768 успешных поисков
(49.1%), empirical blocks — 358/768 (46.6%), dense Adam — 9/768 (1.2%).
При правильных индексах и новых случайных значениях codebook успех 100%;
перенос старых значений при тех же индексах даёт 0%. При 48 весах VAE и
empirical blocks достигают 12.8% и 17.4%: ограничением становится поиск
и выбор комбинации. Преимущество VAE перед простым block prior не установлено.

Теорема для гладкого одночастотного population-прототипа доказана при явных
условиях. Теорема C4 объединяет маски и codebook: при ограниченной внутренней
размерности и ненулевом покрытии подходящих подпространств вероятность
успеха остаётся положительной независимо от исходного числа весов, тогда
как upper bound для dense GD с фиксированной покоординатной дисперсией
убывает с размерностью. Расширение на minibatch SGD требует проекции амплитуды и достаточно
большого batch. Эти условия не сертифицированы для фактических экспериментов
с batch=64 и без проекции; теорию и эмпирические результаты следует различать.

## Запуск

Команды выполняются из корня репозитория. Нужны Python, NumPy, PyTorch с CUDA
и Matplotlib; здесь использовался `/home/udeneev-av/myvenv/bin/python3`,
PyTorch 2.6.0+cu118 и V100. GPU-нумерацию нужно выбирать по доступности.

```bash
python3 -m paper_vae_solver.experiments matched --device cuda:0 --output paper_vae_solver/runs/matched_v2
python3 -m paper_vae_solver.experiments scaling --device cuda:2
python3 -m paper_vae_solver.experiments allocation --device cuda:5
python3 -m paper_vae_solver.replication --output paper_vae_solver/runs/replication_v2
python3 -m paper_vae_solver.population
python3 -m paper_vae_solver.correlation
python3 -m paper_vae_solver.tanh_control
MPLCONFIGDIR=/tmp/paper_vae_solver_matplotlib python3 -m paper_vae_solver.report
```

Codebook-серия имеет отдельный `codebook_protocol.json`. Выполненные команды
использовали свободные GPU 6 и 7 и временное хранилище `/dev/shm`:

```bash
python3 -m paper_vae_solver.codebook_experiments --blocks 2 --device cuda:6 --output /dev/shm/udeneev-av-paper-vae-solver-codebook-v1-j2
python3 -m paper_vae_solver.codebook_experiments --blocks 4 6 --device cuda:7 --output /dev/shm/udeneev-av-paper-vae-solver-codebook-v1-j4j6
MPLCONFIGDIR=/tmp/paper_vae_solver_matplotlib python3 -m paper_vae_solver.codebook_report --sources /dev/shm/udeneev-av-paper-vae-solver-codebook-v1-j2 /dev/shm/udeneev-av-paper-vae-solver-codebook-v1-j4j6
```

Последняя команда экспортирует компактные данные в `runs/codebook_v1`.
После очистки временного хранилища отчёт можно пересоздать из этой копии:

```bash
MPLCONFIGDIR=/tmp/paper_vae_solver_matplotlib python3 -m paper_vae_solver.codebook_report
```

Завершённые вычислительные запуски по умолчанию возвращаются без повторного
обучения. Для новой версии основного опыта укажите новый `--output`. У
replication есть `--bank-source`: он позволяет оценить новую схему поиска на
сохранённых банках, не собирать и не обучать их заново. В выполненном v2
использован именно этот режим после исправления контроля перебора.

`report` только читает результаты и пересоздаёт таблицы/графики. Скрипты
population/correlation/tanh_control используют указанные внутри GPU и пути;
изменённый протокол следует сохранять в новой версии каталога запуска.

## Файлы

`protocol.json` фиксирует общий дизайн; `runs/*/plan.json` — фактический
протокол и происхождение каждого запуска. Сырые кандидаты и выбранные решения
сохранены в NPZ; независимые банки и checkpoints находятся в
`runs/replication_v2/bank_*`. В `results/summary.json` — машинные сводки,
в `results/metrics.csv` — плоская таблица, в `results/figures` — рисунки.
Codebook-данные хранятся отдельно в `runs/codebook_v1`, сводки — в
`results/codebook_summary.json` и `results/codebook_metrics.csv`. В копию
входят raw NPZ, source-банки, checkpoints генераторов, планы и исходники;
временные файлы для чтения опубликованных результатов больше не нужны.

Исходники каждого опыта сохранены рядом с его результатами. Первые версии
matched и replication содержат документированную ошибку повторной инициализации
в enumeration; для выводов используются только v2. Остальные методы от
исправления не изменились. Все прежние результаты `sine_mask_vae` сохранены.

Software-тесты не создавались и не запускались. Выполнены сами научные
эксперименты, минимальная статическая проверка новых Python-файлов и просмотр
экспортированных рисунков. Basic Memory на Mac была недоступна через MCP;
актуальное состояние исследования сохранено в этой папке.

При итоговой сверке 25 октября 2026 числа основных mask-опытов и независимых
банков повторно получены из raw NPZ. Для всех 24 192 codebook-поисков
проверено соответствие сохранённого решения минимуму empirical selection
loss; все 42 группы «метод × размерность» совпали с опубликованными числами.
Это проверка происхождения результатов, без нового обучения.
