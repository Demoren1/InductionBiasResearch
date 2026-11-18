# Индексатор по одной source-задаче

Прямое сравнение отношений весов на двух чистых general/random структурах.
Условия и ограничения описаны в [протоколе](PROTOCOL.md).
Результат сохраняется в `artifacts/single_task_recovery_2026-11-14/` и
`paper/theory/SINGLE_TASK_RECOVERY_EXPERIMENT_2026-11-14.md`.

| Модуль | Действие |
|---|---|
| `prepare` | Новые численные source-задачи и наблюдения; истина в private |
| `controls` | Отдельный oracle-контроль масштабов и ограниченного log-шума |
| `source --phase source` | Неизменённая dense-подготовка одной задачи |
| `source --phase g34` | Неизменённый G34 на том же входе |
| `learner --phase extract` | Банк из прямых разностей log-модулей |
| `learner --phase refit` | Численная подгонка и отдельный source-отбор |
| `audit` | Структурная истина открывается после обеих source selections |
| `transfer --phase prepare` | Четыре новые target-функции на геометрию |
| `transfer --phase fit` | Исходный adapter: 18 Adam/2000 + 4 GN/48 |
| `transfer --phase score` | Все predictions замораживаются до чтения ответов |
| `plots` | Графики PNG/PDF по сохранённым результатам без повторного обучения |

Каждый вызов использует `--root` с одним новым каталогом. Вызовы source,
learner и transfer fit принимают `--case case_04` либо `case_05`.
Source, learner refit и transfer fit используют доступный GPU через
`--device cuda:0`; для параллельного запуска задайте разные
`CUDA_VISIBLE_DEVICES`. В исходном запуске использовались GPU 0 и 1.
Среда — `/home/udeneev-av/myvenv/bin/python`, с
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2`.

Пример начала нового запуска из корня репозитория:

```bash
/home/udeneev-av/myvenv/bin/python -m experiments.single_task_recovery_v1.prepare \
  --root artifacts/local/single_task_recovery_new
```

После подготовки выполняются controls и source-подготовка для обеих
структур, затем G34, ratio extract и ratio refit для обеих структур.
После их завершения выполняются audit, transfer prepare, transfer fit
для обеих структур и transfer score. Готовые каталоги извлечения
и численной target-подгонки не перезаписываются.

Код не оценивает пространство контрастов по SVD и не достраивает неполные
группы произвольным кластеризатором. Если слияния не дали полного
индексатора, это фиксируется как отсутствие кандидата. Поэтому этот
опыт проверяет конструктивное правило повторяющихся пар и его небольшой
proposal-банк; общий поиск продолжений через длинные циклы не реализован.

В исходном запуске перед приведением кода к PEP 8 сохранён исполнявшийся
снимок в `code_snapshot/executed/` вместе с SHA-256. Форматирование,
упорядочивание импортов и сокращение длинных строк не меняли расчёты.
Software tests не создавались и не запускались.

Построение графиков завершённого опыта:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/ibr-mpl-cache \
  /home/udeneev-av/myvenv/bin/python -m experiments.single_task_recovery_v1.plots \
  --root artifacts/single_task_recovery_2026-11-14
```
