# Cooperative generator/evaluator

Основной алгоритм и его определения описаны в [best_algo.md](../mds/best_algo.md).
Запускатели из корня проекта: [pattern.sh](../sh_scripts/pattern.sh) и
[deepsets.sh](../sh_scripts/deepsets.sh). Оба запускают bootstrap, затем search;
набор аргументов данных сохраняется между фазами.

В каждой обучающей задаче есть свой Transformer-генератор и functional bank.
Общий evaluator обучается один раз на cross-task fresh fits начальных масок,
происходящих из functional banks, затем замораживается на bootstrap и search.
По умолчанию выход каждой генерации имеет фиксированный бюджет $K$. Один joint
update минимизирует сумму трёх членов: frozen evaluator cost в собственном контексте,
попарное agreement генераторов и distillation к лучшим реальным маскам train-архива:

$$
L = L_{\mathrm{quality}} + \lambda L_{\mathrm{agreement}} + \mu L_{\mathrm{distill}}.
$$

Agreement и distillation — MSE между hard-масками после Hungarian alignment; для
градиента используется straight-through маска. По умолчанию $\lambda=0.1$ и
$\mu=0.1$. Distillation рассматривает все верхние подходящие маски архива,
измеренные на train-задачах; маска не обязана уже превосходить dense. Joint
objective не включает отдельные BCE или reconstruction шаги. Reconstruction
pretraining по умолчанию выключен (`GENERATOR_PRETRAIN_EPOCHS=0`).

На refresh поиск объединяет предложения всех генераторов, случайные маски,
буквальный top real archive и его edge-swap мутации. Evaluator ранжирует весь пул по
общему train score. Для acquisition budget $B$ уже измеренные top-$B$ кандидаты
переиспользуют labels; дополнительно до $B$ лучших unseen масок получают новые
fits, каждый на всех train-задачах. Так кэш не расходует бюджет новых измерений.
Fits пополняют replay, реальный archive и functional banks, но не обновляют
замороженный evaluator. Итоговая маска выбирается по независимым
selection-наблюдениям. Held-out задачи открываются после фиксации frozen результата.

Начальный сбор functional maps сохранён: для каждой train-задачи строятся карты
из fitted teachers разных плотностей на её приватном probe. Source-query ошибка
используется для отбора начальных карт внутри плотности; она не служит quality
label для evaluator. Все уникальные начальные маски, включая маски карт и
control/random кандидаты, проходят реальные fits на каждой train-задаче и
попадают в replay. Evaluator один раз обучается на real cross-fits масок из
неизменяемого начального набора functional banks. Принадлежность задаётся самой
маской и не меняется при росте банков. Labels масок вне этого набора — включая
random, control и последующие acquisition candidates — сохраняются для
archive/replay и feedback генераторам, но не обучают evaluator.

## Pattern defaults

В `pattern.sh` по умолчанию обучаются паттерны
`0000 0001 0011 0101 0110 0111 1000 1001 1010 1100 1110 1111`; held-out паттерны —
`0010 0100 1011 1101`. Наборы разделены по reversal/complement-орбитам. Их можно
переопределить переменными `TRAIN_PATTERNS` и `TEST_PATTERNS`; значения одинаковы
для bootstrap и search.

Скрипт использует GPUs `0 1 2 4 5` по умолчанию, строит банки и запускает обе фазы:

```bash
GPU_IDS="0 1 2 4 5" bash sh_scripts/pattern.sh
```

Укажи `BOOTSTRAP_GENERATORS=1`, чтобы включить generator exploration в bootstrap;
скрипт передаст `--bootstrap-generators` и в bootstrap, и в search. Иначе bootstrap
обновляет functional banks и один раз обучает evaluator, без обучения генераторов.
После этого evaluator остаётся замороженным. `EVALUATOR_EPOCHS` задаёт только это
начальное обучение. В `deepsets.sh` `REFRESH_EVERY` задаёт cadence раундов новых
измерений; в `pattern.sh` он равен 2. Refresh добавляет fits и feedback, но не
обновляет evaluator. Bootstrap сохраняет policy `initial_bank_only` в checkpoint v7.
`GE_OUT`, `GE_SEED`,
`CHILD_STEPS`, `GENERATOR_EPOCHS`, `UPDATES_PER_EPOCH`, `AGREEMENT_WEIGHT` и
`ELITE_DISTILLATION_WEIGHT` позволяют задать путь, seed, бюджеты и веса. Дополнительные
аргументы командной строки передаются в search.

`deepsets.sh` использует тот же joint objective и цикл real-fit feedback. Числа
train и held-out задач задаются `TRAIN_TASKS` и `TEST_TASKS`; данные читаются из
`DATA_ROOT` (по умолчанию `datasets/mnist8m`). Для обоих скриптов `WARM_START_FROM`
позволяет начать search с подготовленного train-only bootstrap. При warm-start от
старого bootstrap с online-trained evaluator его веса переинициализируются и
evaluator обучается заново только на masks исходного functional-bank набора.
Warm-start сохраняет этот membership и frozen evaluator даже если live banks
генераторов позже пополнились feedback-картами.

## Структура пакета

Публичные команды остаются `python -m generator_evaluator.cooperative_run`,
`python -m generator_evaluator.run` и `python -m generator_evaluator.evaluate_frozen`;
эти корневые модули служат тонкими compatibility entry points. Реализация сгруппирована
по ответственности:

- `data/` содержит общие типы задач и адаптеры pattern и DeepSets данных.
- `models/` содержит Transformer-генератор.
- `training/` содержит optimizer updates, joint и staged objectives, а также
  исполнение генераторов на устройствах.
- `search/` содержит policy предложений, schedule, evaluator quality, priors и
  consensus controls.
- `evaluation/` содержит pattern и DeepSets fits, измерения, параллельное
  исполнение и frozen evaluation.
- `storage/` отвечает за artifacts, functional banks, runtime state, warm starts,
  Toeplitz controls и progress.
- `runners/` оркестрирует cooperative и legacy runs.

Pattern batching объединяется в `evaluation/pattern.py`, а cooperative measurement
dispatch — в `evaluation/parallel.py`. Поддержка совместимости со старыми bank
pickles сохраняется.
