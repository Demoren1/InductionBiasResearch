# G34: fixed mask против free mask

> Архивировано 8 ноября 2026. Сырые results/ удалены. [Сохранённый аудит и восстановление](../../2026-11-08/README.md). Ниже — исторические команды: перед новым обучением нужно вернуть пакеты в experiments/ и заново подготовить данные.

В обоих вариантах сохраняются специальный ноль и четыре обучаемых
знаковых кода. Free снимает фиксацию бинарного support внутри joint
projection: любая позиция выбирает из 0 и ±1,…,±4. Численный solver
и остальные этапы исходного G34 сохранены.

Актуальная постановка — ACTIVE_PROTOCOL.md. Исходный PROTOCOL.md
содержал лишние контроли, отменённые после уточнения пользователя;
их результаты не входят в текущее сравнение. Для актуального запуска
использовать resume.py, а не исходный run.py --phase complete.

Новый запуск из корня репозитория:

```bash
/home/udeneev-av/myvenv/bin/python -m experiments.g34_support_v1.resume \
  --phase setup --base /tmp/g34-mask-new \
  --reference /home/udeneev-av/InductionBiasResearch/experiments/g34_theory_v1/results/20261108_general/confirmation
/home/udeneev-av/myvenv/bin/python -m experiments.g34_support_v1.resume \
  --phase complete --base /tmp/g34-mask-new
```

Продолжение использует GPU0–7, по одному внешнему заданию на карту,
сначала выполняет12 source jobs, затем12 target jobs. Перед открытием
query responses и teacher I фиксируются все выбранные структуры и
predictions. Target числа свежие, одинаковые данные/бюджеты.

THEORY_NOTES.md объясняет объединение фиксированных масок, нулевой
assignment, условное margin-восстановление, ошибки порогов и ограничения
идентифицируемости. Main report: G34_SUPPORT_MASK_REPORT_2026-11-08.md
в корне проекта. Raw: results/20261108; compact artifacts:
artifacts/g34_support_2026-11-08.
