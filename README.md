# Learning Transferable Network Structure

Извлекаем общую связность и группы равных весов из решений нескольких
задач. На новой задаче структура фиксируется, а кодбук, сдвиги и свободный
выходной слой обучаются заново. Основной метод — совместная калиброванная
проекция G34 с функциональным выбором индексатора.

Состояние на 12 ноября 2026 года:

- [Статья](paper/main.tex) и [PDF](paper/main.pdf): метод, основные
  теоретические результаты и доказательства в приложении.
- [Основное сравнение](paper/BENCHMARK_REPORT.md) на Pattern 0110,
  синтетическом DeepSets, Aug-Omniglot и парных суммах MNIST.
- [Компактные данные основного сравнения](artifacts/benchmark_2026-11-07/README.md).
- [Контроль упрощения G34](artifacts/anchor_start_2026-11-10/README.md):
  полный поиск восстанавливает 15/24 индексаторов, упрощения — 11/24.
- [Взвешенные сигнатуры и проверка предпосылок](artifacts/theorem_pair_recovery_weighted_2026-11-12/CAREFUL_CHECK.md).
- [Сигнатуры на Omniglot](artifacts/omniglot_signature_2026-11-12/REVIEW.md)
  и [общей синтетике](artifacts/general_signature_2026-11-12/REVIEW.md).
  Эти исследования не подтвердили общую замену G34.
- [Теория](paper/theory/README.md).

## Рабочие каталоги

| Каталог | Назначение |
|---|---|
| `paper/` | Статья, таблицы, рисунки и выбранные индексаторы |
| `experiments/paper_structure_benchmark_v1/` | Данные, G34 и конкуренты, численная адаптация и аудит |
| `experiments/sharing_structure_research_v1/` | Зарегистрированное ядро G34 с сохранёнными контрольными суммами |
| `experiments/theorem_pair_recovery_v1/` | Сигнатуры пространства контрастов и диагностика теоремы |
| `experiments/omniglot_signature_v1/` | Проверка первого геометрического кандидата на Omniglot |
| `experiments/general_signature_v1/` | Перенос на новые численные задачи общей синтетики |
| `artifacts/` | Компактные результаты, предсказания, ответы и выбранные состояния |
| `archive/` | Исторические отчёты и журнал очистки |

Старые экспериментальные деревья, тяжёлые ансамбли, промежуточные
кэши и резервные архивы удалены при [уборке 12 ноября](archive/2026-11-12/README.md).
Точные результаты статьи и новых проверок сохранены. Границы воспроизведения
описаны в [README артефактов](artifacts/README.md). Прежний исследовательский
код доступен в истории Git; рабочие методы не зависят от архивных пакетов.

## Аудит без обучения

Python 3.12 и зависимости из `requirements-reporting.txt`:

```bash
python -m pip install -r requirements-reporting.txt
python -m experiments.paper_structure_benchmark_v1.reporting.evaluate \
  --base artifacts/benchmark_2026-11-07 --phase published-score
python -m experiments.paper_structure_benchmark_v1.reporting.anchor_audit \
  --base artifacts/anchor_start_2026-11-10 --output /tmp/ibr-anchor-audit
python -m experiments.paper_structure_benchmark_v1.reporting.signature_audit \
  --base artifacts/general_signature_2026-11-12 --dataset general \
  --output /tmp/ibr-general-audit.json
python -m experiments.paper_structure_benchmark_v1.reporting.signature_audit \
  --base artifacts/omniglot_signature_2026-11-12 --dataset omniglot \
  --output /tmp/ibr-omniglot-audit.json
```

Аудит не обучает модели и не требует GPU. Зависимости обучения находятся
в `requirements.txt`; команды запуска — в [README кода](experiments/README.md).
Новый запуск создаёт отдельные каталоги, не перезаписывая завершённые серии.

## Сборка статьи

```bash
make -C paper
./pack_overleaf.sh
```

Нужны TeX и `latexmk`. Историческое сравнение конкурентов и новый
Omniglot-прогон разделены в статье: итоговые эпизоды в них различаются.
