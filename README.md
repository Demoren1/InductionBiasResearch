# Learning Transferable Network Structure

Извлекаем общую связность и группы равных весов из решений нескольких
задач. На новой задаче структура фиксируется, а кодбук, сдвиги и свободный
выходной слой обучаются заново. Текущий метод — совместная калиброванная
проекция с последующим функциональным выбором индексатора.

Состояние на 7 ноября 2026 года:

- [Рукопись](paper/main.tex) и [PDF](paper/main.pdf).
- [Результаты четырёх наборов и ограничения](paper/BENCHMARK_REPORT.md).
- [Рабочий код и команды](experiments/paper_structure_benchmark_v1/README.md).
- [Компактные данные для независимого аудита](artifacts/benchmark_2026-11-07/README.md).
- [История исследований и очистки](archive/README.md).

## Рабочие каталоги

| Каталог | Назначение |
|---|---|
| `paper/` | Статья, итоговые таблицы, рисунки и выбранные индексаторы |
| `experiments/paper_structure_benchmark_v1/` | Подготовка задач, методы, подгонка, запуск и отчёты |
| `experiments/sharing_structure_research_v1/` | Зарегистрированное исходное ядро G34; сохранено без изменений |
| `artifacts/benchmark_2026-11-07/` | Точные предсказания, ответы для аудита, выбранные структуры и протоколы |
| `archive/` | Предыдущие эксперименты, заметки, отчёты и снимок исходников до уборки |

Полные исходные данные и состояния запусков остаются локально в
`experiments/paper_structure_benchmark_v1/results/20261107/` и
`artifacts/local/`. Они исключены из новых коммитов. Опубликованный компактный
комплект позволяет пересчитать итоговые метрики и пересобрать все рисунки
статьи без этих каталогов.

## Пересчёт результатов без обучения

Использовано окружение Python 3.12. Для аудита нужны пакеты из
`requirements-reporting.txt`:

```bash
python -m pip install -r requirements-reporting.txt
python -m experiments.paper_structure_benchmark_v1.reporting.evaluate \
  --base artifacts/benchmark_2026-11-07 --phase published-score
MPLCONFIGDIR=/tmp/ibr-mpl-cache \
  python -m experiments.paper_structure_benchmark_v1.reporting.render_report \
  --base artifacts/benchmark_2026-11-07 --paper paper
make -C paper
```

Для последней команды необходимы TeX и `latexmk`. GPU и повторное обучение
для аудита и сборки статьи не нужны. Полные зависимости обучения приведены
в `requirements.txt`; использованное CUDA-окружение описано в README кода.

Сравнение включает Pattern 0110, синтетический DeepSets, Aug-Omniglot и
уменьшенную multitask-задачу сумм пар MNIST. Это общий полносвязный стенд;
границы адаптаций Yeh, MSR и LUT-Q описаны в отчёте. Статья сохраняет
пометки `in progress` для ещё не завершённой теоретической части.
