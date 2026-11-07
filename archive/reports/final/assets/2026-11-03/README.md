# Рисунки итогового отчёта от 3 ноября

[Отчёт](../../FINAL_REPORT_2026-11-03.md). SVG и PNG содержат одинаковые
рисунки; SVG сохраняют векторный текст и линии.

| Файлы | Содержание и происхождение |
|---|---|
| `method.tex`, `.pdf`, `.svg`, `.png` | TikZ/LaTeX-схема текущей процедуры |
| `pattern_weights.*`, `pattern_alignment.*`, `pattern_indexators.*` | Сохранённые query-selected веса шаблона 0010; только перестановки нейронов и имён кодов |
| `pattern_heatmap_data.npz` | Исходные и согласованные матрицы, перестановки и индексаторы |
| `pattern_learning.*` | Истории четырёх выбранных стартов Pattern |
| `sine_landscape.*` | Аналитическая Gaussian-MSE и сохранённые стационарные точки исходной функции |
| `sine_results.*`, `sine_summary.json` | Исторический matched_v2 с partwise VAE; не новый AR-опыт |
| `sine_summary_origin.json` | Git-ревизия, исходный путь и SHA-256 исторической сводки |
| `sine_current_results.*`, `sine_current_structures.*` | Новый опыт с текущим AR: все методы и распределения предложений |
| `sine_current_learning.*`, `sine_current_prior.*` | Реальные train/population истории выбранных моделей и обучение MADE |
| `sine_current_curves.csv`, `sine_current_sources.json` | Новые кривые и SHA-256 артефактов запуска |
| `dsd_example.*` | Новая иллюстративная реализация шума с сохранённым сигналом; не исходный обучающий пример |
| `dsd_results.*`, `dsd_learning.*` | Метрики и истории пяти независимых банков Dynamic Signal |
| `dsd_indexator_analytic.*` | Аналитический индексатор: наблюдаемый порядок, группировка входов и увеличенный блок 8 × 2 |
| `dsd_indexators_obtained.*` | Опорные и AR-индексаторы пяти банков, карты пропущенных и лишних связей и неверных кодов |
| `dsd_indexators_data.npz`, `dsd_indexators_recovery.csv`, `dsd_indexators_sources.json` | Все 160 выбранных матриц до и после согласования, ошибки по задачам и SHA-256 источников |
| `report_data.json`, `learning_curves.csv`, `sources.json` | Числа и источники Pattern, исторических синусов и Dynamic Signal |

Пересобрать рисунки из имеющихся результатов, из корня репозитория:

```bash
MPLCONFIGDIR=/tmp/induction-report-mpl python final_mds/assets/2026-11-03/build_figures.py
MPLCONFIGDIR=/tmp/induction-report-mpl python -m experiments.sine_ar_current_v1.report
MPLCONFIGDIR=/tmp/induction-report-mpl python -m experiments.dynamic_signal_v1.indexator_figures
```

Эти команды не обучают сети. Новый эксперимент, его протокол и исходные
результаты находятся в `experiments/sine_ar_current_v1/`.

Схема компилируется `pdflatex -output-directory=/tmp method.tex` из этой
папки. PDF преобразован в SVG через `PyMuPDF 1.24.14` методом
`Document[0].get_svg_image()`; PNG — `get_pixmap(matrix=Matrix(2, 2))`.
TeX-формулы остаются векторными контурами, внешние шрифты SVG не требуются.

В новом sine-опыте сохранён точный исполнявшийся `run_source.py` и
`made_source.py`. Рабочие Python-файлы после запуска только отформатированы
по PEP 8; исходный снимок сохраняет первоначальное содержимое.
# Дополнение: случайный поиск структуры

`random_structure_search/search_and_transfer.svg` и `.png` показывают
64 поколения случайного структурного поиска и независимое сравнение на
16 новых задачах. `random_structure_search/obtained_masks.svg` и `.png`
показывают найденную маску, частоты банка, выбранное предложение генератора
и случайный контроль. Порядок входов наблюдаемый; teacher не используется.

Подпись AR обозначает ветку эксперимента: в этом запуске validation
выбрала checkpoint MADE на шаге 0, фактически независимые частоты.
Полезного переноса структуры не обнаружено. Числа и ограничения —
[в отдельном отчёте](../../RANDOM_STRUCTURE_SEARCH_2026-11-03.md).
Скрипт — `experiments/random_structure_search_v1/report.py`, данные,
код замороженного запуска и SHA-256 —
`experiments/random_structure_search_v1/results/search_20261103/`.
