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
| `report_data.json`, `learning_curves.csv`, `sources.json` | Числа и источники Pattern, исторических синусов и Dynamic Signal |

Пересобрать рисунки из имеющихся результатов, из корня репозитория:

```bash
MPLCONFIGDIR=/tmp/induction-report-mpl python final_mds/assets/2026-11-03/build_figures.py
MPLCONFIGDIR=/tmp/induction-report-mpl python -m experiments.sine_ar_current_v1.report
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
