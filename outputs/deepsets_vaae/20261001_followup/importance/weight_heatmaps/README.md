# Heatmaps обученных весов

Пример выбран заранее: seed 4100, target task 0, бюджет 256 наборов, init 0. Цвет — эффективный подписанный вес W×M; неактивные связи равны точно нулю. Колонки сохранены в исходном порядке. Во всех направлениях одна цветовая шкала.

![dense: dense comparison](dense_vs_dense_weights.png)
Матрицы 784×32: строки — входные пиксели, колонки — скрытые нейроны. Слева dense, справа dense. Это один парный пример, не средняя карта.

![dense: filters](dense_filters.png)
Каждый квадрат — одна колонка в геометрии 28×28. Показывает расположение эффективных связей; без функциональной проверки не доказывает полезность фильтра.

![agreement: dense comparison](dense_vs_agreement_weights.png)
Матрицы 784×32: строки — входные пиксели, колонки — скрытые нейроны. Слева dense, справа agreement. Это один парный пример, не средняя карта.

![agreement: filters](agreement_filters.png)
Каждый квадрат — одна колонка в геометрии 28×28. Показывает расположение эффективных связей; без функциональной проверки не доказывает полезность фильтра.

![importance_activity_weighted: dense comparison](dense_vs_importance_activity_weighted_weights.png)
Матрицы 784×32: строки — входные пиксели, колонки — скрытые нейроны. Слева dense, справа importance_activity_weighted. Это один парный пример, не средняя карта.

![importance_activity_weighted: filters](importance_activity_weighted_filters.png)
Каждый квадрат — одна колонка в геометрии 28×28. Показывает расположение эффективных связей; без функциональной проверки не доказывает полезность фильтра.

![importance_function_gradient: dense comparison](dense_vs_importance_function_gradient_weights.png)
Матрицы 784×32: строки — входные пиксели, колонки — скрытые нейроны. Слева dense, справа importance_function_gradient. Это один парный пример, не средняя карта.

![importance_function_gradient: filters](importance_function_gradient_filters.png)
Каждый квадрат — одна колонка в геометрии 28×28. Показывает расположение эффективных связей; без функциональной проверки не доказывает полезность фильтра.

![importance_raw_abs_weight: dense comparison](dense_vs_importance_raw_abs_weight_weights.png)
Матрицы 784×32: строки — входные пиксели, колонки — скрытые нейроны. Слева dense, справа importance_raw_abs_weight. Это один парный пример, не средняя карта.

![importance_raw_abs_weight: filters](importance_raw_abs_weight_filters.png)
Каждый квадрат — одна колонка в геометрии 28×28. Показывает расположение эффективных связей; без функциональной проверки не доказывает полезность фильтра.

![importance_train_deletion_positive: dense comparison](dense_vs_importance_train_deletion_positive_weights.png)
Матрицы 784×32: строки — входные пиксели, колонки — скрытые нейроны. Слева dense, справа importance_train_deletion_positive. Это один парный пример, не средняя карта.

![importance_train_deletion_positive: filters](importance_train_deletion_positive_filters.png)
Каждый квадрат — одна колонка в геометрии 28×28. Показывает расположение эффективных связей; без функциональной проверки не доказывает полезность фильтра.
