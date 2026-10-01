# Перестроенный функциональный банк DeepSets

## Результат

Основной source-audit — точная эмпирическая iid-population метрика по 3000 audit-изображениям. Средний NMSE sparse решений равен 0.231809, paired dense controls — 0.204134; sparse лучше в 448 из 10240 matched сравнений task/replica. В банке восемь seed, четыре фиксированных source cost-вектора из одной synthetic task family, 10240 sparse fit-решений и 256 dense-control решений.

$$
\mathbb{E}\left[\frac{(\sum_{i=1}^5 e_i)^2}{5}\right]=\mathbb{E}[e^2]+4\mathbb{E}[e]^2
$$

Это source-bank сравнение. Известные индивидуальные метки $c[\mathrm{digit}]$ доступны при построении source teachers; они не передаются target-задаче, где остаются только наблюдаемые метки наборов. Четыре cost-вектора из одной synthetic family не являются четырьмя независимыми seed или четырьмя families.

## Топология против fit-решений

В 10240 sparse fit-строк найдено 6666 точных бинарных mask-топологий и 10240 уникальных точных эффективных состояний $(W\odot M,b,a,o)$. Идентичная support-маска поэтому не означает идентичный fit. У dense controls 256 отдельных fit-решений используют одну topology all-on и 256 эффективных состояний. Train/held-out teacher split — 256/64 в каждой task×seed ячейке; 24 ячейки имеют повтор маски через split, но exact effective-state дубликатов нет. Held-out teachers исключены из pooled context; held-out reconstruction для обученного генератора здесь не измерялся.

## Exact и Monte Carlo audit

Вторичный Monte Carlo audit использует 2048 сэмплированных iid-наборов по 5 изображений. Replay сохраненных индексов дает максимальное отличие от сохраненных MC значений 0.0; это проверяет replay, но exact audit остается главным. Solver provenance разделяет первоначальное stochastic v3 предложение, exact-population pilots с варьированием L2 при предложенном lr=0.002 на двух meta-validation cost vectors и actual production snapshot. Все 10240/10240 sparse и 256/256 dense fits получили plateau-флаг.

Полный отчет, методика, caveats, density/recipe таблицы и контекстные фигуры: [RESULTS_RU.md](../../../outputs/deepsets_vaae/20261001_rebuilt_functional_bank/RESULTS_RU.md).

![audit_nmse_by_density.png](../../../outputs/deepsets_vaae/20261001_rebuilt_functional_bank/figures/audit_nmse_by_density.png)

По X — доля активных связей; по Y — точный ожидаемый NMSE для iid-наборов из пяти элементов на 3000 audit-изображениях. Панели разделяют random и dense-functional маски; пунктир показывает dense. Точки — средние по восьми seed, интервалы описательные. Средние sparse-оценки уступают dense при всех проверенных плотностях.

![paired_dense_gaps.png](../../../outputs/deepsets_vaae/20261001_rebuilt_functional_bank/figures/paired_dense_gaps.png)

По X — плотность и способ построения маски; по Y — exact sparse NMSE минус paired dense NMSE. В каждом box — восемь seed-средних по четырём исходным задачам. Отрицательная разность благоприятна sparse; это не распределение тысяч независимых задач.

![representative_masks_and_signed_weights.png](../../../outputs/deepsets_vaae/20261001_rebuilt_functional_bank/figures/representative_masks_and_signed_weights.png)

Пример seed 4100, task 0, replica 0, dense-functional variant 0. По X — пиксель 0–783, по Y — hidden-нейрон 0–31. Слева binary mask: белый — 0, чёрный — 1. Справа реальные signed $W\odot M$ на общей симметричной шкале: красный — положительный вес, синий — отрицательный. Это отдельные обученные состояния, не усреднение.

![monte_carlo_vs_exact_audit.png](../../../outputs/deepsets_vaae/20261001_rebuilt_functional_bank/figures/monte_carlo_vs_exact_audit.png)

По X — точный ожидаемый audit NMSE; по Y — вторичная Monte Carlo оценка по 2048 iid-наборам из пяти элементов. Панели показывают sparse и paired dense; цвет обозначает плотность. Пунктир $y=x$ показывает совпадение оценок. Разброс отражает конечное число сэмплированных наборов.

![source_population_training_curves.png](../../../outputs/deepsets_vaae/20261001_rebuilt_functional_bank/figures/source_population_training_curves.png)

По X — шаг Adam. Слева — точный source NMSE без L2, справа — NMSE на 512 source-validation наборах. Линии — средние по восьми seed для dense и sparse, полосы — межseed стандартное отклонение. Plateau проверяется отдельно по fit-цели с L2; query остаётся диагностикой.

![psi_two_teachers.png](../../../outputs/deepsets_vaae/20261001_rebuilt_functional_bank/seed_4100/figures/psi_two_teachers.png)

Seed 4100, task 0: по X — 128 общих probe-изображений, по Y — aligned hidden-нейрон. Цвет показывает signed вклад $\psi_j(x)$ на общей симметричной шкале. Показаны train и held-out учителя; held-out исключён из context. Это полные профили вкладов, без дополнительного отсечения значений.

![q_signed_mean.png](../../../outputs/deepsets_vaae/20261001_rebuilt_functional_bank/seed_4100/figures/q_signed_mean.png)

Seed 4100: по X — пиксель, по Y — aligned hidden-нейрон. Цвет — средний signed $q_{ij}(x)$ на probe-выборке; шкала симметрична относительно нуля. Знак различает положительный и отрицательный вклад связи.

![q_abs_mean.png](../../../outputs/deepsets_vaae/20261001_rebuilt_functional_bank/seed_4100/figures/q_abs_mean.png)

Seed 4100: по X — пиксель, по Y — aligned hidden-нейрон. Цвет — средний абсолютный $q_{ij}(x)$ на probe-выборке. Это производная статистика полного функционального представления; бинарная маска хранится отдельно.
