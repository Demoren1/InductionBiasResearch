"""Generate figures and a concise evidence-based Pattern report."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

NAMES = {
    "oracle_indices": "Аналитический индексатор",
    "inferred_indices": "Извлечённый индексатор",
    "vae_sample": "VAE, без обучения z",
    "vae_z": "VAE + обучение z",
    "untrained_vae_z": "Необученный VAE + z",
    "direct_indices": "Прямое обучение индексатора",
    "random_indices": "Случайный индексатор",
    "dense_adam": "Плотный Adam",
    "dense_sgd": "Плотный SGD",
}


def main():
    root = Path(__file__).resolve().parents[2]
    run = root / "pattern/runs/codebook_20261028_v4"
    assets = root / "mds/assets/2026-10-28/pattern_codebook"
    assets.mkdir(parents=True, exist_ok=True)
    summary = []
    for method, name in NAMES.items():
        data = np.load(run / (method + ".npz"))
        query = data["query_loss"].reshape(4, 32)
        selected = query.argmin(1)
        population = data["population_accuracy"].reshape(4, 32)
        test = data["test_accuracy"].reshape(4, 32)
        row = {
            "method": method,
            "name": name,
            "perfect_starts": int((population == 1).sum()),
            "mean_population_accuracy": float(population.mean()),
            "selected_population_accuracy": population[np.arange(4), selected].tolist(),
            "selected_test_accuracy": test[np.arange(4), selected].tolist(),
            "selected_perfect_tasks": int(
                (population[np.arange(4), selected] == 1).sum()
            ),
        }
        summary.append(row)
    (run / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    with (assets / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    extracted = np.load(run / "extracted.npz")["indices"]
    delta = np.arange(11)[:, None] - np.arange(8)[None]
    truth = np.where((delta >= 0) & (delta < 4), delta + 1, 0)
    table = np.zeros((4, 4), dtype=int)
    for actual, expected in zip(extracted[truth > 0], truth[truth > 0]):
        if actual:
            table[actual - 1, expected - 1] += 1
    rows, columns = linear_sum_assignment(-table)
    mapping = np.zeros(5, dtype=int)
    mapping[rows + 1] = columns + 1
    aligned = mapping[extracted]
    bank = np.load(run / "bank.npz")
    train_rows = bank["split_ids"] == 0
    counts = {
        str(task): [
            int(((bank["task_ids"] == task) & (bank["split_ids"] == role)).sum())
            for role in range(3)
        ]
        for task in range(12)
    }
    audit = {
        "support_correct_of_32": int(((extracted > 0) & (truth > 0)).sum()),
        "tying_correct_of_32_up_to_class_names": int(table[rows, columns].sum()),
        "training_maps": int(train_rows.sum()),
        "counts_task_train_validation_test": counts,
        "minimum_training_query_accuracy": float(
            bank["query_accuracy"][train_rows].min()
        ),
        "mean_training_query_accuracy": float(
            bank["query_accuracy"][train_rows].mean()
        ),
        "maximum_bank_query_bce": float(bank["query_loss"].max()),
        "source_only_alignment_and_selection": True,
        "analytic_labels_used_for_diagnostics_only": True,
    }
    (run / "audit.json").write_text(json.dumps(audit, indent=2))
    plt.rcParams.update({"font.size": 11, "figure.dpi": 150})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    frequency = np.load(run / "extracted.npz")["support_frequency"]
    axes[0].imshow(frequency, vmin=0, vmax=1, cmap="Blues")
    axes[0].set_title("Общая частота после выравнивания")
    axes[1].imshow(truth, vmin=0, vmax=4, cmap="viridis")
    axes[1].set_title("Аналитический индексатор")
    axes[2].imshow(aligned, vmin=0, vmax=4, cmap="viridis")
    axes[2].set_title("Извлечено из решений")
    for axis in axes:
        axis.set_xlabel("Скрытый нейрон")
        axis.set_ylabel("Вход")
    fig.savefig(assets / "indexators.png")
    plt.close(fig)
    labels = [row["name"] for row in summary]
    positions = np.arange(len(summary))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    axes[0].barh(
        positions, [r["perfect_starts"] / 128 * 100 for r in summary], color="#008e88"
    )
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Безошибочные старты, %")
    axes[0].set_xlim(0, 100)
    axes[1].barh(
        positions,
        [r["mean_population_accuracy"] * 100 for r in summary],
        color="#284b75",
    )
    axes[1].set_yticks(positions, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Средняя точность одного старта, %")
    axes[1].set_xlim(0, 100)
    fig.savefig(assets / "comparison.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for method in [
        "vae_z",
        "direct_indices",
        "dense_adam",
        "dense_sgd",
        "untrained_vae_z",
    ]:
        history = json.loads((run / (method + "_history.json")).read_text())
        values = np.array([r["query_accuracy"] for r in history])
        ax.plot([r["step"] for r in history], values.mean(1) * 100, label=NAMES[method])
    ax.set_xlabel("Шаг обучения на новой задаче")
    ax.set_ylabel("Средняя точность на query, %")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.savefig(assets / "training.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    point_rows = []
    for dataset in ["MNIST", "FashionMNIST"]:
        path = (
            root
            / "experiments/structure_search_v2/runs"
            / ("pointset_pilot_" + dataset)
        )
        data = json.loads((path / "results.json").read_text())
        for method in [
            "dense_adam",
            "untied_adam",
            "deepsets_given",
            "dense_sorted_adam",
        ]:
            rows = [r for r in data if r["method"] == method]
            point_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "accuracy": float(
                        np.mean([r["validation_accuracy"] for r in rows])
                    ),
                    "parameters": rows[0]["parameters"],
                }
            )
    for i, dataset in enumerate(["MNIST", "FashionMNIST"]):
        values = [r["accuracy"] * 100 for r in point_rows if r["dataset"] == dataset]
        ax.bar(np.arange(4) + (i - 0.5) * 0.36, values, 0.36, label=dataset)
    ax.set_xticks(
        np.arange(4),
        [
            "Плотный Adam",
            "Без общих весов\nмежду точками",
            "DeepSets задан",
            "Adam после\nсортировки точек",
        ],
    )
    ax.set_ylabel("Точность на source-validation, %")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.savefig(assets / "pointset_feasibility.png")
    plt.close(fig)
    (assets / "pointset_feasibility.json").write_text(json.dumps(point_rows, indent=2))
    pilot = (
        root / "experiments/structure_search_v2/runs/scrambled_pattern_source_pilot_v1"
    )
    history = json.loads((pilot / "history.json").read_text())
    controls = json.loads((pilot / "controls_summary.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(
        [row["generation"] for row in history],
        [row["best_source_nmse"] for row in history],
        label="Эволюционный поиск",
    )
    axes[0].axhline(
        controls["frozen_random_codebooks_best_mean_nmse"],
        color="#888888",
        linestyle="--",
        label="Фиксированные случайные структуры",
    )
    axes[0].set_xlabel("Поколение")
    axes[0].set_ylabel("NMSE на source-query")
    axes[0].legend(fontsize=8)
    dense = float(np.mean(controls["dense_source_best_nmse"]))
    axes[1].bar(
        ["Случайные\nструктуры", "Эволюция", "Плотный\nAdam", "Правильная\nструктура"],
        [
            controls["frozen_random_codebooks_best_mean_nmse"],
            history[-1]["best_source_nmse"],
            dense,
            controls["oracle_codebook_best_mean_nmse"],
        ],
        color="#284b75",
    )
    axes[1].set_ylabel("NMSE на source-query ↓")
    axes[1].set_ylim(0, 1)
    fig.savefig(assets / "scrambled_source_pilot.png")
    plt.close(fig)
    table_lines = []
    for row in summary:
        selected_accuracy = np.mean(row["selected_population_accuracy"]) * 100
        test_accuracy = np.mean(row["selected_test_accuracy"]) * 100
        table_lines.append(
            f"| {row['name']} | {row['perfect_starts']}/128 | "
            f"{row['mean_population_accuracy'] * 100:.2f}% | "
            f"{selected_accuracy:.2f}% | {test_accuracy:.2f}% |"
        )
    report = r"""# Pattern: извлечение индексатора и перенос через NF+VAE

28 октября 2026 года. Завершённый эксперимент на исходном Pattern; это первый этап общей проверки трёх примеров.

## Задача и точное решение

На вход подаётся строка из 11 бит. Нужно определить, встречается ли в ней заданный шаблон из четырёх бит. Например, строка `11001011100` содержит шаблон `0010`. Всего возможны 2048 входов. В исходных задачах использовали 12 шаблонов, а для проверки переноса оставили четыре новых: `0010`, `0100`, `1011`, `1101`.

Переведём биты входа и шаблона в значения −1 и +1. Один скрытый нейрон проверяет одно из восьми положений шаблона:

$$
\begin{aligned}
x&\in\{-1,+1\}^{11},\qquad p\in\{-1,+1\}^{4},\\
h_j(x)&=\operatorname{ReLU}\!\left(\sum_{r=1}^{4}p_r x_{j+r-1}-3\right),\qquad j=1,\ldots,8,\\
y(x)&=\mathbf{1}\!\left\{\sum_{j=1}^{8}h_j(x)>0\right\}.
\end{aligned}
$$

При полном совпадении сумма четырёх произведений равна 4, поэтому отклик нейрона равен 1. Одно несовпадение уменьшает сумму до 2, и отклик становится нулевым. Это точное решение задачи. Для шаблона `0010` четыре коэффициента фильтра равны −1, −1, +1, −1.

## Что переносим

Численные значения весов хранятся в кодбуке, а индексатор указывает, какое значение поставить на каждую связь:

$$
\begin{aligned}
W_{ij}&=C_{I_{ij}},\
I&\in\{0,1,2,3,4\}^{11\times8},\
C_0&=0,\
f(x)&=a\sum_{j=1}^{8}\operatorname{ReLU}\!\left(\sum_{i=1}^{11}W_{ij}x_i+b\right)+c.
\end{aligned}
$$

На новой задаче обучаются четыре значения кодбука и три общих числа: скрытое смещение, выходной коэффициент и выходное смещение. Всего семь численных параметров. В варианте NF+VAE дополнительно обучается латентный вектор из 16 чисел; параметры самого VAE заморожены. Независимые веса связей здесь не обучаются. Плотная сеть имеет 105 свободных параметров.

## Банк, выравнивание и VAE

На 12 исходных задачах собрали 6144 обычные сети: по 512 случайных инициализаций на шаблон. Каждую обучали Adam и постепенно прореживали до 32 связей, после прореживания возвращая численные параметры к исходной инициализации. Было шесть стадий по 1024 шага. Этот сбор — отдельные предварительные затраты, они не входят в бюджет решения новой задачи.

В рабочем банке оставили лучшую четверть решений **для каждой задачи отдельно**, по BCE на исходной проверочной выборке. Почти идеальная точность не требуется: даже неполное решение может содержать полезную структуру. Исключаются решения с BCE выше 0,5. Банк содержит 1536 решений: 1152 для обучения VAE, 192 для выбора его состояния и 192 в резерве. Начальные состояния сетей между этими частями не пересекаются. У обучающей части средняя исходная точность — __BANK_MEAN__%, минимальная — __BANK_MIN__%.

Перед обучением VAE обязательно выровняли скрытые столбцы. Венгерский алгоритм сопоставляет столбцы каждой маски общей опорной маске. Опорную маску уточняли по исходным обучающим решениям; среди начальных вариантов выбирали тот, с которым решения лучше согласуются. Аналитическая маска при этом не использовалась.

После выравнивания сравнили, как веса каждой связи меняются между исходными задачами. Связи с похожими профилями объединяются в группы общих значений. Здесь заранее заданы размеры модели: восемь скрытых нейронов, четыре связи на нейрон при извлечении общей маски и четыре группы по восемь связей. Расположение связей и состав групп алгоритму не сообщаются. Извлечены все 32 правильные связи и все 32 правильные принадлежности к группам с точностью до переименования четырёх значений кодбука.

![Общая маска и индексаторы](assets/2026-10-28/pattern_codebook/indexators.png)

NF получает веса сети, градиент ошибки и функциональный отклик её связей. Для исходных задач используются отдельные NF-энкодеры и один общий категориальный декодер VAE. Он восстанавливает **отобранные выровненные маски** и группы общих весов. Состояние VAE выбирается по ошибке восстановления на отложенных сетях исходных задач. Аналитический индексатор используется только для итогового сравнения и контрольного метода.

## Сравниваемые методы

**Извлечённый индексатор** сразу использует найденную общую структуру и обучает кодбук. **VAE без обучения z** декодирует случайный латентный вектор и обучает только численные параметры. **VAE + z** одновременно обучает кодбук и латентный вектор, меняющий дискретный индексатор через замороженный декодер. Для дискретных назначений применяется straight-through оценка градиента.

**Прямое обучение индексатора** начинает с того же обученного декодера, но затем меняет все категориальные оценки связей независимо. Поэтому этот вариант тоже использует перенос; это проверка необходимости латентного ограничения. **Необученный VAE + z** проверяет роль предварительного обучения. Ещё два контроля — случайный индексатор и заранее заданный аналитический индексатор. Плотные Adam и SGD обучают обычную сеть без кодбука.

На каждом новом шаблоне у каждого метода 32 старта по 4096 шагов на одних и тех же 768 обучающих входах. Шаг и лучший старт выбираются по 256 другим входам. Итоговый test содержит 512 входов. Настройки численного обучения выбраны на исходных шаблонах. Дополнительно проверяется точность на **всех 2048 входах**; это полная проверка конечной задачи, а не отдельная независимая выборка. Показатели из разных стартов относятся к четырём задачам, а не к 128 независимым задачам.

## Результаты

«Безошибочный старт» означает ноль ошибок на всех 2048 входах. Последние два столбца относятся к одной модели на задачу, выбранной по query среди 32 стартов.

| Метод | Безошибочные старты | Средняя точность старта, все входы | Выбранные модели, все входы | Выбранные модели, test |
| :-- | --: | --: | --: | --: |
__TABLE__

![Сравнение методов](assets/2026-10-28/pattern_codebook/comparison.png)

Общая структура извлечена правильно: с фиксированным извлечённым индексатором решены все четыре новые задачи без ошибок. **VAE с обучаемым z тоже решил все четыре задачи**, а в отдельных стартах дал 84/128 безошибочных решений против 1/128 у плотного Adam и 0/128 у плотного SGD. Без обучения z у VAE только один безошибочный старт. У необученного декодера их нет: предварительно выученная структура существенна.

При этом VAE не лучший по каждому показателю. Прямое обучение индексатора даёт более высокую среднюю точность старта, хотя точных решений у него меньше. В отдельных запусках VAE ошибается существенно. Поэтому вывод здесь — преимущество по частоте нахождения **полного решения**, а не превосходство по любой метрике.

![Ход обучения](assets/2026-10-28/pattern_codebook/training.png)

Pattern подтверждает работу связки «банк → выравнивание → общая структура → новый кодбук» и перенос через замороженный VAE. Однако его истинный индексатор одинаков для всех шаблонов. Этот эксперимент ещё не доказывает, что VAE умеет составлять принципиально новые сложные индексаторы, и не заменяет проверку на более крупной задаче.

## Два известных датасета: только предварительная проверка постановки

Для MNIST и Fashion-MNIST проверили DeepSets на неупорядоченных множествах точек. Изображение уменьшается до 14×14; каждая из 196 точек содержит строку, столбец и яркость. Метки классов остаются исходными. Во время обучения порядок точек случайно меняется. Использованы 6000 исходных обучающих изображений и 2000 других исходных изображений для проверки, два старта по 2048 шагов. Официальный test ещё не использовался.

| Метод | MNIST | Fashion-MNIST |
| :-- | --: | --: |
| Плотный Adam, случайный порядок точек | 20,68% | 28,75% |
| Отдельная MLP для каждой позиции, без общих весов | 27,95% | 67,20% |
| DeepSets с заданными общими весами | 48,98% | 69,93% |
| Плотный Adam после восстановления порядка точек | 93,80% | 83,73% |

![Предварительная проверка DeepSets](assets/2026-10-28/pattern_codebook/pointset_feasibility.png)

Здесь DeepSets задан вручную, поэтому таблица **не является результатом поиска структуры нашим методом**. Восстановление порядка делает обычный Adam сильным. На Fashion-MNIST преимущество общих весов над отдельными MLP пока небольшое. Эти проверки помогают выбрать содержательную постановку; два примера извлечения структуры на известных датасетах ещё не завершены.

## Более сложный синтетический вариант

Следующая постановка — скрытая перестановка 64 входов, восемь групп по восемь входов и четыре общих ReLU-фильтра. Каждый фильтр имеет восемь коэффициентов: в кодбуке 32 различных обучаемых значения вместо четырёх. Плотная матрица имеет размер 64×32 и 256 полезных связей. Порядок входов и группы заранее алгоритму не сообщаются.

$$
\begin{aligned}
h_{gk}(x)&=\operatorname{ReLU}\!\left(\sum_{r=1}^{8}C_{kr}x_{\pi(8g+r)}+b_k\right),\\
y(x)&=\frac{1}{\sqrt{32}}\sum_{g=0}^{7}\sum_{k=1}^{4}a_k h_{gk}(x),\qquad x\sim\mathcal{N}(0,I_{64}).
\end{aligned}
$$

Здесь перестановка входов общая между задачами, а численные значения кодбука, смещений и выходных коэффициентов меняются. Точное решение задано учителем. Даже число разбиений входов на восемь неупорядоченных групп примерно равно 4,5×10⁴⁷; полный перебор непригоден.

Проведён первый сбор на четырёх исходных задачах: 64 кандидата, 24 поколения; структуры меняли перестановками входов, а их кодбуки обучали Adam. Получены 1536 состояний кандидатов, из них 1351 различная структура. Безошибочность для попадания в этот предварительный банк не требовалась. Ошибка NMSE — MSE, делённая на дисперсию целевого выхода.

| Исходный контроль | NMSE ↓ |
| :-- | --: |
| Фиксированные случайные структуры | 0,7753 |
| Эволюционный поиск структур | 0,6323 |
| Плотный Adam | 0,1171 |
| Правильная структура, новый кодбук | около 5×10⁻¹⁴ |

![Исходный поиск в сложном варианте](assets/2026-10-28/pattern_codebook/scrambled_source_pilot.png)

Изменение структур помогает относительно фиксированных случайных вариантов, но пока проигрывает плотному Adam. При правильной структуре численное обучение легко решает задачу. Значит, текущая трудность — найти индексатор. Это проверки исходных задач; перенос через NF+VAE на новых задачах этой модификации ещё не проверен. Сравнение на новых задачах потребует общего бюджета и дополнительных контролей.

Отдельный контроль по вторым моментам исходных наблюдений восстановил группы и соответствие всех 64 входов. Он использует гауссовское распределение входов и не перебирает полные индексаторы. Это самостоятельный способ предложить структуру из данных, а не результат извлечения из банка через NF+VAE; смешивать эти выводы нельзя.

## Воспроизводимость

Код: `pattern/codebook/`. Основной протокол: `pattern/codebook/protocol_v4.json`. Итоговые массивы и сохранённые модели: `pattern/runs/codebook_20261028_v4/`. Полный исходный сбор: `pattern/runs/codebook_20261028_v2/raw_collection/`. Предшествующие варианты сохранены для проверки влияния отбора, выравнивания и исправления категориальных целей. Неизменённые плотные базовые методы переиспользуются; происхождение записано в `reused_results.json`.

Машинные таблицы: [comparison.csv](assets/2026-10-28/pattern_codebook/comparison.csv), [pointset_feasibility.json](assets/2026-10-28/pattern_codebook/pointset_feasibility.json). Результаты предварительных проверок изображений и веса моделей находятся в `experiments/structure_search_v2/runs/`.

Источники постановок: [Deep Sets](https://arxiv.org/abs/1703.06114), [Torchvision datasets](https://docs.pytorch.org/vision/main/datasets.html), [официальный Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist). Использование DeepSets на изображениях как множествах точек — наша проверочная постановка; это не стандартное сравнение архитектур на исходной растровой задаче.
"""
    report = report.replace("__TABLE__", "\n".join(table_lines))
    report = report.replace(
        "__BANK_MEAN__", f"{audit['mean_training_query_accuracy'] * 100:.2f}"
    )
    report = report.replace(
        "__BANK_MIN__", f"{audit['minimum_training_query_accuracy'] * 100:.2f}"
    )
    (root / "mds/PATTERN_CODEBOOK_2026-10-28.md").write_text(report)


if __name__ == "__main__":
    main()
