"""Build the concise pilot report and figures from completed saved runs."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = ROOT / "mds/assets/2026-10-28/compositional_sine"
LABELS = {
    "learned_parts": "Подбор восстановленных частей",
    "blind_structures": "32 случайных полных сочетания",
    "fixed_source": "Исходные веса без изменения",
    "wrong_parts": "Неверное расположение частей",
    "dense_sgd": "Плотный SGD",
    "dense_adam": "Плотный Adam",
    "source_start_adam": "Исходные веса + Adam",
    "rescaled_source_linear": "Направления, новый масштаб + МНК",
    "rescaled_source_adam": "Направления, новый масштаб + Adam",
}
METHODS = list(LABELS)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    runs = []
    for folder in ["pilot_v1", "gpu_30_v1"]:
        path = HERE / "results" / folder
        plan = json.loads((path / "protocol.json").read_text())
        summary = json.loads((path / "summary.json").read_text())
        control = path / "source_scale_control/summary.json"
        if control.exists():
            summary.update(json.loads(control.read_text()))
        runs.append((path, plan, summary))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(17, 5), layout="constrained", sharey=True
    )
    for ax, (_, plan, summary) in zip(axes, runs):
        rows = [summary.get(method) for method in METHODS]
        values = [
            100 * row["successes"] / row["tasks"] if row else np.nan
            for row in rows
        ]
        colors = [
            "#008577"
            if method == "learned_parts"
            else "#d17932"
            if method in ("dense_sgd", "dense_adam")
            else "#8896a8"
            for method in METHODS
        ]
        ax.barh(np.arange(len(METHODS)), values, color=colors)
        ax.set(
            yticks=np.arange(len(METHODS)),
            yticklabels=[LABELS[method] for method in METHODS],
            xlim=(0, 125),
            xticks=[0, 25, 50, 75, 100],
            xlabel="Успешные задачи, %",
            title=f"{plan['groups']} частей · "
            f"{2 ** plan['groups']:,} структур".replace(",", " "),
        )
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=0.15)
        for index, row in enumerate(rows):
            ax.text(
                values[index] + 2 if row else 2,
                index,
                f"{row['successes']}/{row['tasks']}"
                if row
                else "не запускался",
                va="center",
            )
    axes[0].invert_yaxis()
    for suffix in ["png", "svg"]:
        fig.savefig(ASSETS / f"comparison.{suffix}", dpi=180)
    plt.close(fig)

    path, plan, summary = runs[1]
    tasks = np.load(path / "target_tasks.npz")
    models = np.load(path / "fitted_models.npz")
    index = plan["target"]["validation_tasks"]
    truth_w, truth_a = tasks["weights"][index], tasks["amplitudes"][index]
    fitted_w = models["0_learned_parts_weights"]
    fitted_a = models["0_learned_parts_amplitudes"]
    rng = np.random.default_rng(28109999)
    x = rng.standard_normal((256, truth_w.shape[0]))
    truth_y = np.sin(x @ truth_w) @ truth_a
    fitted_y = np.sin(x @ fitted_w) @ fitted_a
    adam_y = (
        np.sin(x @ models["0_dense_adam_weights"])
        @ models["0_dense_adam_amplitudes"]
    )
    truth_v = truth_w / np.linalg.norm(truth_w, axis=0)
    fitted_v = fitted_w / np.linalg.norm(fitted_w, axis=0)
    matching = np.abs(fitted_v.T @ truth_v).argmax(1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    axes[0].scatter(
        truth_y,
        fitted_y,
        s=14,
        alpha=0.6,
        label="Восстановленные части",
        color="#008577",
    )
    axes[0].scatter(
        truth_y,
        adam_y,
        s=14,
        alpha=0.5,
        label="Плотный Adam",
        color="#d17932",
    )
    limit = np.max(np.abs(truth_y)) * 1.05
    axes[0].plot([-limit, limit], [-limit, limit], "--", color="#172b4d")
    axes[0].set(
        xlabel="Ответ учителя",
        ylabel="Предсказание",
        title="Первая новая задача · 30 частей",
    )
    axes[0].legend(fontsize=9)
    truth_frequency = np.linalg.norm(truth_w, axis=0)
    fitted_frequency = np.empty_like(truth_frequency)
    fitted_frequency[matching] = np.linalg.norm(fitted_w, axis=0)
    axes[1].plot(
        range(1, len(matching) + 1),
        truth_frequency,
        "o-",
        color="#172b4d",
        label="Учитель",
    )
    axes[1].plot(
        range(1, len(matching) + 1),
        fitted_frequency,
        "x--",
        color="#008577",
        label="Найденные частоты",
    )
    axes[1].set(
        xlabel="Номер части",
        ylabel="Частота",
        title="Численные значения обучены заново",
    )
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.15)
    for suffix in ["png", "svg"]:
        fig.savefig(ASSETS / f"example.{suffix}", dpi=180)
    plt.close(fig)

    table = [
        (
            "| Метод | Успех: 20 частей | MSE: 20 частей | "
            "Успех: 30 частей | MSE: 30 частей |"
        ),
        "| --- | --- | --- | --- | --- |",
    ]
    for method in METHODS:
        cells = [LABELS[method]]
        for _, _, results in runs:
            row = results.get(method)
            if row:
                cells.extend(
                    [
                        f"{row['successes']}/{row['tasks']}",
                        f"{row['mean_mse']:.6f}",
                    ]
                )
            else:
                cells.extend(["—", "—"])
        table.append("| " + " | ".join(cells) + " |")
    records = []
    for _, plan, summary in runs:
        for method, row in summary.items():
            records.append(
                {
                    "groups": plan["groups"],
                    "structures": 2 ** plan["groups"],
                    "method": method,
                    **row,
                }
            )
    with (ASSETS / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    text = r"""# Перенос частей при миллионе и миллиарде структур

Пилот · 28 октября 2026.

## Задача

Проверяем, можно ли извлечь повторяющиеся части из простых решений
и собрать новое решение при слишком большом числе сочетаний для полного
перебора. Используем 20 или 30 частей, у каждой два варианта.
Число допустимых структур — 1 048 576 и 1 073 741 824 соответственно.

У каждой части восемь полезных входов. Их расположение скрыто одной общей
перестановкой координат. Ответ учителя — сумма синусоид:

$$
\begin{aligned}
J&\in\{20,30\},\qquad x\sim\mathcal N(0,I_{8J}),\\
y_s(x)&=\frac1{\sqrt J}\sum_{j=1}^{J}
a_j\sin(\omega_jv_{j,s_j}^{\top}x),\qquad s_j\in\{0,1\},\\
a_j&\in[0.8,1.2],\qquad \omega_j\in[3.8,4.2].
\end{aligned}
$$

Каждый вариант части — вектор длины 1 с восемью ненулевыми весами.
Их знаки соответствуют шаблонам 11110000 или 11001100.
У разных частей полезные входы не пересекаются. Полную структуру
задают 20 или 30 двоичных индексов.

## Аналитическое решение

Ученик — сумма синусоид с обучаемыми весами и выходными коэффициентами.
Точное решение существует: выбрать правильный вариант каждой части,
умножить его вектор на частоту учителя и взять выходной коэффициент
учителя, поделённый на корень из числа частей:

$$
w_j=\omega_jv_{j,s_j},\qquad
c_j=\frac{a_j}{\sqrt J}
\quad\Longrightarrow\quad f(x)=y_s(x),\quad R=0.
$$

Разные части используют независимые входы. Поэтому их можно сначала
сравнивать по связи с ответом новой задачи по отдельности, затем совместно
уточнять коэффициенты. Миллион или миллиард полных сочетаний перечислять
для этого не требуется.

## Пример найденного решения

На первой новой задаче с 30 частями восстановлены все 30 выбранных
вариантов. Частоты, которых не было в исходном банке, подобраны заново.
На рисунке показаны ответы на 256 новых входах и найденные частоты.
Пример взят по номеру задачи, без отбора по качеству.

![Предсказания и восстановленные частоты](assets/2026-10-28/compositional_sine/example.png)

## Данные и методы

Для каждой размерности собрали отдельный банк из 40 или 60 элементарных
низкочастотных задач — по одной на каждый вариант части.
Это задачи с одной синусоидой и частотой от 0,8 до 1,2.
Каждую решали плотным SGD с четырёх стартов по 2048 шагов.
Полученные веса разделили на три кластера: отрицательные, нулевые
и положительные. Совпадение ненулевых координат позволило объединить
варианты в пары. Все 40 и 60 частей восстановлены правильно.
Границы групп и правильные индексы новых задач методу не передавались.

Таким образом, в банке есть все элементарные части, но нет полных
многокомпонентных сочетаний новых задач. Пилот проверяет сборку знакомых
частей; совершенно новые элементарные части здесь не проверялись.

**Подбор восстановленных частей:** на 4096 примерах новой задачи
сравнить два варианта каждой части и 64 возможные частоты от 0,125 до 8.
Затем подобрать амплитуды методом наименьших квадратов и дважды уточнить
частоты по остаткам. Истинные частоты учителя методу неизвестны.

**32 случайных полных сочетания:** выбрать целиком 32 структуры
из восстановленных частей. Для каждой подобрать частоты и амплитуды
тем же способом и выбрать лучшую по отдельным 4096 примерам.

**Исходные веса без изменения:** зафиксировать все низкочастотные векторы
банка и обучить только выходные коэффициенты.
**Неверное расположение частей:** повторить подбор частей после новой
перестановки их входных координат.

**Плотные SGD и Adam:** обучать все входные веса и выходные коэффициенты
с новых случайных стартов. **Исходные веса + Adam:** начать с обученных
исходных векторов, затем разрешить всем весам меняться.
VAE в этом пилоте не использовался.

Дополнительный контроль — **нормированные исходные направления**.
Сохраняем направления всех исходных векторов, выбираем один общий масштаб
из 0,5/1/2/4/8 по двум настроечным задачам, затем обучаем выходные
коэффициенты методом наименьших квадратов (МНК). Выбран масштаб 4.
В GPU-проверке дополнительно обучаем все веса Adam с такого старта;
его шаг также выбран по настроечным задачам.

На каждую размерность взяли две отдельные задачи для настройки
оптимизаторов и восемь новых задач для итоговой проверки.
Пилот с 20 частями: восемь стартов по 1024 шага, сеть из 40 синусоид.
Усиленная проверка с 30 частями на V100: 16 стартов по 4096 шагов;
число синусоид 30 или 60, масштаб старта и шаг выбраны на настройке.
Для Adam со стартом из исходных весов шаг настроен отдельно
в усиленной проверке. Размер минибатча — 64.

## Результаты

Успех — точная MSE ниже 0,1 по нормальному распределению входов.
Ошибка вычисляется по аналитической формуле среднего произведения
синусоид. В таблице один результат — лучший выбранный поиск на одной
новой задаче; это не доля отдельных стартов.

![Сравнение методов](assets/2026-10-28/compositional_sine/comparison.png)

{{TABLE}}

Подбор извлечённых частей решает новые задачи с очень малой ошибкой.
Случайный выбор целых сочетаний, сохранение исходной частоты
и неверное расположение частей такого результата не дают.

Контроль масштаба важен для интерпретации: перенос исходных направлений
с новым масштабом также решает задачи, даже без явного выбора
дискретного сочетания. Значит, полезен извлечённый словарь направлений,
а специальная процедура выбора частей не является единственным
успешным методом. Узкий диапазон новых частот облегчает этот контроль.
Adam с нормированными исходными направлениями также дал 8/8 успехов
в усиленной проверке. Поэтому результат относится к пользе исходного
представления для обучения, а не к невозможности решить задачу градиентом.

Результат поддерживает перенос структуры в пространстве,
где полный перебор уже не укладывается в использованный бюджет.
Но найденная структура делает задачу разложимой на части:
экспоненциальное число сочетаний само по себе не означает, что
единственный альтернативный способ — полный перебор.

Это два небольших пилота, по одному исходному банку и восьми новым
задачам на размерность. Исходные задачи содержат отдельные готовые части;
выделение частей из сложных смешанных решений здесь не проверялось.
Спектральный подбор использует специальную
форму синусоидальной модели. Его затраты не приравниваются к затратам
градиентного обучения по числу стартов или предъявлений примеров.
Доказательства превосходства над всеми градиентными методами
или преимущества самого VAE здесь нет.

Исходные банки потребовали 160 и 240 обучений соответственно.
Их подготовка и настройка методов оплачиваются отдельно от поиска
на одной новой задаче; общая экономия FLOPs не измерялась.

## Воспроизводимость

[Постановка и команды](../experiments/compositional_sine_v2/README.md),
[протокол первого пилота](../experiments/compositional_sine_v2/protocol.json),
[протокол усиленной проверки](../experiments/compositional_sine_v2/protocol_gpu_30.json).
В папках results/pilot_v1 и results/gpu_30_v1 сохранены исходные
и найденные веса, отдельные результаты, настройки, контрольные суммы
протоколов и точные копии запущенного скрипта.
"""
    text = text.replace("{{TABLE}}", "\n".join(table))
    (ROOT / "mds/COMPOSITIONAL_SINE_PILOT_2026-10-28.md").write_text(text)
    print("Saved report, 2 figures and CSV from both completed runs")


if __name__ == "__main__":
    main()
