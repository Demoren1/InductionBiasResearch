"""Summarize the sealed real-network experiment and draw research figures."""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from bank import DIRECTORY, RUN, SEEDS, families
from core import save_json
from train import load_cards, restore


LABELS = {
    "dense": "Dense SGD",
    "dense_tuned": "Tuned dense SGD",
    "dense_long": "Long dense SGD",
    "random": "Random masks",
    "elite_pool": "Replay source masks",
    "vanilla": "Ordinary NF + VAE",
    "partwise_training": "Partwise NF + VAE",
    "empirical_product": "Independent frequencies",
    "oracle": "Known correct mask",
}
COLORS = {
    "dense": "#808b96",
    "dense_tuned": "#808b96",
    "dense_long": "#808b96",
    "random": "#db9735",
    "elite_pool": "#aaa5aa",
    "vanilla": "#ac638d",
    "partwise_training": "#277dad",
    "empirical_product": "#39966a",
    "oracle": "#b4c8ad",
}


def wilson(successes, count):
    z = 1.959963984540054
    p = successes / count
    center = (p + z * z / (2 * count)) / (1 + z * z / count)
    half = (
        z
        * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count))
        / (1 + z * z / count)
    )
    return [center - half, center + half]


def audit():
    records = json.loads((RUN / "models.json").read_text())
    _, roles = load_cards()
    summaries = {}
    heatmaps = {}
    for variant in ("vanilla", "partwise_training"):
        counts = np.zeros((4, 4), dtype=int)
        total, valid = 0, 0
        for record in records:
            if record["variant"] != variant:
                continue
            model = restore(record)
            rng = np.random.default_rng(330101 + record["seed"])
            latent = torch.tensor(
                rng.standard_normal((131072, 16)), dtype=torch.float32
            )
            with torch.no_grad():
                scores = model.decode(latent).flatten(1).numpy()
            indices = np.argsort(-scores, axis=1, kind="stable")[:, :2]
            masks = np.zeros((len(scores), 8, 2), dtype=bool)
            flat = masks.reshape(-1, 16)
            np.put_along_axis(flat, indices, True, axis=1)
            right_roles = (
                (masks.sum(1) == 1).all(1)
                & (masks[:, :4, 0].sum(1) == 1)
                & (masks[:, 4:, 1].sum(1) == 1)
            )
            a = masks[:, :4, 0].argmax(1)
            b = masks[:, 4:, 1].argmax(1)
            np.add.at(counts, (a[right_roles], b[right_roles]), 1)
            total += len(masks)
            valid += int(right_roles.sum())
        distribution = counts / counts.sum()
        independent = (
            distribution.sum(1)[:, None] * distribution.sum(0)[None, :]
        )
        positive = distribution > 0
        mutual_information = float(
            (
                distribution[positive]
                * np.log(distribution[positive] / independent[positive])
            ).sum()
        )
        summaries[variant] = {
            "draws": total,
            "valid_fraction": valid / total,
            "train_mass": sum(counts[a, b] for a, b in families("train"))
            / total,
            "test_mass": sum(counts[a, b] for a, b in families("test"))
            / total,
            "pair_mutual_information_nats": mutual_information,
            "all_draw_pair_frequencies": (counts / total).tolist(),
        }
        heatmaps[variant] = counts / total
    unique = set()
    for _, labels in roles[0]:
        unique.update(tuple(mask.flatten().tolist()) for mask in labels)
    summaries["unique_training_masks"] = len(unique)
    save_json(RUN / "proposal_audit.json", summaries)
    return summaries, heatmaps


def main():
    torch.set_num_threads(1)
    if not (RUN / "TEST_COMPLETE.json").exists():
        raise RuntimeError("Test evaluation must finish first")
    records = [
        json.loads(p.read_text()) for p in (RUN / "test").glob("*.json")
    ]
    results, curves, all_values = {}, {}, {}
    for method in LABELS:
        selected, by_seed, retained = [], {}, 0
        candidate_count, candidate_success, correct_support_count = 0, 0, 0
        by_family = {}
        curve_values = {budget: [] for budget in (1, 4, 16, 32, 64)}
        for record in records:
            if record["method"] != method:
                continue
            name = f"{method}_a{record['a']}_b{record['b']}_f{record['function']}_s{record['model_seed']}"
            with np.load(RUN / "test" / f"{name}.npz") as data:
                selected.extend(data["selected"].tolist())
                by_seed.setdefault(str(record["model_seed"]), []).extend(
                    data["selected"].tolist()
                )
                family = f"a{record['a']}_b{record['b']}"
                by_family.setdefault(family, []).extend(
                    data["selected"].tolist()
                )
                candidate_count += len(data["candidate_losses"])
                candidate_success += int(
                    (data["candidate_losses"] < 0.1).sum()
                )
                expected = np.zeros((8, 2), dtype=np.float32)
                expected[record["a"], 0] = 1
                expected[record["b"] + 4, 1] = 1
                correct_support_count += int(
                    (data["masks"] == expected).all((1, 2)).sum()
                )
                if record["budget"] == 64:
                    population = data["candidate_losses"].reshape(64, 64)
                    empirical = data["candidate_empirical"].reshape(64, 64)
                    for budget in curve_values:
                        index = empirical[:, :budget].argmin(1)
                        curve_values[budget].extend(
                            population[np.arange(64), index].tolist()
                        )
            retained += record.get("success_retained_after_unmask", 0)
        values = np.asarray(selected)
        n, k = len(values), int((values < 0.1).sum())
        results[method] = {
            "success": k,
            "searches": n,
            "rate": k / n,
            "wilson95_conditional": wilson(k, n),
            "mean_selected_population_mse": float(values.mean()),
            "median_selected_population_mse": float(np.median(values)),
            "candidate_count": candidate_count,
            "candidate_success": candidate_success,
            "candidate_success_rate": candidate_success / candidate_count,
            "exact_correct_sparse_support_fraction": correct_support_count
            / candidate_count,
            "family_rates": {
                family: float((np.asarray(v) < 0.1).mean())
                for family, v in by_family.items()
            },
            "seed_rates": {
                seed: float((np.asarray(v) < 0.1).mean())
                for seed, v in by_seed.items()
            },
        }
        if method == "partwise_training":
            results[method]["retained_after_unmask"] = retained
        curves[method] = {
            str(b): float((np.asarray(v) < 0.1).mean())
            for b, v in curve_values.items()
            if v
        }
        all_values[method] = values
    audit_summary, heatmaps = audit()
    geometry = json.loads((DIRECTORY / "geometry.json").read_text())
    population_geometry = json.loads(
        (DIRECTORY / "geometry_population.json").read_text()
    )
    selection = json.loads((RUN / "selection.json").read_text())
    bank = json.loads((RUN / "BANK_COMPLETE.json").read_text())
    manifest = json.loads((RUN / "bank/manifest.json").read_text())
    models = json.loads((RUN / "models.json").read_text())
    summary = {
        "results": results,
        "curves": curves,
        "bank": bank,
        "proposal_audit": audit_summary,
        "geometry": geometry,
        "population_geometry": population_geometry,
        "selection": selection,
        "offline": {
            "bank_sgd_steps": bank["fits"] * 1024,
            "bank_training_examples": bank["fits"] * 1024 * 64,
            "bank_seconds": sum(r["seconds"] for r in manifest),
            "vae_training_seconds": sum(r["seconds"] for r in models),
        },
        "interval_scope": "Conditional restart intervals; shared minibatches within context, 24 contexts, no task-population confidence claim",
    }
    save_json(RUN / "summary.json", summary)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
        }
    )
    fig, axes = plt.subplots(
        2, 2, figsize=(13.2, 9.0), constrained_layout=True
    )
    spec = manifest[0]["task"]
    coordinate = np.linspace(-4, 4, 1000)
    true = spec["small"][0] * np.sin(spec["low"][0] * coordinate) + np.sin(
        spec["high"][0] * coordinate
    )
    ax = axes[0, 0]
    ax.plot(
        coordinate,
        true,
        color="#45515e",
        lw=1.3,
        label="Target: two frequencies",
    )
    for key, label, color in (
        ("broad_low_frequency", "Poor local minimum", "#db9735"),
        ("good_high_frequency", "Good local minimum", "#277dad"),
    ):
        vector = np.asarray(geometry[key]["parameters"])
        ax.plot(
            coordinate,
            vector[16] * np.sin(vector[0] * coordinate),
            lw=1.6,
            color=color,
            label=label,
        )
    ax.set(
        xlabel="Relevant input coordinate",
        ylabel="Prediction",
        title="A. A real regression task, MSE loss",
    )
    ax.legend(fontsize=8, loc="upper right")
    data = np.load(DIRECTORY / "geometry.npz")
    ax = axes[0, 1]
    ax.plot(data["frequencies"], data["profile"], color="#45515e", lw=2)
    ax.set(
        xlabel="Input frequency weight (output amplitude optimized)",
        ylabel="Single-head empirical MSE",
        title="B. Smooth competing minima emerge from prediction",
    )
    ax.annotate(
        "Broad, poor",
        xy=(0.96, 0.50),
        xytext=(0.55, 0.34),
        arrowprops={"arrowstyle": "->"},
        color="#b27622",
    )
    ax.annotate(
        "Sharper, good",
        xy=(3.92, 0.042),
        xytext=(4.3, 0.15),
        arrowprops={"arrowstyle": "->"},
        color="#277dad",
    )
    ax.text(
        0.98,
        0.94,
        "Both Hessians positive definite\nGood / poor curvature = 6.8×",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=9,
    )
    ax = axes[1, 0]
    displayed = [m for m in LABELS if m != "dense"]
    rates = np.array([results[m]["rate"] for m in displayed]) * 100
    intervals = (
        np.array([results[m]["wilson95_conditional"] for m in displayed]) * 100
    )
    positions = np.arange(len(displayed))
    ax.barh(
        positions, rates, color=[COLORS[m] for m in displayed], height=0.66
    )
    ax.errorbar(
        rates,
        positions,
        xerr=np.maximum(
            0, np.stack((rates - intervals[:, 0], intervals[:, 1] - rates))
        ),
        fmt="none",
        color="#37414a",
        lw=0.8,
        capsize=2,
    )
    ax.set_yticks(positions, [LABELS[m] for m in displayed])
    ax.invert_yaxis()
    ax.set(
        xlim=(0, 108),
        xlabel="Successful searches (%)",
        title="C. Unseen combinations; equal training-example budget",
    )
    for p, rate in zip(positions, rates):
        ax.text(rate + 2.0, p, f"{rate:.1f}%", va="center", fontsize=8)
    ax = axes[1, 1]
    for method in (
        "dense_tuned",
        "random",
        "vanilla",
        "partwise_training",
        "empirical_product",
    ):
        points = curves[method]
        ax.plot(
            [int(b) for b in points],
            [100 * value for value in points.values()],
            "o-",
            label=LABELS[method],
            color=COLORS[method],
            ms=4,
        )
    ax.set(
        xlabel="Candidates × 1024 SGD steps × batch 64",
        ylabel="Successful searches (%)",
        title="D. The frozen generator proposes; SGD trains weights",
    )
    ax.legend(fontsize=8)
    fig.suptitle(
        "A sparse-mask generator reaches a better minimum in a sine neural network",
        fontsize=14,
    )
    fig.savefig(RUN / "experiment.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), constrained_layout=True)
    for ax, method in zip(axes, ("vanilla", "partwise_training")):
        matrix = heatmaps[method] * 100
        shown = ax.imshow(matrix, vmin=0, vmax=13, cmap="Blues")
        for a in range(4):
            for b in range(4):
                split = (
                    "train"
                    if (b - a) % 4 in (0, 1)
                    else ("val" if (b - a) % 4 == 2 else "test")
                )
                ax.text(
                    b,
                    a,
                    f"{matrix[a, b]:.1f}%\n{split}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if matrix[a, b] > 7 else "#25344b",
                )
        ax.set(
            title=LABELS[method],
            xlabel="Part B",
            ylabel="Part A",
            xticks=range(4),
            yticks=range(4),
        )
    fig.colorbar(
        shown, ax=axes, shrink=0.8, label="Fraction of all prior proposals (%)"
    )
    fig.savefig(RUN / "combinations.png")
    plt.close(fig)

    table = "\n".join(
        f"| {LABELS[m]} | {r['success']}/{r['searches']} | {100 * r['rate']:.2f}% | {r['mean_selected_population_mse']:.4f} |"
        for m, r in results.items()
    )
    text = f"""# Нейросеть с двумя конкурирующими частотами

## Постановка

Вход — восемь независимых нормальных признаков, выход — два числа.
Маленькая нейросеть имеет два синусоидальных нейрона и обучаемые выходные
амплитуды: всего 18 параметров. На каждый выход приходится один нейрон.

$$
f_j(x)=c_j\\sin(w_j^T x), \\qquad
y_j=\\alpha_j\\sin(\\ell_j x_{{r_j}})+\\sin(h_j x_{{r_j}}).
$$

Низкая частота лежит в диапазоне 0.65–0.75, высокая — 3.8–4.2,
слабая амплитуда — 0.35–0.45. Первый релевантный признак выбирается
из первых четырёх, второй — из последних четырёх. Один нейрон
не может точно воспроизвести обе частоты. Обучение слабой низкой
частоты даёт плохой минимум, сильной высокой — хороший. Это намеренно
маленькая сеть с ограниченной ёмкостью и синусоидальной активацией.

Обучение использует только обычную MSE предсказаний и настоящий autograd
SGD на свежих минибатчах из 64 примеров. Нет заданной функции качества
по расстоянию до параметров teacher, переключения между чашами,
добавленного шума к градиенту или IMP.

Маска оставляет ровно две из 16 входных связей. Оба выходных коэффициента
всегда обучаются. NF/VAE получает нормированные веса и градиенты
эмпирической MSE по активным связям; источник не содержит test-масок.

## Контроль до обучения генератора

В пилоте плотный SGD дал 0/8192 хороших сетей при четырёх масштабах
инициализации. Правильная маска при масштабе 2.0 дала 309/2048 успехов
после 2048 шагов; это диагностический oracle, не метод поиска.
Пилот использовал общий RNG для начальных весов и данных; основной
протокол разделяет эти потоки. Исходный пилот сохранён отдельно.

На обычной эмпирической MSE по 16 384 примерам два решения дополнительно
доведены до стационарных точек. Плохое: MSE {geometry["broad_low_frequency"]["empirical_mse"]:.5f},
хорошее: {geometry["good_high_frequency"]["empirical_mse"]:.5f}.
Градиенты меньше 5e-8; оба гессиана положительно определены.
Геометрическая средняя кривизна хорошего выше в
{geometry["curvature_ratio_good_over_broad"]:.2f} раза. Это локальная резкость
в данных координатах параметров; вероятность попадания измеряется
отдельно. Глобальная оптимальность хорошего решения не утверждается.

Поскольку основной SGD использует свежие примеры, отдельно проверена
и ожидаемая MSE: оба её минимума также имеют положительные гессианы,
градиенты меньше 8e-9, MSE соответственно
{population_geometry["broad_low_frequency"]["population_mse"]:.6f} и
{population_geometry["good_high_frequency"]["population_mse"]:.6f}.
Отношение средних геометрических кривизн —
{population_geometry["curvature_ratio_good_over_broad"]:.2f}.

## Банк и замороженный NF + VAE

Восемь сочетаний входят в source train, четыре в validation, четыре
в новый структурный test; разбиение определено до обучения.
Все отдельные части встречаются в train. У каждого source-сочетания
две функции с разными частотами/амплитудами, всего 16 source-задач.

Равномерно выбираются маски из всех 120 вариантов двух связей.
Роли source train/validation присваиваются до обучения и отбора.
Каждая сеть обучается 1024 шагами SGD. Порог отбора — измеренная MSE
меньше 0.1 на 4096 новых примерах; поддержка teacher в отборе не используется.
Получено {bank["fits"]:,} обученных сетей, {bank["elites"]} элит,
{audit_summary["unique_training_masks"]} разных масок. Сохранены и неудачные попытки.

Использована прежняя NF-архитектура, латентный код 16 и один общий
decoder 16→128→128→16. Вариант partwise_training сохранён:
при реконструкции каждой выходной колонки чужая половина латентного
кода заменяется независимым нормальным шумом. KL-вес 0.05,
тот же BCE + straight-through MSE масок. Обычный VAE отличается только
отсутствием этой замены. Обучающие цели — только настоящие source-маски.
Роли выходов здесь заданы самой двухвыходной задачей, поэтому цели
канонические у обоих вариантов. Новые полные пары в обучение не добавляются.
Все три seed сохранены, удачный seed не выбирался.

На новой задаче encoder и decoder заморожены. Сэмпл из стандартного
нормального prior, один проход decoder и глобальный top-2 дают маску.
Для каждого кандидата заново обучаются веса сети и выходные амплитуды.
Лучший кандидат выбирается по измеренной MSE на свежих примерах.

## Результат

Итог — четыре не встречавшиеся пары, две новые функции на пару,
три независимых потока запусков, по 64 поиска на контекст:
1536 поисков на метод. Успех означает population MSE ниже 0.1.
Для нормальных входов эта обычная MSE имеет точный интеграл; он
используется для диагностики, оценки validation и итогового измерения,
SGD оптимизирует ошибки настоящих примеров. Кандидат выбирается по
эмпирической ошибке, не по знанию правильных частот или маски.

| Метод | Успехи | Вероятность успеха | Средняя MSE выбранного |
|---|---:|---:|---:|
{table}

У всех методов бюджет обучения — 65 536 SGD-обновлений отдельных сетей,
или 4 194 304 примера на один поиск. Обычно это 64 старта по 1024 шага;
длинный SGD — восемь стартов по 8192 шага. Его настройки выбраны на
validation среди трёх learning rate и четырёх масштабов инициализации:
{selection["dense_options"]}. Моментум, Adam и все возможные способы
подбора начальных весов не проверялись. NF не гарантирует преимущество
над любым возможным градиентным алгоритмом.

После снятия маски и ещё 1024 шагов плотного SGD сохранились
{results["partwise_training"]["retained_after_unmask"]}/{results["partwise_training"]["success"]}
хороших решений partwise NF/VAE. Значит, хорошая точка доступна
и плотной сети; разрежение помогает туда попасть.

![Эксперимент](experiment.png)

![Новые сочетания](combinations.png)

![Обучение и восстановление](training.png)

training_metrics.json содержит train/validation Recall@2 масок на замороженных
лучших checkpoint. Сохранённая история VAE содержит validation loss,
KL и exact mask recovery; исторические train loss и Recall@2 по эпохам
не записывались. Они не восстанавливаются вымышленными кривыми.
Отдельные 128 source-траекторий SGD на политику служат иллюстрацией
обучения; известная маска в них — диагностический oracle. В основное
сравнение поисков эти дополнительные траектории не входят.

## Что показывает результат и его границы

Неудачные запуски SGD неоднородны. В исходном пилоте при масштабе
инициализации 2.0 у 97.9% плотных сетей обе выходные амплитуды стали
меньше 0.05 по модулю. При угасших амплитудах градиенты входных весов
слабы. Часть конечнобюджетных неудач связана с такой задержкой,
поэтому достижение широкого плохого минимума не приписывается каждому
неудачному запуску. Наличие двух строгих минимумов и частота достижения
хорошего решения проверены отдельно.

Это перенос способа сочетать части в настоящую оптимизацию предсказаний,
с гладкой стандартной loss и естественными для выбранной сети локальными
минимумами. Проверена синусоидальная сеть с двумя известными выходными
ролями и ограниченной ёмкостью. Перенос на ReLU/MLP, неизвестные модули
или произвольные реальные данные ещё не проверен. Геометрия исследована
на одной source-функции; вероятность успеха — на всех test-контекстах.

Независимые частоты частей — обязательный простой контроль:
они изучены из того же банка, с одинаковым весом source-задач,
а правило «одна связь на выход» выводится из элит. Этот контроль
отделяет пользу композиционного предположения от сложности VAE.
Replay исходных элит не создаёт test-пары.

Равенство бюджета относится к онлайн-обучению весов. Банк стоил
{summary["offline"]["bank_training_examples"]:,} обучающих примеров
({summary["offline"]["bank_seconds"]:.1f} с в векторизованном GPU-запуске),
шесть VAE — {summary["offline"]["vae_training_seconds"]:.1f} с CPU.
Банк равен 4096 бюджетам одиночного онлайн-поиска. Это не сравнение
полной стоимости с нуля. Выбор масок и измерение качества учитываются
отдельно от числа обучающих примеров. Векторизованное время не равно
времени последовательного применения метода.

Интервалы Wilson в summary.json описывают условные повторы начальных
весов. Внутри контекста методы и старты разделяют поток минибатчей;
всего 24 контекста. Это не интервалы для популяции произвольных задач.

## Артефакты

plan.json задаёт протокол, selection.json фиксирует validation,
bank/ хранит все попытки и элиты, models/ — все шесть checkpoint,
test/ — сырые результаты и маски, source/ — снимок реализации,
summary.json — агрегаты. Старый искусственный опыт не изменён.
"""
    (RUN / "REPORT.md").write_text(text)
    print(
        json.dumps(
            {"results": results, "proposal_audit": audit_summary}, indent=2
        )
    )


if __name__ == "__main__":
    main()
