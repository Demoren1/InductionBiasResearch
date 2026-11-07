"""Audit support/query alternating search and render selected indexators."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import digest, save_json
from .evaluate import align_indices
from .report import categorical, matched_columns, save_figure
from .sharing_report import evaluate

LABELS = {
    "joint_support": "Совместно: support",
    "alternating_support": "Поочерёдно: support/support",
    "alternating_query": "Поочерёдно: support/query",
}


def audit_search(root):
    audit = {}
    for regime in ("full", "masked"):
        for phase in ("pilot", "confirmation"):
            history = json.loads((root / regime / phase / "history.json").read_text())
            norms = {}
            for key in ("numeric", "architecture"):
                values = [
                    value
                    for row in history
                    if row["gradient_norms"]
                    for value in row["gradient_norms"][key]
                ]
                norms[f"{key}_gradient_norm_min_max"] = [min(values), max(values)]
            audit[f"{regime}/{phase}"] = {
                "maximum_hard_forward_error": max(
                    row["maximum_hard_forward_error"] for row in history
                ),
                **norms,
            }
    output = root / "masked"
    choice = json.loads((output / "pilot_selection.json").read_text())["families"][
        "joint_support"
    ]
    with (
        np.load(output / "teacher_audit_only.npz") as teacher,
        np.load(output / "pilot/hard_fits.npz") as fits,
    ):
        audit["post_freeze_pilot_joint_masked_recovery"] = [
            {
                "seed": row["seed"],
                "recovery": align_indices(
                    fits["indices"][row["row"]], teacher["indices"], 4
                )[0],
            }
            for row in choice["per_seed"]
        ]
    save_json(root / "gradient_and_pilot_audit.json", audit)


def search_diagnostics(output):
    families = json.loads((output / "summary.json").read_text())["families"]
    rows = json.loads((output / "confirmation/hard_fits.json").read_text())["rows"]
    history = json.loads((output / "confirmation/history.json").read_text())
    variance = json.loads((output / "specification.json").read_text())[
        "target_variance"
    ]
    with np.load(output / "confirmation/snapshots.npz") as archive:
        probabilities = archive["policy_probabilities"]
    diagnostics = {}
    for family, result in families.items():
        diagnostics[family] = []
        for candidate in result["per_seed"]:
            row = rows[candidate["row"]]
            record = next(
                item for item in history if item["step"] == row["search_step"]
            )
            policy = probabilities[row["snapshot"], row["root"]]
            active = policy.argmax(-1) != 0
            diagnostics[family].append(
                {
                    "seed": row["seed"],
                    "search_step": row["search_step"],
                    "search_support_nmse": float(
                        np.mean(record["train_mse"][row["root"]]) / variance
                    ),
                    "search_query_nmse": float(
                        np.mean(record["query_mse"][row["root"]]) / variance
                    ),
                    "selection_nmse_after_Cfit": candidate["selection_nmse"],
                    "MAP_edges": int(active.sum()),
                    "mean_MAP_probability": float(policy.max(-1).mean()),
                    "median_zero_probability_on_active_edges": float(
                        np.median(policy[..., 0][active])
                    )
                    if active.any()
                    else None,
                }
            )
    save_json(output / "search_diagnostics.json", diagnostics)
    return diagnostics


def figures(output, reference):
    families = json.loads((output / "summary.json").read_text())["families"]
    with np.load(output / "aligned_states.npz") as archive:
        aligned = dict(archive)
    with np.load(output / "confirmation/hard_fits.npz") as archive:
        fitted = dict(archive)
    fitted_rows = json.loads((output / "confirmation/hard_fits.json").read_text())[
        "rows"
    ]
    with np.load(output / "confirmation/snapshots.npz") as archive:
        snapshots = dict(archive)
    selected = {}
    for family, result in families.items():
        primary = result["primary"]
        number, restart = primary["row"], primary["query_selected_restart"]
        row = fitted_rows[number]
        selected[family] = {
            "selection": primary,
            "settings": row,
            "raw_indexator": fitted["indices"][number].tolist(),
            "aligned_indexator": aligned[f"row_{number}_indices"].tolist(),
            "numeric": fitted["numeric"][number, restart].tolist(),
            "weights": fitted["weights"][number, restart].tolist(),
            "train_mse": float(fitted["train_mse"][number, restart]),
            "query_mse": float(fitted["query_mse"][number, restart]),
            "selection_mse": float(fitted["selection_mse"][number, restart]),
        }
    save_json(output / "selected_indexators.json", selected)
    truth, order = aligned["true_indices"], aligned["row_order"]
    with np.load(output / "teacher_audit_only.npz") as archive:
        true_codes = np.r_[0.0, archive["numeric"][:4]]
    true_weights = true_codes[np.abs(truth)] * np.sign(truth)
    with np.load(reference / "selected_states.npz") as archive:
        dense = matched_columns(archive["dense_shared_adam_weights"], true_weights)
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 150})
    figure, axes = plt.subplots(2, 3, figsize=(14.5, 9.3), constrained_layout=True)
    image = categorical(axes.flat[0], truth[order], 4, "Истинный индексатор")
    for axis, (family, label) in zip(list(axes.flat)[1:4], LABELS.items()):
        result = families[family]
        primary = result["primary"]
        number = primary["row"]
        image = categorical(
            axis,
            aligned[f"row_{number}_indices"][order],
            4,
            f"{label}\nошибок {primary['recovery']['categorical_errors']}/256; "
            f"точных масок {result['exact_masks']}/3",
        )
    colorbar = figure.colorbar(
        image,
        ax=list(axes.flat),
        ticks=range(-4, 5),
        shrink=0.5,
        pad=0.04,
        location="bottom",
        fraction=0.035,
    )
    colorbar.set_label("Индекс кодбука")
    result = families["alternating_query"]
    primary = result["primary"]
    query_weights = fitted["weights"][primary["row"], primary["query_selected_restart"]]
    query_weights = query_weights[:, aligned[f"row_{primary['row']}_columns"]]
    for axis, matrix, title in (
        (axes.flat[4], dense, "Плотный Adam: веса\nисходный контроль NMSE 0.0323"),
        (
            axes.flat[5],
            query_weights,
            f"Support/query: веса после C-fit\nвыбранный I: test NMSE {primary['test_nmse']:.4g}",
        ),
    ):
        maximum = np.abs(matrix).max()
        image = axis.imshow(matrix[order], cmap="RdBu_r", vmin=-maximum, vmax=maximum)
        axis.set_title(title)
        axis.set_xlabel("Скрытый нейрон")
        axis.set_ylabel("Координата входа")
        figure.colorbar(image, ax=axis, shrink=0.65)
    title = "Полная матрица" if output.name == "full" else "На ранее найденной маске"
    figure.suptitle(f"DeepSets: {title}; hard Gumbel, 4 сэмпла", fontsize=15)
    save_figure(figure, output / "figures/indexators")

    chosen = fitted_rows[primary["row"]]
    matrix = snapshots["policy_probabilities"][chosen["snapshot"], chosen["root"]]
    matrix = matrix[order][:, aligned[f"row_{primary['row']}_columns"]]
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)
    for axis, values, label, maximum in (
        (axes[0], matrix[..., 0], "Вероятность нуля", 1),
        (axes[1], matrix.max(-1), "Вероятность выбранного кода", 1),
        (
            axes[2],
            -(matrix * np.log(np.maximum(matrix, 1e-30))).sum(-1),
            "Энтропия",
            np.log(matrix.shape[-1]),
        ),
    ):
        image = axis.imshow(values, vmin=0, vmax=maximum, cmap="viridis")
        axis.set_title(label)
        axis.set_xlabel("Скрытый нейрон")
        axis.set_ylabel("Координата входа")
        figure.colorbar(image, ax=axis, shrink=0.7)
    figure.suptitle(
        f"{title}: support/query, seed {chosen['seed']}, шаг {chosen['search_step']}"
    )
    save_figure(figure, output / "figures/probabilities")

    history = json.loads((output / "confirmation/history.json").read_text())
    roots = json.loads((output / "confirmation/rows.json").read_text())
    variance = json.loads((output / "specification.json").read_text())[
        "target_variance"
    ]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    for family, label in LABELS.items():
        number = next(
            i for i, r in enumerate(roots) if r["family"] == family and r["seed"] == 223
        )
        steps = [r["step"] for r in history]
        for axis, key in zip(list(axes.flat)[:2], ("train_mse", "query_mse")):
            axis.plot(
                steps,
                [np.mean(r[key][number]) / variance for r in history],
                label=label,
            )
            axis.set_yscale("log")
        axes.flat[2].plot(steps, [r["MAP_edges"][number] for r in history], label=label)
        chosen_rows = [i for i, r in enumerate(fitted_rows) if r["root"] == number]
        axes.flat[3].plot(
            [fitted_rows[i]["search_step"] for i in chosen_rows],
            [fitted["selection_mse"][i].mean() / variance for i in chosen_rows],
            marker="o",
            label=label,
        )
        axes.flat[3].set_yscale("log")
    for axis, label in zip(
        axes.flat,
        (
            "Support NMSE при поиске",
            "Query NMSE при поиске",
            "Число связей в MAP",
            "Selection NMSE после C-fit",
        ),
    ):
        axis.set_title(label)
        axis.set_xlabel("Шаг поиска")
        axis.grid(alpha=0.2)
    axes.flat[3].legend(fontsize=8)
    figure.suptitle(f"{title}: парный seed 223; query обучает P только в support/query")
    save_figure(figure, output / "figures/curves")


def report(root, reference):
    records, costs, diagnostics = [], {}, {}
    for regime in ("full", "masked"):
        output = root / regime
        families = json.loads((output / "summary.json").read_text())["families"]
        records.extend(
            {"regime": regime, "family": name, **families[name]} for name in LABELS
        )
        diagnostics[regime] = search_diagnostics(output)
        figures(output, reference)
        seconds = 0
        forwards = 0
        for phase in ("pilot", "confirmation"):
            complete = json.loads((output / phase / "complete.json").read_text())
            seconds += complete["seconds"]
            forwards += complete["effective_structure_forwards"]
        for name in (
            "pilot/hard_fits.npz",
            "confirmation/hard_fits.npz",
            "fresh_calibration.npz",
            "fresh_evaluation.npz",
        ):
            with np.load(output / name) as archive:
                seconds += float(archive["seconds"])
        costs[regime] = {
            "training_minutes": seconds / 60,
            "structure_forwards": forwards,
        }
    save_json(root / "comparison.json", {"rows": records, "costs": costs})
    text = [
        "# DeepSets: support-C / query-P при hard sampling",
        "",
        "Первая проверка разделения обновлений на одном ранее изученном синтетическом учителе. Это first-order alternating: градиент через численную адаптацию C не вычисляется. Прямой проход выбирает один из девяти кодов [0,C1,…,C4,−C1,…,−C4] на каждую позицию; backward использует смещённый Gumbel-Softmax STE.",
        "",
        "| Область | Метод | Средняя selection NMSE | Средняя test NMSE, 3 seed | Точных масок/3 | Точных I/3 | Связей выбранного I | Средняя test NMSE, 32 свежих C |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in records:
        test_mean = np.mean([r["test_nmse"] for r in row["per_seed"]])
        text.append(
            f"| {'Полная матрица' if row['regime'] == 'full' else 'Выученная маска'} "
            f"| {LABELS[row['family']]} | {row['selection_nmse_mean']:.6g} "
            f"| {test_mean:.6g} | {row['exact_masks']}/3 | {row['exact_indexators']}/3 "
            f"| {row['primary']['recovery']['active_edges']} | {row['fresh']['mean_test_nmse']:.6g} |"
        )
    text.extend(["", "## Результат сравнения", ""])
    for regime in ("full", "masked"):
        subset = {row["family"]: row for row in records if row["regime"] == regime}
        joint, query = subset["joint_support"], subset["alternating_query"]
        ratio = query["selection_nmse_mean"] / joint["selection_nmse_mean"]
        text.append(
            f"- {regime}: отношение средней selection NMSE support/query к совместному контролю — {ratio:.4g}; "
            f"точных масок {query['exact_masks']}/3 против {joint['exact_masks']}/3, "
            f"точных I {query['exact_indexators']}/3 против {joint['exact_indexators']}/3."
        )
    audit = json.loads((root / "gradient_and_pilot_audit.json").read_text())
    pilot_recovery = audit["post_freeze_pilot_joint_masked_recovery"]
    pilot_exact = sum(
        item["recovery"]["categorical_errors"] == 0 for item in pilot_recovery
    )
    masked_joint = next(
        row
        for row in records
        if row["regime"] == "masked" and row["family"] == "joint_support"
    )
    fresh_values = masked_joint["fresh"]["all_test_nmse"]
    text.extend(
        [
            "",
            f"После freeze отдельно проверен пилот совместного обучения на маске: точный I в {pilot_exact}/{len(pilot_recovery)} seed. На confirmation точных I {masked_joint['exact_indexators']}/3. Успех отдельных seed не является устойчивым восстановлением. Лучшая выбранная совместным обучением структура имеет одну ошибку шэринга при правильной маске.",
            "",
            f"Для этой выбранной структуры медиана test NMSE32 свежих C равна {np.median(fresh_values):.6g}, максимум {max(fresh_values):.6g}; {masked_joint['fresh']['successes_below_0_01']}/32 стартов ниже0.01. Усреднённая ошибка включает численные неудачи, но даже хорошие старты не исправляют ошибочное дискретное назначение.",
            "",
            "Записанные нормы градиентов P ненулевые во всех стадиях; максимальная ошибка равенства hard forward выбранному one-hot равна0. Это исключает отсутствие градиента P и подмену hard forward мягкой смесью в этих запусках. Градиент через адаптацию C и отдельная длительная адаптация C для каждой сэмплированной структуры требуют другого эксперимента.",
        ]
    )
    text.extend(
        [
            "",
            "## Ошибки при поиске до фиксации I",
            "",
            "Средние по выбранным source-снимкам трёх seed; support/query усредняются по восьми случайным структурам при текущем C. Selection после C-fit выше относится уже к фиксированному MAP I и не является той же моделью. Снижение query, по которой обучается P, само по себе не доказывает улучшения структуры.",
            "",
            "| Область | Метод | Support NMSE при поиске | Query NMSE при поиске | Связей в MAP, среднее |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for regime, families in diagnostics.items():
        for family, label in LABELS.items():
            values = families[family]
            text.append(
                f"| {regime} | {label} | {np.mean([r['search_support_nmse'] for r in values]):.6g} "
                f"| {np.mean([r['search_query_nmse'] for r in values]):.6g} "
                f"| {np.mean([r['MAP_edges'] for r in values]):.4g} |"
            )
    text.extend(
        [
            "",
            "## Обновления и выбор",
            "",
            "В каждой итерации C обновляется на support с отсоединённым P. Совместный контроль вычисляет градиент P на том же support minibatch при C до обновления. Поочерёдный support/support использует обновлённый C. Поочерёдный support/query использует обновлённый C и query minibatch. В двух поочерёдных вариантах C и численная голова отсоединены от графа архитектурного loss.",
            "",
            "Все три режима имеют одинаковые12000 обновлений C и12000 обновлений P, четыре hard-структуры на каждой стадии, два forward/backward на итерацию. Один кодбук во время поиска. На одном seed совпадают начальные alpha/C, support/query minibatch и Gumbel noise; в двух стадиях итерации используется один и тот же шум. Радиус C нормирован по начальной hard MAP матрице, без soft-to-hard сдвига масштаба при запуске.",
            "",
            "На семью шесть настроек: C lr0.003/0.01 и P lr0.003/0.01/0.03. Радиус0.5; sparsity0.03 для full и0 для masked; поздний entropy0.02; P=softmax(alpha), температура только surrogate1→0.3. Влияние переноса массы и число численных реплик здесь не варьируются.",
            "",
            "Pilot221/222 выбирает одну настройку каждой семьи; confirmation223/224/225 проверяет её без нового подбора. Внутри seed снимок0/4000/8000/12000 выбирается по средней selection ошибке четырёх C-fit: исходный C плюс три свежих Gaussian,3000 support шагов, численные checkpoints выбирает query. Query в support/query участвует в поиске P и не считается независимой валидацией; независимый source-fold selection используется для выбора структуры и гиперпараметров.",
            "",
            "После заморозки структуры и source-настроек выполняется калибровка C-fit и32 новых старта C. Heldout8192 новых входов создаётся после freeze. Все данные относятся к той же функции; это не перенос на другого учителя. Три seed дают предварительную оценку, не достаточную для вывода об устойчивом преимуществе метода.",
            "",
            "Full разрешает все256 позиций, истинное число32 поиску не передаётся. Masked использует ранее найденную DARTS маску; внутри неё нулевой код остаётся доступен. Успех masked означает условное восстановление шэринга, а не полный поиск с нуля.",
            "",
            "## Выбранные индексаторы",
            "",
            "Полные I, C, W и исходные метрики: full/selected_indexators.json и masked/selected_indexators.json. Группы, знаки и перестановки нейронов согласуются с teacher только для аудита и рисунков. Adam — ранее выполненный плотный контроль на этой задаче, здесь не переобучается.",
            "",
            "![Полный поиск](full/figures/indexators.png)",
            "",
            "![Поиск на маске](masked/figures/indexators.png)",
            "",
            "![Вероятности полного поиска](full/figures/probabilities.png)",
            "",
            "![Вероятности на маске](masked/figures/probabilities.png)",
            "",
            "![Кривые полного поиска](full/figures/curves.png)",
            "",
            "![Кривые на маске](masked/figures/curves.png)",
            "",
            "## Затраты",
            "",
        ]
    )
    for regime, cost in costs.items():
        text.append(
            f"- {regime}: измеренное обучение {cost['training_minutes']:.2f} мин.; оценок hard-структур {cost['structure_forwards']}."
        )
    text.extend(
        [
            "",
            "Исходники выполненных стадий, регистрации, P/alpha/C, hard-refit и свежие старты сохранены. Программные тесты не создавались и не запускались.",
            "",
        ]
    )
    (root / "REPORT.md").write_text("\n".join(text))
    save_json(
        root / "complete.json",
        {
            "sha256": {
                str(p.relative_to(root)): digest(p)
                for p in root.rglob("*")
                if p.is_file() and p.name != "complete.json"
            },
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    (args.root / "reporting_code_snapshot.py").write_bytes(Path(__file__).read_bytes())
    if not args.report_only:
        for regime in ("full", "masked"):
            evaluate(args.root / regime)
        audit_search(args.root)
    report(args.root, args.reference)


if __name__ == "__main__":
    main()
