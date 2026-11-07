"""Audit frozen sine consensus searches using exact Gaussian population risk."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import digest, save_json, teacher_predict
from .evaluate import align_indices, sine_population
from .model import score
from .report import categorical, save_figure
from .sine_capacity import constrained_minimum

METHODS = {
    "learned": "Найденный I, свежий C",
    "shuffled": "Перемешанный I",
    "oracle": "Истинный I",
    "dense_random": "Плотный Adam, случайный старт",
    "dense_moment": "Плотный Adam, moment-start",
}


def capacity(index, direction, spec):
    direction = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    unit = direction / norm
    adjusted = {
        **spec,
        "frequency": spec["frequency"] * norm,
        "low_frequency": spec["low_frequency"] * norm,
    }
    if not np.any(index):
        return {
            "population_mse": float(
                sine_population(
                    np.zeros_like(direction), 0.0, {"direction": direction}, spec
                )
            ),
            "direction_projection_cosine": 0.0,
            "weight_norm": 0.0,
        }
    return constrained_minimum(index, unit, adjusted)


def risks(output, result, row, restart, truth, spec, x, y, floor):
    population = np.maximum(
        sine_population(
            result["weights"][row, ..., 0],
            result["numeric"][row, :, spec["codes"] + 1],
            truth,
            spec,
        ),
        0.0,
    )
    test = score(
        torch.tensor(x),
        torch.tensor(y),
        torch.tensor(result["weights"][row : row + 1]),
        torch.tensor(result["numeric"][row : row + 1]),
        spec,
    ).numpy()[0]
    return {
        "selected_train_mse": float(result["train_mse"][row, restart]),
        "mean_train_mse": float(result["train_mse"][row].mean()),
        "selected_query_mse": float(result["query_mse"][row, restart]),
        "selected_population_mse": float(population[restart]),
        "mean_population_mse": float(population.mean()),
        "selected_test_mse": float(test[restart]),
        "mean_test_mse": float(test.mean()),
        "population_gap": float(population[restart] - floor),
        "below_0_1": int((population < 0.1).sum()),
        "within_0_001_of_floor": int((population <= floor + 0.001).sum()),
        "restarts": len(population),
        "selected_restart": restart,
        "all_population_mse": population.tolist(),
        "all_test_mse": test.tolist(),
    }


def evaluate_source(root, frequency):
    output = root / f"sine_{frequency}"
    frozen = json.loads((output / "source_frozen.json").read_text())
    numerical = json.loads((output / "numerical_frozen.json").read_text())
    if digest(output / "source_frozen.json") != numerical["source_sha256"]:
        raise RuntimeError("Source changed after numerical calibration")
    for descriptor in (frozen, numerical):
        for name, expected in descriptor["sha256"].items():
            if digest(output / name) != expected:
                raise RuntimeError(f"Frozen state changed: {output / name}")
    spec = json.loads((output / "specification.json").read_text())
    plan = json.loads((output / "protocol.json").read_text())
    best_of = {}
    for method in METHODS:
        path = output / "numerics" / f"{method}_calibration.npz"
        with np.load(path) as archive:
            restarts = archive["query_mse"].argmin(-1)
            row = int(
                archive["selection_mse"][np.arange(len(restarts)), restarts].argmin()
            )
        best_of[method] = {
            "row": row,
            "restart": int(restarts[row]),
            "calibration_sha256": digest(path),
        }
    # Additional best-of-restarts deployment rule uses source validation only.
    # Freeze it before reading truth or creating heldout observations.
    best_of_path = output / "numerics/best_of_selection.json"
    if not best_of_path.exists():
        save_json(
            best_of_path,
            {
                "frozen_utc": datetime.now(UTC).isoformat(),
                "criterion": "query chooses one of8calibration starts per numerical setting; selection chooses the setting. Additional source-only deployment diagnostic, distinct from the registered mean8 calibration and32fresh check.",
                "choices": best_of,
            },
        )
    elif json.loads(best_of_path.read_text())["choices"] != best_of:
        raise RuntimeError("Previously frozen best-of source choice changed")
    with np.load(output / "teacher_audit_only.npz") as archive:
        truth = dict(archive)
    floor = capacity(truth["indices"], truth["direction"], spec)["population_mse"]
    test_path = output / "heldout.npz"
    if test_path.exists():
        with np.load(test_path) as archive:
            x, y = archive["test_x"], archive["test_y"]
    else:
        rng = np.random.default_rng(plan["seed"] + 17100000)
        x = rng.normal(size=(spec["test"], spec["inputs"])).astype(np.float32)
        y = teacher_predict(x, truth, spec)
        np.savez_compressed(test_path, test_x=x, test_y=y)
        save_json(
            output / "heldout_manifest.json",
            {
                "created_utc": datetime.now(UTC).isoformat(),
                "numerical_frozen_sha256": digest(output / "numerical_frozen.json"),
                "heldout_sha256": digest(test_path),
            },
        )
    choice = json.loads((output / "architecture_selection.json").read_text())
    with np.load(output / "confirmation/hard_fits.npz") as archive:
        fitted = dict(archive)
    states = {
        "truth_indices": truth["indices"],
        "row_order": np.argsort(truth["indices"][:, 0]),
        "true_weights": spec["frequency"] * truth["direction"],
    }
    configurations = []
    for entry in choice["ranking"]:
        records = []
        for row in entry["per_seed"]:
            number = row["row"]
            restart = int(fitted["query_mse"][number].argmin())
            index = fitted["indices"][number]
            recovery, aligned, _ = align_indices(index, truth["indices"], spec["codes"])
            record = {
                "seed": row["seed"],
                "row": number,
                "search_step": row["search_step"],
                "selection_mse": row["selection_mse"],
                "recovery": recovery,
                "capacity": capacity(index, truth["direction"], spec),
                **risks(output, fitted, number, restart, truth, spec, x, y, floor),
            }
            records.append(record)
            states[f"row_{number}_aligned_indices"] = aligned
        configurations.append(
            {
                "configuration": entry["configuration"],
                "settings": entry["per_seed"][0],
                "selection_mse": entry["selection_mse"],
                "per_seed": records,
                "exact_indexators": sum(
                    row["recovery"]["categorical_errors"] == 0 for row in records
                ),
            }
        )
    primary = choice["primary"]
    source_row = primary["row"]
    index = fitted["indices"][source_row]
    primary_record = next(
        row for row in configurations[0]["per_seed"] if row["row"] == source_row
    )
    states["indices"] = index
    states["hard_weights"] = fitted["weights"][
        source_row, primary_record["selected_restart"], :, 0
    ]
    states["hard_numeric"] = fitted["numeric"][
        source_row, primary_record["selected_restart"]
    ]
    with np.load(output / "confirmation/snapshots.npz") as archive:
        states["latent"] = archive["latent"][primary["snapshot"], primary["root"], :, 0]
    numerical_choices = json.loads((output / "numerics/selection.json").read_text())
    methods = {}
    for method, selected in numerical_choices.items():
        with np.load(output / "numerics" / f"{method}_fresh.npz") as archive:
            result = dict(archive)
        restart = selected["query_selected_restart"]
        methods[method] = {
            "settings": selected["settings"],
            **risks(output, result, 0, restart, truth, spec, x, y, floor),
        }
        states[method + "_weights"] = result["weights"][0, restart, :, 0]
        states[method + "_numeric"] = result["numeric"][0, restart]
    calibrated = {}
    for method, entry in best_of.items():
        with np.load(output / "numerics" / f"{method}_calibration.npz") as archive:
            result = dict(archive)
        calibrated[method] = risks(
            output, result, entry["row"], entry["restart"], truth, spec, x, y, floor
        )
    with np.load(output / "cached_controls/dense_long.npz") as archive:
        result = dict(archive)
    rows = json.loads((output / "cached_controls/dense_long.json").read_text())["rows"]
    number = min(
        (i for i, row in enumerate(rows) if row["method"] == "adam"),
        key=lambda i: result["selection_mse"][i].mean(),
    )
    restart = int(result["query_mse"][number].argmin())
    methods["dense_long_cached"] = {
        "settings": rows[number],
        **risks(output, result, number, restart, truth, spec, x, y, floor),
    }
    summary = {
        "frequency": frequency,
        "population_reference": floor,
        "source_function_previously_studied": True,
        "configurations": configurations,
        "source_primary": primary_record,
        "numerics": methods,
        "best_validation_calibration": calibrated,
    }
    np.savez_compressed(output / "selected_states.npz", **states)
    save_json(output / "summary.json", summary)
    save_json(
        output / "selected_indexator.json",
        {
            "selection": primary,
            "raw_indexator": index.tolist(),
            "aligned_for_display": states[f"row_{source_row}_aligned_indices"].tolist(),
            "fresh_numeric": states["learned_numeric"].tolist(),
            "fresh_weights": states["learned_weights"].tolist(),
        },
    )
    print(
        "SOURCE",
        frequency,
        "floor",
        floor,
        "exactI",
        configurations[0]["exact_indexators"],
        "fresh",
        methods["learned"],
        flush=True,
    )


def evaluate_transfer(root):
    folder = root / "transfer"
    frozen = json.loads((folder / "source_frozen.json").read_text())
    registration = json.loads((root / "transfer_registration.json").read_text())
    if digest(folder / "learned_structures.npz") != frozen["structures_sha256"]:
        raise RuntimeError("Transferred structures changed")
    with np.load(folder / "teacher_audit_only.npz") as archive:
        directions, oracle = archive["directions"], archive["indices"]
    with np.load(folder / "learned_structures.npz") as archive:
        assignments = archive["indices"]
    capacities = {}
    spec = json.loads((folder / "task_0/specification.json").read_text())
    for record, index in zip(frozen["records"], assignments):
        values = [
            capacity(index, direction, spec)["population_mse"]
            for direction in directions
        ]
        capacities[record["method"]] = {
            "mean_population_mse": float(np.mean(values)),
            "below_0_1": int((np.array(values) < 0.1).sum()),
            "directions": len(values),
            "all_population_mse": values,
        }
    tasks = []
    for task in range(registration["trained_functions"]):
        output = folder / f"task_{task}"
        choices = json.loads((output / "choices_frozen.json").read_text())
        for name, expected in choices["sha256"].items():
            if digest(output / name) != expected:
                raise RuntimeError(f"Target state changed: {output / name}")
        spec = json.loads((output / "specification.json").read_text())
        plan = json.loads((output / "protocol.json").read_text())
        truth = {"direction": directions[task]}
        floor = capacity(oracle, directions[task], spec)["population_mse"]
        path = output / "heldout.npz"
        if path.exists():
            with np.load(path) as archive:
                x, y = archive["test_x"], archive["test_y"]
        else:
            rng = np.random.default_rng(plan["seed"] + 19100000 + 100003 * task)
            x = rng.normal(size=(spec["test"], spec["inputs"])).astype(np.float32)
            y = teacher_predict(x, truth, spec)
            np.savez_compressed(path, test_x=x, test_y=y)
            save_json(
                output / "heldout_manifest.json",
                {
                    "created_utc": datetime.now(UTC).isoformat(),
                    "choices_frozen_sha256": digest(output / "choices_frozen.json"),
                    "heldout_sha256": digest(path),
                },
            )
        fits = {}
        methods = {}
        for method, entry in choices["choices"].items():
            name = entry["fit"]
            if name not in fits:
                with np.load(output / (name + ".npz")) as archive:
                    fits[name] = dict(archive)
            methods[method] = {
                "settings": entry["settings"],
                **risks(
                    output,
                    fits[name],
                    entry["row"],
                    entry["restart"],
                    truth,
                    spec,
                    x,
                    y,
                    floor,
                ),
            }
        tasks.append({"task": task, "population_reference": floor, "methods": methods})
    aggregate = {}
    for method in tasks[0]["methods"]:
        aggregate[method] = {
            field: float(np.mean([task["methods"][method][field] for task in tasks]))
            for field in (
                "selected_population_mse",
                "mean_population_mse",
                "selected_test_mse",
                "mean_test_mse",
                "below_0_1",
                "within_0_001_of_floor",
            )
        }
    save_json(
        folder / "summary.json",
        {
            "tasks": tasks,
            "aggregate": aggregate,
            "analytic_capacities_not_training": capacities,
        },
    )


def plots(root, summaries):
    figure, axes = plt.subplots(1, 3, figsize=(14, 8), constrained_layout=True)
    for axis, (frequency, summary) in zip(axes, summaries.items()):
        with np.load(root / f"sine_{frequency}/selected_states.npz") as archive:
            states = dict(archive)
        rows = summary["configurations"][0]["per_seed"]
        matrix = np.column_stack(
            [
                states["truth_indices"],
                *[states[f"row_{row['row']}_aligned_indices"] for row in rows],
            ]
        )
        image = categorical(
            axis,
            matrix[states["row_order"]],
            3,
            f"f={frequency}: точный I {summary['configurations'][0]['exact_indexators']}/3",
        )
        axis.set_xticks(
            range(4),
            [
                "Истина",
                *[
                    f"seed {row['seed']}\nошибок {row['recovery']['categorical_errors']}"
                    for row in rows
                ],
            ],
        )
        axis.set_xlabel("Индексатор после выбора по selection")
    figure.colorbar(
        image,
        ax=list(axes),
        location="bottom",
        ticks=range(-3, 4),
        shrink=0.5,
        fraction=0.035,
    )
    save_figure(figure, root / "figures/indexators")

    figure, axes = plt.subplots(1, 3, figsize=(16, 8), constrained_layout=True)
    for axis, (frequency, summary) in zip(axes, summaries.items()):
        with np.load(root / f"sine_{frequency}/selected_states.npz") as archive:
            states = dict(archive)
        reference = states["true_weights"]
        vectors = [reference]
        for key in (
            "latent",
            "hard_weights",
            "learned_weights",
            "dense_random_weights",
            "dense_moment_weights",
        ):
            vector = states[key]
            vectors.append(vector * (1 if vector @ reference >= 0 else -1))
        matrix = np.column_stack(vectors)[states["row_order"]]
        maximum = np.abs(matrix).max()
        image = axis.imshow(
            matrix, cmap="RdBu_r", vmin=-maximum, vmax=maximum, aspect="auto"
        )
        axis.set_xticks(
            range(6),
            [
                "Истинный\nfv",
                "Z при\nсохранении",
                "I/C после\nrefit",
                "I/C,\nсвежий C",
                "Adam\nrandom",
                "Adam\nmoment",
            ],
            rotation=45,
            ha="right",
        )
        axis.set_title(f"f={frequency}: численные веса, общий масштаб")
        axis.set_ylabel("Координата, сортировка по истинным группам")
        figure.colorbar(image, ax=axis, shrink=0.6)
    save_figure(figure, root / "figures/weights")

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for axis, (frequency, summary) in zip(axes, summaries.items()):
        output = root / f"sine_{frequency}"
        rows = json.loads((output / "confirmation/rows.json").read_text())
        history = json.loads((output / "confirmation/history.json").read_text())
        configuration = summary["configurations"][0]["configuration"]
        ids = [i for i, row in enumerate(rows) if row["configuration"] == configuration]
        for field, label in (
            ("train_mse", "Z: train"),
            ("query_mse", "Z: query"),
            ("hard_query_mse", "C[I]: query до refit"),
        ):
            axis.plot(
                [row["step"] for row in history],
                [np.asarray(row[field])[ids].mean() for row in history],
                label=label,
            )
        axis.axhline(
            summary["population_reference"],
            color="black",
            linestyle=":",
            label="Population-эталон",
        )
        axis.set_title(f"f={frequency}, среднее 3 seed")
        axis.set_xlabel("Обновлений поиска")
        axis.set_ylabel("MSE")
        axis.set_yscale("log")
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    save_figure(figure, root / "figures/search_curves")

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for axis, (frequency, summary) in zip(axes, summaries.items()):
        for method, label in (
            ("learned", "Найденный I"),
            ("oracle", "Истинный I"),
            ("dense_random", "Adam random"),
            ("dense_moment", "Adam moment"),
        ):
            with np.load(
                root / f"sine_{frequency}/numerics/{method}_fresh.npz"
            ) as result:
                line = axis.plot(
                    result["history_steps"],
                    result["history_query"][:, 0].mean(-1),
                    label=label,
                )[0]
                axis.plot(
                    result["history_steps"],
                    result["history_train"][:, 0].mean(-1),
                    color=line.get_color(),
                    linestyle="--",
                    alpha=0.7,
                )
        axis.axhline(summary["population_reference"], color="black", linestyle=":")
        axis.set_title(f"f={frequency}: среднее 32 свежих стартов")
        axis.set_xlabel("Шагов численного обучения")
        axis.set_ylabel("MSE: query сплошная, train пунктир")
        axis.set_yscale("log")
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    save_figure(figure, root / "figures/numerical_curves")

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for axis, (frequency, summary) in zip(axes, summaries.items()):
        output = root / f"sine_{frequency}"
        spec = json.loads((output / "specification.json").read_text())
        with np.load(output / "teacher_audit_only.npz") as archive:
            direction = archive["direction"].astype(float)
        with np.load(output / "selected_states.npz") as archive:
            states = dict(archive)
        unit = direction / np.linalg.norm(direction)
        radii = np.linspace(0.00001, 10, 1001)
        weights = radii[:, None] * unit
        low, high = spec["low_frequency"] * direction, spec["frequency"] * direction
        correlation = sum(
            amplitude
            * 0.5
            * (
                np.exp(-np.square(weights - target).sum(-1) / 2)
                - np.exp(-np.square(weights + target).sum(-1) / 2)
            )
            for amplitude, target in ((spec["low_amplitude"], low), (1.0, high))
        )
        amplitude = correlation / (-0.5 * np.expm1(-2 * radii**2))
        values = sine_population(weights, amplitude, {"direction": direction}, spec)
        axis.plot(radii, values, color="black", label="Вдоль v, оптимальная амплитуда")
        for method, label, marker, size, color in (
            ("learned", "Найденный I + свежий C", "o", 85, "C0"),
            ("dense_random", "Adam random", "x", 50, "C1"),
            ("dense_moment", "Adam moment", "+", 70, "C2"),
            ("oracle", "Истинный I", ".", 25, "C3"),
        ):
            axis.scatter(
                np.linalg.norm(states[method + "_weights"]),
                summary["numerics"][method]["selected_population_mse"],
                label=label,
                s=size,
                marker=marker,
                facecolors="none" if method == "learned" else color,
                edgecolors=color,
                zorder=5 if method == "learned" else 3,
            )
        axis.set_title(f"f={frequency}: Gaussian population risk")
        axis.set_xlabel("Норма вектора весов")
        axis.set_ylabel("MSE")
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    save_figure(figure, root / "figures/landscape")


def report(root):
    summaries = {
        frequency: json.loads((root / f"sine_{frequency}/summary.json").read_text())
        for frequency in (4, 6, 8)
    }
    plots(root, summaries)
    lines = [
        "# Проекционное согласование на сумме двух синусоид",
        "",
        "Проверен лучший вариант предыдущей серии DeepSets: непрерывный Z сохраняется, каждые 50 шагов точная проекция совместно выбирает I/C для Z+U, затем обновляется U. Z не перезаписывается через C[I], кодбук внутри поисковой траектории не обучается отдельным task-SGD.",
        "",
        "Устойчивого выхода к глобальному минимуму не получено. При f=6 выбранный свежий кодбук дал population MSE 0.050383 против эталона 0.049975, 6/32 новых численных стартов оказались в пределах 0.001 от эталона. При f=4/8 ошибки существенно выше. Точный I не восстановлен ни в одном из трёх подтверждающих запусков на каждой частоте. На новых численных функциях перенос найденного шэринга не подтвердился.",
        "",
        "Сохраняется исходная задача: 24 стандартных Gaussian-входа, y=0.4sin(0.7vᵀx)+sin(fvᵀx), f=4/6/8, один неизвестный трёхгрупповой вектор v единичной нормы. Ученик — одна a·sin(wᵀx), без фазы и offset. Поэтому нулевая ошибка недостижима: лучший population-риск около 0.05, плохой локальный — около 0.5. Это одна исходная структура и один v при трёх частотах, не три независимых семейства.",
        "",
        "## Результат на исходной функции",
        "",
        "Ниже точная Gaussian population MSE, без Monte Carlo-шума. Параметры выбраны исключительно по query/selection. Fresh означает 32 новых численных старта фиксированного source-I; таблица отдельно показывает результат query-selected старта и среднее всех 32.",
        "",
        "| f | Population-эталон | Наш I, лучший C по validation | Наш I, fresh selected | Наш I, fresh mean32 | Adam random, fresh selected | Adam moment, fresh selected | Истинный I, fresh selected |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for frequency, summary in summaries.items():
        methods = summary["numerics"]
        lines.append(
            f"| {frequency} | {summary['population_reference']:.7g} | {summary['best_validation_calibration']['learned']['selected_population_mse']:.7g} | {methods['learned']['selected_population_mse']:.7g} | {methods['learned']['mean_population_mse']:.7g} | {methods['dense_random']['selected_population_mse']:.7g} | {methods['dense_moment']['selected_population_mse']:.7g} | {methods['oracle']['selected_population_mse']:.7g} |"
        )
    lines.extend(
        [
            "",
            "![Ландшафт риска и реальные решения](figures/landscape.png)",
            "",
            "Кривая показывает риск для весов вдоль истинного v с аналитически оптимальной амплитудой. Точки — реально обученные параметры; они могут отклоняться от v. Истинное направление используется только при построении рисунка после freeze.",
            "",
            "Столбец «лучший C по validation» — дополнительная проверка правила deployment: query выбирает старт из восьми при каждой настройке, selection выбирает настройку. Это выбор уже обученных calibration-параметров, а не новый32-start прогон. Решение заморожено до открытия population/test; основной fresh-протокол по-прежнему выбирает численную настройку по mean selection восьми стартов, затем проверяет32новых.",
            "",
            "## Структура и численная сходимость",
            "",
            "| f | Точный I | Ошибок выбранного I | Лучший достижимый риск в его подпространстве | Fresh MSE<0.1 | Fresh MSE≤эталон+0.001 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for frequency, summary in summaries.items():
        source, fresh = summary["source_primary"], summary["numerics"]["learned"]
        lines.append(
            f"| {frequency} | {summary['configurations'][0]['exact_indexators']}/3 | {source['recovery']['categorical_errors']}/24 | {source['capacity']['population_mse']:.7g} | {fresh['below_0_1']}/32 | {fresh['within_0_001_of_floor']}/32 |"
        )
    lines.extend(
        [
            "",
            "![Истинные и найденные индексаторы всех трёх стартов](figures/indexators.png)",
            "",
            "![Z, дискретные и плотные веса Adam](figures/weights.png)",
            "",
            "Строки упорядочены по истинным группам только для отображения. Имена/знаки групп выровнены после freeze; при отображении численных весов допускается общий переворот w и амплитуды одной sin. Значения весов не нормируются для рисунка. У источника две группы близки к C/−C, поэтому хорошая ошибка функции ещё не доказывает правильный шэринг.",
            "",
            "## Исходные параметры DeepSets и подбор для синусов",
            "",
            "Сравнивается mean population MSE трёх source-selected hard-C-refit решений. Это этап поиска с четырьмя численными стартами на снимок, до последующей калибровки и 32 свежих C. Обе настройки проверены на тех же seed 403–405.",
            "",
            "| f | Исходные radius0.5/rho100/lambda0.3 | Настройка после пилота | Mean population MSE после подбора |",
            "| --- | ---: | --- | ---: |",
        ]
    )
    for frequency, summary in summaries.items():
        fixed = next(
            row
            for row in summary["configurations"]
            if row["settings"]["radius"] == 0.5
            and row["settings"]["rho_final"] == 100
            and row["settings"]["nonzero_penalty"] == 0.3
        )
        chosen = summary["configurations"][0]
        config = chosen["settings"]
        lines.append(
            f"| {frequency} | {np.mean([row['selected_population_mse'] for row in fixed['per_seed']]):.7g} | radius{config['radius']:g}/rho{config['rho_final']:g}/lambda{config['nonzero_penalty']:g} | {np.mean([row['selected_population_mse'] for row in chosen['per_seed']]):.7g} |"
        )
    lines.extend(
        [
            "",
            "## Протокол",
            "",
            "32 конфигурации до пилота: radius 0.5/2/6/8, rho_final 1/100, lambda 0/0.003/0.03/0.3. Dual=0.25, Adam lr=0.01; 2000 dense warmup+10000 Z-обновлений, проекция каждые 50. Pilot 401/402 выбирает одну настройку; confirmation 403–405 проверяет её и исходные параметры DeepSets (radius0.5,rho100,lambda0.3). Все 24 позиции доступны, число ненулевых связей и истинные группы не задаются. Три независимых численных кода и связанные отрицательные двойники, фиксированный ноль.",
            "",
            "Каждый из четырёх снимков I получает 3000 шагов hard-C-refit: один исходный и три новых Gaussian старта. Query выбирает численные checkpoints, средняя selection четырёх стартов выбирает I/config. Для выбранного I калибруются 12 численных настроек (4 нормы×3 скорости) по mean selection восьми стартов; затем обучаются 32 новых старта. Истинный I, случайно перемешанные группы и плотные Adam random/moment проверяются отдельно. Moment-start использует только mean(support_x*support_y) с 2% jitter. Подбор и ошибки всех стартов лежат в summary.json каждой частоты.",
            "",
            "Support/query/selection 16384/4096/8192 взяты из прежнего эксперимента. Архивные calibration состояния dense/oracle повторно используются после проверки frozen SHA256, одинаковых наблюдений и fit-протокола; их код и происхождение сохранены в cached_controls/provenance.json. Все новые 32-start проверки действительно обучены. Архивный dense-long даёт 2×12000 обновлений вместо 8×3000; его результаты также в summary.json. Нынешнее исследование использует ранее изученную исходную функцию, а не независимую подтверждающую задачу.",
            "",
            "![Поиск: средние train/query всех трёх seed](figures/search_curves.png)",
            "",
            "![Численная адаптация: средние 32 свежих стартов](figures/numerical_curves.png)",
            "",
            "На поисковых кривых train считается по всему support. В численных кривых train — первые 512 support-примеров, query — все 4096; полные support-ошибки выбранных checkpoints сохранены в summary.json. Популяционный риск и новый test16384 открываются после source/numerical freeze и не участвуют в выборе.",
            "",
        ]
    )
    transfer_path = root / "transfer/summary.json"
    if transfer_path.exists():
        transfer = json.loads(transfer_path.read_text())
        lines.extend(
            [
                "## Настоящий перенос",
                "",
                "Четыре новые функции с новыми Gaussian-значениями трёх истинных групп, направления нормированы. Общая целевая частота f=6: все три source-I сравниваются на одинаковых функциях и X. Структуры заморожены до генерации направлений. Каждый I фиксирован; численные параметры полностью новые. 12 настроек×8 стартов×3000 шагов на метод, query checkpoints/start, mean selection настройки, новый test после freeze. Dense random/moment получают тот же численный протокол.",
                "",
                "| Метод | Mean population MSE, query-selected | Mean population MSE, все8 | Mean test MSE, query-selected |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for name, row in transfer["aggregate"].items():
            lines.append(
                f"| {name} | {row['selected_population_mse']:.7g} | {row['mean_population_mse']:.7g} | {row['selected_test_mse']:.7g} |"
            )
        lines.extend(
            [
                "",
                "Дополнительно для 32 новых направлений аналитически оценивается лучший достижимый риск в подпространстве каждого I (одномерная минимизация, амплитуда устраняется аналитически). Это представимость при идеальном численном подборе, не результаты обученного переноса.",
                "",
                "| Source-I | Mean best-achievable population MSE | Направлений с риском<0.1 |",
                "| --- | ---: | ---: |",
            ]
        )
        for name, row in transfer["analytic_capacities_not_training"].items():
            lines.append(
                f"| {name} | {row['mean_population_mse']:.7g} | {row['below_0_1']}/{row['directions']} |"
            )
        names = list(transfer["aggregate"])
        figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
        positions = np.arange(len(names))
        axis.bar(
            positions - 0.18,
            [transfer["aggregate"][name]["selected_population_mse"] for name in names],
            width=0.36,
            label="Query-selected старт",
        )
        axis.bar(
            positions + 0.18,
            [transfer["aggregate"][name]["mean_population_mse"] for name in names],
            width=0.36,
            label="Среднее всех 8",
        )
        axis.axhline(
            summaries[6]["population_reference"],
            color="black",
            linestyle=":",
            label="Population-эталон ≈0.05",
        )
        axis.set_xticks(positions, names, rotation=35, ha="right")
        axis.set_ylabel("Population MSE, среднее четырёх новых функций")
        axis.legend(fontsize=8)
        axis.grid(axis="y", alpha=0.2)
        save_figure(figure, root / "figures/transfer")
        lines.extend(["", "![Реально обученный перенос](figures/transfer.png)", ""])
    seconds = 0.0
    for frequency in summaries:
        output = root / f"sine_{frequency}"
        for phase in ("pilot", "confirmation"):
            seconds += json.loads((output / phase / "complete.json").read_text())[
                "seconds"
            ]
            with np.load(output / phase / "hard_fits.npz") as archive:
                seconds += float(archive["seconds"])
        for path in (output / "numerics").glob("*.npz"):
            with np.load(path) as archive:
                seconds += float(archive["seconds"])
    if transfer_path.exists():
        for output in (root / "transfer").glob("task_*"):
            for name in ("fixed_fits", "dense_random", "dense_moment"):
                with np.load(output / (name + ".npz")) as archive:
                    seconds += float(archive["seconds"])
    lines.extend(
        [
            "",
            f"Измеренное новое обучение, включая поиск/кандидатные refit/калибровку/fresh и новый перенос: {seconds / 60:.2f} мин. Переиспользованные старые calibration-фиты не включены повторно. Суммарная офлайн-стоимость структурного поиска выше одного dense-fit; одинаковые числа шагов не означают одинаковые FLOPs. Программные тесты не создавались и не запускались; Ruff новых файлов прошёл. Исходный frozen-метод DeepSets и старые результаты не менялись.",
            "",
        ]
    )
    (root / "REPORT.md").write_text("\n".join(lines))
    (root / "evaluation_code_snapshot.py").write_bytes(Path(__file__).read_bytes())
    registration = json.loads((root / "registration.json").read_text())
    frozen_method = Path(registration["method"])
    method = json.loads((frozen_method / "method.json").read_text())
    for name, expected in method["copied_sha256"].items():
        if digest(frozen_method / name) != expected:
            raise RuntimeError(f"Original frozen method copy changed: {name}")
    for name in (
        "consensus_search.py",
        "codebook_projection.py",
        "model.py",
        "train.py",
        "run.py",
        "evaluate.py",
        "data.py",
    ):
        if digest(root / "code_snapshot" / name) != digest(
            Path(__file__).parent / name
        ):
            raise RuntimeError(
                f"Shared executed source changed during the experiment: {name}"
            )
    save_json(
        root / "method_integrity.json",
        {
            "checked_utc": datetime.now(UTC).isoformat(),
            "frozen_archive_unchanged": True,
            "shared_modules_match_executed_snapshot": True,
        },
    )
    save_json(
        root / "summary.json",
        {
            "source": summaries,
            "new_training_seconds": seconds,
            "transfer": str(transfer_path.relative_to(root))
            if transfer_path.exists()
            else None,
        },
    )
    save_json(
        root / "complete.json",
        {
            "completed_utc": datetime.now(UTC).isoformat(),
            "sha256": {
                str(path.relative_to(root)): digest(path)
                for path in root.rglob("*")
                if path.is_file() and path.name != "complete.json"
            },
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--stage", choices=("source", "transfer", "report"), required=True
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == "source":
        for frequency in (4, 6, 8):
            evaluate_source(args.root, frequency)
    elif args.stage == "transfer":
        evaluate_transfer(args.root)
    else:
        report(args.root)


if __name__ == "__main__":
    main()
