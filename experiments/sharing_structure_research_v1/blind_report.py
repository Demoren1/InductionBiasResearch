"""Freeze held-out predictions, then audit structures and render G33 results."""

import argparse
import csv
import itertools
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .blind_contract import canonical_index, digest, load_sizes, save_json
from .blind_numeric import numpy_predict

BASE = Path(__file__).resolve().parent
INDEXED = ("geometric_initialization", "functional_polish")
METHODS = ("dense_adam", "dense_repaired", *INDEXED)
LABELS = {
    "dense_adam": "Dense Adam",
    "dense_repaired": "Dense + GN + revival",
    "geometric_initialization": "Geometry + free head",
    "functional_polish": "Indexed + GN",
}


def dense_predict(x, p, sizes):
    d, h, o = [sizes[k] for k in ("inputs", "hidden", "outputs")]
    w = p[: d * h].reshape(d, h)
    b = p[d * h : d * h + h]
    a = p[d * h + h : -o].reshape(h, o)
    return np.maximum(x @ w + b, 0) @ a / math.sqrt(h) + p[-o:]


def freeze(root):
    protocol = json.loads((root / "protocol.json").read_text())
    registration = json.loads((root / "method_frozen.json").read_text())
    for name, expected in registration["code_sha256"].items():
        if digest(BASE / name) != expected:
            raise ValueError(f"Registered learner changed: {name}")
    if digest(root / "data_frozen.json") != registration["data_frozen_sha256"]:
        raise ValueError("Data registration changed after method freeze")
    data = json.loads((root / "data_frozen.json").read_text())
    for name, expected in data["public_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Registered public data changed: {name}")
    receipts, choices = {}, {}
    for row in protocol["cases"]:
        for condition in protocol["conditions"]:
            key = f"{row['name']}_c{condition}"
            folder = root / "fits" / key
            if not (folder / "job_complete.json").exists():
                raise ValueError(f"Learner has not completed: {key}")
            receipt = root / "receipts" / f"{key}.json"
            content = json.loads(receipt.read_text())
            if content["status"] != "complete" or content["violations"]:
                raise ValueError(f"Observation boundary failed: {key}")
            receipts[str(receipt.relative_to(root))] = digest(receipt)
            selected = json.loads((folder / "selected.json").read_text())
            if digest(folder / "selected.npz") != selected["sha256"]["selected.npz"]:
                raise ValueError(f"Frozen selected model changed: {key}")
            for owner, filename in (
                ("fits", "selected.npz"),
                ("dense", "fitted.npz"),
                ("dense", "repaired.npz"),
            ):
                path = root / owner / key / filename
                choices[str(path.relative_to(root))] = digest(path)
    out = root / "evaluation"
    out.mkdir(exist_ok=True)
    if (out / "predictions_frozen.json").exists():
        raise ValueError("Predictions already frozen; use audit without refreezing")
    prediction_hashes = {}

    def deny_private(event, args):
        if (
            event == "open"
            and args
            and not isinstance(args[0], int)
            and "private" in Path(args[0]).parts
        ):
            raise PermissionError("Teacher data closed until predictions freeze")

    sys.addaudithook(deny_private)
    for row in protocol["cases"]:
        case = row["name"]
        sizes = load_sizes(root, case)
        for condition in protocol["conditions"]:
            key = f"{case}_c{condition}"
            chosen = dict(np.load(root / "fits" / key / "selected.npz"))
            dense = np.load(root / "dense" / key / "fitted.npz")["parameters"]
            repaired = np.load(root / "dense" / key / "repaired.npz")["parameters"]
            predictions = {method: [] for method in METHODS}
            with np.load(root / "public" / key / "test.npz") as test:
                for task in range(protocol["sources"]):
                    x = test[f"x_{task}"]
                    for method, parameters in (
                        ("dense_adam", dense),
                        ("dense_repaired", repaired),
                    ):
                        predictions[method].append(
                            dense_predict(x, parameters[task], sizes)
                        )
                    for method in INDEXED:
                        predictions[method].append(
                            numpy_predict(
                                x,
                                chosen[f"{method}_index"],
                                chosen[f"{method}_parameters"][task],
                                sizes["codes"],
                                sizes["outputs"],
                            )
                        )
            path = out / f"{key}_predictions.npz"
            np.savez_compressed(
                path, **{k: np.stack(v) for k, v in predictions.items()}
            )
            prediction_hashes[path.name] = digest(path)
    save_json(
        out / "predictions_frozen.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "code_sha256": registration["code_sha256"],
            "public_data_hashes_verified": len(data["public_sha256"]),
            "selected_models": choices,
            "receipts": receipts,
            "prediction_sha256": prediction_hashes,
            "teacher_opened": False,
            "test_labels_used": False,
            "test_inputs_used_only_after_all_models_selected": True,
            "cases": len(protocol["cases"]),
            "conditions": len(protocol["conditions"]),
        },
    )
    print(json.dumps({"predictions_frozen": len(prediction_hashes)}), flush=True)


def sharing_metrics(aligned, truth):
    left, right = np.triu_indices(truth.size, 1)
    found, true = aligned.ravel(), truth.ravel()
    predicted = (found[left] != 0) & (np.abs(found[left]) == np.abs(found[right]))
    expected = (true[left] != 0) & (np.abs(true[left]) == np.abs(true[right]))
    signs = np.sign(found[left] * found[right]) == np.sign(true[left] * true[right])
    correct = int(np.count_nonzero(predicted & expected & signs))
    precision = correct / max(int(predicted.sum()), 1)
    recall = correct / max(int(expected.sum()), 1)
    return {
        "sharing_precision": precision,
        "sharing_recall": recall,
        "sharing_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "sharing_correct_pairs": correct,
        "sharing_predicted_pairs": int(predicted.sum()),
        "sharing_true_pairs": int(expected.sum()),
    }


def raw_weight(p, sizes):
    d, h = sizes["inputs"], sizes["hidden"]
    return p[: d * h].reshape(d, h)


def match_raw_columns(weight, truth):
    unit = weight / np.maximum(np.linalg.norm(weight, axis=0), 1e-12)
    target = truth / np.maximum(np.linalg.norm(truth, axis=0), 1e-12)
    cost = ((unit[:, :, None] - target[:, None, :]) ** 2).sum(0)
    source, dest = linear_sum_assignment(cost)
    return weight[:, source[np.argsort(dest)]]


def audit(root):
    frozen = json.loads((root / "evaluation/predictions_frozen.json").read_text())
    for filename, expected in frozen["prediction_sha256"].items():
        if digest(root / "evaluation" / filename) != expected:
            raise ValueError(f"Frozen prediction changed: {filename}")
    for filename, expected in frozen["selected_models"].items():
        if digest(root / filename) != expected:
            raise ValueError(f"Frozen model changed: {filename}")
    from experiments.darts_structured_tasks_v1.evaluate import align_indices

    protocol = json.loads((root / "protocol.json").read_text())
    generation = json.loads((root / "private/generation.json").read_text())
    out = root / "evaluation"
    rows, cases = [], []
    matrices = {}
    for definition in generation["cases"]:
        case, dataset = definition["name"], definition["dataset"]
        sizes = load_sizes(root, case)
        truth = dict(np.load(root / "private" / f"{case}.npz"))
        clean = np.stack(
            [
                np.load(root / "private" / f"{case}_clean_test.npz")[f"y_{t}"]
                for t in range(protocol["sources"])
            ]
        )
        cases.append({**definition, "sizes": sizes})
        for condition, sigma in enumerate(generation["noise"]):
            key = f"{case}_c{condition}"
            selected = dict(np.load(root / "fits" / key / "selected.npz"))
            fits = dict(np.load(root / "fits" / key / "fitted.npz"))
            choices = json.loads((root / "fits" / key / "selected.json").read_text())
            noisy = np.stack(
                [
                    np.load(root / "public" / key / "test.npz")[f"y_{t}"]
                    for t in range(protocol["sources"])
                ]
            )
            preds = dict(np.load(out / f"{key}_predictions.npz"))
            true_canonical = canonical_index(truth["index"], sizes["codes"])
            bank_exact = np.all(fits["indices"] == true_canonical, axis=(1, 2))
            sources = [
                json.loads((root / "source" / f"{key}_t{t}" / "trace.json").read_text())
                for t in range(protocol["sources"])
            ]
            matrices[f"{key}_truth"] = truth["index"]
            for method in METHODS:
                value = preds[method]
                row = {
                    "case": case,
                    "dataset": dataset,
                    "variant": definition.get("variant", dataset),
                    "bias_style": definition["bias_style"],
                    "condition": condition,
                    "true_noise_sd_audit_only": sigma,
                    "method": method,
                    "test_clean_mse": float(np.mean((value - clean) ** 2)),
                    "test_noisy_mse": float(np.mean((value - noisy) ** 2)),
                    "test_clean_nmse": float(
                        np.mean((value - clean) ** 2) / np.var(clean)
                    ),
                    "bank_size": len(fits["indices"]),
                    "true_index_in_bank_audit_only": bool(bank_exact.any()),
                }
                if method in INDEXED:
                    metric, aligned, columns = align_indices(
                        selected[f"{method}_index"], truth["index"], sizes["codes"]
                    )
                    row.update(metric)
                    row.update(sharing_metrics(aligned, truth["index"]))
                    row["exact_index"] = metric["categorical_errors"] == 0
                    row["selected_candidate"] = choices["methods"][method]["candidate"]
                    matrices[f"{key}_{method}_raw"] = selected[f"{method}_index"]
                    matrices[f"{key}_{method}_aligned"] = aligned
                    matrices[f"{key}_{method}_columns"] = columns
                rows.append(row)
            save_json(out / f"{key}_noise_estimates.json", sources)
    np.savez_compressed(out / "selected_indices.npz", **matrices)
    save_json(
        out / "selected_indices.json",
        {key: value.tolist() for key, value in matrices.items()},
    )
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (out / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for (dataset, sigma, method), members in itertools.groupby(
        sorted(
            rows,
            key=lambda r: (r["dataset"], r["true_noise_sd_audit_only"], r["method"]),
        ),
        key=lambda r: (r["dataset"], r["true_noise_sd_audit_only"], r["method"]),
    ):
        members = list(members)
        item = {
            "dataset": dataset,
            "sigma_audit_only": sigma,
            "method": method,
            "cases": len(members),
            "mean_clean_mse": float(np.mean([m["test_clean_mse"] for m in members])),
            "mean_clean_nmse": float(np.mean([m["test_clean_nmse"] for m in members])),
            "true_index_in_bank": sum(
                m["true_index_in_bank_audit_only"] for m in members
            ),
        }
        if method in INDEXED:
            item.update(
                {
                    "exact": sum(m["exact_index"] for m in members),
                    "index_errors": sum(m["categorical_errors"] for m in members),
                    "index_positions": sum(m["categorical_positions"] for m in members),
                    **{
                        name: float(np.mean([m[name] for m in members]))
                        for name in (
                            "mask_f1",
                            "mask_recall",
                            "sharing_f1",
                            "sharing_recall",
                        )
                    },
                }
            )
        summary.append(item)
    save_json(out / "results.json", {"rows": rows, "summary": summary, "cases": cases})
    with (out / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(dict.fromkeys(k for r in summary for k in r))
        )
        writer.writeheader()
        writer.writerows(summary)
    render(root, rows, cases, matrices)
    write_report(root, rows, summary)
    save_json(
        out / "audit_complete.json",
        {
            "completed_utc": datetime.now(UTC).isoformat(),
            "predictions_frozen_sha256": digest(out / "predictions_frozen.json"),
            "report_source_sha256": digest(Path(__file__)),
            "report_sha256": digest(root / "REPORT.md"),
            "teacher_used_only_after_predictor_freeze": True,
            "forbidden_reads": 0,
            "rows": len(rows),
            "outputs": {
                p.name: digest(p)
                for p in out.glob("*")
                if p.is_file() and p.name != "audit_complete.json"
            },
        },
    )
    print(json.dumps({"summary": summary}), flush=True)


def write_report(root, rows, summary):
    def aggregate(dataset, sigma, method):
        return next(
            row
            for row in summary
            if (row["dataset"], row["sigma_audit_only"], row["method"])
            == (dataset, sigma, method)
        )

    labels = {
        "pattern_0110": "Pattern 0110",
        "deepsets": "DeepSets",
        "general": "General",
    }
    text = (BASE / "METHOD_CURRENT.md").read_text()
    text = text.replace(
        "(METHOD_G32_KNOWN_PRIORS.md)", "(../../../METHOD_G32_KNOWN_PRIORS.md)"
    )
    text = text.replace(
        "(results/20261107/blind_g33_fixed_confirmation/REPORT.md)",
        "(#независимая-проверка)",
    )
    text += "\n## Независимая проверка\n\n"
    text += (
        "Алгоритм и девять его исходников зафиксированы до основной серии. "
        "Одна программа без информации о семействе запускается на десяти анонимных случаях "
        "с четырьмя source-задачами каждый и двумя парными условиями шума. "
        "Два случая Pattern, два DeepSets и шесть общих геометрий. У Pattern одна схема0110 "
        "с двумя наборами численных параметров. Условия шума парные, "
        "их нельзя считать новыми независимыми структурами. Бюджеты выбраны на старых пилотах. "
        "Истинные параметры основной серии открыты только после сохранения всех выбранных "
        "моделей и предсказаний Q. Перенастройки по этому аудиту не было.\n\n"
        "Pattern использует вещественную ReLU-функцию счётчика 0110; итоговая бинарная "
        "классификация отдельно в новой серии не проверялась. DeepSets действительно суммирует "
        "общую двухканальную ReLU-функцию восьми точек; роли точек переставлены и скрыты от learner. "
        "Общая синтетика содержит local, quartet_mix, random; у части задач b независим по нейронам. "
        "Свободная выходная голова учится во всех семействах. Значение sigma ниже раскрыто только аудитору.\n\n"
        "| Данные | sigma | Точный I: геометрия | Точный I: +GN | Истинный I в банке | Ошибки +GN | Mask F1 +GN | Sharing F1 +GN |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    for dataset, dataset_label in labels.items():
        for sigma in (0.0, 0.2):
            geo = aggregate(dataset, sigma, "geometric_initialization")
            fit = aggregate(dataset, sigma, "functional_polish")
            text += (
                f"| {dataset_label} | {sigma:.2f} | {geo['exact']}/{geo['cases']} | "
                f"{fit['exact']}/{fit['cases']} | {fit['true_index_in_bank']}/{fit['cases']} | "
                f"{fit['index_errors']}/{fit['index_positions']} | {100 * fit['mask_f1']:.2f}% | "
                f"{100 * fit['sharing_f1']:.2f}% |\n"
            )
    text += (
        "\nSharing F1 сравнивает пары активных позиций, которые используют один код, "
        "с проверкой относительного знака. Нулевой фон не раздувает эту метрику. "
        "Ошибки индексов считаются после разрешённых глобальных симметрий; средние F1 равновесны по случаям.\n\n"
        "| Данные | sigma | Dense Adam MSE | Dense +GN+revival MSE | Geometry MSE | Indexed +GN MSE |\n"
        "|---|---:|---:|---:|---:|---:|\n"
    )
    for dataset, dataset_label in labels.items():
        for sigma in (0.0, 0.2):
            errors = [
                aggregate(dataset, sigma, method)["mean_clean_mse"]
                for method in METHODS
            ]
            text += (
                f"| {dataset_label} | {sigma:.2f} | "
                + " | ".join(f"{x:.2e}" for x in errors)
                + " |\n"
            )
    text += (
        "\nMSE измеряется на закрытом чистом ответе Q после фиксации предсказаний; "
        "шумные test MSE и NMSE также сохранены в CSV. Это средние по случаям, "
        "задачам, примерам и выходам, а не выбор лучшего source или test-старта. "
        "Dense Adam не выдаёт дискретного I; структурный успех ему не приписывается.\n\n"
        "## Диагностика выбранных решений\n\n"
    )
    native = [r for r in rows if r["method"] == "functional_polish"]
    failed = [r for r in native if not r["exact_index"]]
    missing = sum(not r["true_index_in_bank_audit_only"] for r in failed)
    selection = len(failed) - missing
    text += (
        f"Численно уточнённый вариант выбрал точный I в {len(native) - len(failed)}/{len(native)} случаях. "
        f"Из {len(failed)} структурных неудач в {missing} истинного I не было в банке; "
        f"в {selection} он был, но функциональный выбор предпочёл другой I. "
        "Первый случай требует улучшения извлечения/банка, второй может быть связан "
        "с численным минимумом, конечным шумом или функциональной неидентифицируемостью. "
        "Эти причины нельзя различить по одной MSE; универсальная причина без отдельного контроля не утверждается.\n\n"
        "| Случай | Данные | sigma | Ошибки geometry | Ошибки +GN | I в банке | Чистая MSE +GN |\n"
        "|---|---|---:|---:|---:|---:|---:|\n"
    )
    for row in native:
        geo = next(
            r
            for r in rows
            if r["case"] == row["case"]
            and r["condition"] == row["condition"]
            and r["method"] == "geometric_initialization"
        )
        text += (
            f"| {row['case']} | {labels[row['dataset']]} | {row['true_noise_sd_audit_only']:.2f} | "
            f"{geo['categorical_errors']} | {row['categorical_errors']} | "
            f"{'да' if row['true_index_in_bank_audit_only'] else 'нет'} | {row['test_clean_mse']:.2e} |\n"
        )
    text += (
        "\nМалая MSE при неверном I не считается структурным успехом. "
        "Результат не доказывает глобальную оптимальность и не является сравнением "
        "с неизменёнными авторскими реализациями LUT-Q/LC. Предыдущие результаты с известным b "
        "имеют иной интерфейс данных и бюджет, поэтому прямое сравнение процентов было бы неверным.\n\n"
        "## Хитмапы, кривые и исходные числа\n\n"
        "Показан первый зарегистрированный случай каждого семейства, source0, шумное условие. "
        "Плотные веса — реальные обученные веса, только столбцы сопоставлены для просмотра; "
        "они не масштабированы до teacher. У каждой матрицы весов своя цветовая шкала. "
        "Категориальные матрицы согласованы только разрешёнными симметриями.\n\n"
        "![Хитмапы](evaluation/representative_heatmaps.png)\n\n"
        "![Все индексаторы без шума](evaluation/all_indices_c0.png)\n\n"
        "![Все индексаторы с шумом](evaluation/all_indices_c1.png)\n\n"
        "Loss показан по реально измеренным шагам. На этапе GN индексатор фиксирован, "
        "поэтому изменение recall во время численной подгонки не рисуется.\n\n"
        "![Train и H0](evaluation/training_curves.png)\n\n"
        "![Recall](evaluation/structural_recall.png)\n\n"
        "- [Полная таблица CSV](evaluation/comparison.csv), [сводная CSV](evaluation/summary.csv).\n"
        "- [Выбранные индексаторы JSON](evaluation/selected_indices.json), [NPZ](evaluation/selected_indices.npz).\n"
        "- [Заморозка предсказаний](evaluation/predictions_frozen.json), [результаты JSON](evaluation/results.json).\n"
        "- [Фактическая версия исходников](method_frozen.json), квитанции реальных чтений — receipts/.\n\n"
        "Основная серия не использует повторные измерения. Высокоточный активный pilot03 "
        "восстановил три пилотных I, но потребовал порядка10^11–10^12 эквивалентных ответов/source. "
        "Экономный активный контроль терял гиперплоскости. Пилот ориентаций улучшил "
        "численную подгонку DeepSets, но оставил9/256 ошибок выбранного I; "
        "это отрицательный пилот, а не подтверждение основной серии.\n"
    )
    (root / "REPORT.md").write_text(text)


def render(root, rows, cases, matrices):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    plt.rcParams.update({"font.size": 9, "figure.dpi": 130})
    out = root / "evaluation"
    colors = [
        "#912c7c",
        "#c05354",
        "#d78942",
        "#c5b242",
        "#f7f7f7",
        "#58a7a1",
        "#3776b5",
        "#6657a5",
        "#358350",
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-4.5, 5), 9)
    for condition in (0, 1):
        fig, axes = plt.subplots(
            len(cases), 3, figsize=(11, 2 * len(cases)), squeeze=False
        )
        for r, case in enumerate(cases):
            key = f"{case['name']}_c{condition}"
            for col, method in enumerate(("truth", *INDEXED)):
                matrix = matrices[
                    f"{key}_truth" if method == "truth" else f"{key}_{method}_aligned"
                ]
                axes[r, col].imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
                title = f"{case['name']} | {case['dataset']} | "
                if method == "truth":
                    title += "true I (audit)"
                else:
                    row = next(
                        a
                        for a in rows
                        if a["case"] == case["name"]
                        and a["condition"] == condition
                        and a["method"] == method
                    )
                    title += f"{LABELS[method]}: {row['categorical_errors']} errors"
                axes[r, col].set_title(title, fontsize=9)
                axes[r, col].set_xlabel("Hidden unit")
                axes[r, col].set_ylabel("Input coordinate")
        fig.tight_layout()
        fig.savefig(out / f"all_indices_c{condition}.png")
        plt.close(fig)
    representatives = [
        next(c for c in cases if c["dataset"] == name)
        for name in ("pattern_0110", "deepsets", "general")
    ]
    fig, axes = plt.subplots(3, 4, figsize=(15, 9))
    for r, case in enumerate(representatives):
        key, sizes = f"{case['name']}_c1", case["sizes"]
        truth = dict(np.load(root / "private" / f"{case['name']}.npz"))
        true_w = (
            np.sign(truth["index"])
            * np.r_[0, truth["codes"][0]][np.abs(truth["index"])]
        )
        p = np.load(root / "dense" / key / "fitted.npz")["parameters"][0]
        dense_w = match_raw_columns(raw_weight(p, sizes), true_w)
        for col, matrix in enumerate((true_w, dense_w)):
            bound = max(float(np.max(np.abs(matrix))), 1e-12)
            image = axes[r, col].imshow(
                matrix, cmap="coolwarm", vmin=-bound, vmax=bound, aspect="auto"
            )
            fig.colorbar(image, ax=axes[r, col], shrink=0.75)
            axes[r, col].set_title(
                f"{case['dataset']} | "
                + ("true W1" if col == 0 else "actual Dense Adam W1")
            )
        for col, method in enumerate(INDEXED, 2):
            matrix = matrices[f"{key}_{method}_aligned"]
            k = sizes["codes"]
            local_cmap = ListedColormap(colors[4 - k : 5 + k])
            local_norm = BoundaryNorm(np.arange(-k - 0.5, k + 1), 2 * k + 1)
            image = axes[r, col].imshow(
                matrix, cmap=local_cmap, norm=local_norm, aspect="auto"
            )
            fig.colorbar(
                image,
                ax=axes[r, col],
                ticks=np.arange(-sizes["codes"], sizes["codes"] + 1),
                shrink=0.75,
            )
            row = next(
                a
                for a in rows
                if a["case"] == case["name"]
                and a["condition"] == 1
                and a["method"] == method
            )
            axes[r, col].set_title(
                f"{LABELS[method]} | {row['categorical_errors']} index errors"
            )
        for ax in axes[r]:
            ax.set_xlabel("Hidden unit (audit alignment)")
            ax.set_ylabel("Input coordinate")
    fig.suptitle(
        "First registered case per family, source 0, noisy condition; raw dense weights are not rescaled",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "representative_heatmaps.png")
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for r, case in enumerate(representatives):
        key = f"{case['name']}_c1"
        dense = dict(np.load(root / "dense" / key / "fitted.npz"))
        config = json.loads((root / "dense" / key / "frozen.json").read_text())
        start = config["selected_starts"][0]
        factor = dense["scale"][0] ** 2
        axes[r, 0].semilogy(
            dense["steps"],
            np.maximum(dense["train_history"][:, start] * factor, 1e-16),
            label="Train",
        )
        axes[r, 0].semilogy(
            dense["steps"],
            np.maximum(dense["validation_history"][:, start] * factor, 1e-16),
            label="H0",
        )
        axes[r, 0].set_title(f"{case['dataset']}: selected Adam start, source 0")
        axes[r, 0].set_xlabel("Adam step")
        fits = dict(np.load(root / "fits" / key / "fitted.npz"))
        chosen = json.loads((root / "fits" / key / "selected.json").read_text())[
            "methods"
        ]["functional_polish"]["candidate"]
        trial = int(fits["validation_history"][:, chosen, 0].min(0).argmin())
        for name, label in (("train_history", "Train"), ("validation_history", "H0")):
            values = fits[name][:, chosen, 0, trial]
            axes[r, 1].semilogy(
                np.arange(len(values)), np.maximum(values, 1e-16), label=label
            )
        axes[r, 1].set_title("Indexed GN, source 0 (I fixed)")
        axes[r, 1].set_xlabel("Gauss-Newton step (I fixed)")
        for ax in axes[r]:
            ax.set_ylabel("Measured MSE")
            ax.legend()
            ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "training_curves.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, field, title in zip(
        axes,
        ("mask_recall", "sharing_recall"),
        ("Mask recall", "Signed sharing pair recall"),
    ):
        for offset, method in enumerate(INDEXED):
            values = [
                np.mean(
                    [
                        r[field]
                        for r in rows
                        if r["dataset"] == name
                        and r["condition"] == 1
                        and r["method"] == method
                    ]
                )
                for name in ("pattern_0110", "deepsets", "general")
            ]
            ax.bar(
                np.arange(3) + (offset - 0.5) * 0.35,
                values,
                width=0.35,
                label=LABELS[method],
            )
        ax.set_xticks(np.arange(3), ["Pattern 0110", "DeepSets", "General"])
        ax.set_ylim(0, 1.05)
        ax.set_title(title + " (audit only)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "structural_recall.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("freeze", "audit"), required=True)
    args = parser.parse_args()
    (freeze if args.phase == "freeze" else audit)(args.root.resolve())


if __name__ == "__main__":
    main()
