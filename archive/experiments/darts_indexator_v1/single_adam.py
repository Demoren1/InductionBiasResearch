"""Dense Adam controls and weight heatmaps for the fixed Pattern 0110 run."""

import argparse
import itertools
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from pattern.datasets import input_table, labels_for

from .data import BASE, digest, load_archive, save_json
from .fit import fit_fixed, initialization, predict
from .model import metrics
from .report import fitting_curves, save_figure, table

KINDS = ("dense_shared", "dense_full")
LABELS = ("Dense Adam: shared head (91)", "Dense Adam: full MLP (105)")


def fit_rows(data, rows, spec, seed, device):
    initial = initialization(rows, seed)
    for number, row in enumerate(rows):
        if row["kind"] == "dense_shared":
            initial[number, 89:96] = 0
            initial[number, 97:104] = 0
    indices = np.broadcast_to(np.arange(1, 89).reshape(11, 8), (len(rows), 11, 8))
    return fit_fixed(
        data, indices, rows, spec, seed, device, initial_parameters=initial
    )


def calibrate(output, device):
    plan = json.loads((output / "protocol.json").read_text())
    folder = output / "adam_baseline"
    folder.mkdir(exist_ok=False)
    settings = [
        {"lr": rate, "norm": scale}
        for rate, scale in itertools.product((0.001, 0.003, 0.01, 0.03), (0.1, 0.3))
    ]
    protocol = {
        "registered_utc": datetime.now(UTC).isoformat(),
        "requested_by_user": "Compare SGD or Adam heatmaps; Adam chosen",
        "posthoc_extension_after_main_test": True,
        "main_architecture_sha256": digest(output / "frozen.npz"),
        "source_sha256": digest(output / "source.npz"),
        "methods": list(KINDS),
        "settings": settings,
        "calibration_restarts": 8,
        "evaluation_restarts": 32,
        "fit": plan["fit"],
        "rules": [
            "Only fixed Pattern 0110, original 768/256/512/512 observation partitions.",
            "Dense shared-head control has 88 independent weights and the same shared bias/readout/output offset as DARTS: 91 functional parameters. Full MLP has per-hidden biases/readouts: 105 parameters.",
            "Ordinary simultaneous Adam numeric updates, no architecture variables; same 4096 support updates, balanced BCE, L2 and clipping as fixed-I codebook fits.",
            "Query chooses each checkpoint; average selection BCE over eight starts chooses rate/scale separately per method. Then 32 fresh starts, query chooses a single start. Test does not rank baseline settings or starts.",
            "This comparison was requested after main-test inspection. Main DARTS indexator and settings remain frozen. No claim that the extension preceded the main results.",
            "Hidden-column matching to true signed filters uses cosine similarity only for display after baseline selection; no thresholding, quantization, sign flipping or teacher-based model selection.",
        ],
    }
    save_json(folder / "protocol.json", protocol)
    rows = [
        {"kind": kind, "task": 0, "setting": setting, "restart": restart, **value}
        for kind in KINDS
        for setting, value in enumerate(settings)
        for restart in range(protocol["calibration_restarts"])
    ]
    save_json(folder / "calibration_rows.json", rows)
    result = fit_rows(
        load_archive(output / "source.npz"),
        rows,
        plan["fit"],
        plan["seed"] + 15000001,
        device,
    )
    np.savez_compressed(folder / "calibration_fits.npz", **result)
    selected = {}
    for kind in KINDS:
        ranked = []
        for number, value in enumerate(settings):
            ids = [
                i
                for i, row in enumerate(rows)
                if row["kind"] == kind and row["setting"] == number
            ]
            ranked.append(
                {
                    **value,
                    "setting": number,
                    "selection_bce": float(result["selection_metrics"][ids, 0].mean()),
                }
            )
        selected[kind] = {
            "chosen": min(ranked, key=lambda row: row["selection_bce"]),
            "all_settings": ranked,
        }
    save_json(folder / "settings.json", selected)
    print(json.dumps(selected, indent=2), flush=True)


def evaluate(output, device):
    plan = json.loads((output / "protocol.json").read_text())
    folder = output / "adam_baseline"
    settings = json.loads((folder / "settings.json").read_text())
    rows = [
        {"kind": kind, "task": 0, "restart": restart, **settings[kind]["chosen"]}
        for kind in KINDS
        for restart in range(32)
    ]
    result = fit_rows(
        load_archive(output / "source.npz"),
        rows,
        plan["fit"],
        plan["seed"] + 16000001,
        device,
    )
    chosen = [
        int(np.argmin(result["query_metrics"][i * 32 : (i + 1) * 32, 0]) + i * 32)
        for i in range(len(KINDS))
    ]
    np.savez_compressed(
        folder / "frozen.npz",
        parameters=result["parameters"][chosen],
        selected_rows=np.array(chosen),
    )
    save_json(
        folder / "frozen.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "selected_rows": chosen,
            "parameters_sha256": digest(folder / "frozen.npz"),
            "settings_sha256": digest(folder / "settings.json"),
            "baseline_test_evaluated": False,
            "main_test_already_evaluated": True,
        },
    )
    heldout = load_archive(output / "heldout.npz")
    _, population_x = input_table()
    population_y = labels_for(population_x, "0110")
    parameters = torch.tensor(result["parameters"], device=device)
    indices = torch.tensor(result["indices"], dtype=torch.long, device=device)
    full_head = torch.tensor(
        [row["kind"] == "dense_full" for row in rows], device=device
    )
    payloads = {
        "test": (torch.tensor(heldout["test_x"]), torch.tensor(heldout["test_y"][0])),
        "population": (population_x, population_y),
    }
    with torch.no_grad():
        for split, (inputs, target) in payloads.items():
            result[split + "_metrics"] = (
                metrics(
                    predict(inputs.to(device), indices, parameters, full_head),
                    target.to(device)[None],
                )
                .cpu()
                .numpy()
            )
    np.savez_compressed(folder / "evaluation_fits.npz", **result)
    save_json(folder / "evaluation_rows.json", rows)
    summary = {}
    for method, (kind, selected) in enumerate(zip(KINDS, chosen)):
        ids = np.arange(method * 32, (method + 1) * 32)
        summary[kind] = {
            "label": LABELS[method],
            "parameters": 91 if method == 0 else 105,
            "selected_row": selected,
            "setting": settings[kind]["chosen"],
            "mean_test_metrics": result["test_metrics"][ids].mean(0).tolist(),
            "selected_test_metrics": result["test_metrics"][selected].tolist(),
            "selected_population_metrics": result["population_metrics"][
                selected
            ].tolist(),
        }
    save_json(folder / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def weight_heatmap(output, folder, summary):
    truth = np.zeros((11, 8))
    for column in range(8):
        truth[column : column + 4, column] = [-1, 1, 1, -1]
    structures = json.loads((output / "selected_indexators.json").read_text())[
        "primary"
    ]
    frozen = load_archive(output / "frozen.npz")
    codebook = np.r_[0, frozen["parameters"][:2]]
    raw_darts = codebook[frozen["indices"]]
    darts = np.empty_like(raw_darts)
    darts[:, structures["target_columns"]] = raw_darts[:, structures["source_columns"]]
    baseline = load_archive(folder / "evaluation_fits.npz")
    matrices, records = [truth, darts], {}
    for kind in KINDS:
        raw = baseline["parameters"][summary[kind]["selected_row"], :88].reshape(11, 8)
        similarity = (
            raw.T
            @ truth
            / np.maximum(
                np.linalg.norm(raw, axis=0)[:, None]
                * np.linalg.norm(truth, axis=0)[None],
                1e-12,
            )
        )
        source, target = linear_sum_assignment(-similarity)
        aligned = np.empty_like(raw)
        aligned[:, target] = raw[:, source]
        matrices.append(aligned)
        records[kind] = {
            "raw": raw.tolist(),
            "aligned": aligned.tolist(),
            "source_columns": source.tolist(),
            "target_columns": target.tolist(),
            "mean_matched_cosine": float(similarity[source, target].mean()),
        }
    limit = max(float(np.abs(matrix).max()) for matrix in matrices)
    figure, axes = plt.subplots(1, 4, figsize=(18, 6), layout="constrained")
    main = json.loads((output / "evaluation_summary.json").read_text())
    titles = [
        "True signed filters (diagnostic)",
        f"DARTS: fixed I, fine-tuned C\nTest BA {100 * main['inherited_primary_test'][1]:.2f}%",
    ]
    titles += [
        f"{LABELS[i]}\nTest BA {100 * summary[kind]['selected_test_metrics'][1]:.2f}%"
        for i, kind in enumerate(KINDS)
    ]
    for axis, weights, title in zip(axes, matrices, titles):
        plotted = axis.imshow(weights, cmap="RdBu_r", vmin=-limit, vmax=limit)
        axis.set_title(title, fontsize=10)
        axis.set_xticks(range(8))
        axis.set_yticks(range(11))
        axis.set_xlabel("Hidden unit (matched for display)")
        axis.set_ylabel("Input coordinate")
    figure.colorbar(
        plotted, ax=list(axes), shrink=0.8, label="Actual input weight; common scale"
    )
    figure.suptitle(
        "Pattern 0110: actual weights after Adam training\nTeacher matching only reorders hidden columns after selection; no normalization or quantization"
    )
    save_figure(figure, folder, "weight_comparison")
    np.savez_compressed(
        folder / "weight_heatmaps.npz",
        names=np.array(["truth", "darts", *KINDS]),
        weights=np.stack(matrices),
    )
    save_json(
        folder / "weight_heatmaps.json",
        {
            "darts": darts.tolist(),
            "truth": truth.tolist(),
            "dense": records,
            "teacher_used_only_for_display": True,
        },
    )


def report(output):
    folder = output / "adam_baseline"
    summary = json.loads((folder / "summary.json").read_text())
    (folder / "figures").mkdir(exist_ok=True)
    weight_heatmap(output, folder, summary)
    fits = load_archive(folder / "evaluation_fits.npz")
    fitting_curves(
        folder,
        "loss_recall",
        "Dense Adam: all 32 fresh starts averaged per control",
        fits,
        [(LABELS[i], np.arange(i * 32, (i + 1) * 32)) for i in range(2)],
    )
    main = json.loads((output / "evaluation_summary.json").read_text())
    rows = [
        [
            "DARTS: fixed I, fresh C",
            5,
            f"{100 * main['methods']['primary']['mean_test_metrics'][1]:.2f}",
            f"{100 * main['methods']['primary']['selected_test_metrics'][1]:.2f}",
        ]
    ]
    rows += [
        [
            row["label"],
            row["parameters"],
            f"{100 * row['mean_test_metrics'][1]:.2f}",
            f"{100 * row['selected_test_metrics'][1]:.2f}",
        ]
        for row in summary.values()
    ]
    content = [
        "## Дополнение по запросу пользователя: плотный Adam",
        "В DARTS-поиске обновления поочерёдные: три шага C на support при фиксированном P, один шаг логитов P на query при фиксированном C. Unrolled-контроль дифференцирует через один виртуальный Adam-шаг C. После argmax меняются только численные параметры.",
        "Adam-контроли одновременно обучают все численные веса на support. Первый сохраняет общий hidden bias и readout, как DARTS; второй — обычный MLP с отдельными bias/readout. Оба имеют 11 входов и 8 ReLU. 8 режимов LR/масштаба × 8 стартов, средний selection BCE выбирает режим. Затем 32 свежих старта по 4096 шагов; query выбирает checkpoint и старт. Сравнение добавлено после просмотра основного test по новому запросу пользователя; основной I не изменялся, test не выбирал Adam-настройки.",
        table(
            [
                "Метод",
                "Численных параметров",
                "Test BA: среднее стартов, %",
                "Test BA: query-выбор, %",
            ],
            rows,
        ),
        "Хитмапы показывают реальные численные входные веса на общей цветовой шкале. Столбцы переставлены после выбора для визуального сопоставления с истинными окнами. Для Adam использовано сопоставление по косинусу без изменения знаков, масштаба весов или квантования. Bias и readout не показаны на матрице; они сохранены в NPZ.",
        "![Сравнение весов](figures/weight_comparison.png)",
        "![Adam loss/recall](figures/loss_recall.png)",
    ]
    (folder / "REPORT.md").write_text("\n\n".join(content) + "\n")
    main_report = output / "REPORT.md"
    existing = main_report.read_text().split(
        "\n\n## Дополнение по запросу пользователя: плотный Adam"
    )[0]
    addition = "\n\n".join(content).replace("](figures/", "](adam_baseline/figures/")
    main_report.write_text(existing.rstrip() + "\n\n" + addition + "\n")
    for name in ("single_adam.py", "README.md"):
        shutil.copy2(BASE / name, output / "analysis_snapshot" / name)
    save_json(
        folder / "complete.json",
        {
            "completed_utc": datetime.now(UTC).isoformat(),
            "software_tests_run": False,
            "summary": summary,
        },
    )
    complete = json.loads((output / "complete.json").read_text())
    complete["adam_baseline_added_utc"] = datetime.now(UTC).isoformat()
    complete["artifacts"] = {
        str(path.relative_to(output)): digest(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "complete.json"
    }
    save_json(output / "complete.json", complete)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("calibrate", "evaluate", "report"))
    parser.add_argument(
        "--output", type=Path, default=BASE / "results/single_0110_20261104"
    )
    parser.add_argument("--device", default="cuda:6")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == "report":
        report(args.output)
    elif args.stage == "calibrate":
        calibrate(args.output, torch.device(args.device))
    else:
        evaluate(args.output, torch.device(args.device))


if __name__ == "__main__":
    main()
