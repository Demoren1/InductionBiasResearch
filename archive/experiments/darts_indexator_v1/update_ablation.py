"""Paired ablation of alternating and simultaneous P/C updates on Pattern 0110."""

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pattern.datasets import input_table, labels_for

from .data import BASE, digest, load_archive, save_json
from .model import metrics, soft_predict
from .report import fitting_curves, save_figure, table
from .search import train
from .single import (
    collect,
    combined,
    fine_tune,
    fit_fixed,
    fresh_parameters,
    predict,
    select,
)
from .single_report import align, draw_index

MODES = {
    "alternate3": ("alternating", 3, "Alternate 3:1"),
    "alternate1": ("alternating", 1, "Alternate 1:1"),
    "synchronous_split": ("synchronous_split", 1, "Synchronous: two losses"),
    "joint_support": ("joint_support", 1, "Joint: support loss"),
}


def prepare(output):
    original = json.loads((output / "protocol.json").read_text())
    folder = output / "update_ablation"
    folder.mkdir(exist_ok=False)
    save_json(
        folder / "protocol.json",
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "user_request": "Ablate alternating versus joint codebook and indexator training",
            "posthoc_after_main_test": True,
            "modes": {
                name: {
                    "update_mode": value[0],
                    "numeric_steps_per_outer": value[1],
                    "label": value[2],
                }
                for name, value in MODES.items()
            },
            "search_seeds": [5, 6, 7],
            "fixed_configuration_ids": [10, 11],
            "source_sha256": digest(output / "source.npz"),
            "main_frozen_sha256": digest(output / "frozen.npz"),
            "rules": [
                "Original fixed Pattern 0110, three indices C0=0/C1/C2, same 11x8 ReLU model and five numerical parameters, four ongoing numerical replicas.",
                "Use the two main-search selected rate/annealing/sparsity configurations: numeric_lr=.03, alpha_lr=.03, annealed, sparsity=0/.01. No further tuning based on ablation test or true structure.",
                "Three new paired search seeds. Identical alpha/C initialization and 512 numeric warmup. Every outer step draws the same three support batches and independent query batch in every mode. Modes with one numeric update use the first support batch and discard the other two.",
                "Alternate3 applies three support C steps then a query P step; alternate1 one support C step then a query P step. Synchronous split calculates both partial gradients at the same old state (C on support, P on query), then applies both Adam updates.",
                "Joint support calculates C and P gradients of the same support loss at the same old state and applies both updates. Query supplies no architecture gradient in this mode. Same P entropy/sparsity and C L2 penalties; no sign/order constraints after initialization.",
                "All modes have 6000 P updates; alternate3 has 18000 C updates plus warmup, other modes 6000 plus warmup. This explicitly ablates the ratio, not an equal numeric-update budget. Runtime and update counts are saved.",
                "At the same five saved checkpoints take argmax and continue inherited C for 4096 steps. Query chooses numeric checkpoints; average selection BCE over four C picks per-seed checkpoints. Average over three search seeds chooses configuration separately per mode; primary seed by selection BCE.",
                "Freeze all mode winners before ablation test evaluation. Report all three search seeds per mode, with 32 fresh C starts and the already-frozen main numeric setting. Test and true structural agreement do not rank candidates.",
            ],
        },
    )
    for name, (mode, count, _) in MODES.items():
        child = folder / name
        child.mkdir()
        plan = json.loads(json.dumps(original))
        plan["output"] = str(child)
        plan["search"].update(
            {
                "pilot_seeds": [5, 6, 7],
                "fixed_configuration_ids": [10, 11],
                "update_mode": mode,
                "numeric_steps_per_outer": count,
                "paired_batch_streams": True,
            }
        )
        save_json(child / "protocol.json", plan)
        shutil.copy2(output / "source.npz", child / "source.npz")
        shutil.copy2(output / "heldout.npz", child / "heldout.npz")
    snapshot = folder / "code_snapshot"
    snapshot.mkdir()
    for path in BASE.iterdir():
        if path.suffix in (".py", ".json", ".md"):
            shutil.copy2(path, snapshot / path.name)
    print("PREPARED", folder, flush=True)


def run_mode(output, name, device):
    child = output / "update_ablation" / name
    plan = json.loads((child / "protocol.json").read_text())
    train(plan, child, "pilot", device, 0, 1)
    collect(plan, child, "pilot")
    fine_tune(plan, child, "pilot", device, 0, 1)
    select(plan, child, "pilot")


def freeze(output):
    folder = output / "update_ablation"
    selected, matrices, parameters = [], [], []
    for name in MODES:
        child = folder / name
        selection = json.loads((child / "pilot_selection.json").read_text())
        candidates = load_archive(child / "pilot/hard_candidates.npz")
        fits = combined(child / "pilot")
        for seed in selection["selected_group"]["seeds"]:
            ids = np.flatnonzero(fits["entry_ids"] == seed["entry"])
            chosen = ids[np.argmin(fits["query_metrics"][ids, 0])]
            selected.append(
                {
                    "mode": name,
                    "label": MODES[name][2],
                    "source_selection": seed,
                    "primary": seed["entry"] == selection["primary"]["entry"],
                    "selected_group_mean_bce": selection["selected_group"][
                        "mean_selection_bce"
                    ],
                }
            )
            matrices.append(candidates["indices"][seed["entry"]])
            parameters.append(fits["parameters"][chosen])
    np.savez_compressed(
        folder / "frozen.npz",
        indices=np.stack(matrices),
        parameters=np.stack(parameters),
    )
    save_json(
        folder / "frozen.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "rows": selected,
            "state_sha256": digest(folder / "frozen.npz"),
            "main_fresh_setting_sha256": digest(output / "fresh_setting.json"),
            "ablation_test_evaluated": False,
            "main_test_already_evaluated": True,
        },
    )
    print("FROZEN", len(selected), "indexators", flush=True)


def validation_audit(output):
    folder = output / "update_ablation"
    (folder / "figures").mkdir(exist_ok=True)
    data = load_archive(output / "source.npz")
    inputs = torch.tensor(data["selection_x"])
    target = torch.tensor(data["selection_y"])[None, :, None]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    audit = {}
    for axis, (name, (_, _, label)) in zip(axes.ravel(), MODES.items()):
        child = folder / name
        selection = json.loads((child / "pilot_selection.json").read_text())
        seeds = selection["selected_group"]["seeds"]
        origins = [row["origin"] for row in seeds]
        snapshot = load_archive(child / origins[0]["folder"] / "snapshots.npz")
        history = json.loads(
            (child / origins[0]["folder"] / "history.json").read_text()
        )
        root_ids = [origin["root"] for origin in origins]
        fits = combined(child / "pilot")
        entries = json.loads((child / "pilot/hard_candidates.json").read_text())
        curves = {
            key: []
            for key in (
                "train",
                "query",
                "soft_validation",
                "hard_validation",
                "fine_tuned_validation",
            )
        }
        for number, step in enumerate(snapshot["steps"]):
            record = next(row for row in history if row["step"] == step)
            temperature = torch.tensor(record["temperature"])[:, None, None, None]
            probabilities = (
                torch.tensor(snapshot["alpha"][number]) / temperature
            ).softmax(-1)
            numeric = torch.tensor(snapshot["numeric"][number])
            hard = torch.nn.functional.one_hot(probabilities.argmax(-1), 3).float()
            with torch.no_grad():
                soft_score = (
                    metrics(soft_predict(inputs, probabilities, numeric), target)
                    .mean((1, 2))
                    .numpy()
                )
                hard_score = (
                    metrics(soft_predict(inputs, hard, numeric), target)
                    .mean((1, 2))
                    .numpy()
                )
            curves["train"].append(float(np.array(record["train"])[root_ids, 0].mean()))
            curves["query"].append(float(np.array(record["query"])[root_ids, 0].mean()))
            curves["soft_validation"].append(float(soft_score[root_ids, 0].mean()))
            curves["hard_validation"].append(float(hard_score[root_ids, 0].mean()))
            selected_entries = [
                entry["entry"]
                for entry in entries
                if entry["root"] in root_ids and entry["step"] == step
            ]
            fit_ids = np.isin(fits["entry_ids"], selected_entries)
            curves["fine_tuned_validation"].append(
                float(fits["selection_metrics"][fit_ids, 0].mean())
            )
        steps = snapshot["steps"].tolist()
        for key, title in (
            ("train", "Support soft"),
            ("query", "Architecture-query soft"),
            ("soft_validation", "Independent validation: soft"),
            ("fine_tuned_validation", "Independent validation: argmax + C fit"),
        ):
            axis.plot(steps, curves[key], marker="o", label=title)
        axis.set_title(label)
        axis.set_xlabel("P updates")
        axis.set_ylabel("Balanced BCE, mean over 3 paired seeds")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
        audit[name] = {
            "configuration": selection["selected_group"]["configuration"],
            "steps": steps,
            **curves,
            "primary_selected_step": selection["primary"]["origin"]["step"],
            "mean_selected_hard_validation_bce": selection["selected_group"][
                "mean_selection_bce"
            ],
            "best_soft_validation_step": steps[
                int(np.argmin(curves["soft_validation"]))
            ],
            "validation_enters_gradients": False,
        }
    figure.suptitle(
        "True validation, separate from architecture-query; measured before ablation test\nSame configuration across all three seeds per mode; selection uses hard I after C fitting"
    )
    save_figure(figure, folder, "validation_loss")
    save_json(
        folder / "validation_audit.json",
        {
            "audited_utc": datetime.now(UTC).isoformat(),
            "ablation_test_evaluated": False,
            "modes": audit,
        },
    )
    print(json.dumps(audit, indent=2), flush=True)


def evaluate(output, device):
    folder = output / "update_ablation"
    plan = json.loads((output / "protocol.json").read_text())
    frozen = json.loads((folder / "frozen.json").read_text())
    state = load_archive(folder / "frozen.npz")
    setting = json.loads((output / "fresh_setting.json").read_text())
    restarts = 32
    initial = fresh_parameters(restarts, setting["norm"], plan["seed"] + 17000001)
    indices = np.repeat(state["indices"], restarts, axis=0)
    fits = fit_fixed(
        load_archive(output / "source.npz"),
        indices,
        np.tile(initial, (len(state["indices"]), 1)),
        [setting["lr"]] * len(indices),
        plan["fit"],
        plan["seed"] + 17000001,
        device,
    )
    heldout = load_archive(output / "heldout.npz")
    _, population_x = input_table()
    population_y = labels_for(population_x, "0110")
    payloads = {
        "test": (torch.tensor(heldout["test_x"]), torch.tensor(heldout["test_y"][0])),
        "population": (population_x, population_y),
    }
    index_tensor = torch.tensor(indices, dtype=torch.long, device=device)
    numeric = torch.tensor(fits["parameters"], device=device)
    warm_index = torch.tensor(state["indices"], dtype=torch.long, device=device)
    warm_numeric = torch.tensor(state["parameters"], device=device)
    warm = {}
    with torch.no_grad():
        for split, (x, y) in payloads.items():
            fits[split + "_metrics"] = (
                metrics(
                    predict(x.to(device), index_tensor, numeric), y.to(device)[None]
                )
                .cpu()
                .numpy()
            )
            warm[split] = (
                metrics(
                    predict(x.to(device), warm_index, warm_numeric), y.to(device)[None]
                )
                .cpu()
                .numpy()
            )
    np.savez_compressed(folder / "evaluation_fits.npz", **fits)
    records = []
    for number, row in enumerate(frozen["rows"]):
        ids = np.arange(number * restarts, (number + 1) * restarts)
        chosen = ids[np.argmin(fits["query_metrics"][ids, 0])]
        records.append(
            {
                **row,
                "fit_number": number,
                "selected_row": int(chosen),
                "mean_test_metrics": fits["test_metrics"][ids].mean(0).tolist(),
                "selected_test_metrics": fits["test_metrics"][chosen].tolist(),
                "selected_population_metrics": fits["population_metrics"][
                    chosen
                ].tolist(),
                "inherited_test_metrics": warm["test"][number].tolist(),
                "inherited_population_metrics": warm["population"][number].tolist(),
            }
        )
    save_json(folder / "summary.json", records)
    print(json.dumps(records, indent=2), flush=True)


def report(output):
    folder = output / "update_ablation"
    (folder / "figures").mkdir(exist_ok=True)
    records = json.loads((folder / "summary.json").read_text())
    state = load_archive(folder / "frozen.npz")
    truth = np.zeros((11, 8), dtype=int)
    for column in range(8):
        truth[column : column + 4, column] = [1, 2, 2, 1]
    structures, primary = [], []
    for row, indices in zip(records, state["indices"]):
        aligned, structural = align(indices, truth)
        structures.append(
            {**row, **structural, "raw": indices.tolist(), "aligned": aligned.tolist()}
        )
        if row["primary"]:
            primary.append(structures[-1])
    save_json(folder / "selected_indexators.json", structures)
    main_i = json.loads((output / "selected_indexators.json").read_text())["primary"]
    panels = [
        (truth, "True I (diagnostic)"),
        (np.array(main_i["aligned"]), "Original alternate 3:1; 12/88 errors"),
    ]
    panels += [
        (
            np.array(row["aligned"]),
            f"{row['label']}; {row['categorical_errors']}/88 errors",
        )
        for row in primary
    ]
    figure, axes = plt.subplots(2, 3, figsize=(14, 11), layout="constrained")
    for axis, (matrix, title) in zip(axes.ravel(), panels):
        draw_index(axis, matrix, title)
    figure.suptitle(
        "Pattern 0110: paired update ablation; selected indexators\nSame paired search seeds; all teacher alignment is post-selection only"
    )
    save_figure(figure, folder, "indexator_comparison")
    baseline = load_archive(output / "adam_baseline/weight_heatmaps.npz")
    weights = [baseline["weights"][0], baseline["weights"][1]]
    titles = ["True signed filters", "Original DARTS: fine-tuned C"]
    for row in primary:
        number = row["fit_number"]
        raw = np.r_[0, state["parameters"][number, :2]][state["indices"][number]]
        aligned = np.empty_like(raw)
        aligned[:, row["target_columns"]] = raw[:, row["source_columns"]]
        weights.append(aligned)
        titles.append(
            f"{row['label']}\nInherited C test BA {100 * row['inherited_test_metrics'][1]:.2f}%"
        )
    weights += [baseline["weights"][2], baseline["weights"][3]]
    titles += ["Dense Adam: shared head", "Dense Adam: full MLP"]
    bound = max(float(np.abs(weight).max()) for weight in weights)
    figure, axes = plt.subplots(2, 4, figsize=(17, 11), layout="constrained")
    for axis, weight, title in zip(axes.ravel(), weights, titles):
        plotted = axis.imshow(weight, cmap="RdBu_r", vmin=-bound, vmax=bound)
        axis.set_title(title, fontsize=10)
        axis.set_xticks(range(8))
        axis.set_yticks(range(11))
        axis.set_xlabel("Hidden unit (matched for display)")
        axis.set_ylabel("Input coordinate")
    figure.colorbar(
        plotted,
        ax=list(axes.ravel()),
        shrink=0.7,
        label="Actual weight; common color scale",
    )
    figure.suptitle(
        "Pattern 0110: weights after freezing I and fine-tuning C; dense Adam controls\nNo weight normalization, sign flips or quantization"
    )
    save_figure(figure, folder, "weight_comparison")
    np.savez_compressed(
        folder / "weight_heatmaps.npz",
        weights=np.stack(weights),
        titles=np.array(titles),
    )
    fits = load_archive(folder / "evaluation_fits.npz")
    fitting_curves(
        folder,
        "fresh_codebook_loss_recall",
        "Paired update ablation: 32 fresh C starts per selected primary I",
        fits,
        [
            (
                row["label"],
                np.arange(row["fit_number"] * 32, (row["fit_number"] + 1) * 32),
            )
            for row in primary
        ],
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for row in primary:
        origin = row["source_selection"]["origin"]
        history = json.loads(
            (folder / row["mode"] / origin["folder"] / "history.json").read_text()
        )
        steps = [record["step"] for record in history]
        root = origin["root"]
        for axis, key, component, scale in (
            (axes[0, 0], "train", 0, 1),
            (axes[0, 1], "query", 0, 1),
            (axes[1, 0], "query", 2, 100),
            (axes[1, 1], "entropy", None, 1),
        ):
            values = [
                record[key][root] if component is None else record[key][root][component]
                for record in history
            ]
            axis.plot(steps, np.array(values) * scale, label=row["label"])
    for axis, ylabel in zip(
        axes.ravel(),
        ("Support soft BCE", "Query soft BCE", "Query recall, %", "Assignment entropy"),
    ):
        axis.set_xlabel("P updates")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    save_figure(figure, folder, "search_loss_recall")
    rows, fresh_rows = [], []
    for row in primary:
        seeds = sorted(
            [entry for entry in structures if entry["mode"] == row["mode"]],
            key=lambda entry: entry["source_selection"]["origin"]["search_seed"],
        )
        rows.append(
            [
                row["label"],
                row["source_selection"]["origin"]["id"],
                f"{row['selected_group_mean_bce']:.2f}",
                ", ".join(
                    f"{100 * seed['inherited_test_metrics'][1]:.2f}" for seed in seeds
                ),
                f"{100 * np.mean([seed['inherited_test_metrics'][1] for seed in seeds]):.2f}",
                f"{100 * row['inherited_test_metrics'][1]:.2f}",
                row["categorical_errors"],
                sum(seed["categorical_exact"] for seed in seeds),
            ]
        )
        fresh_rows.append(
            [
                row["label"],
                f"{100 * np.mean([seed['selected_test_metrics'][1] for seed in seeds]):.2f}",
                f"{100 * row['selected_test_metrics'][1]:.2f}",
                f"{100 * row['mean_test_metrics'][1]:.2f}",
            ]
        )
    lines = [
        "## Абляция по запросу пользователя: порядок обновлений P и C",
        "Это дополнение после первоначального test. Исходный выбор не меняется. Две заранее выбранные настройки поиска проверены на трёх новых парных seed 5/6/7. Начальные P/C и warmup совпадают; на каждом внешнем шаге совпадают три support minibatch и независимый query minibatch. 1:1 варианты используют первый support batch. Всего 6000 обновлений P; 3:1 получает 18000 обновлений C, остальные 6000, плюс общий warmup512. Таким образом отдельно меняются соотношение обновлений, их порядок и обучающий loss.",
        "Alternate3/1: сначала C на support, затем P на query. Synchronous split: оба градиента вычисляются до обновления чисел, C по support и P по query. Joint support: оба градиента вычисляются по одному support loss; query не обучает P. Везде независимые состояния Adam, одинаковые регуляризаторы, четыре кодбука и argmax с последующим дообучением C4096.",
        "Checkpoint и конфигурация выбираются по прежнему source-only правилу после дообучения исходных C. Для всех трёх I выбранной конфигурации — по32 свежих C с общей ранее зафиксированной численной настройкой. Старт C выбирается поquery; первичный search seed — поselection. Ни test, ни совпадение с истинным I не ранжируют модели.",
        "Основная таблица: продолжаем исходные C после фиксации I, как в предложенном пользователем алгоритме. Три seed перечислены в порядке5/6/7. Свежие кодбуки — отдельный контроль ниже.",
        table(
            [
                "Режим",
                "Config",
                "Mean selection BCE",
                "Test BA: три seed, %",
                "Среднее seed, %",
                "Primary BA, %",
                "Primary ошибок I/88",
                "Точных I/3",
            ],
            rows,
        ),
        table(
            [
                "Режим: свежие C",
                "Среднее BA трёх seed, %",
                "Primary: query-выбранный C, %",
                "Primary: среднее32C, %",
            ],
            fresh_rows,
        ),
        "У некоторых неосновных seed свежие C с общим режимом сходятся хуже унаследованных чисел; это ограничение общей численной настройки. Качество самого алгоритма продолжения C оценивается первой таблицей.",
        "![Выбранные индексаторы](figures/indexator_comparison.png)",
        "![Веса и Adam](figures/weight_comparison.png)",
        "![Фактические кривые поиска](figures/search_loss_recall.png)",
        "![Независимая validation до итогового test](figures/validation_loss.png)",
        "![Свежие кодбуки](figures/fresh_codebook_loss_recall.png)",
    ]
    content = "\n\n".join(lines) + "\n"
    (folder / "REPORT.md").write_text(content)
    main_report = output / "REPORT.md"
    existing = main_report.read_text().split(
        "\n\n## Абляция по запросу пользователя: порядок обновлений P и C"
    )[0]
    main_report.write_text(
        existing.rstrip()
        + "\n\n"
        + content.replace("](figures/", "](update_ablation/figures/")
    )
    snapshot = output / "analysis_snapshot"
    for name in ("update_ablation.py", "search.py", "README.md"):
        shutil.copy2(BASE / name, snapshot / name)
    save_json(
        folder / "complete.json",
        {
            "completed_utc": datetime.now(UTC).isoformat(),
            "primary_results": primary,
            "software_tests_run": False,
        },
    )
    complete = json.loads((output / "complete.json").read_text())
    complete["update_ablation_added_utc"] = datetime.now(UTC).isoformat()
    complete["artifacts"] = {
        str(path.relative_to(output)): digest(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "complete.json"
    }
    save_json(output / "complete.json", complete)
    print(
        table(
            [
                "Mode",
                "Config",
                "Selection",
                "Seeds",
                "Mean",
                "Primary",
                "Errors",
                "Exact",
            ],
            rows,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "run", "validate", "freeze", "evaluate", "report")
    )
    parser.add_argument("--mode", choices=tuple(MODES), default="alternate3")
    parser.add_argument(
        "--output", type=Path, default=BASE / "results/single_0110_20261104"
    )
    parser.add_argument("--device", default="cuda:6")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == "prepare":
        prepare(args.output)
    elif args.stage == "run":
        run_mode(args.output, args.mode, torch.device(args.device))
    elif args.stage == "freeze":
        freeze(args.output)
    elif args.stage == "validate":
        validation_audit(args.output)
    elif args.stage == "evaluate":
        evaluate(args.output, torch.device(args.device))
    else:
        report(args.output)


if __name__ == "__main__":
    main()
