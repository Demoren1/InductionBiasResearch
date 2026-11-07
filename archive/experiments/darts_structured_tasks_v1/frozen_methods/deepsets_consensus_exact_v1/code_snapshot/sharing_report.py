"""Evaluate frozen sharing ablations and render their actual indexators."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap

from .data import digest, save_json, teacher_predict
from .evaluate import align_indices
from .model import score


def evaluate(output):
    frozen = json.loads((output / "source_frozen.json").read_text())
    for name, expected in frozen["sha256"].items():
        if digest(output / name) != expected:
            raise RuntimeError(f"Frozen source state changed: {name}")
    plan = json.loads((output / "protocol.json").read_text())
    spec = json.loads((output / "specification.json").read_text())
    choices = json.loads((output / "family_selection.json").read_text())["families"]
    with np.load(output / "confirmation/hard_fits.npz") as archive:
        fitted = dict(archive)
    fresh_choices = json.loads((output / "fresh_selection.json").read_text())
    with np.load(output / "fresh_evaluation.npz") as archive:
        fresh = dict(archive)
    save_json(
        output / "final_frozen.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "source_frozen": frozen,
            "sha256": {
                name: digest(output / name)
                for name in (
                    "fresh_selection.json",
                    "fresh_evaluation.npz",
                    "family_selection.json",
                    "confirmation/hard_fits.npz",
                )
            },
        },
    )
    with np.load(output / "teacher_audit_only.npz") as archive:
        truth = dict(archive)
    heldout_seed = plan["seed"] + plan.get("sharing_heldout_offset", 7100000)
    rng = np.random.default_rng(heldout_seed)
    x = rng.normal(size=(8192, spec["inputs"])).astype(np.float32)
    y = teacher_predict(x, truth, spec)
    np.savez_compressed(output / "heldout.npz", test_x=x, test_y=y)
    save_json(
        output / "heldout_manifest.json",
        {
            "seed": heldout_seed,
            "created_utc": datetime.now(UTC).isoformat(),
            "heldout_sha256": digest(output / "heldout.npz"),
            "exploratory_task_previously_inspected": True,
        },
    )
    x_tensor, y_tensor = torch.tensor(x), torch.tensor(y)
    chosen_rows = [
        row["row"] for family in choices.values() for row in family["per_seed"]
    ]
    losses = (
        score(
            x_tensor,
            y_tensor,
            torch.tensor(fitted["weights"][chosen_rows]),
            torch.tensor(fitted["numeric"][chosen_rows]),
            spec,
        ).numpy()
        / spec["target_variance"]
    )
    by_row = dict(zip(chosen_rows, losses))
    fresh_losses = (
        score(
            x_tensor,
            y_tensor,
            torch.tensor(fresh["weights"]),
            torch.tensor(fresh["numeric"]),
            spec,
        ).numpy()
        / spec["target_variance"]
    )
    families, aligned_states = {}, {}
    for family, entry in choices.items():
        records = []
        for row in entry["per_seed"]:
            number = row["row"]
            restart = int(fitted["query_mse"][number].argmin())
            recovery, aligned, columns = align_indices(
                fitted["indices"][number], truth["indices"], spec["codes"]
            )
            records.append(
                {
                    "seed": row["seed"],
                    "row": number,
                    "search_step": row["search_step"],
                    "selection_nmse": row["selection_mse"] / spec["target_variance"],
                    "test_nmse": float(by_row[number][restart]),
                    "test_mean_over_four": float(by_row[number].mean()),
                    "query_selected_restart": restart,
                    "recovery": recovery,
                }
            )
            aligned_states[f"row_{number}_indices"] = aligned
            aligned_states[f"row_{number}_columns"] = columns
        primary = min(records, key=lambda row: row["selection_nmse"])
        fresh_row = next(
            i for i, row in enumerate(fresh_choices["rows"]) if row["family"] == family
        )
        fresh_restart = fresh_choices["query_selected_restarts"][fresh_row]
        configuration = entry["per_seed"][0]
        families[family] = {
            "configuration": entry["configuration"],
            "commitment": configuration["commitment"],
            "settings": configuration,
            "selection_nmse_mean": entry["selection_mse"] / spec["target_variance"],
            "per_seed": records,
            "exact_indexators": sum(
                row["recovery"]["categorical_errors"] == 0 for row in records
            ),
            "exact_masks": sum(row["recovery"]["mask_f1"] == 1 for row in records),
            "primary": primary,
            "fresh": {
                "source_row": fresh_choices["rows"][fresh_row]["source_row"],
                "settings": fresh_choices["rows"][fresh_row],
                "query_selected_test_nmse": float(
                    fresh_losses[fresh_row, fresh_restart]
                ),
                "mean_test_nmse": float(fresh_losses[fresh_row].mean()),
                "all_test_nmse": fresh_losses[fresh_row].tolist(),
                "successes_below_0_01": int((fresh_losses[fresh_row] < 0.01).sum()),
                "restarts": int(fresh_losses.shape[1]),
            },
        }
    np.savez_compressed(
        output / "aligned_states.npz",
        true_indices=truth["indices"],
        row_order=truth["permutation"],
        **aligned_states,
    )
    summary = {
        "families": families,
        "teacher_used_only_after_source_freeze": True,
        "exploratory": True,
        "observations": "New test inputs for one previously studied teacher; new search seeds are not independent functions.",
    }
    save_json(output / "summary.json", summary)
    print(
        json.dumps(
            {
                name: {
                    "lambda": row["commitment"],
                    "selection": row["selection_nmse_mean"],
                    "exact": row["exact_indexators"],
                    "masks": row["exact_masks"],
                    "primary_errors": row["primary"]["recovery"]["categorical_errors"],
                    "primary_nmse": row["primary"]["test_nmse"],
                }
                for name, row in families.items()
            },
            indent=2,
        ),
        flush=True,
    )


def report(output):
    summary = json.loads((output / "summary.json").read_text())
    families = summary["families"]
    plan = json.loads((output / "protocol.json").read_text())
    specification = json.loads((output / "specification.json").read_text())
    labels = {
        "baseline": "Исходный DARTS",
        "no_sparsity": "Без sparsity",
        "no_entropy": "Без entropy",
        "gentle_cooling": "Мягкое охлаждение",
        "commit_constant": "Постоянный commitment",
        "commit_early": "Ранний commitment",
        "commit_late": "Поздний commitment",
        "commit_early_no_sparsity": "Commitment без sparsity",
        "commit_early_no_entropy": "Commitment без entropy",
        "commit_early_slow_alpha": "Commitment, медленный P",
    }
    for method, title in (
        ("logits", "Свободные логиты"),
        ("distance_soft", "Расстояния, мягкий проход"),
        ("nearest_ste", "Ближайший код, STE"),
    ):
        for warmup in (0, 1000, 3000):
            labels[f"{method}_w{warmup}"] = f"{title}; warmup {warmup}"
    labels["logits_commit_w0"] = "Логиты + commitment; warmup 0"
    for family in families:
        if family.startswith("full_ste_l1_"):
            labels[family] = "Полная матрица, STE; L1=" + family.removeprefix(
                "full_ste_l1_"
            )
    with np.load(output / "aligned_states.npz") as archive:
        states = dict(archive)
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 160})
    colors = plt.get_cmap("turbo")(np.linspace(0.03, 0.97, 9))
    colors[4] = [0.96, 0.96, 0.96, 1]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-4.5, 5), len(colors))
    panels = [("Истинный I", states["true_indices"])]
    for family, result in families.items():
        primary = result["primary"]
        panels.append(
            (
                (
                    f"{labels[family]}\nλ={result['commitment']:g}; "
                    f"ошибок {primary['recovery']['categorical_errors']}/256"
                ),
                states[f"row_{primary['row']}_indices"],
            )
        )
    columns = 3 if len(panels) <= 6 else 4
    row_count = (len(panels) + columns - 1) // columns
    figure, axes = plt.subplots(
        row_count,
        columns,
        figsize=(4 * columns, 4 * row_count),
        constrained_layout=True,
    )
    for axis, (title, matrix) in zip(axes.flat, panels):
        image = axis.imshow(
            matrix[states["row_order"]],
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        axis.set_title(title)
        axis.set_xlabel("Скрытый нейрон")
        axis.set_ylabel("Координата")
    for axis in list(axes.flat)[len(panels) :]:
        axis.set_visible(False)
    figure.colorbar(image, ax=axes, ticks=range(-4, 5), shrink=0.6)
    figure.suptitle("DeepSets: source-selected индексаторы абляций", fontsize=16)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    for extension in ("png", "svg"):
        figure.savefig(figures / f"indexators.{extension}", bbox_inches="tight")
    plt.close(figure)

    history = json.loads((output / "confirmation/history.json").read_text())
    chosen = sorted(families, key=lambda key: families[key]["selection_nmse_mean"])
    baseline = next(
        name for name in ("baseline", "logits_w0", "full_ste_l1_0") if name in families
    )
    chosen = [baseline] + [name for name in chosen if name != baseline][:3]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    variance = json.loads((output / "specification.json").read_text())[
        "target_variance"
    ]
    for name in chosen:
        root = families[name]["settings"]["root"]
        steps = [record["step"] for record in history]
        for axis, key in zip(axes, ("query_mse", "hard_query_mse")):
            values = [np.mean(record[key][root]) / variance for record in history]
            axis.plot(steps, values, label=labels[name])
            axis.set_yscale("log")
            axis.set_xlabel("Шаг поиска P")
            axis.set_ylabel("Query NMSE")
            axis.grid(alpha=0.2)
    axes[0].set_title("Модель при обучении")
    axes[1].set_title("Жёсткая модель до дообучения C")
    axes[1].legend(fontsize=8)
    figure.suptitle(f"Парные кривые seed {families[baseline]['settings']['seed']}")
    for extension in ("png", "svg"):
        figure.savefig(figures / f"soft_hard_curves.{extension}", bbox_inches="tight")
    plt.close(figure)
    conditional = (output / "learned_mask.npz").exists()
    description = (
        "Условный поиск шэринга на ранее найденной самим DARTS маске. "
        "Общий бюджет12000 включает masked warmup. Одна численная реплика "
        "во время поиска, затем один исходный и три свежих кодбука для C-fit. "
        if conditional
        else "Поиск маски и шэринга с нуля, четыре численные реплики. "
    )
    if specification.get("full_connectivity_search"):
        description = (
            "Полный поиск подключений и шэринга с нуля без готовой маски. "
            "Ближайший дискретный код и identity STE, одна численная реплика "
            "во время поиска. При C-fit один исходный и три свежих кодбука. "
        )
    description += (
        "Истинные группы и роли координат не передаются поиску. "
        "Словарь [0,C1,…,C4,−C1,…,−C4]. "
        f"Pilot {plan['search']['pilot_seeds']}; "
        f"confirmation {plan['search']['confirmation_seeds']}. "
        "Настройки выбираются по среднему hard-refit selection NMSE, "
        "teacher открывается после freeze. C-fit3000 шагов для каждого снимка."
    )
    lines = [
        "# DeepSets: абляции DARTS и commitment",
        "",
        description,
        "",
        "| Вариант | λ | Selection NMSE, среднее 3 seed | Точных I/3 | Точных масок/3 | Test NMSE выбранного seed | Ошибок I/256 | Свежих C с NMSE<0.01/32 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for family, result in families.items():
        primary = result["primary"]
        lines.append(
            f"| {labels[family]} | {result['commitment']:g} | "
            f"{result['selection_nmse_mean']:.6g} | "
            f"{result['exact_indexators']}/3 | {result['exact_masks']}/3 | "
            f"{primary['test_nmse']:.6g} | "
            f"{primary['recovery']['categorical_errors']} | "
            f"{result['fresh']['successes_below_0_01']}/32 |"
        )
    lines += [
        "",
        "![Индексаторы](figures/indexators.png)",
        "",
        "![Мягкая и жёсткая модели](figures/soft_hard_curves.png)",
        "",
        "Свежие C — новые численные инициализации на той же функции, не перенос на новые задачи. Все per-seed результаты и все32 численных старта сохранены в summary.json. Изменения относительно уже просмотренных результатов — исследовательские абляции, не новый независимый benchmark. Программные тесты не запускались.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if not args.report_only:
        evaluate(args.output)
    report(args.output)


if __name__ == "__main__":
    main()
