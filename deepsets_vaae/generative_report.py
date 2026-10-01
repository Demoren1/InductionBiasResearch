"""Strict report writer for the post-VAE generative-model comparison.

The runner creates the experiment artefacts; this module only validates and
summarises them.  It deliberately refuses partial experiments instead of
guessing missing target metrics or samples.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/deepsets_vaae/20261001_other_generators"
SEEDS = tuple(range(4100, 4108))
BUDGETS = (32, 64, 128, 256)
OLD_METHODS = (
    "functional_mean_small", "functional_mean_large", "functional_vae_small",
    "functional_vae_large", "raw_vae_large", "random", "dense",
)
GENERATIVE_METHODS = ("diffusion", "flow_matching", "gnn_flow", "set_transformer_flow", "gan")
METHODS = OLD_METHODS + GENERATIVE_METHODS
TASKS = tuple(range(8))
REPLICAS = tuple(range(4))
K = 7526
F, H = 784, 32
T975_DF7 = 2.364624251  # two-sided 95% Student t multiplier, df=7


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _save(fig: plt.Figure, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=170, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def _ci(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (len(SEEDS),) or not np.isfinite(array).all():
        raise ValueError("paired seed values must contain eight finite values")
    mean = float(array.mean())
    sem = float(array.std(ddof=1) / math.sqrt(len(array)))
    return {"n": len(array), "df": len(array) - 1, "mean": mean,
            "sem": sem, "ci95_low": mean - T975_DF7 * sem,
            "ci95_high": mean + T975_DF7 * sem}


def _validate_protocol(out: Path) -> dict[str, Any]:
    protocol = _json(out / "protocol.json")
    expected = {
        "seeds": list(SEEDS), "methods": list(METHODS), "budgets": list(BUDGETS),
        "source_tasks": 4, "source_keep": 256, "source_density": .2,
        "target_density": .3, "target_edges": K,
    }
    bad = {key: (protocol.get(key), value) for key, value in expected.items()
           if protocol.get(key) != value}
    if bad:
        raise ValueError(f"Protocol does not match the declared comparison: {bad}")
    return protocol


def _validate_target_records(rows: Any, seed: int, population: str) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"seed {seed}/{population}: records must be a list")
    expected = {(task, budget, method, replica) for task in TASKS for budget in BUDGETS
                for method in METHODS for replica in REPLICAS}
    seen: set[tuple[int, int, str, int]] = set()
    checked: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"seed {seed}/{population}: non-object target row")
        key = (int(row.get("task", -1)), int(row.get("support_size", -1)),
               str(row.get("method", "")), int(row.get("init", -1)))
        if key in seen or key not in expected:
            raise ValueError(f"seed {seed}/{population}: duplicate/unexpected target row {key}")
        _finite(row.get("mse"), f"seed {seed}/{population}/{key}: mse")
        _finite(row.get("validation_mse"), f"seed {seed}/{population}/{key}: validation_mse")
        seen.add(key)
        checked.append(row)
    if seen != expected:
        raise ValueError(f"seed {seed}/{population}: got {len(seen)} rows, expected {len(expected)}")
    return checked


def _curve_value(row: dict[str, Any], *names: str) -> float | None:
    for name in names:
        if name in row and row[name] is not None:
            return _finite(row[name], f"loss curve/{name}")
    return None


def _load_fit(out: Path, seed: int, method: str) -> dict[str, Any]:
    folder = out / f"seed_{seed}" / method
    fit = _json(folder / "fit.json")
    for key, value in (("method", method), ("experiment_seed", seed)):
        if fit.get(key) != value:
            raise ValueError(f"{folder}/fit.json: expected {key}={value!r}, got {fit.get(key)!r}")
    if not isinstance(fit.get("training_seed"), int):
        raise ValueError(f"{folder}/fit.json must record integer training_seed")
    if not isinstance(fit.get("converged"), bool):
        raise ValueError(f"{folder}/fit.json must record boolean converged")
    if "status" in fit and not isinstance(fit["status"], str):
        raise ValueError(f"{folder}/fit.json status must be a string")
    for key in ("best_step", "stop_step", "parameter_count"):
        if int(fit.get(key, -1)) < 0:
            raise ValueError(f"{folder}/fit.json: invalid {key}")
    normalization = fit.get("normalization")
    metrics = fit.get("sample_metrics")
    curve = fit.get("loss_curve")
    if not isinstance(normalization, dict) or not isinstance(metrics, dict) or not isinstance(curve, list) or not curve:
        raise ValueError(f"{folder}/fit.json lacks normalization, sample_metrics, or loss_curve")
    for key in ("mean", "std"):
        _finite(normalization.get(key), f"{folder}/normalization/{key}")
    if float(normalization["std"]) <= 0:
        raise ValueError(f"{folder}/normalization/std must be positive")
    for key in ("variance_ratio", "pairwise_iou", "train_nearest_mse", "validation_nearest_mse",
                "mean_map_mse", "mean_map_vs_heldout_mse"):
        _finite(metrics.get(key), f"{folder}/sample_metrics/{key}")
    previous = -1
    for row in curve:
        if not isinstance(row, dict) or int(row.get("step", -1)) <= previous:
            raise ValueError(f"{folder}/fit.json loss_curve must have increasing positive steps")
        previous = int(row["step"])
        if _curve_value(row, "stochastic_loss", "train_loss", "validation_loss", "generator_loss",
                        "probe_train_loss", "probe_validation_loss") is None:
            raise ValueError(f"{folder}/fit.json curve row has no recognized loss")
    samples_path = folder / "samples.pt"
    if not samples_path.is_file():
        raise FileNotFoundError(f"Missing required samples: {samples_path}")
    payload = torch.load(samples_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"{samples_path}: expected dict")
    for key, shape in (("samples", (4, 32, F, H)), ("score", (F, H)), ("masks", (4, F, H))):
        value = torch.as_tensor(payload.get(key))
        if tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{samples_path}: {key} must be finite with shape {shape}")
        if key == "masks":
            if not bool(torch.all((value == 0) | (value == 1))) or not bool(torch.all(value.sum((1, 2)) == K)):
                raise ValueError(f"{samples_path}: masks must be binary exact-{K}")
    return fit


def _plot_loss_curves(out: Path, fits: dict[tuple[int, str], dict[str, Any]]) -> None:
    target = out / "loss_curves"
    for seed in SEEDS:
        for method in GENERATIVE_METHODS:
            fit = fits[(seed, method)]
            curve = fit["loss_curve"]
            x = np.asarray([int(row["step"]) for row in curve])
            if method == "gan":
                _plot_gan_loss_curve(target, seed, fit)
                continue
            fig, ax = plt.subplots(figsize=(7.2, 4.1))
            drawn = 0
            for keys, label, color in (
                (("stochastic_loss",), "стохастическая train", "#4c78a8"),
                (("train_loss", "deterministic_train_loss"), "train", "#59a14f"),
                (("validation_loss", "heldout_loss"), "held-out", "#e15759"),
                (("critic_loss",), "critic (GAN)", "#9467bd"),
            ):
                y = [_curve_value(row, *keys) for row in curve]
                if any(value is not None for value in y):
                    ax.plot(x, [np.nan if value is None else value for value in y], label=label, color=color, linewidth=1.4)
                    drawn += 1
            if not drawn:
                raise ValueError(f"No plottable losses for {seed}/{method}")
            for step, label, color in ((int(fit["best_step"]), "лучший source validation", "#222222"),
                                       (int(fit["stop_step"]), "остановка", "#777777")):
                ax.axvline(step, color=color, linestyle="--", linewidth=.9, label=label)
            ax.set_xlabel("update")
            ax.set_ylabel("source objective (меньше лучше)")
            ax.set_title(f"{method}, seed {seed}: source-only обучение")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
            _save(fig, target, f"seed_{seed}_{method}")


def _plot_gan_loss_curve(target: Path, seed: int, fit: dict[str, Any]) -> None:
    """GAN game losses have different units from the fixed source-map probe."""
    curve = fit["loss_curve"]
    x = np.asarray([int(row["step"]) for row in curve])
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.1))
    left = (("probe_train_loss", "projected_w1_train", "train_loss"), "source train: projected W1", "#59a14f"), \
           (("probe_validation_loss", "projected_w1_validation", "validation_loss", "heldout_loss"), "source held-out: projected W1", "#e15759")
    right = (("generator_loss", "gen_loss", "stochastic_loss"), "generator objective", "#4c78a8"), \
            (("critic_loss", "discriminator_loss"), "critic objective", "#9467bd")
    for ax, groups, title, ylabel in (
        (axes[0], left, "GAN: fixed source-map probe", "projected W1 (меньше лучше)"),
        (axes[1], right, "GAN: optimization game", "objective (units differ)"),
    ):
        drawn = 0
        for keys, label, color in groups:
            y = [_curve_value(row, *keys) for row in curve]
            if any(value is not None for value in y):
                ax.plot(x, [np.nan if value is None else value for value in y], label=label, color=color, linewidth=1.4)
                drawn += 1
        if drawn != len(groups):
            raise ValueError(f"GAN {seed}: missing required {title} series")
        for step, label, color in ((int(fit["best_step"]), "best", "#222222"),
                                   (int(fit["stop_step"]), "stop", "#777777")):
            ax.axvline(step, color=color, linestyle="--", linewidth=.9, label=label)
        ax.set_title(title); ax.set_xlabel("update"); ax.set_ylabel(ylabel); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle(f"gan, seed {seed}: probe и game losses показаны раздельно")
    _save(fig, target, f"seed_{seed}_gan")


def _target_summary(records: dict[str, dict[int, list[dict[str, Any]]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for population, by_seed in records.items():
        pop: dict[str, Any] = {}
        for budget in BUDGETS:
            values: dict[str, list[float]] = {method: [] for method in METHODS}
            for seed in SEEDS:
                rows = by_seed[seed]
                for method in METHODS:
                    cell = [float(row["mse"]) for row in rows if int(row["support_size"]) == budget
                            and str(row["method"]) == method]
                    if len(cell) != len(TASKS) * len(REPLICAS):
                        raise ValueError(f"Incomplete cell {population}/{seed}/{budget}/{method}")
                    values[method].append(float(np.mean(cell)))
            entry: dict[str, Any] = {method: {**_ci(values[method]), "seed_values": values[method]}
                                     for method in METHODS}
            contrasts: dict[str, Any] = {}
            for method in GENERATIVE_METHODS:
                for baseline in ("functional_mean_large", "random", "dense", "functional_vae_large"):
                    delta = (np.asarray(values[method]) - np.asarray(values[baseline])).tolist()
                    contrasts[f"{method}_minus_{baseline}"] = {**_ci(delta), "seed_values": delta}
            entry["contrasts"] = contrasts
            pop[str(budget)] = entry
        result[population] = pop
    return result


def _plot_target(out: Path, summary: dict[str, Any]) -> None:
    colors = {"functional_mean_large": "#2f7f5f", "functional_vae_large": "#d55e00",
              "random": "#777777", "dense": "#222222", "diffusion": "#4c78a8",
              "flow_matching": "#59a14f", "gnn_flow": "#e15759", "set_transformer_flow": "#9467bd", "gan": "#f28e2b"}
    display = ("functional_mean_large", "functional_vae_large", "random", "dense") + GENERATIVE_METHODS
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, population in zip(axes, ("old", "fresh")):
        for method in display:
            means = [summary[population][str(b)][method]["mean"] for b in BUDGETS]
            lo = [summary[population][str(b)][method]["ci95_low"] for b in BUDGETS]
            hi = [summary[population][str(b)][method]["ci95_high"] for b in BUDGETS]
            ax.plot(BUDGETS, means, marker="o", linewidth=1.35, label=method, color=colors[method])
            ax.fill_between(BUDGETS, lo, hi, color=colors[method], alpha=.12)
        ax.set_xscale("log", base=2)
        ax.set_xticks(BUDGETS, [str(x) for x in BUDGETS])
        ax.grid(alpha=.2)
        ax.set_title(f"{population}: среднее по 8 seed, 95% t-CI")
        ax.set_xlabel("число размеченных target-наборов")
    axes[0].set_ylabel("NMSE = MSE / 5 (меньше лучше)")
    axes[1].legend(fontsize=7, ncol=2)
    _save(fig, out, "target_learning_curves")

    baselines = ("functional_mean_large", "random", "dense", "functional_vae_large")
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8), sharex=True)
    x = np.arange(len(GENERATIVE_METHODS))
    width = .19
    for ax, baseline in zip(axes.flat, baselines):
        for j, budget in enumerate(BUDGETS):
            data = [summary["fresh"][str(budget)]["contrasts"][f"{method}_minus_{baseline}"]
                    for method in GENERATIVE_METHODS]
            means = [d["mean"] for d in data]
            errors = [[d["mean"] - d["ci95_low"] for d in data],
                      [d["ci95_high"] - d["mean"] for d in data]]
            ax.bar(x + (j - 1.5) * width, means, width, yerr=errors, capsize=2, label=str(budget))
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_title(f"fresh: генератор − {baseline}")
        ax.set_ylabel("парная Δ NMSE")
        ax.grid(axis="y", alpha=.2)
        ax.legend(title="budget", fontsize=7)
    axes[1, 0].set_xticks(x, GENERATIVE_METHODS, rotation=18, ha="right")
    axes[1, 1].set_xticks(x, GENERATIVE_METHODS, rotation=18, ha="right")
    _save(fig, out, "target_contrasts_fresh")


def _plot_sample_metrics(out: Path, fits: dict[tuple[int, str], dict[str, Any]]) -> dict[str, Any]:
    names = ("variance_ratio", "pairwise_iou", "train_nearest_mse", "validation_nearest_mse",
             "mean_map_mse", "mean_map_vs_heldout_mse")
    summary: dict[str, Any] = {}
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    x = np.arange(len(GENERATIVE_METHODS))
    for ax, metric in zip(axes.flat, names):
        values = [[_finite(fits[(seed, method)]["sample_metrics"][metric], metric)
                   for seed in SEEDS] for method in GENERATIVE_METHODS]
        ax.boxplot(values, positions=x, widths=.58, showfliers=False)
        for j, cell in enumerate(values):
            ax.scatter(np.full(len(cell), j), cell, s=12, color="#333333", alpha=.65, zorder=3)
        ax.set_xticks(x, GENERATIVE_METHODS, rotation=18, ha="right", fontsize=8)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=.2)
        summary[metric] = {method: {**_ci(values[i]), "seed_values": values[i]}
                           for i, method in enumerate(GENERATIVE_METHODS)}
    _save(fig, out, "sample_diversity_metrics")
    return summary


def _checkpoint_method_index(checkpoint: dict[str, Any], method: str, init: int) -> int:
    methods = checkpoint.get("methods")
    if not isinstance(methods, list):
        raise ValueError("target checkpoint lacks methods")
    found = [int(item["model_index"]) for item in methods
             if str(item.get("method")) == method and int(item.get("init", -1)) == init]
    if len(found) != 1:
        raise ValueError(f"target checkpoint has no unique {method}/init{init}")
    return found[0]


def _plot_heatmaps(out: Path) -> None:
    seed, task, budget, init = 4100, 0, 256, 0
    checkpoint_path = out / f"seed_{seed}" / "fresh_weights" / f"target_task{task}_budget{budget}.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing heatmap checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    weight = torch.as_tensor(checkpoint.get("weight"))
    masks = torch.as_tensor(checkpoint.get("masks"))
    if weight.ndim != 3 or tuple(weight.shape[1:]) != (F, H) or masks.shape != weight.shape:
        raise ValueError("target checkpoint has unexpected weight/mask shape")
    compare = ("dense", "functional_mean_large", "functional_vae_large") + GENERATIVE_METHODS
    effective = []
    binary = []
    score_maps = []
    for method in compare:
        index = _checkpoint_method_index(checkpoint, method, init)
        effective.append((weight[index] * masks[index]).numpy())
        binary.append(masks[index].numpy())
        if method in GENERATIVE_METHODS:
            sample_path = out / f"seed_{seed}" / method / "samples.pt"
            payload = torch.load(sample_path, map_location="cpu", weights_only=True)
            score_maps.append(torch.as_tensor(payload["score"]).numpy())
        else:
            score_maps.append(np.full((F, H), np.nan, dtype=np.float32))
    stack = np.stack(effective)
    limit = float(np.abs(stack).max())
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("effective trained weights have no nonzero finite scale")
    folder = out / "weight_heatmaps"
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2), constrained_layout=True)
    for ax, method, value in zip(axes.flat, compare, effective):
        image = ax.imshow(value.T, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit, interpolation="nearest")
        ax.set_title(method, fontsize=9); ax.set_xlabel("pixel"); ax.set_ylabel("hidden")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.72, label="обученный W × M")
    _save(fig, folder, "fresh_task0_budget256_effective_weights")
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2), constrained_layout=True)
    for ax, method, value in zip(axes.flat, compare, binary):
        image = ax.imshow(value.T, aspect="auto", cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(method, fontsize=9); ax.set_xlabel("pixel"); ax.set_ylabel("hidden")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.72, label="бинарная маска")
    _save(fig, folder, "fresh_task0_budget256_binary_masks")
    fig, axes = plt.subplots(1, len(GENERATIVE_METHODS), figsize=(16, 3.7), constrained_layout=True)
    for ax, method, value in zip(axes, GENERATIVE_METHODS, score_maps[3:]):
        image = ax.imshow(value.T, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_title(f"{method}: score")
        ax.set_xlabel("pixel"); ax.set_ylabel("hidden")
        fig.colorbar(image, ax=ax, shrink=.8)
    _save(fig, folder, "seed4100_generated_scores")
    np.savez_compressed(folder / "fresh_task0_budget256_heatmap_arrays.npz",
                        methods=np.asarray(compare), effective_weights=np.stack(effective),
                        binary_masks=np.stack(binary), generated_scores=np.stack(score_maps))
    (folder / "metadata.json").write_text(json.dumps({
        "seed": seed, "population": "fresh", "target_task": task, "budget": budget, "init": init,
        "methods": list(compare), "weight_scale_symmetric": limit,
        "caption": "W×M uses selected target checkpoint; masks and scores are displayed separately."
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _source_references(out: Path) -> list[str]:
    lines: list[str] = []
    for name in ("parentsources.json", "article_notes.json"):
        path = out / name
        if path.is_file():
            value = _json(path)
            lines.append(f"- [{name}]({name}) — сохранённые источники/заметки, использованные для выбора архитектур: `{json.dumps(value, ensure_ascii=False)}`")
    if not lines:
        protocol = _json(out / "protocol.json")
        articles = protocol.get("articles")
        if articles is not None:
            lines.append("- `protocol.json` → `articles`: " + json.dumps(articles, ensure_ascii=False))
    if not lines:
        lines.append("- Файлы `parentsources.json` и `article_notes.json` не найдены; отчёт не подменяет их ссылками.")
    return lines


def _write_report(out: Path, protocol: dict[str, Any], summary: dict[str, Any],
                  sample_summary: dict[str, Any], stability: dict[str, Any]) -> None:
    fresh = summary["fresh"]["256"]
    rows = []
    for method in GENERATIVE_METHODS:
        cell = fresh[method]
        baseline = fresh["contrasts"][f"{method}_minus_functional_mean_large"]
        rows.append(f"| {method} | {cell['mean']:.4f} [{cell['ci95_low']:.4f}; {cell['ci95_high']:.4f}] | {baseline['mean']:+.4f} [{baseline['ci95_low']:+.4f}; {baseline['ci95_high']:+.4f}] |")
    stability_rows = [f"| {method} | {stability['by_method'][method]['stable']} / {stability['by_method'][method]['attempts']} | "
                      f"{', '.join(f'{key}: {value}' for key, value in stability['by_method'][method]['statuses'].items()) or 'status не записан'} |"
                      for method in GENERATIVE_METHODS]
    text = [
        "# Другие генеративные модели для функциональных карт DeepSets",
        "",
        "Генераторы обучены на том же неизменённом банке: четыре source-задачи, по 205 aligned functional train-карт и 51 held-out source-карте на seed. Модель получает pooled source-карты с условием source-task; target cost-векторы, target validation и target test не используются для выбора checkpoint, маски или sparsity. Все пять моделей здесь — самостоятельные генераторы карт, не VAE и не PCA.",
        "",
        "Все target-сравнения ниже exploratory: обе группы из восьми cost-задач уже были просмотрены в VAE-контроле. Интервалы — парные 95% Student t (df=7) по seed; множественные сравнения не корректировались.",
        "",
        "## Схема и фиксированная sparsity",
        "",
        f"Размер карты — {F}×{H}={F*H}; каждая новая маска проходит общий exact-top-{K} projection (30%). Adaptive sparsity намеренно отложена: её нельзя подбирать по target validation/test, иначе сравнение протекает. Следующий отдельный протокол может проверить source-only joint utility с penalty на число рёбер (L0/hard-concrete) и nested validation; это ещё не тестировалось. Прямая `functional_mean_large` остаётся сильным baseline.",
        "",
        "Методы: diffusion и flow matching моделируют распределение карт; `gnn_flow` использует двудольный граф pixel–hidden; `set_transformer_flow` рассматривает hidden-columns как множество; GAN использует генератор/критик. Во всех вариантах task-condition и нормализация учатся только на pooled 4×205 source-train картах; held-out 4×51 используется для source-only checkpoint selection.",
        "",
        "## Статус source-only попыток",
        "",
        "| Метод | stable / все попытки | сохранённые статусы |",
        "|---|---:|---|",
        *stability_rows,
        "",
        "Лимит GPU updates сам по себе не означает сходимость. В частности, GAN может быть сохранён как exploratory unstable game; его результаты не заменяются и не дорисовываются, а статус остаётся в сводке. Для GAN слева на кривой показан source fixed-probe, справа — generator/critic objectives в разных единицах; направление «меньше лучше» относится к probe, но не к game losses.",
        "",
        "## Target NMSE, fresh, budget 256",
        "",
        "| Метод | NMSE, среднее [95% CI] | Δ к functional mean large [95% CI] |",
        "|---|---:|---:|",
        *rows,
        "",
        "NMSE = MSE/5, меньше лучше. Значение Δ ниже нуля означает меньшую ошибку, но не доказывает обобщение вне этих восьми фиксированных задач.",
        "",
        "## Графики",
        "",
        "- [Кривые target-качества](target_learning_curves.png): линии — средние по seed, полупрозрачные области — 95% t-CI.",
        "- [Парные контрасты на fresh задачах](target_contrasts_fresh.png): каждый столбец — generator minus baseline; ниже нуля лучше генератора.",
        "- [Разнообразие source samples](sample_diversity_metrics.png): variance ratio, парная IoU и расстояния до train/held-out карт.",
        "- [Все 40 source loss curves](loss_curves/): train/held-out objectives, выбранный checkpoint и остановка. GAN-графики раздельно показывают fixed probe и generator/critic game losses.",
        "- [W×M heatmap](weight_heatmaps/fresh_task0_budget256_effective_weights.png), [бинарные маски](weight_heatmaps/fresh_task0_budget256_binary_masks.png) и [scores генераторов](weight_heatmaps/seed4100_generated_scores.png): фиксированный seed 4100, fresh task 0, budget 256, init 0; шкала W×M общая и знаковая.",
        "",
        "## Воспроизводимость",
        "",
        "- [protocol.json](protocol.json) фиксирует банк, методы и target protocol; старый банк не изменялся.",
        "- [summary.json](summary.json) содержит все seed means и paired contrasts; [figure_data.npz](figure_data.npz) — численные массивы графиков.",
        "- Каждый `seed_<n>/<method>/fit.json` содержит source-only loss curve и sample diagnostics; `samples.pt` — сэмплы, score и exact-K masks. Target checkpoints находятся в `weights/` и `fresh_weights/`.",
        "",
        "## Источники и адаптации",
        "",
        *_source_references(out),
        "",
        f"Метаданные отчёта: seeds={list(SEEDS)}, budgets={list(BUDGETS)}, methods={list(METHODS)}.",
    ]
    (out / "REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def _stability_summary(fits: dict[tuple[int, str], dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, Any] = {}
    for method in GENERATIVE_METHODS:
        cells = [fits[(seed, method)] for seed in SEEDS]
        statuses: dict[str, int] = {}
        for fit in cells:
            status = str(fit.get("status", "not_recorded"))
            statuses[status] = statuses.get(status, 0) + 1
        by_method[method] = {"attempts": len(cells), "stable": sum(bool(cell["converged"]) for cell in cells),
                             "unstable": sum(not bool(cell["converged"]) for cell in cells), "statuses": statuses}
    return {"attempts": len(SEEDS) * len(GENERATIVE_METHODS),
            "stable": sum(value["stable"] for value in by_method.values()),
            "unstable": sum(value["unstable"] for value in by_method.values()), "by_method": by_method}


def build(out: Path) -> dict[str, Any]:
    out = Path(out)
    protocol = _validate_protocol(out)
    records: dict[str, dict[int, list[dict[str, Any]]]] = {"old": {}, "fresh": {}}
    fits: dict[tuple[int, str], dict[str, Any]] = {}
    metadata: dict[str, Any] = {"fits": {}}
    for seed in SEEDS:
        result = _json(out / f"seed_{seed}" / "results.json")
        if int(result.get("seed", -1)) != seed:
            raise ValueError(f"seed_{seed}/results.json has wrong seed")
        records["old"][seed] = _validate_target_records(result.get("records"), seed, "old")
        records["fresh"][seed] = _validate_target_records(result.get("fresh_records"), seed, "fresh")
        for method in GENERATIVE_METHODS:
            fit = _load_fit(out, seed, method)
            fits[(seed, method)] = fit
            metadata["fits"][f"seed_{seed}/{method}"] = {
                key: fit[key] for key in ("best_step", "stop_step", "parameter_count", "normalization", "sample_metrics", "sampling")
                if key in fit
            }
    stability = _stability_summary(fits)
    _plot_loss_curves(out, fits)
    target = _target_summary(records)
    _plot_target(out, target)
    sample = _plot_sample_metrics(out, fits)
    _plot_heatmaps(out)
    result = {"status": "PASS", "protocol": protocol, "target": target,
              "sample_metrics": sample, "fit_metadata": metadata, "stability": stability}
    (out / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    arrays: dict[str, np.ndarray] = {}
    for population in ("old", "fresh"):
        for budget in BUDGETS:
            for method in METHODS:
                arrays[f"target_{population}_budget{budget}_{method}_seedvalues"] = np.asarray(
                    target[population][str(budget)][method]["seed_values"], dtype=np.float64)
    for metric, values in sample.items():
        for method, cell in values.items():
            arrays[f"sample_{metric}_{method}_seedvalues"] = np.asarray(cell["seed_values"], dtype=np.float64)
    np.savez_compressed(out / "figure_data.npz", **arrays)
    (out / "figure_data.json").write_text(json.dumps({"target": target, "sample_metrics": sample}, indent=2,
                                                       ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    _write_report(out, protocol, target, sample, stability)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = build(args.out)
    print(json.dumps({"status": result["status"], "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
