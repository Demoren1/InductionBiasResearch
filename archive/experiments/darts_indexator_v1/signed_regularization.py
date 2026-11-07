"""Ablate sign-pair exclusivity and normalized value variance on Pattern 0110."""

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

from . import single
from .data import BASE, digest, load_archive, save_json
from .model import (
    assignment_values,
    assignment_variance,
    canonical_signed_probabilities,
    metrics,
    soft_predict,
)
from .report import save_figure, table
from .signed_codebook import (
    CODE_LABELS,
    diagnose,
    draw_index,
    evaluate,
    freeze,
    search_phase,
)

VARIANTS = {"control": 5, "pair_product": 5, "value_variance": 5}


def projection_ablation(parent, source_root, destination, variants, device):
    destination.mkdir(exist_ok=False)
    save_json(
        destination / "registration.json",
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "source": str(source_root),
            "variants": list(variants),
            "rules": [
                "Keep the original source-selected configuration, search seeds and snapshots fixed. Change only hardening: move min(P+C,P-C) from each twin to zero, then argmax.",
                "This algebraically preserves the soft weights for every codebook and prevents both twins having positive mass. It does not prevent cancellation across different base codes.",
                "Continue the same four inherited pre-argmax codebooks for4096 steps, same rates and batch stream as the raw hardening. Query chooses numerical checkpoints. The primary search seed is unchanged.",
                "Freeze all projected I and source fits before new heldout evaluation. Existing task test is known; this is a posthoc explanatory ablation, not an independent benchmark.",
            ],
        },
    )
    prepared = {}
    for variant in variants:
        folder = source_root / variant
        plan = json.loads((folder / "protocol.json").read_text())
        state = json.loads((folder / "frozen.json").read_text())
        choices = state["selected_group"]["seeds"]
        indices, initials, raw_p, projected_p = [], [], [], []
        for choice in choices:
            origin = choice["origin"]
            archive = load_archive(folder / origin["folder"] / "snapshots.npz")
            history = json.loads(
                (folder / origin["folder"] / "history.json").read_text()
            )
            temperature = next(row for row in history if row["step"] == origin["step"])[
                "temperature"
            ][origin["root"]]
            p = torch.softmax(
                torch.tensor(archive["alpha"][origin["snapshot"], origin["root"]])
                / temperature,
                -1,
            )
            canonical = canonical_signed_probabilities(p)
            raw_p.append(p.numpy())
            projected_p.append(canonical.numpy())
            indices.append(CODE_LABELS[canonical.argmax(-1).numpy()])
            initials.append(archive["numeric"][origin["snapshot"], origin["root"], 0])
        result = single.fit_fixed(
            load_archive(folder / "source.npz"),
            np.repeat(np.stack(indices), 4, 0),
            np.stack(initials).reshape(-1, 5),
            [choice["origin"]["numeric_lr"] for choice in choices for _ in range(4)],
            plan["fit"],
            plan["seed"] + 10000001,
            device,
        )
        np.savez_compressed(destination / (variant + "_fits.npz"), **result)
        np.savez_compressed(
            destination / (variant + "_projection.npz"),
            raw=np.stack(raw_p),
            projected=np.stack(projected_p),
            indices=np.stack(indices),
            initial=np.stack(initials),
        )
        prepared[variant] = {
            "choices": choices,
            "primary_seed": state["primary"]["origin"]["search_seed"],
        }
        print("PROJECTED SOURCE", variant, flush=True)
    save_json(
        destination / "source_frozen.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "source": prepared,
            "heldout_evaluated": False,
        },
    )
    heldout = load_archive(parent / "heldout.npz")
    _, all_x = input_table()
    test_x = torch.tensor(heldout["test_x"], device=device)
    test_y = torch.tensor(heldout["test_y"][0], device=device)[None]
    all_x = all_x.to(device)
    all_y = labels_for(all_x, "0110")[None]
    summary = {}
    with torch.no_grad():
        for variant, source in prepared.items():
            fits = load_archive(destination / (variant + "_fits.npz"))
            projection = load_archive(destination / (variant + "_projection.npz"))
            index = torch.tensor(fits["indices"], dtype=torch.long, device=device)
            numerical = torch.tensor(fits["parameters"], device=device)
            test = (
                metrics(single.predict(test_x, index, numerical), test_y).cpu().numpy()
            )
            population = (
                metrics(single.predict(all_x, index, numerical), all_y).cpu().numpy()
            )
            seeds = []
            for number, choice in enumerate(source["choices"]):
                ids = np.arange(number * 4, (number + 1) * 4)
                selected = ids[fits["query_metrics"][ids, 0].argmin()]
                canonical = torch.tensor(projection["projected"][number], device=device)
                numeric = torch.tensor(
                    projection["initial"][number][None, None], device=device
                )
                immediate = torch.nn.functional.one_hot(canonical.argmax(-1), 5).float()
                immediate_score = (
                    metrics(
                        soft_predict(test_x, immediate[None], numeric),
                        test_y[None, :, None],
                    )
                    .mean((0, 1, 2))
                    .cpu()
                    .tolist()
                )
                seeds.append(
                    {
                        "seed": choice["origin"]["search_seed"],
                        "test": test[selected].tolist(),
                        "population": population[selected].tolist(),
                        "immediate_test": immediate_score,
                        "structure": diagnose(projection["indices"][number], True),
                    }
                )
            primary = next(
                row for row in seeds if row["seed"] == source["primary_seed"]
            )
            raw = torch.tensor(projection["raw"])
            normalized = torch.tensor(projection["projected"])
            numerical = torch.tensor(projection["initial"][:, None])
            values = assignment_values(numerical, 5)
            old_weights = torch.einsum("sihk,strk->strih", raw, values)
            new_weights = torch.einsum("sihk,strk->strih", normalized, values)
            summary[variant] = {
                "primary": primary,
                "by_seed": seeds,
                "max_soft_weight_difference": float(
                    (old_weights - new_weights).abs().max()
                ),
                "max_probability_sum_difference": float(
                    (raw.sum(-1) - normalized.sum(-1)).abs().max()
                ),
                "max_pair_product": float(
                    (normalized[..., 1:3] * normalized[..., 3:5]).abs().max()
                ),
            }
            np.savez_compressed(
                destination / (variant + "_evaluation.npz"),
                test=test,
                population=population,
            )
    save_json(destination / "summary.json", summary)
    (destination / "figures").mkdir()
    figure, axes = plt.subplots(
        len(variants),
        3,
        figsize=(12, 4.8 * len(variants)),
        squeeze=False,
        layout="constrained",
    )
    source_summary = json.loads((source_root / "summary.json").read_text())
    for number, (variant, source) in enumerate(prepared.items()):
        projection = load_archive(destination / (variant + "_projection.npz"))
        selected = next(
            i
            for i, choice in enumerate(source["choices"])
            if choice["origin"]["search_seed"] == source["primary_seed"]
        )
        old = projection["raw"][selected]
        new = projection["projected"][selected]
        point = np.unravel_index(
            np.abs(new[..., 0] - old[..., 0]).argmax(), new.shape[:2]
        )
        for axis, values, title in (
            (axes[number, 0], old[..., 0], "Original P0"),
            (axes[number, 1], new[..., 0], "P0 after moving cancelling mass"),
        ):
            plotted = axis.imshow(values, cmap="viridis", vmin=0, vmax=1)
            axis.set_title(variant + ": " + title)
            axis.set_xticks(range(8))
            axis.set_yticks(range(11))
            axis.scatter(
                [point[1]], [point[0]], facecolors="none", edgecolors="red", s=180
            )
            figure.colorbar(plotted, ax=axis, shrink=0.75)
        before = source_summary[variant]["primary_test"][1]
        after = summary[variant]["primary"]["test"][1]
        draw_index(
            axes[number, 2],
            projection["indices"][selected],
            f"Projected argmax I: warm BA {100 * before:.2f}% -> {100 * after:.2f}%",
        )
    figure.suptitle(
        "Strict within-pair exclusivity before argmax; soft weights preserved\nSame selected states, four inherited C, same numeric budget; no penalty coefficient"
    )
    save_figure(figure, destination, "canonicalization")
    rows = []
    for variant, result in summary.items():
        old_seeds = {
            row["origin"]["search_seed"]: row
            for row in source_summary[variant]["by_seed"]
        }
        for row in result["by_seed"]:
            rows.append(
                [
                    variant,
                    row["seed"],
                    f"{100 * old_seeds[row['seed']]['test'][1]:.2f}",
                    f"{100 * row['immediate_test'][1]:.2f}",
                    f"{100 * row['test'][1]:.2f}",
                    f"{100 * row['population'][1]:.2f}",
                    f"{100 * row['structure']['mask_f1']:.2f}",
                ]
            )
    lines = [
        "# Удаление взаимно гасящейся массы перед argmax",
        "Для каждой пары +Ck/-Ck общая масса переносится в нулевую категорию. Правило не использует C, данные или учителя, сохраняет мягкую W для любого кодбука и сумму вероятностей. После преобразования не более одного слота каждой пары имеет положительную массу.",
        "$$\n\\begin{aligned}\nm_k&=\\min(P_{+k},P_{-k}),\\\\\nP'_0&=P_0+2\\sum_k m_k,\\\\\nP'_{+k}&=P_{+k}-m_k,\\qquad P'_{-k}=P_{-k}-m_k.\n\\end{aligned}\n$$",
        "Это отдельная абляция дискретизации: прежние выбранные конфигурация, search seed и snapshot сохранены. Все четыре исходных C обучаются с теми же4096 шагами и batch stream; query выбирает численный checkpoint. Ни новый search, ни подбор штрафа не требуются. Новый I и source fits заморожены до heldout. Задача и её test уже известны, поэтому анализ исследовательский и posthoc.",
        table(
            [
                "Метод",
                "Seed",
                "Прежний warm BA,%",
                "Новый argmax BA,%",
                "Новый warm BA,%",
                "BA всех2048,%",
                "F1 маски,%",
            ],
            rows,
        ),
        "Компенсация между разными базовыми кодами может сохраниться. Точное восстановление всей структуры со всех seed не гарантируется; группировки и маска оцениваются отдельно. Свежие C для новых projected I здесь не оценивались.",
        "![P0 до/после и выбранные I](figures/canonicalization.png)",
        "Сохранены исходные/преобразованные P, I, исходные C, все fits и метрики, max разность мягких W, сумм вероятностей и произведений противоположных слотов. Программные тесты не создавались и не запускались.",
    ]
    (destination / "REPORT.md").write_text("\n\n".join(lines) + "\n")
    print(
        "PROJECTION SUMMARY",
        {
            name: {
                "BA": row["primary"]["test"][1],
                "maskF1": row["primary"]["structure"]["mask_f1"],
                "softW_difference": row["max_soft_weight_difference"],
            }
            for name, row in summary.items()
        },
        flush=True,
    )


def prepare(parent):
    output = parent / "signed_codebook_20261104/regularization"
    output.mkdir(exist_ok=False)
    save_json(
        output / "registration.json",
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "variants": VARIANTS,
            "penalty_coefficients": [0.01, 0.1, 1.0],
            "rules": [
                "Exactly [0,C1,C2,-C1,-C2], two unconstrained trainable code values plus shared b/a/d. No single-base-code experiment.",
                "All models use identical initial P/C per search seed, same minibatches, numeric LR.03, alpha LR.01, sparsity0, annealed temperature1->.05 and late entropy.02. Only the new penalty differs.",
                "Pair product is mean over88 positions of sum_k P(+Ck)P(-Ck). It suppresses a code and its own twin; mixtures across different base codes can still cancel.",
                "Value variance is mean E[(v-E[v])^2] divided by mean(C1^2,C2^2), including v=0 and all signed values. Codebook values are detached in this penalty; only P gets its gradient. Normalization prevents an incentive to shrink all code values.",
                "Both penalties have constant coefficients from the first architecture update. Numerical C updates still optimize support BCE and the unchanged L2 only.",
                "Pilot seeds17/18/19, three penalty coefficients each; control has one zero-penalty setting. Mean hard selection BCE chooses one coefficient for each method, then confirm on20/21/22 with the same coefficient.",
                "Four numerical replicas,512 numeric warmup,6000 joint steps, snapshots0/1500/3000/4500/6000; argmax I followed by4096 inherited-C steps. Query chooses numeric checkpoints; independent selection chooses architecture checkpoints/configuration/seed. No truth or test ranking.",
                "Freeze the configuration winner and best individual source candidate before test. Fresh C controls use the same32 Gaussian starts and previous lr.03/norm.5; compare old fitting and64-step C/-C warmup with both branches charged to4096.",
                "All previous task tests are known. This is an exploratory follow-up with new search seeds, not a new benchmark. No software tests, commits or publication.",
            ],
        },
    )
    for variant in VARIANTS:
        folder = output / variant
        folder.mkdir()
        plan = json.loads(
            (parent / "signed_codebook_20261104/paired/protocol.json").read_text()
        )
        plan["output"] = str(folder)
        plan["search"].update(
            {
                "architecture_learning_rates": [0.01],
                "sparsity_penalties": [0.0],
                "pair_penalties": [0.01, 0.1, 1.0]
                if variant == "pair_product"
                else [0.0],
                "variance_penalties": [0.01, 0.1, 1.0]
                if variant == "value_variance"
                else [0.0],
                "pilot_seeds": [17, 18, 19],
                "confirmation_seeds": [20, 21, 22],
                "confirmation_top_configs": 1,
            }
        )
        plan["rules"] = [
            "See ../registration.json for this paired regularizer ablation."
        ]
        save_json(folder / "protocol.json", plan)
        shutil.copy2(parent / "source.npz", folder / "source.npz")
    snapshot = output / "code_snapshot"
    snapshot.mkdir()
    for name in (
        "signed_regularization.py",
        "signed_codebook.py",
        "search.py",
        "model.py",
        "single.py",
        "data.py",
    ):
        shutil.copy2(BASE / name, snapshot / name)
    print("PREPARED", output, flush=True)


def report(parent):
    output = parent / "signed_codebook_20261104/regularization"
    (output / "figures").mkdir(exist_ok=True)
    summary = json.loads((output / "summary.json").read_text())
    structures, projections = {}, {}
    rows, seed_rows, individual_rows, fresh_rows = [], [], [], []
    weights, titles = [], []
    figure, axes = plt.subplots(3, 2, figsize=(10, 15), layout="constrained")
    for number, (variant, row) in enumerate(summary.items()):
        folder = output / variant
        index = np.array(row["indices"])
        diagnostic = diagnose(index, True)
        seed_structures = [
            diagnose(np.array(seed["indices"]), True) for seed in row["by_seed"]
        ]
        individual = diagnose(np.array(row["individual"]["indices"]), True)
        structures[variant] = {
            "primary": diagnostic,
            "by_seed": seed_structures,
            "individual": individual,
        }
        draw_index(
            axes[number, 0],
            np.array(diagnostic["truth"]),
            variant + ": closest equivalent true grouping",
        )
        draw_index(
            axes[number, 1],
            np.array(diagnostic["aligned"]),
            f"Selected I: {diagnostic['errors']}/88 group errors; mask F1 {100 * diagnostic['mask_f1']:.2f}%",
        )
        codebook = np.r_[0, row["parameters"][:2]]
        raw = codebook[np.abs(index)] * np.sign(index)
        aligned = np.empty_like(raw)
        aligned[:, diagnostic["target_columns"]] = raw[:, diagnostic["source_columns"]]
        weights.append(aligned)
        titles.append(variant + ": actual weights")
        state = json.loads((folder / "frozen.json").read_text())
        origin = state["primary"]["origin"]
        snapshots = load_archive(folder / origin["folder"] / "snapshots.npz")
        p = load_archive(folder / "selected_probabilities.npz")["probabilities"]
        if not np.array_equal(CODE_LABELS[p.argmax(-1)], index):
            raise ValueError("Probability and frozen I artifacts disagree")
        numeric = torch.tensor(
            snapshots["numeric"][origin["snapshot"], origin["root"]][None]
        )
        with torch.no_grad():
            variances = (
                assignment_variance(torch.tensor(p[None]), numeric)
                .mean((0, 1, 2))
                .numpy()
            )
            values = assignment_values(numeric, 5).numpy()[0, 0]
        soft = np.einsum("ihk,rk->rih", p, values)
        hard = np.stack(
            [
                np.r_[0, replica[:2]][np.abs(index)] * np.sign(index)
                for replica in numeric.numpy()[0, 0]
            ]
        )
        pair = (p[..., 1:3] * p[..., 3:5]).sum(-1)
        relative = np.linalg.norm(
            (soft - hard).reshape(4, -1), axis=1
        ) / np.linalg.norm(soft.reshape(4, -1), axis=1).clip(min=1e-12)
        ambiguous = np.unravel_index(variances.argmax(), variances.shape)
        projections[variant] = {
            "mean_pair_product": float(pair.mean()),
            "max_pair_product": float(pair.max()),
            "mean_normalized_variance": float(variances.mean()),
            "max_normalized_variance": float(variances.max()),
            "mean_relative_soft_argmax_weight_error": float(relative.mean()),
            "max_variance_position_zero_based": [int(value) for value in ambiguous],
            "probabilities_at_position": p[ambiguous].tolist(),
            "soft_weights_at_position": soft[:, ambiguous[0], ambiguous[1]].tolist(),
            "argmax_weights_at_position": hard[:, ambiguous[0], ambiguous[1]].tolist(),
            "snapshot_effective_values": values.tolist(),
        }
        pfigure, paxes = plt.subplots(2, 3, figsize=(12, 10), layout="constrained")
        for code in range(5):
            axis = paxes.ravel()[code]
            plotted = axis.imshow(p[..., code], cmap="viridis", vmin=0, vmax=1)
            axis.set_title(f"P: label {CODE_LABELS[code]}")
            axis.set_xticks(range(8))
            axis.set_yticks(range(11))
            pfigure.colorbar(plotted, ax=axis, shrink=0.7)
        draw_index(paxes.ravel()[-1], index, "Raw I = argmax P")
        pfigure.suptitle(
            variant + ": selected P and indexator; -1/-2 are signed code references"
        )
        save_figure(pfigure, output, variant + "_probabilities")
        coefficient = origin.get("pair_penalty", 0.0) + origin.get(
            "variance_penalty", 0.0
        )
        rows.append(
            [
                variant,
                coefficient,
                f"{100 * row['relaxation']['soft'][1]:.2f}",
                f"{100 * row['relaxation']['argmax'][1]:.2f}",
                f"{100 * row['primary_test'][1]:.2f}",
                f"{100 * diagnostic['mask_f1']:.2f}",
                f"{sum(value['mask_exact'] for value in seed_structures)}/3",
                f"{relative.mean():.3f}",
            ]
        )
        for seed, structure in zip(row["by_seed"], seed_structures, strict=True):
            seed_rows.append(
                [
                    variant,
                    seed["origin"]["search_seed"],
                    seed["origin"]["step"],
                    structure["errors"],
                    f"{100 * structure['mask_f1']:.2f}",
                    f"{100 * seed['test'][1]:.2f}",
                    f"{100 * seed['population'][1]:.2f}",
                ]
            )
        individual_rows.append(
            [
                variant,
                individual["errors"],
                f"{100 * individual['mask_f1']:.2f}",
                f"{100 * row['individual']['test'][1]:.2f}",
                f"{100 * row['individual']['population'][1]:.2f}",
            ]
        )
        for name, result in row["fresh"].items():
            if name.startswith("primary"):
                fresh_rows.append(
                    [
                        variant,
                        name,
                        f"{100 * result['mean_test_ba']:.2f}",
                        f"{result['perfect_test']}/32",
                        f"{100 * result['selected_test'][1]:.2f}",
                    ]
                )
    figure.suptitle(
        "Five assignment classes, two free code values: regularizer ablation\nGround-truth matching is diagnostic only, after selection"
    )
    save_figure(figure, output, "selected_indexators")
    baseline = load_archive(parent / "adam_baseline/weight_heatmaps.npz")
    matrices = [baseline["weights"][0], *weights, baseline["weights"][2]]
    labels = [
        "True signed filters",
        *titles,
        "Dense Adam: shared head, previous control",
    ]
    bound = max(float(np.abs(value).max()) for value in matrices)
    figure, axes = plt.subplots(2, 3, figsize=(14, 10), layout="constrained")
    for axis, values, title in zip(axes.ravel(), matrices, labels):
        plotted = axis.imshow(values, cmap="RdBu_r", vmin=-bound, vmax=bound)
        axis.set_title(title, fontsize=10)
        axis.set_xticks(range(8))
        axis.set_yticks(range(11))
    axes.ravel()[-1].set_visible(False)
    figure.colorbar(
        plotted,
        ax=list(axes.ravel()[:-1]),
        shrink=0.8,
        label="Actual weight; common scale",
    )
    figure.suptitle(
        "Actual weights, no normalization or quantization; columns matched after freezing"
    )
    save_figure(figure, output, "weight_comparison")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for variant in VARIANTS:
        folder = output / variant
        state = json.loads((folder / "frozen.json").read_text())
        fits = single.combined(folder / "confirmation")
        ids = np.flatnonzero(fits["entry_ids"] == state["primary"]["entry"])
        axes[0].plot(
            fits["history_steps"], fits["query_history"][:, ids].mean(1), label=variant
        )
        axes[1].plot(
            fits["history_steps"],
            100 * fits["recall_history"][:, ids].mean(1),
            label=variant,
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Mean query BCE, four inherited C")
    axes[1].set_ylabel("Mean query positive recall, %")
    for axis in axes:
        axis.set_xlabel("Numeric updates after argmax")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.15)
    save_figure(figure, output, "inherited_curves")
    figure, axis = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for variant in ("pair_product", "value_variance"):
        calibration = json.loads(
            (output / variant / "pilot_selection.json").read_text()
        )
        groups = sorted(
            calibration["groups"],
            key=lambda row: (
                row["seeds"][0]["origin"]["pair_penalty"]
                + row["seeds"][0]["origin"]["variance_penalty"]
            ),
        )
        coefficients = [
            row["seeds"][0]["origin"]["pair_penalty"]
            + row["seeds"][0]["origin"]["variance_penalty"]
            for row in groups
        ]
        axis.plot(
            coefficients,
            [row["mean_selection_bce"] for row in groups],
            marker="o",
            label=variant,
        )
    control = json.loads((output / "control/pilot_selection.json").read_text())[
        "selected_group"
    ]["mean_selection_bce"]
    axis.axhline(control, color="grey", linestyle="--", label="control, no new penalty")
    axis.set_xscale("log")
    axis.set_xlabel("Penalty coefficient, constant from first P update")
    axis.set_ylabel("Mean pilot selection BCE over three search seeds")
    axis.set_title("Source-only tuning before confirmation/test")
    axis.legend()
    axis.grid(alpha=0.15)
    save_figure(figure, output, "penalty_calibration")
    save_json(output / "selected_indexators.json", structures)
    save_json(output / "projection_diagnostics.json", projections)
    lines = [
        "# Pattern 0110: пресечение компенсации между знаковыми кодами",
        "Во всех трёх вариантах ровно [0,c1,c2,-c1,-c2], два свободных числа кодбука и b/a/d. Сравниваются контроль, произведение вероятностей противоположных слотов одной пары и дисперсия численных значений смеси. Это мягкие штрафы, не строгие дискретные запреты.",
        "$$\nL_{\\mathrm{pair}}=\\lambda\\frac1{88}\\sum_{ij}\\sum_{k=1}^2 P_{ij,+k}P_{ij,-k}.\n$$",
        "Первый штраф подавляет смесь внутри своей пары. Компенсация между разными базовыми кодами остаётся возможной.",
        "$$\nL_{\\mathrm{var}}=\\lambda\\frac1{88}\\sum_{ij}\\frac{\\sum_k P_{ij,k}(v_k-\\bar w_{ij})^2}{\\max(\\tfrac12(c_1^2+c_2^2),\\varepsilon)},\\qquad v=(0,c_1,c_2,-c_1,-c_2).\n$$",
        "Второй штраф включает нулевое значение и любые разные численные значения. Он допускает распределение массы между фактически одинаковыми значениями: их выбор через argmax не меняет вес. Масштаб нормирован, C отсоединён от графа этого штрафа, чтобы он обновлял только P. Числа C по-прежнему учатся по support BCE и прежнему L2.",
        "Все начальные P/C и minibatch одинаковы для одного search seed. Numeric LR.03, alpha LR.01, sparsity0, прежние annealing/entropy. Коэффициенты .01/.1/1 действуют с первого архитектурного шага. Pilot17/18/19 выбирает один коэффициент по mean hard selection BCE, confirmation20/21/22 его проверяет. Максимум6000 joint шагов, четыре C,512 warmup; пять снимков и4096 дообучения C после argmax. Query выбирает численные checkpoint, независимая selection выбирает I и настройки; teacher/test не ранжируют кандидатов.",
        table(
            [
                "Метод",
                "Lambda",
                "Soft test BA,%",
                "Argmax BA,%",
                "Warm hard BA,%",
                "F1 маски,%",
                "Точных масок/seed",
                "Относительный скачок W",
            ],
            rows,
        ),
        "Soft/argmax — среднее четырёх текущих C до дообучения; warm — кодбук, выбранный по query после дообучения. Точные маски и строгие группировки оцениваются отдельно: одинаковые роли могут использовать разные почти равные signed-значения.",
        table(
            [
                "Метод",
                "Search seed",
                "Снимок",
                "Ошибок групп/88",
                "F1 маски,%",
                "Warm test BA,%",
                "BA всех2048,%",
            ],
            seed_rows,
        ),
        "Лучший индивидуальный кандидат по source selection среди всего pilot/confirmation также заморожен до test; он оценивает возможность найти решение и не заменяет основной выбор конфигурации.",
        table(
            ["Метод", "Ошибок групп/88", "F1 маски,%", "Test BA,%", "BA всех2048,%"],
            individual_rows,
        ),
        "![Выбранные индексаторы](figures/selected_indexators.png)",
        "![Контроль P](figures/control_probabilities.png)",
        "![Штраф внутри пары](figures/pair_product_probabilities.png)",
        "![Штраф дисперсии](figures/value_variance_probabilities.png)",
        "![Фактические веса и Adam](figures/weight_comparison.png)",
        "![Продолжение исходных C](figures/inherited_curves.png)",
        "![Калибровка штрафов на source](figures/penalty_calibration.png)",
        "Свежие C — отдельный контроль с теми же32 Gaussian-стартами, lr.03/norm.5, checkpoint/старт по query. warmup64 выбирает C/-C; бюджет4096 учитывает отброшенную ветку. Тест уже изучался в предыдущих сериях: это исследовательская абляция на известной задаче, новые search seed не образуют новый benchmark.",
        table(
            [
                "Метод",
                "Fresh процедура",
                "Mean test BA,%",
                "Точных/32",
                "Query-selected BA,%",
            ],
            fresh_rows,
        ),
        "Полные оценки каждой пары I/C, происхождение выбора, P, исходники, фактические кривые и диагностика компенсации сохранены в NPZ/JSON. Программные тесты не создавались и не запускались.",
    ]
    projection_path = output / "projection/summary.json"
    if projection_path.exists():
        projected = json.loads(projection_path.read_text())
        projection_rows = [
            [
                variant,
                f"{100 * summary[variant]['primary_test'][1]:.2f}",
                f"{100 * result['primary']['immediate_test'][1]:.2f}",
                f"{100 * result['primary']['test'][1]:.2f}",
                f"{100 * result['primary']['population'][1]:.2f}",
                f"{100 * result['primary']['structure']['mask_f1']:.2f}",
            ]
            for variant, result in projected.items()
        ]
        lines += [
            "## Точная несовместимость двух знаков перед argmax",
            "Вместо нового loss можно убрать взаимно гасящуюся массу из каждой пары и перенести её в нулевой слот. Для любого C мягкая W сохраняется, а произведение вероятностей внутри пары становится ровно нулевым. Это отдельная абляция hardening на тех же выбранных конфигурации/seed/snapshot и четырёх исходных C. Поиск P не повторяется, численный бюджет после argmax тот же4096. Подробности и все три seed — [projection/REPORT.md](projection/REPORT.md).",
            "$$\n\\begin{aligned}\nm_k&=\\min(P_{+k},P_{-k}),\\\\\nP'_0&=P_0+2\\sum_km_k,\\\\\nP'_{+k}&=P_{+k}-m_k,\\quad P'_{-k}=P_{-k}-m_k.\n\\end{aligned}\n$$",
            table(
                [
                    "Метод",
                    "Прежний warm BA,%",
                    "Новый argmax BA,%",
                    "Новый warm BA,%",
                    "BA всех2048,%",
                    "F1 маски,%",
                ],
                projection_rows,
            ),
            "![Нулевой индекс и I после переноса](projection/figures/canonicalization.png)",
            "На исходном пятизначном кодбуке seed15 тот же приём исправил единственную лишнюю связь: обычный immediate argmax90.71%, после канонизации100%; после одинакового C-fit98.15%→100%, маска стала точной. Два других выбранных seed от этого не стали точными. [Отдельная проверка](projection_original/REPORT.md).",
            "На новом control seed20 функция после канонизации тоже решается на100% всех2048, однако объявленная маска содержит44 связи вместо32. Причина отличается: один свободный код используется как почти нулевой (в исходном raw-fit C1≈-0.0203 при C2≈1.969), поэтому ненулевые имена индексов не обязательно обозначают существенные веса. Перенос внутри пары не устраняет такой дубль нулевого кода. Это демонстрирует различие качества функции, точности бинарной маски и точности группировки.",
            "В проверенной сетке постоянных коэффициентов .01/.1/1 штрафы не дали преимущества; меньшие коэффициенты и позднее включение здесь не исследовались. Канонизация строго устраняет только компенсацию между собственными двойниками и не гарантирует восстановление GT или сходимость каждого search seed.",
        ]
    (output / "REPORT.md").write_text("\n\n".join(lines) + "\n")
    snapshot = output / "analysis_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in (
        "signed_regularization.py",
        "signed_codebook.py",
        "search.py",
        "model.py",
        "single.py",
        "data.py",
    ):
        shutil.copy2(BASE / name, snapshot / name)
    save_json(
        output / "complete.json",
        {
            "completed_utc": datetime.now(UTC).isoformat(),
            "software_tests_run": False,
            "structures": structures,
            "projections": projections,
            "artifacts": {
                str(path.relative_to(output)): digest(path)
                for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "complete.json"
            },
        },
    )
    print(
        table(
            [
                "Method",
                "Lambda",
                "Soft BA",
                "Argmax BA",
                "Warm BA",
                "Mask F1",
                "Exact masks",
                "W jump",
            ],
            rows,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "prepare",
            "pilot",
            "confirmation",
            "freeze",
            "evaluate",
            "project",
            "project_original",
            "report",
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=BASE / "results/single_0110_20261104"
    )
    parser.add_argument("--variant", choices=tuple(VARIANTS))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    output = args.output / "signed_codebook_20261104/regularization"
    device = torch.device(args.device)
    if args.stage == "prepare":
        prepare(args.output)
    elif args.stage in ("pilot", "confirmation"):
        for variant in [args.variant] if args.variant else VARIANTS:
            search_phase(output / variant, args.stage, device)
    elif args.stage == "freeze":
        freeze(output, VARIANTS)
    elif args.stage == "evaluate":
        evaluate(args.output, output, VARIANTS, device)
    elif args.stage == "project":
        projection_ablation(
            args.output, output, output / "projection", VARIANTS, device
        )
    elif args.stage == "project_original":
        projection_ablation(
            args.output,
            args.output / "signed_codebook_20261104",
            output / "projection_original",
            {"paired": 5},
            device,
        )
    else:
        report(args.output)


if __name__ == "__main__":
    main()
