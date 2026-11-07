"""Compare [0,C1,C2] with [0,C1,C2,-C1,-C2] on fixed Pattern 0110."""

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
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.optimize import linear_sum_assignment

from pattern.datasets import input_table, labels_for

from . import single
from .data import BASE, digest, load_archive, save_json
from .model import initial_numeric, metrics, soft_predict
from .report import save_figure, table
from .search import train

VARIANTS = {"baseline": 3, "paired": 5}
CODE_LABELS = np.array([0, 1, 2, -1, -2], dtype=np.int16)
COLORS = ["#4879b8", "#73a9dc", "#edf0f2", "#e49d48", "#ac72b3"]


def prepare(parent):
    output = parent / "signed_codebook_20261104"
    output.mkdir(exist_ok=False)
    save_json(
        output / "registration.json",
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "variants": VARIANTS,
            "source_sha256": digest(parent / "source.npz"),
            "rules": [
                "Only fixed Pattern0110, same 768/256/512/512 support/query/selection/test inputs as previous experiments. No single-base-code variant.",
                "Both variants have two unconstrained trainable code values C1/C2 and shared b/a/d: five numerical parameters. Paired classes are exactly 0,C1,C2,-C1,-C2, not four independent numerical values.",
                "Use identical initial C1<0<C2 across numerical replicas in both models, same numeric seeds, support minibatch streams, and generic zero prior .5. Alpha shapes differ, so initial effective weight matrices need not coincide. No sign constraints during optimization.",
                "Joint support P/C learning, four numeric replicas,512 numeric warmup,6000 joint updates, snapshots0/1500/3000/4500/6000. Annealed softmax and entropy penalty as the previous tuned joint control.",
                "Compare alpha LR .003/.01/.03 crossed with sparsity0/.01 at numeric LR.03. Pilot search seeds11/12/13, top2 configurations by mean hard selection BCE confirmed on14/15/16.",
                "At each snapshot freeze raw argmax I; negative classes become signed integer labels. Continue all four inherited C for4096 steps with reset Adam moments. Query chooses numeric checkpoint, mean selection BCE chooses architecture checkpoint/configuration, then primary seed.",
                "Also freeze the best individual source candidate across all pilot/confirmation checkpoints before test; do not substitute it for the main configuration winner.",
                "Freeze both primary architectures and source selections before test. Compare inherited C,32 common unconstrained Gaussian fresh starts at previously selected lr.03/norm.5, and the previously source-selected64-step C/-C initialization. The4096 budget charges both warmup branches.",
                "Truth is inaccessible to search/selection. Post-freeze I diagnostics allow column permutations and global group/sign symmetries; paired truth may use one opposite-sign pair or two independent groups. Record mask recovery separately.",
                "This is exploratory research on a previously inspected task, not a new benchmark. No software tests, commits, publication, or deleted experiments restored.",
            ],
        },
    )
    for variant, classes in VARIANTS.items():
        folder = output / variant
        folder.mkdir()
        plan = json.loads((parent / "protocol.json").read_text())
        plan.update({"output": str(folder), "classes": classes})
        plan["search"].update(
            {
                "numeric_learning_rates": [0.03],
                "architecture_learning_rates": [0.003, 0.01, 0.03],
                "schedules": ["annealed"],
                "sparsity_penalties": [0.0, 0.01],
                "pilot_seeds": [11, 12, 13],
                "confirmation_seeds": [14, 15, 16],
                "confirmation_orders": ["first"],
                "update_mode": "joint_support",
                "numeric_steps_per_outer": 1,
                "paired_batch_streams": True,
            }
        )
        plan["rules"] = [
            "Inherited single-task protocol is superseded by ../registration.json.",
            f"Exactly {classes} assignment classes; two numerical code parameters.",
        ]
        save_json(folder / "protocol.json", plan)
        shutil.copy2(parent / "source.npz", folder / "source.npz")
    snapshot = output / "code_snapshot"
    snapshot.mkdir()
    for name in ("signed_codebook.py", "search.py", "model.py", "single.py", "data.py"):
        shutil.copy2(BASE / name, snapshot / name)
    print("PREPARED", output, flush=True)


def oracle_control(parent, device):
    output = parent / "signed_codebook_20261104"
    original = load_archive(
        parent / "numeric_optimization_20261104/original_oracle.npz"
    )
    plan = json.loads((parent / "protocol.json").read_text())
    indices = original["indices"].copy()
    indices[indices == 1] = -1
    initial = original["initial_parameters"].copy()
    initial[:, 0] *= -1
    result = single.fit_fixed(
        load_archive(parent / "source.npz"),
        indices,
        initial,
        [0.03] * len(initial),
        plan["fit"],
        plan["seed"] + 14000001,
        device,
    )
    np.savez_compressed(output / "fixed_I_control.npz", **result)
    summary = {
        "purpose": "Reparameterize the same physical fixed oracle I as -C1,+C2; negative slots alone should not change fitting when I cannot move.",
        "max_query_bce_difference": float(
            np.abs(
                original["query_metrics"][:, 0] - result["query_metrics"][:, 0]
            ).max()
        ),
        "original_perfect_selection": int(
            (original["selection_metrics"][:, 1] >= 1 - 1e-6).sum()
        ),
        "paired_perfect_selection": int(
            (result["selection_metrics"][:, 1] >= 1 - 1e-6).sum()
        ),
        "starts": len(initial),
    }
    save_json(output / "fixed_I_control.json", summary)
    print("FIXED I", summary, flush=True)


def search_phase(folder, phase, device):
    plan = json.loads((folder / "protocol.json").read_text())
    train(plan, folder, phase, device, 0, 1)
    entries, indices, initial = [], [], []
    for shard in sorted((folder / phase / "search").glob("shard_*")):
        specifications = json.loads((shard / "roots.json").read_text())
        snapshots = load_archive(shard / "snapshots.npz")
        for snapshot, step in enumerate(snapshots["steps"]):
            for root, specification in enumerate(specifications):
                classes = snapshots["alpha"][snapshot, root].argmax(-1)
                indices.append(CODE_LABELS[classes])
                initial.append(snapshots["numeric"][snapshot, root, 0])
                entries.append(
                    {
                        **specification,
                        "entry": len(entries),
                        "step": int(step),
                        "snapshot": snapshot,
                        "root": root,
                        "folder": str(shard.relative_to(folder)),
                    }
                )
    destination = folder / phase
    save_json(destination / "hard_candidates.json", entries)
    np.savez_compressed(
        destination / "hard_candidates.npz",
        indices=np.stack(indices),
        inherited_numeric=np.stack(initial),
    )
    restarts = plan["search"]["numeric_restarts"]
    result = single.fit_fixed(
        load_archive(folder / "source.npz"),
        np.repeat(np.stack(indices), restarts, 0),
        np.stack(initial).reshape(-1, 5),
        [entry["numeric_lr"] for entry in entries for _ in range(restarts)],
        plan["fit"],
        plan["seed"] + 10000001,
        device,
    )
    np.savez_compressed(
        destination / "hard_fits_0.npz",
        entry_ids=np.repeat(np.arange(len(entries)), restarts),
        **result,
    )
    single.select(plan, folder, phase)
    print("FINISHED", folder.name, phase, len(entries), "hard candidates", flush=True)


def freeze(output, variants):
    for variant in variants:
        folder = output / variant
        plan = json.loads((folder / "protocol.json").read_text())
        single.freeze(plan, folder)
        choices = []
        for phase in ("pilot", "confirmation"):
            row = json.loads((folder / phase / "ranking.json").read_text())[0]
            choices.append({"phase": phase, **row})
        selected = min(choices, key=lambda row: row["selection_bce"])
        phase = folder / selected["phase"]
        archive = load_archive(phase / "hard_candidates.npz")
        fits = single.combined(phase)
        rows = np.flatnonzero(fits["entry_ids"] == selected["entry"])
        row = rows[np.argmin(fits["query_metrics"][rows, 0])]
        np.savez_compressed(
            folder / "individual.npz",
            indices=archive["indices"][selected["entry"]],
            parameters=fits["parameters"][row],
        )
        save_json(folder / "individual.json", selected)
    save_json(
        output / "frozen.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "heldout_evaluated": False,
            "architectures": {
                variant: digest(output / variant / "frozen.npz") for variant in variants
            },
            "fresh_lr": 0.03,
            "fresh_norm": 0.5,
            "fresh_starts": 32,
            "fresh_initialization_seed_offset": 26000001,
            "fresh_batch_seed_offset": 27000001,
            "fresh_policies": [0, 64],
        },
    )


def evaluate(parent, output, variants, device):
    frozen = json.loads((output / "frozen.json").read_text())
    for variant in variants:
        folder = output / variant
        plan = json.loads((folder / "protocol.json").read_text())
        state = load_archive(folder / "frozen.npz")
        source = load_archive(folder / "source.npz")
        initial = (
            initial_numeric(
                [plan["seed"] + frozen["fresh_initialization_seed_offset"]],
                1,
                frozen["fresh_starts"],
                frozen["fresh_norm"],
                torch.device("cpu"),
                codes=2,
            )
            .numpy()
            .reshape(-1, 5)
        )
        truth = np.zeros((11, 8), dtype=np.int16)
        for column in range(8):
            truth[column : column + 4, column] = [1, 2, 2, 1]
        for warmup in frozen["fresh_policies"]:
            indices = np.repeat(np.stack([state["indices"], truth]), len(initial), 0)
            result = single.fit_fixed(
                source,
                indices,
                np.tile(initial, (2, 1)),
                [frozen["fresh_lr"]] * len(indices),
                {**plan["fit"], "paired_initialization_steps": warmup},
                plan["seed"] + frozen["fresh_batch_seed_offset"],
                device,
            )
            np.savez_compressed(folder / f"fresh_{warmup}.npz", **result)
    save_json(
        output / "numerical_frozen.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "heldout_evaluated": False,
            "fits": {
                str(path.relative_to(output)): digest(path)
                for path in sorted(output.glob("*/fresh_*.npz"))
            },
        },
    )
    heldout = load_archive(parent / "heldout.npz")
    _, all_x = input_table()
    test_x = torch.tensor(heldout["test_x"], device=device)
    test_y = torch.tensor(heldout["test_y"][0], device=device)[None]
    population_x = all_x.to(device)
    population_y = labels_for(population_x, "0110")[None]

    def scores(index, parameters):
        index = torch.tensor(index, dtype=torch.long, device=device)
        numerical = torch.tensor(parameters, device=device)
        with torch.no_grad():
            return {
                "test": metrics(single.predict(test_x, index, numerical), test_y)
                .cpu()
                .numpy(),
                "population": metrics(
                    single.predict(population_x, index, numerical), population_y
                )
                .cpu()
                .numpy(),
            }

    summary = {}
    for variant, classes in variants.items():
        folder = output / variant
        chosen = json.loads((folder / "confirmation_selection.json").read_text())
        state = load_archive(folder / "frozen.npz")
        fits = single.combined(folder / "confirmation")
        seeds = []
        for row, index in zip(
            chosen["selected_group"]["seeds"], state["by_seed"], strict=True
        ):
            rows = np.flatnonzero(fits["entry_ids"] == row["entry"])
            selected = rows[np.argmin(fits["query_metrics"][rows, 0])]
            current = scores(index[None], fits["parameters"][selected : selected + 1])
            seeds.append(
                {
                    **row,
                    "indices": index.tolist(),
                    "parameters": fits["parameters"][selected].tolist(),
                    "test": current["test"][0].tolist(),
                    "population": current["population"][0].tolist(),
                }
            )
        individual = load_archive(folder / "individual.npz")
        individual_scores = scores(
            individual["indices"][None], individual["parameters"][None]
        )
        primary = scores(state["indices"][None], state["parameters"][None])
        origin = chosen["primary"]["origin"]
        snapshot = load_archive(folder / origin["folder"] / "snapshots.npz")
        history = json.loads((folder / origin["folder"] / "history.json").read_text())
        temperature = next(row for row in history if row["step"] == origin["step"])[
            "temperature"
        ][origin["root"]]
        probabilities = torch.softmax(
            torch.tensor(
                snapshot["alpha"][origin["snapshot"], origin["root"]], device=device
            )
            / temperature,
            -1,
        )
        numeric = torch.tensor(
            snapshot["numeric"][origin["snapshot"], origin["root"]][None], device=device
        )
        hard = torch.nn.functional.one_hot(
            probabilities.argmax(-1), probabilities.shape[-1]
        ).float()
        relaxation = {}
        with torch.no_grad():
            for name, values in (("soft", probabilities), ("argmax", hard)):
                relaxation[name] = (
                    metrics(
                        soft_predict(test_x, values[None], numeric),
                        test_y[None, :, None],
                    )
                    .mean((0, 1, 2))
                    .cpu()
                    .tolist()
                )
        np.savez_compressed(
            folder / "selected_probabilities.npz",
            probabilities=probabilities.cpu().numpy(),
            indices=state["indices"],
            temperature=np.array(temperature),
        )
        fresh = {}
        for warmup in frozen["fresh_policies"]:
            result = load_archive(folder / f"fresh_{warmup}.npz")
            current = scores(result["indices"], result["parameters"])
            for number, name in enumerate(("primary", "oracle")):
                rows = np.arange(number * 32, (number + 1) * 32)
                selected = rows[result["query_metrics"][rows, 0].argmin()]
                fresh[f"{name}_warmup{warmup}"] = {
                    "mean_test_ba": float(current["test"][rows, 1].mean()),
                    "perfect_test": int((current["test"][rows, 1] >= 1 - 1e-6).sum()),
                    "selected_test": current["test"][selected].tolist(),
                    "selected_population": current["population"][selected].tolist(),
                }
            np.savez_compressed(folder / f"fresh_{warmup}_evaluation.npz", **current)
        summary[variant] = {
            "classes": classes,
            "parameters": state["parameters"].tolist(),
            "indices": state["indices"].tolist(),
            "selected_configuration": chosen["selected_group"]["configuration"],
            "mean_selection_bce": chosen["selected_group"]["mean_selection_bce"],
            "primary_test": primary["test"][0].tolist(),
            "primary_population": primary["population"][0].tolist(),
            "by_seed": seeds,
            "individual": {
                "indices": individual["indices"].tolist(),
                "test": individual_scores["test"][0].tolist(),
                "population": individual_scores["population"][0].tolist(),
            },
            "relaxation": relaxation,
            "fresh": fresh,
        }
        print("EVALUATED", variant, summary[variant]["primary_test"], flush=True)
    save_json(output / "summary.json", summary)


def diagnose(index, paired):
    choices = (-2, -1, 1, 2) if paired else (1, 2)
    best = None
    for zero_code, one_code in itertools.permutations(choices, 2):
        truth = np.zeros((11, 8), dtype=np.int16)
        for column in range(8):
            truth[column : column + 4, column] = [
                zero_code,
                one_code,
                one_code,
                zero_code,
            ]
        source, target = linear_sum_assignment(
            (index[:, :, None] != truth[:, None, :]).sum(0)
        )
        aligned = np.empty_like(index)
        aligned[:, target] = index[:, source]
        errors = int((aligned != truth).sum())
        if best is None or errors < best[0]:
            best = errors, aligned, truth, source, target
    source, target = linear_sum_assignment(
        -((index != 0).astype(int).T @ (best[2] != 0).astype(int))
    )
    support = np.empty_like(index, dtype=bool)
    support[:, target] = index[:, source] != 0
    expected = best[2] != 0
    tp = int((support & expected).sum())
    fp = int((support & ~expected).sum())
    fn = int((~support & expected).sum())
    return {
        "raw": index.tolist(),
        "errors": best[0],
        "exact": best[0] == 0,
        "aligned": best[1].tolist(),
        "truth": best[2].tolist(),
        "source_columns": best[3].tolist(),
        "target_columns": best[4].tolist(),
        "mask_f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "mask_exact": fp == 0 and fn == 0,
        "active": int((index != 0).sum()),
        "label_counts": {
            str(code): int((index == code).sum()) for code in (0, 1, 2, -1, -2)
        },
    }


def draw_index(axis, index, title):
    axis.imshow(
        index, cmap=ListedColormap(COLORS), norm=BoundaryNorm(np.arange(-2.5, 3), 5)
    )
    for row, column in np.ndindex(index.shape):
        axis.text(
            column, row, str(index[row, column]), ha="center", va="center", fontsize=8
        )
    axis.set_title(title, fontsize=10)
    axis.set_xticks(range(8))
    axis.set_yticks(range(11))
    axis.set_xlabel("Hidden unit")
    axis.set_ylabel("Input coordinate")


def report(parent):
    output = parent / "signed_codebook_20261104"
    (output / "figures").mkdir(exist_ok=True)
    summary = json.loads((output / "summary.json").read_text())
    structures = {}
    figure, axes = plt.subplots(2, 3, figsize=(13, 11), layout="constrained")
    actual_weights, weight_titles = [], []
    for number, (variant, row) in enumerate(summary.items()):
        index = np.array(row["indices"])
        diagnostics = diagnose(index, variant == "paired")
        seeds = [
            diagnose(np.array(seed["indices"]), variant == "paired")
            for seed in row["by_seed"]
        ]
        structures[variant] = {
            "primary": diagnostics,
            "by_seed": seeds,
            "exact_seeds": sum(seed["exact"] for seed in seeds),
            "individual": diagnose(
                np.array(row["individual"]["indices"]), variant == "paired"
            ),
        }
        truth = np.array(diagnostics["truth"])
        aligned = np.array(diagnostics["aligned"])
        draw_index(
            axes[number, 0], truth, variant + ": closest equivalent true I (diagnostic)"
        )
        draw_index(
            axes[number, 1], aligned, f"Selected I: {diagnostics['errors']}/88 errors"
        )
        draw_index(axes[number, 2], index, "Raw selected I, before column matching")
        values = np.r_[0, row["parameters"][:2]]
        raw_weights = values[np.abs(index)] * np.sign(index)
        weights = np.empty_like(raw_weights)
        weights[:, diagnostics["target_columns"]] = raw_weights[
            :, diagnostics["source_columns"]
        ]
        actual_weights.append(weights)
        weight_titles.append(variant + ": actual selected weights")
        probabilities = load_archive(output / variant / "selected_probabilities.npz")
        p = probabilities["probabilities"]
        if not np.array_equal(CODE_LABELS[p.argmax(-1)], index):
            raise ValueError("Selected probability and indexator artifacts disagree")
        frozen = json.loads((output / variant / "frozen.json").read_text())
        origin = frozen["primary"]["origin"]
        snapshots = load_archive(output / variant / origin["folder"] / "snapshots.npz")
        numeric = snapshots["numeric"][origin["snapshot"], origin["root"], 0]
        values = numeric[:, :2]
        if variant == "paired":
            values = np.concatenate([values, -values], -1)
        soft_weights = np.einsum("ihk,rk->rih", p[..., 1:], values)
        hard_weights = np.stack(
            [
                np.r_[0, replica[:2]][np.abs(index)] * np.sign(index)
                for replica in numeric
            ]
        )
        relative_error = np.linalg.norm(
            (soft_weights - hard_weights).reshape(4, -1), axis=1
        ) / np.linalg.norm(soft_weights.reshape(4, -1), axis=1).clip(min=1e-12)
        effective = np.array([0, *row["parameters"][:2]])
        if variant == "paired":
            effective = np.r_[effective, -np.array(row["parameters"][:2])]
        structures[variant]["projection"] = {
            "relative_soft_to_argmax_weight_errors": relative_error.tolist(),
            "selected_effective_values": effective.tolist(),
            "mean_probability_max": float(p.max(-1).mean()),
        }
        ambiguous = np.unravel_index(p.max(-1).argmin(), p.shape[:2])
        structures[variant]["projection"]["most_ambiguous_position"] = {
            "raw_input_hidden_zero_based": [int(value) for value in ambiguous],
            "probabilities_in_class_order": p[ambiguous].tolist(),
            "soft_weights_four_replicas": soft_weights[
                :, ambiguous[0], ambiguous[1]
            ].tolist(),
            "argmax_weights_four_replicas": hard_weights[
                :, ambiguous[0], ambiguous[1]
            ].tolist(),
            "snapshot_effective_values": np.concatenate(
                [np.zeros((4, 1)), values], -1
            ).tolist(),
        }
        pfigure, paxes = plt.subplots(2, 3, figsize=(12, 10), layout="constrained")
        for code in range(p.shape[-1]):
            axis = paxes.ravel()[code]
            plotted = axis.imshow(p[..., code], cmap="viridis", vmin=0, vmax=1)
            axis.set_title(f"P: label {CODE_LABELS[code]}")
            axis.set_xticks(range(8))
            axis.set_yticks(range(11))
            pfigure.colorbar(plotted, ax=axis, shrink=0.7)
        for axis in paxes.ravel()[p.shape[-1] : -1]:
            axis.set_visible(False)
        draw_index(paxes.ravel()[-1], index, "I = argmax, mapped to signed labels")
        if variant == "paired":
            for axis in paxes.ravel():
                axis.scatter(
                    [ambiguous[1]],
                    [ambiguous[0]],
                    s=180,
                    facecolors="none",
                    edgecolors="red",
                    linewidths=1.4,
                )
        pfigure.suptitle(
            f"{variant}: frozen source-selected P; sum P = 1, temperature {float(probabilities['temperature']):.2f}"
            + (
                "\nRed circle: C1/-C1 mixture nearly cancels; argmax creates a nonzero weight"
                if variant == "paired"
                else ""
            )
        )
        save_figure(pfigure, output, variant + "_probabilities")
    figure.suptitle(
        "Pattern 0110: free versus sign-paired codebook\nLabels -1/-2 mean -C1/-C2; C values remain unconstrained; alignment only after selection"
    )
    save_figure(figure, output, "selected_indexators")
    figure, axes = plt.subplots(2, 3, figsize=(13, 11), layout="constrained")
    for number, variant in enumerate(VARIANTS):
        diagnostic = structures[variant]["individual"]
        origin = json.loads((output / variant / "individual.json").read_text())[
            "origin"
        ]
        draw_index(
            axes[number, 0],
            np.array(diagnostic["truth"]),
            variant + ": closest equivalent true grouping",
        )
        draw_index(
            axes[number, 1],
            np.array(diagnostic["aligned"]),
            f"Best by validation: I errors {diagnostic['errors']}/88; mask F1 {100 * diagnostic['mask_f1']:.0f}%",
        )
        draw_index(
            axes[number, 2],
            np.array(diagnostic["raw"]),
            f"Raw I: seed {origin['search_seed']}, snapshot {origin['step']}",
        )
    figure.suptitle(
        "Best individual I by independent validation, frozen before test\nBoth solve all 2048 inputs exactly; paired uses several labels for the same pattern roles"
    )
    save_figure(figure, output, "best_individual_indexators")
    baseline = load_archive(parent / "adam_baseline/weight_heatmaps.npz")
    weights = [baseline["weights"][0], *actual_weights, baseline["weights"][2]]
    titles = [
        "True signed filter weights",
        *weight_titles,
        "Dense Adam: shared head (previous control)",
    ]
    bound = max(float(np.abs(values).max()) for values in weights)
    figure, axes = plt.subplots(1, 4, figsize=(15, 5.5), layout="constrained")
    for axis, values, title in zip(axes, weights, titles, strict=True):
        plotted = axis.imshow(values, cmap="RdBu_r", vmin=-bound, vmax=bound)
        axis.set_title(title, fontsize=10)
        axis.set_xticks(range(8))
        axis.set_yticks(range(11))
        axis.set_xlabel("Hidden unit, matched for display")
        axis.set_ylabel("Input coordinate")
    figure.colorbar(
        plotted, ax=list(axes), shrink=0.8, label="Actual weight, common scale"
    )
    figure.suptitle(
        "No normalization or quantization; only hidden-column matching after freezing"
    )
    save_figure(figure, output, "weight_comparison")
    np.savez_compressed(
        output / "weight_heatmaps.npz",
        weights=np.stack(weights),
        titles=np.array(titles),
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for variant, color in (("baseline", "#4879b8"), ("paired", "#65a58d")):
        folder = output / variant
        frozen = json.loads((folder / "frozen.json").read_text())
        fits = single.combined(folder / "confirmation")
        rows = np.flatnonzero(fits["entry_ids"] == frozen["primary"]["entry"])
        for key, style, name in (
            ("train_history", "-", "support"),
            ("query_history", "--", "query"),
        ):
            axes[0].plot(
                fits["history_steps"],
                fits[key][:, rows].mean(1),
                color=color,
                linestyle=style,
                label=variant + " " + name,
            )
        axes[1].plot(
            fits["history_steps"],
            100 * fits["recall_history"][:, rows].mean(1),
            color=color,
            label=variant,
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Mean balanced BCE over four inherited C")
    axes[1].set_ylabel("Mean query positive recall, %")
    for axis in axes:
        axis.set_xlabel("Numeric updates after freezing I")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.15)
    figure.suptitle("Continue inherited C; query chooses each numerical checkpoint")
    save_figure(figure, output, "inherited_curves")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout="constrained")
    for column, variant in enumerate(VARIANTS):
        folder = output / variant
        frozen = json.loads((folder / "frozen.json").read_text())
        origin = frozen["primary"]["origin"]
        history = json.loads((folder / origin["folder"] / "history.json").read_text())
        snapshots = load_archive(folder / origin["folder"] / "snapshots.npz")
        source = load_archive(folder / "source.npz")
        for key, style in (("train", "-"), ("query", "--")):
            axes[0, column].plot(
                [row["step"] for row in history],
                [row[key][origin["root"]][0] for row in history],
                linestyle=style,
                label="support" if key == "train" else key,
            )
        validation = []
        with torch.no_grad():
            for number, step in enumerate(snapshots["steps"]):
                temperature = next(row for row in history if row["step"] == step)[
                    "temperature"
                ][origin["root"]]
                p = torch.softmax(
                    torch.tensor(snapshots["alpha"][number, origin["root"]])
                    / temperature,
                    -1,
                )
                numeric = torch.tensor(
                    snapshots["numeric"][number, origin["root"]][None]
                )
                score = metrics(
                    soft_predict(torch.tensor(source["selection_x"]), p[None], numeric),
                    torch.tensor(source["selection_y"])[None, :, None],
                )
                validation.append(float(score[..., 0].mean()))
        axes[0, column].plot(
            snapshots["steps"], validation, marker="o", label="independent selection"
        )
        axes[0, column].axvline(
            origin["step"], color="grey", alpha=0.5, label="selected snapshot"
        )
        axes[0, column].set_title(variant + ": soft search of selected seed")
        axes[0, column].set_ylabel("Mean balanced BCE over four C")
        entries = json.loads((folder / "confirmation/hard_candidates.json").read_text())
        fits = single.combined(folder / "confirmation")
        for seed in summary[variant]["by_seed"]:
            search_seed = seed["origin"]["search_seed"]
            ids = [
                number
                for number, entry in enumerate(entries)
                if entry["id"] == seed["origin"]["id"]
                and entry["search_seed"] == search_seed
            ]
            steps = [entries[number]["step"] for number in ids]
            losses = [
                float(fits["selection_metrics"][fits["entry_ids"] == number, 0].mean())
                for number in ids
            ]
            axes[1, column].plot(
                steps, losses, marker="o", label=f"search seed {search_seed}"
            )
        axes[1, column].set_title(variant + ": hard I after continuing inherited C")
        axes[1, column].set_ylabel("Independent selection BCE, mean over four C")
    for axis in axes.ravel():
        axis.set_yscale("log")
        axis.set_xlabel("Joint P/C search updates")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.15)
    figure.suptitle(
        "Validation monitoring: source-only selection; test does not rank candidates"
    )
    save_figure(figure, output, "validation_curves")
    save_json(output / "selected_indexators.json", structures)
    fixed = json.loads((output / "fixed_I_control.json").read_text())
    rows, seed_rows, fresh_rows, individual_rows, projection_rows = [], [], [], [], []
    for variant, row in summary.items():
        diagnostic = structures[variant]
        rows.append(
            [
                variant,
                row["classes"],
                5,
                diagnostic["primary"]["errors"],
                f"{100 * diagnostic['primary']['mask_f1']:.2f}",
                f"{100 * row['primary_test'][1]:.2f}",
                f"{diagnostic['exact_seeds']}/3",
            ]
        )
        for seed, structure in zip(row["by_seed"], diagnostic["by_seed"], strict=True):
            seed_rows.append(
                [
                    variant,
                    seed["origin"]["search_seed"],
                    seed["origin"]["architecture_lr"],
                    seed["origin"]["sparsity_penalty"],
                    seed["origin"]["step"],
                    structure["errors"],
                    f"{100 * seed['test'][1]:.2f}",
                ]
            )
        for name, result in row["fresh"].items():
            fresh_rows.append(
                [
                    variant,
                    name,
                    f"{100 * result['mean_test_ba']:.2f}",
                    f"{result['perfect_test']}/32",
                    f"{100 * result['selected_test'][1]:.2f}",
                ]
            )
        individual_choice = json.loads(
            (output / variant / "individual.json").read_text()
        )
        individual_rows.append(
            [
                variant,
                individual_choice["phase"],
                individual_choice["origin"]["search_seed"],
                diagnostic["individual"]["errors"],
                f"{100 * diagnostic['individual']['mask_f1']:.2f}",
                f"{100 * row['individual']['test'][1]:.2f}",
                f"{100 * row['individual']['population'][1]:.2f}",
            ]
        )
        projection_rows.append(
            [
                variant,
                f"{100 * row['relaxation']['soft'][1]:.2f}",
                f"{100 * row['relaxation']['argmax'][1]:.2f}",
                f"{100 * row['primary_test'][1]:.2f}",
                f"{np.mean(diagnostic['projection']['relative_soft_to_argmax_weight_errors']):.3f}",
            ]
        )
    lines = [
        "# Pattern 0110: удвоение индексов через C и -C",
        "Сравниваются исходные [0,c1,c2] и ровно предложенные пользователем [0,c1,c2,-c1,-c2]. В обоих случаях два свободных числа кодбука и b/a/d, всего пять численных параметров. Четыре ненулевых значения второго варианта не независимы. Базовые c1/c2 свободно меняют знак; отдельный положительный модуль или clamp не вводится. В hard-артефактах отрицательные значения I означают ссылку на -C, а не физический знак веса.",
        "При поиске W=(P+1-P-1)c1+(P+2-P-2)c2. Это позволяет менять знак эффективного веса через P, без обязательной смены знака c. Во время выбора не передаются истинные окна, связи, численность групп или I. При фиксированном I неиспользуемые дополнительные классы сами по себе не меняют задачу обучения чисел.",
        f"Контроль фиксированного истинного I: та же физическая функция переписана через -c1,+c2, с соответствующим отражением начального c1. Из {fixed['starts']} стартов точных на selection {fixed['paired_perfect_selection']} против исходных {fixed['original_perfect_selection']}; max различие query BCE {fixed['max_query_bce_difference']:.2g}. Это контроль параметризации, а не поиск структуры.",
        "Совместное обучение P/C на support; query выбирает численные checkpoint. Сетка numeric LR.03, alpha .003/.01/.03 × sparsity0/.01, annealed softmax1→.05, entropy.02. Pilot seeds11/12/13, по18 запусков каждого варианта. Две настройки каждого варианта подтверждены на14/15/16. Четыре C,512 warmup,6000 joint steps. После argmax все четыре исходных C дообучаются4096 шагов. Mean независимой selection BCE выбирает checkpoint, затем конфигурацию по среднему трёх seed, затем основной seed. Численные и minibatch seed общие, начальные P различаются из-за разного количества классов.",
        table(
            [
                "Вариант",
                "Классов I",
                "Численных параметров",
                "Ошибок I/88",
                "F1 маски, %",
                "Warm test BA, %",
                "Точных I на новых seed",
            ],
            rows,
        ),
        "Способность найти хотя бы одно решение и устойчивость выбранной конфигурации — разные оценки. Ниже отдельный лучший индивидуальный кандидат по независимой selection среди всего одинакового бюджета pilot/confirmation. Он выбран и заморожен до test и не подменяет основной configuration-based выбор выше.",
        table(
            [
                "Вариант",
                "Этап",
                "Seed",
                "Ошибок I/88",
                "F1 маски, %",
                "Test BA, %",
                "BA всех 2048, %",
            ],
            individual_rows,
        ),
        "Для paired нет единственного именования истинного I: одинаковую функцию могут задать один противоположный ±код либо две независимые группы. Диагностика после freeze берёт ближайшее представление двух ролей паттерна, учитывает глобальные имена/знаки групп и перестановку нейронов. Отдельно считается бинарная маска. Эти преобразования не участвуют в выборе.",
        table(
            [
                "Вариант",
                "Search seed",
                "Alpha LR",
                "Sparsity",
                "Snapshot",
                "Ошибок I",
                "Warm test BA, %",
            ],
            seed_rows,
        ),
        "Разрыв soft→argmax оценивается с исходными C до дообучения, затем сравнивается с hard I после дообучения. Метрики soft/argmax усреднены по четырём текущим C, warm — выбранный по query кодбук. Относительная ошибка весов — норма разности soft- и hard-матриц, делённая на норму soft-матрицы, средняя по тем же четырём C.",
        table(
            [
                "Вариант",
                "Soft test BA, %",
                "Argmax test BA, %",
                "Warm hard test BA, %",
                "Относительный разрыв W",
            ],
            projection_rows,
        ),
        "У paired после дообучения значения C1/C2 могут быть почти противоположны, так что C1 и -C2, а также C2 и -C1 образуют близкие значения. Вероятностная смесь тогда имеет несколько почти эквивалентных категорий. Это возможная причина усложнения дискретизации и разделения одинаковых ролей между именами групп; причинная абляция этой гипотезы не проводилась. Фактические значения и ошибка проекции сохранены в selected_indexators.json.",
        "В выбранном paired-I обнаружен конкретный механизм лишней связи: input11/hidden3 в исходном порядке имеет P(C1)=49.79%, P(-C1)=50.12%, P(0)=0.0103%. При C1 около -1.70 мягкий вес около +0.006, а argmax даёт +1.70. Эта позиция отмечена красным на P-хитмапах. Средняя максимальная вероятность по всей матрице 99.42% скрывает одну такую неоднозначную позицию; относительный скачок всей W составляет 17.6%. Наблюдение объясняет эту лишнюю связь, но само по себе не доказывает причину результатов всех поисковых запусков. При четырёх signed-метках разные имена также могут задавать почти одинаковые физические веса, поэтому строгие ошибки группировки нужно читать вместе с F1 связей и качеством функции.",
        "![Истинные и выбранные индексаторы](figures/selected_indexators.png)",
        "![Лучшие индивидуальные индексаторы по validation](figures/best_individual_indexators.png)",
        "![P исходного варианта](figures/baseline_probabilities.png)",
        "![P знаковых пар](figures/paired_probabilities.png)",
        "![Фактические веса и Adam](figures/weight_comparison.png)",
        "![Реальные кривые дообучения исходных C](figures/inherited_curves.png)",
        "![Контроль validation и выбор snapshot](figures/validation_curves.png)",
        "Свежие C проверяются отдельно:32 общих нормированных Gaussian-старта без ограничения знаков, ранее подобранные lr.03/norm.5. warmup0 — прежний fit; warmup64 — выбор C/-C по query, суммарный бюджет4096 учитывает обе ветки. Oracle использует те же начальные функции в обоих вариантах. Выбор лучшего старта по query корректен; среднее и доля успешных описывают бюджет перезапусков.",
        table(
            [
                "Вариант",
                "Fresh I / процедура",
                "Mean test BA, %",
                "Точных test",
                "Query-selected test BA, %",
            ],
            fresh_rows,
        ),
        "Отдельный лучший индивидуальный source-кандидат заморожен до test; он не заменяет основной выбор по конфигурации. Его I и метрики сохранены в individual.npz/.json и summary.json. Это исследовательский опыт на уже известном паттерне, новые seed проверяют инициализацию, не новый независимый benchmark. Программные тесты не создавались и не запускались.",
    ]
    (output / "REPORT.md").write_text("\n\n".join(lines) + "\n")
    snapshot = output / "analysis_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in ("signed_codebook.py", "search.py", "model.py", "single.py", "data.py"):
        shutil.copy2(BASE / name, snapshot / name)
    save_json(
        output / "complete.json",
        {
            "completed_utc": datetime.now(UTC).isoformat(),
            "structures": structures,
            "software_tests_run": False,
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
                "Variant",
                "Classes",
                "Numeric parameters",
                "I errors",
                "Mask F1",
                "Warm test BA",
                "Exact new seeds",
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
            "oracle",
            "pilot",
            "confirmation",
            "freeze",
            "evaluate",
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
    device = torch.device(args.device)
    if args.stage == "prepare":
        prepare(args.output)
    elif args.stage == "oracle":
        oracle_control(args.output, device)
    elif args.stage in ("pilot", "confirmation"):
        for variant in [args.variant] if args.variant else VARIANTS:
            search_phase(
                args.output / "signed_codebook_20261104" / variant, args.stage, device
            )
    elif args.stage == "freeze":
        freeze(args.output / "signed_codebook_20261104", VARIANTS)
    elif args.stage == "evaluate":
        evaluate(
            args.output, args.output / "signed_codebook_20261104", VARIANTS, device
        )
    else:
        report(args.output)


if __name__ == "__main__":
    main()
