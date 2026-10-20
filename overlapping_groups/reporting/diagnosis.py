"""Audit inner-fit undertraining and latent optimization from saved runs."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from ..io import checked_load, save_json


def diagnose(root):
    torch.set_num_threads(1)
    directories = sorted(
        p.parent for p in root.glob("wg_*/family_*/COMPLETE.json")
    )
    config = json.loads((directories[0] / "protocol.json").read_text())[
        "config"
    ]
    inner_label = f"{config['search']['inner_steps']} {config['search'].get('inner_optimizer', 'sgd').upper()}"
    selections = [checked_load(p / "selection.pt") for p in directories]
    diagnostics = [
        json.loads((p / "diagnostics.json").read_text()) for p in directories
    ]
    entropy = np.array([s["adaptation_query_entropy"] for s in selections])
    oracle = np.array([d["oracle_short_query_bce"] for d in diagnostics])
    methods = [
        "best_initial",
        "gumbel",
        "evolution",
        "random_budget",
        "training_bank",
    ]
    scores = {
        method: np.array(
            [s["adaptation_query_bce"][method] for s in selections]
        )
        for method in methods
    }
    payload = {
        "jobs": len(directories),
        "adaptation_query_entropy": float(entropy.mean()),
        "oracle_short_query_bce": float(oracle.mean()),
        "oracle_short_excess_bce": float((oracle - entropy).mean()),
        "oracle_fraction_constant_predictor_gap_closed": float(
            np.mean((np.log(2) - oracle) / (np.log(2) - entropy))
        ),
        "proxy_scores": {
            method: float(values.mean()) for method, values in scores.items()
        },
        "random_beats_gumbel_on_proxy": int(
            (scores["random_budget"] < scores["gumbel"]).sum()
        ),
        "random_beats_evolution_on_proxy": int(
            (scores["random_budget"] < scores["evolution"]).sum()
        ),
        "oracle_beats_gumbel_on_proxy": int((oracle < scores["gumbel"]).sum()),
        "oracle_beats_evolution_on_proxy": int(
            (oracle < scores["evolution"]).sum()
        ),
        "method_pair_total": len(directories),
    }
    histories = {
        method: [
            json.loads((p / f"{method}_history.json").read_text())
            for p in directories
        ]
        for method in ["gumbel", "evolution"]
    }
    for method, values in histories.items():
        payload[method + "_acceptance_rate"] = float(
            np.mean(
                [
                    sum(row["accepted_restarts"] for row in history[1:])
                    / (40 * 8)
                    for history in values
                ]
            )
        )
    gradients = [
        row["gradient_norm"]
        for history in histories["gumbel"]
        for row in history[1:]
    ]
    payload["gumbel_gradient_norm_median"] = float(np.median(gradients))
    payload["gumbel_gradient_norm_range"] = [
        float(min(gradients)),
        float(max(gradients)),
    ]
    curve_methods = [
        "correct_mask",
        "dense",
        "gumbel",
        "evolution",
        "random_budget",
        "training_bank",
    ]
    curves = {method: [] for method in curve_methods}
    for directory in directories:
        protocol = json.loads((directory / "protocol.json").read_text())
        model = checked_load(directory / "evaluation_models.pt")
        query_entropy = np.array(
            [
                float(F.binary_cross_entropy(y, y))
                for y in [
                    checked_load(
                        root
                        / "observations"
                        / f"family_{protocol['family']:02d}_function_{function}"
                        / "observations.pt"
                    )["query"]["y"]
                    for function in [6, 7]
                ]
            ]
        )
        history = json.loads(
            (directory / "evaluation_history.json").read_text()
        )
        for method in curve_methods:
            index = model["methods"].index(method)
            curve = [
                float(
                    (
                        np.array(row["query_bce"])[
                            :, index * 3 : (index + 1) * 3
                        ]
                        - query_entropy[:, None]
                    ).mean()
                )
                for row in history
            ]
            curves[method].append(curve)
    steps = [row["step"] for row in history]
    payload["evaluation_query_excess_curves"] = {
        method: np.mean(values, axis=0).tolist()
        for method, values in curves.items()
    }
    payload["evaluation_steps"] = steps
    payload["scope"] = (
        "Read-only analysis. Inner fitness uses adaptation functions 4/5; full Adam histories use evaluation functions 6/7. Optimizer, seeds and functions differ, so this is not a controlled optimizer/step ablation."
    )
    save_json(root / "selection_diagnosis.json", payload)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    colors = {
        "gumbel": "#2166ac",
        "evolution": "#d87820",
        "random_budget": "#8562a8",
        "training_bank": "#529780",
        "correct_mask": "#333333",
        "dense": "#bc5656",
    }
    names = {
        "gumbel": "Gumbel",
        "evolution": "Эволюция",
        "random_budget": "Случайный поиск",
        "training_bank": "Выбор из банка",
        "correct_mask": "Правильная маска",
        "dense": "MLP без маски",
    }
    for method in ["gumbel", "evolution"]:
        values = np.array(
            [
                [row["best_bce"] for row in history]
                for history in histories[method]
            ]
        )
        axes[0].plot(
            np.arange(41),
            (values - entropy[:, None]).mean(0),
            label=names[method],
            color=colors[method],
        )
    axes[0].axhline(
        float((oracle - entropy).mean()),
        color=colors["correct_mask"],
        linestyle="--",
        label=f"Точная маска: {inner_label}",
    )
    axes[0].axhline(
        float((scores["random_budget"] - entropy).mean()),
        color=colors["random_budget"],
        linestyle=":",
        label=f"Случайные z: {inner_label}",
    )
    axes[0].set(
        xlabel="Шаг поиска z",
        ylabel="Adaptation query BCE − энтропия",
        title="Что оптимизирует поиск: короткое обучение",
    )
    axes[0].legend(fontsize=8)
    for method in curve_methods:
        values = np.maximum(
            payload["evaluation_query_excess_curves"][method], 1e-8
        )
        axes[1].semilogy(
            steps, values, color=colors[method], label=names[method]
        )
    axes[1].set(
        xlabel="Шаг Adam",
        ylabel="Evaluation query BCE − энтропия",
        title="После фиксации маски: полное обучение",
    )
    axes[1].legend(fontsize=8)
    fig.suptitle(
        "Различие критериев: 64 SGD на функциях 4/5 и 2000 Adam на 6/7\nСреднее по 24 семействам и трём декодерам; данные из сохранённых запусков"
    )
    fig.savefig(root / "figures/selection_diagnosis.png", dpi=170)
    plt.close(fig)
    print(
        json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key != "evaluation_query_excess_curves"
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    diagnose(parser.parse_args().root)
