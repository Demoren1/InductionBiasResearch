"""Post-selection diagnostics from saved candidates; no new MLP training."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from ..io import DEFAULT_BANK, checked_load, save_json
from ..masks import structural_metrics
from ..problem import reference


def audit(root, bank=DEFAULT_BANK):
    torch.set_num_threads(1)
    manifest = json.loads((bank / "manifest.json").read_text())
    controls = checked_load(root / "training_controls.pt")
    summary = json.loads((root / "summary.json").read_text())
    structures = {}
    for family in summary["families"]:
        metrics, _ = structural_metrics(
            controls["bank_masks"],
            reference(manifest["family_definitions"][family], "cpu"),
        )
        structures[family] = {
            "iou": np.array([row["iou"] for row in metrics]),
            "exact": np.array([row["exact_mask"] for row in metrics]),
        }
    rows = []
    for complete in sorted(root.glob("wg_*/family_*/COMPLETE.json")):
        directory = complete.parent
        protocol = json.loads((directory / "protocol.json").read_text())
        family = protocol["family"]
        selection = checked_load(directory / "selection.pt")
        bank = checked_load(directory / "training_bank_selection.pt")
        scores = bank["scores"].numpy()
        exact = structures[family]["exact"]
        diagnostic = json.loads((directory / "diagnostics.json").read_text())
        best_exact = float(scores[exact].min()) if exact.any() else None
        correlation = (
            float(spearmanr(-scores, structures[family]["iou"]).statistic)
            if np.ptp(scores) > 0 and np.ptp(structures[family]["iou"]) > 0
            else None
        )
        row = {
            "family": family,
            "decoder_seed": protocol["decoder_seed"],
            "bank_exact_candidates": int(exact.sum()),
            "bank_selected_exact": bool(exact[bank["selected"]]),
            "bank_best_exact_rank": int((scores < best_exact).sum() + 1)
            if best_exact is not None
            else None,
            "bank_proxy_iou_spearman": correlation,
            "bank_selected_origin_family": controls["origin_families"][
                bank["selected"]
            ],
            "bank_oracle_proxy_margin": float(scores.min())
            - diagnostic["oracle_short_query_bce"],
            "gumbel_oracle_proxy_margin": selection["adaptation_query_bce"][
                "gumbel"
            ]
            - diagnostic["oracle_short_query_bce"],
            **diagnostic,
        }
        for method in ["gumbel", "evolution"]:
            history = json.loads(
                (directory / f"{method}_history.json").read_text()
            )
            row[method + "_proxy_improvement"] = (
                history[0]["best_bce"] - history[-1]["best_bce"]
            )
        rows.append(row)
    available = [row for row in rows if row["bank_exact_candidates"]]
    correlations = [
        row["bank_proxy_iou_spearman"]
        for row in rows
        if row["bank_proxy_iou_spearman"] is not None
    ]
    payload = {
        "jobs": len(rows),
        "rows": rows,
        "families_with_exact_bank_candidates": len(
            {row["family"] for row in available}
        ),
        "jobs_with_exact_bank_candidates": len(available),
        "bank_exact_selections": sum(
            row["bank_selected_exact"] for row in rows
        ),
        "mean_best_exact_bank_rank": float(
            np.mean([row["bank_best_exact_rank"] for row in available])
        )
        if available
        else None,
        "mean_bank_proxy_iou_spearman": float(np.mean(correlations))
        if correlations
        else None,
        "jobs_where_oracle_short_fitness_loses_to_bank": sum(
            row["bank_oracle_proxy_margin"] < 0 for row in rows
        ),
        "random_exact_candidates_total": sum(
            row["random_pool_exact_candidates"] for row in rows
        ),
        "mean_random_max_iou": float(
            np.mean([row["random_pool_max_iou"] for row in rows])
        ),
        "scope": "Saved-candidate diagnostics after selection; no model fitting, no changes to selected masks.",
    }
    save_json(root / "audit.json", payload)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for method, color, label in [
        ("gumbel", "#2166ac", "Gumbel"),
        ("evolution", "#cc7722", "Эволюция"),
        ("training_bank", "#529780", "Выбор из банка"),
    ]:
        values = summary["summary"][method]["iou"]["family_values"]
        axes[0].plot(
            summary["families"], values, marker=".", color=color, label=label
        )
    maximums = [
        np.mean(
            [
                row["random_pool_max_iou"]
                for row in rows
                if row["family"] == family
            ]
        )
        for family in summary["families"]
    ]
    axes[0].plot(
        summary["families"],
        maximums,
        linestyle="--",
        color="#777777",
        label="Максимум в случайном пуле (диагностика)",
    )
    axes[0].set(
        xlabel="Семейство",
        ylabel="IoU полной маски",
        title="Доступные и выбранные маски",
        ylim=(0, 1.03),
    )
    axes[0].legend(fontsize=8)
    for row in available:
        axes[1].scatter(
            row["family"],
            row["bank_best_exact_rank"],
            color="#529780",
            alpha=0.45,
        )
    axes[1].set(
        xlabel="Семейство",
        ylabel="Место лучшей точной маски по короткой BCE",
        title="Может ли критерий выбрать известный ответ?",
    )
    axes[1].axhline(
        1,
        color="#777777",
        linestyle="--",
        label="Выбирается только первое место",
    )
    axes[1].legend(fontsize=8)
    fig.savefig(root / "figures/diagnostics.png", dpi=170)
    plt.close(fig)
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "rows"},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    args = parser.parse_args()
    audit(args.root, args.bank)
