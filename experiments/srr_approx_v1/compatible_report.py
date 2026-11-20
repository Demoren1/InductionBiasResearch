"""Collect family recovery, paired transfer effects and the deployed indexators."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from experiments.sharing_structure_research_v1.blind_contract import save_json
from experiments.srr_approx_v1.compatible import OUTPUT
from experiments.srr_approx_v1.compatible_replication import name
from experiments.srr_approx_v1.compatible_synthetic import CASES
from experiments.srr_approx_v1.group_transfer import read_json


STRATEGIES = ("witness_cover", "round_robin", "uniform48", "local", "exhaustive")
SETTINGS = ("mnist16", "omniglot10")


def collect():
    scalar = {}
    for case in CASES:
        files = sorted((OUTPUT / "scalar_confirmation").glob(f"*_{case}.json"))
        records = [read_json(path) for path in files]
        scalar[case] = {
            "families": len(records),
            "recovered": {
                strategy: sum(r[strategy]["population_family_exact"] for r in records)
                for strategy in STRATEGIES
            },
            "all_witnesses_retained": sum(
                r["witness_cover"]["full_true_frontier_covered"] for r in records
            ),
        }
    folder = OUTPUT / "completion"
    with np.load(OUTPUT / "pattern" / "frozen.npz") as f:
        frozen = {"pattern": f["fixed"]}
    manifest, effects = {}, {}
    for setting in SETTINGS:
        plan = read_json(folder / f"{setting}_plan.json")
        metrics = read_json(folder / f"{setting}_results.json")
        first = read_json(OUTPUT / f"{setting}_results.json")
        repeat = read_json(OUTPUT / "replication" / f"{setting}_results.json")
        policy = metrics["fixed"]
        manifest[setting] = []
        with np.load(folder / f"{setting}_deployment.npz") as bank:
            for j, seed in enumerate(plan["deployment_seeds"], 1):
                key = name(j, policy)
                output_name = f"{setting}_{seed}"
                frozen[output_name] = bank[key]
                manifest[setting].append(
                    {"index": output_name, "seed": seed, "policy": policy}
                )
        controls = ["0000", "coverage", "neutral", "erm"]
        if setting == "omniglot10":
            controls.extend(("wsr_step_75", "four_uniform_mean"))
        effects[setting] = {
            "episodes": metrics["episodes"],
            "method": policy,
            "unrestricted_erm": metrics["erm"],
            "accuracy": {
                k: metrics["accuracy_ce_brier"][k][0] for k in ["fixed", *controls]
            },
            "paired_accuracy_intervals": {
                k: metrics["minus_baseline_ci975"]
                if k == "0000"
                else metrics["paired_ci95"][k]
                for k in controls
            },
            "pilot_768_vs_asrr_ci95": first["paired_acc_difference_ci95"]["0000"],
            "one_stage_replication_1536_vs_asrr_ci975": repeat[
                "new_minus_baseline_ci975"
            ],
            "per_mask_vs_asrr_ci95": metrics["per_seed_minus_baseline"],
        }
    np.savez_compressed(OUTPUT / "fixed_indices.npz", **frozen)
    return {
        "scalar_confirmation": scalar,
        "pattern": read_json(OUTPUT / "pattern" / "results.json"),
        "fixed_indices": manifest,
        "transfer": effects,
        "intervals": "episode-paired mean, low, high in accuracy units, conditional on the fixed data pools and three mask seeds; 97.5% vs ordinary aSRR and 95% for secondary controls",
    }


def plot_effects(effects):
    labels = {
        "0000": "Исходный aSRR",
        "neutral": "Те же блоки,\nравномерные исключения",
        "coverage": "Выбор по доле\nпокрытых задач",
        "wsr_step_75": "WSR",
        "four_uniform_mean": "Четыре случайные маски",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.1, 3.35), layout="constrained")
    for ax, setting, title in zip(axes, SETTINGS, ("MNIST", "Omniglot"), strict=True):
        intervals = effects[setting]["paired_accuracy_intervals"]
        controls = [k for k in labels if k in intervals]
        for i, k in enumerate(controls):
            mean, low, high = 100 * np.asarray(intervals[k])
            ax.errorbar(
                mean,
                i,
                xerr=[[mean - low], [high - mean]],
                fmt="o",
                markersize=5.5,
                capsize=3,
                linewidth=1.6,
                color="#185a74" if k == "0000" else "#686f75",
            )
        ax.axvline(0, color="#9099a2", linewidth=0.9, linestyle="--", zorder=0)
        ax.set_yticks(range(len(controls)), [labels[k] for k in controls], fontsize=9)
        ax.set_ylim(len(labels) - 0.5, -0.5)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Разность точности, п.п.")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(axis="x", alpha=0.16)
        ax.tick_params(axis="y", length=0, pad=6)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#9aa1a8")
    fig.savefig(OUTPUT / "paired_transfer.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    results = collect()
    save_json(OUTPUT / "summary.json", results)
    plot_effects(results["transfer"])
