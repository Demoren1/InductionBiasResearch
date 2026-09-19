"""Summarize the matched global-z ablation and draw paired test results."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, median, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "pattern/outputs/bilevel_mask/generated_sharing_z_ablation_20260919"
REFERENCE = ROOT / "pattern/outputs/bilevel_mask/generated_sharing_balanced_final_20260915"
SUMMARY = ROOT / "mds/data/2026-09-19/generated_sharing_z_ablation.json"
FIGURE = ROOT / "mds/assets/2026-09-19/generated_sharing_z_ablation.png"
COMPUTE_SUMMARY = ROOT / "mds/data/2026-09-19/generated_sharing_z_compute_matched.json"
COMPUTE_FIGURE = ROOT / "mds/assets/2026-09-19/generated_sharing_z_compute_matched.png"
COORD_SUMMARY = ROOT / "mds/data/2026-09-19/generated_sharing_coordinate_only.json"
COORD_FIGURE = ROOT / "mds/assets/2026-09-19/generated_sharing_coordinate_only.png"
PRIMARY_FIGURE = ROOT / "mds/assets/2026-09-19/generated_sharing_learned_vs_one_fixed_z.png"
COMPACT_SUMMARY = ROOT / "mds/data/2026-09-19/z_importance_summary.json"
POLICIES = ("learned", "frozen_bank", "single_fixed")
SEEDS = tuple(range(42, 50))


def stats(values: list[float]) -> dict:
    return {"mean": mean(values), "sd": stdev(values), "median": median(values),
            "min": min(values), "max": max(values)}


def two_sided_sign_p(wins: int, trials: int) -> float:
    smaller = min(wins, trials - wins)
    return min(1.0, 2 * sum(math.comb(trials, index) for index in range(smaller + 1)) / 2**trials)


def summarize() -> dict:
    rows = []
    for seed in SEEDS:
        results = {policy: json.loads((RUNS / f"{policy}_seed{seed}/summary.json").read_text())
                   for policy in POLICIES}
        configs = [result["config"] for result in results.values()]
        assert all(config == configs[0] for config in configs)
        assert all(result["fold_global_z"]["hard_assignment_mismatched_entries"] == 0
                   for result in results.values())
        assert all(result["fold_global_z"]["original_generator_parameters"] == 469
                   for result in results.values())
        assert all(result["fold_global_z"]["folded_coordinate_generator_parameters"] == 405
                   for result in results.values())
        assert len({result["reference_checkpoint_sha256"] for result in results.values()}) == 1
        reference = json.loads((REFERENCE / f"seed{seed}/summary.json").read_text())
        reference_bce = reference["evaluations"]["test"]["strategies"]["generated_sharing"]["mean_query_bce"]
        row = {"seed": seed, "reference_published_test_bce": reference_bce,
               "policies": {policy: {
                   "test_bce": result["test"]["mean_query_bce"],
                   "test_accuracy": result["test"]["mean_query_accuracy"],
                   "test_active_iou": result["test"]["mean_active_iou"],
                   "validation_bce": result["validation"]["mean_query_bce"],
                   "selected_outer_step": result["training"]["selected_outer_step"],
                   "training_seconds": result["training"].get("training_seconds"),
               } for policy, result in results.items()}}
        row["delta_test_bce_vs_learned"] = {
            control: row["policies"][control]["test_bce"] - row["policies"]["learned"]["test_bce"]
            for control in POLICIES[1:]}
        rows.append(row)

    aggregate = {}
    for policy in POLICIES:
        aggregate[policy] = {metric: stats([row["policies"][policy][metric] for row in rows])
                             for metric in ("test_bce", "test_accuracy", "test_active_iou", "validation_bce")}
    differences = {}
    for policy in POLICIES[1:]:
        values = [row["delta_test_bce_vs_learned"][policy] for row in rows]
        wins = sum(value > 0 for value in values)
        n = len(values)
        differences[policy] = {**stats(values), "learned_wins": wins,
                               "seeds": n, "two_sided_sign_p": two_sided_sign_p(wins, n)}
    return {"seeds": list(SEEDS), "policies": list(POLICIES), "per_seed": rows,
            "aggregate": aggregate, "paired_differences_bce_control_minus_learned": differences,
            "reference_reproduction_max_abs_test_bce": max(
                abs(row["policies"]["learned"]["test_bce"] - row["reference_published_test_bce"])
                for row in rows),
            "fold_global_z": {"full_parameters": 469, "folded_parameters": 405,
                              "hard_assignment_mismatch_all_runs": 0}}


def plot(summary: dict, figure: Path = FIGURE, title_suffix: str = "") -> None:
    seeds = summary["seeds"]
    rows = summary["per_seed"]
    colors = {"learned": "#2466A8", "frozen_bank": "#D58023", "single_fixed": "#6F4B9A"}
    labels = {"learned": "optimized z", "frozen_bank": "16 fixed z candidates",
              "single_fixed": "one fixed z"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.1), layout="constrained")
    ax = axes[0]
    for policy in POLICIES:
        values = [row["policies"][policy]["test_bce"] for row in rows]
        ax.plot(seeds, values, "o-", color=colors[policy], lw=2, ms=6, label=labels[policy])
    ax.set_title("Held-out examples of the same 16 patterns" + title_suffix)
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Test BCE (lower is better)")
    ax.set_xticks(seeds)
    ax.grid(alpha=.2)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    width = .34
    positions = list(range(len(seeds)))
    for shift, policy in ((-width/2, "frozen_bank"), (width/2, "single_fixed")):
        values = [row["delta_test_bce_vs_learned"][policy] for row in rows]
        ax.bar([position + shift for position in positions], values, width,
               color=colors[policy], label=labels[policy])
    ax.axhline(0, color="#303030", lw=1)
    ax.set_xticks(positions, seeds)
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Control BCE − optimized-z BCE")
    ax.set_title("Positive bars favor optimizing z")
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False, fontsize=9)
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=180)
    plt.close(fig)


def summarize_compute_matched(base: dict) -> dict:
    rows = []
    for original in base["per_seed"]:
        seed = original["seed"]
        row = {"seed": seed, "policies": {"learned": original["policies"]["learned"]}}
        for policy in POLICIES[1:]:
            result = json.loads((RUNS / f"{policy}_150_seed{seed}/summary.json").read_text())
            assert result["config"]["outer_steps"] == 150
            assert result["fold_global_z"]["hard_assignment_mismatched_entries"] == 0
            assert result["fold_global_z"]["folded_coordinate_generator_parameters"] == 405
            row["policies"][policy] = {
                "test_bce": result["test"]["mean_query_bce"],
                "test_accuracy": result["test"]["mean_query_accuracy"],
                "test_active_iou": result["test"]["mean_active_iou"],
                "validation_bce": result["validation"]["mean_query_bce"],
                "training_seconds": result["training"]["training_seconds"],
                "selected_outer_step": result["training"]["selected_outer_step"],
            }
        row["delta_test_bce_vs_learned"] = {
            control: row["policies"][control]["test_bce"] - row["policies"]["learned"]["test_bce"]
            for control in POLICIES[1:]}
        rows.append(row)
    aggregate = {policy: {metric: stats([row["policies"][policy][metric] for row in rows])
                          for metric in ("test_bce", "test_accuracy", "test_active_iou",
                                         "training_seconds")}
                 for policy in POLICIES}
    differences = {}
    for policy in POLICIES[1:]:
        values = [row["delta_test_bce_vs_learned"][policy] for row in rows]
        wins = sum(value > 0 for value in values)
        n = len(values)
        differences[policy] = {**stats(values), "learned_wins": wins,
                               "seeds": n, "two_sided_sign_p": two_sided_sign_p(wins, n)}
    return {"protocol": "learned z: 100 outer steps; fixed-z controls: 150 outer steps",
            "seeds": list(SEEDS), "policies": list(POLICIES), "per_seed": rows,
            "aggregate": aggregate, "paired_differences_bce_control_minus_learned": differences}


def summarize_coordinate(base: dict, compute: dict) -> dict:
    rows = []
    for original, equal_time in zip(base["per_seed"], compute["per_seed"]):
        seed = original["seed"]
        row = {"seed": seed,
               "learned_100": original["policies"]["learned"],
               "frozen_bank_150": equal_time["policies"]["frozen_bank"]}
        for steps in (100, 150):
            result = json.loads((RUNS / f"coordinate_only_{steps}_seed{seed}/summary.json").read_text())
            assert result["config"]["outer_steps"] == steps
            assert result["fold_global_z"]["original_generator_parameters"] == 405
            assert result["fold_global_z"]["hard_assignment_mismatched_entries"] == 0
            row[f"coordinate_{steps}"] = {
                "test_bce": result["test"]["mean_query_bce"],
                "test_accuracy": result["test"]["mean_query_accuracy"],
                "test_active_iou": result["test"]["mean_active_iou"],
                "training_seconds": result["training"]["training_seconds"],
                "selected_outer_step": result["training"]["selected_outer_step"],
            }
        rows.append(row)
    keys = ("learned_100", "coordinate_100", "coordinate_150", "frozen_bank_150")
    aggregate = {key: {metric: stats([row[key][metric] for row in rows])
                       for metric in ("test_bce", "test_accuracy", "test_active_iou",
                                      "training_seconds")}
                 for key in keys}
    differences = {}
    for key in keys[1:]:
        values = [row[key]["test_bce"] - row["learned_100"]["test_bce"] for row in rows]
        wins = sum(value > 0 for value in values)
        differences[key] = {**stats(values), "learned_wins": wins,
                            "seeds": len(values), "two_sided_sign_p": two_sided_sign_p(wins, len(values))}
    return {"seeds": list(SEEDS), "per_seed": rows, "aggregate": aggregate,
            "paired_differences_bce_control_minus_learned": differences}


def plot_coordinate(summary: dict) -> None:
    seeds = summary["seeds"]
    rows = summary["per_seed"]
    colors = {"learned_100": "#2466A8", "coordinate_100": "#7E4B9B",
              "coordinate_150": "#2D9182", "frozen_bank_150": "#D58023"}
    labels = {"learned_100": "optimized z, 100 steps",
              "coordinate_100": "no z, 100 steps",
              "coordinate_150": "no z, 150 steps",
              "frozen_bank_150": "16 fixed z, 150 steps"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.1), layout="constrained")
    for key in colors:
        axes[0].plot(seeds, [row[key]["test_bce"] for row in rows], "o-",
                     color=colors[key], lw=1.8, ms=5, label=labels[key])
    axes[0].set_title("Explicit 405-parameter coordinate generator")
    axes[0].set_xlabel("Training seed")
    axes[0].set_ylabel("Test BCE (lower is better)")
    axes[0].set_xticks(seeds)
    axes[0].grid(alpha=.2)
    axes[0].legend(frameon=False, fontsize=8)

    width = .35
    positions = list(range(len(seeds)))
    for shift, key in ((-width/2, "coordinate_100"), (width/2, "coordinate_150")):
        values = [row[key]["test_bce"] - row["learned_100"]["test_bce"] for row in rows]
        axes[1].bar([position + shift for position in positions], values, width,
                    color=colors[key], label=labels[key])
    axes[1].axhline(0, color="#303030", lw=1)
    axes[1].set_xticks(positions, seeds)
    axes[1].set_title("Positive bars favor optimizing z")
    axes[1].set_xlabel("Training seed")
    axes[1].set_ylabel("No-z BCE − optimized-z BCE")
    axes[1].grid(axis="y", alpha=.2)
    axes[1].legend(frameon=False, fontsize=9)
    fig.savefig(COORD_FIGURE, dpi=180)
    plt.close(fig)


def plot_primary(base: dict, compute: dict) -> None:
    seeds = base["seeds"]
    learned = [row["policies"]["learned"]["test_bce"] for row in base["per_seed"]]
    fixed_100 = [row["policies"]["single_fixed"]["test_bce"] for row in base["per_seed"]]
    fixed_150 = [row["policies"]["single_fixed"]["test_bce"] for row in compute["per_seed"]]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), layout="constrained")
    ax = axes[0]
    for values, color, label in (
        (learned, "#2466A8", "learned z · 100 steps"),
        (fixed_100, "#794899", "one fixed z · 100 steps"),
        (fixed_150, "#2D9182", "one fixed z · 150 steps"),
    ):
        ax.plot(seeds, values, "o-", color=color, lw=2, ms=6, label=label)
    ax.set_title("Test quality: learned z vs one fixed z")
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Test BCE (lower is better)")
    ax.set_xticks(seeds)
    ax.grid(alpha=.2)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    positions = list(range(len(seeds)))
    width = .35
    for values, shift, color, label in (
        (fixed_100, -width/2, "#794899", "same 100 steps"),
        (fixed_150, width/2, "#2D9182", "similar training time"),
    ):
        ax.bar([position + shift for position in positions],
               [fixed - trained for fixed, trained in zip(values, learned)],
               width, color=color, label=label)
    ax.axhline(0, color="#303030", lw=1)
    ax.set_xticks(positions, seeds)
    ax.set_title("Benefit from learning z, seed by seed")
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Fixed-z BCE − learned-z BCE")
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(PRIMARY_FIGURE, dpi=180)
    plt.close(fig)


def compact_results(base: dict, compute: dict, coordinate: dict) -> dict:
    rows = []
    for short, matched, no_z in zip(base["per_seed"], compute["per_seed"], coordinate["per_seed"]):
        assert short["seed"] == matched["seed"] == no_z["seed"]
        rows.append({
            "seed": short["seed"],
            "learned_z_100": short["policies"]["learned"],
            "one_fixed_z_100": short["policies"]["single_fixed"],
            "one_fixed_z_150": matched["policies"]["single_fixed"],
            "sixteen_fixed_z_150": matched["policies"]["frozen_bank"],
            "no_z_150": no_z["coordinate_150"],
        })
    variants = ("learned_z_100", "one_fixed_z_100", "one_fixed_z_150",
                "sixteen_fixed_z_150", "no_z_150")
    aggregates = {variant: {
        metric: stats([row[variant][metric] for row in rows])
        for metric in ("test_bce", "test_accuracy", "test_active_iou", "training_seconds")
    } for variant in variants}
    return {
        "setting": {"seeds": list(SEEDS), "patterns": 16, "sequence_length": 32,
                    "pattern_length": 4, "test": "new examples of the same pattern identities"},
        "aggregate": aggregates,
        "per_seed": rows,
        "reproduced_published_bce_max_abs_difference": base["reference_reproduction_max_abs_test_bce"],
        "global_z_fold": base["fold_global_z"],
    }


if __name__ == "__main__":
    summary = summarize()
    compute_summary = summarize_compute_matched(summary)
    coordinate_summary = summarize_coordinate(summary, compute_summary)
    compact = compact_results(summary, compute_summary, coordinate_summary)
    COMPACT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    COMPACT_SUMMARY.write_text(json.dumps(compact, indent=2) + "\n")
    plot_primary(summary, compute_summary)
    plot(compute_summary, COMPUTE_FIGURE, "; similar training time")
    print(json.dumps({variant: compact["aggregate"][variant]["test_bce"]
                      for variant in compact["aggregate"]}, indent=2))
