"""Plot ordinary DigitSum transfer for generated, Kronecker, and fixed U."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from .run import TEST_LENGTHS


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = ROOT / "deepsets_z" / "mnist8m" / "outputs" / "v_moe_digit_sum_finetune_raw"
DEFAULT_BINARY_RUNS = (ROOT / "deepsets_z" / "mnist8m" / "outputs" /
                       "v_moe_digit_sum_binary_u")
DEFAULT_DENSE = ROOT / "deepsets_z" / "mnist8m" / "results_authors_scaled" / "paper_mlp_seed42.json"
DEFAULT_FIGURE = ROOT / "mds" / "assets" / "2026-09-19" / "mnist8m_middle_unfreeze.png"
DEFAULT_SUMMARY = ROOT / "deepsets_z" / "mnist8m" / "digit_sum_u_control_summary.json"


RUNS = {
    "generated_frozen": ("conv_router_v_head_seed42.json",
                         "Generated U; middle frozen", "#b55f43", "--"),
    "generated_unfrozen": ("conv_router_v_head_seed42_unfrozen_middle.json",
                           "Generated U; only U frozen", "#08789b", "-"),
    "kronecker_frozen": ("conv_router_v_head_seed42_u_kronecker.json",
                         "Kronecker U (rank 32); middle frozen", "#7a5195", "--"),
    "kronecker_unfrozen": (
        "conv_router_v_head_seed42_u_kronecker_unfrozen_middle.json",
        "Kronecker U (rank 32); only U frozen", "#3c8c56", "-"),
    "random_unfrozen": ("conv_router_v_head_seed42_u_random_unfrozen_middle.json",
                        "Fixed random U; only U frozen", "#d99614", ":"),
    "convolution_unfrozen": (
        "conv_router_v_head_seed42_u_convolution_unfrozen_middle.json",
        "Analytic convolution U; only U frozen", "#5370b7", "-."),
}

BINARY_RUNS = {
    "generated_binary_unfrozen": (
        "conv_router_v_head_seed42_u_generated_binary_unfrozen_middle.json",
        "Generated binary U; only U frozen", "#d1495b", "-"),
    "random_binary_unfrozen": (
        "conv_router_v_head_seed42_u_random_binary_unfrozen_middle.json",
        "Fixed random binary U; only U frozen", "#6c757d", ":"),
}


def load_complete(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing completed run: {path}")
    result = json.loads(path.read_text())
    if "test" not in result:
        raise ValueError(f"Run is incomplete: {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--binary-runs", type=Path, default=DEFAULT_BINARY_RUNS)
    parser.add_argument("--dense", type=Path, default=DEFAULT_DENSE)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    runs = {key: load_complete(args.runs / spec[0])
            for key, spec in RUNS.items()}
    runs.update({key: load_complete(args.binary_runs / spec[0])
                 for key, spec in BINARY_RUNS.items()})
    specifications = {**RUNS, **BINARY_RUNS}
    dense = json.loads(args.dense.read_text())
    summary = {
        "seed": 42,
        "target": "raw ordinary digit sum",
        "runs": {},
        "dense": {str(length): dense["metrics"][f"test_{length}"]
                  for length in TEST_LENGTHS},
    }
    for key, run in runs.items():
        source_root = args.binary_runs if key in BINARY_RUNS else args.runs
        summary["runs"][key] = {
            "source": str(source_root / specifications[key][0]),
            "u_arm": run["config"].get("u_arm", "generated_ortho"),
            "middle_layers_frozen": run.get("frozen_middle_layers", True),
            "best_epoch": run["best_epoch"],
            "best_validation_mae": run["best_validation_mae"],
            "trained_parameters": run["trained_parameters"],
            "test": run["test"],
        }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.2),
                             layout="constrained")
    ax = axes[0]
    for key, (_filename, label, color, linestyle) in specifications.items():
        run = runs[key]
        ax.plot(TEST_LENGTHS,
                [100 * run["test"][str(length)]["exact_round_accuracy"]
                 for length in TEST_LENGTHS],
                color=color, linestyle=linestyle, linewidth=2,
                marker="o", markersize=4, label=label)
    ax.plot(TEST_LENGTHS,
            [100 * dense["metrics"][f"test_{length}"]["exact_round_accuracy"]
             for length in TEST_LENGTHS],
            color="#292929", linestyle="--", linewidth=2,
            marker="o", markersize=4, label="Dense MLP (full-rank U = I)")
    ax.set(xlabel="Set length", ylabel="Exact rounded sum, %",
           xticks=TEST_LENGTHS)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[1]
    # The right panel answers whether Kronecker U changes fine-tuning dynamics;
    # fixed controls remain on the already crowded quality panel.
    for key in ("generated_frozen", "generated_unfrozen",
                "kronecker_frozen", "kronecker_unfrozen",
                "generated_binary_unfrozen", "random_binary_unfrozen"):
        _filename, label, color, linestyle = specifications[key]
        history = runs[key]["history"]
        ax.plot([row["epoch"] for row in history],
                [row["validation"]["mae"] for row in history],
                color=color, linestyle=linestyle, linewidth=2, label=label)
        best = runs[key]["best_epoch"]
        point = next(row for row in history if row["epoch"] == best)
        ax.scatter([best], [point["validation"]["mae"]],
                   color=color, s=38, zorder=3)
    ax.set(xlabel="Fine-tuning epoch", ylabel="Validation MAE", ylim=(0, 0.5))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Ordinary digit sum · matched MoE controls · seed 42 · raw target")
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=180)
    plt.close(fig)
    print(f"Summary: {args.summary}\nFigure: {args.figure}")


if __name__ == "__main__":
    main()
