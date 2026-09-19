"""Visualize how position coverage changes the Deep Sets z ablation."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .summarize import COLORS, LABELS


ROOT = Path(__file__).resolve().parent
ARMS = ("learned_z", "fixed_z", "no_z_one", "no_z_16")
PROTOCOLS = (
    ("First 10 slots only", "summary_prefix"),
    ("Random slots", "summary_random"),
    ("Random slots, strong position input", "summary_position_stress"),
)


def main() -> None:
    rows = [json.loads((ROOT / directory / "summary.json").read_text())
            ["selection_policies"]["in_distribution"]
            for _, directory in PROTOCOLS]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    offsets = np.linspace(-0.27, 0.27, len(ARMS))
    for arm_index, arm in enumerate(ARMS):
        for protocol_index, values in enumerate(rows):
            summary = values[arm]
            mae = np.asarray(summary["lengths"]["100"]["mae"]["per_seed"])
            x = protocol_index + offsets[arm_index]
            # All eight seeds are shown; deterministic jitter prevents overlap.
            jitter = np.linspace(-0.027, 0.027, len(mae))
            axes[0].scatter(x + jitter, mae, s=22, alpha=0.8,
                            color=COLORS[arm], label=LABELS[arm] if protocol_index == 0 else None)
            axes[0].plot([x - 0.07, x + 0.07], [np.median(mae)] * 2,
                         color=COLORS[arm], linewidth=2)
            axes[1].bar(x, summary["constant_assignment_count"], width=0.13,
                        color=COLORS[arm])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("MAE of the sum at set length 100 ↓")
    axes[0].set_title("Each dot is one seed; line is median")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].set_ylabel("Seeds with one U for all 100 slots")
    axes[1].set_ylim(0, 8.6)
    axes[1].set_yticks(range(0, 9, 2))
    axes[1].set_title("Recovered position-independent sharing")
    for ax in axes:
        ax.set_xticks(range(len(PROTOCOLS)), [name for name, _ in PROTOCOLS],
                      rotation=13, ha="right")
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(ROOT / "protocol_comparison.png", dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    main()
