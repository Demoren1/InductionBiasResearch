"""Create a paper-ready illustration of one circular motif-pair task."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Patch
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motif_pair import config  # noqa: E402
from motif_pair.data.generate import make_task_bank  # noqa: E402


A_COLOR = "#1B9E77"
B_COLOR = "#D95F02"
ONE_COLOR = "#264653"
ZERO_COLOR = "#EAF0F6"
INDEX_COLOR = "#5E6A75"
POSITIVE_COLOR = "#18794E"
NEGATIVE_COLOR = "#B42318"


def select_examples(task: config.Task) -> tuple[dict[str, object], dict[str, object]]:
    """Pick a clean, count-matched positive/negative pair for the figure."""
    bank = make_task_bank(task)
    preferred_one_counts = (8, 7, 9, 6, 10, 5, 11, 4, 12)
    preferred_negative_gaps = (8, 7, 9, 10, 4, 3, 6)

    positive_index = None
    selected_one_count = None
    for one_count in preferred_one_counts:
        candidates = torch.nonzero(
            (bank["a_start"] == 0)
            & (bank["delta"] == task.gap)
            & (bank["ones_count"] == one_count),
            as_tuple=False,
        ).flatten()
        if candidates.numel():
            positive_index = int(candidates[0])
            selected_one_count = one_count
            break
    if positive_index is None or selected_one_count is None:
        raise RuntimeError("could not find a visually aligned positive example")

    negative_index = None
    for negative_gap in preferred_negative_gaps:
        if negative_gap == task.gap:
            continue
        candidates = torch.nonzero(
            (bank["a_start"] == 0)
            & (bank["delta"] == negative_gap)
            & (bank["ones_count"] == selected_one_count),
            as_tuple=False,
        ).flatten()
        if candidates.numel():
            negative_index = int(candidates[0])
            break
    if negative_index is None:
        raise RuntimeError("could not find a count-matched hard negative")

    def unpack(index: int) -> dict[str, object]:
        bits = ((bank["x"][index].cpu() + 1) / 2).to(torch.long).tolist()
        return {
            "bits": bits,
            "a_start": int(bank["a_start"][index]),
            "b_start": int(bank["b_start"][index]),
            "delta": int(bank["delta"][index]),
            "ones_count": int(bank["ones_count"][index]),
            "label": int(bank["y"][index]),
        }

    return unpack(positive_index), unpack(negative_index)


def circular_positions(radius: float = 1.0) -> np.ndarray:
    indices = np.arange(config.SEQ_LEN)
    angles = np.pi / 2 - 2 * np.pi * indices / config.SEQ_LEN
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))


def motif_positions(start: int) -> set[int]:
    return {(start + offset) % config.SEQ_LEN for offset in range(config.MOTIF_LEN)}


def draw_example(ax: plt.Axes, example: dict[str, object], task: config.Task,
                 *, positive: bool) -> None:
    positions = circular_positions()
    a_start = int(example["a_start"])
    b_start = int(example["b_start"])
    delta = int(example["delta"])
    bits = list(example["bits"])
    a_positions = motif_positions(a_start)
    b_positions = motif_positions(b_start)

    ax.add_patch(Circle((0, 0), 1.0, facecolor="none", edgecolor="#D6DEE6",
                        linewidth=1.2, linestyle=(0, (2, 3)), zorder=0))

    for index, ((x_coord, y_coord), bit) in enumerate(zip(positions, bits)):
        if index in a_positions:
            edge_color, line_width = A_COLOR, 3.2
        elif index in b_positions:
            edge_color, line_width = B_COLOR, 3.2
        else:
            edge_color, line_width = "#AAB7C4", 1.0
        fill = ONE_COLOR if bit else ZERO_COLOR
        ax.add_patch(Circle((x_coord, y_coord), 0.155, facecolor=fill,
                            edgecolor=edge_color, linewidth=line_width, zorder=3))
        ax.text(x_coord, y_coord, str(bit), ha="center", va="center",
                fontsize=11.5, fontweight="bold",
                color="white" if bit else "#263238", zorder=4)

        label_position = circular_positions(1.26)[index]
        ax.text(label_position[0], label_position[1], str(index), ha="center",
                va="center", fontsize=7.5, color=INDEX_COLOR)

    start_positions = circular_positions(1.0)
    arrow = FancyArrowPatch(
        start_positions[a_start] * 0.79,
        start_positions[b_start] * 0.79,
        arrowstyle="-|>",
        mutation_scale=15,
        connectionstyle="arc3,rad=-0.24",
        linewidth=2.2,
        color="#4C78A8",
        zorder=2,
    )
    ax.add_patch(arrow)

    verdict_color = POSITIVE_COLOR if positive else NEGATIVE_COLOR
    gap_relation = "=" if positive else r"\ne"
    formula = (
        rf"$\Delta=(s_B-s_A)\ \mathrm{{mod}}\ 16={delta}"
        rf"\ {gap_relation}\ g\ \Rightarrow\ y={int(example['label'])}$"
    )
    ax.text(0, -1.48, formula, ha="center", va="center", fontsize=11.2,
            color="#23415C",
            bbox={"boxstyle": "round,pad=0.42", "facecolor": "#F8FAFC",
                  "edgecolor": verdict_color, "linewidth": 1.4})

    panel_title = "Positive example" if positive else "Hard negative"
    symbol = "✓" if positive else "✗"
    ax.set_title(f"{symbol}  {panel_title}", fontsize=13.5, fontweight="semibold",
                 color=verdict_color, pad=15)
    ax.text(0, -1.79, f"sequence: {''.join(map(str, bits))}", ha="center",
            va="center", fontsize=8.5, family="monospace", color="#4B5563")
    ax.set(xlim=(-1.48, 1.48), ylim=(-1.94, 1.35), aspect="equal")
    ax.axis("off")


def main() -> None:
    task = config.Task("001", "110", 5)
    positive, negative = select_examples(task)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titleweight": "semibold",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.35))
    fig.subplots_adjust(left=0.045, right=0.955, bottom=0.15, top=0.91, wspace=0.12)

    draw_example(axes[0], positive, task, positive=True)
    draw_example(axes[1], negative, task, positive=False)

    fig.legend(
        handles=(
            Patch(facecolor="white", edgecolor=A_COLOR, linewidth=3, label="pattern A = 001"),
            Patch(facecolor="white", edgecolor=B_COLOR, linewidth=3, label="pattern B = 110"),
            Patch(facecolor=ONE_COLOR, edgecolor="#AAB7C4", label="bit 1"),
            Patch(facecolor=ZERO_COLOR, edgecolor="#AAB7C4", label="bit 0"),
        ),
        loc="lower center", bbox_to_anchor=(0.5, 0.018), ncols=4,
        frameon=False, fontsize=9.5,
    )

    output = Path(__file__).with_name("motif_pair_A001_B110_G05.pdf")
    fig.savefig(output, format="pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"saved {output}")
    print(f"positive: {positive}")
    print(f"negative: {negative}")


if __name__ == "__main__":
    main()
