"""Plot the gap-held-out protocol and its ideal first-layer supports.

This is a design diagnostic, not an empirical result: it visualizes which gap
values are available to meta-training and which ideal Toeplitz supports must
be produced without observing that gap during generator training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

try:
    from motif_pair import config
    from motif_pair.data.generate import gold_mask
    from motif_pair.evaluation.task_split import validate_split
except ModuleNotFoundError:  # pragma: no cover - direct invocation in motif_pair/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import config  # type: ignore
    from data.generate import gold_mask  # type: ignore
    from evaluation.task_split import validate_split  # type: ignore


def plot_gap_ood_design(split: dict, out_stem: Path) -> tuple[Path, Path]:
    """Save a wide PDF/PNG summary of a gap-held-out split."""
    validate_split(split)
    if split.get("split_kind") != "gap_heldout":
        raise ValueError("the design plot requires a gap_heldout split")
    heldout = set(int(gap) for gap in split["heldout_gaps"])

    # Motif identities do not affect the oracle support, so one valid pair is
    # sufficient to visualize the complete family of structural priors.
    pair = tuple(split.get("shared_pairs", [["000", "001"]])[0])
    figure, axes = plt.subplots(
        1, len(config.GAPS), figsize=(12.0, 2.25), sharex=True, sharey=True,
        constrained_layout=True,
    )
    cmap = ListedColormap(("#f5f5f5", "#2563eb"))
    for axis, gap in zip(axes, config.GAPS):
        task = config.Task(pair[0], pair[1], gap)
        axis.imshow(gold_mask(task), origin="lower", interpolation="nearest",
                    aspect="equal", cmap=cmap, vmin=0, vmax=1)
        is_test = gap in heldout
        axis.set_title(
            f"$g={gap}$\n{'held out' if is_test else 'train'}",
            color="#b42318" if is_test else "#344054",
            fontsize=9,
            fontweight="bold" if is_test else "normal",
        )
        for spine in axis.spines.values():
            spine.set_linewidth(2.0 if is_test else 0.8)
            spine.set_edgecolor("#b42318" if is_test else "#98a2b3")
        axis.set_xticks((0, 15), labels=("0", "15"))
        axis.set_yticks((0, 15), labels=("0", "15"))
        axis.tick_params(labelsize=7, length=2)
    axes[0].set_ylabel("input position", fontsize=9)
    for axis in axes:
        axis.set_xlabel("hidden unit", fontsize=8)
    figure.suptitle(
        "Gap-held-out protocol: ideal first-layer supports",
        fontsize=11,
    )

    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf = out_stem.with_suffix(".pdf")
    png = out_stem.with_suffix(".png")
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return pdf, png


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out-stem", type=Path, required=True)
    args = parser.parse_args()
    split = json.loads(args.split.read_text(encoding="utf-8"))
    pdf, png = plot_gap_ood_design(split, args.out_stem)
    print(f"saved {pdf} and {png}")


if __name__ == "__main__":
    main()
