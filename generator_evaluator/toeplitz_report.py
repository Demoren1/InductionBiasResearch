"""Report proximity to the four-diagonal pattern mask, without child fitting."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from .mask_priors import SlidingWindowMaskPrior


def write_toeplitz_report(out: Path, methods: dict[str, torch.Tensor]) -> dict:
    """Save permutation-aligned masks, edge errors, and structural scores."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    out = Path(out)
    folder = out / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    prior = SlidingWindowMaskPrior()
    reference = prior.mask()
    metrics = {name: prior.diagnostics(mask) for name, mask in methods.items()}
    report = dict(metric="four-diagonal sliding-window active-edge overlap",
                  higher_is_better=True, perfect_score=1.,
                  input_coordinates="fixed", hidden_columns="optimally matched",
                  methods=metrics)

    displayed = dict(methods)
    if "toeplitz" not in displayed:
        displayed["toeplitz"] = reference
    count = len(displayed)
    columns = min(6, count)
    blocks = math.ceil(count / columns)
    fig, axes = plt.subplots(2 * blocks, columns, figsize=(3 * columns, 7 * blocks), squeeze=False)
    for index, (name, mask) in enumerate(displayed.items()):
        block, column = divmod(index, columns)
        top, bottom = axes[2 * block, column], axes[2 * block + 1, column]
        aligned = prior.aligned_mask(mask)
        values = metrics.get(name) or prior.diagnostics(mask)
        top.imshow(aligned, cmap="Greys", vmin=0, vmax=1,
                               aspect="equal", interpolation="nearest")
        top.set_title(f"{name}\nToeplitz score: {values['toeplitz_score']:.1%}")
        # White: absent in both; green: matched edge; red: extra; blue: missing.
        errors = torch.zeros_like(reference)
        errors[(reference == 1) & (aligned == 1)] = 1
        errors[(reference == 0) & (aligned == 1)] = 2
        errors[(reference == 1) & (aligned == 0)] = 3
        bottom.imshow(errors, cmap=ListedColormap(["white", "#269d55", "#dc4949", "#448ad2"]),
                               vmin=0, vmax=3, aspect="equal", interpolation="nearest")
        bottom.set_title(f"Matched {values['matched_edges']}/{values['reference_edges']}\n"
                                  f"Extra {values['extra_edges']}, missing {values['missing_edges']}")
        for axis in (top, bottom):
            axis.set(xlabel="Aligned hidden neuron", xticks=range(8), yticks=range(11))
        if column == 0:
            top.set_ylabel("Input bit")
            bottom.set_ylabel("Input bit")
    for index in range(count, blocks * columns):
        block, column = divmod(index, columns)
        axes[2 * block, column].set_visible(False)
        axes[2 * block + 1, column].set_visible(False)
    fig.suptitle("Four-diagonal reference; green: matched, red: extra, blue: missing")
    fig.tight_layout(rect=(0, 0, 1, .95))
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"toeplitz_masks.{extension}", dpi=150)
    plt.close(fig)
    _write_proposal_plot(out, prior, metrics.get("common", {}).get("toeplitz_score"))
    return report


def _proposal_scores(out, prior):
    """Read legacy and staged proposal artifacts on a common epoch axis."""
    groups = {}
    proposals = []
    for path in (Path(out) / "proposals").glob("*.pt"):
        if not path.stem.startswith(("epoch_", "quality_", "cooperation_", "bootstrap_quality_")):
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        local_epoch = int(path.stem.rsplit("_", 1)[-1])
        proposals.append((payload, local_epoch))
    quality_epochs = max((epoch for payload, epoch in proposals
                          if payload.get("stage") == "quality"), default=0)
    ordered = []
    for payload, local_epoch in proposals:
        epoch = int(payload.get("epoch", local_epoch +
                    (quality_epochs if payload.get("stage") == "cooperation" else 0)))
        ordered.append((epoch, payload))
    for epoch, payload in sorted(ordered, key=lambda item: item[0]):
        for mask, source in zip(payload["masks"], payload["sources"]):
            if source.startswith("generator:"):
                pattern = source.split(":", 1)[1]
                groups.setdefault(pattern, {}).setdefault(epoch, []).append(
                    prior.diagnostics(mask)["toeplitz_score"])
    return groups


def _write_proposal_plot(out, prior, selected_score):
    """Show structural scores of saved generator proposals before acquisition."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    groups = _proposal_scores(out, prior)
    if not groups:
        return
    columns = min(4, len(groups))
    rows = math.ceil(len(groups) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows),
                             sharey=True, squeeze=False)
    for axis, (pattern, epochs) in zip(axes.flat, groups.items()):
        means = []
        for epoch, scores in epochs.items():
            offsets = torch.linspace(-.1, .1, len(scores)).tolist()
            axis.scatter([epoch + delta for delta in offsets], scores,
                         color="#448ad2", alpha=.7)
            means.append(sum(scores) / len(scores))
        axis.plot(list(epochs), means, "o-", color="#22517b", label="Mean proposal score")
        if selected_score is not None:
            axis.axhline(selected_score, color="#dc4949", linestyle="--", label="Selected common mask")
        axis.axhline(1., color="#269d55", linestyle=":", label="Toeplitz reference")
        axis.set(title=f"Generator {pattern}", xlabel="Training epoch", ylabel="Toeplitz edge overlap",
                 xticks=list(epochs), ylim=(0, 1.05))
        axis.yaxis.set_major_formatter(PercentFormatter(1.))
        axis.legend(fontsize=8)
    for axis in list(axes.flat)[len(groups):]:
        axis.set_visible(False)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(out / "figures" / f"toeplitz_generator_proposals.{extension}", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    frozen = torch.load(args.run / "frozen.pt", map_location="cpu", weights_only=False)
    report = write_toeplitz_report(args.run, frozen["methods"])
    for name, row in report["methods"].items():
        print(f"{name}: {row['toeplitz_score']:.1%}, matched {row['matched_edges']}/{row['reference_edges']}, "
              f"extra {row['extra_edges']}, missing {row['missing_edges']}")
    print((args.run / "figures" / "toeplitz_masks.png").resolve())


if __name__ == "__main__":
    main()
