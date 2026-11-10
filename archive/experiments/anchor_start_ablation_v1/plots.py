"""Display frozen indexators in a common order for structural scoring."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgb


def heatmaps(path, panels, codes, title):
    """Plot every supplied condition; truth is used only for display alignment."""
    base_colors = ("#2563eb", "#dc2626", "#16a34a", "#9333ea")[:codes]
    negative = [
        tuple(0.55 + 0.45 * value for value in to_rgb(color))
        for color in reversed(base_colors)
    ]
    palette = ListedColormap(negative + ["#ffffff"] + list(base_colors))
    norm = BoundaryNorm(np.arange(-codes - 0.5, codes + 1), palette.N)
    names = ("Ground truth", "Full search", "Anchor / fixed", "Anchor / free")
    figure, axes = plt.subplots(
        len(panels), 4, figsize=(12, 2.5 * len(panels)), squeeze=False
    )
    for row, panel in enumerate(panels):
        for column, (name, index) in enumerate(zip(names, panel["indices"])):
            axis = axes[row, column]
            image = axis.imshow(
                index, cmap=palette, norm=norm, interpolation="nearest"
            )
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(name)
            if column == 0:
                axis.set_ylabel(panel["label"])
            else:
                errors = int((index != panel["indices"][0]).sum())
                axis.set_xlabel(f"{errors} entry errors", fontsize=9)
    figure.suptitle(title, y=0.995)
    figure.subplots_adjust(top=0.95, bottom=0.025, hspace=0.25, wspace=0.18)
    figure.colorbar(
        image,
        ax=list(axes.flat),
        ticks=np.arange(-codes, codes + 1),
        shrink=0.45,
        label="Signed code index; zero removes a connection",
    )
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
