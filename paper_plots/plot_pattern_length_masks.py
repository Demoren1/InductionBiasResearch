"""Render two fixed evaluated pattern masks from the manuscript's JSON snapshot."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'paper/plots/data/pattern_length_masks.json'
OUTPUT = ROOT / 'paper/plots/pattern_length_masks.pdf'


def plot():
    examples = json.loads(SOURCE.read_text())['examples']
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.05), layout='constrained')
    cmap = ListedColormap(['white', '#111827'])
    for i, example in enumerate(examples):
        mask, ideal = [np.array([[int(bit) for bit in row] for row in example[key]])
                       for key in ['mask', 'ideal']]
        source, target = linear_sum_assignment(mask.T @ ideal, maximize=True)
        order = np.empty(32, dtype=int)
        order[target] = source
        aligned = mask[:, order]
        intersection = np.logical_and(aligned, ideal).sum()
        iou = intersection / np.logical_or(aligned, ideal).sum()
        assert abs(iou - example['reported_iou']) < 1e-6
        for j, (matrix, label) in enumerate([(aligned, 'CVAE'), (ideal, 'Analytic')]):
            ax = axes[2*i+j]
            ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
            ax.set_title(f"{label}, $k={example['length']}$", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(.5)
        axes[2*i].set_xlabel(f'IoU = {iou:.3f}', fontsize=8)
    axes[0].set_ylabel('Input position', fontsize=8)
    fig.supxlabel('Hidden unit (aligned for display)', fontsize=8)
    fig.savefig(OUTPUT, bbox_inches='tight')
    fig.savefig(OUTPUT.with_suffix('.png'), dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(OUTPUT)


if __name__ == '__main__':
    plot()
