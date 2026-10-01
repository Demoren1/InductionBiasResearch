"""Plot a recorded live observation of the eight density-sweep GPUs."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1]
                        / 'outputs/deepsets_vaae/20261001_density_sweep')
    parser.add_argument('--title', default='Наблюдение работы восьми GPU во время density sweep')
    args = parser.parse_args()
    out = args.out.resolve()
    rows = json.loads((out/'gpu_utilization_samples.json').read_text())['samples']
    fig, axes = plt.subplots(4, 2, figsize=(11, 9), sharex=True, sharey=True)
    summaries = {}
    for gpu, ax in enumerate(axes.flat):
        selected = [r for r in rows if r['index'] == gpu]
        values = np.array([r['utilization_percent'] for r in selected])
        summaries[str(gpu)] = {
            'uuid': selected[0]['uuid'], 'samples': len(values),
            'mean_percent': float(values.mean()), 'peak_percent': float(values.max()),
            'fraction_at_least_90_percent': float((values >= 90).mean())}
        ax.plot([r['elapsed_seconds'] for r in selected], values, linewidth=1)
        ax.set_title(f'GPU {gpu}: среднее {values.mean():.1f}%, максимум {values.max():.0f}%')
        ax.set_ylim(-2, 102); ax.grid(alpha=.2)
        if gpu % 2 == 0: ax.set_ylabel('Загрузка, %')
        if gpu >= 6: ax.set_xlabel('Секунды наблюдения')
    fig.suptitle(args.title)
    fig.tight_layout()
    plots = out/'plots'; plots.mkdir(exist_ok=True)
    for suffix in ('png', 'pdf'):
        fig.savefig(plots/f'gpu_utilization.{suffix}', dpi=160, bbox_inches='tight')
    plt.close(fig)
    (out/'gpu_utilization_summary.json').write_text(json.dumps(summaries, indent=2))
    snapshot = out/'source_snapshot'; snapshot.mkdir(exist_ok=True)
    shutil.copyfile(__file__, snapshot/Path(__file__).name)
    print(json.dumps(summaries))


if __name__ == '__main__':
    main()
