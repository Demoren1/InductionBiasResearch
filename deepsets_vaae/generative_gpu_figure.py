"""Render the recorded whole-device observation, without rerunning training."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/deepsets_vaae/20261001_other_generators'

def main():
    data = json.loads((OUT / 'gpu_utilization_samples.json').read_text())
    rows = data['samples']
    start = min(row['timestamp'] for row in rows)
    fig, axes = plt.subplots(4, 2, figsize=(11, 9), sharex=True, sharey=True)
    summary = {}
    for gpu, ax in enumerate(axes.flat):
        selected = [r for r in rows if r['gpu'] == gpu]
        values = np.array([r['utilization_percent'] for r in selected])
        summary[str(gpu)] = dict(samples=len(values), uuid=selected[0]['uuid'],
            mean_percent=float(values.mean()), peak_percent=float(values.max()),
            fraction_at_least_90_percent=float((values >= 90).mean()))
        ax.plot([r['timestamp']-start for r in selected], values, linewidth=.8)
        ax.set_title(f'GPU {gpu}: mean {values.mean():.1f}%, peak {values.max():.0f}%')
        ax.set_ylim(-2, 102); ax.grid(alpha=.2)
        if gpu % 2 == 0: ax.set_ylabel('Utilization, %')
        if gpu >= 6: ax.set_xlabel('Observation time, s')
    fig.suptitle('Whole-device observation: primary generators + overlapping residual fits')
    fig.tight_layout()
    out = OUT / 'plots'; out.mkdir(exist_ok=True)
    for suffix in ['png', 'pdf']:
        fig.savefig(out / f'gpu_utilization.{suffix}', dpi=160, bbox_inches='tight')
    plt.close(fig)
    result = dict(scope=data['scope'], duration_seconds=max(r['timestamp'] for r in rows)-start,
        mean_across_devices_percent=float(np.mean([v['mean_percent'] for v in summary.values()])),
        devices=summary, limitation='Whole-device utilization; includes overlapping fits and other resident processes; not a speed or memory benchmark.')
    (OUT / 'gpu_utilization_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
