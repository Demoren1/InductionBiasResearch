"""Render measured follow-up execution speed and live GPU utilization."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .followup_common import FOLLOWUP


def main():
    out = FOLLOWUP / 'performance'
    measurement = json.loads((out / 'reference_benchmark.json').read_text())
    assert measurement['passed'] and measurement['max_absolute_state_delta'] == 0
    fig, ax = plt.subplots(figsize=(7, 4))
    times = [measurement['sequential_seconds'], measurement['batched_seconds']]
    bars = ax.bar(['Sequential conditions', '8 conditions together'], times,
                  color=['#777777', '#2077b4'])
    for bar, seconds in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, seconds + .2,
                f'{seconds:.2f} s', ha='center')
    ax.set_ylabel('Measured elapsed seconds (lower is better)')
    ax.set_ylim(0, max(times)*1.17)
    ax.set_title(f"Identical saved weights: {measurement['speedup']:.2f}× speedup")
    for suffix in ('png', 'pdf'):
        fig.savefig(out/f'target_speed.{suffix}', dpi=160, bbox_inches='tight')
    plt.close(fig)
    sample_path = out/'utilization_samples.json'
    if sample_path.exists():
        rows = json.loads(sample_path.read_text())['samples']
        fig, ax = plt.subplots(figsize=(10, 4))
        for gpu in range(8):
            selected = [r for r in rows if r['index'] == gpu]
            ax.plot([r['elapsed_seconds'] for r in selected],
                    [r['utilization_percent'] for r in selected], label=f'GPU {gpu}')
        ax.set(xlabel='Seconds since live sampling started',
               ylabel='GPU utilization (%)', ylim=(-2, 102),
               title='Live utilization during follow-up queues')
        ax.legend(ncol=4, fontsize=8); ax.grid(alpha=.2)
        for suffix in ('png', 'pdf'):
            fig.savefig(out/f'gpu_utilization.{suffix}', dpi=160, bbox_inches='tight')
        plt.close(fig)
    snapshot = out/'source_snapshot'; snapshot.mkdir(exist_ok=True)
    for name in ('followup_batched_eval.py', 'followup_common.py',
                 'followup_performance_report.py', 'core.py'):
        shutil.copyfile(Path(__file__).with_name(name), snapshot/name)
    (out/'REPORT.md').write_text(
        '# Ускорение вычислений\n\n'
        'Независимые target-задачи и бюджеты объединены по восемь групп. '
        'Инициализация, выбор обучающих наборов, Adam и выбор checkpoint остаются '
        'отдельными для каждой группы. На каждой выделенной карте очередь допускает '
        'два процесса, чтобы перекрывать подготовку данных и вычисления.\n\n'
        '![Время обучения](target_speed.png)\n\n'
        f'Ось Y — время на двух задачах × четырёх бюджетах × '
        f"{measurement['models_per_condition']} моделях, по 800 шагов. "
        f"Исходный режим: {times[0]:.2f} с; объединённый: {times[1]:.2f} с "
        f"({measurement['speedup']:.2f}×). Это один замер скорости, без интервала "
        'неопределённости. Все восемь выбранных checkpoints совпали по весам '
        'точно; test NMSE и шаг выбора совпали. Небольшое различие служебных '
        'train-метрик не превышает 2.4e-7.\n\n'
        '![Живая загрузка GPU](gpu_utilization.png)\n\n'
        'Ось X — время наблюдения, Y — загрузка каждой карты. Это короткое '
        'наблюдение очереди с разными стадиями, а не парное сравнение старого и '
        'нового режима. Направление importance уже закончено; его карты могут '
        'простаивать. На этапах matching и подготовки данных загрузка GPU '
        'остаётся ниже, чем при обучении target-моделей.\n\n'
        'Попытка объединить все матричные произведения в один strided GEMM '
        'изменила порядок округления и конечные результаты. Она исключена '
        'из итоговых запусков. Рабочий режим сохраняет исходные размеры GEMM '
        'и объединяет остальные операции.\n\n'
        '[Точные замеры и проверка совпадения](reference_benchmark.json), '
        '[сырые наблюдения загрузки](utilization_samples.json).\n')
    print(out/'REPORT.md')


if __name__ == '__main__':
    main()
