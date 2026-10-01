"""Aggregate the completed pilot repeats, retaining repeat as sampling unit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

from .run import write_json
from .figures import extra_figures


def interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    margin = (float(t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
              if len(values) > 1 else None)
    return {'mean': mean, 'ci95': None if margin is None else [mean - margin, mean + margin]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.out.glob('seed_*/results.json'))
    if not paths:
        raise SystemExit('No completed repeats')
    payloads = [json.loads(path.read_text()) for path in paths]
    per_seed = {}
    for payload in payloads:
        groups = {}
        for record in payload['records']:
            key = (record['method'], record['support_size'])
            groups.setdefault(key, []).append(record['mse'])
        per_seed[payload['seed']] = {key: float(np.mean(values)) for key, values in groups.items()}
    keys = sorted(next(iter(per_seed.values())))
    if any(set(rows) != set(keys) for rows in per_seed.values()):
        raise ValueError('Repeated evaluations do not have the same groups')
    aggregate = []
    for method, support in keys:
        values = [rows[(method, support)] for rows in per_seed.values()]
        aggregate.append({'method': method, 'support_size': support, **interval(values),
                          'repeat_values': values})
    comparisons = []
    for baseline in ('mean', 'single_vae', 'random', 'dense'):
        for support in sorted({support for _, support in keys}):
            values = [rows[('agreement', support)] - rows[(baseline, support)]
                      for rows in per_seed.values()]
            comparisons.append({'baseline': baseline, 'support_size': support,
                                'direction': 'negative favors agreement', **interval(values),
                                'paired_repeat_values': values})
    result = {'completed_repeats': len(paths), 'seeds': list(per_seed),
              'inference_unit': 'independent end-to-end training repeat, conditional on fixed task vectors',
              'aggregate': aggregate, 'comparisons': comparisons}
    write_json(args.out / 'summary.json', result)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = {'agreement': 'VAAE agreement', 'mean': 'Aligned mean', 'single_vae': 'Single VAE',
              'random': 'Random mask', 'dense': 'Dense encoder'}
    for method in ('agreement', 'mean', 'single_vae', 'random', 'dense'):
        rows = sorted((row for row in aggregate if row['method'] == method),
                      key=lambda row: row['support_size'])
        ax.plot([row['support_size'] for row in rows], [row['mean'] for row in rows],
                marker='o', label=labels[method])
        if len(paths) > 1:
            ax.fill_between([row['support_size'] for row in rows],
                            [row['ci95'][0] for row in rows],
                            [row['ci95'][1] for row in rows], alpha=.1)
    ax.set_xscale('log', base=2)
    ax.set_xticks(sorted({support for _, support in keys}))
    ax.set_xticklabels(sorted({support for _, support in keys}))
    ax.set_xlabel('Total labeled sets: training + validation (5 images each)')
    ax.set_ylabel('Test MSE / set size (lower is better)')
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / 'learning_curves.png', dpi=180)
    fig.savefig(args.out / 'learning_curves.pdf')
    plt.close(fig)
    lines = ['# DeepSets + VAAE: пилот переноса маски', '',
             f'Завершено независимых повторов: {len(paths)}.', '',
             'Общий поэлементный энкодер и суммирование заданы архитектурой. '
             'Извлекается бинарная маска связей внутри энкодера; веса на новой задаче обучаются с нуля.', '',
             'Четыре исходные и восемь тестовых задач назначают цифрам разные стоимости. '
             'Размер каждого набора — пять изображений. Разбиение задач фиксировано; '
             'интервалы отражают вариативность независимого обучения, условно на этом разбиении.', '',
             'Бюджет размеченных наборов включает обучение и выбор чекпойнта: '
             'около 80% на обучение, 20% на validation. Дополнительные метки новой задачи не используются.', '',
             '| Метод | Наборов | Test MSE / 5 | 95% интервал по повторам |',
             '|---|---:|---:|---|']
    if len(paths) > 1:
        mean_comparisons = [row for row in comparisons if row['baseline'] == 'mean']
        vae_comparisons = [row for row in comparisons if row['baseline'] == 'single_vae']
        random_comparisons = [row for row in comparisons if row['baseline'] == 'random']
        findings = []
        if all(row['ci95'][1] < 0 for row in mean_comparisons):
            findings.append('Agreement снижает ошибку относительно выровненной средней карты на всей сетке бюджетов.')
        if all(row['ci95'][0] <= 0 <= row['ci95'][1] for row in vae_comparisons):
            findings.append('Дополнительный выигрыш agreement относительно одной VAE не установлен: все парные интервалы включают ноль.')
        if not all(row['ci95'][1] < 0 for row in random_comparisons):
            findings.append('Устойчивое преимущество над случайной маской на всей сетке бюджетов не установлено. '
                            'Выигрыш только относительно средней карты недостаточен для вывода об извлечении переносимой структуры.')
        if findings:
            lines[4:4] = [' '.join(findings), '']
    for row in aggregate:
        ci = row['ci95']
        bounds = '—' if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        lines.append(f"| {labels[row['method']]} | {row['support_size']} | {row['mean']:.4f} | {bounds} |")
    lines += ['', 'Парная разница agreement − baseline; отрицательное значение означает выигрыш agreement.', '',
              '| Baseline | Наборов | Разница MSE | 95% интервал |', '|---|---:|---:|---|']
    for row in comparisons:
        ci = row['ci95']
        bounds = '—' if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        lines.append(f"| {labels[row['baseline']]} | {row['support_size']} | {row['mean']:+.4f} | {bounds} |")
    lines += ['', 'Это исследовательский пилот. Он не проверяет открытие симметрии перестановок, '
              'перенос на неизвестные классы цифр или восстановление единственной истинной маски.', '',
              'Важные ограничения: банки сохраняют лучшие 25% кандидатов; VAE каждой задачи обучается '
              'на 32 картах; настройки не подбирались по тестовым задачам. Проверять сходимость '
              'по журналам банков и validation-кривым; увеличивать бюджет только отдельным новым запуском.', '',
              'Пулы имеют разные номера строк и разные точные пиксельные массивы. '
              'MNIST8m содержит аугментации: происхождение от исходных рукописных образцов '
              'не установлено, поэтому независимость по исходному почерку не заявляется.', '']
    diagnostic_paths = [path.parent / 'mask_diagnostics.json' for path in paths]
    if all(path.exists() for path in diagnostic_paths):
        diagnostics = [json.loads(path.read_text())['agreement'] for path in diagnostic_paths]
        metrics = {key: float(np.mean([np.mean(d[key]) for d in diagnostics]))
                   for key in ('soft_loss_initial', 'soft_loss_final',
                               'hard_iou_initial', 'hard_iou_final',
                               'softness_initial', 'softness_final')}
        lines += ['## Диагностика согласования', '',
                  f"Среднее мягкое расхождение: {metrics['soft_loss_initial']:.6f} → {metrics['soft_loss_final']:.6f}.", '',
                  f"Средний IoU бинарных масок разных источников: {metrics['hard_iou_initial']:.4f} → {metrics['hard_iou_final']:.4f}.", '',
                  f"Среднее p(1−p) мягкой маски: {metrics['softness_initial']:.4f} → {metrics['softness_final']:.4f}. "
                  'Большие значения означают менее бинарные вероятности. '
                  'Снижение мягкого расхождения следует оценивать отдельно от согласия итоговых бинарных структур.', '']
    lines += extra_figures(args.out, paths, comparisons, labels)
    if (args.out / 'WEIGHT_HEATMAPS.md').exists():
        lines += ['## Обученные веса dense и VAAE', '',
                  '[Heatmap матриц, бинарных масок и всех 32 фильтров с пояснениями](WEIGHT_HEATMAPS.md). '
                  'Показан заранее выбранный пример; численные параметры всех исходных '
                  'обучений воспроизведены и сохранены в `weighted_exports/`.', '']
    (args.out / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
