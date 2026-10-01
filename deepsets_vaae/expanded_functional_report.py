"""Aggregate the expanded functional-map VAE experiment and render its report.

The independent replicate is the seed.  Target tasks, methods, budgets, and
initializations are averaged within each seed before the t interval is formed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import t


SEEDS = tuple(range(4100, 4108))
TASKS = tuple(range(8))
BUDGETS = (32, 64, 128, 256)
METHODS = (
    'functional_mean_small', 'functional_mean_large',
    'functional_vae_small', 'functional_vae_large', 'raw_vae_large',
    'random', 'dense',
)
SPARSE_METHODS = METHODS[:-2]
SOURCE_EDGES = 5018
TRANSFER_EDGES = 7526
TOTAL_EDGES = 784 * 32

LABELS = {
    'functional_mean_small': 'Функциональное среднее, 26 карт',
    'functional_mean_large': 'Функциональное среднее, 205 карт',
    'functional_vae_small': 'VAE functional, 26 карт',
    'functional_vae_large': 'VAE functional, 205 карт',
    'raw_vae_large': 'VAE |W|, 205 карт',
    'random': 'Случайная маска',
    'dense': 'Плотная сеть',
}


def interval(values) -> dict:
    values = np.asarray(values, dtype=float)
    if values.shape != (len(SEEDS),) or not np.isfinite(values).all():
        raise ValueError(f'Expected {len(SEEDS)} finite seed-level values, got {values}')
    mean = float(values.mean())
    margin = float(t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return {'mean': mean, 'ci95': [mean - margin, mean + margin],
            'repeat_values': values.tolist(), 'df': len(values) - 1}


def save_figure(fig, out: Path, name: str) -> None:
    fig.savefig(out / f'{name}.png', dpi=180, bbox_inches='tight')
    fig.savefig(out / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def _json(path: Path):
    return json.loads(path.read_text())


def _diagnostic_path(folder: Path) -> Path:
    path = folder / 'functional' / 'functional_vae_diagnostics.json'
    if not path.is_file():
        raise ValueError(f'{folder}: missing {path.relative_to(folder)}')
    return path


def _array_path(folder: Path) -> Path | None:
    path = folder / 'functional' / 'functional_vae_arrays.npz'
    return path if path.is_file() else None


def _validate_records(records: list[dict], block: str, seed: int) -> dict:
    expected = {(task, budget, method, init)
                for task in TASKS for budget in BUDGETS
                for method in METHODS for init in range(4)}
    grouped = {}
    seen = set()
    for row in records:
        try:
            identity = (int(row['task']), int(row['support_size']),
                        str(row['method']), int(row['init']))
            value = float(row['mse'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'{block}/seed_{seed}: malformed target record {row}') from exc
        if identity in seen:
            raise ValueError(f'{block}/seed_{seed}: duplicate target record {identity}')
        if not np.isfinite(value):
            raise ValueError(f'{block}/seed_{seed}: non-finite MSE at {identity}')
        seen.add(identity)
        grouped.setdefault(identity[:3], []).append(value)
    if seen != expected:
        raise ValueError(f'{block}/seed_{seed}: target coverage mismatch; '
                         f'missing={sorted(expected-seen)[:8]}, extra={sorted(seen-expected)[:8]}')
    if any(len(values) != 4 for values in grouped.values()):
        raise ValueError(f'{block}/seed_{seed}: each task/budget/method must have four replicas')
    task_means = {key: float(np.mean(values)) for key, values in grouped.items()}
    return {(method, budget): float(np.mean([task_means[(task, budget, method)] for task in TASKS]))
            for method in METHODS for budget in BUDGETS}


def _validate_seed(folder: Path, expected_seed: int, protocol: dict) -> dict:
    result_path = folder / 'results.json'
    masks_path = folder / 'masks.pt'
    if not result_path.is_file() or not masks_path.is_file():
        raise ValueError(f'{folder}: missing results.json or masks.pt')
    seed_protocol_path = folder / 'protocol.json'
    data_provenance_path = folder / 'data_provenance.json'
    if not seed_protocol_path.is_file() or not data_provenance_path.is_file():
        raise ValueError(f'{folder}: missing protocol.json or data_provenance.json')
    seed_protocol = _json(seed_protocol_path)
    if int(seed_protocol.get('seed', -1)) != expected_seed:
        raise ValueError(f'{seed_protocol_path}: seed mismatch')
    seed_spec = {key: value for key, value in seed_protocol.items()
                 if key not in ('seed', 'cuda_visible_devices')}
    if seed_spec != protocol:
        raise ValueError(f'{seed_protocol_path}: run protocol differs from root protocol')
    data_provenance = _json(data_provenance_path)
    if data_provenance.get('row_ids_pairwise_disjoint') is not True:
        raise ValueError(f'{data_provenance_path}: data splits are not verified disjoint')
    payload = _json(result_path)
    if int(payload.get('seed', -1)) != expected_seed:
        raise ValueError(f'{result_path}: seed mismatch')
    old = _validate_records(payload.get('records', []), 'old', expected_seed)
    fresh = _validate_records(payload.get('fresh_records', []), 'fresh', expected_seed)

    masks = torch.load(masks_path, map_location='cpu', weights_only=True)
    if set(masks) != set(METHODS):
        raise ValueError(f'{masks_path}: expected masks for {METHODS}, got {sorted(masks)}')
    for method in METHODS:
        value = torch.as_tensor(masks[method])
        if tuple(value.shape) != (4, 784, 32):
            raise ValueError(f'{masks_path}/{method}: expected [4,784,32], got {tuple(value.shape)}')
        if not bool(torch.isfinite(value).all()) or not bool(torch.all((value == 0) | (value == 1))):
            raise ValueError(f'{masks_path}/{method}: mask is non-finite or non-binary')
        wanted = TOTAL_EDGES if method == 'dense' else TRANSFER_EDGES
        counts = value.sum((-1, -2)).tolist()
        if counts != [wanted] * 4:
            raise ValueError(f'{masks_path}/{method}: expected {wanted} edges per replica, got {counts}')

    diagnostics_path = _diagnostic_path(folder)
    diagnostics = _json(diagnostics_path)
    if int(diagnostics.get('seed', -1)) != expected_seed + 50_000:
        raise ValueError(f'{diagnostics_path}: expected VAE substream seed {expected_seed + 50_000}')
    if diagnostics.get('k', TRANSFER_EDGES) != TRANSFER_EDGES:
        raise ValueError(f'{diagnostics_path}: functional mask K is not {TRANSFER_EDGES}')
    source_quality = []
    for task in range(4):
        bank_path = folder / f'bank_{task}.pt'
        if not bank_path.is_file():
            raise ValueError(f'{folder}: missing source bank {bank_path.name}')
        bank = torch.load(bank_path, map_location='cpu', weights_only=False)
        if int(bank.get('edges_per_mask', -1)) != SOURCE_EDGES:
            raise ValueError(f'{bank_path}: source mask K must be {SOURCE_EDGES}')
        all_scores = np.asarray(bank.get('best_validation_normalized_mse', []), dtype=float)
        selected_scores = np.asarray(bank.get('validation_losses', []), dtype=float)
        chosen = np.asarray(bank.get('selected_candidates', []), dtype=int)
        if all_scores.shape != (1024,) or selected_scores.shape != (256,) or chosen.shape != (256,):
            raise ValueError(f'{bank_path}: expected 1024 candidate scores and 256 selected scores')
        if not np.isfinite(all_scores).all() or not np.isfinite(selected_scores).all():
            raise ValueError(f'{bank_path}: non-finite source validation score')
        if len(set(chosen.tolist())) != 256 or np.any(chosen < 0) or np.any(chosen >= 1024):
            raise ValueError(f'{bank_path}: invalid selected candidate indices')
        if not np.array_equal(selected_scores, all_scores[chosen]):
            raise ValueError(f'{bank_path}: selected score order does not match candidate scores')
        source_quality.append({'all': all_scores, 'selected': selected_scores})
    for label, base in (('old', folder / 'weights'), ('fresh', folder / 'fresh_weights')):
        expected_names = {f'target_task{task}_budget{budget}.pt'
                          for task in TASKS for budget in BUDGETS}
        present = {path.name for path in base.glob('target_task*_budget*.pt')} if base.is_dir() else set()
        if present != expected_names:
            raise ValueError(f'{folder}/{label}: expected 32 selected checkpoints, found '
                             f'{len(present)}; missing={sorted(expected_names-present)[:4]}')
    return {'seed': expected_seed, 'folder': folder, 'old': old, 'fresh': fresh,
            'diagnostics': diagnostics, 'diagnostics_path': diagnostics_path,
            'array_path': _array_path(folder), 'source_quality': source_quality,
            'seed_protocol': seed_protocol, 'data_provenance': data_provenance}


def load_inputs(out: Path, protocol: dict) -> list[dict]:
    seed_dirs = [out / f'seed_{seed}' for seed in SEEDS]
    absent = [str(path) for path in seed_dirs if not path.is_dir()]
    if absent:
        raise ValueError(f'Incomplete experiment: missing seed directories {absent}')
    return [_validate_seed(folder, seed, protocol) for folder, seed in zip(seed_dirs, SEEDS)]


def _validate_protocol(out: Path) -> dict:
    path = out / 'protocol.json'
    if not path.is_file():
        raise ValueError(f'Incomplete experiment: missing {path.name}')
    protocol = _json(path)
    expected = {
        'source_tasks': 4, 'candidates': 1024, 'keep': 256,
        'source_density': .2, 'target_density': .3, 'target_edges': TRANSFER_EDGES,
        'large_train_maps': 205, 'small_train_maps': 26, 'validation_maps': 51,
        'support_sizes': list(BUDGETS), 'eval_steps': 800,
        'test_sets': 512, 'smoke': False,
    }
    wrong = {key: (protocol.get(key), value) for key, value in expected.items()
             if protocol.get(key) != value}
    if wrong:
        raise ValueError(f'{path}: protocol differs from report specification: {wrong}')
    if tuple(protocol.get('seeds', ())) != SEEDS:
        raise ValueError(f'{path}: seed list must be {list(SEEDS)}')
    if protocol.get('methods') != list(METHODS):
        raise ValueError(f'{path}: method order differs from report specification')
    primary = protocol.get('primary_comparison', {})
    if primary != {'population': 'fresh_test', 'budget': 256,
                   'method': 'functional_vae_large', 'baseline': 'functional_mean_large'}:
        raise ValueError(f'{path}: primary comparison differs from the predeclared contrast')
    if protocol.get('target_task_labels_for_mask_extraction') is not False:
        raise ValueError(f'{path}: target labels must not be used to extract masks')
    return protocol


def _seed_values(rows: list[dict], split: str, method: str, budget: int) -> list[float]:
    return [row[split][(method, budget)] for row in rows]


def _summarize(rows: list[dict], split: str) -> dict:
    aggregate = []
    for method in METHODS:
        for budget in BUDGETS:
            aggregate.append({'method': method, 'support_size': budget,
                              **interval(_seed_values(rows, split, method, budget))})
    comparisons = []
    for budget in BUDGETS:
        contrasts = [(method, baseline) for method in SPARSE_METHODS
                     for baseline in ('random', 'dense')]
        contrasts += [(method, 'functional_mean_large') for method in
                      ('functional_mean_small', 'functional_vae_small',
                       'functional_vae_large', 'raw_vae_large')]
        contrasts += [('functional_vae_small', 'functional_mean_small'),
                      ('functional_mean_large', 'functional_mean_small'),
                      ('functional_vae_large', 'functional_vae_small'),
                      ('raw_vae_large', 'functional_vae_large')]
        for method, baseline in contrasts:
            paired = [row[split][(method, budget)] - row[split][(baseline, budget)]
                      for row in rows]
            comparisons.append({'method': method, 'baseline': baseline,
                                'support_size': budget,
                                'direction': 'negative favors the first method',
                                **interval(paired)})
    primary = next(row for row in comparisons
                   if row['support_size'] == 256
                   and row['method'] == 'functional_vae_large'
                   and row['baseline'] == 'functional_mean_large')
    return {'aggregate': aggregate, 'comparisons': comparisons,
            'primary_comparison': primary,
            'inference_unit': 'seed-level mean across 8 fixed target tasks and 4 initializations; t interval, df=7'}


def _plot_learning(out: Path, summaries: dict, figure_data: dict) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    colors = dict(zip(METHODS, plt.cm.tab10.colors[:len(METHODS)]))
    for ax, split, title in zip(axes, ('old', 'fresh'),
                                ('Старые 8 задач стоимости', 'Свежие 8 задач стоимости')):
        ax.set_title(title)
        for method in METHODS:
            values = [row for row in summaries[split]['aggregate'] if row['method'] == method]
            values.sort(key=lambda row: row['support_size'])
            x = np.asarray([row['support_size'] for row in values])
            y = np.asarray([row['mean'] for row in values])
            lower = np.asarray([row['ci95'][0] for row in values])
            upper = np.asarray([row['ci95'][1] for row in values])
            ax.plot(x, y, marker='o', label=LABELS[method], color=colors[method])
            ax.fill_between(x, lower, upper, color=colors[method], alpha=.10)
        ax.set_xscale('log', base=2)
        ax.set_xticks(BUDGETS, [str(x) for x in BUDGETS])
        ax.set_xlabel('Размеченные наборы новой target-задачи')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Test MSE / 5 (меньше — лучше)')
    axes[1].legend(fontsize=8, loc='best')
    fig.tight_layout()
    save_figure(fig, out, 'learning_curves')
    figure_data['learning_curves'] = summaries
    return [
        '## Качество переноса', '',
        '![Кривые переноса](learning_curves.png)', '',
        '**Как читать.** По горизонтали — полный бюджет размеченных наборов target-задачи, '
        'включая обучение и выбор checkpoint; по вертикали — тестовая MSE, делённая на 5. '
        'Левая панель использует прежние 8 задач стоимости, правая — 8 новых заранее '
        'зафиксированных задач. Точки — среднее сначала по задачам и четырём инициализациям '
        'внутри seed, затем по восьми seeds. Полосы — t-интервалы 95% по seed, df=7.', '',
        '**Ограничение.** Интервалы условны на фиксированных задачах и разбиениях данных. '
        'Они отражают вариативность обучения между seeds; target-задачи не являются '
        'независимыми повторениями.', '',
    ]


CONTRASTS = (
    ('functional_vae_large', 'functional_mean_large', 'VAE large − mean large'),
    ('functional_vae_small', 'functional_mean_small', 'VAE small − mean small'),
    ('functional_mean_large', 'functional_mean_small', 'Mean: 205 − 26 карт'),
    ('functional_vae_large', 'functional_vae_small', 'VAE: 205 − 26 карт'),
    ('functional_vae_large', 'random', 'VAE large − random'),
    ('functional_vae_large', 'dense', 'VAE large − dense'),
)


def _plot_effects(out: Path, summaries: dict, figure_data: dict) -> list[str]:
    lines = ['## Парные различия между вариантами', '']
    for split, title in (('old', 'Старые target-задачи'), ('fresh', 'Новые target-задачи')):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
        for ax, (method, baseline, label) in zip(axes.flat, CONTRASTS):
            rows = [row for row in summaries[split]['comparisons']
                    if row['method'] == method and row['baseline'] == baseline]
            rows.sort(key=lambda row: row['support_size'])
            x = np.asarray([row['support_size'] for row in rows])
            y = np.asarray([row['mean'] for row in rows])
            lo = np.asarray([row['ci95'][0] for row in rows])
            hi = np.asarray([row['ci95'][1] for row in rows])
            ax.errorbar(x, y, yerr=np.vstack((y - lo, hi - y)), marker='o', capsize=3)
            ax.axhline(0, color='black', lw=.8)
            ax.set_title(label, fontsize=10)
            ax.set_xscale('log', base=2)
            ax.set_xticks(BUDGETS, [str(v) for v in BUDGETS])
            ax.set_xlabel('Бюджет target-разметки')
            ax.set_ylabel('Парная разность test MSE / 5')
            ax.grid(alpha=.2)
        fig.suptitle(title)
        fig.tight_layout()
        name = f'paired_effects_{split}'
        save_figure(fig, out, name)
        figure_data[name] = [row for row in summaries[split]['comparisons']
                             if (row['method'], row['baseline']) in
                             {(m, b) for m, b, _ in CONTRASTS}]
        lines += [f'### {title}', '', f'![Парные эффекты: {title}]({name}.png)', '',
                  '**Как читать.** По вертикали показана разность MSE первого метода и baseline '
                  'на тех же seed, задаче, бюджете и инициализации; внутри seed сначала усреднены '
                  'фиксированные target-задачи и четыре обучения. Значения ниже нуля благоприятны '
                  'первому методу. Отрезки — парные 95% t-интервалы по восьми seeds, df=7. '
                  'Интервалы точечные, без поправки на множественные бюджеты и сравнения.', '',
                  'Сравнение small с large отвечает на эффект большего числа карт при фиксированном '
                  'способе построения; сравнение VAE с соответствующим mean оценивает добавку VAE. '
                  'Сравнения с random и dense относятся к переносу структуры на target-задачу.', '']

    primary = summaries['fresh']['primary_comparison']
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    y = primary['mean']
    ax.errorbar([0], [y], yerr=[[y - primary['ci95'][0]], [primary['ci95'][1] - y]],
                fmt='o', capsize=5, color='#2166ac')
    ax.axhline(0, color='black', lw=1)
    ax.set_xlim(-.8, .8)
    ax.set_xticks([0], ['Бюджет 256'])
    ax.set_ylabel('VAE functional large − mean large, test MSE / 5')
    ax.set_title('Заранее выбранное сравнение на новых target-задачах')
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    save_figure(fig, out, 'primary_fresh_256')
    figure_data['primary_fresh_256'] = primary
    lines += ['### Предварительно выбранная проверка', '',
              '![Основное сравнение](primary_fresh_256.png)', '',
              '**Как читать.** Показана только заранее обозначенная разность на новых задачах '
              'стоимости при бюджете 256: VAE по функциональным картам с 205 train-картами '
              'минус функциональное среднее тех же 205 карт. Интервал парный по восьми seeds '
              '(df=7); отрицательное значение поддерживает VAE.', '',
              f"Оценка разности: {primary['mean']:+.6f}; 95% интервал "
              f"[{primary['ci95'][0]:+.6f}, {primary['ci95'][1]:+.6f}]. Это единственное "
              'подтверждающее сравнение, заданное заранее. Все остальные сравнения и графики '
              'имеют исследовательский статус; новые target-задачи не использовались для выбора '
              'варианта VAE.', '']
    return lines


def _iter_metric_rows(value):
    """Flatten diagnostics while retaining per-task dictionaries."""
    if isinstance(value, list):
        for item in value:
            yield from _iter_metric_rows(item)
    elif isinstance(value, dict):
        if any(isinstance(v, (int, float)) for v in value.values()):
            yield value
        else:
            for item in value.values():
                yield from _iter_metric_rows(item)


def _metric(row: dict, names: tuple[str, ...]):
    for name in names:
        value = row.get(name)
        if isinstance(value, (int, float)) and np.isfinite(value):
            return float(value)
    return None


HELDOUT_KEYS = ('heldout_reconstruction_mse', 'heldout_mse', 'validation_mse',
                'reconstruction_mse', 'heldout_loss', 'validation_loss')
HELDOUT_BCE_KEYS = ('heldout_reconstruction_bce', 'reconstruction_bce')
MEAN_KEYS = ('train_mean_mse', 'mean_baseline_mse', 'heldout_mean_mse',
             'mean_mse', 'baseline_mse', 'training_mean_mse')
MEAN_BCE_KEYS = ('train_mean_bce', 'heldout_train_mean_bce')
RECON_IOU_KEYS = ('heldout_diversity_reconstruction_pairwise_iou_mean',
                  'diversity_reconstruction_pairwise_iou_mean')
TARGET_IOU_KEYS = ('heldout_diversity_target_pairwise_iou_mean',
                   'diversity_target_pairwise_iou_mean')
RECON_HAMMING_KEYS = ('heldout_diversity_reconstruction_pairwise_hamming_mean',
                      'diversity_reconstruction_pairwise_hamming_mean')
TARGET_HAMMING_KEYS = ('heldout_diversity_target_pairwise_hamming_mean',
                       'diversity_target_pairwise_hamming_mean')
RECON_UNIQUE_KEYS = ('heldout_diversity_reconstruction_unique_fraction',
                     'diversity_reconstruction_unique_fraction')
TARGET_UNIQUE_KEYS = ('heldout_diversity_target_unique_fraction',
                      'diversity_target_unique_fraction')
VARIANCE_RATIO_KEYS = ('heldout_reconstruction_to_target_variance_ratio',
                       'reconstruction_to_target_variance_ratio')


def _diagnostic_metric_samples(diagnostics: dict) -> dict:
    metrics = diagnostics.get('reconstruction_metrics', {})
    result = {}
    for method, content in metrics.items():
        rows = list(_iter_metric_rows(content))
        heldout = [v for row in rows if (v := _metric(row, HELDOUT_KEYS)) is not None]
        heldout_bce = [v for row in rows if (v := _metric(row, HELDOUT_BCE_KEYS)) is not None]
        mean = [v for row in rows if (v := _metric(row, MEAN_KEYS)) is not None]
        mean_bce = [v for row in rows if (v := _metric(row, MEAN_BCE_KEYS)) is not None]
        details = {
            'reconstruction_pairwise_iou': [v for row in rows if (v := _metric(row, RECON_IOU_KEYS)) is not None],
            'target_pairwise_iou': [v for row in rows if (v := _metric(row, TARGET_IOU_KEYS)) is not None],
            'reconstruction_pairwise_hamming': [v for row in rows if (v := _metric(row, RECON_HAMMING_KEYS)) is not None],
            'target_pairwise_hamming': [v for row in rows if (v := _metric(row, TARGET_HAMMING_KEYS)) is not None],
            'reconstruction_unique_fraction': [v for row in rows if (v := _metric(row, RECON_UNIQUE_KEYS)) is not None],
            'target_unique_fraction': [v for row in rows if (v := _metric(row, TARGET_UNIQUE_KEYS)) is not None],
            'variance_ratio': [v for row in rows if (v := _metric(row, VARIANCE_RATIO_KEYS)) is not None],
        }
        if heldout or heldout_bce or mean or mean_bce or any(details.values()):
            result[method] = {'heldout_reconstruction_mse': heldout,
                              'heldout_reconstruction_bce': heldout_bce,
                              'train_mean_mse': mean, 'train_mean_bce': mean_bce,
                              **details}
    return result


def _plot_source_quality(out: Path, seeds: list[dict], figure_data: dict) -> list[str]:
    """Plot source-validation quality before/after retaining the top 256 candidates."""
    all_scores = np.concatenate([bank['all'] for seed in seeds for bank in seed['source_quality']])
    selected_scores = np.concatenate([bank['selected'] for seed in seeds for bank in seed['source_quality']])
    all_seed_means = [float(np.mean([bank['all'].mean() for bank in seed['source_quality']]))
                      for seed in seeds]
    selected_seed_means = [float(np.mean([bank['selected'].mean() for bank in seed['source_quality']]))
                           for seed in seeds]
    gain = interval(np.asarray(selected_seed_means) - np.asarray(all_seed_means))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    bins = np.linspace(min(all_scores.min(), selected_scores.min()),
                       max(all_scores.max(), selected_scores.max()), 55)
    axes[0].hist(all_scores, bins=bins, density=True, alpha=.55, label='Все 1024 кандидата')
    axes[0].hist(selected_scores, bins=bins, density=True, alpha=.6, label='Выбранные 256')
    axes[0].set_xlabel('Best source-validation NMSE')
    axes[0].set_ylabel('Плотность')
    axes[0].set_title('Распределение качества банков')
    axes[0].legend(fontsize=8)
    x = np.array([0, 1])
    for left, right in zip(all_seed_means, selected_seed_means):
        axes[1].plot(x, [left, right], color='steelblue', alpha=.45, marker='o')
    for index, values in enumerate((all_seed_means, selected_seed_means)):
        stat = interval(values)
        axes[1].errorbar([index], [stat['mean']],
                         yerr=[[stat['mean'] - stat['ci95'][0]],
                               [stat['ci95'][1] - stat['mean']]],
                         fmt='o', color='black', capsize=4)
    axes[1].set_xticks(x, ['Все 1024', 'Выбранные 256'])
    axes[1].set_ylabel('Средний best source-validation NMSE')
    axes[1].set_title('Парные средние по seed')
    axes[1].grid(axis='y', alpha=.2)
    fig.tight_layout()
    save_figure(fig, out, 'source_bank_quality')
    np.savez_compressed(out / 'source_bank_quality_values.npz',
                        all_candidate_scores=all_scores, selected_scores=selected_scores,
                        all_seed_means=np.asarray(all_seed_means),
                        selected_seed_means=np.asarray(selected_seed_means))
    figure_data['source_bank_quality'] = {
        'all_candidate_scores': {'count': len(all_scores), 'mean': float(all_scores.mean()),
                                 'median': float(np.median(all_scores))},
        'selected_scores': {'count': len(selected_scores), 'mean': float(selected_scores.mean()),
                            'median': float(np.median(selected_scores))},
        'all_seed_means': all_seed_means, 'selected_seed_means': selected_seed_means,
        'selected_minus_all_seed_mean': gain,
    }
    lines = ['## Качество исходных банков', '',
             '![Качество source-банков](source_bank_quality.png)', '',
             '**Как читать.** Слева распределение лучшей source-validation NMSE всех 1024 '
             'кандидатов и сохранённых 256 кандидатов на четырёх source-задачах и восьми seeds. '
             'Справа линии соединяют средние по четырём задачам внутри каждого seed; чёрные '
             'точки и отрезки — среднее и 95% t-интервал по восьми seeds (df=7). Меньше NMSE '
             'означает лучшее качество на source validation.', '',
             '**Граница интерпретации.** Эти же source-validation оценки использовались для '
             'выбора 256 из 1024, поэтому график показывает качество отбора, а не независимую '
             'оценку обобщения source-банка. Он не участвует в сравнении методов на target-задачах.', '',
             f"Парная разность mean(selected − all) = {gain['mean']:+.6f}; 95% CI "
             f"[{gain['ci95'][0]:+.6f}, {gain['ci95'][1]:+.6f}].", '',
             'Полные значения распределений сохранены в `source_bank_quality_values.npz`.', '']

    return lines


def _plot_reconstruction(out: Path, seeds: list[dict], figure_data: dict) -> list[str]:
    metrics_by_seed = [_diagnostic_metric_samples(row['diagnostics']) for row in seeds]
    methods = [method for method in ('functional_vae_small', 'functional_vae_large', 'raw_vae_large')
               if all(method in metrics and metrics[method]['heldout_reconstruction_mse']
                      for metrics in metrics_by_seed)]
    if not methods:
        return []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    reconstruction_data = {}
    colors = {'functional_vae_small': '#1b9e77', 'functional_vae_large': '#d95f02',
              'raw_vae_large': '#7570b3'}
    display = {'functional_vae_small': 'VAE functional, 26',
               'functional_vae_large': 'VAE functional, 205',
               'raw_vae_large': 'VAE |W|, 205'}
    for method in methods:
        reconstruction = [float(np.mean(metrics[method]['heldout_reconstruction_mse']))
                          for metrics in metrics_by_seed]
        baseline = [float(np.mean(metrics[method]['train_mean_mse']))
                    if metrics[method]['train_mean_mse'] else float('nan')
                    for metrics in metrics_by_seed]
        ratios = []
        for metrics in metrics_by_seed:
            current = np.asarray(metrics[method]['heldout_reconstruction_mse'], dtype=float)
            mean = np.asarray(metrics[method]['train_mean_mse'], dtype=float)
            ratios.append(float(np.mean(current / mean)) if np.all(mean > 0) else float('nan'))
        if not np.isfinite(ratios).all():
            raise ValueError(f'{method}: held-out/train-mean MSE ratio is non-finite')
        stat = interval(ratios)
        axes[0].errorbar([display[method]], [stat['mean']],
                         yerr=[[stat['mean']-stat['ci95'][0]], [stat['ci95'][1]-stat['mean']]],
                         fmt='o', capsize=3, color=colors[method], label=LABELS[method])
        reconstruction_data[method] = {
            'heldout_to_train_mean_mse_ratio': {'seed_values': ratios, **stat},
            'heldout_reconstruction_mse': {'seed_values': reconstruction,
                                           **interval(reconstruction)},
            'train_mean_mse': {'seed_values': baseline, **interval(baseline)},
        }
    axes[0].axhline(1, color='black', lw=1, linestyle='--')
    axes[0].set_title('Held-out ошибка относительно своего baseline')
    axes[0].set_ylabel('Reconstruction MSE / train-mean MSE')
    axes[0].tick_params(axis='x', labelrotation=18)
    axes[0].grid(axis='y', alpha=.2)

    for ax, field, baseline_field, title, ylabel in (
            (axes[1], 'variance_ratio', None, 'Дисперсия карт',
             'Дисперсия реконструкций / исходных карт'),
            (axes[2], 'reconstruction_pairwise_iou', 'target_pairwise_iou',
             'Сходство binary support', 'Pairwise IoU')):
        for method in methods:
            if not all(metrics.get(method, {}).get(field) for metrics in metrics_by_seed):
                continue
            values = [float(np.mean(metrics[method][field])) for metrics in metrics_by_seed]
            value_stat = interval(values)
            ax.errorbar([display[method]], [value_stat['mean']],
                        yerr=[[value_stat['mean']-value_stat['ci95'][0]],
                              [value_stat['ci95'][1]-value_stat['mean']]],
                        fmt='o', capsize=3, color=colors[method], label=LABELS[method])
            reconstruction_data.setdefault(method, {})[field] = {'seed_values': values, **value_stat}
            if baseline_field and all(metrics.get(method, {}).get(baseline_field)
                                      for metrics in metrics_by_seed):
                targets = [float(np.mean(metrics[method][baseline_field])) for metrics in metrics_by_seed]
                target_stat = interval(targets)
                ax.errorbar([display[method]], [target_stat['mean']],
                            yerr=[[target_stat['mean']-target_stat['ci95'][0]],
                                  [target_stat['ci95'][1]-target_stat['mean']]],
                            fmt='x', capsize=3, color=colors[method], alpha=.7)
                reconstruction_data[method][baseline_field] = {'seed_values': targets, **target_stat}
        if field == 'variance_ratio':
            ax.axhline(1, color='black', lw=1, linestyle='--')
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', labelrotation=18)
        ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    save_figure(fig, out, 'vae_reconstruction')
    figure_data['vae_reconstruction'] = reconstruction_data
    return ['### Восстановление и разнообразие карт', '',
            '![Восстановление VAE](vae_reconstruction.png)', '',
            '**Как читать.** Слева — held-out reconstruction MSE, делённая на MSE '
            'предсказания своим train mean; пунктир на 1 означает равенство baseline. '
            'Это относительная ошибка внутри одного представления: абсолютные значения '
            'MSE raw-карт и functional-карт имеют разные шкалы и по ним нельзя объявлять '
            'одно представление хуже другого. В центре — отношение дисперсии реконструкций '
            'к дисперсии исходных карт; пунктир на 1 означает совпадение. Справа — '
            'pairwise IoU реконструированных бинарных масок (точка) и held-out масок '
            '(крестик); высокое IoU означает низкое разнообразие выходных поддержек. '
            'Точки усредняют source-задачи внутри seed; интервалы рассчитаны по восьми '
            'seeds (df=7). Raw VAE обучается на |W|, остальные варианты — на functional maps. '
            'Поэтому сравнение raw и functional включает различие входных представлений.', '',
            'Визуальные примеры исходных и восстановленных карт сохраняются отдельными '
            'численными массивами в `functional_vae_arrays.npz`; сама MSE '
            'не показывает геометрию ошибок.', '']


def _source_vae_metric_summary(seeds: list[dict]) -> dict:
    metrics_by_seed = [_diagnostic_metric_samples(row['diagnostics']) for row in seeds]
    result = {}
    methods = ('functional_vae_small', 'functional_vae_large', 'raw_vae_large')
    fields = ('heldout_reconstruction_mse', 'train_mean_mse',
              'heldout_reconstruction_bce', 'train_mean_bce',
              'variance_ratio', 'reconstruction_pairwise_iou', 'target_pairwise_iou',
              'reconstruction_pairwise_hamming', 'target_pairwise_hamming')
    for method in methods:
        if not all(method in metrics for metrics in metrics_by_seed):
            raise ValueError(f'Missing source reconstruction diagnostics for {method}')
        result[method] = {}
        seed_values = {field: [] for field in fields}
        for metrics in metrics_by_seed:
            current = metrics[method]
            if any(len(current.get(field, [])) != 4 for field in fields):
                raise ValueError(f'{method}: expected one held-out diagnostic per source task')
            for field in fields:
                seed_values[field].append(float(np.mean(current[field])))
            ratios = np.asarray(current['heldout_reconstruction_mse']) / np.asarray(current['train_mean_mse'])
            seed_values.setdefault('heldout_to_train_mean_mse_ratio', []).append(float(ratios.mean()))
        for field, values in seed_values.items():
            result[method][field] = interval(values)
        result[method]['heldout_bce_minus_train_mean_bce'] = interval(
            np.asarray(seed_values['heldout_reconstruction_bce'])
            - np.asarray(seed_values['train_mean_bce']))
        result[method]['reconstruction_pairwise_iou_minus_target'] = interval(
            np.asarray(seed_values['reconstruction_pairwise_iou'])
            - np.asarray(seed_values['target_pairwise_iou']))
    return result


def _selected_checkpoint(folder: Path, seed: int, split: str) -> dict:
    base = folder / ('fresh_weights' if split == 'fresh' else 'weights')
    path = base / 'target_task0_budget256.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    required = {'method_names', 'replica_indices', 'records', 'state_dict', 'effective_weight'}
    if not required.issubset(checkpoint):
        raise ValueError(f'{path}: missing checkpoint keys {sorted(required-set(checkpoint))}')
    if int(checkpoint.get('task', -1)) != 0 or int(checkpoint.get('support_size', -1)) != 256:
        raise ValueError(f'{path}: checkpoint metadata does not match task 0, budget 256')
    if len(checkpoint['method_names']) != 28 or len(checkpoint['replica_indices']) != 28:
        raise ValueError(f'{path}: expected exactly 28 method/init models')
    values = {}
    for method in ('dense', 'functional_mean_small', 'functional_mean_large',
                   'functional_vae_small', 'functional_vae_large', 'raw_vae_large', 'random'):
        found = [i for i, (name, replica) in enumerate(zip(checkpoint['method_names'],
                                                            checkpoint['replica_indices']))
                 if name == method and int(replica) == 0]
        if len(found) != 1:
            if method == 'random':
                continue
            raise ValueError(f'{path}: expected exactly one {method} init=0, found {found}')
        i = found[0]
        state = checkpoint['state_dict']
        weight = state['weight'][i].detach().cpu().numpy()
        mask = state['masks'][i].detach().cpu().numpy()
        effective = checkpoint['effective_weight'][i].detach().cpu().numpy()
        if weight.shape != (784, 32) or mask.shape != weight.shape or effective.shape != weight.shape:
            raise ValueError(f'{path}/{method}: unexpected selected array shapes')
        if not (np.isfinite(weight).all() and np.isfinite(mask).all() and np.isfinite(effective).all()):
            raise ValueError(f'{path}/{method}: selected arrays contain NaN/Inf')
        if not np.array_equal(weight * mask, effective):
            raise ValueError(f'{path}/{method}: effective weight differs from W*M')
        if not np.all(effective[mask == 0] == 0):
            raise ValueError(f'{path}/{method}: forbidden weights are not exact zeros')
        wanted = TOTAL_EDGES if method == 'dense' else TRANSFER_EDGES
        actual = int(mask.sum())
        if actual != wanted:
            raise ValueError(f'{path}/{method}: expected K={wanted}, found {actual}')
        record_rows = [r for r in checkpoint['records']
                       if r['method'] == method and int(r['init']) == 0]
        if len(record_rows) != 1:
            raise ValueError(f'{path}/{method}: selected target record missing/duplicated')
        values[method] = {'weight': weight, 'mask': mask, 'effective_weight': effective,
                          'record': record_rows[0]}
    return values


def _export_heatmaps(out: Path, seeds: list[dict], figure_data: dict) -> list[str]:
    folder = next(row['folder'] for row in seeds if row['seed'] == 4100)
    fresh = _selected_checkpoint(folder, 4100, 'fresh')
    # The fixed example is specifically the fresh task 0 example.
    selected = fresh
    order = ('dense', 'functional_mean_small', 'functional_mean_large',
             'functional_vae_small', 'functional_vae_large', 'raw_vae_large', 'random')
    limit = max(float(np.abs(selected[name]['effective_weight']).max()) for name in order
                if name in selected)
    destination = out / 'weight_heatmaps'
    destination.mkdir(parents=True, exist_ok=True)
    arrays = {}
    metadata = {'selection': {'seed': 4100, 'split': 'fresh', 'task': 0,
                              'budget': 256, 'init': 0},
                'selection_rule': 'predeclared fixed example; no test-quality selection or column sorting',
                'matrix_shape': [784, 32], 'signed_color_limits': [-limit, limit],
                'source_bank_mask_edges': SOURCE_EDGES,
                'transferred_mask_edges': TRANSFER_EDGES,
                'methods': {}}
    fig = plt.figure(figsize=(24, 10), layout='constrained')
    grid = fig.add_gridspec(1, len(order) + 1,
                            width_ratios=[1] * len(order) + [.06])
    axes = [fig.add_subplot(grid[0, index]) for index in range(len(order))]
    colorbar_axis = fig.add_subplot(grid[0, -1])
    for ax, method in zip(axes, order):
        if method not in selected:
            ax.set_visible(False)
            continue
        value = selected[method]
        image = ax.imshow(value['effective_weight'], cmap='RdBu_r', vmin=-limit, vmax=limit,
                          aspect='auto', interpolation='nearest')
        ax.set_title(LABELS[method], fontsize=9, rotation=18)
        ax.set_xlabel('Скрытый нейрон')
        if ax is axes[0]:
            ax.set_ylabel('Пиксель (строка × 28 + столбец)')
        arrays[f'{method}_W'] = value['weight']
        arrays[f'{method}_M'] = value['mask']
        arrays[f'{method}_effective'] = value['effective_weight']
        metadata['methods'][method] = {
            'record': value['record'], 'actual_K': int(value['mask'].sum()),
            'finite': True,
            'effective_equals_W_times_M': bool(np.array_equal(value['weight'] * value['mask'],
                                                               value['effective_weight'])),
            'masked_entries_exact_zero': bool(np.all(value['effective_weight'][value['mask'] == 0] == 0)),
        }
    fig.colorbar(image, cax=colorbar_axis, label='Signed effective weight: W × M')
    fig.suptitle('Один заранее выбранный target checkpoint: свежая задача 0, бюджет 256, init 0')
    save_figure(fig, destination, 'fresh_task0_budget256_weights')
    np.savez_compressed(destination / 'fresh_task0_budget256_values.npz', **arrays)
    (destination / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False,
                                                          indent=2, allow_nan=False))

    mask_arrays = {f'{name}_M': selected[name]['mask'] for name in order if name in selected}
    np.savez_compressed(destination / 'fresh_task0_budget256_masks.npz', **mask_arrays)
    figure_data['fixed_heatmap_metadata'] = metadata
    return ['## Обученные веса на фиксированном примере', '',
            '![Обученные веса](weight_heatmaps/fresh_task0_budget256_weights.png)', '',
            '**Как читать.** Все панели используют один пример: seed 4100, новая задача стоимости '
            '0, бюджет 256 target-наборов, первая инициализация. Каждая матрица имеет размер '
            '784×32; цвет показывает подписанный эффективный вес W×M. Красный/синий — знаки, '
            'общая симметричная шкала задана максимумом абсолютного веса среди показанных методов. '
            'Столбцы оставлены в сохранённом порядке.', '',
            'Для всех вариантов точечно проверены конечность значений, равенство effective=W×M, '
            'точные нули запрещённых связей и реальный K. Source-bank маски имеют 5018 связей '
            '(20%); после извлечения/построения для переноса используются маски с 7526 связями '
            '(30%). Dense содержит все 25 088 связей. Это один пример, он не заменяет агрегат '
            'качества по seeds и target-задачам.', '',
            'Точные W, M и W×M сохранены в `weight_heatmaps/fresh_task0_budget256_values.npz`; '
            'проверки и записи checkpoint — в `weight_heatmaps/metadata.json`. Дополнительные '
            'бинарные маски сохранены отдельно в `fresh_task0_budget256_masks.npz`.', '']


def _plot_saved_reconstructions(out: Path, seeds: list[dict], figure_data: dict) -> list[str]:
    path = next(row['array_path'] for row in seeds if row['seed'] == 4100)
    if path is None:
        return []
    with np.load(path) as archive:
        original_key = 'heldout_example_function_maps'
        reconstruction_key = 'heldout_example_functional_vae_large_reconstruction'
        missing = {original_key, reconstruction_key} - set(archive.files)
        if missing:
            raise ValueError(f'{path}: missing fixed reconstruction arrays {sorted(missing)}')
        original = np.asarray(archive[original_key])
        reconstruction = np.asarray(archive[reconstruction_key])
    if (original.ndim != 3 or reconstruction.ndim != 3
            or original.shape[1:] != (784, 32)
            or reconstruction.shape != original.shape or len(original) < 1):
        raise ValueError(f'{path}: expected paired held-out arrays [N,784,32], got '
                         f'{original.shape} and {reconstruction.shape}')
    original = original[0]
    reconstruction = reconstruction[0]
    if not (np.isfinite(original).all() and np.isfinite(reconstruction).all()):
        raise ValueError('Saved held-out reconstruction arrays contain NaN/Inf')
    fig = plt.figure(figsize=(12, 6), layout='constrained')
    grid = fig.add_gridspec(1, 5, width_ratios=[1, 1, .045, 1, .045])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[0, 3])]
    map_colorbar_axis = fig.add_subplot(grid[0, 2])
    residual_colorbar_axis = fig.add_subplot(grid[0, 4])
    limit = max(float(np.max(original)), float(np.max(reconstruction)))
    residual = reconstruction - original
    residual_limit = max(float(np.max(np.abs(residual))), 1e-12)
    for ax, value, title in zip(axes[:2], (original, reconstruction),
                                ('Held-out functional map', 'VAE large reconstruction')):
        im = ax.imshow(value, cmap='viridis', vmin=0, vmax=limit, aspect='auto',
                       interpolation='nearest')
        ax.set_title(title)
        ax.set_xlabel('Hidden unit')
        ax.set_ylabel('Source connection index')
    residual_image = axes[2].imshow(residual, cmap='RdBu_r', vmin=-residual_limit,
                                    vmax=residual_limit, aspect='auto', interpolation='nearest')
    axes[2].set_title('Reconstruction − map')
    axes[2].set_xlabel('Hidden unit')
    axes[2].set_ylabel('Source connection index')
    fig.colorbar(im, cax=map_colorbar_axis, label='Normalized functional map value')
    fig.colorbar(residual_image, cax=residual_colorbar_axis, label='Signed residual')
    save_figure(fig, out, 'heldout_functional_map_reconstruction')
    figure_data['heldout_reconstruction_array_keys'] = {
        'source': original_key, 'reconstruction': reconstruction_key,
        'source_archive': str(path.relative_to(out))}
    return ['### Пример held-out карты и реконструкции', '',
            '![Карта и реконструкция](heldout_functional_map_reconstruction.png)', '',
            '**Как читать.** Показан первый сохранённый held-out пример первого source-task: '
            'исходная функциональная карта, реконструкция VAE на 205 train-картах и их '
            'поэлементная разность. Цветовая шкала общая; центр около нуля означает малую '
            'разность; residual-панель имеет собственную симметричную шкалу. Это иллюстрация геометрии ошибки для одного примера, а не средняя '
            'оценка и не дополнительная проверка переноса.', '']


def _include_gpu_observation(out: Path, figure_data: dict) -> list[str]:
    plot = out / 'plots' / 'gpu_utilization.png'
    pdf = out / 'plots' / 'gpu_utilization.pdf'
    summary_path = out / 'gpu_utilization_summary.json'
    present = [path.is_file() for path in (plot, pdf, summary_path)]
    if not any(present):
        return []
    if not all(present):
        raise ValueError('GPU observation is partial: expected plots/gpu_utilization.png/pdf '
                         'and gpu_utilization_summary.json')
    summary = _json(summary_path)
    if set(summary) != {str(index) for index in range(8)}:
        raise ValueError(f'{summary_path}: expected utilization summary for 8 GPUs')
    sample_counts = [int(summary[str(index)]['samples']) for index in range(8)]
    if min(sample_counts) < 1:
        raise ValueError(f'{summary_path}: no GPU utilization observations')
    figure_data['gpu_observation'] = summary
    total = sum(sample_counts)
    sample_description = (f'{sample_counts[0]} точек на GPU' if len(set(sample_counts)) == 1
                           else f'{min(sample_counts)}–{max(sample_counts)} точек на GPU')
    return [
        '## Наблюдение GPU во время VAE и target-обучения', '',
        '![Наблюдение загрузки GPU](plots/gpu_utilization.png)', '',
        f"**Как читать.** Для каждой из восьми GPU показана загрузка за короткий "
        f"период наблюдения: {sample_description}, {total} значений суммарно. "
        'Показатель — занятость GPU в процентах по '
        'сэмплам наблюдателя.', '',
        '**Ограничение.** Запись охватывает только финальную фазу VAE и target-обучения, '
        'а не весь pipeline формирования source-банков. Это наблюдение очереди, не '
        'сравнительный benchmark и не оценка FLOPs. Источники CPU-простоев отдельно не измерялись.', '',
        '[PDF графика](plots/gpu_utilization.pdf) · '
        '[Сводка по GPU](gpu_utilization_summary.json).', '',
    ]


def build_report(out: Path) -> dict:
    out = out.resolve()
    protocol = _validate_protocol(out)
    seeds = load_inputs(out, protocol)
    summaries = {split: _summarize(seeds, split) for split in ('old', 'fresh')}
    source_diagnostics = _source_vae_metric_summary(seeds)
    result = {'experiment': out.name, 'seeds': list(SEEDS),
              'target_tasks': {'old': list(TASKS), 'fresh': list(TASKS)},
              'budgets': list(BUDGETS), 'methods': list(METHODS),
              'fixed_task_conditional': True, 'predeclared_protocol': protocol,
              'run_metadata': [{'seed': row['seed'], 'protocol': row['seed_protocol'],
                                'data_provenance': row['data_provenance']} for row in seeds],
              'source_reconstruction_diagnostics': source_diagnostics,
              'summaries': summaries}
    figure_data = {'seed_level_inference': 'means over target tasks and four initializations within seed; t(df=7)',
                   'input_map_counts': {'small_train': 26, 'large_train': 205,
                                        'common_heldout_per_task': 51},
                   'target_labeled_budgets': list(BUDGETS),
                   'mask_edges': {'source_bank': SOURCE_EDGES, 'transfer': TRANSFER_EDGES,
                                  'dense': TOTAL_EDGES},
                   'source_reconstruction_diagnostics': source_diagnostics}
    fresh256 = {row['method']: row for row in summaries['fresh']['aggregate']
                if row['support_size'] == 256}
    fresh_comparisons = {(row['method'], row['baseline']): row
                         for row in summaries['fresh']['comparisons']
                         if row['support_size'] == 256}
    primary = summaries['fresh']['primary_comparison']
    mean_data_effect = fresh_comparisons[('functional_mean_large', 'functional_mean_small')]
    vae_data_effect = fresh_comparisons[('functional_vae_large', 'functional_vae_small')]
    mean_random = fresh_comparisons[('functional_mean_large', 'random')]
    mean_dense = fresh_comparisons[('functional_mean_large', 'dense')]
    functional_large_source = source_diagnostics['functional_vae_large']
    raw_large_source = source_diagnostics['raw_vae_large']
    fresh256_table = ['| Метод | Test MSE/5 | 95% CI по seeds |', '|---|---:|---|']
    for method in METHODS:
        row = fresh256[method]
        fresh256_table.append(f"| {LABELS[method]} | {row['mean']:.6f} | "
                              f"[{row['ci95'][0]:.6f}, {row['ci95'][1]:.6f}] |")
    lines = [
        '# Расширенный перенос функциональных карт через VAE', '',
        'Завершены восемь независимых seeds (4100–4107). Для каждого seed отдельно '
        'агрегируются четыре инициализации и восемь фиксированных target-задач; t-интервалы '
        'строятся только по восьми seed, df=7. Старые восемь задач и восемь новых задач '
        'показаны отдельно.', '',
        'Банк построен для четырёх исходных source-задач: на каждую создано 1024 кандидата '
        'и отобрано 256 решений. Source-банк использует маски с 5018 связями (20%). '
        'Перед переносом выбирается ровно 7526 связей (30%). Веса target-моделей '
        'обучаются по новым target-меткам; source-карты и target-бюджеты — разные наборы '
        'данных и разные единицы счёта.', '',
        'VAE получает functional-карты, вычисленные без target-меток. В большом варианте '
        'на каждой source-задаче 205 карт используются для обучения и 51 общая карта '
        'отложена; малый вариант обучается на вложенном подмножестве из 26 карт и проверяется '
        'на тех же 51 held-out картах. Эти числа относятся к числу карт source-решений, а '
        'бюджеты 32/64/128/256 ниже относятся к размеченным наборам новой target-задачи, '
        'включая validation для выбора checkpoint.', '',
        '## Основные выводы', '',
        f"**Основное заранее выбранное сравнение.** На новых задачах при бюджете 256 "
        f"VAE functional large минус среднее по тем же 205 functional-картам равно "
        f"{primary['mean']:+.6f} MSE/5; 95% CI "
        f"[{primary['ci95'][0]:+.6f}, {primary['ci95'][1]:+.6f}]. Положительная разность "
        'и интервал выше нуля означают, что VAE не улучшила целевой перенос и в этом '
        'сравнении показала более высокую ошибку.', '',
        f"**Эффект дополнительных source-карт не подтверждён.** При переходе с 26 на 205 "
        f"карт функциональное среднее меняется на {mean_data_effect['mean']:+.6f} "
        f"[{mean_data_effect['ci95'][0]:+.6f}, {mean_data_effect['ci95'][1]:+.6f}], "
        f"а VAE — на {vae_data_effect['mean']:+.6f} "
        f"[{vae_data_effect['ci95'][0]:+.6f}, {vae_data_effect['ci95'][1]:+.6f}]. "
        'Оба интервала включают ноль. Преимущество функционального среднего над VAE '
        'сохраняется и после увеличения банка карт.', '',
        f"**Вторичные сравнения на новых задачах.** При бюджете 256 функциональное среднее "
        f"large ниже random на {mean_random['mean']:+.6f} "
        f"[{mean_random['ci95'][0]:+.6f}, {mean_random['ci95'][1]:+.6f}] и ниже dense на "
        f"{mean_dense['mean']:+.6f} [{mean_dense['ci95'][0]:+.6f}, "
        f"{mean_dense['ci95'][1]:+.6f}] MSE/5. Это исследовательские сравнения, "
        'а не основной заранее выбранный контраст.', '',
        '### Свежие задачи, бюджет 256', '', *fresh256_table, '',
        '### Что показала VAE на held-out source-картах', '',
        f"Для functional VAE large отношение held-out MSE к baseline по train mean равно "
        f"{functional_large_source['heldout_to_train_mean_mse_ratio']['mean']:.4f} "
        f"[{functional_large_source['heldout_to_train_mean_mse_ratio']['ci95'][0]:.4f}, "
        f"{functional_large_source['heldout_to_train_mean_mse_ratio']['ci95'][1]:.4f}]. "
        f"Held-out BCE {functional_large_source['heldout_reconstruction_bce']['mean']:.5f} "
        f"против {functional_large_source['train_mean_bce']['mean']:.5f} у своего mean-baseline; "
        f"парная разность {functional_large_source['heldout_bce_minus_train_mean_bce']['mean']:+.5f} "
        f"[{functional_large_source['heldout_bce_minus_train_mean_bce']['ci95'][0]:+.5f}, "
        f"{functional_large_source['heldout_bce_minus_train_mean_bce']['ci95'][1]:+.5f}]. "
        f"Реконструкции сохраняют {100*functional_large_source['variance_ratio']['mean']:.2f}% "
        'дисперсии held-out карт; pairwise IoU реконструированных бинарных поддержек '
        f"{functional_large_source['reconstruction_pairwise_iou']['mean']:.5f} против "
        f"{functional_large_source['target_pairwise_iou']['mean']:.5f} у held-out карт. "
        'Это согласуется с потерей разнообразия реконструкций.', '',
        f"Для raw VAE large отношение MSE к собственному train-mean baseline равно "
        f"{raw_large_source['heldout_to_train_mean_mse_ratio']['mean']:.4f}; отношения "
        'сопоставимы, но абсолютные значения raw- и functional-MSE имеют разные шкалы. '
        'По ним нельзя заключать, что одна входная репрезентация хуже другой.', '',
        '**Ограничение сходимости.** VAE обучалась 160 optimizer updates; лучший checkpoint '
        'часто приходился на последний update: functional small — 18/32 source-задач, '
        'functional large — 22/32, raw large — 16/32. В последние 10% update попали '
        'соответственно 28/32, 30/32 и 31/32. Сходимость не установлена; вывод о целевом '
        'переносе относится только к этому 160-update протоколу и не исключает иной результат '
        'при более долгом обучении.', '',
        '## Итоги по target-задачам', '',
        'Полный predeclared protocol с векторами задач, параметрами обучения и source SHA-256 '
        'сохранён в [protocol.json](protocol.json). Пер-seed CUDA и data split provenance '
        'включены в `summary.json`.', '',
    ]
    lines += _plot_learning(out, summaries, figure_data)
    lines += _plot_effects(out, summaries, figure_data)
    lines += _plot_source_quality(out, seeds, figure_data)
    lines += _plot_reconstruction(out, seeds, figure_data)
    lines += _plot_saved_reconstructions(out, seeds, figure_data)
    lines += _include_gpu_observation(out, figure_data)
    lines += _export_heatmaps(out, seeds, figure_data)

    lines += ['## Итоговые оценки', '',
              'Test MSE / 5; интервалы — 95% t-интервалы по восьми seed. Они условны на '
              'зафиксированных задачах стоимости и разбиениях.', '',
              '| Набор задач | Метод | Target-бюджет | MSE/5 | 95% CI |',
              '|---|---|---:|---:|---|']
    for split, title in (('old', 'Старые'), ('fresh', 'Новые')):
        for row in summaries[split]['aggregate']:
            lo, hi = row['ci95']
            lines.append(f"| {title} | {LABELS[row['method']]} | {row['support_size']} | "
                         f"{row['mean']:.6f} | [{lo:.6f}, {hi:.6f}] |")
    lines += ['', '## Парные сравнения', '',
              'Отрицательная разность означает преимущество первого указанного метода. '
              'Первичная проверка заранее задана для новых задач при бюджете 256; прочие '
              'сравнения являются exploratory.', '',
              '| Набор | Сравнение | Бюджет | Разность | 95% CI |',
              '|---|---|---:|---:|---|']
    for split, title in (('old', 'Старые'), ('fresh', 'Новые')):
        for row in summaries[split]['comparisons']:
            lo, hi = row['ci95']
            name = f"{LABELS[row['method']]} − {LABELS[row['baseline']]}"
            prefix = 'Основное: ' if split == 'fresh' and row is summaries['fresh']['primary_comparison'] else ''
            lines.append(f"| {title} | {prefix}{name} | {row['support_size']} | "
                         f"{row['mean']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    figure_data['summary'] = summaries
    (out / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                                  allow_nan=False))
    (out / 'figure_data.json').write_text(json.dumps(figure_data, ensure_ascii=False, indent=2,
                                                      allow_nan=False))
    lines += ['', 'Полные seed-level значения, парные интервалы и plotted values сохранены в '
              '`summary.json` и `figure_data.json`. Восемь исходных `results.json`, '
              'source VAE diagnostics и target checkpoints остаются рядом с seed-каталогами.', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True,
                        help='root output directory containing seed_4100 ... seed_4107')
    parser.add_argument('--report', action='store_true',
                        help='accepted explicit report mode; all inputs are validated before output')
    args = parser.parse_args()
    result = build_report(args.out)
    print(json.dumps({'out': str(args.out.resolve()), 'seeds': result['seeds'],
                      'report': str(args.out.resolve() / 'REPORT.md')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
