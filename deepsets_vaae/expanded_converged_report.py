"""Strict report for the convergence-based functional VAE repeat.

This report is deliberately separate from ``expanded_functional_report``: it
requires all 96 source-only VAE fits to satisfy the declared plateau rule and
does not carry forward the earlier fresh-task contrast as confirmatory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import expanded_functional_report as control


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'outputs/deepsets_vaae/20261001_converged_functional_vae'
CONTROL_OUT = ROOT / 'outputs/deepsets_vaae/20261001_expanded_functional_vae'
SEEDS = tuple(range(4100, 4108))
TASKS = tuple(range(8))
SOURCE_TASKS = tuple(range(4))
BUDGETS = (32, 64, 128, 256)
METHODS = control.METHODS
VAE_FAMILIES = ('functional_vae_small', 'functional_vae_large', 'raw_vae_large')
SOURCE_EDGES = 5018
TRANSFER_EDGES = 7526
TOTAL_EDGES = 784 * 32
F = 784 * 32


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f'{name}.png', dpi=160, bbox_inches='tight')
    fig.savefig(out / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def _expected_fit_criteria(max_steps: int) -> dict:
    return {
        'min_steps': 1000, 'max_steps': max_steps, 'window_updates': 200,
        'evaluation_interval_updates': 10, 'relative_window_tolerance': .001,
        'validation_improvement_tolerance': .0001,
        'validation_improvement_patience_updates': 400,
        'required_consecutive_eligible_checks': 3,
        'objectives_required': ['stochastic_train', 'deterministic_train', 'validation'],
        'linear_trend_drift_required': True,
    }


def _validate_protocol(out: Path) -> tuple[dict, dict]:
    path = out / 'protocol.json'
    if not path.is_file():
        raise ValueError(f'Missing {path}')
    protocol = _json(path)
    original_path = CONTROL_OUT / 'protocol.json'
    if not original_path.is_file():
        raise ValueError(f'Missing 160-update control protocol: {original_path}')
    original = _json(original_path)
    expected = {
        'seeds': list(SEEDS), 'source_tasks': 4, 'source_candidates': 1024,
        'source_keep': 256, 'source_density': .2, 'target_density': .3,
        'target_edges': TRANSFER_EDGES, 'vae_validation_maps': 51,
        'support_sizes': list(BUDGETS), 'methods': list(METHODS),
        'vae_families': {
            'functional_vae_small': {'train_maps': 26, 'input': 'function_maps'},
            'functional_vae_large': {'train_maps': 205, 'input': 'function_maps'},
            'raw_vae_large': {'train_maps': 205, 'input': 'raw_maps'},
        },
        'convergence': {
            'minimum_epochs': 1000, 'maximum_epochs': 20000,
            'window_epochs': 200, 'plateau_tolerance': .001,
            'patience_epochs': 400,
            'validation_improvement_tolerance': .0001,
            'requires_train_and_validation_plateau': True,
            'max_epochs_is_not_convergence': True,
        },
    }
    bad = {key: (protocol.get(key), value) for key, value in expected.items()
           if protocol.get(key) != value}
    if bad:
        raise ValueError(f'{path}: protocol mismatch: {bad}')
    if protocol.get('comparison_status', {}).get('confirmatory_claim') is not False:
        raise ValueError(f'{path}: convergence repeat must be exploratory, not confirmatory')
    if protocol.get('comparison_status', {}).get('classification') != \
            'all comparisons in this convergence repeat are exploratory':
        raise ValueError(f'{path}: comparison status does not mark all results exploratory')
    if protocol.get('original_protocol_sha256') != _sha(original_path):
        raise ValueError(f'{path}: original source-control protocol hash mismatch')
    if protocol.get('task_vectors') != original.get('task_vectors'):
        raise ValueError(f'{path}: target/source task vectors differ from the 160-update control')
    if protocol.get('target_eval', {}).get('steps') != original.get('eval_steps') or \
            protocol.get('target_eval', {}).get('test_sets') != original.get('test_sets') or \
            protocol.get('target_eval', {}).get('batch_conditions') != original.get('batch_conditions') or \
            protocol.get('target_eval', {}).get('kernel_mode') != original.get('kernel_mode') or \
            protocol.get('target_eval', {}).get('initialization_reference_models') != \
            original.get('initialization_reference_models'):
        raise ValueError(f'{path}: target evaluation settings differ from the control')
    if protocol.get('target_eval', {}).get('old_seed_offset') != 100000 or \
            protocol.get('target_eval', {}).get('fresh_seed_offset') != 300000:
        raise ValueError(f'{path}: target evaluation seeds are not matched to the control')
    for source_file in ('core.py', 'followup_batched_eval.py', 'followup_common.py'):
        if protocol.get('source_sha256', {}).get(source_file) != \
                original.get('source_sha256', {}).get(source_file):
            raise ValueError(f'{path}: target data/evaluation code changed for {source_file}')
    control_spec = control._validate_protocol(CONTROL_OUT)
    if control_spec.get('vae_epochs') != 160:
        raise ValueError('The comparison root is not the frozen 160-update control')
    return protocol, original


def _validate_plateau_report(report: dict, *, family: str, task: int,
                             seed: int) -> None:
    label = f'seed={seed}/{family}/task={task}'
    if report.get('family') != family or int(report.get('task', -1)) != task:
        raise ValueError(f'{label}: fit report identity mismatch')
    if report.get('converged') is not True or report.get('convergence_status') != 'converged':
        raise ValueError(f'{label}: did not converge')
    try:
        stop = int(report['stop_step'])
        best = int(report['best_step'])
        criteria = report['convergence_criteria']
        metrics = report['convergence_metrics']
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'{label}: missing convergence fields') from exc
    try:
        cap = int(report['max_steps_cap'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'{label}: missing maximum-step cap') from exc
    if not (1000 <= stop <= min(cap, 20000)) or not (0 <= best <= stop) or stop % 10:
        raise ValueError(f'{label}: invalid best/stop steps ({best}, {stop})')
    expected_criteria = _expected_fit_criteria(cap)
    if criteria != expected_criteria:
        raise ValueError(f'{label}: per-fit convergence criteria differ: {criteria}')
    if int(metrics.get('window_updates', -1)) != 200 or \
            int(metrics.get('evaluation_interval_updates', -1)) != 10 or \
            int(metrics.get('plateau_checks_consecutive', -1)) < 3 or \
            metrics.get('eligible') is not True or \
            metrics.get('plateau_objectives_passed') is not True or \
            metrics.get('validation_improvement_patience_passed') is not True or \
            int(metrics.get('updates_since_significant_validation_improvement', -1)) < 400:
        raise ValueError(f'{label}: final convergence check did not pass every criterion')
    fields = (
        'stochastic_train_relative_window_change',
        'deterministic_train_relative_window_change', 'validation_relative_window_change',
        'stochastic_train_linear_trend_relative_drift',
        'deterministic_train_linear_trend_relative_drift',
        'validation_linear_trend_relative_drift',
    )
    for field in fields:
        value = metrics.get(field)
        if value is None or not np.isfinite(float(value)) or float(value) > .001:
            raise ValueError(f'{label}: plateau metric {field}={value} exceeds .001')


def _curve_path(seed_folder: Path, fit_report: dict, family: str, task: int) -> Path:
    expected = seed_folder / 'functional' / 'fits' / family / f'task_{task}' / 'loss_curves.json'
    stored = Path(str(fit_report.get('loss_curve_path', '')))
    if not stored.is_absolute():
        stored = (seed_folder / 'functional' / stored).resolve()
    else:
        stored = stored.resolve()
    if stored != expected.resolve():
        raise ValueError(f'{expected}: fit report loss_curve_path points elsewhere: {stored}')
    if not stored.is_file():
        raise ValueError(f'{seed_folder}: missing full loss curve {stored}')
    return stored


def _load_curve(path: Path, report: dict, family: str, task: int, seed: int) -> dict:
    payload = _json(path)
    stop = int(report['stop_step'])
    stochastic = np.asarray(payload.get('stochastic_train_loss_by_update', []), dtype=float)
    evaluations = payload.get('evaluations')
    if payload.get('objective') != report.get('objective') or \
            payload.get('deterministic_objective') != report.get('deterministic_objective') or \
            int(payload.get('evaluation_interval_updates', -1)) != 10:
        raise ValueError(f'{path}: curve metadata differs from the fit report')
    if stochastic.shape != (stop,) or not np.isfinite(stochastic).all():
        raise ValueError(f'{path}: expected {stop} finite stochastic updates, got {stochastic.shape}')
    if not isinstance(evaluations, list) or len(evaluations) != stop // 10 + 1:
        raise ValueError(f'{path}: expected evaluations at step 0 and every 10 updates to {stop}')
    steps, train, valid = [], [], []
    for index, row in enumerate(evaluations):
        step = int(row.get('step', -1))
        if step != index * 10:
            raise ValueError(f'{path}: non-regular deterministic evaluation step {step}')
        steps.append(step)
        train.append(float(row['deterministic_train_objective']))
        valid.append(float(row['validation_objective']))
    if not (np.isfinite(train).all() and np.isfinite(valid).all()):
        raise ValueError(f'{path}: non-finite deterministic train/validation objective')
    if steps[-1] != stop or int(report['best_step']) not in steps:
        raise ValueError(f'{path}: curve lacks stop or best-checkpoint step')
    final = evaluations[-1].get('convergence_metrics')
    if final != report.get('convergence_metrics'):
        raise ValueError(f'{path}: final curve convergence check differs from fit report')
    for key, actual in (
            ('last_validation_objective', valid[-1]),
            ('last_deterministic_train_objective', train[-1]),
            ('last_stochastic_train_objective', stochastic[-1])):
        if not np.isclose(float(report[key]), float(actual), rtol=1e-6, atol=1e-7):
            raise ValueError(f'{path}: {key} differs between report and full curve')
    return {'path': path, 'stochastic': stochastic, 'steps': np.asarray(steps),
            'train': np.asarray(train), 'validation': np.asarray(valid),
            'stop_step': stop, 'best_step': int(report['best_step']),
            'best_validation_objective': float(report['best_validation_objective'])}


def _validate_fit_outputs(folder: Path, seed: int) -> tuple[dict, list[dict]]:
    diagnostics_path = folder / 'functional' / 'functional_vae_diagnostics.json'
    arrays_path = folder / 'functional' / 'functional_vae_arrays.npz'
    if not diagnostics_path.is_file() or not arrays_path.is_file():
        raise ValueError(f'{folder}: missing VAE diagnostics or source map arrays')
    diagnostics = _json(diagnostics_path)
    if diagnostics.get('converged') is not True or \
            diagnostics.get('convergence_status') != 'all_12_fits_converged':
        raise ValueError(f'{diagnostics_path}: not all source-only VAE fits converged')
    if int(diagnostics.get('experiment_seed', -1)) != seed or \
            int(diagnostics.get('seed', -1)) != seed + 50000:
        raise ValueError(f'{diagnostics_path}: source/VAE seed mismatch')
    if diagnostics.get('target_labels_used') is not False or int(diagnostics.get('k', -1)) != TRANSFER_EDGES:
        raise ValueError(f'{diagnostics_path}: target-label leakage or incorrect transfer K')
    if diagnostics.get('shape') != [784, 32] or diagnostics.get('methods') != list(METHODS) or \
            int(diagnostics.get('source_train_maps_per_task', -1)) != 205 or \
            int(diagnostics.get('small_train_maps_per_task', -1)) != 26 or \
            int(diagnostics.get('validation_maps_per_task', -1)) != 51 or \
            diagnostics.get('source_tasks') != list(SOURCE_TASKS):
        raise ValueError(f'{diagnostics_path}: source-map shapes/counts/tasks differ from protocol')
    reports = diagnostics.get('vae')
    if reports is None or reports != diagnostics.get('families') or \
            ('fit_reports' in diagnostics and reports != diagnostics['fit_reports']):
        raise ValueError(f'{diagnostics_path}: duplicated fit report maps disagree')
    if set(reports or {}) != set(VAE_FAMILIES):
        raise ValueError(f'{diagnostics_path}: expected exactly three VAE families')
    curves = []
    for family in VAE_FAMILIES:
        fit_rows = reports[family]
        if not isinstance(fit_rows, list) or len(fit_rows) != 4:
            raise ValueError(f'{diagnostics_path}: {family} must contain 4 source-task fits')
        by_task = {int(row.get('task', -1)): row for row in fit_rows}
        if set(by_task) != set(SOURCE_TASKS):
            raise ValueError(f'{diagnostics_path}: {family} task coverage is not 0..3')
        for task in SOURCE_TASKS:
            fit_report = by_task[task]
            _validate_plateau_report(fit_report, family=family, task=task, seed=seed)
            expected_train = 26 if family == 'functional_vae_small' else 205
            if int(fit_report.get('train_maps', -1)) != expected_train or \
                    int(fit_report.get('validation_maps', -1)) != 51:
                raise ValueError(f'{diagnostics_path}: {family}/task={task} source map counts mismatch')
            path = _curve_path(folder, fit_report, family, task)
            curve = _load_curve(path, fit_report, family, task, seed)
            curve.update({'family': family, 'task': task, 'seed': seed,
                          'report': fit_report})
            curves.append(curve)
    with np.load(arrays_path) as archive:
        required = {
            'heldout_example_function_maps',
            'heldout_example_functional_vae_large_reconstruction',
            'heldout_example_raw_maps',
            'heldout_example_raw_vae_large_reconstruction',
        }
        if not required.issubset(archive.files):
            raise ValueError(f'{arrays_path}: missing exact held-out map/reconstruction pair '
                             f'{sorted(required-set(archive.files))}')
        for left, right in (
                ('heldout_example_function_maps',
                 'heldout_example_functional_vae_large_reconstruction'),
                ('heldout_example_raw_maps', 'heldout_example_raw_vae_large_reconstruction')):
            a, b = np.asarray(archive[left]), np.asarray(archive[right])
            if a.ndim != 3 or a.shape[1:] != (784, 32) or b.shape != a.shape or len(a) < 1:
                raise ValueError(f'{arrays_path}: invalid paired shapes {left}={a.shape}, {right}={b.shape}')
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                raise ValueError(f'{arrays_path}: held-out arrays contain NaN/Inf')
    return diagnostics, curves


def _validate_target_seed(folder: Path, seed: int, protocol: dict,
                          control_seed: dict) -> dict:
    if not (folder / 'COMPLETE').is_file():
        raise ValueError(f'{folder}: missing seed COMPLETE marker')
    for name in ('results.json', 'masks.pt', 'protocol.json', 'data_provenance.json'):
        if not (folder / name).is_file():
            raise ValueError(f'{folder}: missing {name}')
    seed_protocol = _json(folder / 'protocol.json')
    if int(seed_protocol.get('seed', -1)) != seed:
        raise ValueError(f'{folder}: seed protocol mismatch')
    if {key: value for key, value in seed_protocol.items()
            if key not in ('seed', 'cuda_visible_devices')} != protocol:
        raise ValueError(f'{folder}: seed protocol differs from root')
    provenance = _json(folder / 'data_provenance.json')
    if provenance != control_seed['data_provenance']:
        raise ValueError(f'{folder}: target/source row split provenance differs from control')
    for task in SOURCE_TASKS:
        bank = folder / f'bank_{task}.pt'
        expected = CONTROL_OUT / f'seed_{seed}' / f'bank_{task}.pt'
        if not bank.is_symlink() or bank.resolve() != expected.resolve():
            raise ValueError(f'{bank}: source bank is not the frozen original source bank symlink')
    payload = _json(folder / 'results.json')
    if int(payload.get('seed', -1)) != seed:
        raise ValueError(f'{folder}/results.json: seed mismatch')
    old = control._validate_records(payload.get('records', []), 'old', seed)
    fresh = control._validate_records(payload.get('fresh_records', []), 'fresh', seed)
    masks = torch.load(folder / 'masks.pt', map_location='cpu', weights_only=True)
    if tuple(masks) != tuple(METHODS):
        raise ValueError(f'{folder}/masks.pt: method order mismatch')
    for method in METHODS:
        value = torch.as_tensor(masks[method])
        if tuple(value.shape) != (4, 784, 32) or not bool(torch.isfinite(value).all()) or \
                not bool(torch.all((value == 0) | (value == 1))):
            raise ValueError(f'{folder}/masks.pt/{method}: invalid mask')
        expected_k = TOTAL_EDGES if method == 'dense' else TRANSFER_EDGES
        if value.sum((-1, -2)).tolist() != [expected_k] * 4:
            raise ValueError(f'{folder}/masks.pt/{method}: expected {expected_k} edges each')
        if method in ('functional_mean_small', 'functional_mean_large', 'random', 'dense'):
            prior = torch.as_tensor(control_seed['masks'][method])
            if not torch.equal(value, prior):
                raise ValueError(f'{folder}/masks.pt/{method}: source-control mask changed')
    diagnostics, curves = _validate_fit_outputs(folder, seed)
    expected_weights = {f'target_task{task}_budget{budget}.pt'
                        for task in TASKS for budget in BUDGETS}
    for split, path in (('old', folder / 'weights'), ('fresh', folder / 'fresh_weights')):
        present = {item.name for item in path.glob('target_task*_budget*.pt')} if path.is_dir() else set()
        if present != expected_weights:
            raise ValueError(f'{folder}/{split}: target checkpoints incomplete; '
                             f'missing {sorted(expected_weights-present)[:5]}')
    return {'seed': seed, 'folder': folder, 'old': old, 'fresh': fresh,
            'raw_records': {'old': payload['records'], 'fresh': payload['fresh_records']},
            'masks': masks, 'diagnostics': diagnostics, 'curves': curves,
            'array_path': folder / 'functional' / 'functional_vae_arrays.npz',
            'seed_protocol': seed_protocol, 'data_provenance': provenance}


def _load_all(out: Path, protocol: dict) -> tuple[list[dict], list[dict]]:
    if not (out / 'COMPLETE').is_file():
        raise ValueError(f'{out}: missing root COMPLETE marker')
    original_protocol = control._validate_protocol(CONTROL_OUT)
    control_seeds = control.load_inputs(CONTROL_OUT, original_protocol)
    if len(control_seeds) != len(SEEDS):
        raise ValueError('160-update control does not have all eight seeds')
    for prior in control_seeds:
        prior_result = _json(prior['folder'] / 'results.json')
        prior['raw_records'] = {'old': prior_result['records'],
                                'fresh': prior_result['fresh_records']}
        prior['masks'] = torch.load(prior['folder'] / 'masks.pt', map_location='cpu',
                                   weights_only=True)
    control_by_seed = {row['seed']: row for row in control_seeds}
    folders = [out / f'seed_{seed}' for seed in SEEDS]
    missing = [str(path) for path in folders if not path.is_dir()]
    if missing:
        raise ValueError(f'{out}: missing seed directories {missing}')
    seeds = [_validate_target_seed(folder, seed, protocol, control_by_seed[seed])
             for folder, seed in zip(folders, SEEDS)]
    if sum(len(seed['curves']) for seed in seeds) != 96:
        raise ValueError('Expected exactly 96 complete VAE source-task fits')
    return seeds, control_seeds


def _aggregate(rows: list[dict], split: str) -> dict:
    aggregate, paired = [], []
    for method in METHODS:
        for budget in BUDGETS:
            values = [row[split][(method, budget)] for row in rows]
            aggregate.append({'method': method, 'support_size': budget,
                              **control.interval(values)})
    for budget in BUDGETS:
        for index, method in enumerate(METHODS):
            for baseline in METHODS[index + 1:]:
                values = [row[split][(method, budget)] - row[split][(baseline, budget)]
                          for row in rows]
                paired.append({'method': method, 'baseline': baseline,
                               'support_size': budget,
                               'direction': 'negative favors the first method',
                               **control.interval(values)})
    return {'aggregate': aggregate, 'all_pairwise_comparisons': paired,
            'inference_unit': 'mean across 8 fixed target tasks and 4 initializations per seed; t interval, df=7',
            'comparison_status': 'exploratory; no confirmatory comparison in this repeat'}


def _matched_control_differences(seeds: list[dict], control_seeds: list[dict]) -> dict:
    result = {}
    controls_by_seed = {row['seed']: row for row in control_seeds}
    for split in ('old', 'fresh'):
        rows = []
        for row in seeds:
            old_rows = controls_by_seed[row['seed']]['raw_records'][split]
            new_rows = row['raw_records'][split]
            key = lambda item: (int(item['task']), int(item['support_size']),
                                str(item['method']), int(item['init']))
            old = {key(item): float(item['mse']) for item in old_rows}
            new = {key(item): float(item['mse']) for item in new_rows}
            if old.keys() != new.keys():
                raise ValueError(f'{split}/seed={row["seed"]}: 160/converged target cells not paired')
            contrasts = {}
            for method in METHODS:
                for budget in BUDGETS:
                    cells = [new[cell] - old[cell] for cell in old
                             if cell[1] == budget and cell[2] == method]
                    if len(cells) != 8 * 4:
                        raise ValueError(f'{split}/seed={row["seed"]}: incomplete paired cells')
                    contrasts[(method, budget)] = float(np.mean(cells))
            rows.append(contrasts)
        family_summary = []
        for method in VAE_FAMILIES:
            for budget in BUDGETS:
                values = [row[(method, budget)] for row in rows]
                family_summary.append({'method': method, 'support_size': budget,
                                       **control.interval(values)})
        result[split] = {
            'all_methods': [
                {'method': method, 'support_size': budget,
                 **control.interval([row[(method, budget)] for row in rows])}
                for method in METHODS for budget in BUDGETS],
            'vae_only': family_summary,
            'seed_values': [{f'{method}:{budget}': row[(method, budget)]
                             for method in METHODS for budget in BUDGETS} for row in rows],
            'inference_unit': 'paired converged minus 160-update test MSE, averaged over 8 tasks and 4 inits within seed; exploratory, t(df=7)',
        }
    return result


def _contrast_row(summary: dict, method: str, baseline: str, budget: int) -> dict:
    direct = next((row for row in summary['all_pairwise_comparisons']
                   if row['method'] == method and row['baseline'] == baseline
                   and row['support_size'] == budget), None)
    if direct is not None:
        return direct
    reverse = next((row for row in summary['all_pairwise_comparisons']
                    if row['method'] == baseline and row['baseline'] == method
                    and row['support_size'] == budget), None)
    if reverse is None:
        raise ValueError(f'Missing exploratory contrast: {method} − {baseline} at {budget}')
    return {**reverse, 'method': method, 'baseline': baseline,
            'direction': 'negative favors the first method', 'mean': -reverse['mean'],
            'ci95': [-reverse['ci95'][1], -reverse['ci95'][0]],
            'repeat_values': [-value for value in reverse['repeat_values']]}


def _plot_target_learning(out: Path, summaries: dict, figure_data: dict) -> list[str]:
    colors = dict(zip(METHODS, plt.cm.tab10.colors[:len(METHODS)]))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, split, title in zip(axes, ('old', 'fresh'),
                                ('Ранее использованные 8 target-задач', 'Повторно проверенные 8 target-задач')):
        for method in METHODS:
            data = sorted((item for item in summaries[split]['aggregate']
                           if item['method'] == method), key=lambda item: item['support_size'])
            x = np.asarray([item['support_size'] for item in data])
            y = np.asarray([item['mean'] for item in data])
            lo = np.asarray([item['ci95'][0] for item in data])
            hi = np.asarray([item['ci95'][1] for item in data])
            ax.plot(x, y, marker='o', label=control.LABELS[method], color=colors[method])
            ax.fill_between(x, lo, hi, color=colors[method], alpha=.10)
        ax.set_xscale('log', base=2)
        ax.set_xticks(BUDGETS, [str(value) for value in BUDGETS])
        ax.set_xlabel('Размер размеченной выборки target-задачи')
        ax.set_title(title)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Test MSE / 5 (меньше — лучше)')
    axes[1].legend(fontsize=8, loc='best')
    fig.tight_layout()
    _save(fig, out, 'target_learning_curves')
    figure_data['target_learning_curves'] = summaries
    return ['## Результаты target-переноса', '',
            '![Кривые target-переноса](target_learning_curves.png)', '',
            '**Как читать.** По горизонтали указан бюджет целевых меток 32/64/128/256; по вертикали — test MSE/5. '
            'Новые наборы target-меток и пять наборов на test относятся к target-задачам; 26 и 205 карт — '
            'это количество source-карт для VAE и они не входят в эти бюджеты. Точка усредняет задачи и четыре '
            'инициализации внутри seed, а 95% t-интервал строится по восьми seed (df=7). Все сравнения этого '
            'повтора исследовательские: восемь свежих cost-задач уже были просмотрены в 160-update отчёте.', '']


def _plot_effects(out: Path, summaries: dict, figure_data: dict) -> list[str]:
    contrasts = (
        ('functional_vae_large', 'functional_mean_large', 'VAE large − mean large'),
        ('functional_vae_small', 'functional_mean_small', 'VAE small − mean small'),
        ('functional_mean_large', 'functional_mean_small', 'Mean: 205 − 26 карт'),
        ('functional_vae_large', 'functional_vae_small', 'VAE: 205 − 26 карт'),
        ('functional_vae_large', 'random', 'VAE large − random'),
        ('functional_vae_large', 'dense', 'VAE large − dense'),
    )
    lines = ['## Исследовательские target-сравнения', '']
    for split, title in (('old', 'Ранее использованные target-задачи'),
                         ('fresh', 'Повторно использованные target-задачи')):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
        pair_rows = []
        for ax, (method, baseline, label) in zip(axes.flat, contrasts):
            rows = [item for item in summaries[split]['all_pairwise_comparisons']
                    if item['method'] == method and item['baseline'] == baseline]
            rows.sort(key=lambda item: item['support_size'])
            if not rows:
                rows = [item for item in summaries[split]['all_pairwise_comparisons']
                        if item['method'] == baseline and item['baseline'] == method]
                rows = [{**item, 'method': method, 'baseline': baseline,
                         'direction': 'negative favors the first method', 'mean': -item['mean'],
                         'ci95': [-item['ci95'][1], -item['ci95'][0]]} for item in rows]
                rows.sort(key=lambda item: item['support_size'])
            pair_rows.extend(rows)
            x = np.asarray([item['support_size'] for item in rows])
            y = np.asarray([item['mean'] for item in rows])
            lo = np.asarray([item['ci95'][0] for item in rows])
            hi = np.asarray([item['ci95'][1] for item in rows])
            ax.errorbar(x, y, yerr=np.vstack((y - lo, hi - y)), marker='o', capsize=3)
            ax.axhline(0, color='black', lw=.8)
            ax.set_title(label, fontsize=10)
            ax.set_xscale('log', base=2)
            ax.set_xticks(BUDGETS, [str(value) for value in BUDGETS])
            ax.set_xlabel('Target-бюджет')
            ax.set_ylabel('Парная разность Test MSE / 5')
            ax.grid(alpha=.2)
        fig.suptitle(title + ' — все контрасты exploratory')
        fig.tight_layout()
        name = f'target_effects_{split}'
        _save(fig, out, name)
        figure_data[name] = pair_rows
        lines += [f'### {title}', '', f'![Target-эффекты: {title}]({name}.png)', '',
                  '**Как читать.** Показана парная разность первого метода и baseline на тех же seed, задачах, '
                  'бюджетах и инициализациях; отрицательное значение благоприятно первому методу. Интервалы '
                  'получены по средним различий внутри восьми seed (df=7), точечные, без коррекции за множественные '
                  'сравнения. Ни одно сравнение в converged-повторе не объявляется подтверждающим.', '']
    return lines


def _plot_control_delta(out: Path, comparison: dict, figure_data: dict) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    colors = dict(zip(VAE_FAMILIES, ('#1b9e77', '#d95f02', '#7570b3')))
    for ax, split, title in zip(axes, ('old', 'fresh'),
                                ('Ранее использованные target-задачи', 'Повторно использованные target-задачи')):
        for method in VAE_FAMILIES:
            rows = sorted((row for row in comparison[split]['vae_only']
                           if row['method'] == method), key=lambda row: row['support_size'])
            x = np.asarray([row['support_size'] for row in rows])
            y = np.asarray([row['mean'] for row in rows])
            lo = np.asarray([row['ci95'][0] for row in rows])
            hi = np.asarray([row['ci95'][1] for row in rows])
            ax.errorbar(x, y, yerr=np.vstack((y - lo, hi - y)), marker='o', capsize=3,
                        color=colors[method], label=control.LABELS[method])
        ax.axhline(0, color='black', lw=.8)
        ax.set_xscale('log', base=2)
        ax.set_xticks(BUDGETS, [str(value) for value in BUDGETS])
        ax.set_xlabel('Target-бюджет')
        ax.set_title(title)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Converged VAE − 160-update control, Test MSE / 5')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out, 'converged_vs_160_vae_target')
    figure_data['converged_vs_160_vae_target'] = comparison
    return ['## Сопоставление с 160-update VAE', '',
            '![Разность converged и 160-update результатов](converged_vs_160_vae_target.png)', '',
            '**Как читать.** На графике — разность Test MSE/5: converged-run минус 160-update контроль для трёх '
            'VAE-методов на одинаковых source-картах и splits, target cost-векторах, данных и seed-offsets тестовых '
            'выборок. Каждая точка усредняет те же восемь target-задач и четыре инициализации внутри seed; интервалы '
            'парные по восьми seed (df=7). Это сопоставление одного повторного анализа, а не новое подтверждение: '
            'target-задачи уже просматривались ранее. Отрицательные значения означают меньшую ошибку после обучения '
            'VAE до численного плато.', '']


def _plot_convergence_summary(out: Path, seeds: list[dict], figure_data: dict) -> list[str]:
    rows = []
    for seed in seeds:
        for curve in seed['curves']:
            rows.append({'seed': seed['seed'], 'family': curve['family'], 'task': curve['task'],
                         'stop_step': curve['stop_step'], 'best_step': curve['best_step'],
                         'best_validation_objective': curve['best_validation_objective']})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for family, color in zip(VAE_FAMILIES, ('#1b9e77', '#d95f02', '#7570b3')):
        subset = [row for row in rows if row['family'] == family]
        grouped = {seed: [item['stop_step'] for item in subset if item['seed'] == seed]
                   for seed in SEEDS}
        xs, ys = [], []
        for seed in SEEDS:
            for value in grouped[seed]:
                xs.append(seed)
                ys.append(value)
        axes[0].scatter(xs, ys, s=18, alpha=.55, color=color, label=control.LABELS[family])
        fit_values = [item['stop_step'] for item in subset]
        axes[1].hist(fit_values, bins=np.arange(1000, 20001, 500), alpha=.5,
                     color=color, label=control.LABELS[family])
    axes[0].axhline(20000, color='black', lw=.8, linestyle='--', label='Предел 20 000')
    axes[0].set_xlabel('Seed')
    axes[0].set_ylabel('Optimizer updates до плато')
    axes[0].set_title('Остановка каждой source-task VAE')
    axes[0].grid(alpha=.2)
    axes[1].set_xlabel('Optimizer updates до плато')
    axes[1].set_ylabel('Количество fits')
    axes[1].set_title('Распределение длительности обучения')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out, 'convergence_stop_steps')
    figure_data['convergence_stop_steps'] = rows
    return ['## Диагностика остановки по плато', '',
            '![Шаги до плато](convergence_stop_steps.png)', '',
            '**Как читать.** Каждая точка слева — одна source-задача одного семейства VAE (всего 96 fits); справа '
            'показано распределение optimizer updates до трёх последовательных проходов условия плато. Пунктир — '
            'hard cap 20 000 updates; статус `converged` подтверждается выполнением plateau-критериев на stop-step '
            '(в том числе при stop-step, равном cap), а сам cap без этих критериев не считается сходимостью. '
            'Checkpoint выбирается по минимальному validation objective (`best_step`), а `stop_step` показывает '
            'момент подтверждения плато. Числа не являются метрикой target-качества.', '']


def _plot_loss_curve(out: Path, curve: dict) -> str:
    family, task, seed = curve['family'], curve['task'], curve['seed']
    stop, best = curve['stop_step'], curve['best_step']
    updates = np.arange(1, stop + 1)
    stochastic = curve['stochastic'] / F
    eval_steps = curve['steps']
    train = curve['train'] / F
    valid = curve['validation'] / F
    zoom_width = min(stop, max(1000, int(np.ceil(stop * .2))))
    zoom_start = max(1, stop - zoom_width + 1)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.8), sharey=False)
    panels = (
        ('stochastic_train', 'Stochastic train ELBO/F', updates, stochastic, None),
        ('deterministic_train', 'Posterior-mean train ELBO/F', eval_steps, train, None),
        ('validation', 'Validation ELBO/F', eval_steps, valid, None),
    )
    for column, (_, title, x, y, _) in enumerate(panels):
        ax = axes[0, column]
        ax.plot(x, y, color=('#7570b3' if column == 0 else '#1b9e77' if column == 1 else '#d95f02'),
                linewidth=.65 if column == 0 else 1.2)
        ax.axvline(best, color='black', linestyle=':', linewidth=1, label=f'best={best}')
        ax.axvline(stop, color='red', linestyle='--', linewidth=1, label=f'stop={stop}')
        ax.set_title(title)
        ax.set_xlabel('Optimizer update')
        ax.set_ylabel('Objective / F')
        ax.grid(alpha=.2)
    axes[0, 2].legend(fontsize=8)
    for column, (_, title, x, y, _) in enumerate(panels):
        ax = axes[1, column]
        keep = x >= zoom_start
        ax.plot(x[keep], y[keep], color=('#7570b3' if column == 0 else '#1b9e77' if column == 1 else '#d95f02'),
                linewidth=.75 if column == 0 else 1.3)
        ax.axvline(best, color='black', linestyle=':', linewidth=1)
        ax.axvline(stop, color='red', linestyle='--', linewidth=1)
        ax.set_title(f'Последние updates: {zoom_start}–{stop}')
        ax.set_xlim(zoom_start, stop)
        ax.set_xlabel('Optimizer update')
        ax.set_ylabel('Objective / F')
        ax.grid(alpha=.2)
    fig.suptitle(f'{control.LABELS[family]} · seed {seed} · source task {task} · '
                 f'best={best}, stop={stop} updates', y=1.01)
    fig.tight_layout()
    name = f'seed_{seed}_{family}_task_{task}'
    _save(fig, out / 'loss_curves', name)
    return name


def _all_loss_plots(out: Path, seeds: list[dict], figure_data: dict) -> tuple[list[str], dict]:
    destination = out / 'loss_curves'
    destination.mkdir(parents=True, exist_ok=True)
    index = []
    lines = ['## Все кривые обучения: 96 source-only VAE fits', '',
             'Для каждой комбинации seed × семейство × source-task сохранена отдельная PNG- и PDF-фигура. '
             'Каждый рисунок показывает полный ход трёх контролируемых objective и последние updates крупно; '
             'вертикальные линии отмечают checkpoint с лучшим validation objective и остановку по плато.', '',
             '**Нормировка.** Обучаемая функция — полносессионная stochastic ELBO: сумма BCE-with-logits и '
             '0.1×KL, усреднённая по картам. На графиках показан этот objective, делённый на '
             f'F={F:,} входных координат на карту. Это только нормировка оси для сопоставимости, '
             'она не изменяет обучение или критерий остановки. Для validation и deterministic train показана '
             'ELBO при posterior mean; для stochastic train — фактически вычисленная loss каждого update.', '']
    for family in VAE_FAMILIES:
        lines += [f'### {control.LABELS[family]}', '',
                  '| Seed | Source task | Updates до плато | Лучший validation update | Графики |',
                  '|---:|---:|---:|---:|---|']
        family_curves = [curve for seed in seeds for curve in seed['curves']
                         if curve['family'] == family]
        for curve in sorted(family_curves, key=lambda value: (value['seed'], value['task'])):
            name = _plot_loss_curve(out, curve)
            report = curve['report']
            curve_index = {
                'seed': curve['seed'], 'family': family, 'task': curve['task'],
                'plot_png': f'loss_curves/{name}.png', 'plot_pdf': f'loss_curves/{name}.pdf',
                'loss_curve_json': str(curve['path'].relative_to(out)),
                'stop_step': curve['stop_step'], 'best_step': curve['best_step'],
                'best_validation_objective': curve['best_validation_objective'],
                'final_validation_objective': float(report['last_validation_objective']),
                'final_stochastic_train_objective': float(report['last_stochastic_train_objective']),
                'final_deterministic_train_objective': float(report['last_deterministic_train_objective']),
                'convergence_metrics': report['convergence_metrics'],
            }
            index.append(curve_index)
            lines.append(f"| {curve['seed']} | {curve['task']} | {curve['stop_step']} | "
                         f"{curve['best_step']} | [PNG]({curve_index['plot_png']}) · "
                         f"[PDF]({curve_index['plot_pdf']}) |")
        lines += ['', '**Подпись.** Строка каждой таблицы однозначно задаёт одну задачу и seed, а ссылки ведут к '
                  'её полным кривым и позднему увеличению. ELBO/F ниже означает меньшую функцию потерь; plateau '
                  'проверялся одновременно по stochastic train, deterministic train и validation: изменения средних '
                  'двух соседних 200-update окон и модуль нормированного линейного тренда должны быть не выше 0.001. '
                  'Значимое улучшение — новый validation minimum более чем на 0.0001 относительно предыдущего '
                  'minimum; затем должно пройти 400 updates без такого улучшения. Требуется три подряд eligible '
                  'проверки с интервалом 10 updates.', '']
    if len(index) != 96:
        raise ValueError(f'Expected exactly 96 loss plots; generated {len(index)}')
    (out / 'loss_curve_index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2,
                                                         allow_nan=False) + '\n', encoding='utf-8')
    figure_data['loss_curve_index'] = index
    return lines, index


def build_report(out: Path) -> dict:
    out = out.resolve()
    protocol, original_protocol = _validate_protocol(out)
    seeds, control_seeds = _load_all(out, protocol)
    summaries = {split: _aggregate(seeds, split) for split in ('old', 'fresh')}
    matched = _matched_control_differences(seeds, control_seeds)
    diagnostics = control._source_vae_metric_summary(seeds)
    control_diagnostics = control._source_vae_metric_summary(control_seeds)
    source_quality = []
    for current, prior in zip(seeds, control_seeds):
        per_task = []
        for task in SOURCE_TASKS:
            bank_path = current['folder'] / f'bank_{task}.pt'
            bank = torch.load(bank_path, map_location='cpu', weights_only=False)
            all_scores = np.asarray(bank.get('best_validation_normalized_mse', []), dtype=float)
            selected_scores = np.asarray(bank.get('validation_losses', []), dtype=float)
            selected = np.asarray(bank.get('selected_candidates', []), dtype=int)
            if all_scores.shape != (1024,) or selected_scores.shape != (256,) or \
                    selected.shape != (256,) or not np.isfinite(all_scores).all() or \
                    not np.isfinite(selected_scores).all() or not np.array_equal(selected_scores, all_scores[selected]):
                raise ValueError(f'{bank_path}: invalid or incomplete frozen source-bank quality arrays')
            if int(bank.get('edges_per_mask', -1)) != SOURCE_EDGES:
                raise ValueError(f'{bank_path}: source bank mask cardinality mismatch')
            per_task.append({'all': all_scores, 'selected': selected_scores})
        source_quality.append({'seed': current['seed'], 'folder': current['folder'],
                               'diagnostics': current['diagnostics'],
                               'diagnostics_path': current['folder'] / 'functional' / 'functional_vae_diagnostics.json',
                               'array_path': current['array_path'], 'source_quality': per_task})
    figure_data = {
        'experiment': out.name,
        'comparison_status': protocol['comparison_status'],
        'convergence': protocol['convergence'],
        'seed_level_inference': 'means over target tasks and four initializations within seed; t(df=7)',
        'input_map_counts': {'small_train': 26, 'large_train': 205, 'heldout_per_task': 51},
        'target_labeled_budgets': list(BUDGETS),
        'mask_edges': {'source_bank': SOURCE_EDGES, 'transfer': TRANSFER_EDGES,
                       'dense': TOTAL_EDGES},
        'source_reconstruction_converged': diagnostics,
        'source_reconstruction_160_control': control_diagnostics,
        'target_summaries': summaries,
        'matched_160_target_differences': matched,
    }
    fresh256 = {row['method']: row for row in summaries['fresh']['aggregate']
                if row['support_size'] == 256}
    exploratory_vae_mean = _contrast_row(summaries['fresh'], 'functional_vae_large',
                                         'functional_mean_large', 256)
    converged_delta = next(row for row in matched['fresh']['vae_only']
                           if row['method'] == 'functional_vae_large'
                           and row['support_size'] == 256)
    functional_source = diagnostics['functional_vae_large']
    lines = [
        '# Повтор функциональных VAE с проверкой сходимости', '',
        'Отчёт выпускается только после успешного plateau-статуса всех 96 VAE fits: восемь seeds × три семейства × '
        'четыре source-задачи. Для всех четырёх source-задач используется тот же зафиксированный банк из 1024 '
        'кандидатов с выбором 256 и те же source-only raw/functional карты и splits, что в 160-update контроле.', '',
        'Small functional VAE обучается на вложенных 26 функциональных source-картах, large functional VAE — на 205, '
        'raw VAE — на 205 raw-картах; для каждого семейства оставлены те же 51 held-out source-карт. Эти числа '
        'описывают обучение VAE. На target-задаче веса целевой сети обучаются по отдельным бюджетам разметки '
        '32/64/128/256; это разные данные и единицы счёта. Source-bank маски используют K=5018 связей (20%), '
        'а извлечение/перенос — K=7526 (30%).', '',
        '## Критерий остановки', '',
        'Для каждой VAE фиксируется stochastic train ELBO на каждом optimizer update, а deterministic objective '
        'для posterior mean на train и validation — каждые 10 updates. После минимум 1000 updates сравниваются '
        'средние соседних окон длиной 200 updates, а также абсолютный линейный дрейф за те же два окна. Все шесть '
        'относительных изменений/дрейфов (три objective × среднее и тренд) должны быть ≤0.001. Значимое улучшение '
        'validation — новый минимум более чем на 0.0001 относительно предыдущего минимума; после последнего такого '
        'улучшения должно пройти 400 updates. Условие должно '
        'выполниться на трёх последовательных проверках через 10 updates. Жёсткий предел — 20 000 updates; достижение '
        'cap само по себе не считается сходимостью. Target-метки не участвуют в этом критерии или извлечении масок.', '',
        'У всех 96 fits записан явный успешный статус и пройдены критерии plateau; stop-step не трактуется как '
        'успех без проверки метрик. Для извлечения масок используется checkpoint `best_step` с '
        'минимальной validation ELBO; `stop_step` — момент, когда подтверждено численное плато. Сходимость здесь '
        'означает плато выбранных objective на фиксированных картах и не является гарантией глобального минимума.', '',
        '## Интерпретация target-результатов', '',
        'Этот повтор не вводит новое подтверждающее сравнение. Свежие cost-задачи уже были просмотрены в предыдущем '
        '160-update отчёте, поэтому все оценки и контрасты ниже помечены как exploratory. Сопоставление с 160-update '
        'контролем использует те же task vectors, data splits, source-карты и seed-offsets target-тестов; это парная '
        'оценка изменения после более долгого обучения VAE, а не независимая репликация.', '',
        f"**Fresh target задачи, бюджет 256 (exploratory).** Функциональное среднее на 205 картах: "
        f"{fresh256['functional_mean_large']['mean']:.6f} [{fresh256['functional_mean_large']['ci95'][0]:.6f}, "
        f"{fresh256['functional_mean_large']['ci95'][1]:.6f}] MSE/5; converged functional VAE large: "
        f"{fresh256['functional_vae_large']['mean']:.6f} [{fresh256['functional_vae_large']['ci95'][0]:.6f}, "
        f"{fresh256['functional_vae_large']['ci95'][1]:.6f}]. Разность VAE − mean составляет "
        f"{exploratory_vae_mean['mean']:+.6f} [{exploratory_vae_mean['ci95'][0]:+.6f}, "
        f"{exploratory_vae_mean['ci95'][1]:+.6f}] MSE/5; статус сравнения — exploratory.", '',
        f"**Изменение относительно 160-update контроля.** На тех же fresh target задачах и бюджете 256 разность "
        f"converged VAE large − 160-update VAE large равна {converged_delta['mean']:+.6f} "
        f"[{converged_delta['ci95'][0]:+.6f}, {converged_delta['ci95'][1]:+.6f}] MSE/5. Это парное "
        'сопоставление условий повтора, а не новое независимое подтверждение.', '',
        f"**Held-out source карты.** Для functional VAE large отношение reconstruction MSE к собственному "
        f"train-mean baseline равно {functional_source['heldout_to_train_mean_mse_ratio']['mean']:.4f} "
        f"[{functional_source['heldout_to_train_mean_mse_ratio']['ci95'][0]:.4f}, "
        f"{functional_source['heldout_to_train_mean_mse_ratio']['ci95'][1]:.4f}]; реконструкция сохраняет "
        f"{100*functional_source['variance_ratio']['mean']:.2f}% held-out map variance. Источник-диагностика "
        'оценивает reconstruction, а не transfer quality.', '',
        'Все усреднения выполняются сначала по восьми target-задачам и четырём инициализациям в пределах seed, '
        'затем по восьми seeds. Интервалы — точечные t-интервалы 95% по восьми seed (df=7), условные на фиксированных '
        'target-задачах; поправки за множественные сравнения не применялись.', '',
        '## Восстановление source-карт', '',
        'Held-out source reconstruction оценивается относительно собственного train-mean baseline каждого семейства. '
        'Абсолютные MSE raw- и functional-карт имеют разные масштабы, поэтому сравниваются относительные показатели, '
        'а не их абсолютные MSE. IoU реконструированных бинарных поддержек близкий к 1 означает меньшую вариативность '
        'реконструкций и сам по себе не означает лучшую точность.', '',
    ]

    lines += _plot_target_learning(out, summaries, figure_data)
    lines += _plot_effects(out, summaries, figure_data)
    lines += _plot_control_delta(out, matched, figure_data)
    lines += _plot_convergence_summary(out, seeds, figure_data)
    loss_lines, loss_index = _all_loss_plots(out, seeds, figure_data)
    lines += loss_lines
    lines += control._plot_source_quality(out, source_quality, figure_data)
    lines += control._plot_reconstruction(out, seeds, figure_data)
    lines += control._plot_saved_reconstructions(out, seeds, figure_data)
    lines += control._export_heatmaps(out, seeds, figure_data)

    lines += ['## Сводка target test MSE/5', '',
              'Все значения и сравнения ниже имеют exploratory-статус.', '',
              '| Повтор target-задач | Метод | Target-бюджет | Среднее | 95% CI по seeds |',
              '|---|---|---:|---:|---|']
    for split, label in (('old', 'Ранее использованные'), ('fresh', 'Повторно использованные')):
        for row in summaries[split]['aggregate']:
            lo, hi = row['ci95']
            lines.append(f"| {label} | {control.LABELS[row['method']]} | {row['support_size']} | "
                         f"{row['mean']:.6f} | [{lo:.6f}, {hi:.6f}] |")
    lines += ['', '## Разности target test MSE/5 между методами', '',
              'Отрицательная разность означает меньшую ошибку у первого метода. Все строки exploratory.', '',
              '| Повтор | Сравнение | Бюджет | Разность | 95% CI |', '|---|---|---:|---:|---|']
    for split, label in (('old', 'Ранее использованные'), ('fresh', 'Повторно использованные')):
        for row in summaries[split]['all_pairwise_comparisons']:
            lo, hi = row['ci95']
            lines.append(f"| {label} | {control.LABELS[row['method']]} − "
                         f"{control.LABELS[row['baseline']]} | {row['support_size']} | "
                         f"{row['mean']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines += ['', '## Target-разности: converged-run минус 160-update control', '',
              'Это сравнение относится только к трём VAE-вариантам. Четыре source-control mask-метода '
              '(functional means, random, dense) проверены на точное совпадение с сохранённым 160-update контролем.', '',
              '| Набор задач | VAE вариант | Target-бюджет | Разность | 95% CI |',
              '|---|---|---:|---:|---|']
    for split, label in (('old', 'Ранее использованные'), ('fresh', 'Повторно использованные')):
        for row in matched[split]['vae_only']:
            lo, hi = row['ci95']
            lines.append(f"| {label} | {control.LABELS[row['method']]} | {row['support_size']} | "
                         f"{row['mean']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")

    report = {
        'experiment': out.name, 'seeds': list(SEEDS), 'source_tasks': list(SOURCE_TASKS),
        'target_tasks': {'old': list(TASKS), 'fresh': list(TASKS)},
        'target_budgets': list(BUDGETS), 'methods': list(METHODS),
        'all_comparisons_exploratory': True, 'converged_fit_count': 96,
        'predeclared_control_report': str((CONTROL_OUT / 'REPORT.md').resolve()),
        'protocol': protocol,
        'seed_metadata': [{'seed': row['seed'], 'protocol': row['seed_protocol'],
                           'data_provenance': row['data_provenance'],
                           'converged_fits': row['diagnostics']['vae']}
                          for row in seeds],
        'source_reconstruction_converged': diagnostics,
        'source_reconstruction_160_control': control_diagnostics,
        'target_summaries': summaries, 'matched_160_target_differences': matched,
        'loss_curve_index': loss_index,
    }
    (out / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                                  allow_nan=False) + '\n', encoding='utf-8')
    (out / 'figure_data.json').write_text(json.dumps(figure_data, ensure_ascii=False, indent=2,
                                                      allow_nan=False) + '\n', encoding='utf-8')
    lines += ['', '## Артефакты', '',
              'Протокол, curve index, per-seed fit criteria/metrics, маски, восстановленные карты, '
              'веса и численные plotted values сохранены рядом с отчётом. Все графики доступны как PNG и PDF; '
              'список 96 графиков с идентификаторами и шагами остановки — в `loss_curve_index.json`.', '']
    (out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    return report


def snapshot_source(out: Path) -> dict:
    sources = [Path(__file__).resolve(), Path(__file__).with_name('expanded_latent_report.py')]
    missing = [str(current) for current in sources if not current.is_file()]
    if missing:
        raise FileNotFoundError(f'Missing required report source modules: {missing}')
    records = {}
    for current in sources:
        snapshot = out / 'source_snapshot' / current.name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        content = current.read_bytes()
        if snapshot.exists() and snapshot.read_bytes() != content:
            raise ValueError(f'Report source snapshot already exists and differs: {snapshot}')
        snapshot.write_bytes(content)
        records[current.name] = {
            'source': str(current), 'snapshot': str(snapshot.relative_to(out)),
            'sha256': hashlib.sha256(content).hexdigest(),
        }
    (out / 'report_source_snapshot.json').write_text(json.dumps(records, indent=2) + '\n',
                                                    encoding='utf-8')
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--report', action='store_true', help='accepted for parity with the control report')
    parser.add_argument('--snapshot', action='store_true',
                        help='freeze report modules after all report appendices are final')
    args = parser.parse_args()
    report = build_report(args.out)
    snapshot = snapshot_source(args.out.resolve()) if args.snapshot else {}
    print(json.dumps({'experiment': report['experiment'],
                      'converged_fit_count': report['converged_fit_count'],
                      'all_comparisons_exploratory': report['all_comparisons_exploratory'],
                      'report_source_snapshot': snapshot}, ensure_ascii=False))


if __name__ == '__main__':
    main()
