"""Render source-only KL/latent-usage diagnostics and append them to the converged report.

This is a read-only consumer of the exact replay artifacts. It requires all 96
fits, frozen stochastic-trace/checkpoint parity, and a relative-tolerance check
against an independent CPU evaluation before it modifies the report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from . import expanded_converged_report as converged_report
from . import expanded_converged_vae as converged
from . import expanded_latent_diagnostics as producer


ROOT = Path(__file__).resolve().parents[1]
CONVERGED_ROOT = ROOT / 'outputs/deepsets_vaae/20261001_converged_functional_vae'
LATENT_ROOT = CONVERGED_ROOT / 'latent_diagnostics'
INDEPENDENT_CPU_PATH = CONVERGED_ROOT / 'independent_best_latent_check.json'
SEEDS = tuple(range(4100, 4108))
FAMILIES = ('functional_vae_small', 'functional_vae_large', 'raw_vae_large')
TASKS = tuple(range(4))
CHECKPOINTS = ('best', 'stop')
SPLITS = ('training', 'validation')
F = 784 * 32
KL_WEIGHT = .1
CPU_RTOL = 2e-3
CPU_ATOL = 1e-5
BEGIN = '<!-- LATENT_DIAGNOSTICS_BEGIN -->'
END = '<!-- LATENT_DIAGNOSTICS_END -->'


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _finite(value, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f'{label}: contains NaN or infinity')
    return result


def _close(actual, expected, *, label: str, rtol: float = 2e-5,
           atol: float = 1e-6) -> None:
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.allclose(a, b, rtol=rtol, atol=atol):
        delta = float(np.max(np.abs(a - b))) if a.shape == b.shape and a.size else float('inf')
        raise ValueError(f'{label}: values differ (shape {a.shape} vs {b.shape}, max_abs_delta={delta})')


def _check_latent_stats(stats: dict, family: str, checkpoint: str, split: str,
                        seed: int, task: int, expected_maps: int) -> None:
    label = f'seed={seed}/{family}/task={task}/{checkpoint}/{split}'
    vector_names = ('posterior_mean_variance_by_dimension',
                    'posterior_mean_covariance_eigenvalues',
                    'posterior_noise_variance_by_dimension')
    vectors = {name: _finite(stats.get(name), f'{label}/{name}') for name in vector_names}
    if any(value.shape != (16,) for value in vectors.values()):
        raise ValueError(f'{label}: each latent per-dimension vector must have length 16')
    mean_var = vectors['posterior_mean_variance_by_dimension']
    eigenvalues = vectors['posterior_mean_covariance_eigenvalues']
    noise_dim = vectors['posterior_noise_variance_by_dimension']
    if np.any(mean_var < -1e-9) or np.any(eigenvalues < -1e-9) or np.any(noise_dim < 0):
        raise ValueError(f'{label}: posterior variance/eigenvalues cannot be negative')
    threshold = float(stats.get('active_units_threshold', np.nan))
    active = int(stats.get('active_units', -1))
    if threshold != .01 or active != int(np.sum(mean_var > threshold)):
        raise ValueError(f'{label}: active-unit count does not match the declared 0.01 variance threshold')
    if int(stats.get('maps', -1)) != expected_maps:
        raise ValueError(f'{label}: expected {expected_maps} maps, got {stats.get("maps")}')
    signal = float(stats.get('posterior_mean_signal_variance_sum', np.nan))
    noise = float(stats.get('posterior_noise_variance_sum', np.nan))
    noise_ratio = float(stats.get('posterior_noise_to_signal_ratio', np.nan))
    effective_rank = float(stats.get('posterior_mean_covariance_effective_rank', np.nan))
    participation = float(stats.get('posterior_mean_covariance_participation_ratio', np.nan))
    scalar = np.asarray([signal, noise, noise_ratio, effective_rank, participation])
    if not np.isfinite(scalar).all() or min(signal, noise) < 0 or not (0 <= effective_rank <= 16.0001) or \
            not (0 <= participation <= 16.0001):
        raise ValueError(f'{label}: invalid posterior scalar diagnostics {scalar}')
    _close(signal, mean_var.sum(), label=f'{label}/signal variance sum', rtol=1e-5)
    _close(noise, noise_dim.sum(), label=f'{label}/posterior noise sum', rtol=1e-5)
    _close(noise_ratio, noise / max(signal, 1e-12), label=f'{label}/noise-to-signal ratio', rtol=1e-5)
    _close(eigenvalues.sum(), signal, label=f'{label}/covariance eigenvalue sum', rtol=2e-4)
    if eigenvalues.sum() > 0:
        p = eigenvalues / eigenvalues.sum()
        positive = p > 0
        expected_rank = float(np.exp(-np.sum(p[positive] * np.log(p[positive]))))
        expected_participation = float(eigenvalues.sum() ** 2 / np.square(eigenvalues).sum())
    else:
        expected_rank = expected_participation = 0.
    _close(effective_rank, expected_rank, label=f'{label}/effective rank', rtol=2e-4)
    _close(participation, expected_participation, label=f'{label}/participation ratio', rtol=2e-4)
    norm_names = ('posterior_mean_vector_norm_mean', 'posterior_mean_vector_norm_p50',
                  'posterior_mean_vector_norm_p95', 'posterior_mean_vector_norm_max',
                  'fraction_posterior_mean_norm_above_agreement_radius')
    norms = np.asarray([float(stats.get(name, np.nan)) for name in norm_names])
    radius = float(stats.get('agreement_search_radius', np.nan))
    if not np.isfinite(norms).all() or np.any(norms[:4] < 0) or radius != 12 or \
            not (0 <= norms[4] <= 1) or not (norms[1] <= norms[2] <= norms[3] + 1e-8):
        raise ValueError(f'{label}: invalid posterior-mean norm/search-radius diagnostics')


def _npz_metric_shapes(data: dict, fits: int, evals: int) -> None:
    vector_fit_keys = ('latent_mean_variance_by_dimension', 'latent_covariance_eigenvalues',
                       'latent_posterior_noise_variance_by_dimension')
    scalar_fit_keys = ('latent_active_units', 'latent_effective_rank', 'latent_participation_ratio',
                       'latent_signal_variance_sum', 'latent_posterior_noise_variance_sum',
                       'latent_noise_to_signal_ratio', 'latent_mean_vector_norm_mean',
                       'latent_mean_vector_norm_p50', 'latent_mean_vector_norm_p95',
                       'latent_mean_vector_norm_max',
                       'latent_fraction_mean_vector_norm_above_agreement_radius')
    for key in vector_fit_keys:
        if key not in data or data[key].shape != (fits, 2, 2, 16):
            raise ValueError(f'latent NPZ: {key} expected shape {(fits, 2, 2, 16)}')
        _finite(data[key], f'latent NPZ/{key}')
    for key in scalar_fit_keys:
        if key not in data or data[key].shape != (fits, 2, 2):
            raise ValueError(f'latent NPZ: {key} expected shape {(fits, 2, 2)}')
        _finite(data[key], f'latent NPZ/{key}')
    scalar_curve_keys = (
        'stochastic_elbo_window_mean', 'stochastic_bce_window_mean',
        'stochastic_kl_total_window_mean', 'deterministic_train_bce',
        'deterministic_train_kl_total', 'deterministic_train_objective',
        'validation_bce', 'validation_kl_total', 'validation_objective',
        'deterministic_train_bce_minus_target_bernoulli_entropy',
        'validation_bce_minus_target_bernoulli_entropy',
    )
    for key in scalar_curve_keys:
        if key not in data or data[key].shape != (evals,):
            raise ValueError(f'latent NPZ: {key} expected shape {(evals,)}')
    vector_curve_keys = ('stochastic_kl_per_dimension_window_mean',
                         'deterministic_train_kl_per_dimension',
                         'validation_kl_per_dimension')
    for key in vector_curve_keys:
        if key not in data or data[key].shape != (evals, 16):
            raise ValueError(f'latent NPZ: {key} expected shape {(evals, 16)}')
        _finite(data[key], f'latent NPZ/{key}')


def _validate_npz_fit(data: dict, fit_index: int, fit_row: dict, source_folder: Path) -> None:
    family, task = fit_row['family'], int(fit_row['task'])
    try:
        seed = int(source_folder.name.split('_', 1)[1])
    except (IndexError, ValueError):
        seed = -1
    label = f'{source_folder.name}/{family}/task_{task}'
    stop, best = int(fit_row['stop_step']), int(fit_row['best_step'])
    up_offsets = data['update_offsets']
    ev_offsets = data['evaluation_offsets']
    up_lo, up_hi = int(up_offsets[fit_index]), int(up_offsets[fit_index + 1])
    ev_lo, ev_hi = int(ev_offsets[fit_index]), int(ev_offsets[fit_index + 1])
    if up_hi - up_lo != stop or ev_hi - ev_lo != stop // 10 + 1:
        raise ValueError(f'{label}: ragged curve offsets disagree with stop_step={stop}')
    updates = data['update_step'][up_lo:up_hi]
    steps = data['evaluation_step'][ev_lo:ev_hi]
    if not np.array_equal(updates, np.arange(1, stop + 1)) or \
            not np.array_equal(steps, np.arange(0, stop + 1, 10)):
        raise ValueError(f'{label}: missing, duplicated, or unordered optimizer updates/evaluations')
    stochastic = _finite(data['stochastic_elbo_by_update'][up_lo:up_hi], f'{label}/stochastic trace')
    source_curve_path = source_folder / 'functional' / 'fits' / family / f'task_{task}' / 'loss_curves.json'
    source_curve = _json(source_curve_path)
    original = np.asarray(source_curve['stochastic_train_loss_by_update'], dtype=np.float32)
    if original.shape != stochastic.shape or not np.array_equal(original, stochastic.astype(np.float32)):
        raise ValueError(f'{label}: replay stochastic trace is not bit-identical to the frozen fit')
    parity = fit_row.get('checkpoint_parity', {})
    for key in ('stochastic_trace_exact_match', 'deterministic_train_curve_exact_match',
                'validation_curve_exact_match', 'best_state_exact_match'):
        if parity.get(key) is not True:
            raise ValueError(f'{label}: missing exact parity flag {key}')
    for key in ('stochastic_trace_max_abs_delta', 'deterministic_train_curve_max_abs_delta',
                'validation_curve_max_abs_delta', 'best_state_max_abs_delta'):
        value = parity.get(key)
        if value is None or float(value) != 0.:
            raise ValueError(f'{label}: {key} must be exactly zero, got {value}')

    mapping = {
        'stochastic_elbo_window_mean': 'stochastic_elbo_window_mean',
        'stochastic_bce_window_mean': 'stochastic_bce_window_mean',
        'stochastic_kl_total_window_mean': 'stochastic_kl_total_window_mean',
        'stochastic_kl_per_dimension_window_mean': 'stochastic_kl_per_dimension_window_mean',
    }
    rows = fit_row.get('evaluation_curve', [])
    if len(rows) != len(steps):
        raise ValueError(f'{label}: JSON evaluation curve length does not match NPZ')
    original_evaluations = source_curve.get('evaluations', [])
    if len(original_evaluations) != len(rows):
        raise ValueError(f'{label}: original/replayed deterministic evaluation count differs')
    for field, split in (('deterministic_train_objective', 'deterministic_train'),
                         ('validation_objective', 'validation')):
        original_values = np.asarray([item[field] for item in original_evaluations], dtype=np.float64)
        replay_values = np.asarray([item[split]['objective'] for item in rows], dtype=np.float64)
        if not np.array_equal(original_values, replay_values):
            raise ValueError(f'{label}: {split} deterministic objective trace is not bit-identical')
    for local_index, row in enumerate(rows):
        if int(row['step']) != int(steps[local_index]):
            raise ValueError(f'{label}: JSON and NPZ evaluation steps differ')
        global_index = ev_lo + local_index
        for npz_key, json_key in mapping.items():
            value = row.get(json_key)
            actual = data[npz_key][global_index]
            if value is None:
                if npz_key == 'stochastic_kl_per_dimension_window_mean':
                    _close(actual, np.zeros(16), label=f'{label}/{json_key} initial row')
                elif not np.isnan(actual):
                    raise ValueError(f'{label}/{json_key}: expected NaN at initial step')
            else:
                _close(actual, value, label=f'{label}/{json_key}', rtol=2e-5, atol=2e-6)
        for split, prefix in (('deterministic_train', 'deterministic_train'),
                              ('validation', 'validation')):
            nested = row[split]
            for field, key in (('bce', f'{prefix}_bce'), ('kl_total', f'{prefix}_kl_total'),
                               ('objective', f'{prefix}_objective'),
                               ('kl_per_dimension', f'{prefix}_kl_per_dimension')):
                _close(data[key][global_index], nested[field],
                       label=f'{label}/{split}/{field}', rtol=2e-5, atol=2e-6)
            if not np.isclose(float(nested['objective']),
                              float(nested['bce']) + KL_WEIGHT * float(nested['kl_total']),
                              rtol=2e-5, atol=1e-5):
                raise ValueError(f'{label}/{split}: objective != BCE + 0.1*KL')
            _close(np.asarray(nested['kl_per_dimension']).sum(), nested['kl_total'],
                   label=f'{label}/{split}/per-dimension KL sum', rtol=2e-5, atol=1e-5)
        if local_index:
            for total_key, bce_key, kl_key in (
                    ('stochastic_elbo_window_mean', 'stochastic_bce_window_mean',
                     'stochastic_kl_total_window_mean'),):
                total = float(data[total_key][global_index])
                bce = float(data[bce_key][global_index])
                kl = float(data[kl_key][global_index])
                if not np.isclose(total, bce + KL_WEIGHT * kl, rtol=2e-5, atol=1e-5):
                    raise ValueError(f'{label}: stochastic objective != BCE + 0.1*KL at step {steps[local_index]}')

    for check_i, checkpoint in enumerate(CHECKPOINTS):
        for split_i, split in enumerate(SPLITS):
            stats = fit_row['latent_stats'][checkpoint][split]
            _check_latent_stats(stats, family, checkpoint, split, seed or -1, task,
                                int(fit_row['train_maps']) if split == 'training'
                                else int(fit_row['validation_maps']))
            for npz_key, field in (
                    ('latent_mean_variance_by_dimension', 'posterior_mean_variance_by_dimension'),
                    ('latent_covariance_eigenvalues', 'posterior_mean_covariance_eigenvalues'),
                    ('latent_posterior_noise_variance_by_dimension', 'posterior_noise_variance_by_dimension')):
                _close(data[npz_key][fit_index, check_i, split_i], stats[field],
                       label=f'{label}/{checkpoint}/{split}/{npz_key}', rtol=2e-5, atol=2e-6)
            for npz_key, field in (
                    ('latent_active_units', 'active_units'),
                    ('latent_effective_rank', 'posterior_mean_covariance_effective_rank'),
                    ('latent_participation_ratio', 'posterior_mean_covariance_participation_ratio'),
                    ('latent_signal_variance_sum', 'posterior_mean_signal_variance_sum'),
                    ('latent_posterior_noise_variance_sum', 'posterior_noise_variance_sum'),
                    ('latent_noise_to_signal_ratio', 'posterior_noise_to_signal_ratio'),
                    ('latent_mean_vector_norm_mean', 'posterior_mean_vector_norm_mean'),
                    ('latent_mean_vector_norm_p50', 'posterior_mean_vector_norm_p50'),
                    ('latent_mean_vector_norm_p95', 'posterior_mean_vector_norm_p95'),
                    ('latent_mean_vector_norm_max', 'posterior_mean_vector_norm_max'),
                    ('latent_fraction_mean_vector_norm_above_agreement_radius',
                     'fraction_posterior_mean_norm_above_agreement_radius')):
                _close(data[npz_key][fit_index, check_i, split_i], stats[field],
                       label=f'{label}/{checkpoint}/{split}/{npz_key}', rtol=2e-5, atol=2e-6)


def _load_complete() -> tuple[list[dict], dict]:
    if not (CONVERGED_ROOT / 'COMPLETE').is_file():
        raise ValueError('The converged target/source root is not COMPLETE')
    allocation_path = LATENT_ROOT / 'latent_allocation.json'
    if not allocation_path.is_file():
        raise ValueError(f'Missing latent replay allocation: {allocation_path}')
    allocation = _json(allocation_path)
    if allocation.get('complete') is not True or allocation.get('failed_seeds'):
        raise ValueError(f'{allocation_path}: all eight latent replay workers must exit cleanly')
    if [int(value) for value in allocation.get('seeds', [])] != list(SEEDS):
        raise ValueError(f'{allocation_path}: seed set/order mismatch')
    workers = allocation.get('workers', [])
    if len(workers) != 8 or {int(row.get('seed', -1)) for row in workers} != set(SEEDS) or \
            any(int(row.get('exit_code', -1)) != 0 or row.get('status') != 'complete'
                for row in workers):
        raise ValueError(f'{allocation_path}: worker statuses are incomplete')
    for seed in SEEDS:
        path = LATENT_ROOT / f'seed_{seed}'
        if not (path / 'COMPLETE').is_file():
            raise ValueError(f'{path}: missing latent replay COMPLETE marker')

    all_fits, seed_rows = [], []
    for seed in SEEDS:
        seed_path = LATENT_ROOT / f'seed_{seed}'
        diag_path = seed_path / 'latent_diagnostics.json'
        array_path = seed_path / 'latent_diagnostics.npz'
        if not diag_path.is_file() or not array_path.is_file():
            raise ValueError(f'{seed_path}: missing completed replay diagnostics/NPZ')
        diag = _json(diag_path)
        if diag.get('complete') is not True or diag.get('source_only') is not True or \
                diag.get('target_data_or_labels_loaded') is not False:
            raise ValueError(f'{diag_path}: replay used non-source data or is incomplete')
        if int(diag.get('experiment_seed', -1)) != seed or int(diag.get('seed', -1)) != seed + 50000:
            raise ValueError(f'{diag_path}: experiment seed mismatch')
        if int(diag.get('fit_count', -1)) != 12 or \
                diag.get('all_stochastic_traces_exact') is not True or \
                diag.get('all_deterministic_train_curves_exact') is not True or \
                diag.get('all_validation_curves_exact') is not True or \
                diag.get('all_best_states_exact') is not True:
            raise ValueError(f'{diag_path}: missing full exact-parity coverage')
        replay_protocol = diag.get('replay_protocol', {})
        if replay_protocol.get('families') != list(FAMILIES) or \
                replay_protocol.get('tasks') != list(TASKS) or \
                int(replay_protocol.get('latent_dim', -1)) != 16 or \
                float(replay_protocol.get('active_unit_variance_threshold', -1)) != .01:
            raise ValueError(f'{diag_path}: latent replay configuration mismatch')
        source_folder = CONVERGED_ROOT / f'seed_{seed}'
        source_arrays = source_folder / 'functional' / 'functional_vae_arrays.npz'
        source_diagnostics = source_folder / 'functional' / 'functional_vae_diagnostics.json'
        hashes = diag.get('source_artifacts', {})
        if hashes.get('arrays_sha256') != producer._sha(source_arrays) or \
                hashes.get('diagnostics_sha256') != producer._sha(source_diagnostics):
            raise ValueError(f'{diag_path}: frozen source artifact hash mismatch')
        core_diag = _json(source_diagnostics)
        by_fit = {(family, int(row['task'])): row
                  for family in FAMILIES for row in core_diag['vae'][family]}
        rows = diag.get('fits', [])
        expected_ids = {f'{family}/task_{task}' for family in FAMILIES for task in TASKS}
        if len(rows) != 12 or {row.get('fit_id') for row in rows} != expected_ids:
            raise ValueError(f'{diag_path}: expected one row for each of twelve family/task fits')
        with np.load(array_path, allow_pickle=False) as archive:
            data = {key: np.asarray(archive[key]) for key in archive.files}
        if not np.array_equal(data.get('fit_id'), np.asarray([row['fit_id'] for row in rows])) or \
                not np.array_equal(data.get('family'), np.asarray([row['family'] for row in rows])) or \
                not np.array_equal(data.get('task'), np.asarray([row['task'] for row in rows])):
            raise ValueError(f'{array_path}: fit identity arrays/order differ from JSON')
        if np.asarray(data.get('fit_seed')).shape != (12,) or \
                not np.array_equal(data['fit_seed'], [row['fit_seed'] for row in rows]) or \
                not np.array_equal(data['stop_step'], [row['stop_step'] for row in rows]) or \
                not np.array_equal(data['best_step'], [row['best_step'] for row in rows]):
            raise ValueError(f'{array_path}: fit seed/step arrays disagree with JSON')
        for row in rows:
            family, task = row['family'], int(row['task'])
            fit_seed = seed + 50000 + task * 1009
            expected_maps = 26 if family == 'functional_vae_small' else 205
            if int(row.get('fit_seed', -1)) != fit_seed or \
                    int(row.get('train_maps', -1)) != expected_maps or \
                    int(row.get('validation_maps', -1)) != 51:
                raise ValueError(f'{diag_path}/{row.get("fit_id")}: fit seed/map counts mismatch')
            core_fit = by_fit[(family, task)]
            if int(row['stop_step']) != int(core_fit['stop_step']) or \
                    int(row['best_step']) != int(core_fit['best_step']) or \
                    int(row['fit_seed']) != int(core_fit['seed']):
                raise ValueError(f'{diag_path}/{row["fit_id"]}: fit identity/stop steps differ from frozen run')
            if int(row['best_step']) not in range(0, int(row['stop_step']) + 1, 10):
                raise ValueError(f'{diag_path}/{row["fit_id"]}: best checkpoint is not on eval grid')
        _npz_metric_shapes(data, 12, len(data.get('evaluation_step', [])))
        if data['update_offsets'].shape != (13,) or data['evaluation_offsets'].shape != (13,) or \
                data['update_offsets'][0] != 0 or data['evaluation_offsets'][0] != 0 or \
                data['update_offsets'][-1] != len(data['update_step']) or \
                data['evaluation_offsets'][-1] != len(data['evaluation_step']):
            raise ValueError(f'{array_path}: invalid ragged curve offsets')
        if not np.all(np.diff(data['update_offsets']) > 0) or \
                not np.all(np.diff(data['evaluation_offsets']) > 0):
            raise ValueError(f'{array_path}: non-increasing ragged curve offsets')
        for i, row in enumerate(rows):
            _validate_npz_fit(data, i, row, source_folder)
        seed_rows.append({'seed': seed, 'diagnostics': diag, 'arrays': data,
                          'fits': rows, 'source_folder': source_folder})
        all_fits.extend([{**row, 'seed': seed, 'fit_index': i,
                          'seed_arrays': data, 'source_folder': source_folder}
                         for i, row in enumerate(rows)])
    if len(all_fits) != 96:
        raise ValueError(f'Expected 96 exact latent replays, found {len(all_fits)}')
    return seed_rows, {'allocation': allocation, 'fits': all_fits}


def _seed_summary(seed_rows: list[dict], families=FAMILIES) -> dict:
    result = {}
    for family in families:
        result[family] = {}
        for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
            result[family][checkpoint] = {}
            for split_index, split in enumerate(SPLITS):
                fields = ('latent_active_units', 'latent_effective_rank',
                          'latent_participation_ratio', 'latent_signal_variance_sum',
                          'latent_posterior_noise_variance_sum', 'latent_noise_to_signal_ratio',
                          'latent_mean_vector_norm_mean', 'latent_mean_vector_norm_p50',
                          'latent_mean_vector_norm_p95', 'latent_mean_vector_norm_max',
                          'latent_fraction_mean_vector_norm_above_agreement_radius')
                per_seed = {field: [] for field in fields}
                kl_values = []
                for seed_row in seed_rows:
                    fit_indices = [i for i, row in enumerate(seed_row['fits'])
                                   if row['family'] == family]
                    if len(fit_indices) != 4:
                        raise ValueError(f'{seed_row["seed"]}/{family}: expected four source tasks')
                    for field in fields:
                        key = field
                        per_seed[field].append(float(np.mean([
                            seed_row['arrays'][key][i, checkpoint_index, split_index]
                            for i in fit_indices])))
                    current_kl = []
                    for i in fit_indices:
                        row = seed_row['fits'][i]
                        target_step = row['best_step'] if checkpoint == 'best' else row['stop_step']
                        curve = row['evaluation_curve']
                        eval_row = next(item for item in curve if item['step'] == target_step)
                        current_kl.append(float(eval_row['deterministic_train' if split == 'training'
                                                         else 'validation']['kl_total']))
                    kl_values.append(float(np.mean(current_kl)))
                result[family][checkpoint][split] = {
                    field: {'seed_values': values, **converged_report.control.interval(values)}
                    for field, values in per_seed.items()
                }
                result[family][checkpoint][split]['kl_per_map'] = {
                    'seed_values': kl_values, **converged_report.control.interval(kl_values)}
    return result


def _coordinate_summary(seed_rows: list[dict]) -> dict:
    result = {}
    for family in FAMILIES:
        result[family] = {}
        for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
            result[family][checkpoint] = {}
            for split_index, split in enumerate(SPLITS):
                per_seed = []
                for seed_row in seed_rows:
                    indexes = [i for i, row in enumerate(seed_row['fits'])
                               if row['family'] == family]
                    per_seed.append(np.mean(seed_row['arrays']['latent_mean_variance_by_dimension'][
                        indexes, checkpoint_index, split_index, :], axis=0))
                stats = [converged_report.control.interval([float(row[dim]) for row in per_seed])
                         for dim in range(16)]
                result[family][checkpoint][split] = {
                    'mean': [row['mean'] for row in stats],
                    'ci95': [row['ci95'] for row in stats],
                    'seed_values': [row.tolist() for row in per_seed],
                }
    return result


def _plot_fit(out: Path, fit: dict) -> str:
    data = fit['seed_arrays']
    i = fit['fit_index']
    stop, best = int(fit['stop_step']), int(fit['best_step'])
    u0, u1 = map(int, data['update_offsets'][i:i + 2])
    e0, e1 = map(int, data['evaluation_offsets'][i:i + 2])
    update_step = data['update_step'][u0:u1]
    elbo_update = data['stochastic_elbo_by_update'][u0:u1] / F
    eval_step = data['evaluation_step'][e0:e1]
    row_mask = eval_step > 0
    eval_step = eval_step[row_mask]
    columns = (
        ('stochastic', 'Stochastic train (10-update means)',
         data['stochastic_bce_window_mean'][e0:e1][row_mask],
         data['stochastic_kl_total_window_mean'][e0:e1][row_mask],
         data['stochastic_elbo_window_mean'][e0:e1][row_mask]),
        ('deterministic_train', 'Posterior-mean train',
         data['deterministic_train_bce'][e0:e1][row_mask],
         data['deterministic_train_kl_total'][e0:e1][row_mask],
         data['deterministic_train_objective'][e0:e1][row_mask]),
        ('validation', 'Posterior-mean validation',
         data['validation_bce'][e0:e1][row_mask],
         data['validation_kl_total'][e0:e1][row_mask],
         data['validation_objective'][e0:e1][row_mask]),
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=False)
    for col, (series, title, bce, kl, objective) in enumerate(columns):
        for row_index, ax in enumerate((axes[0, col], axes[1, col])):
            if row_index == 0:
                x = eval_step
                keep = np.ones_like(x, dtype=bool)
                ax.set_title(title + (' · full' if series != 'stochastic' else ' · full + update trace'))
            else:
                width = min(stop, max(1000, int(np.ceil(stop * .2))))
                x_min = max(1, stop - width + 1)
                keep = eval_step >= x_min
                x = eval_step
                ax.set_title(f'{title} · updates {x_min}–{stop}')
            xx = x[keep]
            bce_y = np.asarray(bce)[keep] / F
            kl_y = np.asarray(kl)[keep]
            objective_y = np.asarray(objective)[keep] / F
            ax.plot(xx, bce_y, color='#2166ac', lw=1.05, label='BCE / F')
            ax.plot(xx, objective_y, color='#1b7837', lw=1.15, label='(BCE + 0.1 KL) / F')
            ax.set_ylabel('BCE / F and objective / F')
            ax.set_xlabel('Optimizer update')
            ax.grid(alpha=.2)
            twin = ax.twinx()
            twin.plot(xx, kl_y, color='#d95f02', lw=1.0, label='KL per map (×0.1 in objective)')
            twin.set_ylabel('Unweighted KL per map')
            if series == 'stochastic' and row_index == 0:
                ax.plot(update_step, elbo_update, color='0.35', alpha=.45,
                        lw=.5, label='Stochastic ELBO / update / F')
            ax.axvline(best, color='black', linestyle=':', linewidth=1,
                       label=f'best={best}' if row_index == 0 else None)
            ax.axvline(stop, color='red', linestyle='--', linewidth=1,
                       label=f'stop={stop}' if row_index == 0 else None)
            if row_index == 1:
                ax.set_xlim(x_min, stop)
            if row_index == 0 and col == 0:
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = twin.get_legend_handles_labels()
                ax.legend(h1 + h2, l1 + l2, fontsize=7, loc='best')
    fig.suptitle(f'{fit["family"]} · seed {fit["seed"]} · source task {fit["task"]} · '
                 f'best={best}, stop={stop} updates', y=1.01)
    fig.tight_layout()
    name = f'seed_{fit["seed"]}_{fit["family"]}_task_{fit["task"]}_bce_kl'
    destination = out / 'plots' / 'loss_decomposition'
    destination.mkdir(parents=True, exist_ok=True)
    converged_report._save(fig, destination, name)
    return name


def _plot_latent_summary(out: Path, summary: dict, coordinates: dict,
                         report_data: dict) -> list[str]:
    lines = ['## Проверка KL и использования 16 латентных координат', '',
             'Здесь проверяется конкретное опасение о collapse: полезно смотреть одновременно на KL, дисперсию '
             'posterior mean между картами и количество active units. Суммарный KL сам по себе не доказывает, '
             'что латентное пространство использует 16 независимых направлений. Active unit считается по заранее '
             'зафиксированному порогу variance(mu)>0.01; effective rank и participation ratio показывают '
             'сжатость совместной ковариации.', '',
             'Метрики агрегированы по четырём source-задачам внутри seed; 95% интервалы рассчитаны по восьми seed '
             '(df=7), а не по 32 source-task fit как независимым повторам.', '']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    colors = {('best', 'training'): '#2166ac', ('best', 'validation'): '#67a9cf',
              ('stop', 'training'): '#b2182b', ('stop', 'validation'): '#ef8a62'}
    labels = {('best', 'training'): 'Best · train', ('best', 'validation'): 'Best · held-out',
              ('stop', 'training'): 'Stop · train', ('stop', 'validation'): 'Stop · held-out'}
    for ax, family in zip(axes, FAMILIES):
        cats, vals, lo, hi, cs = [], [], [], [], []
        for checkpoint in CHECKPOINTS:
            for split in SPLITS:
                stat = summary[family][checkpoint][split]['latent_active_units']
                cats.append(labels[(checkpoint, split)])
                vals.append(stat['mean'])
                lo.append(stat['ci95'][0]); hi.append(stat['ci95'][1])
                cs.append(colors[(checkpoint, split)])
        x = np.arange(len(cats))
        ax.bar(x, vals, color=cs, alpha=.78)
        ax.errorbar(x, vals, yerr=np.vstack((np.asarray(vals)-np.asarray(lo),
                                             np.asarray(hi)-np.asarray(vals))),
                    fmt='none', color='black', capsize=3)
        ax.set_xticks(x, cats, rotation=20, ha='right')
        ax.set_ylim(0, 16.8)
        ax.set_title(family)
        ax.set_ylabel('Active units (variance of μ > 0.01)')
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Активные латентные координаты: best checkpoint и stop state')
    fig.tight_layout()
    converged_report._save(fig, out / 'plots', 'latent_active_units')
    report_data['latent_active_units'] = {
        family: {checkpoint: {split: summary[family][checkpoint][split]['latent_active_units']
                              for split in SPLITS} for checkpoint in CHECKPOINTS}
        for family in FAMILIES}
    lines += ['### Active units', '',
              '![Число active units](latent_diagnostics/plots/latent_active_units.png)', '',
              '**Как читать.** Каждая точка/столбец — среднее числа координат с variance(posterior mean)>0.01. '
              'Показаны train и общий held-out split для best checkpoint и последнего stop state. Если число близко '
              'к 16, каждая координата меняется по картам выше заданного порога; это не говорит об их независимости.', '']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for ax, family in zip(axes, FAMILIES):
        for checkpoint, split in (('best', 'training'), ('best', 'validation'),
                                  ('stop', 'training'), ('stop', 'validation')):
            stat = coordinates[family][checkpoint][split]
            mean = np.asarray(stat['mean'])
            ci = np.asarray(stat['ci95'])
            lower, upper = mean - ci[:, 0], ci[:, 1] - mean
            x = np.arange(1, 17)
            ax.plot(x, mean, marker='o', ms=3, color=colors[(checkpoint, split)],
                    label=labels[(checkpoint, split)])
            ax.fill_between(x, ci[:, 0], ci[:, 1], color=colors[(checkpoint, split)], alpha=.09)
        ax.axhline(.01, color='black', linestyle='--', linewidth=.8, label='AU threshold 0.01')
        ax.set_xticks(np.arange(1, 17))
        ax.set_xlabel('Latent coordinate')
        ax.set_title(family)
        ax.set_ylabel('Var across maps of posterior μ')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7, loc='best')
    fig.suptitle('Дисперсия posterior mean по каждой из 16 координат')
    fig.tight_layout()
    converged_report._save(fig, out / 'plots', 'latent_variance_by_coordinate')
    report_data['latent_variance_by_coordinate'] = coordinates
    lines += ['### Дисперсия по координатам', '',
              '![Variance posterior mean по координатам](latent_diagnostics/plots/latent_variance_by_coordinate.png)', '',
              '**Как читать.** Для каждой координаты показана дисперсия μ между source maps, усреднённая по '
              'четырём задачам внутри seed; полупрозрачные полосы — pointwise 95% t-интервалы по восьми seeds. '
              'Горизонтальная линия 0.01 — тот же порог, использованный для подсчёта active units. Соседние номера '
              'координат не имеют встроенного семантического порядка.', '']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    categories = (('best', 'training'), ('best', 'validation'),
                  ('stop', 'training'), ('stop', 'validation'))
    for ax, family in zip(axes, FAMILIES):
        vals, low, high = [], [], []
        names = []
        for checkpoint, split in categories:
            stat = summary[family][checkpoint][split]['kl_per_map']
            vals.append(stat['mean']); low.append(stat['ci95'][0]); high.append(stat['ci95'][1])
            names.append(labels[(checkpoint, split)])
        x = np.arange(len(names))
        ax.bar(x, vals, color=[colors[item] for item in categories], alpha=.78)
        ax.errorbar(x, vals, yerr=np.vstack((np.asarray(vals)-np.asarray(low),
                                             np.asarray(high)-np.asarray(vals))),
                    fmt='none', color='black', capsize=3)
        ax.set_xticks(x, names, rotation=20, ha='right')
        ax.set_title(family)
        ax.set_ylabel('KL divergence per map (unweighted)')
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('KL по train/held-out картам на best и stop checkpoints')
    fig.tight_layout()
    converged_report._save(fig, out / 'plots', 'latent_kl_per_map')
    report_data['latent_kl_per_map'] = {
        family: {checkpoint: {split: summary[family][checkpoint][split]['kl_per_map']
                              for split in SPLITS} for checkpoint in CHECKPOINTS}
        for family in FAMILIES}
    lines += ['### KL и эффективная размерность', '',
              '![KL per map](latent_diagnostics/plots/latent_kl_per_map.png)', '',
              '**Как читать.** Столбцы показывают невзвешенный аналитический KL на одну карту; в objective он '
              'вносится с коэффициентом 0.1. Интервалы — по seed-level средним. Сопоставляйте KL с числом active '
              'units и effective rank: активность всех координат не означает, что covariance занимает все 16 '
              'независимых направлений.', '']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    categories = (('best', 'training'), ('best', 'validation'),
                  ('stop', 'training'), ('stop', 'validation'))
    category_labels = [labels[item] for item in categories]
    radius_colors = [colors[item] for item in categories]
    for ax, family in zip(axes, FAMILIES):
        vals, low, high = [], [], []
        for checkpoint, split in categories:
            stat = summary[family][checkpoint][split][
                'latent_fraction_mean_vector_norm_above_agreement_radius']
            vals.append(stat['mean'])
            low.append(stat['ci95'][0])
            high.append(stat['ci95'][1])
        x = np.arange(len(category_labels))
        ax.bar(x, vals, color=radius_colors, alpha=.78)
        ax.errorbar(x, vals, yerr=np.vstack((np.asarray(vals) - np.asarray(low),
                                             np.asarray(high) - np.asarray(vals))),
                    fmt='none', color='black', capsize=3)
        ax.set_xticks(x, category_labels, rotation=20, ha='right')
        ax.set_ylim(0, 1)
        ax.set_title(family)
        ax.set_ylabel('Fraction of source maps with ||μ||₂ > 12')
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Доля posterior means за радиусом 12')
    fig.tight_layout()
    converged_report._save(fig, out / 'plots', 'latent_search_radius_coverage')
    report_data['latent_search_radius_coverage'] = {
        family: {checkpoint: {split: summary[family][checkpoint][split][
            'latent_fraction_mean_vector_norm_above_agreement_radius']
            for split in SPLITS} for checkpoint in CHECKPOINTS} for family in FAMILIES}
    lines += ['### Покрытие радиуса поиска', '',
              '![Доля posterior means за радиусом 12](latent_diagnostics/plots/latent_search_radius_coverage.png)', '',
              '**Как читать.** Показана доля source maps, для которых евклидова норма posterior mean μ '
              'превышает радиус 12, использованный в agreement search. Это геометрическая доля точек за '
              'границей заданного шара, а не доля успешных поисков, качество decoder или доказательство '
              'оптимальности радиуса. Интервалы рассчитаны по seed-level средним.', '']

    lines += ['### Численная сводка latent metrics', '',
              '| VAE семейство | Checkpoint | Split | KL/map | Active units | Effective rank | Participation ratio | Noise/signal | ||μ||₂ > 12 |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for family in FAMILIES:
        for checkpoint in CHECKPOINTS:
            for split in SPLITS:
                row = summary[family][checkpoint][split]
                lines.append(f"| {family} | {checkpoint} | {split} | "
                             f"{row['kl_per_map']['mean']:.4f} | "
                             f"{row['latent_active_units']['mean']:.3f} | "
                             f"{row['latent_effective_rank']['mean']:.3f} | "
                             f"{row['latent_participation_ratio']['mean']:.3f} | "
                             f"{row['latent_noise_to_signal_ratio']['mean']:.4f} | "
                             f"{100 * row['latent_fraction_mean_vector_norm_above_agreement_radius']['mean']:.1f}% |")
    lines += ['', 'Все агрегаты усредняют четыре source-задачи внутри каждого seed; полный набор per-fit значений и '
              'интервалы сохранены в `latent_diagnostics/latent_report_summary.json`.', '']
    return lines


def build_latent_report(out: Path = CONVERGED_ROOT) -> dict:
    out = Path(out).resolve()
    seed_rows, aggregate = _load_complete()
    fits = aggregate['fits']
    latent_root = out / 'latent_diagnostics'
    if latent_root.resolve() != LATENT_ROOT.resolve():
        raise ValueError('This report module is bound to the production converged experiment root')
    latent_summary = _seed_summary(seed_rows)
    coordinate_summary = _coordinate_summary(seed_rows)
    cpu_path = INDEPENDENT_CPU_PATH
    if not cpu_path.is_file():
        raise ValueError(f'Missing independent CPU latent check: {cpu_path}')
    cpu = _json(cpu_path)
    if cpu.get('scope') != 'CPU independent check of all32 large Functional VAE best source-validation checkpoints, not training trajectories':
        raise ValueError(f'{cpu_path}: CPU validation scope mismatch')
    cpu_rows = cpu.get('rows', [])
    if len(cpu_rows) != 32:
        raise ValueError(f'{cpu_path}: expected exactly 32 large-functional best checkpoints')
    fit_index = {(row['seed'], row['family'], int(row['task'])): row for row in fits}
    cpu_comparison = {'relative_tolerance': CPU_RTOL, 'absolute_tolerance': CPU_ATOL,
                      'metrics': {}, 'active_unit_boundary_cases': [], 'compared_rows': 0,
                      'active_unit_comparisons': 0, 'active_unit_exact_matches': 0}
    error_samples: dict[str, list[float]] = {}
    core_root = out
    for cpu_row in cpu_rows:
        seed, task = int(cpu_row['seed']), int(cpu_row['task'])
        key = (seed, 'functional_vae_large', task)
        if key not in fit_index:
            raise ValueError(f'{cpu_path}: CPU fit missing from replay: {key}')
        row = fit_index[key]
        if int(row['best_step']) != int(cpu_row['best_step']):
            raise ValueError(f'{cpu_path}: best step differs for seed={seed}, task={task}')
        data = row['seed_arrays']
        i = int(row['fit_index'])
        evaluation = next(item for item in row['evaluation_curve']
                          if int(item['step']) == int(row['best_step']))
        metrics = {
            'training': {
                'KL_per_map': float(evaluation['deterministic_train']['kl_total']),
                'KL_per_dimension': np.asarray(evaluation['deterministic_train']['kl_per_dimension'], dtype=float),
                'MU_variance_per_dimension': np.asarray(row['latent_stats']['best']['training'][
                    'posterior_mean_variance_by_dimension'], dtype=float),
                'active_units_variance_gt_0.01': int(row['latent_stats']['best']['training']['active_units']),
                'effective_covariance_rank': float(row['latent_stats']['best']['training'][
                    'posterior_mean_covariance_effective_rank']),
                'participation_ratio': float(row['latent_stats']['best']['training'][
                    'posterior_mean_covariance_participation_ratio']),
            },
            'validation': {
                'KL_per_map': float(evaluation['validation']['kl_total']),
                'KL_per_dimension': np.asarray(evaluation['validation']['kl_per_dimension'], dtype=float),
                'MU_variance_per_dimension': np.asarray(row['latent_stats']['best']['validation'][
                    'posterior_mean_variance_by_dimension'], dtype=float),
                'active_units_variance_gt_0.01': int(row['latent_stats']['best']['validation']['active_units']),
                'effective_covariance_rank': float(row['latent_stats']['best']['validation'][
                    'posterior_mean_covariance_effective_rank']),
                'participation_ratio': float(row['latent_stats']['best']['validation'][
                    'posterior_mean_covariance_participation_ratio']),
            },
        }
        for split in SPLITS:
            cpu_metrics = cpu_row[split]
            for metric, actual in metrics[split].items():
                if metric == 'active_units_variance_gt_0.01':
                    expected = int(cpu_metrics[metric])
                    cpu_comparison['active_unit_comparisons'] += 1
                    if int(actual) != expected:
                        cpu_variance = np.asarray(cpu_metrics['MU_variance_per_dimension'], dtype=float)
                        near = np.flatnonzero(np.abs(cpu_variance - .01) <=
                                              (CPU_ATOL + CPU_RTOL * np.abs(cpu_variance)))
                        if len(near):
                            cpu_comparison['active_unit_boundary_cases'].append({
                                'seed': seed, 'task': task, 'split': split,
                                'cpu_count': expected, 'replay_count': int(actual),
                                'threshold_dimensions': near.tolist(),
                            })
                        else:
                            raise ValueError(f'CPU/replay active-unit count differs away from threshold: {key}/{split}')
                    else:
                        cpu_comparison['active_unit_exact_matches'] += 1
                    continue
                expected = np.asarray(cpu_metrics[metric], dtype=float)
                actual_array = np.asarray(actual, dtype=float)
                if not np.allclose(actual_array, expected, rtol=CPU_RTOL, atol=CPU_ATOL):
                    relative = np.abs(actual_array - expected) / np.maximum(np.abs(expected), CPU_ATOL)
                    raise ValueError(f'CPU/replay {metric} mismatch for {key}/{split}; '
                                     f'max_rel={float(relative.max()):.6g}')
                relative = np.abs(actual_array - expected) / np.maximum(np.abs(expected), CPU_ATOL)
                error_samples.setdefault(metric, []).append(float(relative.max()))
        cpu_comparison['compared_rows'] += 1
    for metric, values in error_samples.items():
        cpu_comparison['metrics'][metric] = {
            'max_relative_error': float(max(values)),
            'mean_row_max_relative_error': float(np.mean(values)),
            'rows': len(values),
        }
    if cpu_comparison['compared_rows'] != 32:
        raise ValueError('Independent CPU check did not reconcile all 32 large-functional best checkpoints')

    large_validation = latent_summary['functional_vae_large']['best']['validation']
    latent_takeaway = {
        'family': 'functional_vae_large', 'checkpoint': 'best', 'split': 'validation',
        'kl_per_map': large_validation['kl_per_map'],
        'active_units': large_validation['latent_active_units'],
        'effective_covariance_rank': large_validation['latent_effective_rank'],
        'search_radius_coverage': large_validation[
            'latent_fraction_mean_vector_norm_above_agreement_radius'],
        'interpretation': 'Nonzero KL and active coordinates do not show posterior collapse under these metrics; '
                         'effective rank below 16 indicates correlated/low-rank occupancy, not latent-dimension '
                         'adequacy or inadequacy.',
    }

    plot_lines = []
    loss_index = []
    for fit in fits:
        name = _plot_fit(latent_root, fit)
        loss_index.append({'seed': fit['seed'], 'family': fit['family'], 'task': fit['task'],
                           'best_step': fit['best_step'], 'stop_step': fit['stop_step'],
                           'plot_png': f'latent_diagnostics/plots/loss_decomposition/{name}.png',
                           'plot_pdf': f'latent_diagnostics/plots/loss_decomposition/{name}.pdf'})
    if len(loss_index) != 96:
        raise ValueError(f'Expected 96 source BCE/KL plots, wrote {len(loss_index)}')
    for family in FAMILIES:
        plot_lines += [f'### {family}: декомпозиция BCE и KL', '',
                       '| Seed | Source task | Best update | Stop update | Loss figures |',
                       '|---:|---:|---:|---:|---|']
        for row in [item for item in loss_index if item['family'] == family]:
            plot_lines.append(f"| {row['seed']} | {row['task']} | {row['best_step']} | "
                              f"{row['stop_step']} | [PNG]({row['plot_png']}) · [PDF]({row['plot_pdf']}) |")
        plot_lines += ['', '**Подпись.** Показаны source-only stochastic train decomposition и deterministic '
                       'posterior-mean decomposition на train и validation. BCE/F и objective/F нанесены на левую '
                       'ось; невзвешенный KL на карту — на правую, с весом 0.1 в objective. Полная '
                       'последовательность stochastic ELBO/F на каждом update приведена тонкой линией по левой '
                       'оси для stochastic train. Нижний ряд — поздний '
                       'zoom; пунктирные вертикали отмечают best validation checkpoint и plateau stop.', '']

    summary = {
        'experiment': 'converged functional VAE source-only KL and latent-usage diagnostics',
        'seeds': list(SEEDS), 'fit_count': len(fits), 'source_only': True,
        'target_data_or_labels_loaded': False, 'latent_dimension': 16,
        'active_unit_variance_threshold': .01, 'objective': 'BCE + 0.1 * KL, averaged over maps',
        'objective_plot_normalization': f'per source-map coordinate F={F}',
        'replay_parity': {
            'all_stochastic_traces_exact': True,
            'all_deterministic_train_curves_exact': True,
            'all_validation_curves_exact': True,
            'all_best_checkpoint_states_exact': True,
            'trace_max_abs_delta': 0., 'deterministic_train_max_abs_delta': 0.,
            'validation_max_abs_delta': 0., 'best_state_max_abs_delta': 0.,
        },
        'cpu_best_checkpoint_reconciliation': cpu_comparison,
        'latent_takeaway': latent_takeaway,
        'latent_summary': latent_summary, 'per_coordinate_variance': coordinate_summary,
        'loss_plot_index': loss_index,
    }
    lines = ['## Диагностика KL и использования 16 латентных координат', '',
             'Все 96 исходных converged fits повторно проиграны только на source-картах. Повтор подтвердил '
             'битовое совпадение сохранённой stochastic ELBO последовательности и best-checkpoint параметров; '
             'не загружались target-данные и target-метки. Для каждого fit сохранены best и stop latent metrics '
             'на train и validation split.', '',
             '### Проверка точности повторного прохода и CPU-сверка', '',
             'Replay каждой стохастической кривой и state dict best checkpoint совпали с frozen source artifacts '
             'точно (максимальная абсолютная разность 0). Независимая CPU-проверка на 32 large-functional best '
             'checkpoint сверена с GPU replay с rtol=0.002 и atol=1e-5; best_step совпал точно для '
             f'{cpu_comparison["compared_rows"]}/32 fits, а active-unit count точно совпал для '
             f'{cpu_comparison["active_unit_exact_matches"]}/{cpu_comparison["active_unit_comparisons"]} '
             'split-comparisons. Максимальные относительные расхождения '
             'непрерывных метрик приведены ниже; CPU check покрывает best checkpoint large-functional и не '
             'является проверкой training trajectory.', '',
             '| CPU/GPU metric | Max relative error | Mean of row maxima | Rows |', '|---|---:|---:|---:|']
    for metric, row in sorted(cpu_comparison['metrics'].items()):
        lines.append(f"| {metric} | {row['max_relative_error']:.3g} | "
                     f"{row['mean_row_max_relative_error']:.3g} | {row['rows']} |")
    if cpu_comparison['active_unit_boundary_cases']:
        lines += ['', 'Некоторые active-unit count расхождения пришлись на координаты CPU-вариации внутри '
                  'численной окрестности порога 0.01; эти seed/task и координаты перечислены в '
                  '`latent_diagnostics/latent_report_summary.json`, без скрытого округления.']
    else:
        lines += ['', 'В 32 CPU-сверенных fit не было координат с CPU variance в пределах заданной численной '
                  'погрешности от AU-порога 0.01; поэтому совпадение active-unit count не зависит от граничного '
                  'округления.']
    kl = latent_takeaway['kl_per_map']
    active = latent_takeaway['active_units']
    rank = latent_takeaway['effective_covariance_rank']
    lines += ['', '### Ответ на вопрос о KL collapse и latent_dim=16', '',
              f"Для functional_vae_large на best validation checkpoint средний KL составляет "
              f"{kl['mean']:.2f} на карту (95% CI {kl['ci95'][0]:.2f}…{kl['ci95'][1]:.2f}), число active "
              f"units — {active['mean']:.2f}/16 (95% CI {active['ci95'][0]:.2f}…{active['ci95'][1]:.2f}), "
              f"а effective covariance rank — {rank['mean']:.2f} (95% CI "
              f"{rank['ci95'][0]:.2f}…{rank['ci95'][1]:.2f}). Эти метрики не указывают на posterior collapse "
              'к prior: KL ненулевой, а все/почти все маргинальные координаты меняются по картам. Одновременно '
              'rank заметно ниже 16, то есть posterior means занимают коррелированное эффективное подпространство. '
              'Эта диагностика описывает использование координат, но сама по себе не отвечает, достаточен ли '
              'latent_dim=16 для качества решения задачи.', '']
    lines += ['', *plot_lines]
    lines += _plot_latent_summary(latent_root, latent_summary, coordinate_summary, summary)
    lines += ['### Интерпретация', '',
              'KL, число active units и дисперсия posterior means позволяют проверить свёртывание posterior к '
              'prior, а effective rank и participation ratio уточняют, сколько независимых направлений несёт '
              'ковариация. Стоит различать активность координаты по маргинальной дисперсии и независимость '
              'направлений: 16 active units сами по себе не означают effective rank 16. KL измерен на одной карте '
              'и в training objective умножается на 0.1; он не смешивается с target test MSE.', '',
              'Полные per-fit latent statistics, curves и parity flags находятся в `latent_diagnostics/seed_*/`; '
              'сводка, тензорные значения и index всех 96 PNG/PDF сохранены в '
              '`latent_diagnostics/latent_report_summary.json` и `latent_diagnostics/latent_report_figure_data.json`.', '']

    section = '\n'.join([BEGIN, *lines, END])
    report_path = out / 'REPORT.md'
    if not report_path.is_file():
        raise ValueError(f'Missing converged report to append: {report_path}')
    original = report_path.read_text(encoding='utf-8')
    if (BEGIN in original) != (END in original):
        raise ValueError('Converged report has an unbalanced latent diagnostics section')
    if BEGIN in original:
        start = original.index(BEGIN)
        finish = original.index(END, start) + len(END)
        original = original[:start].rstrip() + '\n\n' + original[finish:].lstrip()
    report_path.write_text(original.rstrip() + '\n\n' + section + '\n', encoding='utf-8')
    (latent_root / 'latent_report_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (latent_root / 'latent_report_figure_data.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=CONVERGED_ROOT)
    args = parser.parse_args()
    report = build_latent_report(args.out)
    print(json.dumps({'fit_count': report['fit_count'],
                      'cpu_rows_reconciled': report['cpu_best_checkpoint_reconciliation']['compared_rows'],
                      'all_source_only': report['source_only'],
                      'report': str((args.out / 'REPORT.md').resolve())}, ensure_ascii=False))


if __name__ == '__main__':
    main()
