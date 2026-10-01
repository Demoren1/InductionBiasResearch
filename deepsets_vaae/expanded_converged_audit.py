"""Independent numerical checks for the convergence repeat."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def drift(values, window_count):
    values = np.asarray(values, dtype=float)
    index = np.arange(len(values), dtype=float)
    index -= index.mean()
    slope = float(index @ (values - values.mean()) / (index @ index))
    return abs(slope * window_count) / max(abs(float(values.mean())), 1e-12)


def audit(root):
    spec = read(root / 'protocol.json')
    source = Path(spec['input_root'])
    assert sha(source / 'protocol.json') == spec['original_protocol_sha256']
    for name, digest in spec['input_artifact_sha256'].items():
        assert sha(source / name) == digest, name
    for name, digest in spec['source_sha256'].items():
        assert sha(root / 'source_snapshot' / name) == digest, name
        assert sha(Path(__file__).parent / name) == digest, name

    fit_rows, paired, control_count = [], [], 0
    methods = spec['methods']
    controls = ['functional_mean_small', 'functional_mean_large', 'random', 'dense']
    totals = {population: {method: [] for method in methods}
              for population in ['records', 'fresh_records']}
    for seed in spec['seeds']:
        folder = root / f'seed_{seed}'
        assert (folder / 'COMPLETE').is_file(), seed
        diag = read(folder / 'functional/functional_vae_diagnostics.json')
        assert diag['converged'] is True
        for family, fits in diag['vae'].items():
            assert len(fits) == 4
            for task, fit in enumerate(fits):
                assert fit['converged'] is True
                curves = read(folder / f'functional/fits/{family}/task_{task}/loss_curves.json')
                losses = np.asarray(curves['stochastic_train_loss_by_update'], dtype=float)
                rows = curves['evaluations']
                stop = fit['stop_step']
                criteria = fit['convergence_criteria']
                assert len(losses) == stop and np.isfinite(losses).all()
                assert stop >= criteria['min_steps'] and stop % 10 == 0
                assert [row['step'] for row in rows] == list(range(0, stop + 1, 10))
                train = np.asarray([row['deterministic_train_objective'] for row in rows])
                valid = np.asarray([row['validation_objective'] for row in rows])
                assert np.isfinite(train).all() and np.isfinite(valid).all()
                assert fit['best_step'] == rows[int(np.argmin(valid))]['step']
                assert fit['best_validation_objective'] == float(valid.min())
                window = criteria['window_updates']
                tolerance = criteria['relative_window_tolerance']
                assert fit['convergence_metrics']['plateau_checks_consecutive'] >= 3
                for final in rows[-3:]:
                    step = final['step']
                    index = step // 10
                    streams = [(losses[:step], window),
                               (train[1:index+1], window // 10),
                               (valid[1:index+1], window // 10)]
                    for values, count in streams:
                        previous = values[-2*count:-count].mean()
                        current = values[-count:].mean()
                        change = abs(current - previous) / max(abs(previous), 1e-12)
                        assert change <= tolerance, (seed, family, task, step, change)
                        assert drift(values[-2*count:], count) <= tolerance
                    check = final['convergence_metrics']
                    assert check['eligible'] and check['validation_improvement_patience_passed']
                    assert check['updates_since_significant_validation_improvement'] >= criteria['validation_improvement_patience_updates']
                fit_rows.append({'seed': seed, 'family': family, 'task': task,
                                 'stop_step': stop, 'best_step': fit['best_step']})
        masks = torch.load(folder / 'masks.pt', map_location='cpu', weights_only=True)
        old_masks = torch.load(source / f'seed_{seed}/masks.pt', map_location='cpu', weights_only=True)
        assert list(masks) == methods
        for method, value in masks.items():
            assert tuple(value.shape) == (4, 784, 32)
            assert torch.all((value == 0) | (value == 1))
            assert torch.all(value.sum((1, 2)) == (25088 if method == 'dense' else 7526))
        for method in controls:
            assert torch.equal(masks[method], old_masks[method])
        result = read(folder / 'results.json')
        old_result = read(source / f'seed_{seed}/results.json')
        for population in totals:
            def identity(row):
                return row['task'], row['support_size'], row['method'], row['init']
            new = {identity(row): row for row in result[population]}
            old = {identity(row): row for row in old_result[population]}
            expected = {(task, budget, method, init) for task in range(8)
                        for budget in [32,64,128,256] for method in methods for init in range(4)}
            assert set(new) == expected and len(result[population]) == 896
            for key, row in new.items():
                assert np.isfinite(row['mse'])
                if key[2] in controls:
                    assert row == old[key], (seed, population, key)
                    control_count += 1
            for method in methods:
                totals[population][method].append(float(np.mean([
                    row['mse'] for row in new.values()
                    if row['support_size'] == 256 and row['method'] == method])))
    assert len(fit_rows) == 96
    summary = {}
    for population, method_values in totals.items():
        summary[population] = {method: float(np.mean(values)) for method, values in method_values.items()}
        delta = np.array(method_values['functional_vae_large']) - np.array(method_values['functional_mean_large'])
        mean = float(delta.mean())
        margin = float(t.ppf(.975, 7) * delta.std(ddof=1) / np.sqrt(8))
        paired.append({'population': population, 'contrast': 'VAE large - functional mean large',
                       'mean': mean, 'ci95': [mean-margin, mean+margin], 'seed_differences': delta.tolist()})
    payload = {'status': 'PASS', 'converged_fits': 96, 'independently_recomputed_last_three_checks': True,
               'unchanged_control_records': control_count, 'fit_stops': fit_rows,
               'target256_mean_NMSE': summary, 'paired_effects': paired,
               'inference': 'exploratory, conditional on already inspected fixed cost tasks'}
    (root / 'parent_audit.json').write_text(json.dumps(payload, indent=2) + '\n')
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    result = audit(parser.parse_args().out)
    print(json.dumps({key: result[key] for key in ['status','converged_fits','unchanged_control_records','target256_mean_NMSE','paired_effects']}, indent=2))
