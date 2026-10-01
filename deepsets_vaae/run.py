"""Reproducible mask-transfer pilot on task-dependent MNIST set scores."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def task_vectors(seed: int = 20261001) -> dict:
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(14, 10))
    vectors -= vectors.mean(axis=1, keepdims=True)
    vectors /= vectors.std(axis=1, keepdims=True)
    return {'seed': seed, 'source': vectors[:4].tolist(),
            'validation': vectors[4:6].tolist(), 'test': vectors[6:].tolist()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('datasets/mnist8m'))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--bank-steps', type=int, default=800)
    parser.add_argument('--eval-steps', type=int, default=800)
    args = parser.parse_args()
    from .core import load_data, build_bank, evaluate_masks
    from .masks import extract_masks

    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'results.json').exists():
        print('Completed artifact exists; nothing to overwrite.', flush=True)
        return
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tasks = task_vectors()
    protocol = {
        'experiment': 'DeepSets VAE agreement mask transfer pilot',
        'date_moscow': '2026-10-01', 'seed': args.seed,
        'device': str(device), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'task_vectors': tasks, 'hidden': 32, 'features': 784,
        'density': .2, 'set_size': 5, 'source_tasks': 4,
        'test_tasks': 8, 'bank_candidates': 128, 'bank_keep': 32,
        'bank_steps': args.bank_steps, 'vae_epochs': 160,
        'agreement_steps': 400, 'mask_starts': 4,
        'support_sizes': [32, 64, 128, 256], 'eval_steps': args.eval_steps,
        'validation_sets': 128, 'test_sets': 512,
        'target_labeled_budget': 'support_sizes count training plus checkpoint-validation labels; 20% validation, capped at validation_sets',
        'scope': 'Symmetric pooling and per-item weight sharing are provided; only connectivity inside the item encoder is learned.',
        'status': 'exploratory pilot; fixed task split; independent bank/VAE/training seeds',
        'target_task_labels_for_mask_selection': False,
        'gpu_selection': 'zero GPU utilization; memory usage ignored',
        'torch': torch.__version__, 'smoke': args.smoke,
    }
    if args.smoke:
        protocol.update(bank_candidates=8, bank_keep=4, bank_steps=3,
                        vae_epochs=2, agreement_steps=2, mask_starts=2,
                        support_sizes=[4, 8], eval_steps=3,
                        validation_sets=8, test_sets=8, test_tasks=1)
    for name in ('core.py', 'masks.py', 'run.py'):
        protocol.setdefault('source_sha256', {})[name] = hashlib.sha256(
            Path(__file__).with_name(name).read_bytes()).hexdigest()
    existing = args.out / 'protocol.json'
    if existing.exists() and json.loads(existing.read_text()) != protocol:
        raise ValueError('Existing protocol differs; use a new output directory')
    write_json(existing, protocol)
    started = time.monotonic()

    def status(stage: str, **details) -> None:
        payload = {'stage': stage, 'elapsed_seconds': time.monotonic() - started,
                   'seed': args.seed, **details}
        write_json(args.out / 'status.json', payload)
        print(json.dumps(payload), flush=True)

    status('loading_data')
    data = load_data(args.data_dir, args.seed, device,
                     per_digit_train=100 if args.smoke else 1000,
                     per_digit_validation=40 if args.smoke else 300,
                     per_digit_test=40 if args.smoke else 300)
    provenance = {}
    for split, values in data.items():
        if isinstance(values, (tuple, list)) and len(values) >= 3:
            ids = values[2].detach().cpu().numpy()
            provenance[split] = {'count': len(ids),
                                'source_ids_sha256': hashlib.sha256(ids.tobytes()).hexdigest()}
    provenance['exact_pixel_duplicates_excluded'] = data.get('exact_pixel_duplicates_excluded', {})
    provenance['row_ids_pairwise_disjoint'] = data.get('row_ids_pairwise_disjoint', False)
    provenance['identity_limit'] = 'Distinct row IDs and exact pixel arrays; original handwriting identity across augmented MNIST8m rows is unavailable.'
    write_json(args.out / 'data_provenance.json', provenance)
    banks = []
    for index, vector in enumerate(tasks['source']):
        path = args.out / f'bank_{index}.pt'
        status('bank', task=index)
        if path.exists():
            bank = torch.load(path, map_location=device, weights_only=False)
        else:
            bank = build_bank(data, torch.tensor(vector, device=device, dtype=torch.float32),
                              args.seed + 1000 * index, device,
                              hidden=32, density=.2,
                              candidates=protocol['bank_candidates'], keep=protocol['bank_keep'],
                              steps=protocol['bank_steps'], batch_size=32, set_size=5)
            torch.save(bank, path)
        banks.append(bank)
    status('vae_and_agreement')
    mask_path = args.out / 'masks.pt'
    if mask_path.exists():
        masks = torch.load(mask_path, map_location=device, weights_only=True)
    else:
        masks, diagnostics = extract_masks(
            banks, args.seed + 50000, device,
            vae_epochs=protocol['vae_epochs'], agreement_steps=protocol['agreement_steps'],
            starts=protocol['mask_starts'], density=.2, latent=16, width=128)
        artifacts = diagnostics.pop('_artifacts', None)
        if artifacts is not None:
            torch.save(artifacts, args.out / 'vae_artifacts.pt')
        write_json(args.out / 'mask_diagnostics.json', diagnostics)
        torch.save({name: mask.cpu() for name, mask in masks.items()}, mask_path)
    masks = {name: value.to(device) for name, value in masks.items()}
    for name, mask in masks.items():
        if not torch.all((mask == 0) | (mask == 1)):
            raise AssertionError(f'{name} is not binary')
        expected = 784 * 32 if name == 'dense' else round(.2 * 784 * 32)
        if not torch.all(mask.sum(dim=(-1, -2)) == expected):
            raise AssertionError(f'{name} has wrong cardinality')
    status('evaluating_target_tasks')
    costs = torch.tensor(tasks['test'][:protocol['test_tasks']], device=device, dtype=torch.float32)
    records = evaluate_masks(
        data, costs, masks, args.seed + 100000, device,
        support_sizes=tuple(protocol['support_sizes']), steps=protocol['eval_steps'],
        batch_size=32, set_size=5, validation_sets=protocol['validation_sets'],
        test_sets=protocol['test_sets'])
    write_json(args.out / 'results.json', {'seed': args.seed, 'records': records,
               'elapsed_seconds': time.monotonic() - started, 'protocol': protocol})
    status('complete', records=len(records))


if __name__ == '__main__':
    main()
