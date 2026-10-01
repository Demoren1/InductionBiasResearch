"""More solutions per source task, then VAE on functional importance maps."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from .core import load_data
from .expanded_bank import build_expanded_bank
from .followup_batched_eval import evaluate_masks_batched
from .followup_common import configure, run_cuda_queue
from .run import task_vectors, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'outputs/deepsets_vaae/20261001_expanded_functional_vae'
SEEDS = list(range(4100, 4108))
METHODS = ['functional_mean_small', 'functional_mean_large',
           'functional_vae_small', 'functional_vae_large', 'raw_vae_large',
           'random', 'dense']
SOURCE_FILES = ['core.py', 'masks.py', 'run.py', 'expanded_bank.py',
                'expanded_functional_vae.py', 'expanded_functional_run.py',
                'followup_importance.py', 'followup_batched_eval.py',
                'followup_common.py']


def protocol(smoke: bool = False) -> dict:
    tasks = task_vectors()
    rng = np.random.default_rng(20261002)
    fresh = rng.normal(size=(8, 10))
    fresh -= fresh.mean(axis=1, keepdims=True)
    fresh /= fresh.std(axis=1, keepdims=True)
    result = {
        'experiment': 'Expanded per-task solution banks and functional-map VAE',
        'date_moscow': '2026-10-01', 'seeds': SEEDS,
        'task_vectors': {**tasks, 'fresh_test': fresh.tolist(), 'fresh_seed': 20261002},
        'source_tasks': 4, 'candidates': 1024, 'keep': 256, 'bank_steps': 800,
        'source_density': .2, 'target_density': .3, 'target_edges': 7526,
        'source_gradient_denominator': 128,
        'large_train_maps': 205, 'small_train_maps': 26, 'validation_maps': 51,
        'shared_alignment': 'raw large-train maps only; held-out maps excluded; small control shares this fixed alignment',
        'vae_epochs': 160, 'vae_latent': 16, 'vae_width': 128,
        'vae_loss': 'sum BCE + 0.1 KL', 'agreement_steps': 400, 'starts': 4,
        'support_sizes': [32, 64, 128, 256], 'eval_steps': 800, 'test_sets': 512,
        'batch_conditions': 8, 'kernel_mode': 'reference',
        'initialization_reference_models': 20, 'gradient_denominator': 20,
        'methods': METHODS,
        'primary_comparison': {'population': 'fresh_test', 'budget': 256,
                               'method': 'functional_vae_large', 'baseline': 'functional_mean_large'},
        'secondary_comparisons': 'exploratory; small versus large VAE/mean, functional versus raw VAE, random and dense',
        'target_task_labels_for_mask_extraction': False,
        'old_test_tasks': 'previously inspected; continuity only',
        'fresh_test_tasks': 'fixed before training; never used for model/density/checkpoint selection outside each target validation split',
        'inference_unit': '8 seed means conditional on fixed cost tasks; pointwise t95 df7, no multiplicity adjustment',
        'smoke': smoke,
    }
    if smoke:
        result.update(candidates=8, keep=4, bank_steps=2, large_train_maps=3,
                      small_train_maps=2, validation_maps=1, vae_epochs=2,
                      agreement_steps=2, starts=4, support_sizes=[8],
                      eval_steps=2, test_sets=8)
    return result


def freeze_sources(out: Path, spec: dict) -> None:
    snap = out / 'source_snapshot'
    snap.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in SOURCE_FILES:
        source = ROOT / 'deepsets_vaae' / name
        content = source.read_bytes()
        target = snap / name
        if target.exists() and target.read_bytes() != content:
            raise ValueError(f'Frozen source differs: {name}')
        target.write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    spec['source_sha256'] = hashes
    bank_phase = out/'bank_phase_protocol.json'
    if bank_phase.exists():
        bank_spec = json.loads(bank_phase.read_text())
        for name, expected in bank_spec['source_sha256'].items():
            if hashlib.sha256((out/'bank_source_snapshot'/name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Bank-phase source snapshot changed: {name}')
        for name in ['expanded_bank.py', 'core.py', 'run.py']:
            if hashes[name] != bank_spec['source_sha256'][name]:
                raise ValueError(f'Bank-training dependency changed between phases: {name}')
        spec['bank_phase_protocol_sha256'] = hashlib.sha256(bank_phase.read_bytes()).hexdigest()
    path = out / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError('Existing protocol differs; use a new output')
    write_json(path, spec)


def worker(out: Path, seed: int, smoke: bool, bank_only: bool = False) -> None:
    configure(seed)
    spec = protocol(smoke)
    parent_spec = out.parent / 'protocol.json'
    if parent_spec.exists():
        saved = json.loads(parent_spec.read_text())
        for name, expected in saved.get('source_sha256', {}).items():
            actual = hashlib.sha256((ROOT / 'deepsets_vaae' / name).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f'Production source changed: {name}')
        spec = saved
    elif not smoke and not bank_only:
        raise ValueError('Production VAE/evaluation requires a frozen parent protocol')
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'COMPLETE').exists():
        return
    write_json(out / 'protocol.json', {**spec, 'seed': seed,
                                      'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')})
    started = time.monotonic()
    def status(stage: str, **extra) -> None:
        payload = {'seed': seed, 'stage': stage, 'elapsed_seconds': time.monotonic()-started, **extra}
        write_json(out / 'status.json', payload)
        print(json.dumps(payload), flush=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    status('loading_data')
    data = load_data(ROOT / 'datasets/mnist8m', seed, device,
                     per_digit_train=100 if smoke else 1000,
                     per_digit_validation=40 if smoke else 300,
                     per_digit_test=40 if smoke else 300)
    write_json(out / 'data_provenance.json', {
        'split_hashes': data.get('split_hashes'),
        'row_ids_pairwise_disjoint': data.get('row_ids_pairwise_disjoint'),
        'exact_pixel_duplicates_excluded': data.get('exact_pixel_duplicates_excluded'),
        'counts': {k: len(v.source_ids) for k,v in data.items() if hasattr(v, 'source_ids')},
        'identity_limit': 'Distinct rows and exact pixels; augmented handwriting identity unavailable',
    })
    banks = []
    for task, costs in enumerate(spec['task_vectors']['source']):
        path = out / f'bank_{task}.pt'
        status('source_bank', task=task)
        if path.exists():
            saved_bank_manifest = out/'bank_artifact_hashes.json'
            if saved_bank_manifest.exists():
                expected = json.loads(saved_bank_manifest.read_text())[path.name]
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise ValueError(f'Cached source bank changed: {path}')
            bank = torch.load(path, map_location='cpu', weights_only=False)
            if tuple(bank['maps'].shape) != (spec['keep'],784,32):
                raise ValueError(f'Cached bank dimensions differ: {path}')
            if len(bank['best_validation_normalized_mse']) != spec['candidates']:
                raise ValueError(f'Cached bank candidate count differs: {path}')
            if bank['training_curves'][-1]['step'] != spec['bank_steps']:
                raise ValueError(f'Cached bank training duration differs: {path}')
        else:
            bank = build_expanded_bank(data, costs, seed+1000*task, device,
                candidates=spec['candidates'], keep=spec['keep'], steps=spec['bank_steps'], density=.2)
            torch.save(bank, path)
        banks.append(bank)
    if bank_only:
        status('bank_complete')
        write_json(out/'bank_timings.json', {'seed':seed, 'elapsed_seconds':time.monotonic()-started})
        (out/'BANK_COMPLETE').write_text('all4 banks complete\n')
        return
    from .expanded_functional_vae import extract_expanded_functional_masks
    functional = out / 'functional'
    mask_path = out / 'masks.pt'
    if mask_path.exists() and (functional / 'functional_vae_diagnostics.json').exists():
        masks = torch.load(mask_path, map_location=device, weights_only=True)
        diagnostics = json.loads((functional/'functional_vae_diagnostics.json').read_text())
    else:
        status('functional_maps_and_vae')
        masks, diagnostics = extract_expanded_functional_masks(
            banks, data, seed+50000, device, functional,
            vae_epochs=spec['vae_epochs'], agreement_steps=spec['agreement_steps'],
            starts=spec['starts'], large_train_maps=spec['large_train_maps'],
            small_train_maps=spec['small_train_maps'], validation_maps=spec['validation_maps'], smoke=smoke)
        torch.save({k:v.detach().cpu() for k,v in masks.items()}, mask_path)
    if list(masks) != METHODS:
        raise ValueError(f'Unexpected methods/order: {list(masks)}')
    for name, mask in masks.items():
        mask = torch.as_tensor(mask)
        assert tuple(mask.shape) == (4,784,32), name
        assert bool(((mask==0)|(mask==1)).all()), name
        assert bool((mask.sum((-1,-2)) == (25088 if name=='dense' else 7526)).all()), name
    eval_kw = dict(support_sizes=tuple(spec['support_sizes']), steps=spec['eval_steps'],
                   test_sets=spec['test_sets'], batch_conditions=8, kernel_mode='reference',
                   initialization_reference_models=20)
    old_tasks = spec['task_vectors']['test'][:1] if smoke else spec['task_vectors']['test']
    fresh_tasks = spec['task_vectors']['fresh_test'][:1] if smoke else spec['task_vectors']['fresh_test']
    status('target_old')
    records = evaluate_masks_batched(data, old_tasks, masks, seed+100000, device,
                                    artifact_dir=out/'weights', **eval_kw)
    status('target_fresh')
    fresh_records = evaluate_masks_batched(data, fresh_tasks, masks, seed+300000, device,
                                          artifact_dir=out/'fresh_weights', **eval_kw)
    write_json(out/'results.json', {'seed':seed, 'records':records, 'fresh_records':fresh_records,
                                   'diagnostics':diagnostics, 'elapsed_seconds':time.monotonic()-started})
    status('complete', records=len(records), fresh_records=len(fresh_records))
    (out/'COMPLETE').write_text('complete\n')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--bank-only', action='store_true')
    args = parser.parse_args()
    if args.launch:
        if args.smoke:
            raise ValueError('Use --seed for smoke; production launch always all8')
        spec = protocol()
        freeze_sources(args.out, spec)
        result = subprocess.run(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used',
                                 '--format=csv,noheader,nounits'],check=True,text=True,capture_output=True)
        rows = [line.split(',') for line in result.stdout.strip().splitlines()]
        selected = [r[1].strip() for r in rows if int(r[2].strip()) == 0]
        if len(selected) != 8:
            raise RuntimeError(f'Expected all8 compute-idle GPUs, found {len(selected)}')
        write_json(args.out/'gpu_selection.json', {'observation':result.stdout,'selected':selected,'memory_considered':False})
        run_cuda_queue('deepsets_vaae.expanded_functional_run',args.out,selected,seeds=SEEDS,
                       workers_per_gpu=1,env_overrides={'DEEPSETS_EVAL_BATCH_CONDITIONS':'8'})
        (args.out/'COMPLETE').write_text('all8 complete\n')
    else:
        if args.seed is None:
            parser.error('--seed required for a worker')
        worker(args.out,args.seed,args.smoke,args.bank_only)


if __name__ == '__main__':
    main()
