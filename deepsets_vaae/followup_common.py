"""Shared, paired protocol for the three independent follow-up experiments."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from .core import evaluate_masks, load_data
from .run import write_json

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / 'outputs/deepsets_vaae/20261001_pilot'
FOLLOWUP = ROOT / 'outputs/deepsets_vaae/20261001_followup'
SEEDS = tuple(range(4100, 4108))
REFERENCE_MODELS = 20


def configure(seed: int) -> None:
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def load_pilot_seed(seed: int, device: str | torch.device = 'cuda:0') -> dict:
    """Return original inputs. `seed` is the experiment seed (4100..4107)."""
    configure(seed)
    folder = PILOT / f'seed_{seed}'
    protocol = json.loads((folder / 'protocol.json').read_text())
    banks = [torch.load(folder / f'bank_{i}.pt', map_location=device,
                        weights_only=False) for i in range(4)]
    masks = torch.load(folder / 'masks.pt', map_location=device, weights_only=True)
    data = load_data(ROOT / 'datasets/mnist8m', seed, device,
                     per_digit_train=1000, per_digit_validation=300, per_digit_test=300)
    return {'data': data, 'banks': banks, 'original_masks': masks,
            'protocol': protocol, 'folder': folder}


def save_provenance(out: Path, seed: int, details: dict, source_files: list[Path]) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    snapshot = out / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    hashes = {}
    for source in source_files:
        source = Path(source)
        content = source.read_bytes()
        target = snapshot / source.name
        if target.exists() and target.read_bytes() != content:
            raise ValueError(f'Frozen source changed: {target}')
        target.write_bytes(content)
        hashes[str(source)] = hashlib.sha256(content).hexdigest()
    protocol = json.loads((PILOT / f'seed_{seed}' / 'protocol.json').read_text())
    write_json(out / 'provenance.json', {
        'seed': seed, 'original_protocol': protocol, 'details': details,
        'source_sha256': hashes, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'torch': torch.__version__, 'date_moscow': '2026-10-01',
        'target_labels_for_mask_extraction': False,
        'initialization_reference_models': REFERENCE_MODELS,
        'gradient_denominator': REFERENCE_MODELS,
    })


def evaluate_followup(data: dict, costs: torch.Tensor, new_masks: dict,
                      seed: int, device: str | torch.device,
                      artifact_dir: str | Path, original_masks: dict) -> list[dict]:
    """Evaluate full pilot protocol; seed is 4100..4107, NOT seed+100000.

    Add all five original controls first. Reference draw sizes and gradient
    scaling stay fixed at 20 models, regardless of how many methods are added.
    This preserves the original paired weight AND readout initialization.
    Each new method must supply four exact-K masks. Save all chosen target
    checkpoints and audit the original controls against their old results.
    """
    if set(new_masks) & set(original_masks):
        raise ValueError('New method names collide with original baselines')
    masks = {name: value.to(device) for name, value in original_masks.items()}
    for name, value in new_masks.items():
        value = torch.as_tensor(value, device=device, dtype=torch.float32)
        if value.shape != (4, 784, 32):
            raise ValueError(f'{name}: expected four [784,32] masks, got {tuple(value.shape)}')
        if not bool(torch.all((value == 0) | (value == 1))):
            raise ValueError(f'{name}: masks must be binary')
        if not bool(torch.all(value.sum((-1, -2)) == 5018)):
            raise ValueError(f'{name}: masks must have exactly 5018 edges')
        masks[name] = value
    artifact_dir = Path(artifact_dir)
    batch_conditions = int(os.environ.get('DEEPSETS_EVAL_BATCH_CONDITIONS', '1'))
    evaluator = evaluate_masks
    execution_kwargs = {}
    if batch_conditions > 1:
        from .followup_batched_eval import evaluate_masks_batched
        evaluator = evaluate_masks_batched
        execution_kwargs['batch_conditions'] = batch_conditions
    records = evaluator(data, costs, masks, seed + 100000, device,
                             support_sizes=(32, 64, 128, 256), steps=800,
                             batch_size=32, set_size=5, validation_sets=128,
                             test_sets=512, artifact_dir=artifact_dir,
                             initialization_reference_models=REFERENCE_MODELS,
                             **execution_kwargs)
    original = json.loads((PILOT / f'seed_{seed}' / 'results.json').read_text())['records']
    key = lambda r: (r['task'], r['support_size'], r['method'], r['init'])
    old = {key(r): r for r in original}
    controls = [r for r in records if r['method'] in original_masks]
    if set(map(key, controls)) != set(old):
        raise AssertionError('Original control record coverage differs')
    deltas = [abs(float(r['mse']) - float(old[key(r)]['mse'])) for r in controls]
    audit = {'seed': seed, 'control_records': len(deltas),
             'max_absolute_mse_delta': max(deltas),
             'mean_absolute_mse_delta': sum(deltas) / len(deltas),
             'threshold': 1e-4, 'passed': max(deltas) <= 1e-4,
             'reference_initialization_models': REFERENCE_MODELS,
             'per_model_gradient_denominator': REFERENCE_MODELS,
             'execution_batch_conditions': batch_conditions,
             'method_names': list(masks), 'records': len(records)}
    write_json(artifact_dir / 'control_audit.json', audit)
    if not audit['passed']:
        raise AssertionError(f'Control reproduction failed: {audit}')
    return records


def run_cuda_queue(module: str, out: str | Path, gpu_uuids: list[str],
                   extra_args: list[str] | None = None, seeds=SEEDS,
                   workers_per_gpu: int = 1, env_overrides: dict | None = None) -> None:
    """Queue independent seeds on reserved GPUs, optionally overlapping workers.

    workers_per_gpu=2 hides CPU alignment/plotting latency behind another seed.
    The device assignment and numerical problem seeds remain unchanged.
    """
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    allocation_path = out / 'allocation.json'
    if allocation_path.exists():
        raise FileExistsError('Use a fresh allocation; do not overwrite a previous launch')
    if workers_per_gpu < 1:
        raise ValueError('workers_per_gpu must be positive')
    allocation = {'module': module, 'launcher_pid': os.getpid(),
                  'gpu_uuids': gpu_uuids, 'seeds': list(seeds),
                  'workers_per_gpu': workers_per_gpu, 'env_overrides': env_overrides or {},
                  'memory_considered': False, 'workers': [], 'complete': False}
    pending = list(seeds)
    active = {}
    failed = []
    while pending or active:
        for gpu in gpu_uuids:
            while sum(row['uuid']==gpu for _,_,row in active.values()) < workers_per_gpu and pending:
                seed = pending.pop(0)
                folder = out / f'seed_{seed}'
                folder.mkdir(exist_ok=True)
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='2',
                           MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
                env.update(env_overrides or {})
                command = [sys.executable, '-m', module, '--seed', str(seed),
                           '--out', str(folder), *(extra_args or [])]
                log = (folder / 'run.log').open('w')
                proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        cwd=ROOT)
                row = {'seed': seed, 'uuid': gpu, 'pid': proc.pid, 'command': command,
                       'output': str(folder)}
                allocation['workers'].append(row)
                active[seed] = (proc, log, row)
                write_json(allocation_path, allocation)
                print(json.dumps({'launched': row}), flush=True)
        for seed in list(active):
            proc, log, row = active[seed]
            code = proc.poll()
            if code is None:
                continue
            log.close()
            row['exit_code'] = code
            if code:
                failed.append(row['seed'])
            del active[seed]
            write_json(allocation_path, allocation)
            print(json.dumps({'finished': row}), flush=True)
        if active:
            time.sleep(5)
    allocation.update(complete=True, failed_seeds=failed)
    write_json(allocation_path, allocation)
    if failed:
        raise RuntimeError(f'Workers failed: {failed}')
