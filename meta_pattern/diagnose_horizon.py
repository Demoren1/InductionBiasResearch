"""Controlled short continuations from one common full-U checkpoint.

This diagnostic changes the inner horizon only. Every arm resets outer Adam,
uses the same training episodes, and is read out at a fixed external budget.
No validation or held-out data participates in its training/checkpoint choice.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

import torch
import torch.nn.functional as F

from .common import (dataset, make_model, save_checkpoint, seed_for, setup,
                     source_hashes, task_splits, write_json)
from .config import Config
from .models import adapt_v, forward_with_u
from .train import stable_clip_grad_norm_


def run(checkpoint, out, horizon, steps, device='cuda', benchmark=False):
    checkpoint, out = Path(checkpoint), Path(out)
    if (out / 'protocol.json').exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    original = Config(**state['config'])
    if original.method != 'generator' or original.train_lengths != (3, 4, 6, 8):
        raise ValueError('Requires a generator trained only at known lengths 3,4,6,8')
    if horizon < 1 or steps < 1:
        raise ValueError('Positive horizon and additional step budget required')
    start_step = state['step']
    config = replace(original, inner_steps=horizon, outer_steps=start_step + steps)
    setup(config.seed, device)
    model = make_model(config, device)
    model.load_state_dict(state['model'])
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.outer_lr)
    tasks = {k: [t for t in task_splits(config)['train'] if t.length == k]
             for k in config.train_lengths}
    protocol = {
        'source_checkpoint': str(checkpoint.resolve()),
        'source_checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'source_step': start_step, 'config': config.to_dict(),
        'additional_steps': steps, 'outer_optimizer_reset': True,
        'selection': 'fixed final checkpoint; no per-arm validation selection',
        'benchmark_only': benchmark, 'source_sha256': source_hashes(),
        'episode_rule': 'original training seeds at source_step + offset; same across horizons',
    }
    write_json(out / 'protocol.json', protocol)
    started = time.monotonic()
    elapsed = []
    for offset in range(1, steps + 1):
        tick = time.monotonic()
        step = start_step + offset
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for slot in range(config.tasks_per_step):
            length = config.train_lengths[((step - 1) * config.tasks_per_step + slot) % len(config.train_lengths)]
            episode_seed = seed_for('train', config.seed if config.data_seed is None else config.data_seed, step, slot)
            task = random.Random(episode_seed).choice(tasks[length])
            support = dataset(config, task, config.support_size, seed_for(episode_seed, 'support'), 'support', device)
            query = dataset(config, task, config.query_size, seed_for(episode_seed, 'query'), 'query', device)
            u = model(length)
            v = adapt_v(u, support['x'], support['y'], steps=horizon, lr=config.inner_lr,
                        seed=seed_for(episode_seed, 'v'), create_graph=True,
                        batch_size=config.batch_size, optimizer=config.inner_optimizer,
                        init_scale=config.init_scale)
            loss = F.binary_cross_entropy_with_logits(
                forward_with_u(query['x'], u, v, seq_len=config.seq_len, hidden=config.hidden), query['y'])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f'Nonfinite loss at offset {offset}, slot {slot}')
            (loss / config.tasks_per_step).backward()
            total += float(loss.detach()) / config.tasks_per_step
        norm = stable_clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
        elapsed.append(time.monotonic() - tick)
        row = {'step': step, 'offset': offset, 'horizon': horizon,
               'query_bce': total, 'gradient_norm_before_clip': norm,
               'step_seconds': elapsed[-1], 'seconds': time.monotonic() - started}
        with (out / 'training.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        if offset == 1 or offset % 10 == 0 or benchmark or offset == steps:
            print(f"HORIZON={horizon} offset={offset}/{steps} loss={total:.5f} grad={norm:.4g} seconds={elapsed[-1]:.3f}", flush=True)
        if not benchmark and (offset % 10 == 0 or offset == steps):
            save_checkpoint(out / 'latest.pt', {
                'config': config.to_dict(), 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'step': step, 'source_sha256': protocol['source_sha256'],
                'diagnostic_protocol': protocol,
            })
    result = {'horizon': horizon, 'additional_steps': steps,
              'median_step_seconds': statistics.median(elapsed[1:] or elapsed),
              'seconds': time.monotonic() - started, 'benchmark_only': benchmark}
    write_json(out / 'done.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--horizon', type=int, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--benchmark', action='store_true')
    print(run(**vars(parser.parse_args())), flush=True)
