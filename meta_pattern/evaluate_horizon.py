"""Frozen known-validation readout shared by all short horizon controls."""
import argparse
from pathlib import Path
import time

import torch

from .calibrate import _batched_logits, batched_adapt_v
from .common import source_hashes, write_json
from .diagnose_fixed_u import (BATCH_SIZE, BUDGETS, KNOWN_LENGTHS, RESTARTS,
                              _aggregate_selection, _episodes, _metrics, _selection_rows)
from .evaluate_interpolation import _file_sha256, load_interpolation_checkpoint


def evaluate(checkpoint, out, device='cuda'):
    checkpoint, out = Path(checkpoint), Path(out)
    if out.exists():
        raise FileExistsError(out)
    state, config, model = load_interpolation_checkpoint(checkpoint, device)
    started = time.monotonic()
    rows, hashes = [], {}
    for length in KNOWN_LENGTHS:
        episodes, length_hashes = _episodes(config, length, device)
        hashes[str(length)] = length_hashes
        cases = [(episode, restart) for episode in episodes for restart in range(RESTARTS)]
        sx = torch.stack([ep['support_x'] for ep, _ in cases])
        sy = torch.stack([ep['support_y'] for ep, _ in cases])
        qx = torch.stack([ep['query_x'] for ep, _ in cases])
        qy = torch.stack([ep['query_y'] for ep, _ in cases])
        seeds = torch.tensor([ep['v_seeds'][r] for ep, r in cases], dtype=torch.int64)
        with torch.no_grad():
            u = tuple(t.detach() for t in model(length))
        snapshots = batched_adapt_v(
            u, sx, sy, steps=max(BUDGETS), lrs=torch.full((len(cases),), .1, device=device),
            seeds=seeds, optimizers=['adam'] * len(cases),
            init_scales=torch.full((len(cases),), .1, device=device),
            batch_size=BATCH_SIZE, checkpoints=BUDGETS)
        for budget in BUDGETS:
            with torch.no_grad():
                sl = _batched_logits(sx, u, snapshots[budget])
                ql = _batched_logits(qx, u, snapshots[budget])
                for i, (ep, restart) in enumerate(cases):
                    rows.append({'basis': 'learned', 'length': length, 'pattern': ep['pattern'],
                                 'budget': budget, 'restart': restart, 'v_seed': int(seeds[i]),
                                 'config': 'adam_lr0.1_init0.1', 'config_index': 3,
                                 'support': _metrics(sl[i], sy[i]), 'query': _metrics(ql[i], qy[i])})
        print(f'READOUT horizon={config.inner_steps} length={length} cases={len(cases)}', flush=True)
    # Only one optimizer is evaluated here.  The fixed-U grid diagnosis has
    # more configurations; its grid-selection label would be misleading here.
    selected = [row for row in _selection_rows(rows)
                if row['strategy'] != 'support_selected_grid']
    result = {
        'checkpoint': str(checkpoint.resolve()), 'checkpoint_sha256': _file_sha256(checkpoint),
        'checkpoint_step': state['step'], 'config': config.to_dict(),
        'horizon': config.inner_steps, 'rows': rows, 'data_hashes': hashes,
        'selection': 'fixed final continuation; all v readout budgets/restarts predeclared',
        'readout_optimizer_configs': ['adam_lr0.1_init0.1'],
        'aggregates': _aggregate_selection(selected),
        'seconds': time.monotonic() - started, 'source_sha256': source_hashes(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    evaluate(**vars(parser.parse_args()))
