"""Matched correction of flat diffusion/FM; reuse completed structured fits.

The first study exposed a rank-deficient flat vector field. This study changes
only its full-dimensional residual path and preserves the first study verbatim.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from . import generative_run as base
from .generative_residual import ResidualFlatTimeField
from .followup_common import run_cuda_queue
from .run import write_json

PHASE1 = base.DEFAULT_OUT
DEFAULT_OUT = base.ROOT / 'outputs/deepsets_vaae/20261001_other_generators_corrected'
PRETRAIN = base.ROOT / 'outputs/deepsets_vaae/20261001_residual_fields'
REUSED = ['gnn_flow', 'set_transformer_flow', 'gan']


def corrected_model(method, device):
    if method in ['diffusion', 'flow_matching']:
        return ResidualFlatTimeField(width=256).to(device)
    return ORIGINAL_FACTORY(method, device)


ORIGINAL_FACTORY = base.make_model


def protocol(flat_only: bool = False) -> dict:
    if not flat_only and not (PHASE1 / 'COMPLETE').is_file():
        raise ValueError('initial study must finish before the matched correction')
    if not flat_only and not (PRETRAIN / 'FLAT_FITS_COMPLETE').is_file():
        raise ValueError('corrected flat fits must finish before final target comparison')
    spec = base.protocol()
    manifest = {}
    for seed in ([] if flat_only else base.SEEDS):
        for method in REUSED:
            folder = PHASE1 / f'seed_{seed}' / method
            if not (folder / 'COMPLETE').is_file():
                raise ValueError(f'missing reusable fit: {folder}')
            for name in ['fit.json', 'model.pt', 'samples.pt']:
                manifest[(folder / name).relative_to(PHASE1).as_posix()] = base.sha(folder / name)
    spec.update(
        experiment='Corrected full-dimensional diffusion/FM; same bank and cached structured/GAN models',
        phase1_root=str(PHASE1), phase1_protocol_sha256=base.sha(PHASE1 / 'protocol.json'),
        reused_model_sha256=manifest, reused_methods=REUSED,
        correction={'models': ['diffusion', 'flow_matching'],
                    'change': 'add zero-initialized time/task scalar coefficient times full input map',
                    'extra_parameters': 257,
                    'matched_initialization': 'base FlatTimeField and initial outputs identical under same training seed',
                    'matched_data_noise_schedule': 'same cached arrays, RNG seeds, batch128, optimizer, source validation probes',
                    'reason': 'unmodified full map output variation rank <=256, cannot represent full-dimensional noise transport',
                    'phase1_flat_models_status': 'architecturally bottlenecked diagnostic controls; not representative diffusion/FM tests',
                    'selection': 'architecture change from source-only rank/loss diagnostics, not target score'},
    )
    if flat_only:
        spec['experiment'] = 'Source-only matched full-rank diffusion/FM fitting phase'
        spec['generator_methods'] = ['diffusion', 'flow_matching']
        spec['methods'] = base.CONTROLS + spec['generator_methods']
        spec['concurrent_execution'] = 'reuse the eight GPUs assigned when idle to the initial study; overlap these source-only fits with its GAN phase'
    else:
        residual_manifest = {}
        for seed in base.SEEDS:
            for method in ['diffusion', 'flow_matching']:
                folder = PRETRAIN / f'seed_{seed}' / method
                if not (folder / 'COMPLETE').is_file():
                    raise ValueError(f'missing corrected fit: {folder}')
                for name in ['fit.json', 'model.pt', 'samples.pt']:
                    residual_manifest[(folder / name).relative_to(PRETRAIN).as_posix()] = base.sha(folder / name)
        spec['residual_fit_root'] = str(PRETRAIN)
        spec['residual_fit_protocol_sha256'] = base.sha(PRETRAIN / 'protocol.json')
        spec['residual_fit_sha256'] = residual_manifest
    for entry in spec['articles']:
        if entry['model'] == 'diffusion':
            entry['adaptation'] = 'full-map x0 DDPM with time/task scalar input residual, no learned VAE/PCA compressor'
        elif entry['model'] == 'flow_matching' and '2210' in entry['url']:
            entry['adaptation'] = 'conditional linear-path field with full-dimensional time/task input residual'
    return spec


def worker(out: Path, seed: int, flat_only: bool = False) -> None:
    spec = base.read(out.parent / 'protocol.json')
    if base.sha(PHASE1 / 'protocol.json') != spec['phase1_protocol_sha256']:
        raise ValueError('phase1 protocol changed')
    out.mkdir(parents=True, exist_ok=True)
    if flat_only:
        base.GENERATORS = ['diffusion', 'flow_matching']
        base.make_model = corrected_model
        base.worker(out, seed, generators_only=True)
        return
    for name, expected in spec['reused_model_sha256'].items():
        if name.startswith(f'seed_{seed}/') and base.sha(PHASE1 / name) != expected:
            raise ValueError(f'reused artifact changed: {name}')
    for method in REUSED:
        src, dst = PHASE1 / f'seed_{seed}' / method, out / method
        if dst.exists() or dst.is_symlink():
            if not dst.is_symlink() or dst.resolve() != src.resolve():
                raise ValueError(f'incorrect reused fit link: {dst}')
        else:
            dst.symlink_to(src.resolve())
    if base.sha(PRETRAIN / 'protocol.json') != spec['residual_fit_protocol_sha256']:
        raise ValueError('corrected fit protocol changed')
    for name, expected in spec['residual_fit_sha256'].items():
        if name.startswith(f'seed_{seed}/') and base.sha(PRETRAIN / name) != expected:
            raise ValueError(f'corrected fit artifact changed: {name}')
    for method in ['diffusion', 'flow_matching']:
        src, dst = PRETRAIN / f'seed_{seed}' / method, out / method
        if dst.exists() or dst.is_symlink():
            if not dst.is_symlink() or dst.resolve() != src.resolve():
                raise ValueError(f'incorrect corrected fit link: {dst}')
        else:
            dst.symlink_to(src.resolve())
    base.make_model = corrected_model
    base.worker(out, seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--flat-only', action='store_true')
    args = parser.parse_args()
    if args.launch:
        spec = protocol(args.flat_only)
        old_deps = base.DEPENDENCIES
        base.DEPENDENCIES = old_deps + ['generative_corrected_run.py', 'generative_residual.py']
        base.freeze(args.out, spec)
        observation = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used',
                                      '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True).stdout
        rows = [r.split(',') for r in observation.strip().splitlines()]
        selected = base.read(PHASE1 / 'gpu_selection.json')['selected'] if args.flat_only else [r[1].strip() for r in rows if int(r[2]) == 0]
        if len(selected) != 8:
            raise RuntimeError('expected all eight GPUs utilization-idle for correction')
        write_json(args.out / 'gpu_selection.json', {'selected': selected, 'observation': observation,
                                                    'memory_considered': False,
                                                    'reused_owned_gpu_assignment': args.flat_only})
        run_cuda_queue('deepsets_vaae.generative_corrected_run', args.out, selected, seeds=base.SEEDS,
                       extra_args=['--flat-only'] if args.flat_only else [])
        (args.out / ('FLAT_FITS_COMPLETE' if args.flat_only else 'COMPLETE')).write_text('all8 complete\n')
    else:
        if args.seed is None:
            parser.error('--seed required')
        worker(args.out, args.seed, args.flat_only)


if __name__ == '__main__':
    main()
