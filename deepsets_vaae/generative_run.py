"""Source-only conditional map generators and paired DeepSets mask transfer.

New artifacts only: the completed VAE/bank studies are immutable inputs.
GNN and Set Transformer parameterize flow matching, not standalone densities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from .core import load_data
from .followup_batched_eval import evaluate_masks_batched
from .followup_common import configure, run_cuda_queue
from .generative_flat import (FlatTimeField, ConditionalGenerator,
                              ConditionalCritic, wgan_gradient_penalty)
from .generative_structured import BipartiteGNNField, SetTransformerField
from .masks import _hard_topk
from .run import write_json

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'outputs/deepsets_vaae/20261001_converged_functional_vae'
DEFAULT_OUT = ROOT / 'outputs/deepsets_vaae/20261001_other_generators'
GENERATORS = ['diffusion', 'flow_matching', 'gnn_flow', 'set_transformer_flow', 'gan']
CONTROLS = ['functional_mean_small', 'functional_mean_large', 'functional_vae_small',
            'functional_vae_large', 'raw_vae_large', 'random', 'dense']
SEEDS = list(range(4100, 4108))
DEPENDENCIES = ['generative_run.py', 'generative_flat.py', 'generative_structured.py', 'generative_report.py',
                'core.py', 'masks.py', 'followup_batched_eval.py', 'followup_common.py', 'run.py']


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def save_pt(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def protocol(smoke: bool = False) -> dict:
    original = read(INPUT / 'protocol.json')
    manifest = {}
    for seed in SEEDS:
        for name in ['functional/functional_vae_arrays.npz', 'masks.pt',
                     'data_provenance.json', 'results.json']:
            path = INPUT / f'seed_{seed}' / name
            manifest[path.relative_to(INPUT).as_posix()] = sha(path)
    return {
        'experiment': 'Conditional functional-map generators on the unchanged expanded bank',
        'date': '2026-10-01', 'input_root': str(INPUT), 'input_sha256': manifest,
        'input_protocol_sha256': sha(INPUT / 'protocol.json'), 'seeds': SEEDS,
        'methods': CONTROLS + GENERATORS, 'generator_methods': GENERATORS,
        'budgets': [32, 64, 128, 256], 'source_tasks': 4,
        'source_keep': 256, 'source_density': 0.2,
        'task_vectors': original['task_vectors'], 'support_sizes': [32, 64, 128, 256],
        'training': {'maps_per_source_task': 205, 'heldout_per_source_task': 51,
                     'conditional_tasks': 4, 'pooled_train_maps': 820, 'pooled_validation_maps': 204,
                     'representation': 'cached normalized aligned functional maps, not weights',
                     'normalization': 'logit clamp 1e-4, scalar mean/std fitted to 820 train maps only',
                     'shared_alignment': 'unchanged raw-training Hungarian alignment from VAE study',
                     'hidden_permutation_augmentation': False,
                     'width_flat': 256, 'width_gnn': 32, 'edge_width_gnn': 8,
                     'width_set_transformer': 128, 'gan_latent': 128,
                     'batch_flat': 128, 'batch_gnn': 128, 'batch_set_transformer': 128,
                     'field_precision': 'CUDA BF16 autocast forward; FP32 parameters, optimizer, MSE and integration; target evaluation unchanged TF32 FP32',
                     'learning_rate': 0.0003, 'gan_learning_rate': 0.0001,
                     'field_lr_schedule': 'source-heldout ReduceLROnPlateau: factor=.5, patience=5 evaluations, threshold=.001, minimum_lr=1e-6',
                     'gan_critic_updates': 3, 'gan_gradient_penalty': 10,
                     'minimum_steps': 2000, 'maximum_steps': 16000, 'eval_every': 100,
                     'plateau_window_evaluations': 5, 'plateau_relative_tolerance': 0.01,
                     'plateau_consecutive_checks': 3, 'patience_steps': 800,
                     'significant_improvement_relative': 0.001,
                     'plateau_requires': 'fixed train and heldout probes; stochastic loss window trend',
                     'maximum_steps_is_not_convergence': True,
                     'GAN_stability': 'fixed train/heldout sliced Wasserstein plus critic and generator trends; minimax losses need not decrease to zero',
                     'checkpoint_selection': 'source heldout objective only; GAN heldout sliced Wasserstein',
                     'tf32': True, 'smoke': smoke},
        'sampling': {'samples_per_source_task': 32, 'diffusion_steps': 200,
                     'diffusion_objective': 'DDPM cosine schedule, x0 prediction, unweighted MSE',
                     'flow_objective': 'independent Gaussian-to-data linear-path conditional flow matching MSE',
                     'flow_integration': 'Heun, 64 uniform steps',
                     'canonicalization': 'Hungarian column match to pooled FUNCTIONAL TRAIN mean only; same for all five generators',
                     'score': 'mean of all 4 x 32 canonicalized generated maps, then hard top-K',
                     'replicas': 'one mask repeated over 4 paired target initializations',
                     'sample_seed': 'experiment seed + 800000 + 1000 * method index'},
        'target_density': 0.3, 'target_edges': 7526,
        'target_eval': {'steps': 800, 'batch_size': 32, 'set_size': 5, 'test_sets': 512,
                        'batch_conditions': 8, 'kernel_mode': 'reference',
                        'initialization_reference_models': 20},
        'comparison_status': 'exploratory: both old/fresh fixed target task vectors have already been inspected',
        'comparison_limitation': 'new generators pool tasks and use sampling mean; existing VAE uses separate models and agreement. FM backbones share objective/data/extraction and batch size, differing capacity.',
        'adaptive_sparsity': 'deferred; fixed top-K prevents opening all edges; future selection must use source-only or nested validation',
        'articles': [
            {'model': 'diffusion', 'url': 'https://arxiv.org/abs/2006.11239', 'adaptation': 'direct full functional-map x0-prediction DDPM; no VAE compressor'},
            {'model': 'flow_matching', 'url': 'https://arxiv.org/abs/2210.02747', 'adaptation': 'conditional independent linear-path flow'},
            {'model': 'gnn_flow', 'url': 'https://arxiv.org/html/2609.32833v1', 'adaptation': 'small bipartite neural graph edge/node velocity field; not paper reproduction'},
            {'model': 'flow_matching', 'url': 'https://arxiv.org/abs/2601.05052', 'adaptation': 'alignment motivates matched-bank control; no full-weight DeepWeightFlow reproduction'},
            {'model': 'gnn_flow', 'url': 'https://arxiv.org/abs/2504.03710', 'adaptation': 'weight-space graph flow motivation'},
            {'model': 'set_transformer_flow', 'url': 'https://arxiv.org/abs/1810.00825', 'adaptation': 'SAB over 32 hidden-neuron columns as equivariant flow field'},
            {'model': 'gan', 'url': 'https://arxiv.org/abs/1704.00028', 'adaptation': 'conditional WGAN-GP on full maps'},
            {'model': 'gan', 'url': 'https://arxiv.org/abs/1901.11058', 'adaptation': 'HyperGAN inspiration only; not its architecture/loss'},
        ],
    }


def freeze(out: Path, spec: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    snapshot = out / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    hashes = {}
    for name in DEPENDENCIES:
        src = ROOT / 'deepsets_vaae' / name
        content = src.read_bytes()
        dst = snapshot / name
        if dst.exists() and dst.read_bytes() != content:
            raise ValueError(f'frozen source differs: {name}')
        dst.write_bytes(content)
        hashes[name] = sha(src)
    spec['source_sha256'] = hashes
    if (out / 'protocol.json').exists() and read(out / 'protocol.json') != spec:
        raise ValueError('existing protocol differs; use fresh root')
    write_json(out / 'protocol.json', spec)


def load_maps(seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    path = INPUT / f'seed_{seed}/functional/functional_vae_arrays.npz'
    with np.load(path) as arr:
        train = torch.tensor(arr['function_train_aligned'], device=device)
        valid = torch.tensor(arr['function_validation_aligned'], device=device)
    if train.shape != (4, 205, 784, 32) or valid.shape != (4, 51, 784, 32):
        raise ValueError('invalid canonical source maps')
    for x in [train, valid]:
        if not bool(torch.isfinite(x).all()) or not bool(((x >= 0) & (x <= 1)).all()):
            raise ValueError('source maps outside finite [0,1]')
    return train, valid


def make_model(method: str, device: torch.device):
    if method in ['diffusion', 'flow_matching']:
        model = FlatTimeField(width=256)
    elif method == 'gnn_flow':
        model = BipartiteGNNField(width=32, edge_width=8)
    elif method == 'set_transformer_flow':
        model = SetTransformerField(width=128)
    elif method == 'gan':
        model = ConditionalGenerator(width=256, latent=128)
    else:
        raise ValueError(method)
    return model.to(device)


def cosine_schedule(steps: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    grid = torch.linspace(0, steps, steps + 1, device=device)
    ab = torch.cos(((grid / steps + .008) / 1.008) * math.pi / 2).square()
    ab = ab / ab[0]
    beta = (1 - ab[1:] / ab[:-1]).clamp(1e-5, .999)
    return beta, (1 - beta).cumprod(0)


def corruption(data: torch.Tensor, task: torch.Tensor, method: str,
               gen: torch.Generator, schedule) -> tuple:
    noise = torch.randn(data.shape, generator=gen, device=data.device)
    if method == 'diffusion':
        indices = torch.randint(len(schedule[1]), (len(data),), generator=gen, device=data.device)
        alpha = schedule[1][indices, None, None]
        x = alpha.sqrt() * data + (1 - alpha).sqrt() * noise
        return x, (indices.float() + 1) / len(schedule[1]), task, data
    t = torch.rand(len(data), generator=gen, device=data.device)
    x = (1 - t[:, None, None]) * noise + t[:, None, None] * data
    return x, t, task, data - noise


def predict(model, x: torch.Tensor, t: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
    # BF16 activations exploit A100 tensor cores and fit the same batch128 on
    # all three backbones; state/targets/MSE/integration remain FP32.
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
        return model(x, t, task).float()


@torch.no_grad()
def probe_loss(model, probe: tuple, batch: int) -> float:
    values = []
    for start in range(0, len(probe[0]), batch):
        x, t, task, y = [p[start:start + batch] for p in probe]
        values.append((predict(model, x, t, task) - y).square().reshape(len(x), -1).mean(1))
    return float(torch.cat(values).mean())


def plateau(rows: list[dict], stochastic: list[float], step: int,
            last_improvement: int, consecutive: int, cfg: dict) -> tuple[dict, int]:
    window = cfg['plateau_window_evaluations']
    statistics = {}
    series = {'train': [r['train_loss'] for r in rows],
              'validation': [r['validation_loss'] for r in rows],
              'stochastic': [float(np.mean(stochastic[max(0, i - 99):i + 1]))
                             for i in range(99, len(stochastic), 100)]}
    if len(rows) < 2 * window or len(series['stochastic']) < 2 * window:
        return {'eligible': False, 'passed': False, 'consecutive': 0}, 0
    for name, vals in series.items():
        values = np.array(vals[-2 * window:])
        previous, current = values[:window].mean(), values[window:].mean()
        scale = max(abs(previous), abs(current), 1e-4)
        statistics[name + '_relative_window_change'] = float(abs(current - previous) / scale)
        slope = np.polyfit(np.arange(len(values)), values, 1)[0]
        statistics[name + '_relative_drift'] = float(abs(slope * window) / scale)
    eligible = step >= cfg['minimum_steps'] and step - last_improvement >= cfg['patience_steps']
    okay = eligible and max(statistics.values()) <= cfg['plateau_relative_tolerance']
    consecutive = consecutive + 1 if okay else 0
    return {**statistics, 'eligible': eligible, 'consecutive': consecutive,
            'since_improvement_steps': step - last_improvement,
            'passed': consecutive >= cfg['plateau_consecutive_checks']}, consecutive


def fit_field(method: str, train: torch.Tensor, valid: torch.Tensor,
              seed: int, folder: Path, cfg: dict, smoke: bool) -> tuple:
    device = train.device
    torch.manual_seed(seed)
    model = make_model(method, device)
    lr = cfg['learning_rate']
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=.5, patience=5, threshold=.001, min_lr=1e-6)
    batch = cfg['batch_gnn'] if method == 'gnn_flow' else (
        cfg['batch_set_transformer'] if method == 'set_transformer_flow' else cfg['batch_flat'])
    schedule = cosine_schedule(200, device)
    flat = train.flatten(0, 1)
    ids = torch.arange(4, device=device).repeat_interleave(train.shape[1])
    probe_gen = torch.Generator(device=device).manual_seed(seed + 111)
    # Fixed, source-only corruption probes; heldout checkpointing sees no target labels.
    n = min(32, train.shape[1], valid.shape[1])
    tp = corruption(train[:, :n].flatten(0, 1), torch.arange(4, device=device).repeat_interleave(n), method, probe_gen, schedule)
    vp = corruption(valid.flatten(0, 1), torch.arange(4, device=device).repeat_interleave(valid.shape[1]), method, probe_gen, schedule)
    gen = torch.Generator(device=device).manual_seed(seed + 222)
    best, best_step, last, consecutive = float('inf'), 0, 0, 0
    best_state = None
    rows, stochastic, pending_losses = [], [], []
    maximum = 3 if smoke else cfg['maximum_steps']
    started = time.monotonic()
    check = {'passed': False}
    first_step = 1
    checkpoint = folder / 'latest_training.pt'
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['state_dict'])
        opt.load_state_dict(saved['optimizer'])
        lr_schedule.load_state_dict(saved['scheduler'])
        gen.set_state(saved['generator'])
        rows, stochastic = saved['rows'], saved['stochastic']
        best, best_step, last, consecutive = saved['best'], saved['best_step'], saved['last'], saved['consecutive']
        best_state = saved['best_state']
        first_step = saved['step'] + 1
    for step in range(first_step, maximum + 1):
        model.train()
        index = torch.randint(len(flat), (batch,), device=device, generator=gen)
        x, t, task, y = corruption(flat[index], ids[index], method, gen, schedule)
        opt.zero_grad(set_to_none=True)
        loss = (predict(model, x, t, task) - y).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        opt.step()
        # Transfer scalar histories once per evaluation, keeping the GPU queue
        # fed between checks instead of synchronizing every optimizer step.
        pending_losses.append(loss.detach())
        if step == 1 or step % cfg['eval_every'] == 0 or step == maximum:
            stochastic.extend(torch.stack(pending_losses).cpu().tolist())
            pending_losses.clear()
            if not all(math.isfinite(value) for value in stochastic[-cfg['eval_every']:]):
                raise FloatingPointError(f'nonfinite {method} training')
            model.eval()
            tr, va = probe_loss(model, tp, batch), probe_loss(model, vp, batch)
            if not math.isfinite(tr + va):
                raise FloatingPointError(f'nonfinite {method} probes')
            if va < best:
                if best == float('inf') or va < best * (1 - cfg['significant_improvement_relative']):
                    last = step
                best, best_step = va, step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            lr_schedule.step(va)
            rows.append({'step': step, 'stochastic_loss': stochastic[-1], 'train_loss': tr,
                         'validation_loss': va, 'learning_rate': opt.param_groups[0]['lr']})
            check, consecutive = plateau(rows, stochastic, step, last, consecutive, cfg)
            progress = {'method': method, 'step': step, 'train': tr, 'validation': va,
                        'best_step': best_step, 'plateau': check,
                        'elapsed_seconds': time.monotonic() - started}
            write_json(folder / 'status.json', progress)
            print(json.dumps(progress), flush=True)
            if step % 1000 == 0 or step == maximum or check['passed']:
                save_pt(checkpoint, {'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                                    'optimizer': opt.state_dict(), 'scheduler': lr_schedule.state_dict(),
                                    'generator': gen.get_state(), 'rows': rows, 'stochastic': stochastic,
                                    'best': best, 'best_step': best_step, 'last': last,
                                    'consecutive': consecutive, 'best_state': best_state, 'step': step})
            if check['passed']:
                break
    model.load_state_dict(best_state)
    model.eval()
    result = {'method': method, 'seed': seed, 'best_step': best_step, 'stop_step': step,
              'best_validation_loss': best, 'converged': bool(check['passed']),
              'convergence_check': check, 'loss_curve': rows, 'stochastic_losses': stochastic,
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'batch_size': batch, 'elapsed_seconds': time.monotonic() - started,
              'objective': 'x0 DDPM MSE' if method == 'diffusion' else 'linear independent flow velocity MSE',
              'checkpoint_selection': 'fixed source heldout corruption probe objective',
              'fit_scope': 'one conditional model, 4 source tasks'}
    save_pt(folder / 'model.pt', {'state_dict': best_state, 'method': method, 'fit': result})
    return model, result


def projected_distance(fake: torch.Tensor, real: torch.Tensor, projection: torch.Tensor) -> float:
    # Same cardinality source-only probes; projected quantile L1 distance.
    a = (fake.flatten(1) @ projection).sort(0).values
    b = (real.flatten(1) @ projection).sort(0).values
    return float((a - b).abs().mean())


def fit_gan(train: torch.Tensor, valid: torch.Tensor, seed: int,
            folder: Path, cfg: dict, smoke: bool) -> tuple:
    device = train.device
    torch.manual_seed(seed)
    model, critic = make_model('gan', device), ConditionalCritic(width=256).to(device)
    go = torch.optim.Adam(model.parameters(), lr=cfg['gan_learning_rate'], betas=(0., .9))
    co = torch.optim.Adam(critic.parameters(), lr=cfg['gan_learning_rate'], betas=(0., .9))
    batch = cfg['batch_flat']
    gen = torch.Generator(device=device).manual_seed(seed + 222)
    flat = train.flatten(0, 1)
    ids = torch.arange(4, device=device).repeat_interleave(train.shape[1])
    n = valid.shape[1]
    pids = torch.arange(4, device=device).repeat_interleave(n)
    zprobe = torch.randn(4 * n, model.latent, device=device, generator=gen)
    projection = torch.randn(784 * 32, 64, device=device, generator=gen)
    projection = projection / projection.norm(dim=0)
    tp, vp = train[:, :n].flatten(0, 1), valid[:, :n].flatten(0, 1)
    best, best_step, last, consecutive = float('inf'), 0, 0, 0
    rows, stochastic, critic_losses, gp_values = [], [], [], []
    maximum = 3 if smoke else cfg['maximum_steps']
    best_state, best_critic = None, None
    started = time.monotonic()
    check = {'passed': False}
    for step in range(1, maximum + 1):
        for p in critic.parameters():
            p.requires_grad_(True)
        for _ in range(cfg['gan_critic_updates']):
            idx = torch.randint(len(flat), (batch,), device=device, generator=gen)
            real, task = flat[idx], ids[idx]
            with torch.no_grad():
                fake = model(torch.randn(batch, model.latent, device=device, generator=gen), task)
            co.zero_grad(set_to_none=True)
            gp = wgan_gradient_penalty(critic, real, fake, task)
            closs = critic(fake, task).mean() - critic(real, task).mean() + cfg['gan_gradient_penalty'] * gp
            closs.backward()
            co.step()
        for p in critic.parameters():
            p.requires_grad_(False)
        task = torch.randint(4, (batch,), device=device, generator=gen)
        go.zero_grad(set_to_none=True)
        fake = model(torch.randn(batch, model.latent, device=device, generator=gen), task)
        gloss = -critic(fake, task).mean()
        gloss.backward()
        go.step()
        stochastic.append(float(gloss.detach()))
        critic_losses.append(float(closs.detach()))
        gp_values.append(float(gp.detach()))
        if not all(math.isfinite(x[-1]) for x in [stochastic, critic_losses, gp_values]):
            raise FloatingPointError('nonfinite GAN training')
        if step == 1 or step % cfg['eval_every'] == 0 or step == maximum:
            with torch.no_grad():
                fake = model(zprobe, pids)
                tr, va = [], []
                for k in range(4):
                    sl = slice(k * n, (k + 1) * n)
                    tr.append(projected_distance(fake[sl], tp[sl], projection))
                    va.append(projected_distance(fake[sl], vp[sl], projection))
                tr, va = float(np.mean(tr)), float(np.mean(va))
            if va < best:
                if best == float('inf') or va < best * (1 - cfg['significant_improvement_relative']):
                    last = step
                best, best_step = va, step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_critic = {k: v.detach().cpu().clone() for k, v in critic.state_dict().items()}
            rows.append({'step': step, 'stochastic_loss': stochastic[-1], 'train_loss': tr,
                         'validation_loss': va, 'generator_loss': stochastic[-1],
                         'critic_loss': critic_losses[-1], 'gradient_penalty': gp_values[-1]})
            # For a game, loss scale and shifts are arbitrary. Still require all
            # trends and distribution probes stationary before calling it stable.
            check, consecutive = plateau(rows, stochastic, step, last, consecutive, cfg)
            if len(critic_losses) >= 1000:
                previous, current = np.mean(critic_losses[-1000:-500]), np.mean(critic_losses[-500:])
                change = abs(current - previous) / max(abs(previous), abs(current), 1.)
                check['critic_relative_window_change'] = float(change)
                if change > cfg['plateau_relative_tolerance']:
                    consecutive = 0
                    check['passed'] = False
                    check['consecutive'] = 0
            print(json.dumps({'method': 'gan', 'step': step, 'train': tr, 'validation': va,
                              'generator': stochastic[-1], 'critic': critic_losses[-1], 'plateau': check}), flush=True)
            write_json(folder / 'status.json', rows[-1] | {'plateau': check})
            if check['passed']:
                break
    model.load_state_dict(best_state)
    model.eval()
    result = {'method': 'gan', 'seed': seed, 'best_step': best_step, 'stop_step': step,
              'best_validation_loss': best, 'converged': bool(check['passed']),
              'convergence_check': check, 'loss_curve': rows, 'stochastic_losses': stochastic,
              'critic_losses': critic_losses, 'gradient_penalties': gp_values,
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'critic_parameter_count': sum(p.numel() for p in critic.parameters()),
              'batch_size': batch, 'elapsed_seconds': time.monotonic() - started,
              'objective': 'WGAN-GP; source fixed projection distances are monitoring/checkpoint metrics',
              'checkpoint_selection': 'fixed source heldout sliced Wasserstein over 64 projections'}
    save_pt(folder / 'model.pt', {'state_dict': best_state, 'critic_state_dict': best_critic,
                                'projection': projection.cpu(), 'method': 'gan', 'fit': result})
    return model, result


@torch.no_grad()
def sample(model, method: str, norm: dict, seed: int, count: int = 32,
           flow_steps: int = 64, batch: int = 32) -> torch.Tensor:
    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(seed)
    # Draw all initial noise in a fixed shape, independent of execution chunks.
    tasks = torch.arange(4, device=device).repeat_interleave(count)
    if method == 'gan':
        initial = torch.randn(4 * count, model.latent, device=device, generator=gen)
    else:
        initial = torch.randn(4 * count, 784, 32, device=device, generator=gen)
    parts = []
    beta, abar = cosine_schedule(200, device)
    lower = (math.log(1e-4 / (1 - 1e-4)) - norm['mean']) / norm['std']
    upper = (math.log((1 - 1e-4) / 1e-4) - norm['mean']) / norm['std']
    for start in range(0, len(tasks), batch):
        task = tasks[start:start + batch]
        x = initial[start:start + batch].clone()
        if method == 'gan':
            x = model(x, task)
        elif method == 'diffusion':
            for i in range(len(beta) - 1, -1, -1):
                t = torch.full((len(x),), (i + 1) / len(beta), device=device)
                x0 = predict(model, x, t, task).clamp(lower, upper)
                previous = abar[i - 1] if i else torch.ones((), device=device)
                alpha = 1 - beta[i]
                x = beta[i] * previous.sqrt() / (1 - abar[i]) * x0 + alpha.sqrt() * (1 - previous) / (1 - abar[i]) * x
                if i:
                    variance = beta[i] * (1 - previous) / (1 - abar[i])
                    x = x + variance.sqrt() * torch.randn(x.shape, device=device, generator=gen)
        else:
            dt = 1 / flow_steps
            for i in range(flow_steps):
                t = torch.full((len(x),), i * dt, device=device)
                v = predict(model, x, t, task)
                w = predict(model, x + dt * v, t + dt, task)
                x = x + .5 * dt * (v + w)
        if not bool(torch.isfinite(x).all()):
            raise FloatingPointError(f'nonfinite {method} samples')
        parts.append(torch.sigmoid(x * norm['std'] + norm['mean']))
    return torch.cat(parts).reshape(4, count, 784, 32)


@torch.no_grad()
def canonicalize(samples: torch.Tensor, reference: torch.Tensor) -> tuple:
    flat = samples.flatten(0, 1)
    r, x = reference.transpose(0, 1), flat.transpose(1, 2)
    cost = r.square().sum(-1)[None, :, None] + x.square().sum(-1)[:, None, :] - 2 * (r[None] @ x.transpose(1, 2))
    orders = []
    for matrix in cost.cpu().numpy():
        _, col = linear_sum_assignment(matrix)
        orders.append(col)
    orders = torch.tensor(np.array(orders), device=samples.device)
    aligned = flat.gather(2, orders[:, None, :].expand_as(flat)).reshape_as(samples)
    return aligned, orders.cpu()


@torch.no_grad()
def sample_metrics(samples: torch.Tensor, train: torch.Tensor, valid: torch.Tensor) -> dict:
    scores = samples.mean((0, 1))
    variances, ious, near_train, near_valid = [], [], [], []
    for k in range(4):
        a, tr, va = [x[k].flatten(1) for x in [samples, train, valid]]
        variances.append(float(a.var(0, unbiased=False).mean() / tr.var(0, unbiased=False).mean().clamp_min(1e-12)))
        hard = _hard_topk(a, 5018)
        inter = hard @ hard.T
        iou = inter / (2 * 5018 - inter).clamp_min(1)
        upper = torch.triu_indices(len(a), len(a), offset=1, device=a.device)
        ious.append(float(iou[upper[0], upper[1]].mean()))
        for real, dest in [(tr, near_train), (va, near_valid)]:
            distance = (a.square().sum(1)[:, None] + real.square().sum(1)[None] - 2 * a @ real.T).clamp_min(0) / a.shape[1]
            dest.append(float(distance.min(1).values.mean()))
    return {'variance_ratio': float(np.mean(variances)), 'pairwise_iou': float(np.mean(ious)),
            'train_nearest_mse': float(np.mean(near_train)), 'validation_nearest_mse': float(np.mean(near_valid)),
            'mean_map_mse': float((scores - train.mean((0, 1))).square().mean()),
            'mean_map_vs_heldout_mse': float((valid - scores).square().mean()),
            'train_mean_vs_heldout_mse': float((valid - train.mean((0, 1))).square().mean()),
            'per_task_variance_ratio': variances, 'per_task_pairwise_iou': ious}


def train_seed(out: Path, seed: int, spec: dict, smoke: bool = False) -> None:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train, valid = load_maps(seed, device)
    logit = torch.logit(train.clamp(1e-4, 1 - 1e-4))
    norm = {'mean': float(logit.mean()), 'std': float(logit.std(unbiased=False))}
    normalized = (logit - norm['mean']) / norm['std']
    normalized_valid = (torch.logit(valid.clamp(1e-4, 1 - 1e-4)) - norm['mean']) / norm['std']
    masks = torch.load(INPUT / f'seed_{seed}/masks.pt', map_location='cpu', weights_only=True)
    if list(masks) != CONTROLS:
        raise ValueError('original mask order mismatch')
    diagnostics = {}
    for idx, method in enumerate(GENERATORS):
        folder = out / method
        folder.mkdir(exist_ok=True)
        if (folder / 'COMPLETE').exists():
            fit = read(folder / 'fit.json')
            saved = torch.load(folder / 'samples.pt', map_location='cpu', weights_only=True)
        else:
            fit_seed = seed + 700000 + idx * 1000
            if method == 'gan':
                model, fit = fit_gan(normalized, normalized_valid, fit_seed, folder, spec['training'], smoke)
            else:
                model, fit = fit_field(method, normalized, normalized_valid, fit_seed, folder, spec['training'], smoke)
            fit['training_seed'] = fit['seed']
            fit['experiment_seed'] = seed
            fit['seed'] = seed
            fit['status'] = ('stable' if fit['converged'] else 'max_steps_unstable') if method == 'gan' else (
                'converged' if fit['converged'] else 'max_steps_not_converged')
            fit['normalization'] = norm
            batch = 32
            samples = sample(model, method, norm, seed + 800000 + idx * 1000,
                             count=4 if smoke else 32, flow_steps=2 if smoke else 64, batch=batch)
            aligned, orders = canonicalize(samples, train.mean((0, 1)))
            score = aligned.mean((0, 1))
            mask = _hard_topk(score.reshape(1, -1), 7526).reshape(1, 784, 32).expand(4, -1, -1).clone()
            fit['sample_metrics'] = sample_metrics(aligned, train, valid)
            fit['sampling'] = spec['sampling'] | {'alignment_reference_sha256': hashlib.sha256(train.mean((0, 1)).cpu().numpy().tobytes()).hexdigest()}
            saved = {'samples': aligned.cpu(), 'unaligned_samples': samples.cpu(),
                     'alignment_orders': orders, 'score': score.cpu(), 'masks': mask.cpu()}
            save_pt(folder / 'samples.pt', saved)
            write_json(folder / 'fit.json', fit)
            (folder / 'COMPLETE').write_text('complete\n')
            del model
            torch.cuda.empty_cache()
        masks[method] = saved['masks']
        diagnostics[method] = fit
        print(json.dumps({'seed': seed, 'method': method, 'fit_complete': True,
                          'converged': fit['converged'], 'sample_metrics': fit['sample_metrics']}), flush=True)
    for method, mask in masks.items():
        expected = 25088 if method == 'dense' else 7526
        if mask.shape != (4, 784, 32) or not bool(((mask == 0) | (mask == 1)).all()) or not bool((mask.sum((1, 2)) == expected).all()):
            raise ValueError(f'invalid final {method} mask')
    save_pt(out / 'masks.pt', masks)
    write_json(out / 'diagnostics.json', diagnostics)
    (out / 'GENERATORS_COMPLETE').write_text('complete\n')


def worker(out: Path, seed: int, smoke: bool = False, generators_only: bool = False) -> None:
    configure(seed)
    out.mkdir(parents=True, exist_ok=True)
    spec = read(out.parent / 'protocol.json')
    for name, expected in spec['source_sha256'].items():
        if sha(ROOT / 'deepsets_vaae' / name) != expected:
            raise ValueError(f'production source changed: {name}')
    for name, expected in spec['input_sha256'].items():
        if name.startswith(f'seed_{seed}/') and sha(INPUT / name) != expected:
            raise ValueError(f'input artifact changed: {name}')
    write_json(out / 'protocol.json', spec | {'seed': seed, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')})
    started = time.monotonic()
    if not (out / 'GENERATORS_COMPLETE').exists():
        train_seed(out, seed, spec, smoke)
    if generators_only:
        return
    masks = torch.load(out / 'masks.pt', weights_only=True, map_location='cpu')
    diagnostics = read(out / 'diagnostics.json')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = load_data(ROOT / 'datasets/mnist8m', seed, device,
                     per_digit_train=1000, per_digit_validation=300, per_digit_test=300)
    provenance = read(INPUT / f'seed_{seed}/data_provenance.json')
    if provenance['split_hashes'] != data['split_hashes'] or data['row_ids_pairwise_disjoint'] is not True:
        raise ValueError('target data splits differ from reference')
    write_json(out / 'data_provenance.json', provenance)
    kw = dict(support_sizes=(32, 64, 128, 256), steps=800, batch_size=32, set_size=5,
              validation_sets=128, test_sets=512, batch_conditions=8,
              kernel_mode='reference', initialization_reference_models=20)
    old = evaluate_masks_batched(data, spec['task_vectors']['test'], masks, seed + 100000, device,
                                 artifact_dir=out / 'weights', **kw)
    fresh = evaluate_masks_batched(data, spec['task_vectors']['fresh_test'], masks, seed + 300000, device,
                                   artifact_dir=out / 'fresh_weights', **kw)
    original = read(INPUT / f'seed_{seed}/results.json')
    audit = {}
    key = lambda r: (r['task'], r['support_size'], r['method'], r['init'])
    for block, rows in [('records', old), ('fresh_records', fresh)]:
        reference = {key(r): r for r in original[block]}
        controls = [r for r in rows if r['method'] in CONTROLS]
        if set(map(key, controls)) != set(reference):
            raise ValueError('control coverage mismatch')
        delta = [abs(float(r['mse']) - float(reference[key(r)]['mse'])) for r in controls]
        audit[block] = {'records': len(delta), 'max_mse_delta': max(delta), 'passed': max(delta) <= 1e-4}
    write_json(out / 'control_audit.json', audit)
    if not all(x['passed'] for x in audit.values()):
        raise ValueError(f'original controls do not reproduce: {audit}')
    expected = {(t, b, m, i) for t in range(8) for b in [32, 64, 128, 256]
                for m in spec['methods'] for i in range(4)}
    for rows in [old, fresh]:
        if len(rows) != len(expected) or set(map(key, rows)) != expected or not all(math.isfinite(r['mse']) for r in rows):
            raise ValueError('target record coverage/nonfinite failure')
    write_json(out / 'results.json', {'seed': seed, 'records': old, 'fresh_records': fresh,
                                     'diagnostics': diagnostics, 'control_audit': audit,
                                     'elapsed_seconds': time.monotonic() - started})
    (out / 'COMPLETE').write_text('complete\n')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--generators-only', action='store_true')
    args = parser.parse_args()
    if args.launch:
        spec = protocol(args.smoke)
        freeze(args.out, spec)
        observation = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used',
                                      '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True).stdout
        rows = [r.split(',') for r in observation.strip().splitlines()]
        selected = [r[1].strip() for r in rows if int(r[2]) == 0]
        if len(selected) != 8:
            raise RuntimeError('expected eight utilization-idle GPUs')
        write_json(args.out / 'gpu_selection.json', {'selected': selected, 'observation': observation,
                                                    'memory_considered': False})
        extra = (['--smoke'] if args.smoke else []) + (['--generators-only'] if args.generators_only else [])
        run_cuda_queue('deepsets_vaae.generative_run', args.out, selected, extra_args=extra,
                       seeds=[4100] if args.smoke else SEEDS)
        if not args.generators_only:
            (args.out / 'COMPLETE').write_text('all8 complete\n')
    else:
        if args.seed is None:
            parser.error('--seed required')
        worker(args.out, args.seed, args.smoke, args.generators_only)


if __name__ == '__main__':
    main()
