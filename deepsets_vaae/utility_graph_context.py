"""Source-bank functional contexts; never reads target labels or test pools."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .core import _sets, centred_costs, load_data

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / 'outputs/deepsets_vaae/20261001_expanded_functional_vae'
FUNCTIONAL = ROOT / 'outputs/deepsets_vaae/20261001_converged_functional_vae'


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


@torch.no_grad()
def build_context(seed, device, out, data=None):
    """Keep the historical train-only alignment, enrich it with signed function data.

    The 51 held-out maps/task are excluded. Teacher checkpoint selection used
    source validation historically; teacher quality is NOT an input feature.
    q_ij(x)=x_i W_eff,ij a_j (1-tanh(pre_j)^2). Its moments can be computed
    without materializing [teacher,probe,pixel,hidden] arrays.
    """
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    data = load_data(ROOT / 'datasets/mnist8m', seed, device) if data is None else data
    path = FUNCTIONAL / f'seed_{seed}/functional/functional_vae_arrays.npz'
    with np.load(path) as arrays:
        rows = np.asarray(arrays['split_permutations'][:, 51:51+205]).copy()
        orders = np.asarray(arrays['train_alignment_orders']).copy()
        maps = torch.from_numpy(np.asarray(arrays['function_train_aligned']).copy()).float()
        probe_rows = np.asarray(arrays['source_train_image_indices'][:128]).copy()
    probe = data['source_train'].features[torch.as_tensor(probe_rows, device=device)]
    functions, signed, rms, active, artifact_hashes = [], [], [], [], {}
    for task in range(4):
        bp = BANK / f'seed_{seed}/bank_{task}.pt'
        artifact_hashes[str(bp)] = file_hash(bp)
        b = torch.load(bp, map_location='cpu', weights_only=False)
        state = b['state_dict']; idx = torch.from_numpy(rows[task])
        order = torch.from_numpy(orders[task]).to(device)
        w = (state['weight'][idx] * state['masks'][idx]).to(device)
        bias = state['bias'][idx].to(device); a = state['readout'][idx].to(device)
        psi_parts, signed_parts, rms_parts, mask_parts = [], [], [], []
        for start in range(0, 205, 32):
            ww = w[start:start+32]; aa = a[start:start+32]
            h = torch.tanh(torch.einsum('pf,mfh->mph', probe, ww)
                           + bias[start:start+32, None, :])
            psi = h * aa[:, None, :]
            gain = (1-h.square()) * aa[:, None, :]
            qm = ww * torch.einsum('pf,mph->mfh', probe, gain) / len(probe)
            q2 = ww.square() * torch.einsum('pf,mph->mfh', probe.square(), gain.square()) / len(probe)
            o = order[start:start+32]
            psi_parts.append(psi.gather(2, o[:, None, :].expand_as(psi)).cpu())
            signed_parts.append(qm.gather(2, o[:, None, :].expand_as(qm)).cpu())
            qr = q2.clamp_min(0).sqrt()
            rms_parts.append(qr.gather(2, o[:, None, :].expand_as(qr)).cpu())
            mask = state['masks'][idx[start:start+32]].to(device)
            mask_parts.append(mask.gather(2, o[:, None, :].expand_as(mask)).cpu())
        functions.append(torch.cat(psi_parts)); signed.append(torch.cat(signed_parts))
        rms.append(torch.cat(rms_parts)); active.append(torch.cat(mask_parts))
        del b, state, w, bias, a
    psi = torch.stack(functions); qmean = torch.stack(signed); qrms = torch.stack(rms)
    mask = torch.stack(active)
    # Per-teacher sensitivity normalization avoids weighting solely by readout scale.
    qs = qrms.flatten(2).amax(-1).clamp_min(1e-8)[..., None, None]
    qm = qmean / qs; qr = qrms / qs
    ps = psi.square().mean((1, 2, 3)).sqrt().clamp_min(1e-8)[:, None, None, None]
    psi_norm = psi / ps
    node = torch.cat((psi_norm.mean((0, 1)).T,
                      psi_norm.std((0, 1), unbiased=False).T), dim=1)
    edge = torch.stack((maps.mean((0, 1)), maps.std((0, 1), unbiased=False),
                        qm.mean((0, 1)), qr.mean((0, 1)),
                        qr.std((0, 1), unbiased=False), mask.mean((0, 1))), dim=-1)
    value = dict(node=node, edge=edge, mean_score=maps.mean((0, 1)),
                 source_mean_scores=maps.mean(1),
                 teacher_scores=maps[:, ::26].reshape(-1, 784, 32),
                 psi=psi, q_signed_mean=qmean, q_rms=qrms,
                 probe_rows=torch.from_numpy(probe_rows),
                 probe_source_ids=data['source_train'].source_ids[torch.as_tensor(probe_rows, device=device)].cpu(),
                 train_rows=torch.from_numpy(rows), alignment_orders=torch.from_numpy(orders))
    torch.save(value, out / 'functional_context.pt')
    meta = dict(seed=seed, teachers=820, source_density=.2, target_density=.3,
                heldout_maps_used=False, teacher_quality_input=False, probe_images=128,
                alignment='historical raw train-only Hungarian orders applied jointly to full states/functions',
                alignment_limitation='equivariant graph field does not make this preprocessing alignment-free',
                representation='full teacher states referenced by immutable hashes; psi profiles and signed/RMS q moments saved',
                archived_abs_gradient_probe_images=4096,
                full_profile_probe_images=128, artifact_sha256=artifact_hashes,
                functional_arrays_sha256=file_hash(path), split_hashes=data['split_hashes'])
    (out / 'context_manifest.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return value, data


def task_context(x, y):
    """Only support set observations, without individual digit labels/cost vectors."""
    z = x.sum(1) / x.shape[1]
    centered = y - y.mean()
    covariance = ((z-z.mean(0)) * centered[:, None]).mean(0)
    covariance = covariance / covariance.square().mean().sqrt().clamp_min(1e-4)
    return torch.cat((z.mean(0), covariance, y.mean()[None],
                      y.std(unbiased=False)[None] / x.shape[1]**.5))


def task_sets(splits, costs, seed, train_name='target_train', query_name='target_validation',
              test_name=None, train_count=205, query_count=51, test_count=512):
    result = []
    device = splits[train_name].features.device
    for i, cost in enumerate(costs):
        gen = torch.Generator(device=device).manual_seed(seed + 10007*(i+1))
        c = centred_costs(cost, device)
        sx, sy = _sets(splits[train_name], c, train_count, 5, gen)
        qx, qy = _sets(splits[query_name], c, query_count, 5, gen)
        row = dict(x=sx, y=sy, qx=qx, qy=qy, task=task_context(sx, sy))
        if test_name is not None:
            tx, ty = _sets(splits[test_name], c, test_count, 5, gen)
            row.update(tx=tx, ty=ty)
        result.append(row)
    return result
