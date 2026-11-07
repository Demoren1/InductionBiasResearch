"""Eliminate code values before solving the column-scale calibration problem."""

import numpy as np
import torch


def components(index, codes):
    parent = np.arange(codes + index.shape[1])

    def root(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for d, h in zip(*np.where(index)):
        parent[root(abs(int(index[d, h])) - 1)] = root(codes + h)
    groups = {}
    for h in range(index.shape[1]):
        groups.setdefault(root(codes + h), []).append(h)
    return list(groups.values())


@torch.no_grad()
def calibrate(index, normal, uncertainty, codes):
    from ..fitting.adaptation import signed_basis

    basis = signed_basis(index, codes, normal.device, normal.dtype)
    tasks, _, hidden = normal.shape
    anchors = torch.zeros((tasks, hidden), dtype=torch.bool, device=normal.device)
    task = torch.arange(tasks, device=normal.device)
    for group in components(index, codes):
        columns = torch.tensor(group, device=normal.device)
        chosen = columns[uncertainty[:, columns].argmin(1)]
        anchors[task, chosen] = True
    cross0 = torch.einsum("dhk,tdh->tkh", basis, normal)
    counts0 = basis.abs().sum(0).T
    norm = normal.square().sum(1)
    scales = torch.ones((tasks, hidden), device=normal.device, dtype=normal.dtype)
    for _ in range(3):
        deviation = uncertainty.clamp_min(0.0003) * scales.abs().clamp_min(0.01)
        weight = deviation.square().reciprocal()
        cross = cross0 * weight[:, None]
        counts = torch.einsum("kh,th->tk", counts0, weight).clamp_min(1e-30)
        schur = torch.diag_embed(norm * weight) - torch.einsum(
            "tkh,tkj->thj", cross / counts[:, :, None], cross
        )
        rhs = -torch.einsum("thj,tj->th", schur, anchors.to(normal.dtype))
        system = schur.clone()
        scale = system.diagonal(dim1=-2, dim2=-1).abs().amax(-1).clamp_min(1)
        system += torch.diag_embed((~anchors).to(normal.dtype) * scale[:, None] * 1e-12)
        system[anchors] = 0
        diagonal = torch.arange(hidden, device=normal.device)
        system[:, diagonal, diagonal] = torch.where(
            anchors, torch.ones_like(scales), system[:, diagonal, diagonal]
        )
        rhs[anchors] = 0
        free = torch.linalg.solve(system, rhs[..., None])[..., 0]
        scales = free + anchors.to(normal.dtype)
        code = torch.einsum("tkh,th->tk", cross, scales) / counts
    return code, scales


@torch.no_grad()
def assign(initial, mask, normal, uncertainty, codes, steps=40):
    index = initial.copy()
    index[~mask] = 0
    if mask.sum() < codes:
        return None
    missing = mask & (index == 0)
    index[missing] = np.resize(np.arange(1, codes + 1), missing.sum())
    symbols = torch.cat((torch.arange(1, codes + 1), -torch.arange(1, codes + 1))).to(
        normal.device
    )
    for step in range(steps):
        code, scales = calibrate(index, normal, uncertainty, codes)
        cost = normal.new_zeros((*index.shape, 2 * codes))
        for begin in range(0, len(normal), 8):
            local = slice(begin, begin + 8)
            centers = torch.cat((code[local], -code[local]), -1)
            target = scales[local, None] * normal[local]
            deviation = uncertainty[local].clamp_min(0.0003) * scales[
                local
            ].abs().clamp_min(0.01)
            standardized = (target[..., None] - centers[:, None, None]) / deviation[
                :, None, :, None
            ]
            cost += (4 * torch.log1p(standardized.square() / 4)).sum(0)
        cost /= len(normal)
        updated = symbols[cost.argmin(-1)].cpu().numpy().astype(np.int16)
        updated[~mask] = 0
        # Keep the prescribed budget available; never use a true group assignment.
        for label in range(1, codes + 1):
            if np.any(np.abs(updated) == label):
                continue
            counts = np.bincount(np.abs(updated).ravel(), minlength=codes + 1)
            allowed = mask & (counts[np.abs(updated)] > 1)
            local = (
                torch.minimum(cost[..., label - 1], cost[..., label - 1 + codes])
                .cpu()
                .numpy()
            )
            position = int(np.where(allowed, local, np.inf).argmin())
            flat = cost.flatten(0, 1)
            updated.ravel()[position] = (
                label
                if flat[position, label - 1] <= flat[position, label - 1 + codes]
                else -label
            )
        if np.array_equal(index, updated):
            break
        index = updated
    code, scales = calibrate(index, normal, uncertainty, codes)
    labels = torch.as_tensor(index, device=normal.device, dtype=torch.long)
    dictionary = torch.cat((code[:, :1] * 0, code, -code), -1)
    categorical = torch.where(labels < 0, codes - labels, labels)
    weight = dictionary[:, categorical]
    safe = torch.where(scales >= 0, 1, -1) * scales.abs().clamp_min(1e-8)
    error = (weight / safe[:, None] - normal) / uncertainty[:, None].clamp_min(0.0003)
    score = float((4 * torch.log1p(error.square() / 4)).mean())
    return index, code.cpu().numpy(), scales.cpu().numpy(), score, step + 1
