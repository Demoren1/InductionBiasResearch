"""Signed equality graph with a registered group-count stopping certificate.

The finite-sample claim requires a linear model, iid Gaussian noise with one
unknown variance, and exactly K nonempty, distinct signed coefficient profiles.
"""

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import f


@torch.no_grad()
def full_relations(fitted, inverse_gram, variance, degrees, alpha):
    tasks, outputs, inputs = fitted.shape
    entries = outputs * inputs
    profiles = fitted.flatten(1)
    covariance = torch.zeros(
        (tasks, entries, entries), device=fitted.device, dtype=fitted.dtype
    )
    for output in range(outputs):
        selected = slice(output * inputs, (output + 1) * inputs)
        covariance[:, selected, selected] = inverse_gram
    diagonal = covariance.diagonal(dim1=-2, dim2=-1)
    cutoff = float(f.isf(alpha / entries**2, tasks, degrees))
    zero_q = (profiles.square() / diagonal).mean(0) / variance
    accepted_zero = (zero_q <= cutoff).cpu().numpy()
    relations = {}
    for sign in (-1, 1):
        difference = profiles[:, :, None] - sign * profiles[:, None, :]
        contrast_variance = (
            diagonal[:, :, None] + diagonal[:, None, :] - 2 * sign * covariance
        )
        # The self positive contrast has variance zero and is omitted below.
        statistic = (difference.square() / contrast_variance.clamp_min(1e-30)).mean(
            0
        ) / variance
        relations[sign] = (statistic <= cutoff).cpu().numpy()
    return (
        accepted_zero,
        relations,
        {
            "cutoff": cutoff,
            "alpha_this_look": alpha,
            "hypotheses": entries**2,
            "degrees": degrees,
            "estimated_variance": float(variance),
        },
    )


def resolve_graph(accepted_zero, relations, codes, shape):
    entries = len(accepted_zero)
    adjacency = np.eye(2 * entries + 1, dtype=bool)
    anchor = 2 * entries
    zero_entries = np.flatnonzero(accepted_zero)
    for shifted in (zero_entries, zero_entries + entries):
        adjacency[shifted, anchor] = True
        adjacency[anchor, shifted] = True
    for sign, accepted in relations.items():
        left, right = np.nonzero(np.triu(accepted, k=1))
        if sign == 1:
            pairs = ((left, right), (left + entries, right + entries))
        else:
            pairs = ((left, right + entries), (left + entries, right))
        for first, second in pairs:
            adjacency[first, second] = True
            adjacency[second, first] = True
    count, components = connected_components(csr_matrix(adjacency), directed=False)
    index = None
    valid_involution = True
    if count == 2 * codes + 1:
        zero = components[anchor]
        mapping, next_label = {zero: 0}, 1
        for entry in range(entries):
            positive, negative = components[entry], components[entry + entries]
            if positive == zero:
                valid_involution &= negative == zero
            elif positive == negative:
                valid_involution = False
            elif positive not in mapping:
                mapping[positive] = next_label
                mapping[negative] = -next_label
                next_label += 1
            else:
                valid_involution &= mapping.get(negative) == -mapping[positive]
        valid_involution &= next_label == codes + 1
        if valid_involution:
            index = np.array(
                [mapping[component] for component in components[:entries]],
                dtype=np.int16,
            ).reshape(shape)
    return (
        index,
        {
            "components": int(count),
            "required_components": 2 * codes + 1,
            "valid_involution": bool(valid_involution),
            "returned_index": index is not None,
            "zero_relations": int(accepted_zero.sum()),
            "pair_relations": {
                str(sign): int(np.triu(accepted, k=1).sum())
                for sign, accepted in relations.items()
            },
        },
        components[:entries].reshape(shape),
    )


def signed_graph(fitted, inverse_gram, variance, degrees, codes, alpha):
    accepted_zero, relations, statistic = full_relations(
        fitted, inverse_gram, variance, degrees, alpha
    )
    index, metadata, components = resolve_graph(
        accepted_zero, relations, codes, fitted.shape[1:]
    )
    return index, {**metadata, **statistic}, components


@torch.no_grad()
def noise_whitened_basis(profiles, noise, codes, diagonal_only=False):
    if diagonal_only:
        noise = torch.diag(noise.diagonal())
    values, vectors = torch.linalg.eigh(noise)
    if float(values.min()) <= 0:
        raise ValueError("Whitening requires positive definite estimated noise")
    root = (vectors * values.sqrt()[None]) @ vectors.T
    inverse = (vectors * values.rsqrt()[None]) @ vectors.T
    matrix = profiles @ profiles.T / profiles.shape[1]
    transformed = inverse @ matrix @ inverse
    spectrum, u = torch.linalg.eigh((transformed + transformed.T) * 0.5)
    physical = root @ u[:, -codes:]
    orthogonal, _ = torch.linalg.qr(physical)
    return orthogonal, spectrum
