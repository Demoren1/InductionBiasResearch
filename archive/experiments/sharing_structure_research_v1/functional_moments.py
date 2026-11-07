"""Population illustration formulas, never used by the G18 graph learner."""

import math

import numpy as np
from scipy.special import ndtr


def gate_probabilities(threshold, correlation):
    """Deterministic Plackett integral for bivariate Gaussian gate probabilities."""
    nodes, weights = np.polynomial.legendre.leggauss(96)
    end = np.arcsin(np.clip(correlation, -1, 1))
    angles = end[..., None] * (nodes + 1) / 2
    left, right = threshold[:, None, None], threshold[None, :, None]
    exponent = -(left**2 - 2 * left * right * np.sin(angles) + right**2)
    exponent /= 2 * np.cos(angles) ** 2
    integral = end / 2 * (np.exp(exponent) * weights).sum(-1)
    probability = ndtr(threshold)[:, None] * ndtr(threshold)[None] + integral / (
        2 * math.pi
    )
    np.fill_diagonal(probability, ndtr(threshold))
    return probability


def population(weights, bias):
    """m0, m1, Hermite m2/m3 and the label-square Hermite m2, bias common."""
    hidden = weights.shape[1]
    gram = weights.T @ weights
    norm = np.sqrt(np.diag(gram))
    threshold = bias / norm
    density = np.exp(-(threshold**2) / 2) / math.sqrt(2 * math.pi)
    m0 = (norm * density + bias * ndtr(threshold)).sum() / math.sqrt(hidden)
    m1 = weights @ ndtr(threshold) / math.sqrt(hidden)
    m2 = (weights * (density / norm)[None]) @ weights.T / math.sqrt(hidden)
    m3 = np.einsum(
        "ih,jh,kh,h->ijk",
        weights,
        weights,
        weights,
        -bias * density / (math.sqrt(hidden) * norm**3),
    )
    probability = gate_probabilities(threshold, gram / np.outer(norm, norm))
    conditional_mean = bias * (1 - gram / np.diag(gram)[None])
    conditional_variance = np.diag(gram)[:, None] - gram**2 / np.diag(gram)[None]
    conditional_sd = np.sqrt(np.maximum(conditional_variance, 0))
    standardized = conditional_mean / np.maximum(conditional_sd, 1e-30)
    conditional_relu = (
        conditional_sd * np.exp(-(standardized**2) / 2) / math.sqrt(2 * math.pi)
    )
    conditional_relu += conditional_mean * ndtr(standardized)
    conditional_relu = np.where(
        conditional_sd > 1e-14, conditional_relu, np.maximum(conditional_mean, 0)
    )
    coefficient = density / norm * conditional_relu.sum(0)
    squared_h2 = 2 / hidden * weights @ (probability + np.diag(coefficient)) @ weights.T
    gradient_anisotropic = squared_h2 - np.outer(m1, m1)
    return {
        "m0": np.asarray(m0),
        "m1": m1,
        "m2": m2,
        "m3": m3,
        "squared_label_h2": squared_h2,
        "gradient_covariance_minus_output_square_I": gradient_anisotropic,
    }
