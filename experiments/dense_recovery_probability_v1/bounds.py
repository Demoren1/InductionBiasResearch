"""Finite-sample bounds for the stated dense-recovery models."""

from __future__ import annotations

import math

from scipy.special import erf


def relation_counts(codes, order):
    """Count primitive balanced relations modulo sign, by squared norm."""
    counts = {
        2: math.comb(codes, 2),
        4: 3 * math.comb(codes, 4),
        6: 3 * math.comb(codes, 3),
    }
    if order == 6:
        counts[6] += 10 * math.comb(codes, 6)
        counts.update(
            {
                8: 20 * math.comb(codes, 5),
                10: 12 * math.comb(codes, 4),
                12: 4 * math.comb(codes, 4),
                14: 6 * math.comb(codes, 3),
            }
        )
    elif order != 4:
        raise ValueError("order must be 4 or 6.")
    return counts


def gaussian_gap_failure(codes, tasks, delta, scale, order=4):
    counts = relation_counts(codes, order)
    return min(
        1.0,
        sum(
            count * erf(delta / (scale * math.sqrt(2 * norm))) ** tasks
            for norm, count in counts.items()
        ),
    )


def uniform_gap_failure(codes, tasks, delta, length, order=4):
    """Use conditional anti-concentration on a largest coefficient."""
    counts = relation_counts(codes, order)
    if order == 4:
        by_maximum = {
            1: counts[2] + counts[4],
            2: counts[6],
        }
    else:
        by_maximum = {
            1: (
                math.comb(codes, 2)
                + 3 * math.comb(codes, 4)
                + 10 * math.comb(codes, 6)
            ),
            2: (
                3 * math.comb(codes, 3)
                + 12 * math.comb(codes, 4)
                + 20 * math.comb(codes, 5)
            ),
            3: 6 * math.comb(codes, 3) + 4 * math.comb(codes, 4),
        }
    return min(
        1.0,
        sum(
            count * min(1.0, 2 * delta / (length * coefficient)) ** tasks
            for coefficient, count in by_maximum.items()
        ),
    )


def overlap_failure(codes, neurons, active_per_neuron):
    required = (codes + 1) // 2
    local = (
        neurons
        * math.comb(codes, required)
        * (required / codes) ** active_per_neuron
    )
    unused = codes * (1 - 1 / codes) ** (neurons * active_per_neuron)
    return min(1.0, local + unused)


def sign_distinct_probability(codes, tasks):
    """Exact birthday probability for normalized code sign trajectories."""
    boxes = 2 ** (tasks - 1)
    if boxes < codes:
        return 0.0
    return math.prod(1 - index / boxes for index in range(codes))


def ols_error_bound(inputs, neurons, tasks, samples, noise, beta, kappa=1.0):
    coordinates = inputs * neurons * tasks
    return (
        noise
        / (kappa * math.sqrt(samples))
        * math.sqrt(2 * math.log(2 * coordinates / beta))
    )
