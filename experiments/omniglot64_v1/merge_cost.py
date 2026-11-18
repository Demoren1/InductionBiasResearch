"""Compensated merges and splits for a bank of neuron-block partitions."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import log_softmax


def indicators(labels):
    return np.eye(int(labels.max()) + 1)[labels]


def joint_basis(partitions):
    columns = np.concatenate(
        [indicators(labels) for labels in partitions], axis=1
    )
    basis, singular, _ = np.linalg.svd(columns, full_matrices=False)
    return basis[:, singular > singular[0] * 1e-10]


def constraint_cost(gram, point, difference):
    """Exact constrained refit loss, including the null-space case."""
    eigenvalues, vectors = np.linalg.eigh(gram)
    nonzero = eigenvalues > np.max(eigenvalues) * 1e-12
    normal = vectors.T @ difference
    if np.linalg.norm(normal[~nonzero]) > 1e-10 * np.linalg.norm(difference):
        return 0.0
    denominator = np.sum(normal[nonzero] ** 2 / eigenvalues[nonzero])
    return float((difference @ point) ** 2 / denominator)


@dataclass
class BlockFit:
    labels: np.ndarray
    metric_design: np.ndarray
    precision: np.ndarray
    codes: np.ndarray
    error: np.ndarray


class QuadraticBank:
    """Independent numerical codebooks per neuron, a common map per block."""

    def __init__(self, directions, metric, importance):
        self.directions = directions
        self.metric = metric
        self.importance = importance
        self.weighted = metric @ directions
        self.total_energy = np.sum(directions * self.weighted, axis=1)

    def fit(self, labels):
        design = indicators(labels)
        metric_design = self.metric @ design
        precision = np.linalg.inv(design.T @ metric_design)
        rhs = design.T @ self.weighted
        codes = precision @ rhs
        error = np.maximum(
            self.total_energy - np.sum(codes * rhs, axis=1), 0.0
        )
        return BlockFit(labels, metric_design, precision, codes, error)

    def assignment(self, fits, rematch):
        tasks, _, hidden = self.directions.shape
        blocks = len(fits)
        width = hidden // blocks
        fixed = np.broadcast_to(np.arange(hidden), (tasks, hidden))
        if not rematch:
            return fixed.reshape(tasks, blocks, width)
        costs = np.stack(
            [fit.error * self.importance for fit in fits], axis=-1
        )
        slots = np.repeat(costs, width, axis=-1)
        return np.stack(
            [linear_sum_assignment(cost.T)[1] for cost in slots]
        ).reshape(tasks, blocks, width)

    def losses(self, fits, assignment):
        return np.array(
            [
                np.take_along_axis(
                    fit.error * self.importance, members, axis=1
                ).sum()
                for fit, members in zip(
                    fits, assignment.transpose(1, 0, 2), strict=True
                )
            ]
        )

    def directions_from_fits(self, fits, assignment):
        estimate = np.empty_like(self.directions)
        rows = np.arange(len(estimate))[:, None, None]
        pixels = np.arange(estimate.shape[1])[None, :, None]
        for block, fit in enumerate(fits):
            members = assignment[:, block, None, :]
            estimate[rows, pixels, members] = np.take_along_axis(
                fit.codes, members, axis=2
            )[:, fit.labels, :]
        return estimate

    def best_merge(self, fit, members):
        left, right = np.triu_indices(fit.codes.shape[1], k=1)
        if not len(left):
            return np.inf, None
        diagonal = np.diagonal(fit.precision, axis1=1, axis2=2)
        inverse_curvature = (
            diagonal[:, left]
            + diagonal[:, right]
            - 2 * fit.precision[:, left, right]
        )
        weights = np.take_along_axis(self.importance, members, axis=1)
        codes = np.take_along_axis(fit.codes, members[:, None, :], axis=2)
        weighted = codes * np.sqrt(weights[:, None, :])
        products = weighted @ weighted.transpose(0, 2, 1)
        squared = np.diagonal(products, axis1=1, axis2=2)
        mismatch = (
            squared[:, left] + squared[:, right] - 2 * products[:, left, right]
        )
        prices = (np.maximum(mismatch, 0) / inverse_curvature).sum(axis=0)
        best = int(np.argmin(prices))
        labels = np.where(fit.labels == right[best], left[best], fit.labels)
        labels -= labels > right[best]
        return float(prices[best]), labels

    def best_split(self, fit, members):
        labels = fit.labels
        counts = np.bincount(labels)
        if counts.max() == 1:
            return 0.0, None
        root_weight = np.sqrt(
            np.take_along_axis(self.importance, members, axis=1)
        )
        residual = self.weighted - fit.metric_design @ fit.codes
        residual = (
            np.take_along_axis(residual, members[:, None, :], axis=2)
            * root_weight[:, None, :]
        )
        diagonal = np.diagonal(self.metric, axis1=1, axis2=2)
        projected = fit.metric_design @ fit.precision
        novelty = diagonal - np.sum(projected * fit.metric_design, axis=-1)
        eligible = np.flatnonzero(counts[labels] > 1)
        gain = np.sum(
            np.square(residual[:, eligible]).sum(axis=-1)
            / novelty[:, eligible],
            axis=0,
        )
        best = int(np.argmax(gain))
        benefit, moved = float(gain[best]), eligible[[best]]

        # Also propose one balanced bisection per sizeable group. Score it with
        # the same refit formula as a singleton; it need not be a good split.
        points = (
            np.take_along_axis(self.directions, members[:, None, :], axis=2)
            * root_weight[:, None, :]
        )
        points = points.transpose(1, 0, 2).reshape(len(labels), -1)
        halves = []
        for group in np.flatnonzero(counts >= 4):
            pixels = np.flatnonzero(labels == group)
            centered = points[pixels] - points[pixels].mean(axis=0)
            direction = centered[
                np.argmax(np.einsum("ij,ij->i", centered, centered))
            ]
            order = np.argsort(centered @ direction, kind="stable")
            halves.append(pixels[order[: len(pixels) // 2]])
        if halves:
            split = np.zeros((len(labels), len(halves)))
            for column, pixels in enumerate(halves):
                split[pixels, column] = 1
            cross = split.T @ fit.metric_design
            within = np.stack(
                [
                    self.metric[:, pixels[:, None], pixels].sum((1, 2))
                    for pixels in halves
                ],
                axis=1,
            )
            novelty = within - np.sum((cross @ fit.precision) * cross, axis=-1)
            gains = np.sum(
                np.square(split.T @ residual).sum(axis=-1) / novelty, axis=0
            )
            winner = int(np.argmax(gains))
            if gains[winner] > benefit:
                benefit, moved = float(gains[winner]), halves[winner]
        larger = labels.copy()
        larger[moved] = len(counts)
        return benefit, larger


def optimize(bank, start, *, exchange, rematch, moves):
    """Take the best proposed split/merge at a fixed overall budget."""
    fits = [bank.fit(labels) for labels in start]
    membership = bank.assignment(fits, rematch)
    loss = bank.losses(fits, membership).sum()
    trace = [{"q": float(loss), "groups": [f.codes.shape[1] for f in fits]}]
    proposals = {}
    for _ in range(moves):
        for block, fit in enumerate(fits):
            if block in proposals:
                continue
            members = membership[:, block]
            saving, split = bank.best_split(fit, members)
            price, merged = bank.best_merge(fit, members)
            if split is None:
                larger, within_price, within = None, np.inf, None
            else:
                larger = bank.fit(split)
                within_price, within = bank.best_merge(larger, members)
            proposals[block] = (
                saving,
                larger,
                price,
                merged,
                within_price,
                within,
            )

        options = []
        for b, (
            saving,
            larger,
            _,
            _,
            within_price,
            within,
        ) in proposals.items():
            if larger is None:
                continue
            options.append((saving - within_price, b, b, {b: within}))
            if exchange:
                for other, (_, _, price, merged, _, _) in proposals.items():
                    if b != other and merged is not None:
                        options.append(
                            (saving - price, b, other, {other: merged})
                        )
        if not options:
            break
        gain, receiving, donor, changed = max(
            options, key=lambda move: move[0]
        )
        if gain <= 1e-10 * max(1.0, loss):
            break
        for b, labels in changed.items():
            fits[b] = bank.fit(labels)
        if receiving != donor:
            fits[receiving] = proposals[receiving][1]
        fixed_loss = bank.losses(fits, membership).sum()
        membership = bank.assignment(fits, rematch)
        improved = bank.losses(fits, membership).sum()
        trace.append(
            {
                "q": float(improved),
                "predicted_gain": float(gain),
                "refit_gain": float(loss - fixed_loss),
                "matching_gain": float(fixed_loss - improved),
                "groups": [f.codes.shape[1] for f in fits],
                "split_block": receiving,
                "merge_block": donor,
            }
        )
        loss = improved
        if rematch:
            proposals.clear()
        else:
            for block in {receiving, donor}:
                proposals.pop(block)
    return [fit.labels for fit in fits], trace


def relu_bound(
    images,
    direction,
    approximation,
    hidden_bias,
    head,
    output_bias,
    *,
    origin=None,
):
    """Measure distillation KL and its empirical bound with matched intercepts."""
    if origin is None:
        origin = images.mean(axis=1, keepdims=True)
    centered = images - origin
    a = centered @ direction
    b = centered @ approximation
    adjusted_bias = hidden_bias[:, None, :] + origin @ direction
    norm = np.linalg.norm(head, axis=2)
    q = ((a - b) ** 2).mean(axis=1) * norm
    dense = np.maximum(a + adjusted_bias, 0) @ head + output_bias[:, None, :]
    compressed = (
        np.maximum(b + adjusted_bias, 0) @ head + output_bias[:, None, :]
    )
    teacher, student = (
        log_softmax(dense, axis=-1),
        log_softmax(compressed, axis=-1),
    )
    kl = (np.exp(teacher) * (teacher - student)).sum(axis=-1).mean(axis=1)
    return kl, q.sum(axis=-1) * norm.sum(axis=-1) / 4
