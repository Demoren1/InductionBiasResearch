"""Jointly supported partial indexators in a finite proposal bank."""

from itertools import combinations
from math import ceil
from pathlib import Path

import numpy as np


OUTPUT = Path("artifacts/asrr_compatible_2026-11-19")
LOCAL = OUTPUT.parent / "local" / OUTPUT.name


class RelationBank:
    """An atomic relation is a signed tie, or one declared block assignment."""

    def __init__(self, relations):
        self.full = tuple(frozenset(model) for model in relations)
        subsets = {
            frozenset(subset)
            for model in self.full
            for k in range(len(model) + 1)
            for subset in combinations(model, k)
        }
        self.partial = sorted(subsets, key=lambda s: (len(s), tuple(sorted(s))))
        self.completes = np.array(
            [[partial <= full for full in self.full] for partial in self.partial]
        )
        self.parent_rows, self.child_rows = [], []
        position = {s: i for i, s in enumerate(self.partial)}
        for i, s in enumerate(self.partial):
            for atom in s:
                self.parent_rows.append(position[s - {atom}])
                self.child_rows.append(i)

    def infer(self, loss, tolerance, coverage=1.0):
        """A task has one witness for *all* the relations in each partial."""
        loss = np.asarray(loss, dtype=float)
        if (
            loss.ndim != 2
            or loss.shape[1] != len(self.full)
            or not np.isfinite(loss).all()
        ):
            raise ValueError(
                "Expected finite held-out losses for every task and proposal"
            )
        if tolerance < 0 or not 0 < coverage <= 1:
            raise ValueError("Invalid risk tolerance or task coverage")
        good = loss <= loss.min(axis=1, keepdims=True) + tolerance
        # Logical matrix product. One witness per task suffices.
        witnesses = good @ self.completes.T
        eligible = witnesses.sum(axis=0) >= ceil(coverage * len(loss))
        maximal = eligible.copy()
        np.logical_and.at(maximal, self.parent_rows, ~eligible[self.child_rows])
        rows = np.flatnonzero(maximal)
        # Necessity is a separate question: every good solution of every task
        # must contain the relation. For trimmed coverage only admission is trimmed.
        necessary = frozenset.intersection(
            *(self.full[i] for i in np.flatnonzero(good.any(axis=0)))
        )
        counts = good.sum(axis=0)
        possible = np.flatnonzero(self.completes[rows].any(axis=0))
        fixed = min(possible, key=lambda j: (-counts[j], loss[:, j].mean(), j))
        return {
            "maximal": tuple(self.partial[i] for i in rows),
            "necessary": necessary,
            "backbone": frozenset.intersection(*(self.partial[i] for i in rows)),
            "fixed": int(fixed),
            "full_coverage": counts / len(loss),
            "solutions_per_task": good.sum(axis=1),
            "witnesses": witnesses[:, rows],
        }

    def witness_cover(self, proxy, budget):
        """Greedily cover conditional witnesses before expensive supervised refits."""
        m = len(self.full)
        if not 1 <= budget <= m:
            raise ValueError("Invalid numerical refit budget")
        ranks = np.argsort(
            np.argsort(proxy, axis=1, kind="stable"), axis=1, kind="stable"
        )
        quality = np.exp(-ranks / ceil(np.sqrt(m)))
        # Candidate, task, partial. Empty and complete specifications are included.
        utility = quality.T[:, :, None] * self.completes.T[:, None, :]
        # A relation that can be completed well in every task gets more weight.
        weights = utility.max(axis=0).min(axis=0) ** 2
        covered = np.zeros(utility.shape[1:])
        selected = []
        for _ in range(budget):
            gain = (np.maximum(utility - covered, 0) * weights).sum(axis=(1, 2))
            gain[selected] = -np.inf
            choice = int(gain.argmax())
            selected.append(choice)
            np.maximum(covered, utility[choice], out=covered)
        return selected


def common_index(left, right):
    """Encode exactly the signed equalities and fixed zeros shared by both maps."""
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape:
        raise ValueError("Indexators must use corresponding parameter positions")
    labels = np.stack((left.ravel(), right.ravel()), axis=1).astype(np.int64)
    active = np.any(labels != 0, axis=1)
    result = np.zeros(len(labels), dtype=np.int64)
    if active.any():
        rows = labels[active]
        first = np.argmax(rows != 0, axis=1)
        sign = np.sign(rows[np.arange(len(rows)), first])
        _, inverse = np.unique(rows * sign[:, None], axis=0, return_inverse=True)
        # Canonicalize by first occurrence and its sign, independent of both gauges.
        _, first_position = np.unique(inverse, return_index=True)
        order = np.argsort(first_position)
        rank = np.empty_like(order)
        rank[order] = np.arange(1, len(order) + 1) * sign[first_position[order]]
        result[active] = rank[inverse] * sign
    return result.reshape(left.shape)


def common_frontier(versions):
    """Exact strict-coverage inference on full signed maps, without listing edges."""
    versions = sorted(versions, key=len)
    if not versions or any(not group for group in versions):
        raise ValueError("Each task needs a nonempty version set")
    current = [np.zeros_like(versions[0][0], dtype=np.int64)]
    for group in versions:
        candidates = {}
        for a in current:
            for b in group:
                c = common_index(a, b)
                candidates[c.tobytes()] = c
        # More independent codes means fewer constraints. At equal dimension,
        # strictly nested linear subspaces cannot occur.
        ordered = sorted(candidates.values(), key=lambda c: np.abs(c).max(initial=0))
        current = []
        for c in ordered:
            if not any(
                np.array_equal(common_index(c, stronger), c) for stronger in current
            ):
                current.append(c)
    return current
