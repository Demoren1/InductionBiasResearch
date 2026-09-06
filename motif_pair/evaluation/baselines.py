"""Leakage-safe mask generators used in motif-pair OOD evaluation."""

from __future__ import annotations

from pathlib import Path

import torch

import config
from models.cvae import load_top_importance, task_condition


def _shape():
    return int(config.SEQ_LEN), int(config.H)


def random_bernoulli(n: int, *, p: float = .375, generator=None, device=None):
    """Independent Bernoulli baseline; expected cardinality is 96."""
    return (torch.rand(n, config.MASK_DIM, generator=generator, device=device) < p).float()


def random_exact(n: int, *, k_active: int | None = None, generator=None, device=None):
    """Uniform fixed-cardinality baseline; this is the primary comparator."""
    k = int(config.K_ACTIVE if k_active is None else k_active)
    scores = torch.rand(n, config.MASK_DIM, generator=generator, device=device)
    top = scores.topk(k, dim=-1).indices
    masks = torch.zeros_like(scores)
    return masks.scatter_(1, top, 1.0)


class ConditionalMean:
    """Train-only mean importance baseline conditioned strictly on the gap.

    A held-out task may use its gap's pool, but all maps in the pool originate
    from ``train_tasks``.  This is intentionally not a per-held-out-task mean.
    """

    def __init__(self, train_tasks, *, ckpt_root: Path | None = None,
                 importance_name: str = "importance.pt", top_frac: float = .1,
                 device: torch.device | str = "cpu", split_path: Path | None = None,
                 expected_provenance: dict | None = None):
        self.train_tasks = list(train_tasks)
        x, c, self.provenance = load_top_importance(self.train_tasks, ckpt_root,
                                                    importance_name, top_frac, device=device,
                                                    split_path=split_path,
                                                    expected_provenance=expected_provenance)
        self.by_condition = {}
        for key in torch.unique(c, dim=0):
            matching = (c == key).all(dim=1)
            self.by_condition[tuple(key.tolist())] = x[matching].mean(dim=0)
        self.global_mean = x.mean(dim=0)

    def __call__(self, task, k_active: int | None = None):
        c = task_condition([task])[0]
        # Every evaluation gap should be represented in meta-train.  The
        # fallback keeps the error readable for deliberately sparse smoke data.
        mean = self.by_condition.get(tuple(c.tolist()), self.global_mean)
        k = int(config.K_ACTIVE if k_active is None else k_active)
        top = mean.topk(k).indices
        mask = torch.zeros_like(mean)
        return mask.scatter_(0, top, 1.0)


def interpolate_train_gap_mean(by_gap: dict[int, torch.Tensor], gap: int) -> tuple[torch.Tensor, dict[str, object]]:
    """Return a train-only continuous mean map for ``gap``.

    The policy is deliberately conservative.  An observed training gap uses
    its own mean.  A gap strictly inside the observed range is linearly
    interpolated between the closest lower and upper *training* gaps.  Outside
    that range we **clamp to the nearest boundary mean** rather than fitting or
    extrapolating a slope.  The latter makes an extrapolation evaluation an
    explicit test of the CVAE, not of an arbitrary baseline extrapolator.
    """
    if not by_gap:
        raise ValueError("cannot interpolate a gap mean with no train maps")
    available = sorted(int(value) for value in by_gap)
    if gap in by_gap:
        return by_gap[gap], {"kind": "observed", "source_gaps": [gap]}
    if gap < available[0]:
        boundary = available[0]
        return by_gap[boundary], {
            "kind": "boundary_clamp_low", "source_gaps": [boundary],
            "extrapolation_policy": "nearest_train_gap_clamp",
        }
    if gap > available[-1]:
        boundary = available[-1]
        return by_gap[boundary], {
            "kind": "boundary_clamp_high", "source_gaps": [boundary],
            "extrapolation_policy": "nearest_train_gap_clamp",
        }
    lower = max(value for value in available if value < gap)
    upper = min(value for value in available if value > gap)
    weight = (gap - lower) / (upper - lower)
    return (1.0 - weight) * by_gap[lower] + weight * by_gap[upper], {
        "kind": "linear_interpolation", "source_gaps": [lower, upper], "weight_upper": weight,
    }


class TrainOnlyGapMean:
    """Gap mean baseline for gap-OOD evaluation without target-gap artifacts.

    Unlike :class:`ConditionalMean`, this class is intended for splits where
    an evaluation gap is absent from meta-training.  Its input consists only
    of selected importance maps from ``train_tasks``.  Internal held-out gaps
    are interpolated with :func:`interpolate_train_gap_mean`; boundary gaps use
    the documented nearest-train-gap clamp rule.
    """

    interpolation_policy = "linear_between_nearest_train_gaps; boundary=nearest_train_gap_clamp"

    def __init__(self, train_tasks, *, ckpt_root: Path | None = None,
                 importance_name: str = "importance.pt", top_frac: float = .1,
                 device: torch.device | str = "cpu", split_path: Path | None = None,
                 expected_provenance: dict | None = None,
                 condition_encoding: str | None = None):
        self.train_tasks = list(train_tasks)
        x, _, self.provenance = load_top_importance(
            self.train_tasks, ckpt_root, importance_name, top_frac, device=device,
            split_path=split_path, expected_provenance=expected_provenance,
            condition_encoding=condition_encoding,
        )
        if len(self.provenance) != len(self.train_tasks):
            raise RuntimeError("importance provenance/task list length mismatch")
        self.by_gap: dict[int, torch.Tensor] = {}
        self._counts: dict[int, int] = {}
        start = 0
        # load_top_importance preserves task order and returns exactly the
        # selected count recorded in each provenance row.  Segmenting by that
        # record avoids inferring an integer gap from a floating scalar
        # condition tensor.
        for task, source in zip(self.train_tasks, self.provenance):
            n_selected = int(source["n_selected"])
            end = start + n_selected
            gap = config.parse_task(task).gap
            values = x[start:end]
            start = end
            if gap not in self.by_gap:
                self.by_gap[gap] = values.sum(dim=0)
                self._counts[gap] = len(values)
            else:
                self.by_gap[gap] = self.by_gap[gap] + values.sum(dim=0)
                self._counts[gap] += len(values)
        if start != len(x):
            raise RuntimeError("importance provenance does not account for every selected map")
        self.by_gap = {gap: total / self._counts[gap] for gap, total in self.by_gap.items()}
        self.available_gaps = tuple(sorted(self.by_gap))

    def mean_for_gap(self, gap: int) -> tuple[torch.Tensor, dict[str, object]]:
        return interpolate_train_gap_mean(self.by_gap, int(gap))

    def describe_gap(self, gap: int) -> dict[str, object]:
        """Return auditable source metadata without exposing a mask."""
        _, description = self.mean_for_gap(gap)
        return {"target_gap": int(gap), "policy": self.interpolation_policy, **description}

    def __call__(self, task, k_active: int | None = None):
        mean, _ = self.mean_for_gap(config.parse_task(task).gap)
        k = int(config.K_ACTIVE if k_active is None else k_active)
        top = mean.topk(k).indices
        mask = torch.zeros_like(mean)
        return mask.scatter_(0, top, 1.0)


def gold_mask(task):
    """Resolve the known synthetic gold support without exposing it to models."""
    from data.generate import ideal_mask
    return ideal_mask(task).reshape(-1).float()
