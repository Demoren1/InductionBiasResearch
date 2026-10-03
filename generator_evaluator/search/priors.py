"""Structural candidate families, independent of task labels and learned weights."""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from generator_evaluator.data.types import topology_id


class MaskPrior(ABC):
    name: str

    @abstractmethod
    def mask(self, *, device="cpu") -> torch.Tensor:
        """Return a hard mask in the domain's fixed input coordinates."""

    def matches(self, candidate: torch.Tensor) -> bool:
        reference = self.mask()
        return candidate.shape == reference.shape and topology_id(candidate) == topology_id(reference)

    def _alignment(self, candidate: torch.Tensor):
        from scipy.optimize import linear_sum_assignment
        candidate = candidate.detach().cpu()
        reference = self.mask()
        if candidate.shape != reference.shape:
            raise ValueError("candidate and structural prior have different shapes")
        topology_id(candidate)  # Validate binary, finite, nonempty input.
        cost = (reference.T[:, None, :] != candidate.T[None, :, :]).sum(-1).numpy()
        rows, columns = linear_sum_assignment(cost)
        return reference, candidate[:, columns], columns, int(cost[rows, columns].sum())

    def aligned_mask(self, candidate: torch.Tensor) -> torch.Tensor:
        """Align hidden columns to the reference without relabelling input bits."""
        return self._alignment(candidate)[1]

    def diagnostics(self, candidate: torch.Tensor) -> dict:
        """Measure active-edge overlap after optimal hidden-column matching.

        The score is the Dice/F1 overlap of active edges, rather than the
        fraction of equal cells, which would reward shared inactive cells.
        For two 32-edge masks it is exactly matched_edges / 32.
        """
        reference, aligned, columns, differing = self._alignment(candidate)
        reference_edges, candidate_edges = int(reference.sum()), int(aligned.sum())
        matched = int(((reference == 1) & (aligned == 1)).sum())
        return dict(prior=self.name, matches=differing == 0, differing_edges=differing,
                    normalized_hamming=differing / reference.numel(),
                    toeplitz_score=2 * matched / (reference_edges + candidate_edges),
                    matched_edges=matched, missing_edges=reference_edges - matched,
                    extra_edges=candidate_edges - matched,
                    reference_edges=reference_edges, candidate_edges=candidate_edges,
                    column_permutation=columns.tolist(),
                    score_definition="2 * matched_edges / (reference_edges + candidate_edges)",
                    identity="modulo hidden-column permutation")


@dataclass(frozen=True)
class SlidingWindowMaskPrior(MaskPrior):
    """One hidden neuron per sliding window: a rectangular banded Toeplitz mask."""
    features: int = 11
    window: int = 4
    name: str = "toeplitz"

    def __post_init__(self):
        if not 1 <= self.window <= self.features:
            raise ValueError("window must fit inside the input sequence")

    @property
    def hidden(self) -> int:
        return self.features - self.window + 1

    def mask(self, *, device="cpu") -> torch.Tensor:
        offset = torch.arange(self.features, device=device)[:, None] - torch.arange(self.hidden, device=device)[None]
        return ((offset >= 0) & (offset < self.window)).float()
