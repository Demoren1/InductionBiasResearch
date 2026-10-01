"""Two-objective learning from actual hard-mask utility and input consistency.

The utility value is supplied by a completed, fresh support-only child fit.
No child-gradient or reconstruction target is required. Ordered Gumbel top-K
samples follow a Plackett--Luce law; its exact score-function gradient trains
the mask policy even though the evaluated mask is binary.
"""
from __future__ import annotations

import torch
from torch import Tensor


def sample_ordered_topk(logits: Tensor, k: int, *, generator=None,
                        gumbel: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
    """Return hard masks, exact ordered-sample log probabilities and rankings.

    Leading axes identify independent draws; the final two axes are edges.
    Sampling indices are detached. Differentiation of gathered logits and
    remaining-set normalizers gives the exact ordered-law score. Using an
    ordered auxiliary sample remains valid for an unordered-mask reward.
    """
    if logits.ndim < 3 or not logits.is_floating_point():
        raise ValueError('logits must be floating [..., features, hidden]')
    if not torch.isfinite(logits).all():
        raise ValueError('logits must be finite')
    flat = logits.flatten(-2)
    if not 0 < k <= flat.shape[-1]:
        raise ValueError('k must select between 1 and all edges')
    if gumbel is None:
        u = torch.rand(logits.shape, device=logits.device, dtype=logits.dtype,
                       generator=generator).clamp(torch.finfo(logits.dtype).eps,
                                                  1-torch.finfo(logits.dtype).eps)
        gumbel = -torch.log(-torch.log(u))
    elif gumbel.shape != logits.shape or not torch.isfinite(gumbel).all():
        raise ValueError('gumbel must be finite with the logits shape')
    order = (flat.detach()+gumbel.flatten(-2)).argsort(-1, descending=True)
    ordered_logits = flat.gather(-1, order)
    # Stable O(number of edges) normalizers: no subtracting near-equal sums.
    denominators = ordered_logits.flip(-1).logcumsumexp(-1).flip(-1)
    log_prob = (ordered_logits[..., :k]-denominators[..., :k]).sum(-1)
    mask = torch.zeros_like(flat).scatter(-1, order[..., :k], 1)
    return mask.reshape_as(logits).detach(), log_prob, order[..., :k].detach()


def quality_policy_loss(log_prob: Tensor, query_nmse: Tensor) -> tuple[Tensor, Tensor]:
    """Unbiased minimization estimator with a leave-one-draw-out baseline.

    Arrays have shape [task, independent policy draw], with at least two
    draws. Every query_nmse is the terminal query error of a fresh child,
    averaged over matched initializations; no query-selected checkpoint.
    Current-draw utility never contributes to its own detached baseline.
    The returned scalar is a gradient estimator, not the numerical NMSE.
    """
    if log_prob.ndim != 2 or query_nmse.shape != log_prob.shape or log_prob.shape[1] < 2:
        raise ValueError('log_prob and query_nmse require [task, draws>=2]')
    if not torch.isfinite(log_prob).all() or not torch.isfinite(query_nmse).all():
        raise ValueError('policy log probability and child query loss must be finite')
    reward = query_nmse.detach().to(log_prob)
    baseline = (reward.sum(-1, keepdim=True)-reward)/(reward.shape[-1]-1)
    advantage = (reward-baseline).detach()
    return (advantage*log_prob).mean(), advantage


def dual_objective(log_prob: Tensor, query_nmse: Tensor,
                   embedding: Tensor, permuted_embedding: Tensor,
                   logits: Tensor, permuted_logits: Tensor, *, coefficient=1.0) -> dict[str, Tensor]:
    """Primary actual utility plus matched-input encoder consistency.

    The annotated E(B) objective is used for gradients. Fixed-output response
    consistency is also penalized, comparing probabilities rather than
    policy draws. The same task, teacher subset and decoder slots are used.
    """
    if coefficient < 0:
        raise ValueError('consistency coefficient must be nonnegative')
    if embedding.shape != permuted_embedding.shape or logits.shape != permuted_logits.shape:
        raise ValueError('original and permuted predictions must have matching shapes')
    policy, advantage = quality_policy_loss(log_prob, query_nmse)
    encoder_consistency = (embedding-permuted_embedding).square().mean()
    response_consistency = (logits.sigmoid()-permuted_logits.sigmoid()).square().mean()
    consistency = encoder_consistency+response_consistency
    return dict(loss=policy+coefficient*consistency,
                policy_gradient_loss=policy, quality_nmse=query_nmse.detach().mean(),
                consistency=consistency, encoder_consistency=encoder_consistency,
                response_consistency=response_consistency, advantage=advantage)
