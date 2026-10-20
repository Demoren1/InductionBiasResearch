"""Extract sparse weights and support-BCE gradients, without extra probes."""

import torch
from torch.nn import functional as F

from ..masks import column_order
from ..models.task import expanded_targets
from ..problem import logits


def extract(state, masks, support):
    if masks.device.type != "cuda":
        raise RuntimeError("Bank feature extraction requires CUDA")
    weights = (state["w"] * masks).detach().requires_grad_()
    prediction = logits(
        {**state, "w": weights}, torch.ones_like(masks), support[0]
    )
    loss = F.binary_cross_entropy_with_logits(
        prediction, expanded_targets(support[1], prediction), reduction="none"
    ).mean(-1)
    (gradient,) = torch.autograd.grad(loss.sum(), weights)
    with torch.no_grad():
        order = column_order(masks)
        indices = order[:, None].expand_as(weights)
        features = torch.stack(
            (
                weights.detach().gather(2, indices),
                (gradient * masks).gather(2, indices),
            ),
            -1,
        )
        result = {
            "features": features,
            "targets": masks.gather(2, indices).bool(),
            "column_orders": order,
            "feature_scale": features.abs()
            .amax((1, 2), keepdim=True)
            .clamp_min(1e-12),
        }
        if not torch.isfinite(features).all():
            raise RuntimeError("Nonfinite bank features")
        return {key: value.cpu().contiguous() for key, value in result.items()}
