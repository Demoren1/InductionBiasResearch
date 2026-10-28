"""Forward model shared by source fitting and independent target fitting."""

import math

import torch


def predict(inputs, weights, activation):
    hidden = torch.einsum("bd,mdh->mbh", inputs, weights)
    if activation == "relu":
        hidden = hidden.relu()
    elif activation == "tanh":
        hidden = hidden.tanh()
    else:
        raise ValueError(f"Unknown activation: {activation}")
    return hidden.sum(-1) / math.sqrt(weights.shape[-1])
