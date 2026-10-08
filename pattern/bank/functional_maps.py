"""Functional derivative profiles of trained, masked ReLU networks."""
from __future__ import annotations
import torch
from torch import Tensor
from torch.nn import functional as F
_FEATURES, _HIDDEN = 11, 8

def extract_pattern_tokens(state: dict[str, Tensor], mask: Tensor, probe_x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    """Make normalized generator tokens from a terminal pattern-child state.

    ``psi`` and all ``q_*`` entries in the returned raw profile are unscaled.
    The returned token concatenates normalized full probe activations, signed,
    absolute and RMS functional maps, and the binary mask.  Initial teachers
    and feedback always pass through this one function.
    """
    if not isinstance(state, dict) or any((name not in state for name in ('w', 'b', 'a', 'c'))):
        raise ValueError('pattern state must contain terminal w, b, a and c tensors')
    weight = torch.as_tensor(state['w'], dtype=torch.float32).detach().cpu()
    bias = torch.as_tensor(state['b'], dtype=torch.float32).detach().cpu()
    readout = torch.as_tensor(state['a'], dtype=torch.float32).detach().cpu()
    offset = torch.as_tensor(state['c'], dtype=torch.float32).detach().cpu()
    mask = torch.as_tensor(mask, dtype=torch.float32).detach().cpu()
    probe_x = torch.as_tensor(probe_x, dtype=torch.float32).detach().cpu()
    if weight.shape != (_FEATURES, _HIDDEN) or mask.shape != weight.shape:
        raise ValueError('pattern state and mask must have shape [11, 8]')
    if bias.shape != (_HIDDEN,) or readout.shape != (_HIDDEN,) or offset.numel() != 1:
        raise ValueError('pattern state biases/readout must have shape [8]')
    if probe_x.ndim != 2 or probe_x.shape[1] != _FEATURES or len(probe_x) < 1:
        raise ValueError('probe_x must have shape [probe_rows, 11]')
    if not all((torch.isfinite(item).all() for item in (weight, bias, readout, offset, mask, probe_x))):
        raise ValueError('functional profile inputs must be finite')
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError('pattern masks must be binary')
    effective = weight * mask
    preactivation = probe_x @ effective + bias
    psi = F.relu(preactivation) * readout
    q = probe_x[:, :, None] * effective[None] * readout[None, None] * (preactivation > 0)[:, None]
    signed, absolute, rms = (q.mean(0), q.abs().mean(0), q.square().mean(0).sqrt())
    psi_scale = psi.square().mean().sqrt().clamp_min(1e-08)
    q_scale = rms.amax().clamp_min(1e-08)
    tokens = torch.cat((psi.div(psi_scale).T, signed.div(q_scale).T, absolute.div(q_scale).T, rms.div(q_scale).T, mask.T), dim=1).contiguous()
    raw = {'psi': psi.contiguous(), 'q_signed': signed.contiguous(), 'q_abs': absolute.contiguous(), 'q_rms': rms.contiguous(), 'psi_scale': psi_scale.reshape(1), 'q_scale': q_scale.reshape(1), 'effective_weights': effective.contiguous()}
    return (tokens, raw)

def extract_maps(state, mask, probe):
    _, raw = extract_pattern_tokens(state, mask, probe)
    return {key: raw[key] for key in ("psi", "q_signed", "q_abs", "q_rms", "psi_scale", "q_scale")}
