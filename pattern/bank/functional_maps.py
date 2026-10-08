"""Functional derivative profiles of trained, masked ReLU networks."""
from __future__ import annotations
import torch
from torch import Tensor
from torch.nn import functional as F
from .imp import logits
_FEATURES, _HIDDEN = 11, 8
NF_CHANNELS = ("weights", "loss_gradient", "functional_map")
NF_BANK_SCHEMA = "pattern.imp_bank.v2"

def _profiles(state, mask, probe_x):
    """Compute unscaled profiles without constructing unused generator tokens."""
    if not isinstance(state, dict) or any((name not in state for name in ('w', 'b', 'a', 'c'))):
        raise ValueError('pattern state must contain terminal w, b, a and c tensors')
    weight, bias, readout, offset, mask, probe_x = (
        torch.as_tensor(value, dtype=torch.float32).detach().cpu()
        for value in (*(state[key] for key in ('w', 'b', 'a', 'c')), mask, probe_x))
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
    return {'psi': psi.contiguous(), 'q_signed': signed.contiguous(), 'q_abs': absolute.contiguous(), 'q_rms': rms.contiguous(), 'psi_scale': psi_scale.reshape(1), 'q_scale': q_scale.reshape(1), 'effective_weights': effective.contiguous()}

def extract_pattern_tokens(state: dict[str, Tensor], mask: Tensor, probe_x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    """Legacy tokens: normalized activations, signed/absolute/RMS maps and mask."""
    raw = _profiles(state, mask, probe_x)
    profiles = [raw[key].div(raw['psi_scale' if key == 'psi' else 'q_scale']).T
                for key in ('psi', 'q_signed', 'q_abs', 'q_rms')]
    return torch.cat((*profiles, torch.as_tensor(mask, dtype=torch.float32).detach().cpu().T), dim=1).contiguous(), raw

def extract_maps(state, mask, probe):
    raw = _profiles(state, mask, probe)
    return {key: raw[key] for key in ("psi", "q_signed", "q_abs", "q_rms", "psi_scale", "q_scale")}


@torch.enable_grad()
def extract_nf_channels(state, mask, support_x, support_y, functional_map):
    """Capture terminal sparse W, d(mean support BCE)/dW and E|q|.

    The gradient includes the fixed pruning mask, so inactive connections have
    zero gradient. It excludes the L2 training penalty and never uses query/test.
    Differentiating a copy leaves the trained network and its gradients untouched.
    """
    state = {key:value.detach().cpu() for key,value in state.items()}
    weight = state["w"].clone().requires_grad_()
    mask = mask.detach().float().cpu()
    prediction = logits({**state,"w":weight},mask,support_x.detach().cpu())
    loss = F.binary_cross_entropy_with_logits(prediction,support_y.detach().cpu())
    gradient, = torch.autograd.grad(loss,weight)
    channels = {"weights":weight.detach()*mask,"loss_gradient":gradient,
                "functional_map":functional_map.detach().cpu()}
    if any(value.shape != (_FEATURES,_HIDDEN) or not torch.isfinite(value).all() for value in channels.values()):
        raise ValueError("NF channels must be finite tensors with shape [11,8]")
    return {**channels,"support_bce":float(loss.detach())}
