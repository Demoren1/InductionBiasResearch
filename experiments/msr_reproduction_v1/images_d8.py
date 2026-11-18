"""Disclosed D8 baseline reconstruction using E2CNN's actual kernel bases."""

import json
import sys

import torch
from torch.nn import functional as F

BASES = {}


def configuration(root, dataset=None, shots=None):
    path = root / "d8_architecture.json"
    if not path.exists():
        return "activation_matched"
    config = json.loads(path.read_text())
    layouts = config.get("layouts", {})
    return layouts.get(
        f"{dataset}_{shots}shot",
        layouts.get(dataset, config.get("layout", "activation_matched")),
    )


def initialize(root, dataset, shots, ways, device):
    sys.path.insert(0, str(root / "dependencies"))
    from e2cnn import gspaces
    from e2cnn import nn as enn
    from e2cnn.nn.init import _generalized_he_init_variances

    space = gspaces.FlipRot2dOnR2(N=8)
    omni = dataset == "omniglot"
    channels = [64] * 4 if omni else [32] * 4
    kernels = [3] * 4
    if not omni and shots == 5:
        channels[:3], kernels[:3] = [128] * 3, [5] * 3
        channels.append(64)
        kernels.append(1)
    layout = configuration(root, dataset, shots)
    if layout not in (
        "activation_matched",
        "trivial_readout",
        "wider_trivial_readout",
    ):
        raise ValueError(f"Unknown D8 reconstruction layout: {layout}")
    if layout == "wider_trivial_readout":
        channels[:-1] = [count * 4 for count in channels[:-1]]
    parameters = {}
    previous_fields, previous_repr = (1 if omni else 3), 1
    for layer, (channels_out, size) in enumerate(zip(channels, kernels)):
        in_rep = (
            space.trivial_repr if previous_repr == 1 else space.regular_repr
        )
        last_trivial = (
            layout != "activation_matched" and layer == len(channels) - 1
        )
        out_rep = space.trivial_repr if last_trivial else space.regular_repr
        output_repr = out_rep.size
        conv = enn.R2Conv(
            enn.FieldType(space, [in_rep]),
            enn.FieldType(space, [out_rep]),
            size,
            bias=False,
        )
        count = conv.weights.numel()
        with torch.no_grad():
            basis = torch.stack(
                [
                    conv.basisexpansion(row).reshape(
                        output_repr, previous_repr, -1
                    )
                    for row in torch.eye(count)
                ]
            )
        deviation = _generalized_he_init_variances(conv.basisexpansion).to(
            device
        )
        fields = channels_out // output_repr
        coefficients = torch.randn(
            fields, previous_fields, count, device=device
        )
        coefficients *= deviation / previous_fields**0.5
        prefix = f"conv{layer}"
        parameters[prefix + ".coefficients"] = coefficients
        parameters[prefix + ".gamma"] = torch.ones(fields, device=device)
        parameters[prefix + ".beta"] = torch.zeros(fields, device=device)
        BASES[prefix] = basis.to(device)
        previous_fields, previous_repr = fields, output_repr
    weight = torch.empty(ways, previous_fields, device=device)
    torch.nn.init.xavier_uniform_(weight)
    parameters["head.weight"] = weight
    parameters["head.bias"] = torch.zeros(ways, device=device)
    parameters = {
        name: value.requires_grad_() for name, value in parameters.items()
    }
    rates = {
        name: torch.tensor(
            0.4 if omni else 0.05, device=device, requires_grad=True
        )
        for name in parameters
    }
    return parameters, rates, len(channels)


def features(parameters, images, layers, dataset):
    x = images
    for layer in range(layers):
        prefix = f"conv{layer}"
        coefficients = parameters[prefix + ".coefficients"]
        basis = BASES[prefix]
        fields, inputs, _ = coefficients.shape
        size = int(basis.shape[-1] ** 0.5)
        output_repr = basis.shape[1]
        weight = torch.einsum("oip,prsk->orisk", coefficients, basis)
        weight = weight.reshape(
            fields * output_repr, inputs * basis.shape[-2], size, size
        )
        x = F.conv2d(
            x,
            weight,
            stride=2 if dataset == "omniglot" else 1,
            padding=size // 2,
        )
        shape = x.shape
        x = x.reshape(shape[0], fields, output_repr, *shape[-2:])
        mean = x.mean((0, 2, 3, 4), keepdim=True)
        variance = (x - mean).square().mean((0, 2, 3, 4), keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + 1e-5)
        x = x * parameters[prefix + ".gamma"][None, :, None, None, None]
        x = F.relu(x + parameters[prefix + ".beta"][None, :, None, None, None])
        x = x.reshape(shape)
        if dataset != "omniglot" and layer < 4:
            x = F.max_pool2d(x, 2)
    # Both spatial and group pooling are required for a D8-invariant classifier.
    x = x.mean((-2, -1)).reshape(x.shape[0], fields, output_repr)
    return x.amax(-1)
