"""Independent NF encoders, a shared decoder, and matched IMP supervision."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..masks import assignments, topk
from ..problem import EDGES, HIDDEN, INPUTS


class GroupedLinear(nn.Module):
    """One independent linear map per function, evaluated with batched GEMM."""

    def __init__(self, tasks, incoming, outgoing, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(tasks, outgoing, incoming))
        self.bias = (
            nn.Parameter(torch.empty(tasks, outgoing)) if bias else None
        )
        bound = 1 / math.sqrt(incoming)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        shape = x.shape[:-1]
        result = torch.bmm(
            x.reshape(x.shape[0], -1, x.shape[-1]),
            self.weight.transpose(1, 2),
        )
        if self.bias is not None:
            result = result + self.bias[:, None]
        return result.reshape(*shape, self.weight.shape[1])


class GroupedNF(nn.Module):
    """The existing column-equivariant NFLayer applied to independent tasks."""

    def __init__(self, tasks, incoming, outgoing):
        super().__init__()
        self.local = GroupedLinear(tasks, incoming, outgoing)
        self.context = GroupedLinear(tasks, incoming, outgoing, bias=False)

    def forward(self, columns):
        return self.local(columns) + self.context(
            columns.mean(2, keepdim=True)
        )


class Encoder(nn.Module):
    """Encode W/G columns invariantly with independent parameters per task."""

    def __init__(self, tasks, options):
        super().__init__()
        channels, width = options["nf_channels"], options["encoder_width"]
        self.first = GroupedNF(tasks, INPUTS * 2, INPUTS * channels)
        self.second = GroupedNF(tasks, INPUTS * channels, INPUTS * channels)
        self.edges = GroupedLinear(tasks, INPUTS * channels, width)
        self.posterior = GroupedLinear(tasks, width, width)
        self.moments = GroupedLinear(tasks, width, options["latent_dim"] * 2)

    def forward(self, features):
        # [task,batch,input,hidden,channel] -> [task,batch,hidden,input*channel].
        columns = features.permute(0, 1, 3, 2, 4).flatten(-2)
        columns = F.gelu(self.first(columns))
        columns = F.gelu(self.second(columns))
        edges = F.gelu(self.edges(columns.mean(2)))
        hidden = F.gelu(self.posterior(edges))
        mu, logvar = self.moments(hidden).chunk(2, -1)
        return mu, logvar.clamp(-12, 8)


class Decoder(nn.Module):
    def __init__(self, options):
        super().__init__()
        width = options["decoder_width"]
        self.body = nn.Sequential(
            nn.Linear(options["latent_dim"], width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, INPUTS * HIDDEN),
        )

    def forward(self, z):
        return self.body(z).reshape(*z.shape[:-1], INPUTS, HIDDEN)


class MaskVAE(nn.Module):
    def __init__(self, tasks, options):
        super().__init__()
        self.encoder = Encoder(tasks, options)
        self.decoder = Decoder(options)

    def forward(self, features, sample):
        mu, logvar = self.encoder(features)
        z = mu + torch.randn_like(mu) * (logvar * 0.5).exp() if sample else mu
        return self.decoder(z), mu, logvar


def matched_target(scores, target, hard_weight):
    """Minimize the actual weighted BCE plus hard-MSE over column orders."""
    with torch.no_grad():
        prediction = scores.detach().reshape(-1, INPUTS, HIDDEN)
        labels = target.detach().reshape_as(prediction)
        positive_weight = (INPUTS * HIDDEN - EDGES) / EDGES
        cost = torch.bmm(F.softplus(prediction).transpose(1, 2), 1 - labels)
        cost += positive_weight * torch.bmm(
            F.softplus(-prediction).transpose(1, 2), labels
        )
        hard = topk(prediction)
        squared = hard.square().sum(1)[:, :, None]
        squared = squared + labels.square().sum(1)[:, None, :]
        squared -= 2 * torch.bmm(hard.transpose(1, 2), labels)
        cost = (1 - hard_weight) * cost + hard_weight * squared
        order = assignments(cost)
        aligned = labels.gather(2, order[:, None].expand_as(labels))
    return aligned.reshape_as(target)


def objective(output, target, beta, hard_weight):
    scores, mu, logvar = output
    target = matched_target(scores, target, hard_weight)
    bce = F.binary_cross_entropy_with_logits(
        scores,
        target,
        pos_weight=scores.new_tensor((INPUTS * HIDDEN - EDGES) / EDGES),
    )
    hard_mse = F.mse_loss(topk(scores, True), target)
    kl = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(-1).mean()
    reconstruction = (1 - hard_weight) * bce + hard_weight * hard_mse
    loss = reconstruction + beta * kl
    return loss, {
        "loss": loss.detach(),
        "reconstruction": reconstruction.detach(),
        "bce": bce.detach(),
        "hard_mse": hard_mse.detach(),
        "kl": kl.detach(),
    }


@torch.no_grad()
def aligned_iou(scores, target):
    masks = topk(scores).reshape(-1, INPUTS, HIDDEN)
    labels = target.reshape_as(masks)
    overlaps = torch.bmm(masks.transpose(1, 2), labels)
    order = assignments(-overlaps)
    intersections = overlaps.gather(2, order[..., None]).sum((1, 2))
    unions = masks.sum((1, 2)) + labels.sum((1, 2)) - intersections
    return intersections / unions.clamp_min(1)
