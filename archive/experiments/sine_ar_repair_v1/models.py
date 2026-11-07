"""Plain, corrupted-context, independent-output AR and historical NF/VAE."""

import torch
from torch import nn
from torch.nn import functional as F

from archive.history.experiments.sine_mask_vae.models import MaskVAE
from pattern.autoregressive.model import IndexMADE


def configurations():
    result = []

    def add(family, **options):
        result.append(
            {
                "id": f"model_{len(result):02d}",
                "family": family,
                "width": 128,
                "lr": 0.001,
                "decay": 0.0,
                "corruption": 0.0,
                "beta": 0.05,
                "task_balanced": False,
                **options,
            }
        )

    for width, lr, decay in [
        (32, 0.001, 0),
        (128, 0.0003, 0),
        (128, 0.001, 0),
        (128, 0.003, 0),
        (128, 0.001, 0.01),
        (128, 0.001, 0.1),
        (256, 0.001, 0),
    ]:
        add("joint", width=width, lr=lr, decay=decay)
    for probability in [0.1, 0.3, 0.6]:
        add("token_noise", corruption=probability)
    for probability in [0.25, 0.5, 1.0]:
        add("cross_noise", corruption=probability)
    for lr in [0.0003, 0.001, 0.003]:
        add("block", width=64, lr=lr)
    for family in ["vae", "partwise_vae"]:
        for beta in [0.005, 0.05, 0.2]:
            add(family, beta=beta, decay=0.01)
    for family in ["mask_vae", "partwise_mask_vae"]:
        add(family, decay=0.01)
    add("joint", task_balanced=True)
    add("cross_noise", corruption=1.0, task_balanced=True)
    return result


class Prior(nn.Module):
    def __init__(self, config, tasks=16):
        super().__init__()
        self.config = config
        family = config["family"]
        if "vae" in family:
            self.net = MaskVAE(tasks)
        elif family == "block":
            self.net = nn.ModuleList(
                [IndexMADE(8, 2, config["width"]) for _ in range(2)]
            )
        else:
            self.net = IndexMADE(16, 2, config["width"])

    def logits(self, tokens):
        if self.config["family"] == "block":
            parts = [self.net[j](tokens[:, j::2]) for j in range(2)]
            return torch.stack(parts, dim=2).reshape(-1, 16, 2)
        return self.net(tokens)

    def corrupt(self, tokens, bank, output=None):
        probability = self.config["corruption"]
        if self.config["family"] == "token_noise":
            change = torch.rand_like(tokens.float()) < probability
            nuisance = torch.rand_like(tokens.float()) < bank.float().mean(0)
            return torch.where(change, nuisance.long(), tokens)
        nuisance = bank[
            torch.randint(len(bank), (len(tokens),), device=tokens.device)
        ]
        change = torch.rand(len(tokens), 1, device=tokens.device) < probability
        foreign = torch.arange(16, device=tokens.device) % 2 != output
        return torch.where(change & foreign, nuisance, tokens)

    def loss(self, masks, features, weights, step, clean=False):
        family = self.config["family"]
        target = masks.reshape(-1, 16).long()
        flat_weight = weights.flatten()
        if "vae" not in family:
            if family == "cross_noise" and not clean:
                train_bank = target[flat_weight > 0]
                values = []
                for output in range(2):
                    observed = self.corrupt(target, train_bank, output)
                    values.append(self.logits(observed)[:, output::2])
                scores = torch.stack(values, dim=2).reshape(-1, 16, 2)
            else:
                observed = target
                if family == "token_noise" and not clean:
                    observed = self.corrupt(target, target[flat_weight > 0])
                scores = self.logits(observed)
            losses = (
                F.cross_entropy(
                    scores.flatten(0, 1), target.flatten(), reduction="none"
                )
                .reshape(-1, 16)
                .mean(1)
            )
            return (losses * flat_weight).sum()

        mu, logvar = self.net.encoder(features)
        if clean:
            latent = mu
        else:
            latent = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        if "partwise" in family and not clean:
            nuisance = torch.randn_like(latent)
            first = self.net.decode(
                torch.cat([latent[..., :8], nuisance[..., 8:]], -1)
            )
            second = self.net.decode(
                torch.cat([nuisance[..., :8], latent[..., 8:]], -1)
            )
            scores = torch.stack([first[..., 0], second[..., 1]], -1)
        else:
            scores = self.net.decode(latent)
        bce = F.binary_cross_entropy_with_logits(
            scores,
            masks.float(),
            pos_weight=scores.new_tensor(7),
            reduction="none",
        ).mean((-2, -1))
        flat = scores.flatten(-2)
        indices = flat.detach().argsort(dim=-1, descending=True, stable=True)[
            ..., :2
        ]
        hard = (
            torch.zeros_like(flat).scatter_(-1, indices, 1).reshape_as(scores)
        )
        soft = scores.sigmoid()
        straight = hard + soft - soft.detach()
        error = (straight - masks).square().mean((-2, -1))
        kl = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(-1)
        beta = self.config["beta"] * (1 if clean else min(1, step / 90))
        return ((0.8 * bce + 0.2 * error + beta * kl) * weights).sum()

    @torch.no_grad()
    def sample(self, count, scale, bank, sampling_noise=False):
        device = next(self.parameters()).device
        family = self.config["family"]
        if "vae" in family:
            latent = torch.randn(count, 16, device=device) * scale
            scores = self.net.decode(latent).flatten(1)
            indices = scores.argsort(dim=-1, descending=True, stable=True)[
                :, :2
            ]
            return torch.zeros_like(scores, dtype=torch.long).scatter_(
                1, indices, 1
            )
        if family == "block":
            values = [
                self.net[j].sample_cached(count, scale) for j in range(2)
            ]
            return torch.stack(values, dim=-1).reshape(-1, 16)
        if not sampling_noise:
            return self.net.sample_cached(count, scale)
        values = torch.zeros(count, 16, dtype=torch.long, device=device)
        for position in range(16):
            observed = self.corrupt(values, bank, position % 2)
            logits = self.net(observed)[:, position] / scale
            values[:, position] = torch.multinomial(
                logits.softmax(-1), 1
            ).squeeze(1)
        return values
