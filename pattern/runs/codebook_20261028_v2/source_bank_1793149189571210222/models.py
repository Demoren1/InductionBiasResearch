"""NF encoders and a shared categorical index decoder."""

import torch
from torch import nn

from pattern.models.task_model import TaskEncoder


class IndexVAE(nn.Module):
    def __init__(self, tasks, options):
        super().__init__()
        self.encoders = nn.ModuleList([TaskEncoder(options) for _ in range(tasks)])
        self.decoder = nn.Sequential(
            nn.Linear(options["latent_dim"], options["decoder_width"]),
            nn.GELU(),
            nn.Linear(options["decoder_width"], options["decoder_width"]),
            nn.GELU(),
            nn.Linear(options["decoder_width"], 11 * 8 * 5),
        )

    def decode(self, latent):
        return self.decoder(latent).reshape(*latent.shape[:-1], 11, 8, 5)

    def forward(self, features, sample=True):
        moments = [encoder(features[i]) for i, encoder in enumerate(self.encoders)]
        mu = torch.stack([item[0] for item in moments])
        logvar = torch.stack([item[1] for item in moments]).clamp(-12, 8)
        latent = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if sample else mu
        return self.decode(latent), mu, logvar


def index_probabilities(scores, straight_through=False):
    """Exactly 32 nonzero assignments; four learned nonzero classes."""
    flat = scores.reshape(-1, 88, 5)
    active_scores = flat[..., 1:].max(-1).values - flat[..., 0]
    keep = active_scores.topk(32, dim=1).indices
    labels = torch.zeros(flat.shape[:2], dtype=torch.long, device=scores.device)
    groups = flat[..., 1:].argmax(-1) + 1
    labels.scatter_(1, keep, groups.gather(1, keep))
    hard = torch.nn.functional.one_hot(labels, 5).to(scores.dtype)
    if straight_through:
        soft = flat.softmax(-1)
        hard = hard + soft - soft.detach()
    return hard.reshape_as(scores)
