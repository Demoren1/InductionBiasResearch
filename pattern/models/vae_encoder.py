import torch
from torch import nn

class VAEEncoder(nn.Module):
    def __init__(self, channels, width, latent_dim):
        super().__init__()
        self.body = nn.Sequential(nn.Flatten(), nn.Linear(11*channels,width),nn.GELU())
        self.mu = nn.Linear(width,latent_dim)
        self.logvar = nn.Linear(width,latent_dim)

    def forward(self, nf_features):
        # Hidden-column mean makes latent distributions invariant to reordering.
        hidden = self.body(nf_features.mean(2))
        return self.mu(hidden), self.logvar(hidden).clamp(-12,8)
