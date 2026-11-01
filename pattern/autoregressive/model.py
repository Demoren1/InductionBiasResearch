import torch
from torch import nn
from torch.nn import functional as F


class MaskedLinear(nn.Linear):
    def __init__(self, inputs, outputs, mask):
        super().__init__(inputs, outputs)
        self.register_buffer("mask", mask.float())

    def forward(self, inputs):
        return F.linear(inputs, self.weight * self.mask, self.bias)


class IndexMADE(nn.Module):
    def __init__(self, positions=88, classes=5, width=256):
        super().__init__()
        self.positions = positions
        self.classes = classes
        inputs = torch.arange(1, positions + 1).repeat_interleave(classes)
        first = torch.arange(width) % (positions - 1) + 1
        second = torch.randperm(width) % (positions - 1) + 1
        self.layers = nn.Sequential(
            MaskedLinear(positions * classes, width, first[:, None] >= inputs),
            nn.GELU(),
            MaskedLinear(width, width, second[:, None] >= first),
            nn.GELU(),
            MaskedLinear(width, positions * classes, inputs[:, None] > second),
        )

    def forward(self, indices):
        values = F.one_hot(indices, self.classes).float().flatten(1)
        return self.layers(values).reshape(-1, self.positions, self.classes)

    @torch.no_grad()
    def sample(self, count, temperature=1.0):
        if temperature < 0:
            raise ValueError("temperature must be nonnegative")
        device = next(self.parameters()).device
        indices = torch.zeros(count, self.positions, dtype=torch.long, device=device)
        for position in range(self.positions):
            logits = self(indices)[:, position]
            if temperature == 0:
                indices[:, position] = logits.argmax(-1)
            else:
                probabilities = (logits / temperature).softmax(-1)
                indices[:, position] = torch.multinomial(probabilities, 1).squeeze(1)
        return indices
