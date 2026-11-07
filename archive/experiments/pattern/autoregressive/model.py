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
        if width >= positions - 1:
            first = torch.arange(width) % (positions - 1) + 1
        else:
            first = torch.arange(width) * (positions - 1) // width + 1
        second = first[torch.randperm(width)]
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

    @torch.no_grad()
    def sample_cached(self, count, temperature=1.0):
        if temperature < 0:
            raise ValueError("temperature must be nonnegative")
        first, second, final = self.layers[0], self.layers[2], self.layers[4]
        first_weight = first.weight * first.mask
        second_weight = second.weight * second.mask
        final_weight = final.weight * final.mask
        initial = first_weight[:, :: self.classes].sum(-1) + first.bias
        activation = initial[None].expand(count, -1).clone()
        indices = torch.zeros(
            count, self.positions, dtype=torch.long, device=initial.device
        )
        for position in range(self.positions):
            hidden = F.gelu(F.linear(F.gelu(activation), second_weight, second.bias))
            start = position * self.classes
            stop = start + self.classes
            logits = F.linear(hidden, final_weight[start:stop], final.bias[start:stop])
            if temperature == 0:
                chosen = logits.argmax(-1)
            else:
                chosen = torch.multinomial(
                    (logits / temperature).softmax(-1), 1
                ).squeeze(-1)
            indices[:, position] = chosen
            activation += (
                first_weight[:, start + chosen].T - first_weight[:, start][None]
            )
        return indices
