from torch import nn

class SharedDecoder(nn.Module):
    def __init__(self, latent_dim, width):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(latent_dim,width),nn.GELU(),
            nn.Linear(width,width),nn.GELU(),nn.Linear(width,88))

    def forward(self,z):
        return self.body(z).reshape(-1,11,8)
