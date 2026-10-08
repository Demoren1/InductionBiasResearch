import torch
from torch import nn
from .nf import NFLayer
from .vae_encoder import VAEEncoder
from .shared_decoder import SharedDecoder
from ..masks import exact_topk

class TaskEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        channels=config["nf_channels"]
        self.nf=nn.Sequential(NFLayer(3,channels),nn.GELU(),NFLayer(channels,channels),nn.GELU())
        self.vae=VAEEncoder(channels,config["encoder_width"],config["latent_dim"])

    def forward(self,x):
        return self.vae(self.nf(x))

class Experiment(nn.Module):
    def __init__(self,tasks,config):
        super().__init__()
        self.encoders=nn.ModuleDict({task:TaskEncoder(config) for task in tasks})
        self.decoder=SharedDecoder(config["latent_dim"],config["decoder_width"])

    def forward(self,task,x,sample=True):
        mu,logvar=self.encoders[task](x)
        z=mu+torch.randn_like(mu)*(logvar*.5).exp() if sample else mu
        logits=self.decoder(z)
        return {"logits":logits,"mask":exact_topk(logits,32,self.training),
                "mu":mu,"logvar":logvar,"z":z}
