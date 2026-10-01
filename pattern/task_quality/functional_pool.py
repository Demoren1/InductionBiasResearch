"""Task-quality learning of positive weights over aligned bank proposals.

The initialization is exactly the previously evaluated uniform functional
baseline. Alignment is an explicit coordinate prior, not learned here.
"""
import math
import torch
from torch import nn
from .generator import FEATURE_DIM,_NoPositionEncoder
from .repaired_generator import repaired_topk_ste

class FunctionalPool(nn.Module):
    def __init__(self,bank,bounded_weights=False):
        super().__init__()
        self.bounded_weights=bounded_weights
        s=bank['q_abs'].double()
        mass=s.sum(1)
        centre=(s*torch.arange(11,dtype=torch.float64)[None,:,None]).sum(1)/mass.clamp_min(1e-30)
        centre=torch.where(mass>0,centre,torch.full_like(centre,float('inf')))
        order=torch.argsort(centre,dim=-1,stable=True)
        aligned=s.gather(2,order[:,None,:].expand(-1,11,-1)).float()
        self.register_buffer('proposals',aligned)
        self.register_buffer('column_order',order)
        self.token_mlp=nn.Sequential(nn.Linear(FEATURE_DIM,64),nn.GELU(),nn.Linear(64,64),nn.GELU())
        self.map_encoder=_NoPositionEncoder(64,1)
        self.support_mlp=nn.Sequential(nn.Linear(12,64),nn.GELU(),nn.Linear(64,64),nn.GELU())
        self.support_encoder=_NoPositionEncoder(64,1)
        self.scorer=nn.Sequential(nn.Linear(192,64),nn.GELU(),nn.Linear(64,1))
        nn.init.zeros_(self.scorer[-1].weight);nn.init.zeros_(self.scorer[-1].bias)

    def forward_with_weights(self,bank,x,y):
        feature,proposals=bank if isinstance(bank,tuple) else (bank,self.proposals)
        if x.ndim==2:x=x.unsqueeze(0);y=y.unsqueeze(0)
        maps=self.map_encoder(self.token_mlp(feature)).mean(1)
        task=self.support_encoder(self.support_mlp(torch.cat((x,y[...,None].to(x.dtype)),-1))).mean(1)
        maps=maps[None].expand(x.size(0),-1,-1)
        task=task[:,None].expand(-1,maps.size(1),-1)
        scores=self.scorer(torch.cat((maps,task,maps*task),-1)).squeeze(-1)
        if self.bounded_weights:
            centered=scores-scores.mean(-1,keepdim=True)
            scale=centered.square().mean(-1,keepdim=True).clamp_min(1.).sqrt()
            scores=2*torch.tanh(centered/scale)
        weights=scores.softmax(-1)
        logits=torch.einsum('bm,mih->bih',weights,proposals)
        return repaired_topk_ste(logits),logits,weights

    def forward(self,bank,x,y):
        mask,logits,_=self.forward_with_weights(bank,x,y)
        return mask,logits
