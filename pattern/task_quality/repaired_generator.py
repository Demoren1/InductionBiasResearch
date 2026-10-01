"""Shift-invariant fixed-mass surrogate and a column-preserving set decoder.

New experimental code; legacy generator and old runs remain unchanged.
No Toeplitz mask, window template, U, or oracle enters either model.
"""
import math
import torch
from torch import nn
from .generator import Generator, FEATURE_DIM, _NoPositionEncoder
from .core import SEQ_LEN, HIDDEN, ACTIVE_EDGES

class _FixedMassSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, k):
        lo=z.amin(-1,keepdim=True)-40
        hi=z.amax(-1,keepdim=True)+40
        for _ in range(55 if z.dtype==torch.float64 else 32):
            mid=(lo+hi)/2
            mass=(z-mid).sigmoid().sum(-1,keepdim=True)
            lo=torch.where(mass>k,mid,lo)
            hi=torch.where(mass>k,hi,mid)
        p=(z-(lo+hi)/2).sigmoid()
        ctx.save_for_backward(p)
        return p

    @staticmethod
    def backward(ctx, upstream):
        (p,)=ctx.saved_tensors
        d=p*(1-p)
        weighted_mean=(upstream*d).sum(-1,keepdim=True)/d.sum(-1,keepdim=True).clamp_min(torch.finfo(d.dtype).tiny)
        return d*(upstream-weighted_mean),None

def fixed_mass_soft(logits, k=ACTIVE_EDGES):
    flat=logits.flatten(-2)
    centered=flat-flat.mean(-1,keepdim=True)
    scale=centered.square().mean(-1,keepdim=True).clamp_min(0.01).sqrt()
    # Ranking is unaffected by centering/scaling; normalization prevents
    # arbitrary growth of score magnitudes from killing the surrogate.
    normalized=centered/scale
    return _FixedMassSigmoid.apply(normalized,k).reshape_as(logits)

def repaired_topk_ste(logits, k=ACTIVE_EDGES):
    flat=logits.flatten(-2)
    hard=torch.zeros_like(flat).scatter(-1,flat.topk(k,dim=-1).indices,1.).reshape_as(logits)
    soft=fixed_mass_soft(logits,k)
    return hard+(soft-soft.detach())

class RepairedLegacy(Generator):
    """Identical legacy architecture, new backward rule only."""
    def forward(self, bank, x, y):
        _,logits=super().forward(bank,x,y)
        return repaired_topk_ste(logits),logits

class ColumnSetGenerator(nn.Module):
    """Eight distinct attention slots read full teacher-column tokens.

    Column/map order is invariant. Each output slot retains its own bank and
    task representation instead of sharing a mean-pooled global vector.
    Edge scores explicitly include input-position × slot interactions.
    Only input identities are encoded; no local-distance/window rule is used.
    """
    def __init__(self):
        super().__init__()
        self.token_mlp=nn.Sequential(nn.Linear(FEATURE_DIM,64),nn.GELU(),nn.Linear(64,64),nn.LayerNorm(64))
        self.map_encoder=_NoPositionEncoder(64,1)
        self.slot_queries=nn.Parameter(torch.empty(HIDDEN,64))
        nn.init.orthogonal_(self.slot_queries)
        self.bank_attention=nn.MultiheadAttention(64,2,dropout=0,batch_first=True)
        self.bank_norm=nn.LayerNorm(64)
        self.support_mlp=nn.Sequential(nn.Linear(SEQ_LEN+1,64),nn.GELU(),nn.Linear(64,64),nn.LayerNorm(64))
        self.support_encoder=_NoPositionEncoder(64,1)
        self.task_attention=nn.MultiheadAttention(64,2,dropout=0,batch_first=True)
        self.task_norm=nn.LayerNorm(64)
        self.slot_encoder=_NoPositionEncoder(64,1)
        self.position_embeddings=nn.Parameter(torch.randn(SEQ_LEN,64)/math.sqrt(64))
        self.row_projection=nn.Sequential(nn.Linear(64,64),nn.LayerNorm(64))
        self.slot_projection=nn.Sequential(nn.Linear(64,64),nn.LayerNorm(64))
        self.edge_mlp=nn.Sequential(nn.Linear(192,64),nn.GELU(),nn.Linear(64,1))

    def forward(self, bank, x, y):
        if x.ndim==2:x=x.unsqueeze(0);y=y.unsqueeze(0)
        columns=self.map_encoder(self.token_mlp(bank))
        memory=columns.reshape(1,-1,64)
        seeds=self.slot_queries.unsqueeze(0)
        pooled,_=self.bank_attention(seeds,memory,memory,need_weights=False)
        slots=self.bank_norm(seeds+pooled).expand(x.size(0),-1,-1)
        support=self.support_encoder(self.support_mlp(torch.cat((x,y[...,None].to(x.dtype)),dim=-1)))
        task,_=self.task_attention(slots,support,support,need_weights=False)
        slots=self.slot_encoder(self.task_norm(slots+task))
        rows=self.row_projection(self.position_embeddings)
        cols=self.slot_projection(slots)
        pair=rows[None,:,None,:]*cols[:,None,:,:]
        bilinear=pair.sum(-1)/math.sqrt(64)
        edge=torch.cat((rows[None,:,None,:].expand(x.size(0),-1,HIDDEN,-1),
                        cols[:,None,:,:].expand(-1,SEQ_LEN,-1,-1),pair),dim=-1)
        logits=bilinear+self.edge_mlp(edge).squeeze(-1)
        return repaired_topk_ste(logits),logits

def make_model(variant):
    if variant=='legacy_fixedmass':return RepairedLegacy()
    if variant=='column_set_fixedmass':return ColumnSetGenerator()
    raise ValueError(variant)
