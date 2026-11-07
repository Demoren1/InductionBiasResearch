"""Hidden-column permutation equivariance with fixed ordered input coordinates.

Specialization to one 11x8 functional edge tensor, rather than the general
multilayer weight-space NF implementation in Zhou et al. (2023).
"""
import torch
from torch import nn

class NFLayer(nn.Module):
    def __init__(self, in_channels, out_channels, rows=11):
        super().__init__()
        self.rows, self.out_channels = rows, out_channels
        self.local = nn.Linear(rows*in_channels, rows*out_channels)
        self.global_context = nn.Linear(rows*in_channels, rows*out_channels, bias=False)

    def forward(self, x):
        # [B,input,hidden,C] -> hidden-column vectors. Both maps share parameters.
        columns = x.permute(0,2,1,3).flatten(-2)
        result = self.local(columns)+self.global_context(columns.mean(1,keepdim=True))
        return result.reshape(x.shape[0],x.shape[2],self.rows,self.out_channels).permute(0,2,1,3)
