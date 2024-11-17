import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.vivim import MambaLayer
from models.layers.swin import Mlp

from torch import einsum
from einops import rearrange

from .osa_utils import *

class MultiViewAttentionFusion(nn.Module):     # NOTE: Spatial attention (MLP style)
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
        window_size = 7,
        with_pe = True,
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.with_pe = with_pe

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.view_attend = nn.Sequential(
            nn.Softmax(dim = 1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        N, C, V, H, W = x.shape
        h = self.heads

        x = x.permute(0, 3, 4, 2, 1).view(N * H * W, V, C)

        # project for queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # scale
        q = q * self.scale

        # sim
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # attention
        attn = self.attend(sim)

        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        
        # merge heads
        out = rearrange(out, 'b h i d -> b i (h d)')

        # combine heads out
        out = self.view_attend(self.to_out(out)) * x
        out = rearrange(out, '(b h w) v c -> b v c h w', b=N , h=H, w=W)
        out = torch.sum(out, dim=1)

        return out


class Fusion(nn.Module):
    def create_fuse_layer(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim * 4, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim * 4, out_dim * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim * 4, out_dim, kernel_size=3, padding=1),
        )

    def __init__(self) -> None:
        super().__init__()

        self.enc1 = MambaLayer(96)
        self.enc2 = MambaLayer(192)
        self.enc3 = MambaLayer(384)
        self.enc4 = MambaLayer(768 + 1)

        self.fuse1 = self.create_fuse_layer(96 + 1 + 192, 96)
        self.fuse2 = self.create_fuse_layer(192 + 1 + 384, 192)
        self.fuse3 = self.create_fuse_layer(384 + 1 + 768 + 1, 384)

        self.fuses = nn.ModuleList([self.fuse3, self.fuse2, self.fuse1])
        self.encs = nn.ModuleList([self.enc3, self.enc2, self.enc1])

        self.view_fuse = MultiViewAttentionFusion(96)

    def forward_step(self, prev_prj, curr_prj, enc, fuse):
        n, _, v, h, w = prev_prj.shape
        prev_prj = prev_prj.permute(0, 2, 1, 3, 4).contiguous().view(n * v, -1, h, w)
            
        n, _, v, h, w = curr_prj.shape
        curr_prj = curr_prj.permute(0, 2, 1, 3, 4).contiguous().view(n * v, -1, h, w)
        prev_prj = F.interpolate(
            prev_prj, size=curr_prj.shape[-2:],
            align_corners=True, mode='bilinear'
        )

        mf = torch.cat([prev_prj, curr_prj], dim=1)
        curr_prj = fuse(mf) + curr_prj[:, :-1]
        curr_prj = curr_prj.view(n, v, -1, h, w).permute(0, 2, 1, 3, 4)

        prev_prj = enc(
            curr_prj
        )

        return prev_prj

    def forward(self, prjs):
        prev_prj = self.enc4(prjs[-1])  # N, C, V, H, W

        prev_prj = self.forward_step(prev_prj, prjs[-2], self.enc3, self.fuse3)
        prev_prj = self.forward_step(prev_prj, prjs[-3], self.enc2, self.fuse2)
        prev_prj = self.forward_step(prev_prj, prjs[-4], self.enc1, self.fuse1)
        
        fused = self.view_fuse(prev_prj)

        return fused
