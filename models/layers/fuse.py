import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.cat import CrossAttention
from models.layers.osa import Block_Attention, BlockCAT, VFBlock
from models.layers.vlstm import ViLViewFuseLayer, ViLViewMergeLayer
from models.layers.weight_init import trunc_normal_
from models.layers.gruunet import GRUUNet


from .osa_utils import *


class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x, out_shape):
        h, w = out_shape
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x[:, :, :h, :w]


class FusionInner(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.fuse_cat = BlockCAT(dim)
        self.fuse = ViLViewFuseLayer(dim)

    def forward(self, prj_feats, prj_src_feats, prj_depths):
        B, V, C, H, W = prj_src_feats.shape
        feats = self.fuse_cat(
            prj_src_feats.view(-1, C, H, W),
            prj_feats.view(-1, C, H, W),
            prj_depths.view(-1, 1, H, W),
        )
        return self.fuse(feats.view(B, V, C, H, W), prj_depths)


class FusionOuter(nn.Module):
    def __init__(self, dim, prev_dim) -> None:
        super().__init__()
        self.fuse_cat = BlockCAT(dim)
        self.fuse_layer = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=1, padding=0),
            nn.GELU(),
            Block_Attention(dim),
        )
        self.fuse = ViLViewFuseLayer(dim)
        self.up = PixelShuffleUpsample(prev_dim, dim, 2)

    def forward(self, prev_feats, prj_feats, prj_src_feats, prj_depths):
        B, V, C, H, W = prj_src_feats.shape
        prj_src_feats = prj_src_feats.view(-1, C, H, W)

        prev_feats = self.up(prev_feats, [H, W])
        prj_feats = self.fuse_layer(torch.cat([prev_feats, prj_src_feats], dim=1))
        feats = self.fuse_cat(
            prj_src_feats.view(-1, C, H, W),
            prj_feats.view(-1, C, H, W),
            prj_depths.view(-1, 1, H, W),
        )
        return self.fuse(feats.view(B, V, C, H, W), prj_depths)


class Merger(nn.Module):
    def __init__(self, dim, prev_dim) -> None:
        super().__init__()
        self.fuse_cat = BlockCAT(dim)
        self.fuse_layer = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=1, padding=0),
            nn.GELU(),
            Block_Attention(dim),
        )
        self.fuse = ViLViewMergeLayer(dim)
        self.up = PixelShuffleUpsample(prev_dim, dim, 2)

    def forward(self, prev_feats, prj_feats, prj_src_feats, prj_depths):
        B, V, C, H, W = prj_src_feats.shape
        prj_src_feats = prj_src_feats.view(-1, C, H, W)

        prev_feats = self.up(prev_feats, [H, W])
        prj_feats = self.fuse_layer(torch.cat([prev_feats, prj_src_feats], dim=1))
        feats = self.fuse_cat(
            prj_src_feats.view(-1, C, H, W),
            prj_feats.view(-1, C, H, W),
            prj_depths.view(-1, 1, H, W),
        )
        return self.fuse(feats.view(B, V, C, H, W), prj_depths)


class Fusion(nn.Module):
    def create_fuse_layer(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim * 4, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim * 4, out_dim * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim * 4, out_dim, kernel_size=3, padding=1),
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def __init__(self) -> None:
        super().__init__()
        self.fuse5 = FusionInner(2048)
        self.fuse4 = FusionOuter(1024, 2048)
        self.fuse3 = FusionOuter(512, 1024)
        self.fuse2 = FusionOuter(256, 512)
        self.fuse1 = Merger(64, 256)

    def forward(self, prj_feats, prj_src_feats, prj_depths):
        f5 = self.fuse5(prj_feats[-1], prj_src_feats[-1], prj_depths[-1])
        f4 = self.fuse4(f5, prj_feats[-2], prj_src_feats[-2], prj_depths[-2])
        f3 = self.fuse3(f4, prj_feats[-3], prj_src_feats[-3], prj_depths[-3])
        f2 = self.fuse2(f3, prj_feats[-4], prj_src_feats[-4], prj_depths[-4])
        f1 = self.fuse1(f2, prj_feats[-5], prj_src_feats[-5], prj_depths[-5])
        return f1