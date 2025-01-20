import torch
import torch.nn as nn

from models.layers.osa import Block_Attention, BlockCAT, VFBlock
from models.layers.vlstm import ViLViewFuseLayer, ViLViewMergeLayer
from models.layers.basic import PixelShuffleUpsample


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
        prj_feats = prj_feats.view(-1, C, H, W)

        prev_feats = self.up(prev_feats, [H, W])
        prj_feats = self.fuse_layer(torch.cat([prev_feats, prj_feats], dim=1))
        feats = self.fuse_cat(
            prj_src_feats,
            prj_feats,
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
        prj_feats = prj_feats.view(-1, C, H, W)

        prev_feats = self.up(prev_feats, [H, W])
        prj_feats = self.fuse_layer(torch.cat([prev_feats, prj_feats], dim=1))
        feats = self.fuse_cat(
            prj_src_feats,
            prj_feats,
        )
        return self.fuse(feats.view(B, V, C, H, W), prj_depths)


class SimpleFusionInner(nn.Module):
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


class SimpleFusionOuter(nn.Module):
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


class SimpleMerger(nn.Module):
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